"""
Drishti — REST API Router
==========================

Endpoint handlers for the Drishti interview pipeline and feedback loop.
"""

from __future__ import annotations

import json
import os
import datetime
from uuid import UUID, uuid4
from typing import Dict, List

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import JSONResponse

from config.logging_config import get_logger
from src.api.schemas import (
    InterviewInitRequest,
    InterviewInitResponse,
    InterviewSubmitRequest,
    InterviewSubmitResponse,
    AgentStatusResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from src.pipelines.interview import get_message_content, get_message_role
from src.utils.api_client import LLMClientError

logger = get_logger(__name__)

router = APIRouter(prefix="/api")

# Thread → agent mapping registry
agent_to_thread_map: Dict[UUID, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record_session_turn(session_id: str, debug_state: dict, user_msg: str, asst_msg: str) -> None:
    """Append the FSM state and conversation turn to a JSONL file for field testing playback."""
    try:
        log_dir = os.path.join(os.getcwd(), "logs", "sessions")
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{session_id}.jsonl")
        
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": session_id,
            "user_intent": debug_state.get("intent", ""),
            "confidence": debug_state.get("confidence", 0),
            "reason": debug_state.get("reason", ""),
            "mode": debug_state.get("mode", ""),
            "energy": debug_state.get("energy", 0),
            "curiosity": debug_state.get("curiosity", 0),
            "certainty": debug_state.get("certainty", 0),
            "topic": debug_state.get("topic", ""),
            "latencies": debug_state.get("latencies", {}),
            "user_message": user_msg,
            "assistant_response": asst_msg
        }
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error("Failed to record session turn: %s", e)


async def _ensure_agent_registered(request: Request, agent_id: UUID) -> None:
    """Ensure agent row exists in DB to satisfy FK constraints."""
    memory_manager = request.app.state.memory_manager
    if memory_manager and memory_manager.client:
        try:
            res = (
                await memory_manager.client.table("agents")
                .select("id")
                .eq("id", str(agent_id))
                .execute()
            )
            if not res or not res.data:
                logger.info("Registering agent %s in DB.", agent_id)
                await memory_manager.client.table("agents").insert(
                    {"id": str(agent_id)}
                ).execute()
        except Exception as e:
            logger.error("Failed to register agent: %s", e)


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


async def llm_client_exception_handler(request: Request, exc: LLMClientError) -> JSONResponse:
    msg = str(exc)
    logger.error("Upstream LLM error: %s", msg)
    if "429" in msg or "rate limit" in msg.lower():
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "Rate Limit", "message": "LLM rate limit exceeded. Retry shortly.", "details": msg},
        )
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"error": "LLM Failure", "message": "Upstream language model error.", "details": msg},
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception at API gateway")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal Server Error", "message": str(exc), "details": ""},
    )


# ---------------------------------------------------------------------------
# POST /api/interview/initialize
# ---------------------------------------------------------------------------


@router.post("/interview/initialize", response_model=InterviewInitResponse)
async def initialize_interview(request: Request, payload: InterviewInitRequest):
    """
    Start a Drishti session. Accepts full identity from Stage 1.
    Returns the opening question shaped by who this person is.
    """
    logger.info("Initializing interview | agent_id=%s | name=%s", payload.agent_id, payload.identity.name)
    await _ensure_agent_registered(request, payload.agent_id)

    thread_id = str(uuid4())
    agent_to_thread_map[payload.agent_id] = thread_id

    initial_state = {
        "agent_id": payload.agent_id,
        "identity": {
            "name": payload.identity.name,
            "age": payload.identity.age,
            "country": payload.identity.country,
            "occupation": payload.identity.occupation,
            "timezone": payload.identity.timezone,
        },
        "messages": [],
        "current_topic": "introduction",
        "stored_memory_count": 0,
        "session_memory_count": 0,
        "insights": [],
        "new_insight_count": 0,
        "persona_labels": [],
        "readiness_missing_info": "Still learning the basics — career, passions, relationships, patterns.",
        "summary_sentences": [],
        "is_complete": False,
        "remaining_turns": 100,
        "consecutive_question_count": 0,
        "conversation_energy": 7,       # Start neutral-positive
        "conversation_curiosity": 5,    # Start neutral
        "conversation_certainty": 5,    # Start neutral
        "reframe_used": False,
        "topic_turn_count": 0,
        "closing_observation": "",
        "evaluation_scores": {
            "Psychology": 0.0,
            "Economy": 0.0,
            "Politics": 0.0,
            "Demographics": 0.0,
        },
    }

    config = {"configurable": {"thread_id": thread_id}}
    graph = request.app.state.graph
    output = await graph.ainvoke(initial_state, config=config)

    messages = output.get("messages", [])
    if not messages:
        raise HTTPException(status_code=500, detail="Failed to generate opening question.")

    next_question = get_message_content(messages[-1])

    return InterviewInitResponse(
        thread_id=thread_id,
        agent_id=output["agent_id"],
        next_question=next_question,
        remaining_turns=output.get("remaining_turns", 100),
        evaluation_scores=output.get("evaluation_scores", {}),
        persona_labels=output.get("persona_labels", []),
        insights=output.get("insights", []),
    )


# ---------------------------------------------------------------------------
# POST /api/interview/submit
# ---------------------------------------------------------------------------


@router.post("/interview/submit", response_model=InterviewSubmitResponse)
async def submit_response(request: Request, payload: InterviewSubmitRequest):
    """
    Submit a user response. Advances the Drishti state machine.
    Returns updated persona labels, any new insights, and summary when complete.
    """
    logger.info("Submitting response | thread_id=%s", payload.thread_id)
    await _ensure_agent_registered(request, payload.agent_id)
    agent_to_thread_map[payload.agent_id] = payload.thread_id

    config = {"configurable": {"thread_id": payload.thread_id}}
    input_state = {
        "messages": [{"role": "user", "content": payload.user_response}]
    }

    graph = request.app.state.graph
    output = await graph.ainvoke(input_state, config=config)

    is_complete = output.get("is_complete", False)
    messages = output.get("messages", [])

    next_question = ""
    if not is_complete and messages:
        last_msg = messages[-1]
        if get_message_role(last_msg) == "assistant":
            next_question = get_message_content(last_msg)
    elif is_complete and messages:
        # Closing message from generate_summary_node
        last_msg = messages[-1]
        if get_message_role(last_msg) == "assistant":
            next_question = get_message_content(last_msg)

    # New insights: pull from output (already in state)
    all_insights = output.get("insights") or []
    new_insights = [i for i in all_insights if i.get("lifecycle_stage") == "candidate"]

    # Build debug state from FSM internals
    debug_state = {
        "intent": output.get("user_intent", "CONTINUE"),
        "confidence": output.get("intent_confidence", 1.0),
        "is_structural": output.get("intent_is_structural", False),
        "reason": output.get("intent_reason", ""),
        "mode": output.get("last_mode", "ASK"),
        "energy": output.get("conversation_energy", 7),
        "curiosity": output.get("conversation_curiosity", 5),
        "certainty": output.get("conversation_certainty", 5),
        "topic": output.get("current_topic", "unknown"),
        "topic_turn_count": output.get("topic_turn_count", 0),
        "support_budget": output.get("support_budget", 0),
        "recent_themes": output.get("recent_themes", []),
        "missing_information": output.get("missing_information", ""),
        "is_ready_for_summary": output.get("is_ready_for_summary", False),
        "latencies": output.get("latencies", {}),
        "build": "v1-freeze",
        "commit": "a41d2bc",
        "model": "llama-3.1-8b",
        "prompt_version": "summary_v9",
        "os_version": "1.0",
        "session_id": payload.thread_id[:7],
    }

    # Record telemetry
    _record_session_turn(
        session_id=payload.thread_id,
        debug_state=debug_state,
        user_msg=payload.user_response,
        asst_msg=next_question
    )

    return InterviewSubmitResponse(
        next_question=next_question,
        remaining_turns=output.get("remaining_turns", 0),
        evaluation_scores=output.get("evaluation_scores", {}),
        is_complete=is_complete,
        persona_labels=output.get("persona_labels", []),
        new_insights=new_insights,
        summary_sentences=output.get("summary_sentences", []),
        debug_state=debug_state,
    )


# ---------------------------------------------------------------------------
# GET /api/agent/{agent_id}/status
# ---------------------------------------------------------------------------


@router.get("/agent/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(request: Request, agent_id: UUID):
    """Fetch current persona, insights, and recent memories."""
    logger.info("Agent status | agent_id=%s", agent_id)

    evaluation_scores = {"Psychology": 0.0, "Economy": 0.0, "Politics": 0.0, "Demographics": 0.0}
    persona_labels: List[Dict] = []
    insights: List[Dict] = []

    thread_id = agent_to_thread_map.get(agent_id) or str(agent_id)
    config = {"configurable": {"thread_id": thread_id}}

    graph = request.app.state.graph
    try:
        state = await graph.aget_state(config=config)
        if state and state.values:
            evaluation_scores = state.values.get("evaluation_scores", evaluation_scores)
            persona_labels = state.values.get("persona_labels", [])
            insights = state.values.get("insights", [])
    except Exception as e:
        logger.warning("Could not read state for thread_id=%s: %s", thread_id, e)

    recent_memories: List[str] = []
    memory_manager = request.app.state.memory_manager
    try:
        if memory_manager and memory_manager.client:
            res = (
                await memory_manager.client.table("memories")
                .select("content")
                .eq("agent_id", str(agent_id))
                .order("created_at", desc=True)
                .limit(5)
                .execute()
            )
            if res and hasattr(res, "data") and res.data:
                raw_memories = [row["content"] for row in res.data]
                # Parse enriched memory JSON to extract readable text
                for raw in raw_memories:
                    try:
                        parsed = json.loads(raw)
                        text = parsed.get("text", raw)
                    except (json.JSONDecodeError, TypeError):
                        text = raw
                    recent_memories.append(text)
    except Exception as e:
        logger.error("Failed to query memories: %s", e)
        recent_memories = []

    return AgentStatusResponse(
        agent_id=agent_id,
        evaluation_scores=evaluation_scores,
        recent_memories=recent_memories,
        persona_labels=persona_labels,
        insights=insights,
    )


# ---------------------------------------------------------------------------
# POST /api/feedback/submit
# ---------------------------------------------------------------------------


@router.post("/feedback/submit", response_model=FeedbackResponse)
async def submit_feedback(request: Request, payload: FeedbackRequest):
    """
    User feedback on a specific summary sentence.

    - 'up': positive signal, logged as reinforcement
    - 'down': contradiction event — relevant insight flagged as is_evolving=True
    - 'edit': user correction stored as high-importance memory

    The surprise_index tells us which observation resonated most — this becomes
    a training signal for prompt improvement.
    """
    logger.info(
        "Feedback received | agent_id=%s | sentence=%d | sentiment=%s",
        payload.agent_id,
        payload.sentence_index,
        payload.sentiment,
    )

    updated_insight_id = None

    # If contradiction or edit: store correction as high-importance memory
    if payload.sentiment in ("down", "edit") and payload.edit_text:
        correction_text = json.dumps({
            "text": f"User correction: {payload.edit_text}",
            "emotion": "neutral",
            "certainty": 1.0,
            "topics": ["self-correction", "feedback"],
            "people": [],
            "time_reference": "present",
            "importance": 9,  # User-provided corrections are always high-importance
            "source": "correction",
            "generated_by_prompt_version": "user_feedback_v1",
        })

        memory_manager = request.app.state.memory_manager
        try:
            if memory_manager and memory_manager.client:
                await memory_manager.add_episodic_memory(
                    agent_id=payload.agent_id,
                    content=correction_text,
                )
                logger.info("Correction stored as high-importance memory.")
        except Exception as e:
            logger.error("Failed to store correction memory: %s", e)

    # If surprise_index is provided, log it (future: use for prompt improvement)
    if payload.surprise_index is not None:
        logger.info(
            "Surprise signal | agent_id=%s | sentence_index=%d",
            payload.agent_id,
            payload.surprise_index,
        )

    message_map = {
        "up": "Thank you. I'll keep building on this.",
        "down": "Noted. I'm updating my understanding.",
        "edit": "Got it. I'll remember what you've shared.",
    }
    message = message_map.get(payload.sentiment, "Thank you.")

    return FeedbackResponse(
        acknowledged=True,
        message=message,
        updated_insight_id=updated_insight_id,
    )
