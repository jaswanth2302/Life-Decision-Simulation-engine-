"""
Drishti — Interview & Insight Pipeline
=======================================

LangGraph state machine implementing the full Drishti interview flow.

Node Graph:
    entry → ask_question_node
                  ↓ (user responds)
            process_response_node
              - enrich and store memory
              - every 3rd stored memory:
                  → extract_insights_node
                  → update_persona_labels_node
                  → assess_readiness_node
                      ↙            ↘
            generate_summary_node   ask_question_node
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from config.logging_config import get_logger
from src.core.state import AgentState
from src.core.schema import (
    GeneratedResponse,
    ReflectionTheme,
    InsightBatch,
    PersonaLabelBatch,
    ReadinessAssessment,
    MemoryEnrichment,
    IntentClassifierOutput,
)
from src.core.memory import MemoryStreamManager
from src.utils.api_client import ResilientLLMClient
from src.utils.prompts import (
    PROMPT_VERSIONS,
    MODE_PROMPTS,
    MEMORY_ENRICHMENT_SYSTEM_PROMPT,
    INSIGHT_EXTRACTION_SYSTEM_PROMPT,
    PERSONA_LABEL_SYSTEM_PROMPT,
    READINESS_ASSESSMENT_SYSTEM_PROMPT,
    DRISHTI_SUMMARY_SYSTEM_PROMPT,
    CLOSING_OBSERVATION_SYSTEM_PROMPT,
    INTENT_CLASSIFIER_SYSTEM_PROMPT,
)
from src.core.analytics import (
    SessionAnalytics,
    ConversationStyleMemory,
    save_session_analytics,
    save_conversation_style,
    load_conversation_style,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Message Helpers
# ---------------------------------------------------------------------------


def get_message_role(msg: Any) -> str:
    """Extract role from dict or LangChain BaseMessage."""
    if isinstance(msg, dict):
        return msg.get("role", "")
    mtype = getattr(msg, "type", "")
    if mtype == "human":
        return "user"
    elif mtype == "ai":
        return "assistant"
    return getattr(msg, "role", "") or mtype


def get_message_content(msg: Any) -> str:
    """Extract content from dict or LangChain BaseMessage."""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return getattr(msg, "content", "")


# ---------------------------------------------------------------------------
# Summary Sentence Output Model
# ---------------------------------------------------------------------------


class EvidenceItemOutput(BaseModel):
    """A single piece of evidence supporting an observation."""
    memory_id: str = Field(default="")
    quote: str = Field(default="")
    interpretation: str = Field(default="")
    weight: float = Field(default=1.0)

class ArgumentOutput(BaseModel):
    """The structured case for why this observation is true."""
    reasoning: str = Field(default="")
    evidence: List[EvidenceItemOutput] = Field(default_factory=list)
    uncertainty: str = Field(default="")
    update_conditions: str = Field(default="")

class SummarySentenceOutput(BaseModel):
    """Structured output for a single summary sentence."""
    text: str
    confidence: float = Field(default=0.8)
    argument: ArgumentOutput = Field(default_factory=ArgumentOutput)
    supporting_insight_ids: List[str] = Field(default_factory=list)
    source_memory_ids: List[str] = Field(default_factory=list)
    is_reframe: bool = False

class SummaryOutput(BaseModel):
    """Structured output for the full 5-sentence summary."""
    sentences: List[SummarySentenceOutput] = Field(
        ..., min_length=5, max_length=5,
        description="Exactly 5 summary sentences with evidence chains."
    )


# ---------------------------------------------------------------------------
# Graph Factory
# ---------------------------------------------------------------------------


def compile_interview_graph(
    llm_client: ResilientLLMClient,
    memory_manager: MemoryStreamManager,
) -> CompiledStateGraph:
    """
    Build and compile the Drishti interview + insight LangGraph state machine.

    The graph handles the full lifecycle:
    question → response → memory → insights → persona → readiness → summary
    """
    workflow = StateGraph(AgentState)

    # -------------------------------------------------------------------
    # Node: Analyze Intent
    # -------------------------------------------------------------------

    async def analyze_intent_node(
        state: AgentState, config: Optional[RunnableConfig] = None
    ) -> dict:
        t0 = time.perf_counter()
        messages = state.get("messages", [])
        if not messages or get_message_role(messages[-1]) != "user":
            return {"user_intent": "CONTINUE", "intent_confidence": 1.0, "intent_is_structural": False}
            
        last_user_text = get_message_content(messages[-1])
        logger.info("analyze_intent_node | agent_id=%s | text=%s", state["agent_id"], last_user_text[:50])

        # ── 1. Repair Detector (Heuristics) ──────────────────────────────
        clarification_phrases = [
            "what do you mean", "huh", "i don't get it", "i dont get it", 
            "can you explain", "explain", "simplify", "say that another way", 
            "i'm confused", "im confused", "what?", "sorry?", "elaborate"
        ]
        text_lower = last_user_text.lower().strip()
        if any(p == text_lower or text_lower.startswith(p) for p in clarification_phrases) or "what do you mean" in text_lower:
            return {
                "user_intent": "CLARIFICATION",
                "intent_confidence": 1.0,
                "intent_is_structural": True,
                "intent_reason": "Matched clarification repair heuristic",
                "latencies": {"Intent Classifier": round((time.perf_counter() - t0) * 1000)},
            }

        result: IntentClassifierOutput = await llm_client.query_structured(
            system_prompt=INTENT_CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=f"Classify this message: '{last_user_text}'",
            target_model=IntentClassifierOutput,
            persona_label="intent_classifier",
        )

        # Apply confidence threshold fallback to CONTINUE
        final_intent = result.intent.value
        if result.confidence < 0.65 and final_intent != "CONTINUE":
            logger.warning("Intent %s below confidence threshold (%.2f). Falling back to CONTINUE.", final_intent, result.confidence)
            final_intent = "CONTINUE"

        latency_ms = round((time.perf_counter() - t0) * 1000)
        
        # Determine reason for debug overlay
        reason = "Normal processing"
        if final_intent == "END_SESSION":
            reason = "Session end requested"
        elif result.is_structural and final_intent in ["CHANGE_TOPIC", "CORRECT_MODEL", "META_QUESTION", "ASK_DRISHTI"]:
            reason = "Structural command bypasses memory"

        return {
            "user_intent": final_intent,
            "intent_confidence": result.confidence,
            "intent_is_structural": result.is_structural,
            "intent_reason": reason,
            "latencies": {"Intent Classifier": latency_ms},
        }

    # -------------------------------------------------------------------
    # Node: Respond  (was: ask_question_node)
    # Python decides the mode. LLM generates the words. Never mixed.
    # -------------------------------------------------------------------

    async def respond_node(
        state: AgentState, config: Optional[RunnableConfig] = None
    ) -> dict:
        t0 = time.perf_counter()
        logger.info("respond_node | agent_id=%s", state["agent_id"])

        identity = state.get("identity") or {}
        name = identity.get("name", "you")
        age = identity.get("age", "unknown")
        country = identity.get("country", "unknown")
        occupation = identity.get("occupation", "unknown")

        # Build history string from last 8 messages
        messages_window = state.get("messages", [])[-8:]
        history = "\n".join(
            f"- {get_message_role(m).capitalize()}: {get_message_content(m)}"
            for m in messages_window
        ) or "None (first turn)"

        # ── Signal Detection ─────────────────────────────────────────────
        all_messages = state.get("messages", [])
        last_user_text = ""
        for m in reversed(all_messages):
            if get_message_role(m) == "user":
                last_user_text = get_message_content(m).lower().strip()
                break

        word_count = len(last_user_text.split())

        ENERGY_DRAIN = [
            "shut up", "stop", "tired", "bored", "idk", "i don't know",
            "i dont know", "fuck", "enough", "whatever", "no more",
            "please stop", "leave me", "ok ok", "ok enough", "not really",
        ]
        ENERGY_FILL = [
            "actually", "that reminds me", "now that i think", "wait",
            "interesting", "yeah exactly", "oh yeah", "that's true",
            "never thought", "good point", "you know what",
        ]
        CURIOSITY_RISE = [
            "wait", "that's actually", "that's true", "how did you know",
            "never thought about", "that reminds me", "now that i think",
            "actually", "you know what", "come to think of it",
        ]
        CURIOSITY_DROP = [
            "idk", "i don't know", "i dont know", "whatever",
            "not sure", "doesn't matter",
        ]
        CERTAINTY_RISE = [
            "i know", "definitely", "i'm sure", "i am sure", "for sure",
            "actually i think i do", "i've decided", "i want to", "i will",
            "i know what i want", "that's it exactly", "yes exactly",
        ]
        CERTAINTY_DROP = [
            "i don't know what i want", "not sure what i want", "i guess",
            "i think maybe", "i'm not sure", "i don't know who i am",
            "lost", "confused about", "no idea",
        ]

        is_long_answer = word_count >= 40
        is_short_answer = word_count <= 5
        has_drain = any(p in last_user_text for p in ENERGY_DRAIN)
        has_fill = any(p in last_user_text for p in ENERGY_FILL) or is_long_answer
        has_curiosity_rise = any(p in last_user_text for p in CURIOSITY_RISE)
        has_curiosity_drop = any(p in last_user_text for p in CURIOSITY_DROP)
        has_certainty_rise = any(p in last_user_text for p in CERTAINTY_RISE)
        has_certainty_drop = any(p in last_user_text for p in CERTAINTY_DROP)

        current_energy = state.get("conversation_energy", 7)
        if has_drain:
            new_energy = max(0, current_energy - 4)  # Hard drop on drain signals
        elif is_short_answer and current_energy < 7:
            new_energy = max(0, current_energy - 1)
        elif has_fill:
            new_energy = min(10, current_energy + 2)
        else:
            new_energy = min(10, current_energy + 1) if current_energy < 5 else current_energy

        current_curiosity = state.get("conversation_curiosity", 5)
        if has_curiosity_rise:
            new_curiosity = min(10, current_curiosity + 3)
        elif is_long_answer:
            new_curiosity = min(10, current_curiosity + 1)
        elif has_curiosity_drop or is_short_answer:
            new_curiosity = max(0, current_curiosity - 2)
        else:
            new_curiosity = current_curiosity

        current_certainty = state.get("conversation_certainty", 5)
        if has_certainty_rise:
            new_certainty = min(10, current_certainty + 3)
        elif has_certainty_drop:
            new_certainty = max(0, current_certainty - 2)
        elif is_long_answer:
            new_certainty = min(10, current_certainty + 1)
        else:
            new_certainty = current_certainty

        consecutive_questions = state.get("consecutive_question_count", 0)
        topic_turn_count = state.get("topic_turn_count", 0) + 1
        reframe_used = state.get("reframe_used", False)
        
        last_mode = state.get("last_mode", "ASK")
        support_budget = state.get("support_budget", 0)
        recent_themes = state.get("recent_themes", [])

        # ── MODE SELECTION (Transition Policy FSM) ──────────────
        import random
        last_was_reflect = last_mode == "REFLECT"
        user_intent = state.get("user_intent", "CONTINUE")

        # 1. Check for Meta-Intents First
        if user_intent == "CHANGE_TOPIC":
            proposed_mode = "CHANGE_TOPIC"
            topic_turn_count = 0
            consecutive_questions = 0
        elif user_intent == "CORRECT_MODEL":
            proposed_mode = "CORRECT_MODEL"
        elif user_intent == "CLARIFICATION":
            proposed_mode = "CLARIFICATION"
        elif user_intent == "META_QUESTION":
            proposed_mode = "META_QUESTION"
        elif user_intent == "ASK_DRISHTI":
            proposed_mode = "ASK_DRISHTI"
        else:
            # 2. Baseline mode based on energy/signals
            if new_energy <= 3 or has_drain:
                proposed_mode = "PAUSE"
            elif consecutive_questions >= 3 or topic_turn_count >= 4:
                proposed_mode = "REFLECT"
            elif new_energy < 6 and new_curiosity < 6:
                proposed_mode = "REFLECT"
            elif new_energy >= 8 and new_curiosity >= 8 and last_mode == "ASK":
                proposed_mode = "ASK" # High energy/curiosity allows consecutive questions
            elif new_energy >= 6 or new_curiosity >= 7:
                if last_was_reflect and random.random() < 0.25:
                    proposed_mode = "REFLECT"
                else:
                    proposed_mode = "ASK"
            else:
                proposed_mode = "REFLECT"

        # Apply Valid Transitions and Budgets
        VALID_TRANSITIONS = {
            "ASK": ["REFLECT", "PAUSE", "ASK"],
            "REFLECT": ["ASK", "REFLECT", "PAUSE"],
            "PAUSE": ["RECOVER"],
            "RECOVER": ["ASK", "REFLECT", "SPACE", "END"],
            "SPACE": ["ASK", "REFLECT", "END"]
        }

        # Override PAUSE if budget exhausted
        if proposed_mode == "PAUSE" and support_budget >= 1:
            proposed_mode = "RECOVER"
            
        # Ensure transition is valid, else fallback
        meta_modes = ["CHANGE_TOPIC", "CORRECT_MODEL", "META_QUESTION", "ASK_DRISHTI", "CLARIFICATION"]
        if proposed_mode not in meta_modes and proposed_mode not in VALID_TRANSITIONS.get(last_mode, ["REFLECT"]):
            # Fallback to the first valid non-pause transition
            valid = [m for m in VALID_TRANSITIONS.get(last_mode, ["REFLECT"]) if m != "PAUSE"]
            proposed_mode = valid[0] if valid else "REFLECT"
            
        mode = proposed_mode

        # Choose reflection sub-type
        if mode == "REFLECT":
            if not reframe_used and new_certainty >= 6:
                reflection_type = "REFRAME"
            elif len(all_messages) > 6:  # Enough history to connect
                reflection_type = "CONNECT"
            else:
                reflection_type = "MIRROR"
        else:
            reflection_type = None

        logger.info(
            "Mode selected | mode=%s | reflection_type=%s | energy=%d | curiosity=%d | certainty=%d | consecutive=%d | last_was_reflect=%s",
            mode, reflection_type, new_energy, new_curiosity, new_certainty, consecutive_questions, last_was_reflect
        )

        # ── Build Insights Context ────────────────────────────────────────
        insights = state.get("insights") or []
        strong_insights = [
            i["text"] for i in insights
            if i.get("lifecycle_stage") in ("supported", "strong")
        ]
        current_insights = (
            "\n".join(f"• {t}" for t in strong_insights[:5])
            if strong_insights
            else "None yet."
        )
        missing_info = state.get("readiness_missing_info") or (
            "Still learning the basics — career, passions, relationships, patterns."
        )

        recent_themes_context = (
            "\n".join(f"• {t}" for t in recent_themes)
            if recent_themes
            else "None yet."
        )

        base_context = (
            f"You are Drishti, a warm perceptive companion talking with {name} "
            f"({age}, {country}, {occupation}).\n"
            f"Recent conversation:\n{history}\n\n"
            f"What you've understood so far:\n{current_insights}\n\n"
            f"Recently explored themes (AVOID repeating these conceptually):\n{recent_themes_context}\n"
        )
        prompt_key = reflection_type if mode == "REFLECT" else mode
        prompt_template = MODE_PROMPTS[prompt_key]
        system_prompt = prompt_template.format(
            base_context=base_context,
            name=name,
            missing_info=missing_info,
        )
        # Construct a raw JSON output block exactly matching GeneratedResponse
        json_example = (
            "{\n"
            f'  "response": "your {prompt_key} response here",\n'
            '  "theme": "ONE_OF_THE_ENUM_VALUES",\n'
            f'  "interaction_type": "{prompt_key}",\n'
            '  "confidence": 0.9\n'
            "}"
        )
        available_themes = [e.value for e in ReflectionTheme if e.value not in recent_themes]
        if not available_themes:
            available_themes = [e.value for e in ReflectionTheme] # Fallback if all used
        enum_list = ", ".join(available_themes)

        user_prompt = {
            "PAUSE":   f"Write a PAUSE response for {name}. No question.",
            "RECOVER": f"Write a RECOVER response for {name}.",
            "SPACE":   f"Write a SPACE response for {name}.",
            "MIRROR":  f"Write a MIRROR reflection for {name}.",
            "CONNECT": f"Write a CONNECT reflection for {name}.",
            "REFRAME": f"Write a REFRAME reflection for {name}. One sentence.",
            "ASK":     f"Write exactly ONE question for {name}.",
            "CHANGE_TOPIC": f"Write a response pivoting to a new topic for {name}.",
            "CORRECT_MODEL": f"Acknowledge {name}'s correction and ask for their perspective.",
            "META_QUESTION": f"Answer {name}'s meta-question gracefully.",
            "ASK_DRISHTI": f"Provide thoughtful advice to {name} based on what you know.",
            "CLARIFICATION": f"Explain your previous message in simpler words.\nRequirements:\n- simpler wording\n- same meaning\n- under 40 words\n- don't introduce new ideas\n- don't mention being an AI.",
        }.get(prompt_key, f"Write a {prompt_key} response for {name}.")
        
        user_prompt += f"\n\nTheme MUST be one of: {enum_list}"
        user_prompt += f"\n\nReturn EXACTLY this JSON format:\n{json_example}"

        result: GeneratedResponse = await llm_client.query_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            target_model=GeneratedResponse,
            persona_label="drishti_interviewer",
        )

        response_text = result.response

        # Track consecutive questions
        is_question = mode == "ASK"
        new_consecutive = (consecutive_questions + 1) if is_question else 0

        # Detect if a Reframe was delivered
        REFRAME_MARKERS = [
            "i don't think you're", "i don't think you are",
            "i think you're not", "this isn't about",
            "i think what's really", "you're not searching for",
            "that's not what this is",
        ]
        used_reframe_now = (reflection_type == "REFRAME") or any(
            m in response_text.lower() for m in REFRAME_MARKERS
        )
        new_reframe_used = reframe_used or used_reframe_now

        # Reset topic entropy on non-question responses or explicit pivots
        new_topic_turns = 0 if not is_question else topic_turn_count

        logger.info(
            "Response generated | mode=%s | energy=%d→%d | consecutive=%d | text=%s",
            mode, current_energy, new_energy, new_consecutive, response_text[:80]
        )

        # State updates
        new_support_budget = support_budget + 1 if mode in ["PAUSE", "RECOVER", "SPACE"] else 0
        
        # Keep rolling window of last 3 themes for cooldown
        new_recent_themes = recent_themes.copy()
        if result.theme:
            new_recent_themes.append(result.theme.value)
        if len(new_recent_themes) > 3:
            new_recent_themes = new_recent_themes[-3:]

        latency_ms = round((time.perf_counter() - t0) * 1000)
        current_latencies = state.get("latencies", {}).copy()
        current_latencies["Response Generation"] = latency_ms

        return {
            "messages": [{"role": "assistant", "content": response_text}],
            "current_topic": "conversation",
            "consecutive_question_count": new_consecutive,
            "conversation_energy": new_energy,
            "conversation_curiosity": new_curiosity,
            "conversation_certainty": new_certainty,
            "topic_turn_count": new_topic_turns,
            "reframe_used": new_reframe_used,
            "last_mode": mode,
            "support_budget": new_support_budget,
            "recent_themes": new_recent_themes,
            "latencies": current_latencies,
        }

    # -------------------------------------------------------------------
    # Node: Process Response
    # -------------------------------------------------------------------

    async def process_response_node(state: AgentState) -> dict:
        """
        After user responds:
        1. Enrich the memory with emotion/topics/certainty
        2. Store in Supabase if importance >= 5
        3. Every 3rd stored memory → trigger insight extraction
        """
        logger.info("process_response_node | agent_id=%s", state["agent_id"])
        import time
        t0 = time.perf_counter()
        
        messages = state.get("messages", [])
        if not messages:
            return {}

        user_response = get_message_content(messages[-1])
        last_question = ""
        for msg in reversed(messages[:-1]):
            if get_message_role(msg) == "assistant":
                last_question = get_message_content(msg)
                break

        memory_text = f"Q: {last_question}\nA: {user_response}"

        # Step 1: Enrich memory with semantic metadata
        enrichment: MemoryEnrichment = await llm_client.query_structured(
            system_prompt=MEMORY_ENRICHMENT_SYSTEM_PROMPT.format(
                memory_text=memory_text
            ),
            user_prompt="Extract semantic metadata from this memory.",
            target_model=MemoryEnrichment,
            persona_label="memory_enricher",
        )

        logger.info(
            "Memory enriched | emotion=%s | importance=%d | topics=%s",
            enrichment.emotion,
            enrichment.importance,
            enrichment.topics,
        )

        stored_count = state.get("stored_memory_count", 0)
        session_count = state.get("session_memory_count", 0)

        # Step 2: Store if importance >= 5
        if enrichment.importance >= 5:
            try:
                memory_content_with_meta = json.dumps({
                    "text": memory_text,
                    "emotion": enrichment.emotion,
                    "certainty": enrichment.certainty,
                    "topics": enrichment.topics,
                    "people": enrichment.people,
                    "time_reference": enrichment.time_reference,
                    "importance": enrichment.importance,
                    "generated_by_prompt_version": PROMPT_VERSIONS["memory_enrichment"],
                })
                await memory_manager.add_episodic_memory(
                    agent_id=state["agent_id"],
                    content=memory_content_with_meta,
                )
                stored_count += 1
                logger.info("Memory stored | stored_count=%d", stored_count)
            except Exception as e:
                logger.error("Failed to store memory: %s", e)
        else:
            logger.info(
                "Memory importance %d < 5 — session context only, not stored.",
                enrichment.importance,
            )

        session_count += 1

        # Legacy score update (kept for API compatibility)
        old_scores = state.get("evaluation_scores") or {}
        new_scores = {
            "Psychology": min(1.0, old_scores.get("Psychology", 0.0) + 0.05),
            "Economy": min(1.0, old_scores.get("Economy", 0.0) + 0.03),
            "Politics": min(1.0, old_scores.get("Politics", 0.0) + 0.02),
            "Demographics": min(1.0, old_scores.get("Demographics", 0.0) + 0.04),
        }

        remaining = max(0, state.get("remaining_turns", 100) - 1)
        
        latency_ms = round((time.perf_counter() - t0) * 1000)
        current_latencies = state.get("latencies", {}).copy()
        current_latencies["Memory Extraction"] = latency_ms

        return {
            "stored_memory_count": stored_count,
            "session_memory_count": session_count,
            "evaluation_scores": new_scores,
            "remaining_turns": remaining,
            "latencies": current_latencies,
        }

    # -------------------------------------------------------------------
    # Node: Extract Insights
    # -------------------------------------------------------------------

    async def extract_insights_node(state: AgentState) -> dict:
        """
        Pull recent memories from Supabase and extract 1–3 new insights.
        Only runs every 3rd stored memory.
        """
        t0 = time.perf_counter()
        logger.info("extract_insights_node | agent_id=%s", state["agent_id"])

        identity = state.get("identity") or {}
        name = identity.get("name", "the user")

        # Retrieve recent memories for insight extraction
        try:
            recent_memories = await memory_manager.retrieve_simulation_context(
                agent_id=state["agent_id"],
                query="personal patterns, habits, values, emotions, relationships",
                limit=6,
            )
        except Exception as e:
            logger.error("Failed to retrieve memories for insight extraction: %s", e)
            recent_memories = []

        if not recent_memories:
            logger.info("No memories available for insight extraction yet.")
            return {}

        existing_insights = state.get("insights") or []
        existing_texts = [i["text"] for i in existing_insights]
        existing_summary = (
            "\n".join(f"• {t}" for t in existing_texts[:10])
            if existing_texts
            else "None yet."
        )

        memory_batch = "\n\n".join(
            f"[Memory {i+1}]: {m}" for i, m in enumerate(recent_memories)
        )

        result: InsightBatch = await llm_client.query_structured(
            system_prompt=INSIGHT_EXTRACTION_SYSTEM_PROMPT.format(
                name=name,
                memory_batch=memory_batch,
                existing_insights=existing_summary,
            ),
            user_prompt="Extract new insights from these memories.",
            target_model=InsightBatch,
            persona_label="insight_extractor",
        )

        # Build new insight dicts with lifecycle metadata
        new_insights = []
        for raw in result.insights:
            text = raw.get("text", "").strip()
            source_ids = raw.get("source_memory_ids", [])
            if not text:
                continue

            # Determine lifecycle stage based on evidence
            evidence_count = len(source_ids)
            if evidence_count >= 3:
                stage = "strong"
                confidence = 0.90
            elif evidence_count >= 2:
                stage = "supported"
                confidence = 0.70
            else:
                stage = "candidate"
                confidence = 0.40

            insight = {
                "text": text,
                "lifecycle_stage": stage,
                "source_memory_ids": source_ids,
                "contradiction_memory_ids": [],
                "confidence": confidence,
                "evidence_count": evidence_count,
                "contradiction_count": 0,
                "is_evolving": False,
                "generated_by_prompt_version": PROMPT_VERSIONS["insight_extraction"],
            }
            new_insights.append(insight)
            logger.info("New insight [%s]: %s", stage, text[:60])

        combined_insights = existing_insights + new_insights
        new_insight_count = (state.get("new_insight_count") or 0) + len(new_insights)
        
        latency_ms = round((time.perf_counter() - t0) * 1000)
        current_latencies = state.get("latencies", {}).copy()
        current_latencies["Insight Extraction"] = latency_ms

        return {
            "insights": combined_insights,
            "new_insight_count": new_insight_count,
            "latencies": current_latencies,
        }

    # -------------------------------------------------------------------
    # Node: Update Persona Labels
    # -------------------------------------------------------------------

    async def update_persona_labels_node(state: AgentState) -> dict:
        """
        Given all current insights, regenerate the human-readable persona labels.
        Only includes labels with confidence >= 0.4 (shown to user at >= 0.4).
        """
        t0 = time.perf_counter()
        logger.info("update_persona_labels_node | agent_id=%s", state["agent_id"])

        insights = state.get("insights") or []
        if not insights:
            return {}

        identity = state.get("identity") or {}
        name = identity.get("name", "the user")

        # Only use supported/strong insights for label generation
        usable = [
            i for i in insights
            if i.get("lifecycle_stage") in ("supported", "strong")
        ]
        if not usable:
            logger.info("No supported/strong insights yet — skipping label update.")
            return {}

        insights_with_confidence = "\n".join(
            f"• [{i['lifecycle_stage'].upper()} | conf={i['confidence']:.0%}] {i['text']}"
            for i in usable
        )

        result: PersonaLabelBatch = await llm_client.query_structured(
            system_prompt=PERSONA_LABEL_SYSTEM_PROMPT.format(
                name=name,
                insights_with_confidence=insights_with_confidence,
            ),
            user_prompt="Generate identity reflection labels from these insights.",
            target_model=PersonaLabelBatch,
            persona_label="persona_labeler",
        )

        labels = []
        for raw in result.labels:
            label = raw.get("label", "").strip()
            if not label:
                continue
            confidence = float(raw.get("confidence", 0.5))
            labels.append({
                "label": label,
                "stars": int(raw.get("stars", 3)),
                "confidence": confidence,
                "evidence_count": int(raw.get("evidence_count", 1)),
                "source_insight_ids": raw.get("source_insight_ids", []),
                "generated_by_prompt_version": PROMPT_VERSIONS["persona_label"],
            })
            logger.info("Persona label: %s (conf=%.0f%%)", label, confidence * 100)

        latency_ms = round((time.perf_counter() - t0) * 1000)
        current_latencies = state.get("latencies", {}).copy()
        current_latencies["Persona Labeling"] = latency_ms

        return {
            "persona_labels": labels,
            "new_insight_count": 0,  # Reset counter after label update
            "latencies": current_latencies,
        }

    # -------------------------------------------------------------------
    # Node: Assess Readiness
    # -------------------------------------------------------------------

    async def assess_readiness_node(state: AgentState) -> dict:
        """
        LLM judgment: can Drishti write a trustworthy 5-sentence summary right now?
        Not a counter. A judgment.
        """
        t0 = time.perf_counter()
        logger.info("assess_readiness_node | agent_id=%s", state["agent_id"])

        insights = state.get("insights") or []
        labels = state.get("persona_labels") or []
        identity = state.get("identity") or {}
        name = identity.get("name", "the user")

        usable_insights = [
            i for i in insights
            if i.get("lifecycle_stage") in ("supported", "strong")
        ]

        insights_summary = "\n".join(
            f"• [{i['lifecycle_stage'].upper()} | evidence={i['evidence_count']}] {i['text']}"
            for i in usable_insights
        ) or "None yet."

        visible_labels = [l for l in labels if l.get("confidence", 0) >= 0.4]
        labels_summary = "\n".join(
            f"• {l['label']} ({'★' * l['stars']}{'☆' * (5 - l['stars'])}) {l['confidence']:.0%}"
            for l in visible_labels
        ) or "None yet."

        result: ReadinessAssessment = await llm_client.query_structured(
            system_prompt=READINESS_ASSESSMENT_SYSTEM_PROMPT.format(
                name=name,
                insight_count=len(usable_insights),
                insights_summary=insights_summary,
                label_count=len(visible_labels),
                labels_summary=labels_summary,
            ),
            user_prompt="Can you write a trustworthy summary right now?",
            target_model=ReadinessAssessment,
            persona_label="readiness_assessor",
        )

        logger.info(
            "Readiness assessment: can_summarize=%s | missing=%s",
            result.can_summarize,
            result.missing_info[:60] if result.missing_info else "N/A",
        )

        return {
            "can_summarize": result.can_summarize,
            "readiness_missing_info": result.missing_info or "",
        }

    # -------------------------------------------------------------------
    # Node: Generate Summary — The Magical Moment
    # -------------------------------------------------------------------

    async def generate_summary_node(state: AgentState) -> dict:
        """
        Generate the 5-sentence Drishti summary with full evidence chains.
        Every sentence traceable to specific memories.
        This is the moment.
        """
        logger.info("generate_summary_node | agent_id=%s", state["agent_id"])

        identity = state.get("identity") or {}
        name = identity.get("name", "you")
        insights = state.get("insights") or []
        labels = state.get("persona_labels") or []

        usable_insights = [
            i for i in insights
            if i.get("lifecycle_stage") in ("supported", "strong")
        ]

        insights_text = "\n".join(
            f"[ID: insight_{j}] {i['text']} (evidence: {i['evidence_count']} memories)"
            for j, i in enumerate(usable_insights)
        )

        # Retrieve supporting memory excerpts
        try:
            memory_excerpts_raw = await memory_manager.retrieve_simulation_context(
                agent_id=state["agent_id"],
                query="key personal patterns, values, defining moments, relationships",
                limit=8,
            )
            # Parse JSON if enriched memory
            memory_excerpts_list = []
            for idx, m in enumerate(memory_excerpts_raw):
                try:
                    parsed = json.loads(m)
                    text = parsed.get("text", m)
                except (json.JSONDecodeError, TypeError):
                    text = m
                memory_excerpts_list.append(f"[memory_{idx}]: {text}")
            memory_excerpts = "\n".join(memory_excerpts_list)
        except Exception as e:
            logger.error("Failed to retrieve memories for summary: %s", e)
            memory_excerpts = "No memories retrieved."

        visible_labels = [l for l in labels if l.get("confidence", 0) >= 0.4]
        persona_labels_text = "\n".join(
            f"• {l['label']} ({'★' * l['stars']}{'☆' * (5 - l['stars'])})"
            for l in visible_labels
        ) or "Still forming."

        missing_info = state.get("readiness_missing_info") or "None."

        result: SummaryOutput = await llm_client.query_structured(
            system_prompt=DRISHTI_SUMMARY_SYSTEM_PROMPT.format(
                name=name,
                insights=insights_text,
                memory_excerpts=memory_excerpts,
                persona_labels=persona_labels_text,
                missing_info=missing_info,
            ),
            user_prompt=f"Write the 5-sentence Drishti summary for {name}.",
            target_model=SummaryOutput,
            persona_label="summary_writer",
        )

        summary_sentences = [
            {
                "text": s.text,
                "confidence": s.confidence,
                "argument": s.argument.model_dump(),
                "supporting_insight_ids": s.supporting_insight_ids,
                "source_memory_ids": s.source_memory_ids,
                "is_reframe": s.is_reframe,
                "generated_by_prompt_version": PROMPT_VERSIONS["summary"],
            }
            for s in result.sentences
        ]

        logger.info(
            "Summary generated | %d sentences | reframes: %d",
            len(summary_sentences),
            sum(1 for s in result.sentences if s.is_reframe),
        )

        # ── Generate the single closing observation ───────────────────────
        closing_observation = ""
        try:
            insights_for_obs = "\n".join(
                f"• {i['text']}" for i in usable_insights[:6]
            ) or "None yet."
            history_for_obs = "\n".join(
                f"- {get_message_role(m).capitalize()}: {get_message_content(m)}"
                for m in (state.get("messages") or [])[-6:]
            )
            obs_result: GeneratedQuestion = await llm_client.query_structured(
                system_prompt=CLOSING_OBSERVATION_SYSTEM_PROMPT.format(
                    name=name,
                    insights=insights_for_obs,
                    history=history_for_obs,
                ),
                user_prompt="Generate the single closing observation sentence.",
                target_model=GeneratedQuestion,
                persona_label="closing_observer",
            )
            closing_observation = obs_result.question
            logger.info("Closing observation: %s", closing_observation)
        except Exception as e:
            logger.warning("Failed to generate closing observation: %s", e)
            closing_observation = ""

        # ── Fire session analytics + update conversation style ────────────
        try:
            all_messages = state.get("messages") or []
            session_id = str(state.get("agent_id", "")) + "-" + str(len(all_messages))
            analytics = SessionAnalytics(
                agent_id=state["agent_id"],
                session_id=session_id,
                state=state,
                messages=all_messages,
            )
            analytics.log_summary()  # Always log to console

            # Persist to Supabase (non-blocking, non-fatal)
            db = memory_manager.client
            await save_session_analytics(db, analytics)

            # Update conversation style meta-memory
            previous_style = await load_conversation_style(db, state["agent_id"])
            new_style = ConversationStyleMemory.from_session(
                agent_id=state["agent_id"],
                analytics=analytics,
                previous=previous_style,
            )
            await save_conversation_style(db, new_style)
        except Exception as e:
            logger.warning("Analytics pipeline failed (non-fatal): %s", e)

        # Append closing message to conversation
        closing = (
            f"I've been listening carefully, {name}. "
            "I've begun to understand the person you are today. "
            "This isn't the truth — it's my best understanding of your story so far."
        )

        return {
            "messages": [{"role": "assistant", "content": closing}],
            "summary_sentences": summary_sentences,
            "closing_observation": closing_observation,
            "is_complete": True,
        }

    # -------------------------------------------------------------------
    # Node: Finalize (no-op, sets is_complete for legacy API compat)
    # -------------------------------------------------------------------

    async def finalize_node(state: AgentState) -> dict:
        return {"is_complete": True}

    # -------------------------------------------------------------------
    # Register Nodes
    # -------------------------------------------------------------------

    workflow.add_node("analyze_intent", analyze_intent_node)
    workflow.add_node("respond", respond_node)
    workflow.add_node("process_response", process_response_node)
    workflow.add_node("extract_insights", extract_insights_node)
    workflow.add_node("update_persona_labels", update_persona_labels_node)
    workflow.add_node("assess_readiness", assess_readiness_node)
    workflow.add_node("generate_summary", generate_summary_node)
    workflow.add_node("finalize", finalize_node)

    # -------------------------------------------------------------------
    # Entry Point Routing
    # -------------------------------------------------------------------

    def route_entry(state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages or get_message_role(messages[-1]) != "user":
            return "respond"
        return "analyze_intent"

    workflow.set_conditional_entry_point(
        route_entry,
        {
            "respond": "respond",
            "analyze_intent": "analyze_intent",
        },
    )

    # -------------------------------------------------------------------
    # After Intent Classification
    # -------------------------------------------------------------------

    def route_after_intent(state: AgentState) -> str:
        intent = state.get("user_intent", "CONTINUE")
        is_structural = state.get("intent_is_structural", False)
        
        route = "process_response"
        reason = "Normal processing"
        memory_saved = True

        if intent == "END_SESSION":
            route = "generate_summary"
            reason = "Session end requested"
            memory_saved = False
        elif is_structural and intent in ["CHANGE_TOPIC", "CORRECT_MODEL", "META_QUESTION", "ASK_DRISHTI"]:
            route = "respond"
            reason = "Structural command bypasses memory"
            memory_saved = False
            
        # Write the structured JSON routing log for debugging
        turn_count = len([m for m in state.get("messages", []) if getattr(m, "role", m.get("type") if isinstance(m, dict) else "") == "user"])
        log_payload = {
            "session_id": str(state.get("agent_id", "unknown")),
            "turn": turn_count,
            "intent": intent,
            "confidence": state.get("intent_confidence", 1.0),
            "mode_before": state.get("last_mode", "ASK"),
            "route": route,
            "energy": state.get("conversation_energy", 7),
            "curiosity": state.get("conversation_curiosity", 5),
            "certainty": state.get("conversation_certainty", 5),
            "topic": state.get("current_topic", "unknown"),
            "memory_saved": memory_saved,
            "reason": reason,
        }
        logger.info("ROUTING_LOG: %s", json.dumps(log_payload))

        # Save reason into state for debug overlay
        state["intent_reason"] = reason

        return route

    workflow.add_conditional_edges(
        "analyze_intent",
        route_after_intent,
        {
            "generate_summary": "generate_summary",
            "respond": "respond",
            "process_response": "process_response",
        },
    )

    workflow.add_edge("respond", END)

    # -------------------------------------------------------------------
    # After Process Response: decide whether to run insight pipeline
    # -------------------------------------------------------------------

    def should_extract_insights(state: AgentState) -> str:
        """Trigger insight extraction every 3rd stored memory."""
        stored = state.get("stored_memory_count", 0)
        if stored > 0 and stored % 3 == 0:
            logger.info("Triggering insight extraction at stored_memory_count=%d", stored)
            return "extract_insights"
        return "respond"

    workflow.add_conditional_edges(
        "process_response",
        should_extract_insights,
        {
            "extract_insights": "extract_insights",
            "respond": "respond",
        },
    )

    # Insight extraction always triggers persona label update
    workflow.add_edge("extract_insights", "update_persona_labels")
    # After label update, assess readiness
    workflow.add_edge("update_persona_labels", "assess_readiness")

    # -------------------------------------------------------------------
    # After Readiness: generate summary or continue interview
    # -------------------------------------------------------------------

    def route_after_readiness(state: AgentState) -> str:
        """
        If can_summarize is True, the assessment said yes.
        Safety: also check we have at least 3 strong insights (MVP testing threshold).
        Hard cap: remaining_turns <= 0.
        """
        can_summarize = state.get("can_summarize", False)
        missing = state.get("readiness_missing_info") or "still needed"
        remaining = state.get("remaining_turns", 100)

        # Hard safety cap
        if remaining <= 0:
            logger.info("Hard turn cap reached — forcing summary generation.")
            return "generate_summary"

        # Check if readiness was granted explicitly by the LLM
        if can_summarize:
            usable_insights = [
                i for i in (state.get("insights") or [])
                if i.get("lifecycle_stage") in ("strong", "supported")
            ]
            if len(usable_insights) >= 3:
                logger.info("Readiness granted — routing to generate_summary.")
                return "generate_summary"
            else:
                logger.info("LLM said ready, but we only have %d/3 strong insights. Continuing.", len(usable_insights))

        logger.info("Not ready yet — continuing interview. Missing: %s", missing[:60])
        return "respond"

    workflow.add_conditional_edges(
        "assess_readiness",
        route_after_readiness,
        {
            "generate_summary": "generate_summary",
            "respond": "respond",
        },
    )

    workflow.add_edge("generate_summary", END)

    return workflow.compile(checkpointer=MemorySaver())
