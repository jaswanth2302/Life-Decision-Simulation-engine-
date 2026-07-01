"""
Generative Agent Simulation Engine — Onboarding State Machine Test Suite
========================================================================

Verifies LangGraph interview transitions, termination constraints,
dynamic gap targeting, and data collection monotonicity. All API calls
are mocked for offline validation.
"""

from __future__ import annotations

import pytest
from typing import Dict, List, Optional
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.state import AgentState
from src.core.memory import MemoryStreamManager
from src.utils.api_client import ResilientLLMClient
from src.pipelines.interview import (
    compile_interview_graph,
    GeneratedQuestion,
    EvaluationResult,
    get_message_role,
    get_message_content,
)


# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------


def _build_mock_llm_client(
    *,
    question_text: str = "What is your primary career goal?",
    psychology_score: float = 0.5,
    economy_score: float = 0.5,
    politics_score: float = 0.5,
    demographics_score: float = 0.5,
) -> ResilientLLMClient:
    """Create a mocked ResilientLLMClient returning structured models."""
    client = ResilientLLMClient.__new__(ResilientLLMClient)

    async def mock_query_structured(
        system_prompt: str,
        user_prompt: str,
        target_model: any,
        persona_label: str = "expert",
    ) -> any:
        if target_model == GeneratedQuestion:
            return GeneratedQuestion(question=question_text)
        elif target_model == EvaluationResult:
            return EvaluationResult(
                psychology_score=psychology_score,
                economy_score=economy_score,
                politics_score=politics_score,
                demographics_score=demographics_score,
            )
        raise ValueError(f"Unexpected target model: {target_model}")

    client.query_structured = AsyncMock(side_effect=mock_query_structured)
    return client


def _build_mock_memory_manager() -> MemoryStreamManager:
    """Create a mocked MemoryStreamManager returning dummy context."""
    manager = MemoryStreamManager.__new__(MemoryStreamManager)
    manager.retrieve_simulation_context = AsyncMock(
        return_value=["Mocked Memory 1", "Mocked Memory 2"]
    )
    manager.add_episodic_memory = AsyncMock(return_value=uuid4())
    return manager


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestInterviewGraphTransitions:
    """Verifies that the state machine routes and terminates correctly."""

    @pytest.mark.asyncio
    async def test_initial_invocation_routes_to_ask_question(self) -> None:
        """Verify that when no messages exist, entry routes to ask_question_node."""
        llm = _build_mock_llm_client(question_text="Targeted question?")
        mem = _build_mock_memory_manager()
        graph = compile_interview_graph(llm, mem)

        agent_id = uuid4()
        initial_state: AgentState = {
            "agent_id": agent_id,
            "messages": [],
            "current_topic": "Psychological Baseline",
            "remaining_turns": 10,
            "evaluation_scores": {
                "Psychology": 0.0,
                "Economy": 0.0,
                "Politics": 0.0,
                "Demographics": 0.0,
            },
            "is_complete": False,
        }

        # Invoke graph with configuration (required for MemorySaver checkpointer)
        config = {"configurable": {"thread_id": "test-thread-initial"}}
        output = await graph.ainvoke(initial_state, config=config)

        # Assertions
        assert len(output["messages"]) == 1
        assert get_message_role(output["messages"][0]) == "assistant"
        assert get_message_content(output["messages"][0]) == "Targeted question?"
        assert output["current_topic"] == "Psychological Baseline"  # Psychology is chosen first as tie-breaker
        assert not output["is_complete"]
        llm.query_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_user_response_updates_scores_and_retrieves_context(self) -> None:
        """Verify that user reply triggers evaluation, memory updates, and target selection."""
        llm = _build_mock_llm_client(
            question_text="Next question?",
            psychology_score=0.8,
            economy_score=0.2,
            politics_score=0.3,
            demographics_score=0.4,
        )
        mem = _build_mock_memory_manager()
        graph = compile_interview_graph(llm, mem)

        agent_id = uuid4()
        state: AgentState = {
            "agent_id": agent_id,
            "messages": [
                {"role": "assistant", "content": "What is your risk tolerance?"},
                {"role": "user", "content": "I prefer low-risk investments."},
            ],
            "current_topic": "Risk Vector",
            "remaining_turns": 10,
            "evaluation_scores": {
                "Psychology": 0.1,
                "Economy": 0.1,
                "Politics": 0.1,
                "Demographics": 0.1,
            },
            "is_complete": False,
        }

        config = {"configurable": {"thread_id": "test-thread-evaluation"}}
        output = await graph.ainvoke(state, config=config)

        # Assertions
        # 1. Memory integration checked
        mem.retrieve_simulation_context.assert_called_once_with(
            agent_id=agent_id, query="I prefer low-risk investments."
        )
        mem.add_episodic_memory.assert_called_once()

        # 2. Score update checked (assert values are monotonically updated)
        assert output["evaluation_scores"]["Psychology"] == 0.8
        assert output["evaluation_scores"]["Economy"] == 0.2
        assert output["evaluation_scores"]["Politics"] == 0.3
        assert output["evaluation_scores"]["Demographics"] == 0.4

        # 3. Decrement turns constraint verified
        assert output["remaining_turns"] == 9

        # 4. Next lowest target selected
        # Economy score is 0.2 (lowest scoring axis). Next focus topic maps to Economy -> Risk Vector
        assert output["current_topic"] == "Risk Vector"
        assert get_message_role(output["messages"][-1]) == "assistant"
        assert get_message_content(output["messages"][-1]) == "Next question?"

    @pytest.mark.asyncio
    async def test_score_saturation_triggers_finalization(self) -> None:
        """Verify that when all evaluation scores exceed 0.85, the graph finalizes."""
        # Setup judge returning saturated scores
        llm = _build_mock_llm_client(
            question_text="Will not be asked",
            psychology_score=0.9,
            economy_score=0.88,
            politics_score=0.92,
            demographics_score=0.85,
        )
        mem = _build_mock_memory_manager()
        graph = compile_interview_graph(llm, mem)

        agent_id = uuid4()
        state: AgentState = {
            "agent_id": agent_id,
            "messages": [
                {"role": "assistant", "content": "Previous question"},
                {"role": "user", "content": "Sufficient reply"},
            ],
            "current_topic": "Risk Vector",
            "remaining_turns": 10,
            "evaluation_scores": {
                "Psychology": 0.8,
                "Economy": 0.8,
                "Politics": 0.8,
                "Demographics": 0.8,
            },
            "is_complete": False,
        }

        config = {"configurable": {"thread_id": "test-thread-saturation"}}
        output = await graph.ainvoke(state, config=config)

        # Assertions
        assert output["is_complete"] is True
        # Since is_complete is True, it routes to finalize and terminates without asking a new question
        assert get_message_role(output["messages"][-1]) == "user"

    @pytest.mark.asyncio
    async def test_exhausted_turns_triggers_finalization(self) -> None:
        """Verify that when remaining_turns hits 0, it terminates even if scores are low."""
        llm = _build_mock_llm_client(
            question_text="Will not be asked",
            psychology_score=0.2,
            economy_score=0.2,
            politics_score=0.2,
            demographics_score=0.2,
        )
        mem = _build_mock_memory_manager()
        graph = compile_interview_graph(llm, mem)

        agent_id = uuid4()
        state: AgentState = {
            "agent_id": agent_id,
            "messages": [
                {"role": "assistant", "content": "Previous question"},
                {"role": "user", "content": "My final answer"},
            ],
            "current_topic": "Psychological Baseline",
            "remaining_turns": 0,
            "evaluation_scores": {
                "Psychology": 0.1,
                "Economy": 0.1,
                "Politics": 0.1,
                "Demographics": 0.1,
            },
            "is_complete": False,
        }

        config = {"configurable": {"thread_id": "test-thread-exhaustion"}}
        output = await graph.ainvoke(state, config=config)

        # Assertions
        assert output["is_complete"] is True
        assert get_message_role(output["messages"][-1]) == "user"

    @pytest.mark.asyncio
    async def test_state_persistence_across_sequential_calls(self) -> None:
        """Verify that state persists across sequential state inputs using MemorySaver."""
        llm = _build_mock_llm_client(
            question_text="Targeted question?",
            psychology_score=0.8,
            economy_score=0.2,
            politics_score=0.3,
            demographics_score=0.4,
        )
        mem = _build_mock_memory_manager()
        graph = compile_interview_graph(llm, mem)

        agent_id = uuid4()
        initial_state: AgentState = {
            "agent_id": agent_id,
            "messages": [],
            "current_topic": "Psychological Baseline",
            "remaining_turns": 10,
            "evaluation_scores": {
                "Psychology": 0.0,
                "Economy": 0.0,
                "Politics": 0.0,
                "Demographics": 0.0,
            },
            "is_complete": False,
        }

        config_1 = {"configurable": {"thread_id": "session-abc-123"}}
        config_2 = {"configurable": {"thread_id": "session-xyz-789"}}

        # 1. Turn 1 (Session 1): Starts interview
        res1 = await graph.ainvoke(initial_state, config=config_1)
        assert len(res1["messages"]) == 1
        assert get_message_role(res1["messages"][0]) == "assistant"
        assert get_message_content(res1["messages"][0]) == "Targeted question?"
        assert res1["remaining_turns"] == 10

        # 2. Turn 1 (Session 2): Starts separate interview, verify isolation
        res2_init = await graph.ainvoke(
            {**initial_state, "remaining_turns": 5}, config=config_2
        )
        assert len(res2_init["messages"]) == 1
        assert res2_init["remaining_turns"] == 5

        # 3. Turn 2 (Session 1): User answers. Resumes graph from checkpoint
        res1_next = await graph.ainvoke(
            {"messages": [{"role": "user", "content": "I prefer low risk."}]},
            config=config_1,
        )
        # Should now have 3 messages: [assistant_question, user_response, next_assistant_question]
        assert len(res1_next["messages"]) == 3
        assert get_message_role(res1_next["messages"][0]) == "assistant"
        assert get_message_role(res1_next["messages"][1]) == "user"
        assert get_message_role(res1_next["messages"][2]) == "assistant"
        assert get_message_content(res1_next["messages"][1]) == "I prefer low risk."
        assert get_message_content(res1_next["messages"][2]) == "Targeted question?"

        # Verify evaluation and score persistence/updates in Session 1
        assert res1_next["remaining_turns"] == 9
        assert res1_next["evaluation_scores"]["Psychology"] == 0.8
        assert res1_next["evaluation_scores"]["Economy"] == 0.2

        # Verify Session 2 remains isolated
        state_s2 = await graph.aget_state(config=config_2)
        assert state_s2.values["remaining_turns"] == 5
        assert len(state_s2.values["messages"]) == 1
