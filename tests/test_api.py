"""
Generative Agent Simulation Engine — FastAPI Integration Test Suite
==================================================================

Verifies route registrations, JSON serialization structures, global exception
handling boundaries, and lifespan integrations under mocked execution states.
"""

from __future__ import annotations

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.main import app
from src.utils.api_client import LLMClientError


# ---------------------------------------------------------------------------
# Test Fixtures & Configurations
# ---------------------------------------------------------------------------


def _setup_mock_services(
    app_instance: any,
    *,
    graph_return_value: dict | None = None,
    memories_data: list | None = None,
) -> tuple[AsyncMock, MagicMock]:
    """Helper to mock app_instance.state attributes for isolated controller testing."""
    # 1. Mock Graph
    mock_graph = AsyncMock()
    if graph_return_value is None:
        graph_return_value = {
            "agent_id": uuid4(),
            "messages": [{"role": "assistant", "content": "What is your primary career goal?"}],
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
    mock_graph.ainvoke.return_value = graph_return_value
    
    # Explicitly configure aget_state as AsyncMock so it can be awaited cleanly
    mock_graph.aget_state = AsyncMock(return_value=MagicMock(values=graph_return_value))
    
    # 2. Mock Memory Manager
    mock_memory_manager = MagicMock()
    mock_supabase = MagicMock()  # Must be MagicMock since .table() chain is synchronous
    
    mock_table = MagicMock()
    mock_select = MagicMock()
    mock_eq = MagicMock()
    mock_order = MagicMock()
    mock_limit = MagicMock()
    
    # Chain methods: .table().select().eq().order().limit().execute()
    mock_supabase.table.return_value = mock_table
    mock_table.select.return_value = mock_select
    mock_select.eq.return_value = mock_eq
    mock_eq.order.return_value = mock_order
    mock_order.limit.return_value = mock_limit
    
    # Mock return payload
    if memories_data is None:
        memories_data = [
            {"content": "Completed college degree in computer science."},
            {"content": "Worked as a software developer for 5 years."},
        ]
    
    execute_result = MagicMock()
    execute_result.data = memories_data
    mock_limit.execute = AsyncMock(return_value=execute_result)
    
    mock_memory_manager.client = mock_supabase
    
    # Cache mocks in the specific running app instance state
    app_instance.state.graph = mock_graph
    app_instance.state.memory_manager = mock_memory_manager
    
    return mock_graph, mock_memory_manager


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_health_check_endpoint() -> None:
    """Verify base endpoint is accessible and returns health metadata."""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


def test_initialize_interview_success() -> None:
    """Verify initialization triggers the graph and yields a valid thread payload."""
    agent_id = uuid4()

    with TestClient(app) as client:
        # Override state objects on client.app inside the test context
        mock_graph, _ = _setup_mock_services(client.app, graph_return_value={
            "agent_id": agent_id,
            "messages": [{"role": "assistant", "content": "Welcome. What is your career goal?"}],
            "current_topic": "Psychological Baseline",
            "remaining_turns": 10,
            "evaluation_scores": {
                "Psychology": 0.0,
                "Economy": 0.0,
                "Politics": 0.0,
                "Demographics": 0.0,
            },
            "is_complete": False,
        })

        payload = {
            "agent_id": str(agent_id),
            "metadata": {"custom_target": 0.90}
        }
        response = client.post("/api/interview/initialize", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "thread_id" in data
        assert data["agent_id"] == str(agent_id)
        assert data["next_question"] == "Welcome. What is your career goal?"
        assert data["remaining_turns"] == 10
        assert data["evaluation_scores"]["Psychology"] == 0.0
        
        # Verify graph invocation structure
        mock_graph.ainvoke.assert_called_once()
        call_args = mock_graph.ainvoke.call_args[0]
        assert call_args[0]["agent_id"] == agent_id
        assert call_args[0]["remaining_turns"] == 10


def test_submit_user_response_advances_dialogue() -> None:
    """Verify that submit reply advances graph turn and increments scores."""
    agent_id = uuid4()
    thread_id = "test-thread-submit"
    
    with TestClient(app) as client:
        # Override state objects on client.app inside the test context
        mock_graph, _ = _setup_mock_services(client.app, graph_return_value={
            "agent_id": agent_id,
            "messages": [
                {"role": "assistant", "content": "Welcome. What is your career goal?"},
                {"role": "user", "content": "I want to be an entrepreneur."},
                {"role": "assistant", "content": "How do you handle financial risks?"},
            ],
            "current_topic": "Risk Vector",
            "remaining_turns": 9,
            "evaluation_scores": {
                "Psychology": 0.4,
                "Economy": 0.1,
                "Politics": 0.0,
                "Demographics": 0.0,
            },
            "is_complete": False,
        })

        payload = {
            "thread_id": thread_id,
            "agent_id": str(agent_id),
            "user_response": "I want to be an entrepreneur."
        }
        response = client.post("/api/interview/submit", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["next_question"] == "How do you handle financial risks?"
        assert data["remaining_turns"] == 9
        assert data["evaluation_scores"]["Psychology"] == 0.4
        assert data["evaluation_scores"]["Economy"] == 0.1
        assert data["is_complete"] is False

        mock_graph.ainvoke.assert_called_once()


def test_submit_response_completion_trigger() -> None:
    """Verify API correctly registers terminal states when graph sets is_complete=True."""
    agent_id = uuid4()
    thread_id = "test-thread-complete"
    
    with TestClient(app) as client:
        # Override state objects on client.app inside the test context
        mock_graph, _ = _setup_mock_services(client.app, graph_return_value={
            "agent_id": agent_id,
            "messages": [
                {"role": "assistant", "content": "Welcome. What is your career goal?"},
                {"role": "user", "content": "I want to be an entrepreneur."},
            ],
            "current_topic": "Risk Vector",
            "remaining_turns": 0,
            "evaluation_scores": {
                "Psychology": 0.9,
                "Economy": 0.9,
                "Politics": 0.95,
                "Demographics": 0.88,
            },
            "is_complete": True,
        })

        payload = {
            "thread_id": thread_id,
            "agent_id": str(agent_id),
            "user_response": "Final answers."
        }
        response = client.post("/api/interview/submit", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Next question should be blank on completion
        assert data["next_question"] == ""
        assert data["remaining_turns"] == 0
        assert data["is_complete"] is True


def test_agent_status_snapshot() -> None:
    """Verify endpoint retrieves metrics from checkpoint and memories from DB."""
    agent_id = uuid4()

    with TestClient(app) as client:
        # Override state objects on client.app inside the test context
        _setup_mock_services(
            client.app,
            graph_return_value={
                "agent_id": agent_id,
                "messages": [],
                "current_topic": "Risk Vector",
                "remaining_turns": 5,
                "evaluation_scores": {
                    "Psychology": 0.75,
                    "Economy": 0.80,
                    "Politics": 0.50,
                    "Demographics": 0.90,
                },
                "is_complete": False,
            },
            memories_data=[
                {"content": "Grew up in a small town."},
                {"content": "Values long-term job security."},
            ]
        )

        response = client.get(f"/api/agent/{agent_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["agent_id"] == str(agent_id)
        assert data["evaluation_scores"]["Psychology"] == 0.75
        assert data["evaluation_scores"]["Economy"] == 0.80
        assert len(data["recent_memories"]) == 2
        assert "Grew up in a small town." in data["recent_memories"]


def test_llm_rate_limit_exception_handling() -> None:
    """Verify that upstream LLM rate limits map cleanly to HTTP 429 response."""
    agent_id = uuid4()

    with TestClient(app, raise_server_exceptions=False) as client:
        # Override state objects on client.app inside the test context
        mock_graph, _ = _setup_mock_services(client.app)
        # Setup graph.ainvoke to raise LLM Rate Limit Error
        mock_graph.ainvoke.side_effect = LLMClientError("Upstream API Rate Limit (429): TPM exceeded")

        payload = {
            "agent_id": str(agent_id),
            "metadata": {}
        }
        response = client.post("/api/interview/initialize", json=payload)
        
        assert response.status_code == 429
        data = response.json()
        assert data["error"] == "Upstream LLM Rate Limit"
        assert "TPM exceeded" in data["details"]


def test_general_internal_error_handling() -> None:
    """Verify that general unhandled exceptions map to HTTP 500 response."""
    agent_id = uuid4()

    with TestClient(app, raise_server_exceptions=False) as client:
        # Override state objects on client.app inside the test context
        mock_graph, _ = _setup_mock_services(client.app)
        # Setup graph.ainvoke to raise generic DB/connection error
        mock_graph.ainvoke.side_effect = RuntimeError("Supabase connection timed out")

        payload = {
            "agent_id": str(agent_id),
            "metadata": {}
        }
        response = client.post("/api/interview/initialize", json=payload)
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal Server Error"
        assert "Supabase connection timed out" in data["details"]
