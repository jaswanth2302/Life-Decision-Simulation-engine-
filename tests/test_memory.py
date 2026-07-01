"""
Generative Agent Simulation Engine — Memory Stream Test Suite
=============================================================

pytest module verifying the Three-Factor Stanford Memory Stream:
1. **Decay & Scoring Math** — Tests that recency, importance, and similarity
   combine accurately, and that highly important old memories can outscore
   recent trivial ones.
2. **Weight Linearity** — Verifies that modifications of w_r, w_i, and w_s
   scale the final computed score perfectly linearly.
3. **Input Resilience** — Ensures long string inputs do not crash the
   embedding layer.
4. **Mocked Database Operations** — Validates Supabase insert and RPC calls
   using a single persistent mock client injected via the constructor.
5. **Configurable Decay Rate** — Verifies p_decay_rate is read from env
   and forwarded in the RPC payload.
"""

from __future__ import annotations

import math
import os
from typing import Any, List
from uuid import UUID, uuid4

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.memory import MemoryImportanceRating, MemoryStreamManager
from src.utils.api_client import EmbeddingGenerationError, ResilientLLMClient


# ---------------------------------------------------------------------------
# Stanford Three-Factor Math Helper
# ---------------------------------------------------------------------------


def calculate_stanford_score(
    hours_elapsed: float,
    importance_score: int,
    cosine_similarity: float,
    w_r: float,
    w_i: float,
    w_s: float,
    lambda_decay: float = 0.01,
) -> float:
    """Python implementation of the Three-Factor memory retrieval score.

    Score = w_r * Recency + w_i * Importance + w_s * Similarity
    """
    # Recency: exp(-lambda * t), bound strictly to [0.0, 1.0]
    recency = math.exp(-lambda_decay * hours_elapsed)
    recency = min(1.0, max(0.0, recency))

    # Importance: linear normalization from [1, 10] to [0.0, 1.0]
    importance = (importance_score - 1.0) / 9.0

    # Score calculation
    return (w_r * recency) + (w_i * importance) + (w_s * cosine_similarity)


# ---------------------------------------------------------------------------
# Helper: Build a mock LLM client and a mock Supabase AsyncClient
# ---------------------------------------------------------------------------


def _build_mock_llm_client(
    *,
    embedding: List[float] | None = None,
    importance: int = 5,
) -> ResilientLLMClient:
    """Return a ResilientLLMClient with mocked async methods."""
    client = ResilientLLMClient.__new__(ResilientLLMClient)
    client.get_embedding = AsyncMock(
        return_value=embedding if embedding is not None else [0.1] * 768
    )
    client.query_structured = AsyncMock(
        return_value=MemoryImportanceRating(importance_score=importance)
    )
    return client


def _build_mock_supabase_client(
    *,
    insert_id: str | None = None,
    rpc_rows: list[dict[str, Any]] | None = None,
) -> AsyncMock:
    """Return a mock AsyncClient wired for table().insert() and rpc() flows."""
    mock_client = AsyncMock()

    # --- table("memories").insert(data).execute() chain ---
    mock_table = MagicMock()
    mock_insert_response = MagicMock()
    mock_insert_response.data = [{"id": insert_id or str(uuid4())}]
    mock_table.insert = MagicMock(return_value=mock_table)
    mock_table.execute = AsyncMock(return_value=mock_insert_response)
    mock_client.table = MagicMock(return_value=mock_table)

    # --- rpc("match_memories", params).execute() chain ---
    mock_rpc_response = MagicMock()
    mock_rpc_response.data = rpc_rows if rpc_rows is not None else []
    mock_client.rpc = MagicMock(return_value=mock_client)
    mock_client.execute = AsyncMock(return_value=mock_rpc_response)

    return mock_client


# ---------------------------------------------------------------------------
# Test 1 — Decay & Scoring Math
# ---------------------------------------------------------------------------


class TestMemoryDecayMath:
    """Verify the decay, importance, and similarity formula rules."""

    def test_important_old_memory_outscores_recent_trivial_memory(self) -> None:
        """Highly important old memory (cosine similarity = 1.0) should
        outscore a very recent trivial memory (weak similarity).
        """
        # Memory A: Life-altering decision, 5 days old (120 hours), perfect query match
        score_A = calculate_stanford_score(
            hours_elapsed=120.0,
            importance_score=10,
            cosine_similarity=1.0,
            w_r=0.4,
            w_i=0.3,
            w_s=0.3,
        )

        # Memory B: Trivial event (eating lunch), 10 mins old (0.16 hours), weak query match
        score_B = calculate_stanford_score(
            hours_elapsed=0.16,
            importance_score=2,
            cosine_similarity=0.2,
            w_r=0.4,
            w_i=0.3,
            w_s=0.3,
        )

        # Print outputs for debugging visibility
        print(f"Memory A (Important/Old) Score: {score_A:.4f}")
        print(f"Memory B (Trivial/New) Score: {score_B:.4f}")

        # Assert Memory A wins
        assert score_A > score_B
        assert score_A > 0.7
        assert score_B < 0.5

    def test_score_scales_linearly_with_weight_modifications(self) -> None:
        """Score must scale perfectly linearly relative to weight parameter shifts."""
        hours = 24.0
        importance = 7
        similarity = 0.8

        # Base factors
        recency = math.exp(-0.01 * hours)

        # Test Case 1: Base weights
        score_1 = calculate_stanford_score(
            hours_elapsed=hours,
            importance_score=importance,
            cosine_similarity=similarity,
            w_r=0.4,
            w_i=0.3,
            w_s=0.3,
        )

        # Test Case 2: Double the recency weight (w_r = 0.8)
        score_2 = calculate_stanford_score(
            hours_elapsed=hours,
            importance_score=importance,
            cosine_similarity=similarity,
            w_r=0.8,
            w_i=0.3,
            w_s=0.3,
        )

        # Delta should be exactly w_delta * recency
        delta_score = score_2 - score_1
        expected_delta = 0.4 * recency

        assert math.isclose(delta_score, expected_delta, rel_tol=1e-9)

        # Test Case 3: Zero out importance, distribute to similarity
        score_3 = calculate_stanford_score(
            hours_elapsed=hours,
            importance_score=importance,
            cosine_similarity=similarity,
            w_r=0.4,
            w_i=0.0,
            w_s=0.6,
        )
        expected_score_3 = (0.4 * recency) + (0.6 * similarity)
        assert math.isclose(score_3, expected_score_3, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Test 2 — MemoryStreamManager with Persistent Injected Client
# ---------------------------------------------------------------------------


class TestMemoryStreamManagerOperations:
    """Verify MemoryStreamManager integration with LLM and database layers.

    All tests inject a mock ``AsyncClient`` directly via the constructor
    rather than patching ``acreate_client``, mirroring the production
    singleton pattern and validating that no new connections are spawned.
    """

    @pytest.mark.asyncio
    async def test_add_episodic_memory_flow(self) -> None:
        """Verify adding a memory generates embedding, rates importance, and inserts."""
        new_memory_id = str(uuid4())
        llm_client = _build_mock_llm_client(importance=8)
        mock_db = _build_mock_supabase_client(insert_id=new_memory_id)

        manager = MemoryStreamManager(
            llm_client=llm_client,
            supabase_client=mock_db,
        )

        agent_id = uuid4()
        memory_content = "Relocating to London for a PM role."
        memory_uuid = await manager.add_episodic_memory(agent_id, memory_content)

        # Assertions
        assert memory_uuid == UUID(new_memory_id)
        llm_client.get_embedding.assert_called_once_with(memory_content)
        llm_client.query_structured.assert_called_once()
        mock_db.table.assert_called_once_with("memories")

        # Verify insert payload
        insert_call = mock_db.table.return_value.insert
        insert_call.assert_called_once_with({
            "agent_id": str(agent_id),
            "content": memory_content,
            "importance_score": 8,
            "embedding": [0.1] * 768,
        })

    @pytest.mark.asyncio
    async def test_retrieve_simulation_context_flow(self) -> None:
        """Verify retrieval triggers embedding query and match_memories RPC call."""
        dummy_query_embedding = [0.5] * 768
        llm_client = _build_mock_llm_client(embedding=dummy_query_embedding)
        mock_db = _build_mock_supabase_client(rpc_rows=[
            {"content": "Memory 1", "importance_score": 9, "score": 0.85},
            {"content": "Memory 2", "importance_score": 6, "score": 0.72},
        ])

        manager = MemoryStreamManager(
            llm_client=llm_client,
            supabase_client=mock_db,
        )

        agent_id = uuid4()
        context = await manager.retrieve_simulation_context(
            agent_id=agent_id,
            query="London job relocation",
            limit=2,
            w_r=0.4,
            w_i=0.3,
            w_s=0.3,
        )

        assert context == ["Memory 1", "Memory 2"]
        llm_client.get_embedding.assert_called_once_with("London job relocation")
        mock_db.rpc.assert_called_once_with(
            "match_memories",
            {
                "query_agent_id": str(agent_id),
                "query_embedding": dummy_query_embedding,
                "w_r": 0.4,
                "w_i": 0.3,
                "w_s": 0.3,
                "match_limit": 2,
                "p_decay_rate": 0.01,
            },
        )

    @pytest.mark.asyncio
    async def test_uninitialized_client_raises_runtime_error(self) -> None:
        """Calling add_episodic_memory without a client must raise RuntimeError."""
        llm_client = _build_mock_llm_client()

        manager = MemoryStreamManager(
            llm_client=llm_client,
            supabase_url="https://example.supabase.co",
            supabase_key="testkey",
        )

        with pytest.raises(RuntimeError, match="no active Supabase client"):
            await manager.add_episodic_memory(uuid4(), "test content")

    @pytest.mark.asyncio
    @patch("src.core.memory.acreate_client")
    async def test_deferred_initialize_creates_client_once(
        self, mock_acreate: MagicMock
    ) -> None:
        """initialize() must call acreate_client exactly once."""
        mock_db = _build_mock_supabase_client()
        mock_acreate.return_value = mock_db

        llm_client = _build_mock_llm_client()
        manager = MemoryStreamManager(
            llm_client=llm_client,
            supabase_url="https://example.supabase.co",
            supabase_key="testkey",
        )

        # First call creates client
        await manager.initialize()
        mock_acreate.assert_called_once()

        # Second call is a safe no-op
        await manager.initialize()
        mock_acreate.assert_called_once()


# ---------------------------------------------------------------------------
# Test 3 — Configurable Decay Rate
# ---------------------------------------------------------------------------


class TestConfigurableDecayRate:
    """Verify that MEMORY_DECAY_RATE env var is respected."""

    @pytest.mark.asyncio
    async def test_custom_decay_rate_forwarded_to_rpc(self) -> None:
        """When MEMORY_DECAY_RATE=0.05, the RPC call must include p_decay_rate=0.05."""
        dummy_embedding = [0.2] * 768
        llm_client = _build_mock_llm_client(embedding=dummy_embedding)
        mock_db = _build_mock_supabase_client(rpc_rows=[])

        with patch.dict(os.environ, {"MEMORY_DECAY_RATE": "0.05"}):
            manager = MemoryStreamManager(
                llm_client=llm_client,
                supabase_client=mock_db,
            )

        agent_id = uuid4()
        await manager.retrieve_simulation_context(
            agent_id=agent_id, query="test query"
        )

        # Extract the p_decay_rate from the RPC call arguments
        rpc_call_args = mock_db.rpc.call_args[0]
        rpc_params = rpc_call_args[1]
        assert rpc_params["p_decay_rate"] == 0.05

    @pytest.mark.asyncio
    async def test_default_decay_rate_is_001(self) -> None:
        """Without MEMORY_DECAY_RATE env var, default should be 0.01."""
        dummy_embedding = [0.2] * 768
        llm_client = _build_mock_llm_client(embedding=dummy_embedding)
        mock_db = _build_mock_supabase_client(rpc_rows=[])

        # Ensure the env var is absent
        env_copy = os.environ.copy()
        env_copy.pop("MEMORY_DECAY_RATE", None)
        with patch.dict(os.environ, env_copy, clear=True):
            manager = MemoryStreamManager(
                llm_client=llm_client,
                supabase_client=mock_db,
            )

        agent_id = uuid4()
        await manager.retrieve_simulation_context(
            agent_id=agent_id, query="test query"
        )

        rpc_call_args = mock_db.rpc.call_args[0]
        rpc_params = rpc_call_args[1]
        assert rpc_params["p_decay_rate"] == 0.01


# ---------------------------------------------------------------------------
# Test 4 — Embedding Input Resilience
# ---------------------------------------------------------------------------


class TestEmbeddingInputResilience:
    """Verify that inputting strings longer than typical limits does not crash the embedding layer."""

    @pytest.mark.asyncio
    async def test_extremely_long_string_does_not_crash_embedding(self) -> None:
        """Test that the embedding layer handles a massive string without crashing."""
        client = ResilientLLMClient(gemini_api_key="gemini-key")

        dummy_embedding = [0.0] * 768
        mock_embedding_obj = MagicMock()
        mock_embedding_obj.values = dummy_embedding
        mock_response = MagicMock()
        mock_response.embeddings = [mock_embedding_obj]

        client._gemini_client = MagicMock()
        client._gemini_client.aio.models.embed_content = AsyncMock(return_value=mock_response)

        try:
            huge_string = "a" * 100000
            result = await client.get_embedding(huge_string)

            assert len(result) == 768
            assert result == dummy_embedding
            from google.genai import types
            client._gemini_client.aio.models.embed_content.assert_called_once_with(
                model="gemini-embedding-001",
                contents=huge_string,
                config=types.EmbedContentConfig(output_dimensionality=768),
            )
        finally:
            await client.close()
