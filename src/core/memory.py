"""
Generative Agent Simulation Engine — Memory Stream Infrastructure
==================================================================

Provides the database integration layer for episodic memory ingestion
and context retrieval. It communicates with Supabase asynchronously,
generates embeddings via Gemini text-embedding-004, calculates psychological
importance scores, and runs the Stanford three-factor retrieval logic.

Connection Management
---------------------
The ``MemoryStreamManager`` accepts an **optional** pre-initialized
``AsyncClient`` via its constructor.  When none is provided, the caller
must invoke the ``initialize()`` coroutine **exactly once** before any
database operations.  This guarantees a single persistent connection is
reused across all high-frequency transactions, eliminating the
``OSError: [Errno 24] Too many open files`` risk inherent in per-call
client creation.
"""

from __future__ import annotations

import os
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field
from supabase import AsyncClient, acreate_client

from config.logging_config import get_logger
from src.utils.api_client import LLMClientError, ResilientLLMClient

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured Pydantic Models for LLM Queries
# ---------------------------------------------------------------------------


class MemoryImportanceRating(BaseModel):
    """Structured Pydantic model for memory importance scoring."""

    importance_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="The social/psychological importance rating of the memory, strictly between 1 (trivial) and 10 (life-altering).",
    )


# ---------------------------------------------------------------------------
# Memory Stream Manager
# ---------------------------------------------------------------------------


class MemoryStreamManager:
    """Manages the agent memory stream, handling uploads, embeddings, and retrievals.

    The manager maintains a **single persistent** ``AsyncClient`` connection
    to Supabase for its entire lifecycle instead of spawning one per call.

    There are two ways to provide the Supabase client:

    1. **Constructor injection** (preferred for testing and singleton patterns):
       Pass an already-initialized ``AsyncClient`` via the ``supabase_client``
       keyword argument.  No further setup is required.

    2. **Deferred initialization** (production convenience):
       Omit ``supabase_client`` and call ``await manager.initialize()``
       once before issuing any database operations.  The manager reads
       ``SUPABASE_URL`` and ``SUPABASE_SERVICE_KEY`` / ``SUPABASE_KEY``
       from the environment and creates the client internally.

    The recency decay rate used in three-factor retrieval defaults to
    ``0.01`` and can be overridden via the ``MEMORY_DECAY_RATE``
    environment variable.
    """

    def __init__(
        self,
        llm_client: ResilientLLMClient,
        *,
        supabase_client: Optional[AsyncClient] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
    ) -> None:
        """Initialize the MemoryStreamManager.

        Parameters
        ----------
        llm_client : ResilientLLMClient
            The resilient LLM client used for embeddings and importance rating.
        supabase_client : AsyncClient | None
            A pre-initialized async Supabase client.  When supplied, the
            manager uses it directly and ``supabase_url`` / ``supabase_key``
            are ignored.
        supabase_url : str | None
            The Supabase project URL.  Used only when ``supabase_client``
            is ``None``.  Falls back to ``SUPABASE_URL`` env var.
        supabase_key : str | None
            The Supabase Service/Anon key.  Used only when
            ``supabase_client`` is ``None``.  Falls back to
            ``SUPABASE_SERVICE_KEY`` or ``SUPABASE_KEY`` env vars.
        """
        self._llm_client = llm_client

        # Configurable decay rate (defaults to 0.01)
        self._decay_rate: float = float(
            os.getenv("MEMORY_DECAY_RATE", "0.01")
        )

        if supabase_client is not None:
            # Injected client — ready to use immediately
            self._client: Optional[AsyncClient] = supabase_client
            self._supabase_url: Optional[str] = None
            self._supabase_key: Optional[str] = None
            logger.info(
                "MemoryStreamManager initialized with injected client | decay_rate=%.4f",
                self._decay_rate,
            )
        else:
            # Deferred initialization — resolve credentials now, connect later
            self._client = None
            self._supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
            self._supabase_key = (
                supabase_key
                or os.environ.get("SUPABASE_SERVICE_KEY")
                or os.environ.get("SUPABASE_KEY")
            )

            if not self._supabase_url or not self._supabase_key:
                raise ValueError(
                    "SUPABASE_URL and SUPABASE_SERVICE_KEY / SUPABASE_KEY "
                    "must be provided or set in environment variables when "
                    "no supabase_client is injected."
                )
            logger.info(
                "MemoryStreamManager initialized (deferred connect) | decay_rate=%.4f",
                self._decay_rate,
            )

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create and store the persistent async Supabase client.

        Must be called **exactly once** when the manager was created
        without an injected ``supabase_client``.  Calling it when the
        client is already set is a safe no-op.
        """
        if self._client is not None:
            logger.debug("initialize() called but client already exists — no-op")
            return

        if not self._supabase_url or not self._supabase_key:
            raise RuntimeError(
                "Cannot initialize: Supabase URL / key not configured."
            )

        logger.info("Creating persistent async Supabase client")
        self._client = await acreate_client(
            self._supabase_url, self._supabase_key
        )
        logger.info("Persistent Supabase client created successfully")

    @property
    def client(self) -> AsyncClient:
        """Return the persistent Supabase client, raising if not yet initialized."""
        if self._client is None:
            raise RuntimeError(
                "MemoryStreamManager has no active Supabase client. "
                "Call `await manager.initialize()` first or inject a "
                "client via the constructor."
            )
        return self._client

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def add_episodic_memory(self, agent_id: UUID, content: str) -> UUID:
        """Generate embedding, rate importance, and insert episodic memory into Supabase.

        Parameters
        ----------
        agent_id : UUID
            The target agent identity UUID.
        content : str
            The text content representing the episodic memory.

        Returns
        -------
        UUID
            The primary key UUID of the newly inserted memory.

        Raises
        ------
        EmbeddingGenerationError
            If embedding generation fails.
        LLMClientError
            If the importance scoring call fails.
        RuntimeError
            If no Supabase client is active, or the insert returns no rows.
        Exception
            For database connection drops or serialization errors.
        """
        logger.info(
            "Adding episodic memory | agent_id=%s | content_len=%d",
            agent_id,
            len(content),
        )

        # 1. Fetch the 768-dimension vector embedding
        embedding: List[float] = await self._llm_client.get_embedding(content)

        # 2. Extract importance from pre-enriched content (set by pipeline's
        #    MemoryEnrichment LLM call). This avoids a redundant Groq API call
        #    that was burning through the free-tier rate limit.
        score = 5  # sensible default
        try:
            import json as _json
            parsed = _json.loads(content)
            if isinstance(parsed, dict) and "importance" in parsed:
                score = int(parsed["importance"])
        except (ValueError, TypeError, _json.JSONDecodeError):
            pass  # not JSON-enriched content — use default

        logger.info(
            "Importance extracted from enriched content | agent_id=%s | rating=%d",
            agent_id,
            score,
        )

        # 3. Format and insert directly into Supabase 'memories' table
        db = self.client
        try:
            insert_payload = {
                "agent_id": str(agent_id),
                "content": content,
                "importance_score": score,
                "embedding": embedding,
            }

            logger.info("Inserting memory into Supabase memories table...")
            response = (
                await db.table("memories").insert(insert_payload).execute()
            )

            if not response.data:
                raise RuntimeError(
                    "Database insert succeeded but returned no rows."
                )

            memory_uuid = UUID(response.data[0]["id"])
            logger.info(
                "Memory stored successfully | memory_id=%s", memory_uuid
            )
            return memory_uuid

        except Exception as e:
            logger.critical(
                "Database error while saving memory | agent_id=%s | error=%s",
                agent_id,
                str(e),
            )
            raise

    async def retrieve_simulation_context(
        self,
        agent_id: UUID,
        query: str,
        limit: int = 5,
        w_r: float = 0.4,
        w_i: float = 0.3,
        w_s: float = 0.3,
    ) -> List[str]:
        """Retrieve relevant memories compiled using the Three-Factor Stanford formula.

        Parameters
        ----------
        agent_id : UUID
            The target agent identity UUID.
        query : str
            The query text to search relevance against.
        limit : int
            The maximum number of memories to return. Defaults to 5.
        w_r : float
            Recency weight. Defaults to 0.4.
        w_i : float
            Importance weight. Defaults to 0.3.
        w_s : float
            Similarity weight. Defaults to 0.3.

        Returns
        -------
        List[str]
            A list of content strings retrieved as context.
        """
        logger.info(
            "Retrieving simulation context | agent_id=%s | query=%s | limit=%d",
            agent_id,
            query,
            limit,
        )

        # 1. Generate the 768 vector embedding for the query string
        query_embedding: List[float] = await self._llm_client.get_embedding(
            query
        )

        # 2. Call the Supabase RPC function 'match_memories'
        db = self.client
        params = {
            "query_agent_id": str(agent_id),
            "query_embedding": query_embedding,
            "w_r": w_r,
            "w_i": w_i,
            "w_s": w_s,
            "match_limit": limit,
            "p_decay_rate": self._decay_rate,
        }

        try:
            logger.info(
                "Calling match_memories RPC | decay_rate=%.4f",
                self._decay_rate,
            )
            response = await db.rpc("match_memories", params).execute()

            # 3. Extract the array of text contents
            memories: List[str] = [row["content"] for row in response.data]
            logger.info("Retrieved %d matched memories", len(memories))
            return memories

        except Exception as e:
            logger.critical(
                "RPC match_memories execution failed | agent_id=%s | error=%s",
                agent_id,
                str(e),
            )
            raise
