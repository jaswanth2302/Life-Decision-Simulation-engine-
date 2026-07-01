"""
Generative Agent Simulation Engine — Reflection Pipeline Test Suite
===================================================================

Comprehensive pytest-asyncio tests verifying the Phase 2 parallel
reflection pipeline's integrity, resilience, and graceful degradation:

1. **Happy-Path Integration Test** — All four experts return valid JSON;
   the resulting ``AgentProfile`` is fully populated and immutable.
2. **Rate-Limit Resilience Test** — Simulates 429 responses followed by
   success; asserts exponential backoff recovery without exceptions.
3. **Graceful Degradation Test** — Political Scientist encounters a
   terminal 500 error; the other three experts succeed; pipeline completes
   with ``political_matrix=None``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIStatusError, RateLimitError
from pydantic import ValidationError

from src.core.schema import (
    AgentProfile,
    DemographicBaseline,
    EconomicDecisionMatrix,
    PsychologicalProfile,
    SocioPoliticalWorldview,
)
from src.pipelines.reflection import (
    EXPERT_PERSONAS,
    generate_expert_reflection_matrix,
)
from src.utils.api_client import (
    LLMClientError,
    LLMRateLimitError,
    LLMServerError,
    ResilientLLMClient,
)


# ═══════════════════════════════════════════════════════════════════════════
# Mock Data — Realistic Expert Analysis Payloads
# ═══════════════════════════════════════════════════════════════════════════

MOCK_TRANSCRIPT: str = (
    "I started my career as a junior data analyst in 2014. The role was "
    "comfortable but I quickly realized I was drawn to the strategic side "
    "of things. After eighteen months I moved into product management at a "
    "fintech startup. That was transformative. I learned to balance "
    "engineering constraints with user needs and business viability. The "
    "pace was relentless but I thrived on the autonomy. Three years later "
    "I transitioned to a senior PM role at a larger organization for "
    "enterprise-scale decision-making exposure. The biggest decision I faced "
    "was relocating internationally. I built a weighted decision matrix with "
    "fifteen factors. I also ran a pre-mortem exercise. The quantitative "
    "analysis narrowly favored staying, but my gut strongly disagreed and "
    "I went with the move. My trust in institutions has eroded but in a "
    "nuanced way. I still believe in institutional frameworks themselves "
    "but my confidence in the people operating them has diminished. I grew "
    "up in a mid-sized suburban area with a strong sense of community but "
    "limited economic diversity. My parents emphasized education as the "
    "primary vehicle for upward mobility."
)

MOCK_PSYCHOLOGIST_RESPONSE: dict[str, Any] = {
    "core_values": [
        "autonomy",
        "intellectual growth",
        "family stability",
        "civic responsibility",
        "meritocratic fairness",
    ],
    "coping_mechanisms": [
        "structured analysis (decision matrices)",
        "pre-mortem catastrophizing",
        "physical exercise under stress",
        "social withdrawal for processing",
    ],
    "cognitive_anchors": [
        "education as upward mobility vehicle",
        "institutional reform over revolution",
        "data-driven reasoning with gut-check override",
    ],
    "emotional_stability_index": 0.72,
}

MOCK_ECONOMIST_RESPONSE: dict[str, Any] = {
    "risk_appetite_score": 0.65,
    "loss_aversion_ratio": 1.85,
    "heuristic_biases": [
        "anchoring to quantitative analysis",
        "status quo bias (mild)",
        "sunk cost sensitivity in career decisions",
        "optimism bias in relocation planning",
    ],
    "resource_allocation_rules": [
        "prioritize long-term career capital over short-term compensation",
        "maintain 6-month emergency fund before discretionary spending",
        "allocate minimum 10% of income to skill development",
        "time-box exploratory activities to prevent scope creep",
    ],
}

MOCK_POLITICAL_RESPONSE: dict[str, Any] = {
    "institutional_trust_score": 0.45,
    "ideological_anchors": [
        "critical institutionalism",
        "reform-oriented pragmatism",
        "evidence-based policy preference",
        "civic engagement as duty",
    ],
    "authority_bias_flag": False,
    "conflict_resolution_style": "collaborative",
}

MOCK_DEMOGRAPHER_RESPONSE: dict[str, Any] = {
    "structural_constraints": [
        "suburban upbringing with limited economic diversity exposure",
        "first-generation professional in technology sector",
        "geographic mobility constrained by family proximity preference",
    ],
    "geographic_context": "suburban mid-sized metro area, temperate climate",
    "socio_economic_tier": "middle",
    "background_intersection_notes": (
        "Subject exhibits strong intergenerational mobility aspirations "
        "shaped by stable but economically homogeneous upbringing."
    ),
}

# Map persona labels to their mock response payloads
_MOCK_RESPONSES: dict[str, dict[str, Any]] = {
    "psychologist": MOCK_PSYCHOLOGIST_RESPONSE,
    "economist": MOCK_ECONOMIST_RESPONSE,
    "political_scientist": MOCK_POLITICAL_RESPONSE,
    "demographer": MOCK_DEMOGRAPHER_RESPONSE,
}


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Create a mock LLM client that returns pre-built payloads
# ═══════════════════════════════════════════════════════════════════════════


def _build_mock_client(
    *,
    failing_persona: str | None = None,
    failure_error: Exception | None = None,
) -> ResilientLLMClient:
    """Build a ResilientLLMClient with mocked query_structured method.

    Parameters
    ----------
    failing_persona : str | None
        If set, this persona label will raise ``failure_error`` instead
        of returning a valid response.
    failure_error : Exception | None
        The exception to raise for the failing persona.
    """
    client = ResilientLLMClient.__new__(ResilientLLMClient)
    # Bypass __init__ — we're mocking the query method directly.
    client._client = AsyncMock()

    original_query = AsyncMock()

    async def mock_query_structured(
        *,
        system_prompt: str,
        user_prompt: str,
        target_model: type,
        persona_label: str = "expert",
    ) -> Any:
        if failing_persona and persona_label == failing_persona:
            raise failure_error  # type: ignore[misc]

        mock_data = _MOCK_RESPONSES[persona_label]
        return target_model.model_validate(mock_data)

    client.query_structured = mock_query_structured  # type: ignore[assignment]
    client.close = AsyncMock()  # type: ignore[assignment]
    return client


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — Happy-Path Integration Test
# ═══════════════════════════════════════════════════════════════════════════


class TestHappyPathIntegration:
    """Verify that the pipeline produces a fully populated AgentProfile
    when all four expert personas return valid responses."""

    @pytest.mark.asyncio
    async def test_full_pipeline_produces_complete_profile(self) -> None:
        """All four experts succeed → fully populated AgentProfile."""
        client = _build_mock_client()

        profile: AgentProfile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT,
            llm_client=client,
        )

        # All matrices should be populated
        assert profile.psychologist_matrix is not None
        assert profile.economist_matrix is not None
        assert profile.political_matrix is not None
        assert profile.demographer_matrix is not None

        # Verify metadata
        assert profile.metadata.agent_id is not None
        assert profile.metadata.created_at is not None
        assert profile.metadata.total_interview_word_count > 0
        assert profile.metadata.target_accuracy_score == 0.85

        # Verify transcript was parsed
        assert len(profile.raw_transcript) > 0

    @pytest.mark.asyncio
    async def test_psychologist_matrix_values(self) -> None:
        """Psychologist matrix should contain the expected mock values."""
        client = _build_mock_client()
        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        psych = profile.psychologist_matrix
        assert psych is not None
        assert "autonomy" in psych.core_values
        assert psych.emotional_stability_index == 0.72
        assert len(psych.coping_mechanisms) == 4

    @pytest.mark.asyncio
    async def test_economist_matrix_values(self) -> None:
        """Economist matrix should contain the expected mock values."""
        client = _build_mock_client()
        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        econ = profile.economist_matrix
        assert econ is not None
        assert econ.risk_appetite_score == 0.65
        assert econ.loss_aversion_ratio == 1.85
        assert len(econ.heuristic_biases) == 4

    @pytest.mark.asyncio
    async def test_profile_is_immutable(self) -> None:
        """The returned AgentProfile must be frozen (immutable)."""
        client = _build_mock_client()
        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        with pytest.raises(ValidationError):
            profile.psychologist_matrix = None  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_profile_json_round_trip(self) -> None:
        """Full profile should survive JSON serialization round-trip."""
        client = _build_mock_client()
        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        json_str = profile.model_dump_json()
        restored = AgentProfile.model_validate_json(json_str)
        assert restored == profile


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — Rate-Limit Resilience Test
# ═══════════════════════════════════════════════════════════════════════════


class TestRateLimitResilience:
    """Verify that the client handles 429 Rate Limit responses correctly,
    performing exponential backoff and eventually recovering."""

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit_then_succeeds(self) -> None:
        """Client should retry on 429 and succeed when the API recovers."""
        # Build a real client but mock the underlying Groq API call
        client = ResilientLLMClient(
            gemini_api_key="test-key",
            base_delay=0.01,  # Very short delay for fast tests
            max_retries=3,
        )

        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = json.dumps(MOCK_PSYCHOLOGIST_RESPONSE)

        # Build the sequence: 429, 429, then success
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        from groq import APIStatusError as GroqAPIStatusError
        import httpx
        rate_limit_error = GroqAPIStatusError(
            "Rate limit exceeded",
            response=httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.groq.com")),
            body=None
        )

        call_count = 0

        async def mock_chat_create(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise rate_limit_error
            return mock_response

        client._groq_client = MagicMock()
        client._groq_client.chat.completions.create = AsyncMock(side_effect=mock_chat_create)

        try:
            result = await client.query_structured(
                system_prompt="You are a psychologist.",
                user_prompt="Analyze this transcript.",
                target_model=PsychologicalProfile,
                persona_label="psychologist",
            )

            # Should have succeeded after retries
            assert isinstance(result, PsychologicalProfile)
            assert result.emotional_stability_index == 0.72
            assert call_count == 3  # 2 failures + 1 success
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises_rate_limit_error(self) -> None:
        """Client should raise LLMRateLimitError when all retries fail."""
        client = ResilientLLMClient(
            gemini_api_key="test-key",
            base_delay=0.01,
            max_retries=3,
        )

        from groq import APIStatusError as GroqAPIStatusError
        import httpx
        rate_limit_error = GroqAPIStatusError(
            "Rate limit exceeded",
            response=httpx.Response(status_code=429, request=httpx.Request("POST", "https://api.groq.com")),
            body=None
        )

        async def always_rate_limit(**kwargs: Any) -> Any:
            raise rate_limit_error

        client._groq_client = MagicMock()
        client._groq_client.chat.completions.create = AsyncMock(side_effect=always_rate_limit)

        try:
            with pytest.raises(LLMRateLimitError):
                await client.query_structured(
                    system_prompt="You are a psychologist.",
                    user_prompt="Analyze this transcript.",
                    target_model=PsychologicalProfile,
                    persona_label="psychologist",
                )
        finally:
            await client.close()


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — Graceful Degradation Test
# ═══════════════════════════════════════════════════════════════════════════


class TestGracefulDegradation:
    """Verify that the pipeline completes successfully when individual
    expert personas fail, leaving their matrices as None."""

    @pytest.mark.asyncio
    async def test_political_scientist_failure_degrades_gracefully(self) -> None:
        """Political scientist 500 failure → political_matrix is None,
        other three matrices are fully populated."""
        server_error = LLMServerError(
            "All 3 retries exhausted for persona=political_scientist"
        )

        client = _build_mock_client(
            failing_persona="political_scientist",
            failure_error=server_error,
        )

        profile: AgentProfile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT,
            llm_client=client,
        )

        # Political matrix should be None due to failure
        assert profile.political_matrix is None

        # All other matrices should be populated
        assert profile.psychologist_matrix is not None
        assert profile.economist_matrix is not None
        assert profile.demographer_matrix is not None

        # Profile metadata should still be valid
        assert profile.metadata.agent_id is not None
        assert profile.metadata.target_accuracy_score == 0.85

    @pytest.mark.asyncio
    async def test_multiple_failures_degrade_gracefully(self) -> None:
        """Multiple expert failures → those matrices are None,
        remaining ones are populated."""
        # Fail both political scientist and economist
        client = _build_mock_client()

        # Override to fail two personas
        original_query = client.query_structured

        call_count_by_persona: dict[str, int] = {}

        async def selective_fail(
            *,
            system_prompt: str,
            user_prompt: str,
            target_model: type,
            persona_label: str = "expert",
        ) -> Any:
            if persona_label in ("political_scientist", "economist"):
                raise LLMServerError(
                    f"Terminal failure for persona={persona_label}"
                )
            mock_data = _MOCK_RESPONSES[persona_label]
            return target_model.model_validate(mock_data)

        client.query_structured = selective_fail  # type: ignore[assignment]

        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        # Failed matrices should be None
        assert profile.political_matrix is None
        assert profile.economist_matrix is None

        # Successful matrices should be populated
        assert profile.psychologist_matrix is not None
        assert profile.demographer_matrix is not None

    @pytest.mark.asyncio
    async def test_all_experts_fail_returns_empty_profile(self) -> None:
        """If all experts fail, profile should still return with all
        matrices as None rather than crashing."""
        client = _build_mock_client()

        async def fail_all(
            *,
            system_prompt: str,
            user_prompt: str,
            target_model: type,
            persona_label: str = "expert",
        ) -> Any:
            raise LLMClientError(f"Total failure for persona={persona_label}")

        client.query_structured = fail_all  # type: ignore[assignment]

        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        assert profile.psychologist_matrix is None
        assert profile.economist_matrix is None
        assert profile.political_matrix is None
        assert profile.demographer_matrix is None

        # Metadata and transcript should still be valid
        assert profile.metadata.agent_id is not None
        assert len(profile.raw_transcript) > 0

    @pytest.mark.asyncio
    async def test_degraded_profile_still_serializable(self) -> None:
        """A degraded profile with None matrices must still round-trip
        through JSON serialization cleanly."""
        server_error = LLMServerError(
            "All 3 retries exhausted for persona=political_scientist"
        )

        client = _build_mock_client(
            failing_persona="political_scientist",
            failure_error=server_error,
        )

        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        json_str = profile.model_dump_json()
        restored = AgentProfile.model_validate_json(json_str)

        assert restored == profile
        assert restored.political_matrix is None
        assert restored.psychologist_matrix is not None


# ═══════════════════════════════════════════════════════════════════════════
# Test 4 — Partial-State Metadata Preservation
# ═══════════════════════════════════════════════════════════════════════════


class TestMetadataPreservation:
    """Verify that metadata fields is_partial_profile and failed_experts
    are correctly populated under failure states."""

    @pytest.mark.asyncio
    async def test_metadata_records_single_failure(self) -> None:
        """Single expert failure should set is_partial_profile=True and list the expert."""
        server_error = LLMServerError("Terminal connection error")
        client = _build_mock_client(
            failing_persona="political_scientist",
            failure_error=server_error,
        )

        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        assert profile.metadata.is_partial_profile is True
        assert profile.metadata.failed_experts == ["political_scientist"]

    @pytest.mark.asyncio
    async def test_metadata_records_multiple_failures(self) -> None:
        """Multiple expert failures should all be listed in failed_experts."""
        client = _build_mock_client()

        # Fail political_scientist and economist
        async def mock_fail_multiple(
            *,
            system_prompt: str,
            user_prompt: str,
            target_model: type,
            persona_label: str = "expert",
        ) -> Any:
            if persona_label in ("political_scientist", "economist"):
                raise LLMServerError("API Error")
            mock_data = _MOCK_RESPONSES[persona_label]
            return target_model.model_validate(mock_data)

        client.query_structured = mock_fail_multiple  # type: ignore[assignment]

        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        assert profile.metadata.is_partial_profile is True
        # Order should match the order they were processed in EXPERT_PERSONAS
        assert set(profile.metadata.failed_experts) == {"political_scientist", "economist"}

    @pytest.mark.asyncio
    async def test_metadata_success_state(self) -> None:
        """When all experts succeed, is_partial_profile must be False."""
        client = _build_mock_client()
        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        assert profile.metadata.is_partial_profile is False
        assert len(profile.metadata.failed_experts) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Test 5 — Truncation Exception Handling
# ═══════════════════════════════════════════════════════════════════════════


class TestTruncationHandling:
    """Verify that LLMClientError is raised when response text is empty (no truncation exposed by genai)."""

    @pytest.mark.asyncio
    async def test_primary_call_truncation_raises_truncation_error(self) -> None:
        """If response text is empty, raise LLMClientError."""
        from src.utils.api_client import LLMClientError

        client = ResilientLLMClient(gemini_api_key="test-key", base_delay=0.01)

        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = ""

        # Mock response with empty text
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        client._groq_client = MagicMock()
        client._groq_client.chat.completions.create = AsyncMock(return_value=mock_response)

        try:
            with pytest.raises(LLMClientError) as exc_info:
                await client.query_structured(
                    system_prompt="Test sys",
                    user_prompt="Test user",
                    target_model=PsychologicalProfile,
                    persona_label="psychologist",
                )
            assert "Empty response content from LLM" in str(exc_info.value)
        finally:
            await client.close()


# ═══════════════════════════════════════════════════════════════════════════
# Test 6 — Repair Ceiling Enforcements
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairCeiling:
    """Verify self-healing attempts are constrained to max_repair_attempts."""

    @pytest.mark.asyncio
    async def test_repair_attempts_exhaustion_raises_validation(self) -> None:
        """If response continues to fail validation, error out at ceiling."""
        # Configure client with max_repair_attempts = 2
        client = ResilientLLMClient(
            gemini_api_key="test-key",
            base_delay=0.01,
            max_repair_attempts=2,
        )

        mock_choice = MagicMock()
        mock_choice.message = MagicMock()
        mock_choice.message.content = json.dumps({"emotional_stability_index": -1.0})

        # Mock a series of invalid JSON outputs
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        client._groq_client = MagicMock()
        client._groq_client.chat.completions.create = AsyncMock(return_value=mock_response)

        try:
            with pytest.raises(LLMClientError) as exc_info:
                await client.query_structured(
                    system_prompt="Test sys",
                    user_prompt="Test user",
                    target_model=PsychologicalProfile,
                    persona_label="psychologist",
                )
            # Should mention repair attempts exhausted
            assert "2 repair attempts exhausted" in str(exc_info.value)
            # The underlying completion creator should have been called 3 times total:
            # 1 primary attempt + 2 repair attempts
            assert client._groq_client.chat.completions.create.call_count == 3
        finally:
            await client.close()

    def test_invalid_repair_attempts_parameter(self) -> None:
        """Client should reject max_repair_attempts out of [1, 2] range."""
        with pytest.raises(ValueError):
            ResilientLLMClient(max_repair_attempts=0, gemini_api_key="test-key")

        with pytest.raises(ValueError):
            ResilientLLMClient(max_repair_attempts=3, gemini_api_key="test-key")


# ═══════════════════════════════════════════════════════════════════════════
# Test 7 — Concurrency Throttling Verification
# ═══════════════════════════════════════════════════════════════════════════


class TestConcurrencyThrottling:
    """Verify that requests are throttled and execute matching the Semaphore ceiling."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self) -> None:
        """Assert that no more than 2 expert queries run concurrently."""
        from src.pipelines.reflection import CONCURRENT_REQUEST_SEMAPHORE

        active_count = 0
        max_seen_concurrency = 0
        lock = asyncio.Lock()

        # Build client mock
        client = _build_mock_client()

        # Wrap query_structured to track active concurrent execution count
        original_query = client.query_structured

        async def tracked_query(*args: Any, **kwargs: Any) -> Any:
            nonlocal active_count, max_seen_concurrency
            async with lock:
                active_count += 1
                if active_count > max_seen_concurrency:
                    max_seen_concurrency = active_count

            # Simulate network latency to allow overlaps
            await asyncio.sleep(0.05)

            try:
                # Delegate to original mock resolver
                return await original_query(*args, **kwargs)
            finally:
                async with lock:
                    active_count -= 1

        client.query_structured = tracked_query  # type: ignore[assignment]

        # Reset the global semaphore to make sure it's clean
        # (It's 2, which matches the pipeline's setting)
        profile = await generate_expert_reflection_matrix(
            MOCK_TRANSCRIPT, llm_client=client
        )

        # Assert all succeeded
        assert profile.psychologist_matrix is not None
        # Max concurrency seen should have been exactly 2 (never 3 or 4)
        assert max_seen_concurrency <= 2
        assert max_seen_concurrency > 1

