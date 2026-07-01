"""
Generative Agent Simulation Engine — Schema Test Suite
======================================================

Comprehensive pytest module verifying Phase 1 data contracts:

1. **Happy-path instantiation** — Full ``AgentProfile`` with rich mock data
   representing a ~2,000-word interview profile with sample expert analyses.
2. **Boundary validation** — Confirms Pydantic ``ValidationError`` is raised
   for out-of-range values (importance > 10, scores > 1.0, invalid domains).
3. **Serialization round-trip** — JSON dump → reload → strict deep equality.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from src.core.schema import (
    AgentMetadata,
    AgentProfile,
    DemographicBaseline,
    EconomicDecisionMatrix,
    MemoryStreamItem,
    PsychologicalProfile,
    SocioPoliticalWorldview,
    TranscriptTurn,
)


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures — Rich Mock Data
# ═══════════════════════════════════════════════════════════════════════════


def _build_mock_transcript() -> list[TranscriptTurn]:
    """Generate a realistic multi-domain transcript (~2,000 words worth)."""
    turns: list[TranscriptTurn] = [
        # --- Professional domain ---
        TranscriptTurn(
            speaker="interviewer",
            text=(
                "Let's start by exploring your professional background. "
                "Can you walk me through your career trajectory, beginning "
                "from your first meaningful role, and describe what motivated "
                "each transition you made along the way?"
            ),
            domain="professional",
            semantic_density_score=0.45,
        ),
        TranscriptTurn(
            speaker="subject",
            text=(
                "Absolutely. I started as a junior data analyst right out of "
                "university in 2014. The role was comfortable but I quickly "
                "realized I was drawn to the strategic side of things rather "
                "than just running reports. After about eighteen months I moved "
                "into a product management role at a mid-size fintech startup. "
                "That was transformative — I learned to balance engineering "
                "constraints with user needs and business viability. The pace "
                "was relentless but I thrived on the autonomy. Three years "
                "later I transitioned to a larger organization as a senior PM, "
                "primarily because I wanted exposure to enterprise-scale "
                "decision-making and cross-functional leadership."
            ),
            domain="professional",
            semantic_density_score=0.82,
        ),
        # --- Decision domain ---
        TranscriptTurn(
            speaker="interviewer",
            text=(
                "Thinking about a major life decision you faced in the last "
                "five years, how did you approach the analysis? Walk me "
                "through your internal framework for weighing options."
            ),
            domain="decision",
            semantic_density_score=0.55,
        ),
        TranscriptTurn(
            speaker="subject",
            text=(
                "The biggest decision was relocating internationally for a "
                "leadership position. My framework was almost embarrassingly "
                "structured — I built a weighted decision matrix with about "
                "fifteen factors: career growth potential, cost of living "
                "differential, proximity to family, healthcare quality, "
                "cultural compatibility, and so on. Each factor got a weight "
                "from one to ten based on long-term importance. I also ran a "
                "pre-mortem exercise imagining I had already accepted and it "
                "had gone badly — what would the failure modes look like? "
                "That reversed analysis was actually more useful than the "
                "forward-looking scoring. In the end, the quantitative "
                "analysis narrowly favored staying put, but my gut strongly "
                "disagreed, and I went with the move."
            ),
            domain="decision",
            semantic_density_score=0.91,
        ),
        # --- Worldview domain ---
        TranscriptTurn(
            speaker="interviewer",
            text=(
                "How would you describe your relationship with societal "
                "institutions — government, media, education systems? "
                "Has your trust evolved over the years?"
            ),
            domain="worldview",
            semantic_density_score=0.50,
        ),
        TranscriptTurn(
            speaker="subject",
            text=(
                "My trust has definitely eroded but in a nuanced way. I still "
                "believe in the institutional frameworks themselves — "
                "democratic governance, independent judiciary, free press — "
                "but my confidence in the people operating within those "
                "frameworks has diminished. I grew up in a household that "
                "had tremendous faith in public institutions, and early in "
                "life I inherited that faith uncritically. Through my "
                "twenties, witnessing policy failures, media sensationalism, "
                "and educational gatekeeping made me more skeptical. I would "
                "not call myself cynical though — I am more of a critical "
                "institutionalist. I think reform from within is possible "
                "but requires sustained civic engagement."
            ),
            domain="worldview",
            semantic_density_score=0.88,
        ),
        # --- Environmental domain ---
        TranscriptTurn(
            speaker="interviewer",
            text=(
                "Tell me about the physical and social environment you grew "
                "up in. How did your surroundings shape your outlook?"
            ),
            domain="environmental",
            semantic_density_score=0.40,
        ),
        TranscriptTurn(
            speaker="subject",
            text=(
                "I grew up in a mid-sized suburban area with a strong sense "
                "of community but limited economic diversity. Most families "
                "in our neighborhood were in similar economic brackets — "
                "solidly middle class with public sector or small business "
                "employment. The environment was safe and stable but also "
                "somewhat insular. Exposure to different perspectives mainly "
                "came through books and later the internet. I think that "
                "combination of security and limited exposure created an "
                "interesting duality — I felt grounded but also hungry "
                "for broader horizons. My parents emphasized education as "
                "the primary vehicle for upward mobility, which became a "
                "deeply internalized value."
            ),
            domain="environmental",
            semantic_density_score=0.79,
        ),
    ]
    return turns


def _build_mock_profile() -> AgentProfile:
    """Construct a fully-populated AgentProfile with all expert matrices."""
    fixed_agent_id = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
    fixed_time = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    transcript = _build_mock_transcript()

    word_count = sum(len(turn.text.split()) for turn in transcript)

    return AgentProfile(
        metadata=AgentMetadata(
            agent_id=fixed_agent_id,
            created_at=fixed_time,
            total_interview_word_count=word_count,
            target_accuracy_score=0.85,
        ),
        raw_transcript=transcript,
        psychologist_matrix=PsychologicalProfile(
            core_values=[
                "autonomy",
                "intellectual growth",
                "family stability",
                "civic responsibility",
                "meritocratic fairness",
            ],
            coping_mechanisms=[
                "structured analysis (decision matrices)",
                "pre-mortem catastrophizing",
                "physical exercise under stress",
                "social withdrawal for processing",
            ],
            cognitive_anchors=[
                "education as upward mobility vehicle",
                "institutional reform over revolution",
                "data-driven reasoning with gut-check override",
            ],
            emotional_stability_index=0.72,
        ),
        economist_matrix=EconomicDecisionMatrix(
            risk_appetite_score=0.65,
            loss_aversion_ratio=1.85,
            heuristic_biases=[
                "anchoring to quantitative analysis",
                "status quo bias (mild)",
                "sunk cost sensitivity in career decisions",
                "optimism bias in relocation planning",
            ],
            resource_allocation_rules=[
                "prioritize long-term career capital over short-term compensation",
                "maintain 6-month emergency fund before discretionary spending",
                "allocate minimum 10% of income to skill development",
                "time-box exploratory activities to prevent scope creep",
            ],
        ),
        political_matrix=SocioPoliticalWorldview(
            institutional_trust_score=0.45,
            ideological_anchors=[
                "critical institutionalism",
                "reform-oriented pragmatism",
                "evidence-based policy preference",
                "civic engagement as duty",
            ],
            authority_bias_flag=False,
            conflict_resolution_style="collaborative",
        ),
        demographer_matrix=DemographicBaseline(
            structural_constraints=[
                "suburban upbringing with limited economic diversity exposure",
                "first-generation professional in technology sector",
                "geographic mobility constrained by family proximity preference",
            ],
            geographic_context="suburban mid-sized metro area, temperate climate",
            socio_economic_tier="middle",
            background_intersection_notes=(
                "Subject exhibits strong intergenerational mobility aspirations "
                "shaped by stable but economically homogeneous upbringing. "
                "Educational attainment viewed as primary differentiation tool."
            ),
        ),
        episodic_memory_stream=[
            MemoryStreamItem(
                memory_id=UUID("11111111-1111-1111-1111-111111111111"),
                content=(
                    "First day at fintech startup — felt both terrified and "
                    "exhilarated by the unstructured environment."
                ),
                timestamp=fixed_time,
                recency_decay_factor=0.35,
                importance_score=8,
                embedding_vector_id="vec_emb_001",
            ),
            MemoryStreamItem(
                memory_id=UUID("22222222-2222-2222-2222-222222222222"),
                content=(
                    "International relocation acceptance call — decided to "
                    "override quantitative analysis with intuition."
                ),
                timestamp=fixed_time,
                recency_decay_factor=0.70,
                importance_score=10,
                embedding_vector_id="vec_emb_002",
            ),
            MemoryStreamItem(
                memory_id=UUID("33333333-3333-3333-3333-333333333333"),
                content=(
                    "Realized during policy debate that personal trust in "
                    "institutions had shifted from inherited faith to earned skepticism."
                ),
                timestamp=fixed_time,
                recency_decay_factor=0.50,
                importance_score=6,
                embedding_vector_id=None,
            ),
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test 1 — Happy-Path Full Instantiation
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentProfileInstantiation:
    """Verify that a fully-populated AgentProfile instantiates correctly."""

    def test_full_profile_creates_successfully(self) -> None:
        """A rich mock profile should instantiate without errors."""
        profile = _build_mock_profile()

        assert profile.metadata.agent_id == UUID(
            "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        )
        assert profile.metadata.target_accuracy_score == 0.85
        assert len(profile.raw_transcript) == 8
        assert profile.psychologist_matrix is not None
        assert profile.economist_matrix is not None
        assert profile.political_matrix is not None
        assert profile.demographer_matrix is not None
        assert len(profile.episodic_memory_stream) == 3

    def test_transcript_domains_are_valid(self) -> None:
        """Every transcript turn must have a valid domain value."""
        profile = _build_mock_profile()
        valid_domains = {"professional", "decision", "worldview", "environmental"}

        for turn in profile.raw_transcript:
            assert turn.domain in valid_domains

    def test_word_count_is_positive(self) -> None:
        """The total interview word count should reflect actual transcript."""
        profile = _build_mock_profile()
        assert profile.metadata.total_interview_word_count > 0

    def test_memory_stream_importance_within_bounds(self) -> None:
        """All memory importance scores must be in [1, 10]."""
        profile = _build_mock_profile()
        for memory in profile.episodic_memory_stream:
            assert 1 <= memory.importance_score <= 10

    def test_expert_matrices_field_values(self) -> None:
        """Spot-check specific expert matrix field values."""
        profile = _build_mock_profile()

        assert profile.psychologist_matrix.emotional_stability_index == 0.72
        assert profile.economist_matrix.risk_appetite_score == 0.65
        assert profile.economist_matrix.loss_aversion_ratio == 1.85
        assert profile.political_matrix.institutional_trust_score == 0.45
        assert profile.political_matrix.authority_bias_flag is False
        assert profile.demographer_matrix.socio_economic_tier == "middle"


# ═══════════════════════════════════════════════════════════════════════════
# Test 2 — Boundary Validation (Pydantic ValidationError)
# ═══════════════════════════════════════════════════════════════════════════


class TestValidationBoundaries:
    """Ensure Pydantic enforces field constraints with ValidationError."""

    def test_importance_score_above_10_rejected(self) -> None:
        """MemoryStreamItem with importance_score > 10 must fail."""
        with pytest.raises(ValidationError):
            MemoryStreamItem(
                content="Test memory",
                importance_score=11,
            )

    def test_importance_score_below_1_rejected(self) -> None:
        """MemoryStreamItem with importance_score < 1 must fail."""
        with pytest.raises(ValidationError):
            MemoryStreamItem(
                content="Test memory",
                importance_score=0,
            )

    def test_semantic_density_above_1_rejected(self) -> None:
        """TranscriptTurn with semantic_density_score > 1.0 must fail."""
        with pytest.raises(ValidationError):
            TranscriptTurn(
                speaker="subject",
                text="Some text",
                domain="professional",
                semantic_density_score=1.5,
            )

    def test_semantic_density_below_0_rejected(self) -> None:
        """TranscriptTurn with semantic_density_score < 0.0 must fail."""
        with pytest.raises(ValidationError):
            TranscriptTurn(
                speaker="subject",
                text="Some text",
                domain="professional",
                semantic_density_score=-0.1,
            )

    def test_invalid_domain_rejected(self) -> None:
        """TranscriptTurn with a domain not in the Literal set must fail."""
        with pytest.raises(ValidationError):
            TranscriptTurn(
                speaker="subject",
                text="Some text",
                domain="invalid_domain",
                semantic_density_score=0.5,
            )

    def test_risk_appetite_above_1_rejected(self) -> None:
        """EconomicDecisionMatrix with risk_appetite_score > 1.0 must fail."""
        with pytest.raises(ValidationError):
            EconomicDecisionMatrix(
                risk_appetite_score=1.01,
            )

    def test_loss_aversion_ratio_zero_rejected(self) -> None:
        """EconomicDecisionMatrix with loss_aversion_ratio == 0 must fail."""
        with pytest.raises(ValidationError):
            EconomicDecisionMatrix(
                risk_appetite_score=0.5,
                loss_aversion_ratio=0.0,
            )

    def test_target_accuracy_above_1_rejected(self) -> None:
        """AgentMetadata with target_accuracy_score > 1.0 must fail."""
        with pytest.raises(ValidationError):
            AgentMetadata(
                target_accuracy_score=1.1,
            )

    def test_institutional_trust_above_1_rejected(self) -> None:
        """SocioPoliticalWorldview with trust > 1.0 must fail."""
        with pytest.raises(ValidationError):
            SocioPoliticalWorldview(
                institutional_trust_score=2.0,
            )

    def test_empty_speaker_rejected(self) -> None:
        """TranscriptTurn with an empty speaker string must fail."""
        with pytest.raises(ValidationError):
            TranscriptTurn(
                speaker="",
                text="Some text",
                domain="professional",
                semantic_density_score=0.5,
            )

    def test_frozen_model_rejects_mutation(self) -> None:
        """Frozen models must reject attribute assignment."""
        turn = TranscriptTurn(
            speaker="subject",
            text="Original text",
            domain="professional",
            semantic_density_score=0.5,
        )
        with pytest.raises(ValidationError):
            turn.text = "Modified text"


# ═══════════════════════════════════════════════════════════════════════════
# Test 3 — Serialization Round-Trip
# ═══════════════════════════════════════════════════════════════════════════


class TestSerializationRoundTrip:
    """Verify JSON dump → reload produces strictly equal model instances."""

    def test_full_profile_json_round_trip(self) -> None:
        """Serialize to JSON, deserialize back, and assert deep equality."""
        original = _build_mock_profile()

        # Dump to JSON string
        json_str: str = original.model_dump_json()

        # Reload from JSON string
        restored = AgentProfile.model_validate_json(json_str)

        # Assert strict equality across the entire deep structure
        assert restored == original

    def test_round_trip_preserves_metadata(self) -> None:
        """Agent metadata must survive serialization perfectly."""
        original = _build_mock_profile()
        json_str = original.model_dump_json()
        restored = AgentProfile.model_validate_json(json_str)

        assert restored.metadata.agent_id == original.metadata.agent_id
        assert restored.metadata.created_at == original.metadata.created_at
        assert (
            restored.metadata.total_interview_word_count
            == original.metadata.total_interview_word_count
        )
        assert (
            restored.metadata.target_accuracy_score
            == original.metadata.target_accuracy_score
        )

    def test_round_trip_preserves_memory_stream(self) -> None:
        """Episodic memory stream must survive serialization identically."""
        original = _build_mock_profile()
        json_str = original.model_dump_json()
        restored = AgentProfile.model_validate_json(json_str)

        assert len(restored.episodic_memory_stream) == len(
            original.episodic_memory_stream
        )
        for orig_mem, rest_mem in zip(
            original.episodic_memory_stream,
            restored.episodic_memory_stream,
            strict=True,
        ):
            assert rest_mem.memory_id == orig_mem.memory_id
            assert rest_mem.content == orig_mem.content
            assert rest_mem.importance_score == orig_mem.importance_score
            assert rest_mem.embedding_vector_id == orig_mem.embedding_vector_id

    def test_round_trip_with_none_matrices(self) -> None:
        """Profile with None expert matrices should round-trip cleanly."""
        profile = AgentProfile(
            metadata=AgentMetadata(),
            raw_transcript=[],
            psychologist_matrix=None,
            economist_matrix=None,
            political_matrix=None,
            demographer_matrix=None,
            episodic_memory_stream=[],
        )

        json_str = profile.model_dump_json()
        restored = AgentProfile.model_validate_json(json_str)

        assert restored == profile
        assert restored.psychologist_matrix is None
        assert restored.economist_matrix is None
        assert restored.political_matrix is None
        assert restored.demographer_matrix is None

    def test_model_dump_dict_round_trip(self) -> None:
        """Verify dict-based serialization also preserves equality."""
        original = _build_mock_profile()

        data_dict = original.model_dump()
        restored = AgentProfile.model_validate(data_dict)

        assert restored == original
