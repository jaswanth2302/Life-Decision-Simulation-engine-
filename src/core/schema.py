"""
Drishti — Core Domain Schema
=============================

Strongly-typed Pydantic v2 models for the Drishti life simulation engine.
The memory architecture is the product. Every model here is designed to
compound in value over years, not just sessions.

Models
------
- RichMemory          — A structured episodic memory with emotion, topics, graph edges.
- MemoryConnection    — An edge in the memory graph linking two memories.
- Insight             — A pattern extracted from memories, with a full lifecycle.
- PersonaLabel        — A human-readable identity reflection (Builder ★★★★★).
- SummarySentence     — One sentence of the Drishti summary with its evidence chain.
- DrishtiSummary      — The complete 5-sentence summary output.
- ReadinessAssessment — Whether the model can write a trustworthy summary now.
- IdentityProfile     — Basic identity from Stage 1 (name, age, country, etc.).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Identity (Stage 1)
# ---------------------------------------------------------------------------


class IdentityProfile(BaseModel):
    """Basic identity collected during the sequential identity form."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1, description="User's first name.")
    age: str = Field(..., min_length=1, description="User's age or age range.")
    country: str = Field(..., min_length=1, description="User's country of residence.")
    occupation: str = Field(..., min_length=1, description="User's current occupation or role.")
    timezone: str = Field(..., min_length=1, description="User's timezone (e.g. 'Asia/Kolkata').")


# ---------------------------------------------------------------------------
# Memory (The Product Foundation)
# ---------------------------------------------------------------------------

EmotionLabel = Literal[
    "joy", "hope", "pride", "excitement", "love", "gratitude",
    "curiosity", "nostalgia", "contentment",
    "sadness", "fear", "anxiety", "anger", "shame", "grief",
    "frustration", "loneliness", "regret",
    "neutral", "ambivalence", "uncertainty",
]

TimeReference = Literal["past", "present", "future", "timeless"]
MemorySource = Literal["conversation", "correction", "reflection", "import"]


class RichMemory(BaseModel):
    """
    A structured episodic memory with full semantic metadata.

    Designed to support queries like "Show me every hopeful memory about
    my startup" two years from now. Build for 2030.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    agent_id: UUID = Field(..., description="Owner of this memory.")
    text: str = Field(..., min_length=1, description="The memory content as stated or derived.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this memory was recorded.",
    )

    # Semantic enrichment — extracted by LLM during memory ingestion
    emotion: EmotionLabel = Field(
        default="neutral",
        description="The dominant emotion associated with this memory.",
    )
    certainty: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="How certain the model is about this memory's interpretation (0–1).",
    )
    importance: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Psychological/social importance (1=trivial, 10=life-altering).",
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Semantic topics this memory touches (e.g. ['career', 'startup', 'building']).",
    )
    people: List[str] = Field(
        default_factory=list,
        description="Named people mentioned in this memory (first names only for privacy).",
    )
    time_reference: TimeReference = Field(
        default="present",
        description="Whether the memory refers to past, present, future, or timeless context.",
    )
    source: MemorySource = Field(
        default="conversation",
        description="Origin of this memory.",
    )

    # Provenance — critical for debugging and iterating prompts
    generated_by_prompt_version: str = Field(
        default="interview_v1",
        description="Which prompt version produced this memory extraction.",
    )

    # Memory graph edges (populated during insight extraction)
    connected_memory_ids: List[str] = Field(
        default_factory=list,
        description=(
            "IDs of related memories. Enables graph traversal queries like "
            "'How did I become more confident?' (V2 feature, edges populated in V1)."
        ),
    )

    # Vector embedding reference (stored in Supabase separately)
    embedding_id: Optional[str] = Field(
        default=None,
        description="Reference to the vector embedding in Supabase pgvector.",
    )


class MemoryConnection(BaseModel):
    """
    An edge in the memory graph.

    Stored in the `memory_connections` Supabase table.
    Enables causal chain traversal in V2.
    """

    model_config = ConfigDict(frozen=True)

    source_id: UUID = Field(..., description="The origin memory.")
    target_id: UUID = Field(..., description="The connected memory.")
    relationship_type: Literal[
        "caused", "followed", "contradicts", "reinforces", "elaborates", "resolves"
    ] = Field(..., description="The semantic relationship between these memories.")
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How strong this connection is (0=weak, 1=definitive).",
    )


# ---------------------------------------------------------------------------
# Insight Lifecycle
# ---------------------------------------------------------------------------

InsightLifecycleStage = Literal[
    "candidate",   # 1 memory supports it — held provisionally, never shown to user
    "supported",   # 2–3 memories — shown with low confidence
    "strong",      # 4+ memories — shown prominently
    "shifting",    # recent memories contradict — shown with "evolving" flag
    "retired",     # contradictions outweigh support — archived, never deleted
]


class Insight(BaseModel):
    """
    A pattern extracted from memories. Not a fact — an observation.

    The lifecycle tracks how confidence in this insight evolves over time.
    A 'shifting' or 'retired' insight is not a failure — it means Drishti
    has noticed the person has changed. That is the whole point.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    text: str = Field(
        ...,
        min_length=1,
        description=(
            "The insight as a specific, grounded observation. "
            "E.g. 'Shows long-term consistency once committed.' "
            "Never generic. Never flattering."
        ),
    )
    lifecycle_stage: InsightLifecycleStage = Field(
        default="candidate",
        description="Current lifecycle stage of this insight.",
    )
    source_memory_ids: List[str] = Field(
        default_factory=list,
        description="IDs of memories that support this insight. Required for explainability.",
    )
    contradiction_memory_ids: List[str] = Field(
        default_factory=list,
        description="IDs of memories that contradict or complicate this insight.",
    )
    confidence: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence in this insight (derived from evidence_count + contradiction_count).",
    )
    evidence_count: int = Field(
        default=1,
        ge=0,
        description="Number of memories that support this insight.",
    )
    contradiction_count: int = Field(
        default=0,
        ge=0,
        description="Number of memories that contradict this insight.",
    )
    is_evolving: bool = Field(
        default=False,
        description="True when the insight is in 'shifting' stage or was recently contradicted by user feedback.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    last_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    generated_by_prompt_version: str = Field(
        default="insight_extraction_v1",
        description="Prompt version that created this insight. Critical for debugging.",
    )


# ---------------------------------------------------------------------------
# Persona Labels ("What I'm Learning About You")
# ---------------------------------------------------------------------------


class PersonaLabel(BaseModel):
    """
    A human-readable identity reflection. Not a psychology label.

    'Builder ★★★★★' not 'Curiosity: 0.82'.
    Humans don't think 'My curiosity is 0.73.' They think 'I'm someone who loves building.'
    """

    model_config = ConfigDict(frozen=True)

    label: str = Field(
        ...,
        min_length=1,
        description=(
            "A short, human-readable identity label. "
            "E.g. 'Builder', 'Deep Thinker', 'Learns by Doing', 'Needs Challenge'. "
            "Not generic psychology terms."
        ),
    )
    stars: int = Field(
        ...,
        ge=1,
        le=5,
        description="Strength of this trait (1=faint signal, 5=defining characteristic).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Model confidence in this label. "
            ">70% → stated as 'I think...'. "
            "40–70% → 'One pattern I notice...'. "
            "<40% → never shown to user."
        ),
    )
    evidence_count: int = Field(
        ...,
        ge=0,
        description="Number of memories/insights that support this label.",
    )
    source_insight_ids: List[str] = Field(
        default_factory=list,
        description="Insight IDs that generated this label. Required for explainability.",
    )
    generated_by_prompt_version: str = Field(
        default="persona_label_v1",
    )


# ---------------------------------------------------------------------------
# Summary Sentence (The Magical Moment)
# ---------------------------------------------------------------------------


class SummarySentence(BaseModel):
    """
    One sentence from the Drishti 5-sentence summary.

    Every sentence is traceable: sentence → insights → memories.
    This is the explainability chain that makes Drishti trustworthy.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        ...,
        min_length=1,
        description=(
            "The observation as it will be shown to the user. "
            "Specific, grounded, never generic, never flattering, never certain. "
            "Uses: 'I think...', 'It seems...', 'One pattern I notice...'"
        ),
    )
    supporting_insight_ids: List[str] = Field(
        default_factory=list,
        description="Insight IDs that generated this sentence.",
    )
    source_memory_ids: List[str] = Field(
        default_factory=list,
        description=(
            "Memory IDs shown when user clicks '▼ Why?'. "
            "At least 2 required for a sentence to be included in the summary."
        ),
    )
    is_reframe: bool = Field(
        default=False,
        description=(
            "True if this sentence contains a reframe — something the user has "
            "never thought about themselves that way. At least one sentence per "
            "summary must be a reframe."
        ),
    )
    generated_by_prompt_version: str = Field(
        default="summary_v1",
    )


class DrishtiSummary(BaseModel):
    """
    The complete Drishti summary — exactly 5 sentences, each with evidence.

    This is the magical moment. The 5-sentence reveal that makes users pause.
    Paced one sentence at a time in the UI, with 2-second gaps.
    """

    model_config = ConfigDict(frozen=True)

    sentences: List[SummarySentence] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly 5 sentences. No more, no less.",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    surprise_sentence_idx: Optional[int] = Field(
        default=None,
        description=(
            "Set after user answers 'Which sentence surprised you the most?' "
            "This becomes a training signal for prompt improvement."
        ),
    )
    generated_by_prompt_version: str = Field(
        default="summary_v1",
    )


# ---------------------------------------------------------------------------
# Readiness Assessment
# ---------------------------------------------------------------------------


class ReadinessAssessment(BaseModel):
    """
    LLM judgment: can Drishti write a trustworthy summary right now?

    Not a counter. A judgment. The model assesses whether it has enough
    specific, grounded evidence to make 5 claims about this person.
    """

    model_config = ConfigDict(frozen=True)

    can_summarize: bool = Field(
        ...,
        description=(
            "True only if Drishti can write a 5-sentence summary where every "
            "sentence is supported by at least 2 specific memories."
        ),
    )
    missing_info: str = Field(
        default="",
        description=(
            "If can_summarize is False: what kind of information would make "
            "the summary more honest? Used to guide the next interview question."
        ),
    )


# ---------------------------------------------------------------------------
# Structured LLM Outputs (for pipeline nodes)
# ---------------------------------------------------------------------------


class MemoryEnrichment(BaseModel):
    """
    Structured output from the memory enrichment LLM call.
    Extracts semantic metadata from a raw memory text.
    """

    emotion: EmotionLabel = Field(default="neutral")
    certainty: float = Field(default=0.8, ge=0.0, le=1.0)
    topics: List[str] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    time_reference: TimeReference = Field(default="present")
    importance: int = Field(default=5, ge=1, le=10)


class InsightBatch(BaseModel):
    """
    Structured output from the insight extraction LLM call.
    Contains 1–3 new insights derived from a batch of memories.
    """

    insights: List[dict] = Field(
        default_factory=list,
        description=(
            "List of insight dicts, each with 'text' (str) and "
            "'source_memory_ids' (list[str])."
        ),
    )


class PersonaLabelBatch(BaseModel):
    """
    Structured output from the persona label LLM call.
    Contains the current set of human-readable identity reflections.
    """

    labels: List[dict] = Field(
        default_factory=list,
        description=(
            "List of label dicts, each with 'label' (str), 'stars' (int 1-5), "
            "'confidence' (float), 'source_insight_ids' (list[str])."
        ),
    )


class UserIntent(str, Enum):
    CONTINUE = "CONTINUE"
    CHANGE_TOPIC = "CHANGE_TOPIC"
    END_SESSION = "END_SESSION"
    CORRECT_MODEL = "CORRECT_MODEL"
    CLARIFICATION = "CLARIFICATION"
    META_QUESTION = "META_QUESTION"
    ASK_DRISHTI = "ASK_DRISHTI"
    UNKNOWN = "UNKNOWN"


class IntentClassifierOutput(BaseModel):
    """Structured output for the intent classifier."""
    intent: UserIntent = Field(description="The primary intent of the user's message.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this classification (0.0 to 1.0)."
    )
    is_structural: bool = Field(
        description="True if this is a purely structural command (e.g. 'next', 'wrap up', 'stop') and contains no new personal information."
    )


class ReflectionTheme(str, Enum):
    EMOTIONAL_OVERLOAD = "EMOTIONAL_OVERLOAD"
    SEARCHING_FOR_MEANING = "SEARCHING_FOR_MEANING"
    SELF_DOUBT = "SELF_DOUBT"
    MOTIVATION = "MOTIVATION"
    PURPOSE = "PURPOSE"
    UNCERTAINTY = "UNCERTAINTY"
    AUTONOMY = "AUTONOMY"
    CONNECTION = "CONNECTION"
    ACHIEVEMENT = "ACHIEVEMENT"
    REST = "REST"
    CHANGE = "CHANGE"
    IDENTITY = "IDENTITY"
    VALUES = "VALUES"
    FEAR = "FEAR"
    GROWTH = "GROWTH"


class GeneratedResponse(BaseModel):
    """Structured output: the next conversational response."""

    response: str = Field(
        ...,
        description="The actual text to speak to the user.",
    )
    theme: ReflectionTheme = Field(
        ...,
        description="The primary conceptual theme of this response.",
    )
    interaction_type: str = Field(
        ...,
        description="The type of interaction (e.g., ASK, REFLECT, PAUSE, RECOVER, SPACE).",
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="How strongly this response fits the evidence (0.0 to 1.0).",
    )
