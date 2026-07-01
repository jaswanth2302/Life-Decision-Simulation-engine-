"""
Drishti — Agent State
======================

LangGraph AgentState TypedDict. Every field here flows through the
interview pipeline and persists via MemorySaver checkpointing.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Central state object for the Drishti interview graph.

    Fields are designed to compound in value over many sessions.
    The memory architecture is the product.
    """

    # -----------------------------------------------------------------------
    # Identity (Stage 1 — collected before interview begins)
    # -----------------------------------------------------------------------
    agent_id: UUID
    identity: Dict[str, str]          # {name, age, country, occupation, timezone}

    # -----------------------------------------------------------------------
    # Conversation
    # -----------------------------------------------------------------------
    messages: Annotated[List[Any], add_messages]
    current_topic: str                # Loose thematic focus for the current question
    consecutive_question_count: int   # Questions asked in a row (input to adaptive reflection formula)
    conversation_energy: int          # 0–10: drops when tired/short/frustrated, rises with long/engaged answers
    conversation_curiosity: int       # 0–10: rises when user says "Wait...", "That's true", volunteers detail
    conversation_certainty: int       # 0–10: how certain the user feels about themselves and what they want
    reframe_used: bool                # Has a Reframe reflection been used this session? (keep them rare)
    topic_turn_count: int             # Turns spent on the same topic — triggers entropy pivot at 4+
    last_mode: str                    # The mode used in the previous turn (ASK, REFLECT, PAUSE, RECOVER, SPACE)
    support_budget: int               # Count of consecutive support interventions (caps at 1)
    recent_themes: List[str]          # Rolling list (last 3) of ReflectionThemes explored
    user_intent: Optional[str]        # Latest classified user intent (e.g., CONTINUE, END_SESSION)
    intent_confidence: Optional[float]# Confidence of the latest intent classification
    intent_is_structural: Optional[bool] # True if the user command was purely structural
    intent_reason: Optional[str]      # Reason for structural pivot or meta intent
    latencies: Dict[str, float]       # Node execution timings in ms

    # -----------------------------------------------------------------------
    # Memory (The Foundation)
    # -----------------------------------------------------------------------
    stored_memory_count: int          # Total memories persisted to Supabase (importance >= 5)
    session_memory_count: int         # Memories processed this session (triggers insight extraction every 3rd)

    # -----------------------------------------------------------------------
    # Insight Engine
    # -----------------------------------------------------------------------
    insights: List[Dict]              # List of Insight dicts with full lifecycle metadata
    new_insight_count: int            # Insights added since last persona label update

    # -----------------------------------------------------------------------
    # Persona ("What I'm Learning About You")
    # -----------------------------------------------------------------------
    persona_labels: List[Dict]        # List of PersonaLabel dicts (only confidence >= 0.4 shown to user)

    # -----------------------------------------------------------------------
    # Summary (The Magical Moment)
    # -----------------------------------------------------------------------
    can_summarize: bool               # Judgment on whether we can generate the final summary
    readiness_missing_info: str       # What's still needed (from ReadinessAssessment)
    summary_sentences: List[Dict]     # List of SummarySentence dicts (exactly 5 when complete)
    closing_observation: str          # Single sentence: "Today I learned that..."
    is_complete: bool

    # -----------------------------------------------------------------------
    # Legacy — kept for API compatibility during transition
    # -----------------------------------------------------------------------
    remaining_turns: int              # Soft safety cap (100) — not the primary termination condition
    evaluation_scores: Dict[str, float]  # Legacy 4-axis scores — still computed, used for coverage heuristic
