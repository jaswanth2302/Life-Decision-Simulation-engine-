"""
Drishti — API Request/Response Schemas
=======================================

Pydantic v2 models for the FastAPI layer.
These are the data contracts between the backend and the Next.js frontend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Interview Init
# ---------------------------------------------------------------------------


class IdentityPayload(BaseModel):
    """Stage 1 identity fields collected by the sequential identity form."""
    name: str = Field(..., min_length=1)
    age: str = Field(..., min_length=1)
    country: str = Field(..., min_length=1)
    occupation: str = Field(..., min_length=1)
    timezone: str = Field(default="UTC")


class InterviewInitRequest(BaseModel):
    agent_id: UUID
    identity: IdentityPayload
    metadata: Optional[Dict[str, Any]] = None


class InterviewInitResponse(BaseModel):
    thread_id: str
    agent_id: UUID
    next_question: str
    remaining_turns: int
    evaluation_scores: Dict[str, float]   # Legacy — kept for UI compat during transition
    persona_labels: List[Dict] = Field(default_factory=list)
    insights: List[Dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Interview Submit
# ---------------------------------------------------------------------------


class InterviewSubmitRequest(BaseModel):
    thread_id: str
    agent_id: UUID
    user_response: str


class InterviewSubmitResponse(BaseModel):
    next_question: str
    remaining_turns: int
    evaluation_scores: Dict[str, float]   # Legacy
    is_complete: bool
    # Drishti fields — populated on every turn
    persona_labels: List[Dict] = Field(default_factory=list)
    new_insights: List[Dict] = Field(default_factory=list)
    # Populated only when is_complete = True
    summary_sentences: List[Dict] = Field(default_factory=list)
    
    # Telemetry
    debug_state: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Agent Status
# ---------------------------------------------------------------------------


class AgentStatusResponse(BaseModel):
    agent_id: UUID
    evaluation_scores: Dict[str, float]
    recent_memories: List[str]
    persona_labels: List[Dict] = Field(default_factory=list)
    insights: List[Dict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """User feedback on a specific summary sentence."""
    agent_id: UUID
    thread_id: str
    sentence_index: int = Field(..., ge=0, le=4, description="Which sentence (0-indexed).")
    sentiment: str = Field(..., description="'up', 'down', or 'edit'.")
    edit_text: Optional[str] = Field(
        default=None,
        description="User's correction when sentiment='edit'.",
    )
    surprise_index: Optional[int] = Field(
        default=None,
        description="Set when user answers 'Which sentence surprised you most?' (0-indexed).",
    )


class FeedbackResponse(BaseModel):
    acknowledged: bool
    message: str
    updated_insight_id: Optional[str] = None
