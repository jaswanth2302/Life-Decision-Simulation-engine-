"""
Drishti — Session Analytics & Conversation Meta-Memory
========================================================

Two responsibilities:

1. SESSION ANALYTICS
   After every conversation ends, log structured metrics to Supabase.
   These power the dashboard that will teach us more than any prompt change:

   - Reflection Acceptance Rate (did user say "yeah", "exactly", "that's true"?)
   - Energy trajectory (start → end)
   - Curiosity trajectory (start → end)
   - Certainty trajectory (start → end)
   - Interview length (turns)
   - Reframe used? (yes/no)
   - Closing observation text
   - Surprise rating (from user feedback)

2. CONVERSATION STYLE META-MEMORY
   Per agent, track HOW this person responds to different conversational moves.
   Not what they said — but how the conversation worked.

   {
     "opens_up_when": ["reflections", "examples", "short questions"],
     "withdraws_when": ["rapid questions", "long responses", "topic changes"],
     "avg_words_per_turn": 32,
     "best_session_length": 14,
     "preferred_depth": "personal"
   }

   Drishti reads this at session start. Over months, it learns how to talk
   to each person individually — not just what they said, but how they think.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from config.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Session Analytics Record
# ---------------------------------------------------------------------------

class SessionAnalytics:
    """Captures the full conversational state at session end for analysis."""

    def __init__(
        self,
        agent_id: UUID,
        session_id: str,
        state: Dict[str, Any],
        messages: List[Any],
    ) -> None:
        identity = state.get("identity") or {}

        # Core metrics
        self.agent_id = str(agent_id)
        self.session_id = session_id
        self.name = identity.get("name", "unknown")
        self.timestamp = datetime.now(timezone.utc).isoformat()

        # Conversation length
        user_turns = [m for m in messages if _get_role(m) == "user"]
        self.total_turns = len(user_turns)

        # Final conversation state
        self.final_energy = state.get("conversation_energy", 0)
        self.final_curiosity = state.get("conversation_curiosity", 0)
        self.final_certainty = state.get("conversation_certainty", 0)

        # Insight quality
        insights = state.get("insights") or []
        strong = [i for i in insights if i.get("lifecycle_stage") in ("strong", "supported")]
        self.insight_count = len(insights)
        self.strong_insight_count = len(strong)

        # Key moments
        self.reframe_used = state.get("reframe_used", False)
        self.closing_observation = state.get("closing_observation", "")

        # Persona labels
        labels = state.get("persona_labels") or []
        self.persona_label_count = len(labels)
        self.top_labels = [l["label"] for l in labels[:3]]

        # Summary quality
        sentences = state.get("summary_sentences") or []
        self.summary_generated = len(sentences) == 5
        self.summary_reframes = sum(1 for s in sentences if s.get("is_reframe"))

        # Conversation style signals (computed from message history)
        self.avg_user_words_per_turn = _compute_avg_words(user_turns)
        self.opens_up_signals = _count_signals(user_turns, ENGAGEMENT_SIGNALS)
        self.withdrawal_signals = _count_signals(user_turns, WITHDRAWAL_SIGNALS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "name": self.name,
            "timestamp": self.timestamp,
            "total_turns": self.total_turns,
            "final_energy": self.final_energy,
            "final_curiosity": self.final_curiosity,
            "final_certainty": self.final_certainty,
            "insight_count": self.insight_count,
            "strong_insight_count": self.strong_insight_count,
            "reframe_used": self.reframe_used,
            "closing_observation": self.closing_observation,
            "persona_label_count": self.persona_label_count,
            "top_labels": json.dumps(self.top_labels),
            "summary_generated": self.summary_generated,
            "summary_reframe_count": self.summary_reframes,
            "avg_user_words_per_turn": self.avg_user_words_per_turn,
            "opens_up_signals": self.opens_up_signals,
            "withdrawal_signals": self.withdrawal_signals,
        }

    def log_summary(self) -> None:
        """Write a human-readable summary to the application log."""
        logger.info(
            "\n"
            "═══════════════════════════════════════════\n"
            "  SESSION COMPLETE — %s\n"
            "═══════════════════════════════════════════\n"
            "  Turns:         %d\n"
            "  Energy:        %d/10\n"
            "  Curiosity:     %d/10\n"
            "  Certainty:     %d/10\n"
            "  Insights:      %d (%d strong)\n"
            "  Reframe used:  %s\n"
            "  Labels:        %s\n"
            "  Summary:       %s\n"
            "  Closing:       %s\n"
            "  Engagement:    +%d / -%d\n"
            "═══════════════════════════════════════════",
            self.name,
            self.total_turns,
            self.final_energy,
            self.final_curiosity,
            self.final_certainty,
            self.insight_count, self.strong_insight_count,
            "YES" if self.reframe_used else "no",
            ", ".join(self.top_labels) or "none yet",
            "complete (5 sentences)" if self.summary_generated else "incomplete",
            self.closing_observation[:80] if self.closing_observation else "none",
            self.opens_up_signals, self.withdrawal_signals,
        )


# ---------------------------------------------------------------------------
# Conversation Style Meta-Memory
# ---------------------------------------------------------------------------

class ConversationStyleMemory:
    """
    Per-agent record of HOW conversations work — not what was said.

    Drishti reads this at session start. Over multiple sessions, it learns
    that this particular person opens up more with reflections than questions,
    or that they disengage when topics change too rapidly.

    That's the difference between Drishti remembering your life
    and Drishti remembering how to talk to you.
    """

    def __init__(self, agent_id: UUID) -> None:
        self.agent_id = str(agent_id)
        self.opens_up_when: List[str] = []
        self.withdraws_when: List[str] = []
        self.avg_words_per_turn: float = 0.0
        self.total_sessions: int = 0
        self.avg_session_length: float = 0.0
        self.reframe_effectiveness: Optional[bool] = None  # Did the reframe work?
        self.best_session_energy: float = 0.0
        self.updated_at: str = ""

    @classmethod
    def from_session(
        cls,
        agent_id: UUID,
        analytics: SessionAnalytics,
        previous: Optional["ConversationStyleMemory"] = None,
    ) -> "ConversationStyleMemory":
        """Derive updated conversation style from a completed session."""
        style = cls(agent_id)

        # Merge with previous data if available
        if previous:
            style.total_sessions = previous.total_sessions + 1
            style.avg_words_per_turn = _running_avg(
                previous.avg_words_per_turn,
                analytics.avg_user_words_per_turn,
                style.total_sessions,
            )
            style.avg_session_length = _running_avg(
                previous.avg_session_length,
                analytics.total_turns,
                style.total_sessions,
            )
        else:
            style.total_sessions = 1
            style.avg_words_per_turn = analytics.avg_user_words_per_turn
            style.avg_session_length = analytics.total_turns

        # Infer conversation style signals from this session
        opens_up = []
        withdraws = []

        if analytics.avg_user_words_per_turn >= 30:
            opens_up.append("detailed questions")
        if analytics.final_curiosity >= 7:
            opens_up.append("reflective conversations")
        if analytics.opens_up_signals >= 3:
            opens_up.append("stories and examples")
        if analytics.final_energy >= 7:
            opens_up.append("paced conversations")

        if analytics.withdrawal_signals >= 3:
            withdraws.append("rapid follow-up questions")
        if analytics.final_energy <= 4:
            withdraws.append("long sessions")
        if analytics.avg_user_words_per_turn <= 10:
            withdraws.append("abstract questions")

        # Merge with previous (keep all unique signals)
        prev_opens = (previous.opens_up_when if previous else [])
        prev_withdraws = (previous.withdraws_when if previous else [])
        style.opens_up_when = list(dict.fromkeys(prev_opens + opens_up))[:6]
        style.withdraws_when = list(dict.fromkeys(prev_withdraws + withdraws))[:6]

        style.best_session_energy = max(
            analytics.final_energy,
            (previous.best_session_energy if previous else 0),
        )
        style.updated_at = datetime.now(timezone.utc).isoformat()

        return style

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "opens_up_when": json.dumps(self.opens_up_when),
            "withdraws_when": json.dumps(self.withdraws_when),
            "avg_words_per_turn": round(self.avg_words_per_turn, 1),
            "total_sessions": self.total_sessions,
            "avg_session_length": round(self.avg_session_length, 1),
            "best_session_energy": self.best_session_energy,
            "updated_at": self.updated_at,
        }

    def to_prompt_context(self) -> str:
        """
        Returns a short natural-language string injected at session start,
        so Drishti knows how to approach this specific person from turn one.
        """
        if self.total_sessions <= 1:
            return ""  # First session — no style data yet

        parts = []
        if self.opens_up_when:
            parts.append(f"This person tends to open up with: {', '.join(self.opens_up_when[:3])}.")
        if self.withdraws_when:
            parts.append(f"They tend to disengage when: {', '.join(self.withdraws_when[:3])}.")
        if self.avg_words_per_turn > 40:
            parts.append("They typically write detailed, thoughtful replies.")
        elif self.avg_words_per_turn < 15:
            parts.append("They typically respond briefly — don't pressure for more.")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Supabase Persistence
# ---------------------------------------------------------------------------

async def save_session_analytics(
    db,
    analytics: SessionAnalytics,
) -> None:
    """Upsert session analytics record into Supabase."""
    try:
        await db.table("session_analytics").insert(analytics.to_dict()).execute()
        logger.info(
            "Session analytics saved | session_id=%s | agent_id=%s",
            analytics.session_id, analytics.agent_id
        )
    except Exception as e:
        # Analytics failure should never break the user experience
        logger.warning("Failed to save session analytics (non-fatal): %s", e)


async def save_conversation_style(
    db,
    style: ConversationStyleMemory,
) -> None:
    """Upsert conversation style meta-memory for this agent."""
    try:
        await (
            db.table("conversation_styles")
            .upsert(style.to_dict(), on_conflict="agent_id")
            .execute()
        )
        logger.info(
            "Conversation style saved | agent_id=%s | sessions=%d",
            style.agent_id, style.total_sessions
        )
    except Exception as e:
        logger.warning("Failed to save conversation style (non-fatal): %s", e)


async def load_conversation_style(
    db,
    agent_id: UUID,
) -> Optional[ConversationStyleMemory]:
    """Load existing conversation style for an agent, if any."""
    try:
        result = (
            await db.table("conversation_styles")
            .select("*")
            .eq("agent_id", str(agent_id))
            .execute()
        )
        if result.data:
            raw = result.data[0]
            style = ConversationStyleMemory(agent_id)
            style.opens_up_when = json.loads(raw.get("opens_up_when", "[]"))
            style.withdraws_when = json.loads(raw.get("withdraws_when", "[]"))
            style.avg_words_per_turn = float(raw.get("avg_words_per_turn", 0))
            style.total_sessions = int(raw.get("total_sessions", 0))
            style.avg_session_length = float(raw.get("avg_session_length", 0))
            style.best_session_energy = float(raw.get("best_session_energy", 0))
            style.updated_at = raw.get("updated_at", "")
            logger.info(
                "Conversation style loaded | agent_id=%s | sessions=%d",
                agent_id, style.total_sessions
            )
            return style
    except Exception as e:
        logger.warning("Failed to load conversation style (non-fatal): %s", e)
    return None


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

ENGAGEMENT_SIGNALS = [
    "actually", "that reminds me", "wait", "that's true", "now that i think",
    "yeah exactly", "oh yeah", "never thought", "interesting", "you know what",
]

WITHDRAWAL_SIGNALS = [
    "idk", "i don't know", "i dont know", "whatever", "enough",
    "stop", "tired", "bored", "ok ok", "not sure",
]


def _get_role(message: Any) -> str:
    if isinstance(message, dict):
        return message.get("role", "")
    return getattr(message, "role", "") or getattr(message, "type", "")


def _get_content(message: Any) -> str:
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


def _compute_avg_words(user_messages: List[Any]) -> float:
    if not user_messages:
        return 0.0
    word_counts = [len(_get_content(m).split()) for m in user_messages]
    return sum(word_counts) / len(word_counts)


def _count_signals(user_messages: List[Any], signals: List[str]) -> int:
    count = 0
    for m in user_messages:
        text = _get_content(m).lower()
        if any(s in text for s in signals):
            count += 1
    return count


def _running_avg(current_avg: float, new_value: float, n: int) -> float:
    """Incremental running average."""
    if n <= 1:
        return new_value
    return current_avg + (new_value - current_avg) / n
