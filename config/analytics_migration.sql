-- Drishti Analytics Tables
-- Run this in the Supabase SQL Editor
-- These tables are append-only / upsert — never breaking, never blocking.

-- ──────────────────────────────────────────────────────────
-- TABLE 1: session_analytics
-- One row per completed Drishti session.
-- Powers the dashboard that will teach us more than prompts.
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS session_analytics (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id              UUID NOT NULL,
    session_id            TEXT NOT NULL UNIQUE,
    name                  TEXT,
    timestamp             TIMESTAMPTZ DEFAULT now(),

    -- Conversation length
    total_turns           INTEGER DEFAULT 0,

    -- Final conversation state axes
    final_energy          INTEGER DEFAULT 0,
    final_curiosity       INTEGER DEFAULT 0,
    final_certainty       INTEGER DEFAULT 0,

    -- Insight quality
    insight_count         INTEGER DEFAULT 0,
    strong_insight_count  INTEGER DEFAULT 0,

    -- Key moments
    reframe_used          BOOLEAN DEFAULT FALSE,
    closing_observation   TEXT DEFAULT '',

    -- Persona
    persona_label_count   INTEGER DEFAULT 0,
    top_labels            JSONB DEFAULT '[]',

    -- Summary quality
    summary_generated     BOOLEAN DEFAULT FALSE,
    summary_reframe_count INTEGER DEFAULT 0,

    -- Engagement signals
    avg_user_words_per_turn FLOAT DEFAULT 0,
    opens_up_signals      INTEGER DEFAULT 0,
    withdrawal_signals    INTEGER DEFAULT 0,

    -- User feedback (filled in via /feedback/submit)
    surprise_rating       INTEGER,          -- 1–5, filled later
    felt_understood       INTEGER,          -- 1–10, filled later
    reflection_accepted   BOOLEAN          -- did user confirm the key reflection?
);

CREATE INDEX IF NOT EXISTS idx_session_analytics_agent_id
    ON session_analytics (agent_id);

CREATE INDEX IF NOT EXISTS idx_session_analytics_timestamp
    ON session_analytics (timestamp DESC);


-- ──────────────────────────────────────────────────────────
-- TABLE 2: conversation_styles
-- One row per agent — upserted after each session.
-- This is how Drishti learns HOW to talk to each person.
-- ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversation_styles (
    agent_id              UUID PRIMARY KEY,
    opens_up_when         JSONB DEFAULT '[]',    -- ["reflections", "examples"]
    withdraws_when        JSONB DEFAULT '[]',    -- ["rapid questions", "long sessions"]
    avg_words_per_turn    FLOAT DEFAULT 0,
    total_sessions        INTEGER DEFAULT 0,
    avg_session_length    FLOAT DEFAULT 0,
    best_session_energy   FLOAT DEFAULT 0,
    updated_at            TIMESTAMPTZ DEFAULT now()
);


-- ──────────────────────────────────────────────────────────
-- USEFUL QUERIES (for your dashboard later)
-- ──────────────────────────────────────────────────────────

-- Average energy per session (trend over time)
-- SELECT DATE(timestamp), AVG(final_energy) FROM session_analytics GROUP BY 1 ORDER BY 1;

-- Reframe usage rate
-- SELECT COUNT(*) FILTER (WHERE reframe_used) * 100.0 / COUNT(*) AS reframe_rate FROM session_analytics;

-- Average session length
-- SELECT AVG(total_turns) FROM session_analytics WHERE summary_generated = TRUE;

-- Most common top labels across all users
-- SELECT label, COUNT(*) FROM (
--   SELECT jsonb_array_elements_text(top_labels) AS label FROM session_analytics
-- ) t GROUP BY label ORDER BY count DESC LIMIT 10;

-- Reflection acceptance rate (once feedback is collected)
-- SELECT COUNT(*) FILTER (WHERE reflection_accepted) * 100.0 / NULLIF(COUNT(*), 0)
-- FROM session_analytics WHERE reflection_accepted IS NOT NULL;
