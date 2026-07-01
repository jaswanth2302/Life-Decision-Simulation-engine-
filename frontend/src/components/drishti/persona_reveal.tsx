"use client";

/**
 * Persona Reveal — The Magical Moment
 * =====================================
 *
 * The transition screen after the interview completes.
 *
 * 1. Brief pause
 * 2. "I've been listening carefully." fades in
 * 3. Five sentences appear one at a time (typewriter, 2s gap)
 *    Each sentence has:
 *    - The observation text
 *    - A subtle "▼ Why?" affordance
 *    - Thumbs up / thumbs down / edit icons
 * 4. After all 5: "Which sentence surprised you the most?"
 * 5. Continue →
 *
 * The pacing is deliberate. Each sentence arrives with weight.
 * The user starts anticipating the next one.
 */

import React, { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useDrishti } from "@/context/SimulationContext";
import { ExplanationCard } from "./explanation_card";

// ---------------------------------------------------------------------------
// Sentence Row
// ---------------------------------------------------------------------------

function SentenceRow({
  sentence,
  index,
  isVisible,
  isExpanded,
  isSurprise,
  onToggleWhy,
  onFeedback,
  onMarkSurprise,
  memoryTexts,
}: {
  sentence: { text: string; source_memory_ids: string[]; is_reframe: boolean };
  index: number;
  isVisible: boolean;
  isExpanded: boolean;
  isSurprise: boolean;
  onToggleWhy: () => void;
  onFeedback: (sentiment: "up" | "down" | "edit") => void;
  onMarkSurprise: () => void;
  memoryTexts: Record<string, string>;
}) {
  const [editMode, setEditMode] = useState(false);
  const [editText, setEditText] = useState("");
  const [feedbackGiven, setFeedbackGiven] = useState<"up" | "down" | "edit" | null>(null);

  if (!isVisible) return null;

  const handleFeedback = (sentiment: "up" | "down" | "edit") => {
    if (sentiment === "edit") {
      setEditMode(true);
    } else {
      setFeedbackGiven(sentiment);
      onFeedback(sentiment);
    }
  };

  const handleEditSubmit = () => {
    if (editText.trim()) {
      setFeedbackGiven("edit");
      setEditMode(false);
      onFeedback("edit");
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.65, ease: "easeOut" }}
      className="space-y-0"
    >
      <div
        className="py-5 group"
        style={{ borderBottom: "1px solid rgba(240,240,248,0.04)" }}
      >
        {/* Reframe badge */}
        {sentence.is_reframe && (
          <div className="mb-2">
            <span
              className="text-[9px] tracking-widest px-2 py-0.5"
              style={{
                color: "rgba(99,102,241,0.6)",
                background: "rgba(99,102,241,0.08)",
                fontFamily: "var(--font-geist-mono)",
                borderRadius: "2px",
              }}
            >
              REFRAME
            </span>
          </div>
        )}

        {/* Sentence text */}
        <p
          className="text-lg font-light leading-relaxed mb-3"
          style={{
            fontFamily: "'Inter', ui-sans-serif, sans-serif",
            color: "rgba(240,240,248,0.85)",
            letterSpacing: "-0.01em",
          }}
        >
          {sentence.text}
        </p>

        {/* Actions row */}
        <div className="flex items-center justify-between">
          {/* Why button */}
          <button
            onClick={onToggleWhy}
            className="text-[10px] tracking-widest transition-colors duration-200"
            style={{
              fontFamily: "var(--font-geist-mono)",
              color: isExpanded ? "rgba(99,102,241,0.7)" : "rgba(240,240,248,0.18)",
              background: "transparent",
              border: "none",
              cursor: "pointer",
            }}
          >
            {isExpanded ? "▲ WHY?" : "▼ WHY?"}
          </button>

          {/* Feedback + surprise */}
          <div className="flex items-center gap-3">
            {/* Surprise marker */}
            <button
              onClick={onMarkSurprise}
              title="This sentence surprised me"
              className="text-sm transition-all duration-200"
              style={{
                color: isSurprise ? "rgba(245,158,11,0.8)" : "rgba(240,240,248,0.12)",
                background: "transparent",
                border: "none",
                cursor: "pointer",
                transform: isSurprise ? "scale(1.2)" : "scale(1)",
              }}
            >
              ✦
            </button>

            {/* Thumbs */}
            {feedbackGiven ? (
              <span
                className="text-[10px] tracking-widest"
                style={{ color: "rgba(240,240,248,0.2)", fontFamily: "var(--font-geist-mono)" }}
              >
                NOTED
              </span>
            ) : (
              <>
                <button
                  onClick={() => handleFeedback("up")}
                  className="text-xs transition-colors duration-200"
                  style={{ color: "rgba(240,240,248,0.2)", background: "transparent", border: "none", cursor: "pointer" }}
                  title="Accurate"
                >
                  👍
                </button>
                <button
                  onClick={() => handleFeedback("down")}
                  className="text-xs transition-colors duration-200"
                  style={{ color: "rgba(240,240,248,0.2)", background: "transparent", border: "none", cursor: "pointer" }}
                  title="Not quite"
                >
                  👎
                </button>
                <button
                  onClick={() => handleFeedback("edit")}
                  className="text-[10px] tracking-widest transition-colors duration-200"
                  style={{ color: "rgba(240,240,248,0.12)", background: "transparent", border: "none", cursor: "pointer", fontFamily: "var(--font-geist-mono)" }}
                  title="Correct it"
                >
                  EDIT
                </button>
              </>
            )}
          </div>
        </div>

        {/* Edit mode */}
        <AnimatePresence>
          {editMode && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-3 space-y-2"
            >
              <textarea
                autoFocus
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                placeholder="What's missing or wrong?"
                rows={2}
                className="w-full bg-transparent resize-none text-sm"
                style={{
                  fontFamily: "'Inter', ui-sans-serif, sans-serif",
                  color: "rgba(240,240,248,0.7)",
                  border: "1px solid rgba(240,240,248,0.1)",
                  padding: "8px 12px",
                  outline: "none",
                  fontWeight: 300,
                  borderRadius: "2px",
                }}
              />
              <div className="flex gap-3">
                <button
                  onClick={handleEditSubmit}
                  className="text-[10px] tracking-widest"
                  style={{
                    color: "rgba(99,102,241,0.7)",
                    background: "transparent",
                    border: "1px solid rgba(99,102,241,0.2)",
                    padding: "4px 12px",
                    cursor: "pointer",
                    fontFamily: "var(--font-geist-mono)",
                    borderRadius: "2px",
                  }}
                >
                  SAVE
                </button>
                <button
                  onClick={() => setEditMode(false)}
                  className="text-[10px] tracking-widest"
                  style={{
                    color: "rgba(240,240,248,0.2)",
                    background: "transparent",
                    border: "none",
                    cursor: "pointer",
                    fontFamily: "var(--font-geist-mono)",
                  }}
                >
                  CANCEL
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Explanation card */}
      <AnimatePresence>
        {isExpanded && (
          <ExplanationCard
            sentence={sentence as any}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Persona Reveal
// ---------------------------------------------------------------------------

export function PersonaReveal() {
  const {
    identity,
    summarySentences,
    expandedSentenceIdx,
    surpriseSentenceIdx,
    toggleExplanation,
    submitSurprise,
    submitFeedback,
    acknowledgeComplete,
  } = useDrishti();

  const name = identity.name || "you";
  const [introStep, setIntroStep] = useState(0); // 0=none, 1=first line, 2=second line, 3=third line, 4=done
  const [visibleCount, setVisibleCount] = useState(0);
  const [showSurprisePrompt, setShowSurprisePrompt] = useState(false);
  const [showContinue, setShowContinue] = useState(false);

  // Phase 1: staggered header intro
  useEffect(() => {
    const timers = [
      setTimeout(() => setIntroStep(1), 800),     // "I've been listening carefully..."
      setTimeout(() => setIntroStep(2), 2800),    // "I think I understand you a little better now."
      setTimeout(() => setIntroStep(3), 4800),    // "Here's what I'm seeing."
      setTimeout(() => setIntroStep(4), 6000),    // Start showing sentences
    ];
    return () => timers.forEach(clearTimeout);
  }, []);

  // Phase 2: reveal sentences one at a time with 2s gaps
  useEffect(() => {
    if (introStep < 4) return;
    if (visibleCount >= summarySentences.length) {
      // All revealed — show surprise prompt after a beat
      const timer = setTimeout(() => setShowSurprisePrompt(true), 1200);
      return () => clearTimeout(timer);
    }

    const delay = visibleCount === 0 ? 800 : 2000;
    const timer = setTimeout(() => {
      setVisibleCount((prev) => prev + 1);
    }, delay);
    return () => clearTimeout(timer);
  }, [introStep, visibleCount, summarySentences.length]);

  // Show continue after surprise prompt
  useEffect(() => {
    if (!showSurprisePrompt) return;
    const timer = setTimeout(() => setShowContinue(true), 1500);
    return () => clearTimeout(timer);
  }, [showSurprisePrompt]);

  // Empty state guard
  if (summarySentences.length === 0) {
    return (
      <div className="h-screen w-full flex items-center justify-center">
        <motion.div
          animate={{ opacity: [0.3, 0.7, 0.3] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="w-2 h-2 rounded-full"
          style={{ background: "rgba(99,102,241,0.6)" }}
        />
      </div>
    );
  }

  // Build a simple memory text lookup from source IDs
  // (in practice these would come from a memo/fetch, for now we show empty texts
  // which will trigger the "overall pattern" fallback in ExplanationCard)
  const memoryTexts: Record<string, string> = {};

  return (
    <div className="h-screen w-full overflow-y-auto flex justify-center">
      <div className="w-full max-w-2xl px-8 py-20">

        {/* Header Sequence */}
        <div className="mb-14 space-y-4">
          <AnimatePresence>
            {introStep >= 1 && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              >
                <p
                  className="text-sm tracking-widest mb-3"
                  style={{
                    color: "rgba(99,102,241,0.5)",
                    fontFamily: "var(--font-geist-mono)",
                  }}
                >
                  DRISHTI
                </p>
                <h2
                  className="text-2xl font-light"
                  style={{
                    fontFamily: "'Inter', ui-sans-serif, sans-serif",
                    color: "rgba(240,240,248,0.6)",
                    fontWeight: 300,
                    letterSpacing: "-0.01em",
                  }}
                >
                  I&apos;ve been listening carefully, {name}.
                </h2>
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {introStep >= 2 && (
              <motion.p
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="text-xl font-light"
                style={{
                  fontFamily: "'Inter', ui-sans-serif, sans-serif",
                  color: "rgba(240,240,248,0.45)",
                  fontWeight: 300,
                  letterSpacing: "-0.01em",
                }}
              >
                I think I understand you a little better now.
              </motion.p>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {introStep >= 3 && (
              <motion.p
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="text-xl font-light"
                style={{
                  fontFamily: "'Inter', ui-sans-serif, sans-serif",
                  color: "rgba(240,240,248,0.45)",
                  fontWeight: 300,
                  letterSpacing: "-0.01em",
                }}
              >
                Here&apos;s what I&apos;m seeing.
              </motion.p>
            )}
          </AnimatePresence>
        </div>

        {/* Sentences */}
        <div className="space-y-0">
          {summarySentences.map((sentence, idx) => (
            <SentenceRow
              key={idx}
              sentence={sentence}
              index={idx}
              isVisible={idx < visibleCount}
              isExpanded={expandedSentenceIdx === idx}
              isSurprise={surpriseSentenceIdx === idx}
              onToggleWhy={() => toggleExplanation(idx)}
              onFeedback={(sentiment) => submitFeedback(idx, sentiment)}
              onMarkSurprise={() => submitSurprise(idx)}
              memoryTexts={memoryTexts}
            />
          ))}
        </div>

        {/* Surprise prompt */}
        <AnimatePresence>
          {showSurprisePrompt && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: "easeOut" }}
              className="mt-12 space-y-4"
            >
              <p
                className="text-base font-light"
                style={{
                  fontFamily: "'Inter', ui-sans-serif, sans-serif",
                  color: "rgba(240,240,248,0.4)",
                }}
              >
                Which sentence surprised you the most?
              </p>
              <p
                className="text-xs"
                style={{
                  color: "rgba(240,240,248,0.18)",
                  fontFamily: "'Inter', ui-sans-serif, sans-serif",
                }}
              >
                Tap the ✦ next to any sentence above.
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Continue */}
        <AnimatePresence>
          {showContinue && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6 }}
              className="mt-10"
            >
              <button
                onClick={acknowledgeComplete}
                style={{
                  fontFamily: "'Inter', ui-sans-serif, sans-serif",
                  fontSize: "13px",
                  fontWeight: 400,
                  color: "rgba(240,240,248,0.35)",
                  background: "transparent",
                  border: "1px solid rgba(240,240,248,0.08)",
                  padding: "10px 28px",
                  cursor: "pointer",
                  letterSpacing: "0.05em",
                  borderRadius: "2px",
                  transition: "all 0.2s ease",
                }}
                onMouseEnter={(e) => {
                  (e.target as HTMLButtonElement).style.color = "rgba(240,240,248,0.6)";
                  (e.target as HTMLButtonElement).style.borderColor = "rgba(240,240,248,0.15)";
                }}
                onMouseLeave={(e) => {
                  (e.target as HTMLButtonElement).style.color = "rgba(240,240,248,0.35)";
                  (e.target as HTMLButtonElement).style.borderColor = "rgba(240,240,248,0.08)";
                }}
              >
                Continue →
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
