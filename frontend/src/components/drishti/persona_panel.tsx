"use client";

/**
 * Drishti Persona Panel
 * ======================
 *
 * The "What I'm Learning About You" sidebar shown during the interview.
 *
 * Displays human-readable identity labels with star ratings.
 * Updates live as the conversation progresses.
 * Each newly-updated label glows for 2 seconds then settles.
 *
 * Only shows labels with confidence >= 0.4 (the threshold below which
 * Drishti doesn't know enough to say something about the person).
 *
 * Empty state: "I'm still learning about you..." with a slow pulse.
 */

import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useDrishti, type PersonaLabel } from "@/context/SimulationContext";

// ---------------------------------------------------------------------------
// Star Rating
// ---------------------------------------------------------------------------

function StarRating({ stars, confidence }: { stars: number; confidence: number }) {
  return (
    <div className="flex items-center gap-0.5">
      {Array.from({ length: 5 }).map((_, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.06, duration: 0.2 }}
          style={{
            fontSize: "10px",
            color: i < stars ? "rgba(99,102,241,0.9)" : "rgba(240,240,248,0.12)",
          }}
        >
          {i < stars ? "★" : "☆"}
        </motion.span>
      ))}
      <span
        className="ml-1.5 text-[10px] tabular-nums"
        style={{ color: "rgba(240,240,248,0.2)", fontFamily: "var(--font-geist-mono)" }}
      >
        {Math.round(confidence * 100)}%
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Individual Label Row
// ---------------------------------------------------------------------------

function LabelRow({
  label,
  isNew,
}: {
  label: PersonaLabel;
  isNew: boolean;
}) {
  const [glowing, setGlowing] = useState(isNew);

  useEffect(() => {
    if (isNew) {
      const timer = setTimeout(() => setGlowing(false), 2200);
      return () => clearTimeout(timer);
    }
  }, [isNew]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: 8 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 8 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="flex items-start justify-between gap-3 py-2.5 px-3 rounded-sm transition-all duration-500"
      style={{
        background: glowing
          ? "rgba(99,102,241,0.06)"
          : "transparent",
        borderLeft: glowing
          ? "1px solid rgba(99,102,241,0.3)"
          : "1px solid transparent",
      }}
    >
      <span
        className="text-sm font-light flex-1 leading-snug"
        style={{
          fontFamily: "'Inter', ui-sans-serif, sans-serif",
          color: glowing
            ? "rgba(240,240,248,0.9)"
            : "rgba(240,240,248,0.65)",
          transition: "color 0.5s ease",
        }}
      >
        {label.label}
      </span>
      <div className="shrink-0 pt-0.5">
        <StarRating stars={label.stars} confidence={label.confidence} />
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Persona Panel
// ---------------------------------------------------------------------------

export function PersonaPanel() {
  const { personaLabels } = useDrishti();
  const prevLabelsRef = useRef<Set<string>>(new Set());
  const [newLabelKeys, setNewLabelKeys] = useState<Set<string>>(new Set());

  // Detect newly added labels for glow animation
  useEffect(() => {
    const visibleLabels = personaLabels.filter((l) => l.confidence >= 0.4);
    const currentKeys = new Set(visibleLabels.map((l) => l.label));
    const prev = prevLabelsRef.current;
    const freshKeys = new Set([...currentKeys].filter((k) => !prev.has(k)));

    if (freshKeys.size > 0) {
      setNewLabelKeys(freshKeys);
      const timer = setTimeout(() => setNewLabelKeys(new Set()), 2500);
      prevLabelsRef.current = currentKeys;
      return () => clearTimeout(timer);
    }
    prevLabelsRef.current = currentKeys;
  }, [personaLabels]);

  const visibleLabels = personaLabels
    .filter((l) => l.confidence >= 0.4)
    .sort((a, b) => b.stars - a.stars || b.confidence - a.confidence);

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div
        className="px-5 py-4 shrink-0"
        style={{ borderBottom: "1px solid rgba(240,240,248,0.05)" }}
      >
        <p
          className="text-[10px] tracking-widest uppercase"
          style={{
            color: "rgba(240,240,248,0.25)",
            fontFamily: "var(--font-geist-mono)",
          }}
        >
          What I&apos;m Learning About You
        </p>
      </div>

      {/* Labels */}
      <div className="flex-1 overflow-y-auto px-2 py-3">
        <AnimatePresence mode="popLayout">
          {visibleLabels.length === 0 ? (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center h-32 gap-3"
            >
              {/* Breathing orb */}
              <motion.div
                animate={{
                  opacity: [0.2, 0.5, 0.2],
                  scale: [0.9, 1.0, 0.9],
                }}
                transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
                className="w-2 h-2 rounded-full"
                style={{ background: "rgba(99,102,241,0.5)" }}
              />
              <p
                className="text-xs text-center leading-relaxed"
                style={{
                  color: "rgba(240,240,248,0.2)",
                  fontFamily: "'Inter', ui-sans-serif, sans-serif",
                  maxWidth: "140px",
                }}
              >
                I&apos;m still learning about you...
              </p>
            </motion.div>
          ) : (
            visibleLabels.map((label) => (
              <LabelRow
                key={label.label}
                label={label}
                isNew={newLabelKeys.has(label.label)}
              />
            ))
          )}
        </AnimatePresence>
      </div>

      {/* Evidence count footer */}
      {visibleLabels.length > 0 && (
        <div
          className="px-5 py-3 shrink-0"
          style={{ borderTop: "1px solid rgba(240,240,248,0.04)" }}
        >
          <p
            className="text-[10px] tracking-widest"
            style={{
              color: "rgba(240,240,248,0.15)",
              fontFamily: "var(--font-geist-mono)",
            }}
          >
            {visibleLabels.length} REFLECTION{visibleLabels.length !== 1 ? "S" : ""} · UPDATING LIVE
          </p>
        </div>
      )}
    </div>
  );
}
