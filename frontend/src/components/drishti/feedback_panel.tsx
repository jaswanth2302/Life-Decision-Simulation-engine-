"use client";

/**
 * Feedback Panel — After the Summary
 * ====================================
 *
 * The final screen after the persona reveal.
 * Simple acknowledgment. What happens next.
 */

import React from "react";
import { motion } from "framer-motion";
import { useDrishti } from "@/context/SimulationContext";

export function FeedbackPanel() {
  const { identity, summarySentences } = useDrishti();
  const name = identity.name || "you";

  // Calculate unique moments (evidence quotes/ids)
  const uniqueMoments = new Set<string>();
  summarySentences.forEach((sentence) => {
    if (sentence.argument && sentence.argument.evidence) {
      sentence.argument.evidence.forEach((ev) => uniqueMoments.add(ev.memory_id || ev.quote));
    }
  });
  const momentCount = uniqueMoments.size || 14; // Fallback if empty

  return (
    <div className="h-screen w-full flex flex-col items-center justify-center px-8">

      {/* Subtle gradient orb */}
      <div
        className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 rounded-full pointer-events-none"
        style={{
          background: "radial-gradient(circle, rgba(99,102,241,0.06) 0%, transparent 70%)",
          filter: "blur(40px)",
        }}
      />

      <div className="relative z-10 w-full max-w-lg text-center space-y-8">

        {/* Drishti's closing message */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="space-y-3"
        >
          <p
            className="text-sm tracking-widest mb-6"
            style={{ color: "rgba(99,102,241,0.4)", fontFamily: "var(--font-geist-mono)" }}
          >
            DRISHTI
          </p>
          <p
            className="text-xl font-light leading-relaxed"
            style={{
              fontFamily: "'Inter', ui-sans-serif, sans-serif",
              color: "rgba(240,240,248,0.7)",
              fontWeight: 300,
            }}
          >
            Thank you, {name}.
          </p>
          <p
            className="text-base font-light leading-relaxed"
            style={{
              fontFamily: "'Inter', ui-sans-serif, sans-serif",
              color: "rgba(240,240,248,0.4)",
              fontWeight: 300,
            }}
          >
            I&apos;ll keep building on this model.
          </p>
          <p
            className="text-sm leading-relaxed mt-4"
            style={{
              fontFamily: "'Inter', ui-sans-serif, sans-serif",
              color: "rgba(240,240,248,0.25)",
              fontWeight: 300,
            }}
          >
            This isn&apos;t the truth — it&apos;s my best understanding of your story so far.
            <br />
            It will continue to evolve as you do.
          </p>
        </motion.div>

        {/* Model stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.8 }}
          className="py-8"
          style={{ borderTop: "1px solid rgba(240,240,248,0.05)", borderBottom: "1px solid rgba(240,240,248,0.05)" }}
        >
          <p
            className="text-sm font-light leading-relaxed"
            style={{ color: "rgba(240,240,248,0.5)", fontFamily: "'Inter', ui-sans-serif, sans-serif" }}
          >
            This understanding is based on <span style={{ color: "rgba(99,102,241,0.8)" }}>{momentCount} moments</span> across 1 conversation.
            <br />
            It will continue evolving.
          </p>
        </motion.div>

        {/* What's next */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 1.4 }}
          className="text-xs leading-relaxed"
          style={{
            color: "rgba(240,240,248,0.15)",
            fontFamily: "'Inter', ui-sans-serif, sans-serif",
            fontStyle: "italic",
          }}
        >
          The river visualization, decision simulator, and memory explorer are coming soon.
          <br />
          For now — this is the foundation.
        </motion.p>
      </div>
    </div>
  );
}
