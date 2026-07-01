"use client";

/**
 * Explanation Card — The Explainability Moat
 * ===========================================
 *
 * Expands when user clicks "▼ Why?" on a summary sentence.
 * Shows the specific reasoning, evidence, confidence, and update conditions.
 */

import React from "react";
import { motion } from "framer-motion";
import { type SummarySentence } from "@/context/SimulationContext";

interface ExplanationCardProps {
  sentence: SummarySentence;
}

export function ExplanationCard({ sentence }: ExplanationCardProps) {
  const arg = sentence.argument;
  
  if (!arg) return null;

  // Render confidence bar (10 blocks)
  const confidencePercent = Math.round((sentence.confidence || 0) * 100);
  const filledBlocks = Math.round((sentence.confidence || 0) * 10);
  const bar = "█".repeat(filledBlocks) + "░".repeat(10 - filledBlocks);

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      style={{ overflow: "hidden" }}
    >
      <div
        className="mt-3 px-5 py-5 rounded-sm space-y-6"
        style={{
          background: "rgba(99,102,241,0.03)",
          border: "1px solid rgba(99,102,241,0.08)",
        }}
      >
        {/* Reasoning */}
        {arg.reasoning && (
          <p
            className="text-sm leading-relaxed"
            style={{
              fontFamily: "'Inter', ui-sans-serif, sans-serif",
              color: "rgba(240,240,248,0.7)",
              fontWeight: 300,
            }}
          >
            {arg.reasoning}
          </p>
        )}

        {/* Evidence */}
        {arg.evidence && arg.evidence.length > 0 && (
          <div className="space-y-4 pt-2">
            <div className="flex items-center gap-3 mb-1">
              <span
                className="text-[10px] tracking-widest"
                style={{ color: "rgba(240,240,248,0.3)", fontFamily: "var(--font-geist-mono)" }}
              >
                BASED ON {arg.evidence.length} MOMENT{arg.evidence.length > 1 ? "S" : ""}
              </span>
              <div className="flex-1 h-px bg-white/5" />
            </div>
            
            {arg.evidence.map((ev, idx) => (
              <div key={idx} className="space-y-2">
                <p
                  className="text-sm italic leading-relaxed"
                  style={{
                    fontFamily: "'Inter', ui-sans-serif, sans-serif",
                    color: "rgba(240,240,248,0.5)",
                    fontWeight: 300,
                  }}
                >
                  &ldquo;{ev.quote}&rdquo;
                </p>
                {ev.interpretation && (
                  <p
                    className="text-[13px] leading-relaxed pl-3 border-l"
                    style={{
                      fontFamily: "'Inter', ui-sans-serif, sans-serif",
                      color: "rgba(99,102,241,0.6)",
                      borderColor: "rgba(99,102,241,0.2)",
                      fontWeight: 300,
                    }}
                  >
                    {ev.interpretation}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Falsifiability / Update Conditions */}
        {arg.update_conditions && (
          <div className="pt-2">
            <p
              className="text-[10px] tracking-widest mb-2"
              style={{
                color: "rgba(240,240,248,0.3)",
                fontFamily: "var(--font-geist-mono)",
              }}
            >
              WHAT WOULD MAKE ME UPDATE THIS UNDERSTANDING?
            </p>
            <p
              className="text-[13px] leading-relaxed"
              style={{
                fontFamily: "'Inter', ui-sans-serif, sans-serif",
                color: "rgba(240,240,248,0.45)",
                fontWeight: 300,
              }}
            >
              {arg.update_conditions}
            </p>
          </div>
        )}

        {/* Confidence Footer */}
        <div className="pt-4 flex items-center justify-between border-t border-white/5">
          <div className="flex items-center gap-3">
            <span
              className="text-[10px] tracking-widest"
              style={{ color: "rgba(240,240,248,0.3)", fontFamily: "var(--font-geist-mono)" }}
            >
              CONFIDENCE
            </span>
            <span
              className="text-xs tracking-[0.2em]"
              style={{ color: "rgba(99,102,241,0.6)", fontFamily: "var(--font-geist-mono)" }}
            >
              {bar}
            </span>
            <span
              className="text-[10px] font-mono"
              style={{ color: "rgba(240,240,248,0.4)" }}
            >
              {confidencePercent}%
            </span>
          </div>
          
          <span
            className="text-[10px] italic"
            style={{ color: "rgba(240,240,248,0.2)" }}
          >
            This may change as we talk more.
          </span>
        </div>
        
      </div>
    </motion.div>
  );
}
