"use client";

/**
 * Drishti Landing Page
 * =====================
 *
 * The first thing a user sees. Calm. Minimal. Deliberate.
 *
 * DRISHTI
 *
 * Your life isn't a timeline.
 * It's a flowing river.
 * Let's understand where you are.
 *
 * [ Begin ]
 *
 * A subtle river shimmer breathes at the bottom of the screen.
 * Text staggered fade-up. Nothing else. No clutter.
 */

import React from "react";
import { motion } from "framer-motion";
import { useDrishti } from "@/context/SimulationContext";

export function LandingPage() {
  const { phase, submitIdentity } = useDrishti();

  // Clicking Begin transitions to IDENTITY phase
  // We need to set phase to IDENTITY — context handles this via submitIdentity
  // But Begin just advances phase, identity is collected next.
  // So we call a direct phase setter via a workaround:
  // The DrishtiProvider exposes no direct setPhase, but we can use
  // submitIdentity with empty data to trigger it... instead let's use
  // a dedicated begin handler.
  const handleBegin = () => {
    // We trigger the IDENTITY phase by dispatching a custom event
    // that DrishtiProvider listens to — or simpler: expose setPhase.
    // Since we exported useDrishti, and the provider has internal setPhase,
    // the cleanest approach is to transition via a dedicated prop.
    // For now we use a trick: the phase setter is called from within.
    // Actually — context only exposes submitIdentity which POSTs to backend.
    // We need a simple phase transition. Let's use a workaround in context.
    // The IDENTITY form itself will call submitIdentity.
    // We just need to navigate to IDENTITY — we'll use a window event approach.
    window.dispatchEvent(new CustomEvent("drishti:begin"));
  };

  // Listen in context... actually the cleanest fix is to just move the
  // phase transition to a simple exported action. Since I control the context,
  // let me use the correct pattern: expose beginOnboarding in context value.
  // For now, render a Begin button that calls the transition via a ref approach.

  return (
    <div className="h-screen w-full flex flex-col items-center justify-center relative overflow-hidden">

      {/* River shimmer — subtle gradient animation at bottom 30% */}
      <div
        className="absolute bottom-0 left-0 right-0 h-[40%] pointer-events-none"
        style={{
          background: "linear-gradient(to top, rgba(99,102,241,0.06) 0%, transparent 100%)",
        }}
      />
      <div className="absolute bottom-0 left-0 right-0 h-[20%] animate-river-shimmer pointer-events-none opacity-60" />

      {/* Subtle grid */}
      <div
        className="absolute inset-0 pointer-events-none opacity-[0.015]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(99,102,241,0.5) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,102,241,0.5) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
        }}
      />

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center text-center px-8 max-w-lg">

        {/* Logo mark */}
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="mb-10"
        >
          <div
            className="w-10 h-10 rounded-full mb-8 mx-auto"
            style={{
              background: "radial-gradient(circle, rgba(99,102,241,0.8) 0%, rgba(99,102,241,0.2) 70%, transparent 100%)",
              boxShadow: "0 0 40px rgba(99,102,241,0.3)",
            }}
          />
        </motion.div>

        {/* Wordmark */}
        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.1, ease: "easeOut" }}
          className="font-display text-5xl font-light tracking-widest mb-10"
          style={{
            fontFamily: "'Inter', ui-sans-serif, sans-serif",
            letterSpacing: "0.25em",
            color: "rgba(240,240,248,0.95)",
          }}
        >
          DRISHTI
        </motion.h1>

        {/* Tagline */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.4, ease: "easeOut" }}
          className="space-y-1 mb-14"
        >
          <p
            className="text-lg font-light"
            style={{
              fontFamily: "'Inter', ui-sans-serif, sans-serif",
              color: "rgba(240,240,248,0.55)",
              lineHeight: 1.7,
            }}
          >
            Your life isn&apos;t a timeline.
          </p>
          <p
            className="text-lg font-light"
            style={{
              fontFamily: "'Inter', ui-sans-serif, sans-serif",
              color: "rgba(240,240,248,0.55)",
              lineHeight: 1.7,
            }}
          >
            It&apos;s a flowing river.
          </p>
          <p
            className="text-base mt-4"
            style={{
              fontFamily: "'Inter', ui-sans-serif, sans-serif",
              color: "rgba(240,240,248,0.3)",
              lineHeight: 1.7,
            }}
          >
            Let&apos;s understand where you are.
          </p>
        </motion.div>

        {/* Begin button */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.8, ease: "easeOut" }}
        >
          <BeginButton />
        </motion.div>

        {/* Tiny attribution */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.7, delay: 1.4 }}
          className="mt-16 text-xs tracking-widest"
          style={{ color: "rgba(240,240,248,0.12)", fontFamily: "var(--font-geist-mono)" }}
        >
          NO LOGIN · NO DASHBOARD · JUST A CONVERSATION
        </motion.p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Begin Button — internal component with hover state
// ---------------------------------------------------------------------------

function BeginButton() {
  const [hovered, setHovered] = React.useState(false);

  return (
    <button
      id="drishti-begin-btn"
      onClick={() => window.dispatchEvent(new CustomEvent("drishti:begin"))}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        fontFamily: "'Inter', ui-sans-serif, sans-serif",
        letterSpacing: "0.15em",
        fontSize: "13px",
        fontWeight: 500,
        color: hovered ? "rgba(240,240,248,0.9)" : "rgba(240,240,248,0.45)",
        border: `1px solid ${hovered ? "rgba(99,102,241,0.6)" : "rgba(240,240,248,0.1)"}`,
        background: hovered ? "rgba(99,102,241,0.08)" : "transparent",
        padding: "14px 40px",
        cursor: "pointer",
        transition: "all 0.25s ease",
        borderRadius: "2px",
        boxShadow: hovered ? "0 0 20px rgba(99,102,241,0.15)" : "none",
      }}
    >
      BEGIN
    </button>
  );
}
