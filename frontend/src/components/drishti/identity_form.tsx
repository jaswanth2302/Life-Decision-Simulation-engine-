"use client";

/**
 * Drishti Identity Form
 * ======================
 *
 * Stage 1: Sequential field reveal. NOT a form.
 *
 * One question at a time. The next appears only after the previous
 * is answered. Pressing Enter advances. No submit button. No progress bar.
 * Generous whitespace. Nothing clinical.
 *
 * Fields: Name → Age → Country → Occupation → Timezone
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useDrishti, type IdentityData } from "@/context/SimulationContext";

// ---------------------------------------------------------------------------
// Field Definitions
// ---------------------------------------------------------------------------

interface FieldDef {
  key: keyof IdentityData;
  question: string;
  placeholder: string;
  type?: string;
}

const FIELDS: FieldDef[] = [
  {
    key: "name",
    question: "What's your name?",
    placeholder: "First name is enough...",
  },
  {
    key: "age",
    question: "How old are you?",
    placeholder: "Your age...",
  },
  {
    key: "country",
    question: "Where are you in the world?",
    placeholder: "Country or city...",
  },
  {
    key: "occupation",
    question: "What do you do?",
    placeholder: "Student, engineer, founder... anything works.",
  },
  {
    key: "timezone",
    question: "What timezone are you in?",
    placeholder: "e.g. Asia/Kolkata, America/New_York, Europe/London...",
  },
];

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function IdentityForm() {
  const { submitIdentity, isProcessing } = useDrishti();
  const [currentField, setCurrentField] = useState(0);
  const [answers, setAnswers] = useState<Partial<IdentityData>>({});
  const [inputValue, setInputValue] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const field = FIELDS[currentField];

  // Auto-focus input on field change
  useEffect(() => {
    const timer = setTimeout(() => {
      inputRef.current?.focus();
    }, 400);
    return () => clearTimeout(timer);
  }, [currentField]);

  const handleAdvance = useCallback(async () => {
    const trimmed = inputValue.trim();
    if (!trimmed) return;

    const newAnswers = { ...answers, [field.key]: trimmed };
    setAnswers(newAnswers);
    setInputValue("");

    if (currentField < FIELDS.length - 1) {
      setCurrentField((prev) => prev + 1);
    } else {
      // All fields collected — submit
      setIsSubmitting(true);
      await submitIdentity({
        name: newAnswers.name || "",
        age: newAnswers.age || "",
        country: newAnswers.country || "",
        occupation: newAnswers.occupation || "",
        timezone: newAnswers.timezone || "UTC",
      });
    }
  }, [inputValue, answers, field?.key, currentField, submitIdentity]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === "Enter") handleAdvance();
    },
    [handleAdvance]
  );

  const progress = currentField / FIELDS.length;

  return (
    <div className="h-screen w-full flex flex-col items-center justify-center px-8 relative">

      {/* Subtle top accent */}
      <div
        className="absolute top-0 left-0 right-0 h-[1px]"
        style={{ background: "linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent)" }}
      />

      {/* Answered fields shown above */}
      <div className="absolute top-20 left-1/2 -translate-x-1/2 w-full max-w-lg px-8 space-y-3">
        <AnimatePresence>
          {FIELDS.slice(0, currentField).map((f, idx) => (
            <motion.div
              key={f.key}
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="flex items-baseline gap-4"
            >
              <span
                className="text-xs tracking-widest shrink-0"
                style={{ color: "rgba(240,240,248,0.2)", fontFamily: "var(--font-geist-mono)" }}
              >
                {f.question}
              </span>
              <span
                className="text-sm"
                style={{
                  color: "rgba(240,240,248,0.5)",
                  fontFamily: "'Inter', ui-sans-serif, sans-serif",
                }}
              >
                {answers[f.key]}
              </span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Current question + input */}
      <div className="w-full max-w-lg">
        <AnimatePresence mode="wait">
          <motion.div
            key={`field-${currentField}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.45, ease: "easeOut" }}
            className="space-y-6"
          >
            {/* Question */}
            <h2
              className="text-2xl font-light"
              style={{
                fontFamily: "'Inter', ui-sans-serif, sans-serif",
                color: "rgba(240,240,248,0.85)",
                letterSpacing: "-0.01em",
              }}
            >
              {field?.question}
            </h2>

            {/* Input */}
            <div
              className="relative"
              style={{ borderBottom: "1px solid rgba(240,240,248,0.1)" }}
            >
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={field?.placeholder}
                disabled={isSubmitting || isProcessing}
                className="w-full bg-transparent py-3 text-lg"
                style={{
                  fontFamily: "'Inter', ui-sans-serif, sans-serif",
                  color: "rgba(240,240,248,0.9)",
                  fontSize: "18px",
                  fontWeight: 300,
                  border: "none",
                  outline: "none",
                }}
              />
              {/* Animated underline on focus */}
              <motion.div
                layoutId="field-underline"
                className="absolute bottom-0 left-0 h-[1px]"
                style={{ background: "rgba(99,102,241,0.6)" }}
                animate={{ width: inputValue ? "100%" : "0%" }}
                transition={{ duration: 0.3 }}
              />
            </div>

            {/* Enter hint */}
            <p
              className="text-xs tracking-widest"
              style={{
                color: "rgba(240,240,248,0.15)",
                fontFamily: "var(--font-geist-mono)",
              }}
            >
              {isSubmitting || isProcessing
                ? "Connecting to Drishti..."
                : currentField < FIELDS.length - 1
                ? "Press ENTER to continue"
                : "Press ENTER to begin"}
            </p>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Progress dots */}
      <div className="absolute bottom-12 flex items-center gap-2">
        {FIELDS.map((_, idx) => (
          <motion.div
            key={idx}
            animate={{
              width: idx === currentField ? 20 : 6,
              opacity: idx <= currentField ? 1 : 0.2,
              background:
                idx < currentField
                  ? "rgba(99,102,241,0.6)"
                  : idx === currentField
                  ? "rgba(99,102,241,0.9)"
                  : "rgba(240,240,248,0.2)",
            }}
            transition={{ duration: 0.3 }}
            style={{ height: 6, borderRadius: 3 }}
          />
        ))}
      </div>
    </div>
  );
}
