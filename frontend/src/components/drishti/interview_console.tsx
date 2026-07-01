"use client";

/**
 * Drishti Interview Console
 * ==========================
 *
 * The conversational heart of Drishti.
 * Left: Chat with Drishti — warm, conversational, typewriter effect.
 * Right: Living persona panel ("What I'm Learning About You").
 *
 * Design:
 * - Drishti messages: soft indigo/slate
 * - User messages: warm, labeled with their name
 * - "Drishti is thinking..." = breathing orb, not loading dots
 * - No turn counter, no military labels, no clinical UI
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useDrishti, type ChatMessage } from "@/context/SimulationContext";
import { PersonaPanel } from "./persona_panel";
import { DebugOverlay } from "./debug_overlay";

// ---------------------------------------------------------------------------
// Typewriter Hook
// ---------------------------------------------------------------------------

function useTypewriter(text: string, speed: number = 18): string {
  const [displayed, setDisplayed] = useState("");
  const indexRef = useRef(0);

  useEffect(() => {
    setDisplayed("");
    indexRef.current = 0;
    if (!text) return;

    const timer = setInterval(() => {
      indexRef.current += 1;
      setDisplayed(text.slice(0, indexRef.current));
      if (indexRef.current >= text.length) clearInterval(timer);
    }, speed);

    return () => clearInterval(timer);
  }, [text, speed]);

  return displayed;
}

// ---------------------------------------------------------------------------
// Message Bubble
// ---------------------------------------------------------------------------

function MessageBubble({
  message,
  isLatest,
  userName,
}: {
  message: ChatMessage;
  isLatest: boolean;
  userName: string;
}) {
  const isUser = message.role === "user";
  const typedContent = useTypewriter(
    isLatest && !isUser ? message.content : "",
    18
  );
  const displayContent = isLatest && !isUser ? typedContent : message.content;
  const stillTyping = isLatest && !isUser && typedContent.length < message.content.length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="flex items-start gap-4 py-4"
      style={{ borderBottom: "1px solid rgba(240,240,248,0.04)" }}
    >
      {/* Role label */}
      <span
        className="text-[10px] tracking-widest shrink-0 mt-0.5 w-16 text-right"
        style={{
          fontFamily: "var(--font-geist-mono)",
          color: isUser ? "rgba(245,158,11,0.5)" : "rgba(99,102,241,0.5)",
        }}
      >
        {isUser ? (userName || "YOU") : "DRISHTI"}
      </span>

      {/* Content */}
      <div
        className="text-base flex-1 leading-relaxed"
        style={{
          fontFamily: "'Inter', ui-sans-serif, sans-serif",
          fontWeight: 300,
          color: isUser
            ? "rgba(245,158,11,0.85)"
            : "rgba(240,240,248,0.8)",
        }}
      >
        {displayContent}
        {stillTyping && (
          <span
            className="inline-block w-0.5 h-4 ml-0.5 align-middle animate-cursor-blink"
            style={{ background: "rgba(99,102,241,0.7)" }}
          />
        )}
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Thinking Indicator
// ---------------------------------------------------------------------------

function ThinkingIndicator() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex items-start gap-4 py-4"
    >
      <span
        className="text-[10px] tracking-widest shrink-0 mt-0.5 w-16 text-right"
        style={{
          fontFamily: "var(--font-geist-mono)",
          color: "rgba(99,102,241,0.5)",
        }}
      >
        DRISHTI
      </span>
      <div className="flex items-center gap-2 mt-1">
        <motion.div
          animate={{ opacity: [0.2, 0.7, 0.2], scale: [0.8, 1.0, 0.8] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
          className="w-1.5 h-1.5 rounded-full"
          style={{ background: "rgba(99,102,241,0.6)" }}
        />
        <span
          className="text-sm font-light"
          style={{
            fontFamily: "'Inter', ui-sans-serif, sans-serif",
            color: "rgba(240,240,248,0.2)",
          }}
        >
          listening...
        </span>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Interview Console
// ---------------------------------------------------------------------------

export function InterviewConsole() {
  const {
    identity,
    messages,
    isProcessing,
    errorMessage,
    sendMessage,
  } = useDrishti();

  const [inputValue, setInputValue] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const userName = (identity.name || "YOU").toUpperCase().slice(0, 8);

  // Auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isProcessing]);

  // Auto-focus
  useEffect(() => {
    if (!isProcessing) {
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  }, [isProcessing]);

  const handleSubmit = useCallback(async () => {
    const trimmed = inputValue.trim();
    if (!trimmed || isProcessing) return;
    setInputValue("");
    await sendMessage(trimmed);
  }, [inputValue, isProcessing, sendMessage]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  return (
    <div className="h-screen w-full flex" style={{ background: "#080810" }}>

      {/* ── Chat Area ───────────────────────────────────────────── */}
      <div
        className="flex-1 flex flex-col"
        style={{ borderRight: "1px solid rgba(240,240,248,0.05)" }}
      >
        {/* Top bar */}
        <div
          className="h-14 flex items-center justify-between px-8 shrink-0"
          style={{ borderBottom: "1px solid rgba(240,240,248,0.05)" }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: "rgba(99,102,241,0.8)", boxShadow: "0 0 6px rgba(99,102,241,0.5)" }}
            />
            <span
              className="text-sm font-light"
              style={{
                fontFamily: "'Inter', ui-sans-serif, sans-serif",
                color: "rgba(240,240,248,0.4)",
                letterSpacing: "0.02em",
              }}
            >
              Drishti
            </span>
          </div>
          {isProcessing && (
            <span
              className="text-[10px] tracking-widest"
              style={{ color: "rgba(99,102,241,0.4)", fontFamily: "var(--font-geist-mono)" }}
            >
              THINKING
            </span>
          )}
        </div>

        {/* Messages */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-8 py-2"
        >
          {messages.map((msg, idx) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              isLatest={idx === messages.length - 1}
              userName={userName}
            />
          ))}

          <AnimatePresence>
            {isProcessing && <ThinkingIndicator />}
          </AnimatePresence>
        </div>

        {/* Input area */}
        <div
          className="shrink-0"
          style={{ borderTop: "1px solid rgba(240,240,248,0.05)" }}
        >
          {errorMessage && (
            <div
              className="px-8 py-2 text-xs"
              style={{
                color: "rgba(248,113,113,0.8)",
                background: "rgba(248,113,113,0.04)",
                borderBottom: "1px solid rgba(248,113,113,0.1)",
                fontFamily: "var(--font-geist-mono)",
              }}
            >
              ⚠ {errorMessage}
            </div>
          )}
          <div className="flex items-end gap-4 px-8 py-4">
            <div
              className="w-4 h-4 rounded-full shrink-0 mb-1"
              style={{
                background: isProcessing
                  ? "rgba(240,240,248,0.1)"
                  : "rgba(245,158,11,0.5)",
                transition: "background 0.3s",
              }}
            />
            <textarea
              ref={inputRef}
              rows={1}
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                // Auto-resize
                e.target.style.height = "auto";
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
              }}
              onKeyDown={handleKeyDown}
              disabled={isProcessing}
              placeholder={isProcessing ? "" : "Speak freely..."}
              className="flex-1 bg-transparent resize-none leading-relaxed"
              style={{
                fontFamily: "'Inter', ui-sans-serif, sans-serif",
                fontSize: "15px",
                fontWeight: 300,
                color: "rgba(240,240,248,0.85)",
                border: "none",
                outline: "none",
                opacity: isProcessing ? 0.3 : 1,
                transition: "opacity 0.3s",
                overflow: "hidden",
              }}
            />
            <button
              onClick={handleSubmit}
              disabled={isProcessing || !inputValue.trim()}
              className="shrink-0 mb-1 text-xs tracking-widest transition-all duration-200"
              style={{
                fontFamily: "var(--font-geist-mono)",
                color:
                  isProcessing || !inputValue.trim()
                    ? "rgba(240,240,248,0.1)"
                    : "rgba(99,102,241,0.6)",
                background: "transparent",
                border: "none",
                cursor:
                  isProcessing || !inputValue.trim() ? "not-allowed" : "pointer",
              }}
            >
              SEND
            </button>
          </div>
        </div>
      </div>

      {/* ── Persona Panel ───────────────────────────────────────── */}
      <div
        className="w-72 shrink-0"
        style={{ background: "rgba(15,15,26,0.6)" }}
      >
        <PersonaPanel />
      </div>

      <DebugOverlay />
    </div>
  );
}
