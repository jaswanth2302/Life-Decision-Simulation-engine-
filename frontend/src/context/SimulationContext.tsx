"use client";

/**
 * Drishti — Central Application State
 * =====================================
 *
 * Manages the full session lifecycle:
 * LANDING → IDENTITY → INTERVIEW → PERSONA_REVEAL → FEEDBACK
 *
 * The memory architecture is the product. This context reflects that
 * by treating persona labels, insights, and summary sentences as
 * first-class state — not telemetry side data.
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import { v4 as uuidv4 } from "uuid";
import {
  drishtiApi,
  type InterviewInitResponse,
  type InterviewSubmitResponse,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type AppPhase =
  | "LANDING"
  | "IDENTITY"
  | "INTERVIEW"
  | "PERSONA_REVEAL"
  | "FEEDBACK";

export interface ChatMessage {
  id: string;
  role: "assistant" | "user";
  content: string;
  timestamp: number;
}

export interface PersonaLabel {
  label: string;
  stars: number;        // 1–5
  confidence: number;   // 0–1
  evidence_count: number;
  source_insight_ids: string[];
}

export interface Insight {
  text: string;
  lifecycle_stage: "candidate" | "supported" | "strong" | "shifting" | "retired";
  confidence: number;
  evidence_count: number;
  source_memory_ids: string[];
  is_evolving: boolean;
}

export interface EvidenceItem {
  memory_id: string;
  quote: string;
  interpretation: string;
  weight: number;
}

export interface Argument {
  reasoning: string;
  evidence: EvidenceItem[];
  uncertainty: string;
  update_conditions: string;
}

export interface SummarySentence {
  text: string;
  confidence: number;
  argument: Argument;
  supporting_insight_ids: string[];
  source_memory_ids: string[];
  is_reframe: boolean;
}

export interface IdentityData {
  name: string;
  age: string;
  country: string;
  occupation: string;
  timezone: string;
}

// ---------------------------------------------------------------------------
// Context Shape
// ---------------------------------------------------------------------------

interface DrishtiState {
  phase: AppPhase;
  agentId: string;
  threadId: string;
  identity: IdentityData;
  messages: ChatMessage[];
  personaLabels: PersonaLabel[];
  insights: Insight[];
  summarySentences: SummarySentence[];
  expandedSentenceIdx: number | null;   // which "Why?" is open
  surpriseSentenceIdx: number | null;   // user's answer to "Which surprised you?"
  isProcessing: boolean;
  errorMessage: string;
  debugState: any | null;
}

interface DrishtiActions {
  beginOnboarding: () => void;
  submitIdentity: (data: IdentityData) => Promise<void>;
  sendMessage: (text: string) => Promise<void>;
  toggleExplanation: (idx: number) => void;
  submitSurprise: (idx: number) => void;
  submitFeedback: (sentenceIdx: number, sentiment: "up" | "down" | "edit", editText?: string) => Promise<void>;
  acknowledgeComplete: () => void;
}

type DrishtiContextValue = DrishtiState & DrishtiActions;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

const defaultIdentity: IdentityData = {
  name: "", age: "", country: "", occupation: "", timezone: "",
};

const initialState: DrishtiState = {
  phase: "LANDING",
  agentId: "",
  threadId: "",
  identity: defaultIdentity,
  messages: [],
  personaLabels: [],
  insights: [],
  summarySentences: [],
  expandedSentenceIdx: null,
  surpriseSentenceIdx: null,
  isProcessing: false,
  errorMessage: "",
  debugState: null,
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const DrishtiContext = createContext<DrishtiContextValue | null>(null);

export function useDrishti(): DrishtiContextValue {
  const ctx = useContext(DrishtiContext);
  if (!ctx) throw new Error("useDrishti must be used within DrishtiProvider");
  return ctx;
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function DrishtiProvider({ children }: { children: ReactNode }) {
  const [phase, setPhase] = useState<AppPhase>("LANDING");
  const [agentId, setAgentId] = useState("");
  const [threadId, setThreadId] = useState("");
  const [identity, setIdentity] = useState<IdentityData>(defaultIdentity);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [personaLabels, setPersonaLabels] = useState<PersonaLabel[]>([]);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [summarySentences, setSummarySentences] = useState<SummarySentence[]>([]);
  const [expandedSentenceIdx, setExpandedSentenceIdx] = useState<number | null>(null);
  const [surpriseSentenceIdx, setSurpriseSentenceIdx] = useState<number | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [debugState, setDebugState] = useState<any | null>(null);

  // -------------------------------------------------------------------------
  // Begin Onboarding — transitions LANDING → IDENTITY
  // -------------------------------------------------------------------------
  const beginOnboarding = useCallback(() => {
    setPhase("IDENTITY");
  }, []);

  // Listen for the drishti:begin custom event from LandingPage
  React.useEffect(() => {
    const handler = () => beginOnboarding();
    window.addEventListener("drishti:begin", handler);
    return () => window.removeEventListener("drishti:begin", handler);
  }, [beginOnboarding]);

  // -------------------------------------------------------------------------
  // Submit Identity — transitions IDENTITY → INTERVIEW
  // -------------------------------------------------------------------------
  const submitIdentity = useCallback(async (data: IdentityData) => {
    setIsProcessing(true);
    setErrorMessage("");
    setIdentity(data);

    const newAgentId = uuidv4();
    setAgentId(newAgentId);

    try {
      const res: InterviewInitResponse = await drishtiApi.initializeInterview(
        newAgentId,
        data
      );

      setThreadId(res.thread_id);
      setPersonaLabels((res.persona_labels as PersonaLabel[]) || []);
      setInsights((res.insights as Insight[]) || []);

      const openingMsg: ChatMessage = {
        id: uuidv4(),
        role: "assistant",
        content: res.next_question,
        timestamp: Date.now(),
      };
      setMessages([openingMsg]);
      setPhase("INTERVIEW");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to connect.";
      setErrorMessage(msg);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  // -------------------------------------------------------------------------
  // Send Message — the core interview loop
  // -------------------------------------------------------------------------
  const sendMessage = useCallback(
    async (text: string) => {
      if (!threadId || !agentId || isProcessing) return;
      setIsProcessing(true);
      setErrorMessage("");

      const userMsg: ChatMessage = {
        id: uuidv4(),
        role: "user",
        content: text,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, userMsg]);

      try {
        const res: InterviewSubmitResponse = await drishtiApi.submitResponse(
          threadId,
          agentId,
          text
        );

        if (res.debug_state) {
          setDebugState(res.debug_state);
        }

        // Update living persona panel
        if (res.persona_labels && res.persona_labels.length > 0) {
          setPersonaLabels(res.persona_labels as PersonaLabel[]);
        }

        // Merge new insights
        if (res.new_insights && res.new_insights.length > 0) {
          setInsights((prev) => {
            const existingTexts = new Set(prev.map((i) => i.text));
            const truly_new = (res.new_insights as Insight[]).filter(
              (i) => !existingTexts.has(i.text)
            );
            return [...prev, ...truly_new];
          });
        }

        if (res.is_complete) {
          // Store summary sentences and transition to reveal
          if (res.summary_sentences && res.summary_sentences.length > 0) {
            setSummarySentences(res.summary_sentences as SummarySentence[]);
          }
          if (res.next_question) {
            const closingMsg: ChatMessage = {
              id: uuidv4(),
              role: "assistant",
              content: res.next_question,
              timestamp: Date.now(),
            };
            setMessages((prev) => [...prev, closingMsg]);
          }
          // Brief delay before transitioning for emotional pacing
          setTimeout(() => setPhase("PERSONA_REVEAL"), 2000);
        } else if (res.next_question) {
          const assistantMsg: ChatMessage = {
            id: uuidv4(),
            role: "assistant",
            content: res.next_question,
            timestamp: Date.now(),
          };
          setMessages((prev) => [...prev, assistantMsg]);
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Failed to send.";
        setErrorMessage(msg);
      } finally {
        setIsProcessing(false);
      }
    },
    [threadId, agentId, isProcessing]
  );

  // -------------------------------------------------------------------------
  // Explanation toggle (▼ Why?)
  // -------------------------------------------------------------------------
  const toggleExplanation = useCallback((idx: number) => {
    setExpandedSentenceIdx((prev) => (prev === idx ? null : idx));
  }, []);

  // -------------------------------------------------------------------------
  // Surprise signal
  // -------------------------------------------------------------------------
  const submitSurprise = useCallback(
    (idx: number) => {
      setSurpriseSentenceIdx(idx);
      // Fire and forget — send to backend
      drishtiApi
        .submitFeedback(agentId, threadId, idx, "up", undefined, idx)
        .catch((e) => console.error("Surprise signal failed:", e));
    },
    [agentId, threadId]
  );

  // -------------------------------------------------------------------------
  // Feedback
  // -------------------------------------------------------------------------
  const submitFeedback = useCallback(
    async (
      sentenceIdx: number,
      sentiment: "up" | "down" | "edit",
      editText?: string
    ) => {
      try {
        await drishtiApi.submitFeedback(
          agentId,
          threadId,
          sentenceIdx,
          sentiment,
          editText
        );
        if (sentiment === "down" || sentiment === "edit") {
          setPhase("FEEDBACK");
        }
      } catch (e) {
        console.error("Feedback submission failed:", e);
      }
    },
    [agentId, threadId]
  );

  // -------------------------------------------------------------------------
  // Acknowledge complete (go to FEEDBACK from PERSONA_REVEAL)
  // -------------------------------------------------------------------------
  const acknowledgeComplete = useCallback(() => {
    setPhase("FEEDBACK");
  }, []);

  const value: DrishtiContextValue = {
    phase,
    agentId,
    threadId,
    identity,
    beginOnboarding,
    messages,
    personaLabels,
    insights,
    summarySentences,
    expandedSentenceIdx,
    surpriseSentenceIdx,
    isProcessing,
    errorMessage,
    debugState,
    submitIdentity,
    sendMessage,
    toggleExplanation,
    submitSurprise,
    submitFeedback,
    acknowledgeComplete,
  };

  return (
    <DrishtiContext.Provider value={value}>{children}</DrishtiContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Legacy compatibility export (SimulationContext was used in old page.tsx)
// ---------------------------------------------------------------------------
export const SimulationProvider = DrishtiProvider;
export const useSimulation = useDrishti;
