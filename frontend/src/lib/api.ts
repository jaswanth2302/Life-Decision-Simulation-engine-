/**
 * Drishti — API Client
 * =====================
 *
 * TypeScript SDK for the Drishti FastAPI backend.
 * All interfaces mirror the Pydantic v2 schemas exactly.
 */

// ---------------------------------------------------------------------------
// Response Interfaces
// ---------------------------------------------------------------------------

export interface PersonaLabel {
  label: string;
  stars: number;
  confidence: number;
  evidence_count: number;
  source_insight_ids: string[];
  generated_by_prompt_version?: string;
}

export interface Insight {
  text: string;
  lifecycle_stage: "candidate" | "supported" | "strong" | "shifting" | "retired";
  confidence: number;
  evidence_count: number;
  source_memory_ids: string[];
  contradiction_memory_ids?: string[];
  is_evolving: boolean;
}

export interface SummarySentence {
  text: string;
  supporting_insight_ids: string[];
  source_memory_ids: string[];
  is_reframe: boolean;
  generated_by_prompt_version?: string;
}

export interface IdentityPayload {
  name: string;
  age: string;
  country: string;
  occupation: string;
  timezone: string;
}

export interface InterviewInitResponse {
  thread_id: string;
  agent_id: string;
  next_question: string;
  remaining_turns: number;
  evaluation_scores: Record<string, number>;
  persona_labels: PersonaLabel[];
  insights: Insight[];
}

export interface InterviewSubmitResponse {
  next_question: string;
  remaining_turns: number;
  evaluation_scores: Record<string, number>;
  is_complete: boolean;
  persona_labels: PersonaLabel[];
  new_insights: Insight[];
  summary_sentences: SummarySentence[];
  debug_state?: any;
}

export interface AgentStatusResponse {
  agent_id: string;
  evaluation_scores: Record<string, number>;
  recent_memories: string[];
  persona_labels: PersonaLabel[];
  insights: Insight[];
}

export interface FeedbackResponse {
  acknowledged: boolean;
  message: string;
  updated_insight_id: string | null;
}

export interface ApiError {
  error: string;
  message: string;
  details: string;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  "https://web-production-6c97a.up.railway.app";

class DrishtiApiClient {
  private readonly baseUrl: string;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...options.headers },
      ...options,
    });

    if (!response.ok) {
      let errorPayload: ApiError;
      try {
        errorPayload = await response.json();
      } catch {
        errorPayload = {
          error: `HTTP ${response.status}`,
          message: response.statusText,
          details: "",
        };
      }
      throw new Error(`[DRISHTI_API] ${errorPayload.error}: ${errorPayload.message}`);
    }

    return response.json() as Promise<T>;
  }

  /** Initialize a Drishti session with full identity from Stage 1. */
  async initializeInterview(
    agentId: string,
    identity: IdentityPayload
  ): Promise<InterviewInitResponse> {
    return this.request<InterviewInitResponse>("/api/interview/initialize", {
      method: "POST",
      body: JSON.stringify({ agent_id: agentId, identity }),
    });
  }

  /** Submit a user response and advance the state machine. */
  async submitResponse(
    threadId: string,
    agentId: string,
    text: string
  ): Promise<InterviewSubmitResponse> {
    return this.request<InterviewSubmitResponse>("/api/interview/submit", {
      method: "POST",
      body: JSON.stringify({
        thread_id: threadId,
        agent_id: agentId,
        user_response: text,
      }),
    });
  }

  /** Fetch current agent status (for polling). */
  async getAgentStatus(agentId: string): Promise<AgentStatusResponse> {
    return this.request<AgentStatusResponse>(`/api/agent/${agentId}/status`);
  }

  /** Submit feedback on a summary sentence. */
  async submitFeedback(
    agentId: string,
    threadId: string,
    sentenceIndex: number,
    sentiment: "up" | "down" | "edit",
    editText?: string,
    surpriseIndex?: number
  ): Promise<FeedbackResponse> {
    return this.request<FeedbackResponse>("/api/feedback/submit", {
      method: "POST",
      body: JSON.stringify({
        agent_id: agentId,
        thread_id: threadId,
        sentence_index: sentenceIndex,
        sentiment,
        edit_text: editText || null,
        surprise_index: surpriseIndex ?? null,
      }),
    });
  }
}

export const drishtiApi = new DrishtiApiClient();

// Legacy export for compatibility
export const engineApi = drishtiApi;
