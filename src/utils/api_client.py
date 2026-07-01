"""
Generative Agent Simulation Engine — Resilient LLM Client
==========================================================

Provides an asynchronous, fault-tolerant wrapper around the Groq SDK for
structured expert analysis extraction, and the Google GenAI SDK for
embedding generation. The client enforces type-safe responses via JSON mode
with Pydantic schema injection, retries transient failures with exponential
backoff + jitter, and performs automated repair prompts when the LLM output
fails Pydantic validation.

Key Capabilities
----------------
- **Structured Output Enforcement**: Uses Groq JSON mode with the target
  Pydantic schema injected into the system prompt for structured responses.
- **Exponential Backoff with Jitter**: Handles 429 Rate Limit and 5xx Server
  Error codes gracefully.
- **Self-Healing Repair Prompt**: On Pydantic parse failure, issues a one-time
  repair prompt containing the raw validation error, asking the LLM to
  correct its output structure.
- **Embedding Generation**: Uses Google GenAI SDK for 768-dim vector embeddings.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from typing import Any, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from groq import AsyncGroq, APIStatusError as GroqAPIStatusError
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Generic type variable for Pydantic model targets
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class LLMClientError(Exception):
    """Raised when the LLM client exhausts all retry and repair attempts."""


class LLMRateLimitError(LLMClientError):
    """Raised specifically when rate limiting persists after all retries."""


class LLMServerError(LLMClientError):
    """Raised when the server returns 5xx errors after all retries."""


class LLMTruncationError(LLMClientError):
    """Raised when the LLM response is truncated due to token length limit."""


class EmbeddingGenerationError(LLMClientError):
    """Raised when Google AI Studio embedding generation fails or dimension is incorrect."""


# ---------------------------------------------------------------------------
# Resilient LLM Client
# ---------------------------------------------------------------------------


class ResilientLLMClient:
    """Asynchronous, fault-tolerant LLM client with structured output enforcement.

    Parameters
    ----------
    gemini_api_key : str, optional
        Explicit Google API key for embeddings; falls back to
        ``GOOGLE_AI_API_KEY`` or ``GEMINI_API_KEY`` environment variables.
    groq_api_key : str, optional
        Explicit Groq API key for text generation; falls back to
        ``GROQ_API_KEY`` environment variable.
    model : str
        The Groq model identifier for completions. Defaults to
        ``"llama-3.3-70b-versatile"``.
    base_delay : float
        Base delay in seconds for exponential backoff. Defaults to ``4.0``.
    max_retries : int
        Maximum number of retry attempts for transient errors. Defaults to ``5``.
    min_call_interval : float
        Minimum seconds between consecutive API calls. Prevents burst-firing
        through Groq free tier RPM limits. Defaults to ``2.0``.
    temperature : float
        Sampling temperature for LLM completions. Defaults to ``0.3`` for
        deterministic expert analysis.
    max_repair_attempts : int
        Maximum number of self-healing repair attempts for validation failures.
        Must be between 1 and 2. Defaults to ``1``.
    """

    def __init__(
        self,
        *,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        base_delay: float = 4.0,
        max_retries: int = 5,
        min_call_interval: float = 2.0,
        temperature: float = 0.3,
        max_repair_attempts: int = 1,
    ) -> None:
        if not (1 <= max_repair_attempts <= 2):
            raise ValueError("max_repair_attempts must be between 1 and 2")
        self._max_repair_attempts: int = max_repair_attempts

        # --- Groq client for text generation ---
        self._groq_api_key = (
            groq_api_key or os.environ.get("GROQ_API_KEY")
        )
        if not self._groq_api_key:
            logger.warning("Groq API Key not found. Falling back to dummy key for testing/initialization.")
            self._groq_api_key = "gsk_dummy_key_for_testing"

        self._groq_client: AsyncGroq = AsyncGroq(api_key=self._groq_api_key)

        # --- Google GenAI client for embeddings only ---
        self._gemini_api_key: Optional[str] = (
            gemini_api_key
            or os.environ.get("GOOGLE_AI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not self._gemini_api_key:
            logger.warning("Google AI API Key not found. Embeddings will be unavailable.")

        self._gemini_client: genai.Client = genai.Client(api_key=self._gemini_api_key)
        self._model: str = model
        self._base_delay: float = base_delay
        self._max_retries: int = max_retries
        self._min_call_interval: float = min_call_interval
        self._last_call_time: float = 0.0  # Monotonic timestamp of last API call
        self._temperature: float = temperature

        logger.info(
            "ResilientLLMClient initialized | model=%s | max_retries=%d | base_delay=%.1fs | min_call_interval=%.1fs | max_repair_attempts=%d | groq_key=%s | gemini_key=%s",
            self._model,
            self._max_retries,
            self._base_delay,
            self._min_call_interval,
            self._max_repair_attempts,
            self._groq_api_key is not None,
            self._gemini_api_key is not None,
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    async def query_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        target_model: Type[T],
        persona_label: str = "expert",
    ) -> T:
        """Send a structured query and return a validated Pydantic model instance.

        Parameters
        ----------
        system_prompt : str
            The system-level instruction defining the expert persona.
        user_prompt : str
            The user-level prompt containing the transcript to analyze.
        target_model : Type[T]
            The Pydantic model class that the response must conform to.
        persona_label : str
            Human-readable label for logging (e.g. ``"psychologist"``).

        Returns
        -------
        T
            A validated instance of ``target_model``.

        Raises
        ------
        LLMClientError
            If all retry and repair attempts are exhausted.
        """
        logger.info(
            "Starting structured query | persona=%s | target=%s",
            persona_label,
            target_model.__name__,
        )

        raw_content: str = await self._call_with_retries(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            target_model=target_model,
            persona_label=persona_label,
        )

        # --- Attempt primary parse ---
        try:
            parsed: T = target_model.model_validate_json(raw_content)
            logger.info(
                "Primary parse succeeded | persona=%s | target=%s",
                persona_label,
                target_model.__name__,
            )
            return parsed
        except ValidationError as parse_err:
            logger.warning(
                "Primary parse failed for %s — initiating repair sequence | error=%s",
                persona_label,
                str(parse_err)[:500],
            )
            return await self._attempt_repair_loop(
                original_content=raw_content,
                parse_error=parse_err,
                system_prompt=system_prompt,
                target_model=target_model,
                persona_label=persona_label,
            )

    # -------------------------------------------------------------------
    # Internal: Retry Logic with Exponential Backoff + Jitter
    # -------------------------------------------------------------------

    async def _call_with_retries(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        target_model: Type[T],
        persona_label: str,
    ) -> str:
        """Execute the Groq API call with exponential backoff retry logic.

        Returns the raw content string from the LLM response.
        """
        last_exception: Optional[Exception] = None

        # Inject the Pydantic schema into the system prompt for structured output
        schema_json: str = json.dumps(target_model.model_json_schema(), indent=2)
        enhanced_system_prompt: str = (
            f"{system_prompt}\n\n"
            f"You MUST respond with valid JSON matching this exact schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Return ONLY the JSON object. No additional text or commentary."
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                # Enforce minimum interval between calls to avoid bursting
                now = time.monotonic()
                elapsed = now - self._last_call_time
                if elapsed < self._min_call_interval:
                    wait = self._min_call_interval - elapsed
                    logger.debug(
                        "Rate pacing | waiting %.1fs before next call | persona=%s",
                        wait, persona_label,
                    )
                    await asyncio.sleep(wait)
                self._last_call_time = time.monotonic()

                response = await self._groq_client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                )

                content: str | None = response.choices[0].message.content
                if not content:
                    raise LLMClientError(
                        f"Empty response content from LLM for persona={persona_label}"
                    )

                logger.info(
                    "API call succeeded | persona=%s | attempt=%d/%d",
                    persona_label,
                    attempt,
                    self._max_retries,
                )
                return content

            except GroqAPIStatusError as api_err:
                status = api_err.status_code
                if status == 429:
                    last_exception = api_err
                    delay = self._compute_backoff_delay(attempt)
                    logger.warning(
                        "Rate limit hit (429) | persona=%s | attempt=%d/%d | backing off %.2fs",
                        persona_label, attempt, self._max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                elif 500 <= status < 600:
                    last_exception = api_err
                    delay = self._compute_backoff_delay(attempt)
                    logger.warning(
                        "Server error (%d) | persona=%s | attempt=%d/%d | backing off %.2fs",
                        status, persona_label, attempt, self._max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Non-retryable API error (%d) | persona=%s | error=%s",
                        status, persona_label, str(api_err),
                    )
                    raise LLMClientError(
                        f"Non-retryable API error {status} for persona={persona_label}: {api_err}"
                    ) from api_err

        # All retries exhausted
        error_msg = f"All {self._max_retries} retries exhausted for persona={persona_label}"
        logger.error(error_msg)

        if isinstance(last_exception, GroqAPIStatusError) and last_exception.status_code == 429:
            raise LLMRateLimitError(error_msg) from last_exception
        elif isinstance(last_exception, GroqAPIStatusError) and 500 <= last_exception.status_code < 600:
            raise LLMServerError(error_msg) from last_exception
        else:
            raise LLMClientError(error_msg) from last_exception

    # -------------------------------------------------------------------
    # Internal: Self-Healing Repair Loop
    # -------------------------------------------------------------------

    async def _attempt_repair_loop(
        self,
        *,
        original_content: str,
        parse_error: ValidationError,
        system_prompt: str,
        target_model: Type[T],
        persona_label: str,
    ) -> T:
        """Issue self-healing repair prompts up to max_repair_attempts ceiling.

        Sends the malformed output and the parsing error back to the LLM,
        requesting a corrected JSON payload.
        """
        current_content: str = original_content
        current_error: ValidationError = parse_error

        for repair_attempt in range(1, self._max_repair_attempts + 1):
            logger.info(
                "Issuing repair prompt | persona=%s | attempt=%d/%d",
                persona_label,
                repair_attempt,
                self._max_repair_attempts,
            )

            error_details: str = str(current_error)[:2000]
            schema_json: str = json.dumps(target_model.model_json_schema(), indent=2)

            repair_prompt: str = (
                "Your previous response failed structural validation. "
                "Please fix the JSON output to match the required schema exactly.\n\n"
                f"--- VALIDATION ERROR ---\n{error_details}\n\n"
                f"--- REQUIRED SCHEMA ---\n{schema_json}\n\n"
                f"--- YOUR ORIGINAL OUTPUT ---\n{current_content[:3000]}\n\n"
                "Return ONLY the corrected JSON with no additional commentary."
            )

            try:
                repair_response = await self._groq_client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": repair_prompt},
                    ],
                    temperature=0.1,  # Lower temperature for repair precision
                    response_format={"type": "json_object"},
                )

                repair_content: str | None = repair_response.choices[0].message.content
                if not repair_content:
                    raise LLMClientError(f"Empty repair response for persona={persona_label}")

                try:
                    repaired: T = target_model.model_validate_json(repair_content)
                    logger.info(
                        "Repair prompt succeeded | persona=%s | attempt=%d/%d",
                        persona_label,
                        repair_attempt,
                        self._max_repair_attempts,
                    )
                    return repaired
                except ValidationError as next_err:
                    current_content = repair_content
                    current_error = next_err
                    logger.warning(
                        "Repair attempt %d failed to validate for %s | error=%s",
                        repair_attempt,
                        persona_label,
                        str(next_err)[:500],
                    )

            except (LLMClientError, GroqAPIStatusError) as api_err:
                error_msg = f"API error during repair attempt {repair_attempt} for persona={persona_label}: {api_err}"
                logger.error(error_msg)
                raise LLMClientError(error_msg) from api_err

        # All repair attempts exhausted
        final_msg = (
            f"All {self._max_repair_attempts} repair attempts exhausted "
            f"for persona={persona_label}. Final error: {current_error}"
        )
        logger.error(final_msg)
        raise LLMClientError(final_msg) from current_error

    # -------------------------------------------------------------------
    # Internal: Backoff Computation
    # -------------------------------------------------------------------

    def _compute_backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay with jitter.

        Formula: ``base_delay * 2^(attempt-1) + uniform_jitter(0, 1)``
        """
        exponential: float = self._base_delay * (2 ** (attempt - 1))
        jitter: float = random.uniform(0.0, 1.0)  # noqa: S311
        return exponential + jitter

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def get_embedding(self, text: str) -> List[float]:
        """Generate a 768-dimension vector embedding using the Google GenAI SDK.

        Uses ``gemini-embedding-001`` (the current default embedding model)
        via the official ``google.genai`` async client.
        """
        if not self._gemini_api_key:
            raise EmbeddingGenerationError("Gemini/Google API key is not configured.")

        attempt = 1
        current_error: Optional[Exception] = None

        while attempt <= self._max_retries:
            try:
                response = await self._gemini_client.aio.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )

                embedding = response.embeddings[0].values

                if not isinstance(embedding, list) or len(embedding) == 0:
                    raise EmbeddingGenerationError(
                        f"Embedding dimension mismatch. Expected non-empty float list, "
                        f"got {len(embedding) if isinstance(embedding, list) else type(embedding)}"
                    )

                logger.info(
                    "Embedding generated successfully | dim=%d | attempt=%d/%d",
                    len(embedding),
                    attempt,
                    self._max_retries,
                )
                return embedding

            except (APIError, EmbeddingGenerationError) as e:
                current_error = e
                delay = self._compute_backoff_delay(attempt)
                logger.warning(
                    "EMBEDDING RETRY | attempt=%d/%d | error=%s | backing off for %.2fs",
                    attempt,
                    self._max_retries,
                    str(e),
                    delay,
                )
                await asyncio.sleep(delay)
                attempt += 1

        final_msg = f"Failed to generate embedding after {self._max_retries} attempts."
        logger.error(final_msg)
        raise EmbeddingGenerationError(final_msg) from current_error

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    async def close(self) -> None:
        """Gracefully close the underlying HTTP clients."""
        res = self._groq_client.close()
        if asyncio.iscoroutine(res) or hasattr(res, "__await__"):
            await res
        logger.info("ResilientLLMClient closed")
