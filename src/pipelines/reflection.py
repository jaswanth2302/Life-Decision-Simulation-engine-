"""
Generative Agent Simulation Engine — Parallel Reflection Pipeline
=================================================================

Orchestrates the concurrent execution of four specialized social-science
expert personas against a sanitized interview transcript. Each persona
analyzes the transcript through its disciplinary lens and produces a
strongly-typed sub-matrix that slots into the immutable ``AgentProfile``.

Pipeline Flow
-------------
1. Parse sanitized transcript → ``List[TranscriptTurn]`` + compute word count.
2. Initialize fresh ``AgentMetadata`` with tracking timestamps.
3. Fire all four expert queries concurrently via ``asyncio.gather``.
4. Gracefully handle per-persona failures (log + leave matrix as ``None``).
5. Assemble and return the frozen ``AgentProfile`` aggregate root.

Expert Personas
---------------
- **Virtual Psychologist** → ``PsychologicalProfile``
- **Behavioral Economist** → ``EconomicDecisionMatrix``
- **Political Scientist** → ``SocioPoliticalWorldview``
- **Demographer / Sociologist** → ``DemographicBaseline``
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, List, NamedTuple, Optional, Type

from pydantic import BaseModel

from config.logging_config import configure_logging, get_logger
from src.core.schema import (
    AgentMetadata,
    AgentProfile,
    DemographicBaseline,
    EconomicDecisionMatrix,
    PsychologicalProfile,
    SocioPoliticalWorldview,
    TranscriptTurn,
)
from src.utils.api_client import LLMClientError, ResilientLLMClient

# Initialize structured logging
configure_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Expert Persona Definitions
# ---------------------------------------------------------------------------


class ExpertPersonaSpec(NamedTuple):
    """Specification for a single expert persona in the reflection pipeline."""

    label: str
    system_prompt: str
    target_model: Type[BaseModel]
    profile_field: str


EXPERT_PERSONAS: List[ExpertPersonaSpec] = [
    ExpertPersonaSpec(
        label="psychologist",
        system_prompt=(
            "You are a Virtual Psychologist specializing in personality "
            "profiling and cognitive-behavioral analysis. Your task is to "
            "analyze interview transcripts and extract a structured "
            "psychological profile.\n\n"
            "From the transcript provided, identify and output:\n"
            "- core_values: An ordered list of the subject's fundamental "
            "values that drive their decisions and worldview.\n"
            "- coping_mechanisms: Habitual strategies the subject uses to "
            "manage stress, uncertainty, and adversity.\n"
            "- cognitive_anchors: Key mental reference points and frameworks "
            "that anchor the subject's reasoning patterns.\n"
            "- emotional_stability_index: A float from 0.0 (highly volatile) "
            "to 1.0 (exceptionally stable), reflecting the subject's "
            "emotional regulation capacity observed in the transcript.\n\n"
            "Base your analysis exclusively on evidence present in the "
            "transcript. Be precise and avoid speculation beyond what the "
            "text supports. Output valid JSON matching the required schema."
        ),
        target_model=PsychologicalProfile,
        profile_field="psychologist_matrix",
    ),
    ExpertPersonaSpec(
        label="economist",
        system_prompt=(
            "You are a Behavioral Economist specializing in decision theory "
            "and prospect theory analysis. Your task is to analyze interview "
            "transcripts and extract a structured economic decision matrix.\n\n"
            "From the transcript provided, identify and output:\n"
            "- risk_appetite_score: A float from 0.0 (extreme risk aversion) "
            "to 1.0 (maximum risk seeking), based on the subject's revealed "
            "preferences and decision patterns.\n"
            "- loss_aversion_ratio: The ratio of perceived loss impact vs. "
            "equivalent gain (Kahneman-Tversky baseline ~2.25). Adjust based "
            "on observed decision behavior.\n"
            "- heuristic_biases: A list of identified cognitive biases "
            "affecting the subject's economic and resource decisions "
            "(e.g. 'anchoring', 'sunk cost fallacy', 'status quo bias').\n"
            "- resource_allocation_rules: Behavioral rules governing how "
            "the subject allocates scarce resources (time, money, attention, "
            "career capital).\n\n"
            "Ground every assessment in specific transcript evidence. "
            "Output valid JSON matching the required schema."
        ),
        target_model=EconomicDecisionMatrix,
        profile_field="economist_matrix",
    ),
    ExpertPersonaSpec(
        label="political_scientist",
        system_prompt=(
            "You are a Political Scientist specializing in institutional "
            "analysis and socio-political belief systems. Your task is to "
            "analyze interview transcripts and extract a structured "
            "socio-political worldview profile.\n\n"
            "From the transcript provided, identify and output:\n"
            "- institutional_trust_score: A float from 0.0 (deep distrust) "
            "to 1.0 (complete trust), reflecting the subject's relationship "
            "with societal institutions (government, media, education).\n"
            "- ideological_anchors: Key ideological positions and political "
            "orientations that anchor the subject's worldview.\n"
            "- authority_bias_flag: Boolean indicating whether the subject "
            "exhibits a measurable tendency to defer to authority figures.\n"
            "- conflict_resolution_style: The subject's preferred approach "
            "to resolving disagreements and conflicts (e.g. 'collaborative', "
            "'competitive', 'avoidant', 'accommodating').\n\n"
            "Derive all assessments from explicit transcript evidence. "
            "Output valid JSON matching the required schema."
        ),
        target_model=SocioPoliticalWorldview,
        profile_field="political_matrix",
    ),
    ExpertPersonaSpec(
        label="demographer",
        system_prompt=(
            "You are a Demographer and Sociologist specializing in structural "
            "analysis and intersectional demographics. Your task is to analyze "
            "interview transcripts and extract a structured demographic "
            "baseline profile.\n\n"
            "From the transcript provided, identify and output:\n"
            "- structural_constraints: Systemic or structural factors that "
            "constrain or shape the subject's life trajectory (e.g. "
            "'limited economic mobility', 'geographic isolation').\n"
            "- geographic_context: Anonymized geographic context describing "
            "the subject's environment (region type, urban/rural, climate).\n"
            "- socio_economic_tier: Broad classification of the subject's "
            "socio-economic positioning ('lower', 'middle', 'upper-middle', "
            "'upper').\n"
            "- background_intersection_notes: Free-text notes on "
            "intersecting identity factors relevant to understanding the "
            "subject's structural position (anonymized, no PII).\n\n"
            "Rely strictly on transcript evidence. Avoid assumptions not "
            "supported by the text. Output valid JSON matching the required "
            "schema."
        ),
        target_model=DemographicBaseline,
        profile_field="demographer_matrix",
    ),
]


# ---------------------------------------------------------------------------
# Transcript Parsing
# ---------------------------------------------------------------------------


def _parse_transcript_to_turns(sanitized_transcript: str) -> List[TranscriptTurn]:
    """Parse a sanitized transcript string into structured TranscriptTurn objects.

    This is a lightweight parser that treats the full transcript as a single
    turn from the subject in the ``professional`` domain. In production,
    a more sophisticated parser would segment by speaker markers and
    classify domains via NLP.

    Parameters
    ----------
    sanitized_transcript : str
        The PII-scrubbed interview transcript text.

    Returns
    -------
    List[TranscriptTurn]
        A list containing the parsed transcript turns.
    """
    if not sanitized_transcript.strip():
        return []

    # Phase 2 baseline: wrap the full transcript as a single subject turn.
    # A production parser would segment speaker turns and classify domains.
    return [
        TranscriptTurn(
            speaker="subject",
            text=sanitized_transcript.strip(),
            domain="professional",
            semantic_density_score=0.75,
        )
    ]


# ---------------------------------------------------------------------------
# Expert Query Execution & Concurrency Throttling
# ---------------------------------------------------------------------------

# Global semaphore to throttle concurrent LLM API requests and prevent hitting TPM/RPM ceilings.
CONCURRENT_REQUEST_SEMAPHORE = asyncio.Semaphore(2)


async def _execute_expert_query(
    client: ResilientLLMClient,
    persona: ExpertPersonaSpec,
    transcript_text: str,
) -> BaseModel:
    """Execute a single expert persona query against the LLM under semaphore throttle.

    Parameters
    ----------
    client : ResilientLLMClient
        The resilient LLM client to use for the query.
    persona : ExpertPersonaSpec
        The expert persona specification.
    transcript_text : str
        The sanitized transcript text to analyze.

    Returns
    -------
    BaseModel
        The validated Pydantic model instance from the expert analysis.
    """
    user_prompt: str = (
        "Analyze the following interview transcript and produce your "
        "expert assessment. Output ONLY valid JSON matching the required "
        "schema.\n\n"
        f"--- INTERVIEW TRANSCRIPT ---\n{transcript_text}\n"
        "--- END TRANSCRIPT ---"
    )

    async with CONCURRENT_REQUEST_SEMAPHORE:
        logger.info(
            "Acquired semaphore slot for persona=%s | starting request",
            persona.label,
        )
        return await client.query_structured(
            system_prompt=persona.system_prompt,
            user_prompt=user_prompt,
            target_model=persona.target_model,
            persona_label=persona.label,
        )


# ---------------------------------------------------------------------------
# Main Pipeline Entrypoint
# ---------------------------------------------------------------------------


async def generate_expert_reflection_matrix(
    sanitized_transcript: str,
    *,
    llm_client: Optional[ResilientLLMClient] = None,
    api_key: Optional[str] = None,
) -> AgentProfile:
    """Generate a complete AgentProfile by orchestrating parallel expert analysis.

    This is the primary entrypoint of the Phase 2 reflection pipeline. It
    parses the sanitized transcript, fires all four expert personas
    concurrently (throttled to 2 at a time), and assembles the results
    into an immutable ``AgentProfile``.

    Parameters
    ----------
    sanitized_transcript : str
        The PII-scrubbed interview transcript to analyze.
    llm_client : ResilientLLMClient | None
        An optional pre-configured LLM client. If ``None``, a new client
        is created using the provided ``api_key`` or environment defaults.
    api_key : str | None
        OpenAI API key. Used only when ``llm_client`` is not provided.

    Returns
    -------
    AgentProfile
        The fully assembled agent profile with all available expert matrices.
        Failed expert matrices are set to ``None`` rather than crashing
        the pipeline, and metadata is populated with failure status.
    """
    logger.info("=" * 70)
    logger.info("REFLECTION PIPELINE STARTED")
    logger.info("=" * 70)

    # --- Step 1: Parse transcript ---
    transcript_turns: List[TranscriptTurn] = _parse_transcript_to_turns(
        sanitized_transcript
    )
    word_count: int = sum(len(turn.text.split()) for turn in transcript_turns)

    logger.info(
        "Transcript parsed | turns=%d | word_count=%d",
        len(transcript_turns),
        word_count,
    )

    # --- Step 2: Initialize or use provided LLM client ---
    owns_client: bool = llm_client is None
    client: ResilientLLMClient = llm_client or ResilientLLMClient(
        gemini_api_key=api_key
    )

    try:
        # --- Step 3: Fire all four expert queries concurrently (throttled via Semaphore) ---
        logger.info(
            "Launching %d expert persona queries concurrently (semaphore limit = 2)",
            len(EXPERT_PERSONAS),
        )

        tasks = [
            _execute_expert_query(client, persona, sanitized_transcript)
            for persona in EXPERT_PERSONAS
        ]

        results: list[BaseModel | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # --- Step 4: Process results with graceful degradation and state preservation ---
        expert_matrices: dict[str, Any] = {}
        failed_expert_labels: List[str] = []

        for persona, result in zip(EXPERT_PERSONAS, results):
            if isinstance(result, BaseException):
                logger.critical(
                    "EXPERT FAILURE | persona=%s | error_type=%s | error=%s | "
                    "Matrix will be set to None — pipeline continues with "
                    "degraded accuracy.",
                    persona.label,
                    type(result).__name__,
                    str(result)[:500],
                )
                expert_matrices[persona.profile_field] = None
                failed_expert_labels.append(persona.label)
            else:
                logger.info(
                    "Expert analysis complete | persona=%s | "
                    "matrix=%s populated successfully",
                    persona.label,
                    persona.profile_field,
                )
                expert_matrices[persona.profile_field] = result

        # --- Step 5: Construct metadata with partial state information ---
        is_partial = len(failed_expert_labels) > 0
        metadata = AgentMetadata(
            total_interview_word_count=word_count,
            target_accuracy_score=0.85,
            is_partial_profile=is_partial,
            failed_experts=failed_expert_labels,
        )

        logger.info(
            "Agent metadata instantiated | agent_id=%s | created_at=%s | is_partial=%s | failed_experts=%s",
            metadata.agent_id,
            metadata.created_at.isoformat(),
            is_partial,
            failed_expert_labels,
        )

        # --- Step 6: Assemble the AgentProfile ---
        populated_count: int = sum(
            1 for v in expert_matrices.values() if v is not None
        )
        logger.info(
            "Assembling AgentProfile | populated_matrices=%d/%d",
            populated_count,
            len(EXPERT_PERSONAS),
        )

        profile = AgentProfile(
            metadata=metadata,
            raw_transcript=transcript_turns,
            psychologist_matrix=expert_matrices.get("psychologist_matrix"),
            economist_matrix=expert_matrices.get("economist_matrix"),
            political_matrix=expert_matrices.get("political_matrix"),
            demographer_matrix=expert_matrices.get("demographer_matrix"),
            episodic_memory_stream=[],
        )

        logger.info("=" * 70)
        logger.info(
            "REFLECTION PIPELINE COMPLETE | agent_id=%s | "
            "matrices_populated=%d/%d | accuracy_target=%.2f | is_partial=%s",
            profile.metadata.agent_id,
            populated_count,
            len(EXPERT_PERSONAS),
            profile.metadata.target_accuracy_score,
            profile.metadata.is_partial_profile,
        )
        logger.info("=" * 70)

        return profile

    finally:
        if owns_client:
            await client.close()
