"""
Generative Agent Simulation Engine — PII Sanitizer
===================================================

Provides differential-privacy utilities for scrubbing Personally Identifiable
Information (PII) from raw interview transcripts before storage. Named entities
are replaced with indexed placeholder tokens (e.g. ``[City_1]``, ``[Employer_1]``)
to preserve structural integrity while protecting subject identity.

.. warning::
    This is a **Phase 1 shell implementation** using regex heuristics.
    Phase 2 will integrate NER-based entity recognition for higher recall.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Pattern Registry
# ---------------------------------------------------------------------------

# Each entry: (compiled regex, category label)
# The category label is used to generate tokens like [Category_N].
_PII_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    # Email addresses
    (
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
        ),
        "Email",
    ),
    # Phone numbers (various formats)
    (
        re.compile(
            r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        ),
        "Phone",
    ),
    # Social Security Numbers (US format)
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "SSN",
    ),
    # URLs
    (
        re.compile(
            r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
        ),
        "URL",
    ),
    # Common proper noun patterns following contextual keywords
    # "at <Company>", "for <Company>", "with <Company>"
    (
        re.compile(
            r"(?:at|for|with|from|joined)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}(?:\s+(?:Inc|LLC|Corp|Ltd|Co|Group|Technologies|Solutions|Partners|Consulting|University|College|Institute)\.?))",
            re.MULTILINE,
        ),
        "Employer",
    ),
    # "in <City>" / "to <City>" / "from <City>" patterns
    (
        re.compile(
            r"(?:in|to|from|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}(?:,\s*[A-Z]{2})?)\b",
            re.MULTILINE,
        ),
        "City",
    ),
    # Standalone capitalized names (first + last) — loose heuristic
    (
        re.compile(
            r"\b([A-Z][a-z]{1,15}\s+[A-Z][a-z]{1,20})\b"
        ),
        "Person",
    ),
]


def sanitize_transcript(text: str) -> str:
    """Scrub PII from raw transcript text using regex-based heuristics.

    Detected entities are replaced with indexed placeholder tokens to
    maintain referential consistency within the document. For example,
    the first detected city becomes ``[City_1]``, the second ``[City_2]``,
    and repeated occurrences of the same entity reuse the same token.

    Parameters
    ----------
    text : str
        The raw transcript text to sanitize.

    Returns
    -------
    str
        The sanitized transcript with PII replaced by placeholder tokens.

    Examples
    --------
    >>> sanitize_transcript("I worked at Google Inc in San Francisco.")
    'I worked at [Employer_1] in [City_1].'
    """
    if not text:
        return text

    # Track already-seen entity values to reuse the same placeholder index.
    entity_registry: Dict[str, Dict[str, str]] = {}

    def _get_placeholder(category: str, value: str) -> str:
        """Return a consistent placeholder token for a given entity."""
        if category not in entity_registry:
            entity_registry[category] = {}

        category_map = entity_registry[category]
        if value not in category_map:
            index = len(category_map) + 1
            category_map[value] = f"[{category}_{index}]"

        return category_map[value]

    sanitized: str = text

    for pattern, category in _PII_PATTERNS:
        matches = list(pattern.finditer(sanitized))

        # Process matches in reverse order to preserve character offsets.
        for match in reversed(matches):
            # Some patterns capture a group; others match the full span.
            if match.lastindex and match.lastindex >= 1:
                # Replace only the captured group, keeping surrounding text.
                group_start = match.start(1)
                group_end = match.end(1)
                entity_value = match.group(1)
                placeholder = _get_placeholder(category, entity_value)
                sanitized = (
                    sanitized[:group_start]
                    + placeholder
                    + sanitized[group_end:]
                )
            else:
                entity_value = match.group(0)
                placeholder = _get_placeholder(category, entity_value)
                sanitized = (
                    sanitized[: match.start()]
                    + placeholder
                    + sanitized[match.end() :]
                )

    return sanitized
