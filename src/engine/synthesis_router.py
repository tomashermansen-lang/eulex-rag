"""Synthesis mode detection for cross-law queries.

This module determines the appropriate synthesis mode based on query semantics
and corpus scope, enabling the RAG engine to generate appropriately structured
responses for different types of cross-law questions.

Synthesis Modes:
    SINGLE      - Traditional single-corpus query (default)
    AGGREGATION - "What do all laws say about X?"
    COMPARISON  - "Compare AI-Act and GDPR on X"
    UNIFIED     - Best answer regardless of source
    ROUTING     - "Which law covers X?" (identification only)

Detection is based on keyword patterns in the query and the specified corpus scope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .corpus_resolver import CorpusResolver


class SynthesisMode(Enum):
    """Synthesis mode for multi-corpus queries."""

    SINGLE = "single"  # Traditional single-corpus (default)
    AGGREGATION = "aggregation"  # "What do all laws say about X?"
    COMPARISON = "comparison"  # "Compare AI-Act and GDPR on X"
    UNIFIED = "unified"  # Best answer regardless of source
    ROUTING = "routing"  # "Which law covers X?"
    DISCOVERY = "discovery"  # Topic-based corpus identification


@dataclass(frozen=True)
class SynthesisContext:
    """Context for synthesis operation."""

    mode: SynthesisMode
    target_corpora: tuple[str, ...]  # Corpora to query
    comparison_pairs: tuple[tuple[str, str], ...] | None = None  # For COMPARISON mode
    routing_only: bool = False  # True if no deep synthesis needed


# ---------------------------------------------------------------------------
# Detection Patterns
# ---------------------------------------------------------------------------

# COMPARISON patterns - detect comparative analysis intent
COMPARISON_PATTERNS = [
    # English
    r"\bcompare\b.*\band\b",  # "compare X and Y"
    r"\bcomparing\b.*\band\b",  # "comparing X and Y"
    r"\bvs\.?\b",  # "X vs Y" or "X vs. Y"
    r"\bversus\b",  # "X versus Y"
    r"\bdifference(?:s)?\s+between\b",  # "differences between"
    r"\bhow\s+does?\b.*\bcompare\b",  # "how does X compare to Y"
    # Danish
    r"\bforskellen?\s+mellem\b",  # "forskellen mellem"
    r"\bsammenlign\b",  # "sammenlign"
    r"\bsammenlignet\s+med\b",  # "sammenlignet med"
]

# ROUTING patterns - detect law identification intent
ROUTING_PATTERNS = [
    # English
    r"\bwhich\s+law\b",  # "which law"
    r"\bwhat\s+law\s+(?:covers?|governs?|applies|regulates?)\b",  # "what law covers"
    r"\bwhich\s+(?:eu\s+)?regulation\b",  # "which regulation"
    r"\bwhat\s+legislation\b",  # "what legislation"
    r"\bidentify\s+(?:the\s+)?law\b",  # "identify the law"
    # Danish
    r"\bhvilken\s+lov\b",  # "hvilken lov"
    r"\bhvad\s+lov\b",  # "hvad lov" (informal)
    r"\bhvilken\s+forordning\b",  # "hvilken forordning"
]

# AGGREGATION patterns - detect multi-law summary intent
AGGREGATION_PATTERNS = [
    # English
    r"\ball\s+(?:eu\s+)?laws?\b",  # "all laws"
    r"\bevery\s+law\b",  # "every law"
    r"\bacross\s+(?:eu\s+)?laws?\b",  # "across laws"
    r"\beach\s+law\b",  # "each law"
    r"\ball\s+(?:eu\s+)?regulations?\b",  # "all regulations"
    r"\bwhat\s+do\s+(?:eu\s+)?laws?\s+say\b",  # "what do laws say"
    # Danish
    r"\balle\s+love\b",  # "alle love"
    r"\bhver\s+lov\b",  # "hver lov"
    r"\bpå\s+tværs\s+af\s+love\b",  # "på tværs af love"
]


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    """Check if text matches any of the given regex patterns."""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def detect_synthesis_mode(
    question: str,
    corpus_scope: str,  # "single" | "explicit" | "all"
    selected_corpora: list[str] | None,
    resolver: "CorpusResolver",
) -> SynthesisContext:
    """Detect synthesis mode from query and scope.

    Decision Logic:
        1. If corpus_scope == "single": return SINGLE mode
        2. Detect comparison keywords → COMPARISON mode
        3. Detect routing keywords → ROUTING mode (no deep synthesis)
        4. Detect aggregation keywords → AGGREGATION mode
        5. Default for multi-corpus: UNIFIED mode

    Args:
        question: The user's question
        corpus_scope: "single" | "explicit" | "all"
        selected_corpora: List of corpus IDs when scope is "explicit"
        resolver: CorpusResolver for alias matching and corpus lookup

    Returns:
        SynthesisContext with mode and target corpora
    """
    # Resolve target corpora based on scope
    if corpus_scope == "single":
        return SynthesisContext(
            mode=SynthesisMode.SINGLE,
            target_corpora=tuple(selected_corpora or []),
            routing_only=False,
        )

    if corpus_scope == "explicit":
        target_corpora = tuple(selected_corpora or [])
    else:  # "all"
        target_corpora = tuple(resolver.get_all_corpus_ids())

    # 2. Check for comparison intent
    if _matches_any_pattern(question, COMPARISON_PATTERNS):
        # Extract mentioned corpora for comparison pairs
        mentioned = resolver.mentioned_corpus_keys(question)
        if mentioned:
            target_corpora = tuple(mentioned)

        comparison_pairs = None
        if len(target_corpora) >= 2:
            # Create pairs for comparison (first two for now)
            comparison_pairs = ((target_corpora[0], target_corpora[1]),)

        return SynthesisContext(
            mode=SynthesisMode.COMPARISON,
            target_corpora=target_corpora,
            comparison_pairs=comparison_pairs,
            routing_only=False,
        )

    # 3. Check for routing intent (identification only)
    if _matches_any_pattern(question, ROUTING_PATTERNS):
        return SynthesisContext(
            mode=SynthesisMode.ROUTING,
            target_corpora=target_corpora,
            routing_only=True,
        )

    # 4. Check for aggregation intent
    if _matches_any_pattern(question, AGGREGATION_PATTERNS):
        return SynthesisContext(
            mode=SynthesisMode.AGGREGATION,
            target_corpora=target_corpora,
            routing_only=False,
        )

    # 5. Default: UNIFIED mode for multi-corpus queries
    return SynthesisContext(
        mode=SynthesisMode.UNIFIED,
        target_corpora=target_corpora,
        routing_only=False,
    )
