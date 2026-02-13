"""Corpus discovery classifier for AI Law Discovery.

Single Responsibility: Classify which corpora are relevant to a query
using a three-stage pipeline (alias detection, retrieval probe, optional
LLM disambiguation). Returns ranked matches with confidence scores.

Does NOT import rag.py, chromadb, or openai. All external dependencies
are injected as callables.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types (frozen, immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveryMatch:
    """A single corpus matched by discovery."""

    corpus_id: str
    confidence: float  # 0.0 - 1.0
    reason: str  # "alias_match" | "retrieval_probe" | "llm_disambiguation"


@dataclass(frozen=True)
class DiscoveryResult:
    """Complete discovery output with gating decision."""

    matches: tuple[DiscoveryMatch, ...]
    resolved_scope: str  # "single" | "explicit" | "abstain"
    resolved_corpora: tuple[str, ...]
    gate: str  # "AUTO" | "SUGGEST" | "ABSTAIN"


@dataclass(frozen=True)
class DiscoveryConfig:
    """Discovery configuration — all defaults match settings.yaml."""

    enabled: bool = True
    probe_top_k: int = 10
    auto_threshold: float = 0.75
    suggest_threshold: float = 0.65
    ambiguity_margin: float = 0.10
    max_corpora: int = 5
    llm_disambiguation: bool = True
    disambiguation_model: str | None = None
    # Scoring weights: avg similarity + best similarity (review fix: removed w_count)
    w_similarity: float = 0.50
    w_best: float = 0.50
    # Vague-query guard: if more than this many corpora pass suggest_threshold
    # but none reaches auto_threshold, the query lacks specificity → ABSTAIN
    max_suggest_corpora: int = 2


# ---------------------------------------------------------------------------
# Distance-to-similarity conversion
# ---------------------------------------------------------------------------


def distance_to_similarity(distance: float) -> float:
    """Convert ChromaDB L2 distance to similarity score.

    Uses Gaussian decay exp(-d²/2) for better score discrimination.
    The 1/(1+d) mapping compresses the useful range [0.2, 1.5] into
    a narrow band (~0.40-0.83), making threshold gating unreliable.
    Gaussian spreads the same range across [0.32, 0.98].

    Maps: distance 0 → 1.0, distance ~0.76 → ~0.75, distance ∞ → 0.0.
    Handles negative distances safely.
    """
    d = max(0.0, distance)
    return math.exp(-(d * d) / 2.0)


# ---------------------------------------------------------------------------
# Stage 1: Alias Detection
# ---------------------------------------------------------------------------


def _stage_alias_detection(
    question: str,
    resolver: object,
) -> list[DiscoveryMatch]:
    """Detect explicitly named corpora via alias matching.

    Uses CorpusResolver.mentioned_corpus_keys() for deterministic,
    word-boundary regex matching. Matches get confidence 1.0.
    """
    keys = resolver.mentioned_corpus_keys(question)  # type: ignore[attr-defined]
    return [
        DiscoveryMatch(corpus_id=key, confidence=1.0, reason="alias_match")
        for key in keys
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_corpus(distances: list[float], config: DiscoveryConfig) -> float:
    """Score a single corpus from its retrieval probe distances.

    Uses threshold-based scoring (review fix): weights average similarity
    and best-hit similarity. Returns 0.0 for empty results.
    """
    if not distances:
        return 0.0

    similarities = [distance_to_similarity(d) for d in distances]
    avg_sim = sum(similarities) / len(similarities)
    best_sim = max(similarities)

    return config.w_similarity * avg_sim + config.w_best * best_sim


# ---------------------------------------------------------------------------
# Stage 2: Retrieval Probe
# ---------------------------------------------------------------------------


def _stage_retrieval_probe(
    question: str,
    corpus_ids: list[str],
    probe_fn: Callable[[str, str, int], list[tuple[dict, float]]],
    config: DiscoveryConfig,
) -> list[DiscoveryMatch]:
    """Probe each corpus with a shallow vector search.

    Queries each corpus collection with low top-k, scores results by
    average and best similarity. Returns matches sorted by confidence desc.
    """
    matches: list[DiscoveryMatch] = []

    for corpus_id in corpus_ids:
        try:
            results = probe_fn(question, corpus_id, config.probe_top_k)
        except Exception:
            logger.warning("Probe failed for corpus %s", corpus_id, exc_info=True)
            continue

        if not results:
            continue

        distances = [dist for _meta, dist in results]
        confidence = _score_corpus(distances, config)

        matches.append(
            DiscoveryMatch(
                corpus_id=corpus_id,
                confidence=round(confidence, 4),
                reason="retrieval_probe",
            )
        )

    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Stage Merging
# ---------------------------------------------------------------------------


def _merge_stages(
    alias_matches: list[DiscoveryMatch],
    probe_matches: list[DiscoveryMatch],
    llm_matches: list[DiscoveryMatch],
) -> list[DiscoveryMatch]:
    """Merge results from all stages. Alias > LLM > Probe for same corpus."""
    seen: dict[str, DiscoveryMatch] = {}

    # Alias matches take highest priority
    for m in alias_matches:
        seen[m.corpus_id] = m

    # LLM disambiguation results (if any) override probe
    for m in llm_matches:
        if m.corpus_id not in seen:
            seen[m.corpus_id] = m

    # Probe results fill in the rest
    for m in probe_matches:
        if m.corpus_id not in seen:
            seen[m.corpus_id] = m

    merged = list(seen.values())
    merged.sort(key=lambda m: m.confidence, reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Confidence Gating
# ---------------------------------------------------------------------------


def _apply_confidence_gating(
    matches: list[DiscoveryMatch],
    config: DiscoveryConfig,
) -> DiscoveryResult:
    """Apply confidence thresholds to determine routing gate."""
    if not matches:
        return DiscoveryResult(
            matches=(),
            resolved_scope="abstain",
            resolved_corpora=(),
            gate="ABSTAIN",
        )

    auto_corpora = [m for m in matches if m.confidence >= config.auto_threshold]
    suggest_corpora = [m for m in matches if m.confidence >= config.suggest_threshold]

    if auto_corpora:
        # Only AUTO-tier corpora go into resolved_corpora. SUGGEST-tier
        # matches remain visible in the full matches list (for sidebar)
        # but are not confident enough for the banner or auto-search.
        selected = auto_corpora[: config.max_corpora]
        return DiscoveryResult(
            matches=tuple(matches),
            resolved_scope="explicit",
            resolved_corpora=tuple(m.corpus_id for m in selected),
            gate="AUTO",
        )

    if suggest_corpora:
        # Vague-query guard: too many suggest matches without any AUTO
        # indicates a non-specific query (e.g. "hvad er loven?")
        if len(suggest_corpora) > config.max_suggest_corpora:
            logger.info(
                "ABSTAIN: %d corpora above suggest threshold (max %d) — query too vague",
                len(suggest_corpora), config.max_suggest_corpora,
            )
            return DiscoveryResult(
                matches=tuple(matches),
                resolved_scope="abstain",
                resolved_corpora=(),
                gate="ABSTAIN",
            )

        selected = suggest_corpora[: config.max_corpora]
        return DiscoveryResult(
            matches=tuple(matches),
            resolved_scope="explicit",
            resolved_corpora=tuple(m.corpus_id for m in selected),
            gate="SUGGEST",
        )

    return DiscoveryResult(
        matches=tuple(matches),
        resolved_scope="abstain",
        resolved_corpora=(),
        gate="ABSTAIN",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def discover_corpora(
    question: str,
    corpus_ids: list[str],
    probe_fn: Callable[[str, str, int], list[tuple[dict, float]]],
    resolver: object,
    config: DiscoveryConfig,
    llm_fn: Callable[[str], str] | None = None,
) -> DiscoveryResult:
    """Discover which corpora are relevant to a question.

    Three-stage pipeline:
    1. Alias detection (deterministic, <1ms)
    2. Retrieval probe (semantic, <200ms)
    3. LLM disambiguation (optional, <500ms)

    Returns a DiscoveryResult with ranked matches and gating decision.
    All external dependencies are injected — no rag.py/chromadb imports.
    """
    if not corpus_ids:
        return DiscoveryResult(
            matches=(),
            resolved_scope="abstain",
            resolved_corpora=(),
            gate="ABSTAIN",
        )

    # Stage 1: Alias detection
    alias_matches = _stage_alias_detection(question, resolver)

    # Stage 2: Retrieval probe
    try:
        probe_matches = _stage_retrieval_probe(
            question, corpus_ids, probe_fn, config,
        )
    except Exception:
        logger.error("Discovery probe failed entirely", exc_info=True)
        probe_matches = []

    # Stage 3: LLM disambiguation (optional)
    llm_matches: list[DiscoveryMatch] = []
    if (
        config.llm_disambiguation
        and llm_fn is not None
        and len(probe_matches) >= 2
    ):
        top_two = probe_matches[:2]
        gap = top_two[0].confidence - top_two[1].confidence
        if gap < config.ambiguity_margin:
            try:
                llm_matches = _stage_llm_disambiguation(
                    question, probe_matches[:5], llm_fn,
                )
            except Exception:
                logger.warning("LLM disambiguation failed, using probe results", exc_info=True)

    # Merge stages
    merged = _merge_stages(alias_matches, probe_matches, llm_matches)

    # Apply confidence gating
    return _apply_confidence_gating(merged, config)


def _stage_llm_disambiguation(
    question: str,
    candidates: list[DiscoveryMatch],
    llm_fn: Callable[[str], str],
) -> list[DiscoveryMatch]:
    """Use LLM to disambiguate top candidates.

    Asks the LLM which corpus is most relevant. Returns re-ranked
    matches with adjusted confidence.
    """
    corpus_list = ", ".join(m.corpus_id for m in candidates)
    prompt = (
        f"Given the question: '{question}'\n"
        f"Which of these EU regulations is most relevant: {corpus_list}?\n"
        f"Reply with just the regulation identifier."
    )
    response = llm_fn(prompt).strip().lower()

    # Boost the LLM-selected corpus
    result: list[DiscoveryMatch] = []
    for m in candidates:
        if m.corpus_id.lower() in response:
            boosted = min(1.0, m.confidence + 0.15)
            result.append(
                DiscoveryMatch(
                    corpus_id=m.corpus_id,
                    confidence=round(boosted, 4),
                    reason="llm_disambiguation",
                )
            )
        else:
            result.append(m)

    result.sort(key=lambda m: m.confidence, reverse=True)
    return result
