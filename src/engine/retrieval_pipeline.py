"""Modular Retrieval Pipeline for RAG.

This module implements a clean, testable retrieval pipeline with immutable
dataclasses and pure functions. It replaces the spaghetti code in rag.py's
answer_structured() method.

Pipeline Stages:
    1. VectorRetrieval   → Initial embedding search
    2. CitationExpansion → Graph-based article discovery + chunk injection
    3. HybridRerank      → 4-factor scoring (vec + bm25 + citation + role)
    4. ContextSelection  → Citable filter + diversity + top-k cap

Design Principles:
    - Frozen dataclasses for immutable input/output
    - No shared mutable state
    - Debug/observability built into each stage result
    - Dependency injection for testability

Usage:
    from src.engine.retrieval_pipeline import execute_pipeline, PipelineConfig

    result = execute_pipeline(
        input=PipelineInput(...),
        config=PipelineConfig.from_settings(),
        query_fn=...,
        inject_fn=...,
        is_citable_fn=...,
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Tuple

from ..common.config_loader import RankingWeights, load_settings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the entire pipeline."""

    # Pool sizes
    retrieval_pool_size: int = 50
    max_context_legal: int = 20
    max_context_engineering: int = 15

    # 4-factor ranking weights
    alpha_vec: float = 0.25
    beta_bm25: float = 0.25
    gamma_cite: float = 0.35
    delta_role: float = 0.15

    # Citation expansion
    citation_expansion_enabled: bool = True
    max_expansion: int = 30
    bump_bonus: float = 0.15
    min_weight: float = 0.15

    # Corpus
    corpus_id: str = ""

    @classmethod
    def from_settings(cls, corpus_id: str = "") -> "PipelineConfig":
        """Load config from settings.yaml."""
        try:
            settings = load_settings()
            rw = settings.ranking_weights or RankingWeights()

            return cls(
                retrieval_pool_size=settings.retrieval_pool_size or 50,
                max_context_legal=settings.max_context_legal or 20,
                max_context_engineering=settings.max_context_engineering or 15,
                alpha_vec=rw.alpha_vec,
                beta_bm25=rw.beta_bm25,
                gamma_cite=rw.gamma_cite,
                delta_role=rw.delta_role,
                citation_expansion_enabled=True,  # From citation_expansion config
                max_expansion=30,
                bump_bonus=0.15,
                min_weight=0.15,
                corpus_id=corpus_id,
            )
        except Exception:
            return cls(corpus_id=corpus_id)


# ---------------------------------------------------------------------------
# Core Data Types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved chunk - immutable."""

    chunk_id: str
    document: str
    metadata: Dict[str, Any]
    distance: float

    def anchor_key(self) -> str | None:
        """Return normalized anchor like 'article:6' or 'annex:iii'."""
        m = self.metadata or {}

        article = str(m.get("article") or "").strip()
        if article:
            return f"article:{article}".lower()

        annex = str(m.get("annex") or "").strip()
        annex_point = str(m.get("annex_point") or "").strip()
        if annex:
            if annex_point:
                return f"annex:{annex}:{annex_point}".lower()
            return f"annex:{annex}".lower()

        recital = str(m.get("recital") or "").strip()
        if recital:
            return f"recital:{recital}".lower()

        return None

    def to_hit_tuple(self) -> Tuple[str, Dict[str, Any]]:
        """Convert to legacy (document, metadata) tuple format."""
        return (self.document, dict(self.metadata))


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with scoring breakdown from hybrid rerank."""

    chunk: RetrievedChunk
    vec_score: float
    bm25_score: float
    citation_score: float
    role_score: float
    final_score: float


@dataclass(frozen=True)
class SelectedChunk:
    """A chunk selected for context with citation info."""

    chunk: RetrievedChunk
    is_citable: bool
    precise_ref: str | None = None
    rank: int = 0


# ---------------------------------------------------------------------------
# Stage 1: Vector Retrieval
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VectorRetrievalInput:
    """Input for vector retrieval stage."""

    question: str
    k: int
    where_filter: Dict[str, Any] | None = None


@dataclass(frozen=True)
class VectorRetrievalResult:
    """Output from vector retrieval stage."""

    chunks: Tuple[RetrievedChunk, ...]
    retrieval_mode: str  # "standard" | "filtered"
    duration_ms: float


def execute_vector_retrieval(
    input: VectorRetrievalInput,
    *,
    query_fn: Callable[[str, int, Dict[str, Any] | None], Tuple[List[Tuple[str, str, Dict[str, Any]]], List[float]]],
) -> VectorRetrievalResult:
    """Execute vector retrieval stage.

    Args:
        input: Retrieval parameters
        query_fn: Function that takes (question, k, where) and returns
                  (list of (id, doc, meta) tuples, list of distances)

    Returns:
        VectorRetrievalResult with retrieved chunks
    """
    start = time.perf_counter()

    raw_hits, distances = query_fn(input.question, input.k, input.where_filter)

    chunks: List[RetrievedChunk] = []
    for i, (chunk_id, doc, meta) in enumerate(raw_hits):
        dist = distances[i] if i < len(distances) else 0.0
        chunks.append(RetrievedChunk(
            chunk_id=chunk_id,
            document=doc,
            metadata=dict(meta) if meta else {},
            distance=dist,
        ))

    duration_ms = (time.perf_counter() - start) * 1000
    mode = "filtered" if input.where_filter else "standard"

    return VectorRetrievalResult(
        chunks=tuple(chunks),
        retrieval_mode=mode,
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Stage 2: Citation Expansion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CitationExpansionInput:
    """Input for citation expansion stage."""

    chunks: Tuple[RetrievedChunk, ...]
    question: str
    corpus_id: str
    max_expansion: int = 30
    min_weight: float = 0.15


@dataclass(frozen=True)
class CitationExpansionResult:
    """Output from citation expansion stage."""

    chunks: Tuple[RetrievedChunk, ...]  # Original + injected
    hint_anchors: FrozenSet[str]  # For downstream ranking
    chunks_injected: int
    injected_anchors: Tuple[str, ...]  # Which anchors were injected
    duration_ms: float


def execute_citation_expansion(
    input: CitationExpansionInput,
    *,
    inject_fn: Callable[[str, str, int], List[Tuple[str, str, Dict[str, Any], float]]] | None = None,
    expansion_enabled: bool = True,
) -> CitationExpansionResult:
    """Execute citation expansion stage.

    Discovers related articles via citation graph and optionally injects
    chunks for anchors not already in the pool.

    Args:
        input: Expansion parameters with current chunks
        inject_fn: Function to fetch chunks for a specific anchor.
                   Takes (question, anchor_key, k) and returns list of
                   (id, doc, meta, distance) tuples.
        expansion_enabled: Whether to perform expansion

    Returns:
        CitationExpansionResult with expanded chunks and hint anchors
    """
    start = time.perf_counter()

    if not expansion_enabled or not input.chunks:
        return CitationExpansionResult(
            chunks=input.chunks,
            hint_anchors=frozenset(),
            chunks_injected=0,
            injected_anchors=(),
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    # Get citation expansion from graph
    try:
        from .citation_expansion import (
            get_citation_expansion_for_query,
            is_citation_expansion_enabled,
        )

        if not is_citation_expansion_enabled():
            return CitationExpansionResult(
                chunks=input.chunks,
                hint_anchors=frozenset(),
                chunks_injected=0,
                injected_anchors=(),
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        # Extract metadatas for expansion
        metadatas = [dict(c.metadata) for c in input.chunks]

        expansion_articles = get_citation_expansion_for_query(
            corpus_id=input.corpus_id,
            question=input.question,
            retrieved_metadatas=metadatas,
            max_expansion=input.max_expansion,
        )

    except Exception:
        expansion_articles = []

    if not expansion_articles:
        return CitationExpansionResult(
            chunks=input.chunks,
            hint_anchors=frozenset(),
            chunks_injected=0,
            injected_anchors=(),
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    # Build hint anchors from expansion
    hint_anchors: set[str] = set()
    for art in expansion_articles:
        art_str = str(art).strip()
        if art_str.upper().startswith("ANNEX:"):
            # Handle ANNEX:III or ANNEX:III:5 format
            hint_anchors.add(art_str.lower())
        else:
            hint_anchors.add(f"article:{art_str}".lower())

    # Find which anchors are already covered
    existing_anchors: set[str] = set()
    for chunk in input.chunks:
        anchor = chunk.anchor_key()
        if anchor:
            existing_anchors.add(anchor)
            # Also add parent anchor for punkt-level
            if anchor.count(":") >= 2:
                parts = anchor.split(":")
                parent = f"{parts[0]}:{parts[1]}"
                existing_anchors.add(parent)

    # Inject chunks for missing anchors
    chunks_list = list(input.chunks)
    injected_anchors: List[str] = []

    if inject_fn:
        missing_anchors = hint_anchors - existing_anchors
        for anchor in sorted(missing_anchors)[:10]:  # Cap injection
            try:
                injected = inject_fn(input.question, anchor, 3)
                for chunk_id, doc, meta, dist in injected:
                    # Check for duplicates
                    if any(c.chunk_id == chunk_id for c in chunks_list):
                        continue
                    chunks_list.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        document=doc,
                        metadata=dict(meta) if meta else {},
                        distance=dist,
                    ))
                    injected_anchors.append(anchor)
            except Exception:
                pass

    duration_ms = (time.perf_counter() - start) * 1000

    return CitationExpansionResult(
        chunks=tuple(chunks_list),
        hint_anchors=frozenset(hint_anchors),
        chunks_injected=len(injected_anchors),
        injected_anchors=tuple(injected_anchors),
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Stage 3: Hybrid Rerank
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridRerankInput:
    """Input for hybrid rerank stage."""

    chunks: Tuple[RetrievedChunk, ...]
    question: str
    hint_anchors: FrozenSet[str]
    user_profile: str  # "LEGAL" | "ENGINEERING"
    weights: RankingWeights | None = None


@dataclass(frozen=True)
class HybridRerankResult:
    """Output from hybrid rerank stage."""

    scored_chunks: Tuple[ScoredChunk, ...]  # Sorted by final_score descending
    query_intent: str
    duration_ms: float


def execute_hybrid_rerank(
    input: HybridRerankInput,
) -> HybridRerankResult:
    """Execute hybrid rerank stage with 4-factor scoring.

    Score = α*vec_sim + β*bm25 + γ*citation + δ*role

    Args:
        input: Rerank parameters with chunks and hint anchors

    Returns:
        HybridRerankResult with scored and sorted chunks
    """
    import math

    start = time.perf_counter()

    if not input.chunks:
        return HybridRerankResult(
            scored_chunks=(),
            query_intent="UNKNOWN",
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    # Import ranking components
    from .helpers import classify_query_intent
    from .ranking import Ranker
    from .concept_config import extract_anchors_from_metadata

    # Get query intent
    query_intent = classify_query_intent(input.question)

    # Use provided weights or defaults
    w = input.weights or RankingWeights()

    # Extract data for scoring
    documents = [c.document for c in input.chunks]
    metadatas = [c.metadata for c in input.chunks]
    distances = [c.distance for c in input.chunks]

    # 1. Vector similarity (lower distance => higher similarity)
    vec_sims = [1.0 / (1.0 + float(d)) for d in distances]
    vec_norm = _normalize_scores(vec_sims)

    # 2. BM25 lexical scoring
    bm25_raw = Ranker._bm25_scores(query=input.question, documents=documents)
    bm25_norm = _normalize_scores(bm25_raw)

    # 3. Citation signal: 1.0 if chunk matches any hint anchor, else 0.0
    citation_signals: List[float] = []
    for meta in metadatas:
        chunk_anchors = extract_anchors_from_metadata(meta)
        has_match = bool(chunk_anchors & set(input.hint_anchors))
        citation_signals.append(1.0 if has_match else 0.0)

    # 4. Role signal: alignment with query intent and user profile
    role_signals: List[float] = []
    for meta, doc in zip(metadatas, documents):
        role_info = Ranker.classify_role(meta, doc)
        role = str(role_info.get("role") or Ranker.ROLE_OBLIGATIONS_NORMATIVE)
        role_conf = float(role_info.get("confidence") or 0.0)
        delta = Ranker._rerank_delta_for_role(
            role=role,
            role_confidence=role_conf,
            intent=query_intent,
            user_profile=input.user_profile,
        )
        # Normalize to [0, 1] range
        normalized_delta = max(0.0, min(1.0, (delta + 0.15) / 0.33))
        role_signals.append(normalized_delta)

    # Calculate final scores
    scored_chunks: List[ScoredChunk] = []
    for i, chunk in enumerate(input.chunks):
        final_score = (
            w.alpha_vec * vec_norm[i]
            + w.beta_bm25 * bm25_norm[i]
            + w.gamma_cite * citation_signals[i]
            + w.delta_role * role_signals[i]
        )
        scored_chunks.append(ScoredChunk(
            chunk=chunk,
            vec_score=vec_norm[i],
            bm25_score=bm25_norm[i],
            citation_score=citation_signals[i],
            role_score=role_signals[i],
            final_score=final_score,
        ))

    # Sort by final score descending
    scored_chunks.sort(key=lambda x: x.final_score, reverse=True)

    duration_ms = (time.perf_counter() - start) * 1000

    return HybridRerankResult(
        scored_chunks=tuple(scored_chunks),
        query_intent=query_intent,
        duration_ms=duration_ms,
    )


def _normalize_scores(values: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


# ---------------------------------------------------------------------------
# Stage 4: Context Selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextSelectionInput:
    """Input for context selection stage."""

    scored_chunks: Tuple[ScoredChunk, ...]
    max_context: int
    user_profile: str  # "LEGAL" | "ENGINEERING"
    hint_anchors: FrozenSet[str] = frozenset()  # Anchors to prioritize for diversity


@dataclass(frozen=True)
class ContextSelectionResult:
    """Output from context selection stage."""

    selected: Tuple[SelectedChunk, ...]
    citable_count: int
    unique_anchors: FrozenSet[str]
    duration_ms: float


def execute_context_selection(
    input: ContextSelectionInput,
    *,
    is_citable_fn: Callable[[Dict[str, Any], str], Tuple[bool, str | None]],
) -> ContextSelectionResult:
    """Execute context selection stage with diversity guard.

    Ensures hint anchors (from citation expansion) are represented in context,
    then fills remaining slots by hybrid rerank order.

    Pass 1: Include at least one chunk per hint anchor (diversity guarantee)
    Pass 2: Fill remaining slots by score order (quality guarantee)

    Args:
        input: Selection parameters with scored chunks and hint anchors
        is_citable_fn: Function that takes (metadata, doc_text) and returns
                       (is_citable, precise_ref_override)

    Returns:
        ContextSelectionResult with selected chunks
    """
    start = time.perf_counter()

    # Pre-filter to citable chunks only
    citable_scored: List[Tuple[int, ScoredChunk, str | None]] = []
    for rank, scored in enumerate(input.scored_chunks):
        is_citable, precise_ref = is_citable_fn(scored.chunk.metadata, scored.chunk.document)
        if is_citable:
            citable_scored.append((rank, scored, precise_ref))

    citable_count = len(citable_scored)

    selected: List[SelectedChunk] = []
    selected_indices: set[int] = set()
    unique_anchors: set[str] = set()

    def _add_chunk(idx: int, rank: int, scored: ScoredChunk, precise_ref: str | None) -> bool:
        """Add a chunk to selected. Returns True if added."""
        if idx in selected_indices:
            return False
        if len(selected) >= input.max_context:
            return False

        anchor = scored.chunk.anchor_key()
        if anchor:
            unique_anchors.add(anchor)

        selected.append(SelectedChunk(
            chunk=scored.chunk,
            is_citable=True,
            precise_ref=precise_ref,
            rank=rank + 1,
        ))
        selected_indices.add(idx)
        return True

    # Pass 1: Diversity guarantee - ensure hint anchors are represented
    # For each hint anchor, find the highest-scoring citable chunk that matches
    if input.hint_anchors:
        covered_hints: set[str] = set()

        # Build a map from hint anchor -> best rank in citable_scored
        # This lets us process hint anchors in score order (best chunks first)
        # rather than alphabetically, which biases toward annex over article.
        hint_best_rank: Dict[str, int] = {}
        for idx, (rank, scored, precise_ref) in enumerate(citable_scored):
            chunk_anchor = scored.chunk.anchor_key()
            if not chunk_anchor:
                continue
            for hint_anchor in input.hint_anchors:
                # Check if this chunk matches this hint
                if chunk_anchor == hint_anchor:
                    pass  # exact match
                elif hint_anchor.startswith(chunk_anchor + ":"):
                    pass  # hint is more specific than chunk
                elif chunk_anchor.startswith(hint_anchor + ":"):
                    pass  # chunk is more specific than hint
                else:
                    continue  # no match

                # Track best (lowest) rank for this hint
                if hint_anchor not in hint_best_rank or idx < hint_best_rank[hint_anchor]:
                    hint_best_rank[hint_anchor] = idx

        # Sort hint anchors by their best rank (highest-scoring first)
        # Hints with no matching chunks go to the end
        sorted_hints = sorted(
            input.hint_anchors,
            key=lambda h: (hint_best_rank.get(h, 999999), h),
        )

        for hint_anchor in sorted_hints:
            if hint_anchor in covered_hints:
                continue
            if len(selected) >= input.max_context:
                break

            # Find best matching chunk for this hint anchor
            for idx, (rank, scored, precise_ref) in enumerate(citable_scored):
                if idx in selected_indices:
                    continue
                chunk_anchor = scored.chunk.anchor_key()
                if not chunk_anchor:
                    continue

                # Check for match (exact or parent match for punkt-level)
                matches = False
                if chunk_anchor == hint_anchor:
                    matches = True
                elif hint_anchor.startswith(chunk_anchor + ":"):
                    # hint is more specific (e.g., annex:iii:5), chunk is parent (annex:iii)
                    matches = True
                elif chunk_anchor.startswith(hint_anchor + ":"):
                    # chunk is more specific, hint is parent
                    matches = True

                if matches:
                    _add_chunk(idx, rank, scored, precise_ref)
                    covered_hints.add(hint_anchor)
                    # Also mark parent anchors as covered
                    if ":" in hint_anchor:
                        parts = hint_anchor.split(":")
                        for i in range(1, len(parts)):
                            parent = ":".join(parts[:i+1])
                            covered_hints.add(parent)
                    break

    # Pass 2: Fill remaining slots by score order
    for idx, (rank, scored, precise_ref) in enumerate(citable_scored):
        if len(selected) >= input.max_context:
            break
        _add_chunk(idx, rank, scored, precise_ref)

    duration_ms = (time.perf_counter() - start) * 1000

    return ContextSelectionResult(
        selected=tuple(selected),
        citable_count=citable_count,
        unique_anchors=frozenset(unique_anchors),
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineInput:
    """Input for the full pipeline."""

    question: str
    corpus_id: str
    user_profile: str  # "LEGAL" | "ENGINEERING"
    where_filter: Dict[str, Any] | None = None
    explicit_article_refs: Tuple[str, ...] = ()
    explicit_annex_refs: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    """Output from the full pipeline."""

    context_result: ContextSelectionResult
    vector_result: VectorRetrievalResult
    expansion_result: CitationExpansionResult
    rerank_result: HybridRerankResult
    total_duration_ms: float

    @property
    def debug_summary(self) -> Dict[str, Any]:
        """Return debug info compatible with eval scorer."""
        return {
            # Stage 1: Vector retrieval
            "vector_retrieval": {
                "count": len(self.vector_result.chunks),
                "mode": self.vector_result.retrieval_mode,
                "duration_ms": round(self.vector_result.duration_ms, 2),
            },

            # Stage 2: Citation expansion
            "citation_expansion_articles": list(self.expansion_result.hint_anchors),
            "chunks_injected": self.expansion_result.chunks_injected,
            "anchor_hints": {
                "hint_anchors": list(self.expansion_result.hint_anchors),
                "injected_anchors": list(self.expansion_result.injected_anchors),
            },

            # Stage 3: Hybrid rerank
            "hybrid_rerank": {
                "enabled": True,
                "query_intent": self.rerank_result.query_intent,
                "top_10_scores": [
                    {
                        "chunk_id": sc.chunk.chunk_id,
                        "anchor": sc.chunk.anchor_key(),
                        "vec": round(sc.vec_score, 3),
                        "bm25": round(sc.bm25_score, 3),
                        "cite": round(sc.citation_score, 3),
                        "role": round(sc.role_score, 3),
                        "final": round(sc.final_score, 3),
                    }
                    for sc in self.rerank_result.scored_chunks[:10]
                ],
                "duration_ms": round(self.rerank_result.duration_ms, 2),
            },

            # Stage 4: Context selection
            "context_selection": {
                "citable_count": self.context_result.citable_count,
                "unique_anchors": list(self.context_result.unique_anchors),
                "final_count": len(self.context_result.selected),
                "duration_ms": round(self.context_result.duration_ms, 2),
            },

            # Timing
            "total_duration_ms": round(self.total_duration_ms, 2),
        }

    def get_hits(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get hits in legacy format for compatibility."""
        return [sel.chunk.to_hit_tuple() for sel in self.context_result.selected]

    def get_distances(self) -> List[float]:
        """Get distances for selected chunks."""
        return [sel.chunk.distance for sel in self.context_result.selected]

    def get_ids(self) -> List[str]:
        """Get chunk IDs for selected chunks."""
        return [sel.chunk.chunk_id for sel in self.context_result.selected]

    def get_metadatas(self) -> List[Dict[str, Any]]:
        """Get metadatas for selected chunks."""
        return [dict(sel.chunk.metadata) for sel in self.context_result.selected]


def execute_pipeline(
    input: PipelineInput,
    config: PipelineConfig,
    *,
    query_fn: Callable[[str, int, Dict[str, Any] | None], Tuple[List[Tuple[str, str, Dict[str, Any]]], List[float]]],
    inject_fn: Callable[[str, str, int], List[Tuple[str, str, Dict[str, Any], float]]] | None = None,
    is_citable_fn: Callable[[Dict[str, Any], str], Tuple[bool, str | None]],
) -> PipelineResult:
    """Execute the full retrieval pipeline.

    Stages:
        1. VectorRetrieval   → Initial embedding search
        2. CitationExpansion → Graph-based discovery + chunk injection
        3. HybridRerank      → 4-factor scoring
        4. ContextSelection  → Citable filter + top-k cap

    Args:
        input: Pipeline parameters
        config: Pipeline configuration
        query_fn: Function for vector retrieval
        inject_fn: Optional function for chunk injection
        is_citable_fn: Function to check if chunk is citable

    Returns:
        PipelineResult with all stage results
    """
    start = time.perf_counter()

    # Determine max_context based on profile
    max_context = (
        config.max_context_engineering
        if input.user_profile == "ENGINEERING"
        else config.max_context_legal
    )

    # Stage 1: Vector Retrieval
    vector_result = execute_vector_retrieval(
        VectorRetrievalInput(
            question=input.question,
            k=config.retrieval_pool_size,
            where_filter=input.where_filter,
        ),
        query_fn=query_fn,
    )

    # Stage 2: Citation Expansion
    expansion_result = execute_citation_expansion(
        CitationExpansionInput(
            chunks=vector_result.chunks,
            question=input.question,
            corpus_id=input.corpus_id,
            max_expansion=config.max_expansion,
            min_weight=config.min_weight,
        ),
        inject_fn=inject_fn,
        expansion_enabled=config.citation_expansion_enabled,
    )

    # Stage 3: Hybrid Rerank
    rerank_result = execute_hybrid_rerank(
        HybridRerankInput(
            chunks=expansion_result.chunks,
            question=input.question,
            hint_anchors=expansion_result.hint_anchors,
            user_profile=input.user_profile,
            weights=RankingWeights(
                alpha_vec=config.alpha_vec,
                beta_bm25=config.beta_bm25,
                gamma_cite=config.gamma_cite,
                delta_role=config.delta_role,
            ),
        ),
    )

    # Stage 4: Context Selection (with diversity guard for hint anchors)
    context_result = execute_context_selection(
        ContextSelectionInput(
            scored_chunks=rerank_result.scored_chunks,
            max_context=max_context,
            user_profile=input.user_profile,
            hint_anchors=expansion_result.hint_anchors,  # Pass for diversity
        ),
        is_citable_fn=is_citable_fn,
    )

    total_duration_ms = (time.perf_counter() - start) * 1000

    return PipelineResult(
        context_result=context_result,
        vector_result=vector_result,
        expansion_result=expansion_result,
        rerank_result=rerank_result,
        total_duration_ms=total_duration_ms,
    )
