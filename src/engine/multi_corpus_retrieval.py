"""Multi-corpus retrieval with Reciprocal Rank Fusion (RRF).

This module coordinates parallel retrieval across multiple corpora and fuses
results using RRF, the standard method for merging ranked results from
multiple retrievers.

RRF formula: score(d) = sum(1 / (k + rank_i(d))) for each corpus i

Design Principles:
    - Frozen dataclasses for immutable input/output
    - Dependency injection for testability (factory functions)
    - No shared mutable state
    - Each chunk tagged with corpus_id for source attribution

Usage:
    from src.engine.multi_corpus_retrieval import (
        MultiCorpusInput,
        MultiCorpusConfig,
        execute_multi_corpus_retrieval,
    )

    result = execute_multi_corpus_retrieval(
        input=MultiCorpusInput(...),
        config=MultiCorpusConfig(),
        query_fn_factory=lambda corpus_id: ...,
        inject_fn_factory=lambda corpus_id: ...,
        is_citable_fn=...,
    )
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from .retrieval_pipeline import (
    PipelineConfig,
    PipelineInput,
    PipelineResult,
    RetrievedChunk,
    ScoredChunk,
    execute_pipeline,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiCorpusConfig:
    """Configuration for multi-corpus retrieval."""

    retrieval_pool_size: int = 50  # Per corpus
    merged_pool_size: int = 100  # After RRF fusion
    rrf_k: int = 60  # RRF smoothing constant
    max_workers: int = 16  # ThreadPoolExecutor workers
    timeout_secs: float = 3.0  # Per-corpus timeout

    @classmethod
    def from_settings(cls, **overrides: Any) -> MultiCorpusConfig:
        """Create config from settings.yaml performance section."""
        from ..common.config_loader import get_performance_settings

        perf = get_performance_settings()
        defaults = {
            "max_workers": perf["max_retrieval_workers"],
            "timeout_secs": perf["retrieval_timeout_secs"],
        }
        defaults.update(overrides)
        return cls(**defaults)


@dataclass(frozen=True)
class MultiCorpusInput:
    """Input for multi-corpus retrieval."""

    question: str
    corpus_ids: Tuple[str, ...]  # Corpora to query
    user_profile: str  # LEGAL | ENGINEERING
    where_filter: Dict[str, Any] | None = None  # Additional filters


@dataclass(frozen=True)
class MultiCorpusResult:
    """Result from multi-corpus retrieval."""

    fused_chunks: Tuple[ScoredChunk, ...]  # Corpus-tagged, RRF-scored
    per_corpus_hits: Dict[str, int]  # corpus_id -> hit count
    duration_ms: float
    debug: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Type Aliases for Factories
# ---------------------------------------------------------------------------

# query_fn: (question, k, where_filter) -> (list of (id, doc, meta), list of distances)
QueryFn = Callable[
    [str, int, Dict[str, Any] | None],
    Tuple[List[Tuple[str, str, Dict[str, Any]]], List[float]],
]

# inject_fn: (question, anchor_key, k) -> list of (id, doc, meta, distance)
InjectFn = Callable[[str, str, int], List[Tuple[str, str, Dict[str, Any], float]]]

# is_citable_fn: (metadata, doc_text) -> (is_citable, precise_ref_override)
IsCitableFn = Callable[[Dict[str, Any], str], Tuple[bool, str | None]]


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------


def apply_rrf_fusion(
    per_corpus_results: Dict[str, List[ScoredChunk]],
    k: int = 60,
    max_results: int = 100,
) -> List[ScoredChunk]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    For each unique chunk across all corpora:
        rrf_score = sum(1 / (k + rank_in_corpus)) for each corpus containing it

    Args:
        per_corpus_results: Dict mapping corpus_id to list of ScoredChunks
                           (ordered by rank, 0-indexed)
        k: RRF smoothing constant (default 60, robust across systems)
        max_results: Maximum results to return after fusion

    Returns:
        List of ScoredChunks sorted by RRF score (highest first),
        with final_score set to the RRF score.
    """
    # Build a map of (corpus_id, chunk_id) -> (rank, ScoredChunk)
    # Using (corpus_id, chunk_id) as key because same chunk_id can appear
    # in different corpora with different content
    chunk_rrf_scores: Dict[Tuple[str, str], float] = {}
    chunk_data: Dict[Tuple[str, str], ScoredChunk] = {}

    for corpus_id, chunks in per_corpus_results.items():
        for rank, scored_chunk in enumerate(chunks):
            # RRF uses 1-indexed ranks
            rrf_contribution = 1.0 / (k + rank + 1)

            # Key is (corpus_id, chunk_id) to handle same chunk in different corpora
            key = (corpus_id, scored_chunk.chunk.chunk_id)

            if key not in chunk_rrf_scores:
                chunk_rrf_scores[key] = 0.0
                chunk_data[key] = scored_chunk

            chunk_rrf_scores[key] += rrf_contribution

    # Create new ScoredChunks with RRF score as final_score
    fused_chunks: List[ScoredChunk] = []
    for key, rrf_score in chunk_rrf_scores.items():
        original = chunk_data[key]
        # Create new ScoredChunk with RRF score
        fused_chunk = ScoredChunk(
            chunk=original.chunk,
            vec_score=original.vec_score,
            bm25_score=original.bm25_score,
            citation_score=original.citation_score,
            role_score=original.role_score,
            final_score=rrf_score,
        )
        fused_chunks.append(fused_chunk)

    # Sort by RRF score descending
    fused_chunks.sort(key=lambda x: x.final_score, reverse=True)

    # Apply max_results limit
    return fused_chunks[:max_results]


# ---------------------------------------------------------------------------
# Multi-Corpus Retrieval Coordinator
# ---------------------------------------------------------------------------


def execute_multi_corpus_retrieval(
    input: MultiCorpusInput,
    config: MultiCorpusConfig,
    *,
    query_fn_factory: Callable[[str], QueryFn],
    inject_fn_factory: Callable[[str], InjectFn | None],
    is_citable_fn: IsCitableFn,
) -> MultiCorpusResult:
    """Execute retrieval across multiple corpora and fuse results with RRF.

    Process:
    1. For each corpus_id in input.corpus_ids:
       - Build query function for that corpus
       - Execute the retrieval pipeline
       - Tag results with corpus_id
    2. Apply RRF to merge results across corpora
    3. Return fused_chunks with global ranking

    Args:
        input: Multi-corpus retrieval parameters
        config: Configuration for pool sizes and RRF
        query_fn_factory: Factory that creates query function for a corpus
        inject_fn_factory: Factory that creates inject function for a corpus
        is_citable_fn: Function to check if chunk is citable

    Returns:
        MultiCorpusResult with fused chunks and per-corpus hit counts
    """
    start = time.perf_counter()

    per_corpus_results: Dict[str, List[ScoredChunk]] = {}
    per_corpus_hits: Dict[str, int] = {}
    debug_info: Dict[str, Any] = {"per_corpus": {}}

    if not input.corpus_ids:
        return MultiCorpusResult(
            fused_chunks=(),
            per_corpus_hits={},
            duration_ms=0.0,
            debug=debug_info,
        )

    def _execute_single_corpus(corpus_id: str) -> Tuple[str, List[ScoredChunk], float]:
        """Execute retrieval for a single corpus. Returns (corpus_id, chunks, duration_ms)."""
        corpus_start = time.perf_counter()
        query_fn = query_fn_factory(corpus_id)
        inject_fn = inject_fn_factory(corpus_id)

        pipeline_config = PipelineConfig.from_settings(corpus_id=corpus_id)
        pipeline_config = PipelineConfig(
            retrieval_pool_size=config.retrieval_pool_size,
            max_context_legal=pipeline_config.max_context_legal,
            max_context_engineering=pipeline_config.max_context_engineering,
            alpha_vec=pipeline_config.alpha_vec,
            beta_bm25=pipeline_config.beta_bm25,
            gamma_cite=pipeline_config.gamma_cite,
            delta_role=pipeline_config.delta_role,
            citation_expansion_enabled=pipeline_config.citation_expansion_enabled,
            max_expansion=pipeline_config.max_expansion,
            bump_bonus=pipeline_config.bump_bonus,
            min_weight=pipeline_config.min_weight,
            corpus_id=corpus_id,
        )

        pipeline_input = PipelineInput(
            question=input.question,
            corpus_id=corpus_id,
            user_profile=input.user_profile,
            where_filter=input.where_filter,
        )

        pipeline_result = execute_pipeline(
            input=pipeline_input,
            config=pipeline_config,
            query_fn=query_fn,
            inject_fn=inject_fn,
            is_citable_fn=is_citable_fn,
        )

        corpus_chunks: List[ScoredChunk] = []
        for scored in pipeline_result.rerank_result.scored_chunks:
            tagged_metadata = dict(scored.chunk.metadata)
            tagged_metadata["corpus_id"] = corpus_id
            tagged_chunk = RetrievedChunk(
                chunk_id=scored.chunk.chunk_id,
                document=scored.chunk.document,
                metadata=tagged_metadata,
                distance=scored.chunk.distance,
            )
            corpus_chunks.append(ScoredChunk(
                chunk=tagged_chunk,
                vec_score=scored.vec_score,
                bm25_score=scored.bm25_score,
                citation_score=scored.citation_score,
                role_score=scored.role_score,
                final_score=scored.final_score,
            ))

        corpus_duration = (time.perf_counter() - corpus_start) * 1000
        return corpus_id, corpus_chunks, corpus_duration

    # Execute retrieval in parallel across corpora
    executor = ThreadPoolExecutor(max_workers=config.max_workers)
    future_to_corpus = {
        executor.submit(_execute_single_corpus, cid): cid
        for cid in input.corpus_ids
    }

    # Collect results with per-corpus timeout
    deadline = time.perf_counter() + config.timeout_secs
    try:
        for future in as_completed(future_to_corpus, timeout=config.timeout_secs):
            corpus_id = future_to_corpus[future]
            try:
                cid, chunks, dur_ms = future.result(timeout=0)
                per_corpus_results[cid] = chunks
                per_corpus_hits[cid] = len(chunks)
                debug_info["per_corpus"][cid] = {
                    "hits": len(chunks),
                    "duration_ms": round(dur_ms, 2),
                }
            except Exception as e:
                per_corpus_results[corpus_id] = []
                per_corpus_hits[corpus_id] = 0
                debug_info["per_corpus"][corpus_id] = {
                    "hits": 0,
                    "error": str(e),
                }
    except TimeoutError:
        # Mark remaining futures as timed out
        for future, corpus_id in future_to_corpus.items():
            if corpus_id not in per_corpus_results:
                per_corpus_results[corpus_id] = []
                per_corpus_hits[corpus_id] = 0
                debug_info["per_corpus"][corpus_id] = {
                    "hits": 0,
                    "error": f"Timeout after {config.timeout_secs}s",
                }
                future.cancel()

    executor.shutdown(wait=False, cancel_futures=True)

    # Apply RRF fusion
    fused_chunks = apply_rrf_fusion(
        per_corpus_results,
        k=config.rrf_k,
        max_results=config.merged_pool_size,
    )

    duration_ms = (time.perf_counter() - start) * 1000

    debug_info["rrf"] = {
        "k": config.rrf_k,
        "input_corpora": len(input.corpus_ids),
        "fused_count": len(fused_chunks),
    }
    debug_info["total_duration_ms"] = round(duration_ms, 2)

    return MultiCorpusResult(
        fused_chunks=tuple(fused_chunks),
        per_corpus_hits=per_corpus_hits,
        duration_ms=duration_ms,
        debug=debug_info,
    )
