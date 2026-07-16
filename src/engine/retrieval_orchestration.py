"""Retrieval orchestration — adapter layer between RAGEngine and retrieval pipeline.

Standalone functions extracted from RAGEngine (Step 6.3).
Each function takes explicit parameters instead of self.*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, TYPE_CHECKING

from . import citations
from . import metadata_helpers
from . import query_helpers
from .retrieval import Retriever
from .retrieval_pipeline import (
    PipelineConfig,
    PipelineInput,
    SelectedChunk,
    execute_pipeline,
)
from .multi_corpus_retrieval import (
    MultiCorpusInput,
    MultiCorpusConfig,
    MultiCorpusResult,
    execute_multi_corpus_retrieval,
)

if TYPE_CHECKING:
    from .planning import UserProfile


# ---------------------------------------------------------------------------
# Simple utilities
# ---------------------------------------------------------------------------


def get_collection_for_corpus(chroma: Any, corpus_id: str) -> Any:
    """Get ChromaDB collection for a specific corpus."""
    collection_name = f"{corpus_id}_documents"
    return chroma.get_or_create_collection(collection_name)


def get_case_law_collection_for_corpus(chroma: Any, corpus_id: str) -> Any:
    """Get the case law ChromaDB collection for a specific corpus."""
    return chroma.get_or_create_collection(f"{corpus_id}_case_law")


def make_query_fn(
    retriever: Any,
    collection: Any,
) -> Callable:
    """Create a query function adapter for the retrieval pipeline."""

    def query_fn(
        q: str,
        k: int,
        where: Dict[str, Any] | None,
    ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[float]]:
        ids, docs, metas, dists = retriever._query_collection_raw(
            collection=collection,
            question=q,
            k=k,
            where=where,
        )
        return list(zip(ids, docs, metas, strict=False)), dists

    return query_fn


def _anchor_to_where_hint(anchor: str, corpus_id: str) -> Dict[str, Any] | None:
    """Parse an anchor string (e.g. 'article:5') into a ChromaDB where-hint dict.

    Returns None if the anchor is not a recognised format.
    """
    if ":" not in anchor:
        return None
    kind, value = anchor.split(":", 1)
    kind = kind.strip().lower()
    value = value.strip()
    if kind not in {"article", "recital", "annex"}:
        return None

    where_hint: Dict[str, Any] = {"corpus_id": corpus_id}
    if kind == "annex" and ":" in value:
        parts = value.split(":")
        where_hint["annex"] = metadata_helpers.normalize_annex_for_chroma(parts[0])
        if len(parts) > 1:
            where_hint["annex_point"] = parts[1]
    elif kind == "annex":
        where_hint["annex"] = metadata_helpers.normalize_annex_for_chroma(value)
    else:
        where_hint[kind] = value
    return where_hint


def make_inject_fn(
    retriever: Any,
    collection: Any,
    corpus_id: str,
) -> Callable:
    """Create an anchor injection function for the retrieval pipeline."""

    def inject_fn(
        q: str,
        anchor: str,
        k: int,
    ) -> List[Tuple[str, str, Dict[str, Any], float]]:
        where_hint = _anchor_to_where_hint(anchor, corpus_id)
        if where_hint is None:
            return []
        ids, docs, metas, dists = retriever._query_collection_raw(
            collection=collection,
            question=q,
            k=k,
            where=where_hint,
            track_state=False,
        )
        return list(zip(ids, docs, metas, dists, strict=False))

    return inject_fn


# ---------------------------------------------------------------------------
# Modular retrieval (single corpus)
# ---------------------------------------------------------------------------


def modular_retrieval(
    *,
    question: str,
    resolved_profile: "UserProfile",
    where_for_retrieval: Dict[str, Any] | None,
    corpus_id: str,
    retriever: Any,
    collection: Any,
    is_citable_fn: Callable,
) -> Dict[str, Any]:
    """Run the modular retrieval pipeline for a single corpus.

    Returns a dict with keys: hits, distances, retrieved_ids, retrieved_metas,
    run_meta_updates, selected_chunks, total_retrieved, citable_count,
    pipeline_result.
    """
    from .planning import UserProfile as UP

    config = PipelineConfig.from_settings(corpus_id=corpus_id)

    profile_str = "ENGINEERING" if resolved_profile == UP.ENGINEERING else "LEGAL"
    pipeline_input = PipelineInput(
        question=question,
        corpus_id=corpus_id,
        user_profile=profile_str,
        where_filter=where_for_retrieval,
    )

    pipeline_result = execute_pipeline(
        input=pipeline_input,
        config=config,
        query_fn=make_query_fn(retriever, collection),
        inject_fn=make_inject_fn(retriever, collection, corpus_id),
        is_citable_fn=is_citable_fn,
    )

    hits = pipeline_result.get_hits()
    distances = pipeline_result.get_distances()
    retrieved_ids = pipeline_result.get_ids()
    retrieved_metas = pipeline_result.get_metadatas()

    run_meta_updates = {
        "modular_pipeline": pipeline_result.debug_summary,
        "anchor_hints": {
            "hint_anchors": list(pipeline_result.expansion_result.hint_anchors),
            "citation_expansion_articles": list(
                pipeline_result.expansion_result.hint_anchors
            ),
            "anchor_hints_applied": True,
            "injected_added": pipeline_result.expansion_result.chunks_injected,
            "citation_source": "modular_pipeline",
        },
        "hybrid_rerank": {
            "enabled": True,
            "query_intent": pipeline_result.rerank_result.query_intent,
            "top_scores": [
                {
                    "anchor": sc.chunk.anchor_key(),
                    "vec": round(sc.vec_score, 3),
                    "bm25": round(sc.bm25_score, 3),
                    "cite": round(sc.citation_score, 3),
                    "role": round(sc.role_score, 3),
                    "final": round(sc.final_score, 3),
                }
                for sc in pipeline_result.rerank_result.scored_chunks[:10]
            ],
        },
    }

    return {
        "hits": list(hits),
        "distances": list(distances),
        "retrieved_ids": list(retrieved_ids),
        "retrieved_metas": list(retrieved_metas),
        "run_meta_updates": run_meta_updates,
        "selected_chunks": pipeline_result.context_result.selected,
        "total_retrieved": len(pipeline_result.vector_result.chunks),
        "citable_count": pipeline_result.context_result.citable_count,
        "pipeline_result": pipeline_result,
    }


# ---------------------------------------------------------------------------
# Cross-law (multi-corpus) retrieval
# ---------------------------------------------------------------------------


def execute_cross_law_retrieval(
    *,
    question: str,
    corpus_ids: Tuple[str, ...],
    resolved_profile: "UserProfile",
    where_filter: Dict[str, Any] | None,
    retriever: Any,
    chroma: Any,
    is_citable_fn: Callable,
) -> MultiCorpusResult:
    """Execute multi-corpus retrieval with RRF fusion.

    Returns the raw MultiCorpusResult for conversion by the caller.
    """
    multi_input = MultiCorpusInput(
        question=question,
        corpus_ids=corpus_ids,
        user_profile=str(resolved_profile.value)
        if hasattr(resolved_profile, "value")
        else str(resolved_profile),
        where_filter=where_filter,
    )
    multi_config = MultiCorpusConfig()

    def query_fn_factory(cid: str) -> Callable:
        collection = get_collection_for_corpus(chroma, cid)

        def query_fn(
            q: str, k: int, where: Dict[str, Any] | None
        ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[float]]:
            effective_where = dict(where or {})
            effective_where["corpus_id"] = cid
            ids, docs, metas, dists = retriever._query_collection_raw(
                collection=collection,
                question=q,
                k=k,
                where=effective_where,
            )
            return list(zip(ids, docs, metas, strict=False)), dists

        return query_fn

    def inject_fn_factory(cid: str) -> Callable:
        collection = get_collection_for_corpus(chroma, cid)

        def inject_fn(
            q: str, anchor: str, k: int
        ) -> List[Tuple[str, str, Dict[str, Any], float]]:
            where_hint = _anchor_to_where_hint(anchor, cid)
            if where_hint is None:
                return []
            ids, docs, metas, dists = retriever._query_collection_raw(
                collection=collection,
                question=q,
                k=k,
                where=where_hint,
            )
            return list(zip(ids, docs, metas, dists, strict=False))

        return inject_fn

    return execute_multi_corpus_retrieval(
        input=multi_input,
        config=multi_config,
        query_fn_factory=query_fn_factory,
        inject_fn_factory=inject_fn_factory,
        is_citable_fn=is_citable_fn,
    )


def convert_multi_corpus_result(
    multi_result: MultiCorpusResult,
    resolved_profile: "UserProfile",
) -> Dict[str, Any]:
    """Convert MultiCorpusResult to flat dict format for _RetrievalResult construction.

    Returns dict with: hits, distances, retrieved_ids, retrieved_metas,
    run_meta_updates, selected_chunks, total_retrieved, citable_count.
    """
    from ..common.config_loader import load_settings
    from .planning import UserProfile as UP

    hits: List[Tuple[str, Dict[str, Any]]] = []
    distances: List[float] = []
    retrieved_ids: List[str] = []
    retrieved_metas: List[Dict[str, Any]] = []
    selected_chunks_list: List[SelectedChunk] = []

    for i, scored_chunk in enumerate(multi_result.fused_chunks):
        chunk = scored_chunk.chunk
        meta = dict(chunk.metadata)

        hits.append((chunk.document, meta))
        distances.append(chunk.distance)
        retrieved_ids.append(chunk.chunk_id)
        retrieved_metas.append(meta)

        is_citable = citations._is_citable_metadata(meta)
        selected_chunks_list.append(
            SelectedChunk(
                chunk=chunk,
                is_citable=is_citable,
                precise_ref=None,
                rank=i,
            )
        )

    settings = load_settings()
    context_cap = (
        settings.max_context_legal
        if resolved_profile == UP.LEGAL
        else settings.max_context_engineering
    )
    capped_selected = tuple(selected_chunks_list[:context_cap])

    run_meta_updates = {
        "multi_corpus_retrieval": {
            "enabled": True,
            "per_corpus_hits": multi_result.per_corpus_hits,
            "duration_ms": round(multi_result.duration_ms, 2),
            "fused_count": len(multi_result.fused_chunks),
        },
        "laws_searched": list(multi_result.per_corpus_hits.keys()),
    }

    return {
        "hits": hits,
        "distances": distances,
        "retrieved_ids": retrieved_ids,
        "retrieved_metas": retrieved_metas,
        "run_meta_updates": run_meta_updates,
        "selected_chunks": capped_selected,
        "total_retrieved": len(multi_result.fused_chunks),
        "citable_count": sum(1 for sc in selected_chunks_list if sc.is_citable),
    }


# ---------------------------------------------------------------------------
# query_with_where decomposition (Step 7.11)
# ---------------------------------------------------------------------------


@dataclass
class QueryWithWhereResult:
    """Result from execute_query_with_where — pure data, no engine state."""

    hits: List[Tuple[str, Dict[str, Any]]]
    distances: List[float]
    ids: List[str]
    metadatas: List[Dict[str, Any]]
    query_where: Dict[str, Any] | None
    collection_name: str | None
    sibling_expansion: Dict[str, Any]


def _try_article_shortcut(
    question: str,
    k: int,
    where: Dict[str, Any] | None,
    collection: Any,
    collection_name: str | None,
    retriever: Any,
) -> QueryWithWhereResult | None:
    """Attempt single-article pre-filtered retrieval. Returns None on miss."""
    article_refs = query_helpers._extract_article_refs(question)
    if len(article_refs) != 1:
        return None
    try:
        where_article = dict(where or {})
        where_article["article"] = article_refs[0]
        filtered, filtered_distances = retriever._query_collection_with_distances(
            collection=collection, question=question, k=k, where=where_article,
        )
        if not filtered:
            return None
        return QueryWithWhereResult(
            hits=filtered, distances=filtered_distances,
            ids=[], metadatas=[], query_where=where_article,
            collection_name=collection_name,
            sibling_expansion=_disabled_expansion(),
        )
    except Exception:  # noqa: BLE001
        return None


def _disabled_expansion(count: int = 0) -> Dict[str, Any]:
    return {"enabled": False, "original_count": count, "expanded_count": count, "siblings_added": 0}


def _apply_sibling_expansion(
    retriever: Any, collection: Any, ids: List, docs: List, metas: List, distances: List,
) -> tuple[List, List, List, List, Dict[str, Any]]:
    """Expand results with sibling chunks if enabled. Returns updated lists + expansion info."""
    from ..common.config_loader import get_sibling_expansion_settings

    sibling_settings = get_sibling_expansion_settings()
    expand_fn = getattr(retriever, "_expand_to_siblings", None)
    if not (sibling_settings.get("enabled", False) and ids and callable(expand_fn)):
        return ids, docs, metas, distances, _disabled_expansion(len(ids) if ids else 0)

    original_count = len(ids)
    max_siblings = sibling_settings.get("max_siblings", 2)
    ids, docs, metas, distances = expand_fn(
        collection=collection, ids=ids, documents=docs,
        metadatas=metas, distances=distances, max_siblings=max_siblings,
    )
    return ids, docs, metas, distances, {
        "enabled": True, "original_count": original_count,
        "expanded_count": len(ids), "siblings_added": len(ids) - original_count,
        "max_siblings": max_siblings,
    }


def _select_precise_hits(
    precise: tuple, imprecise: tuple, k: int,
) -> tuple[List, List, List]:
    """Select up to k hits, preferring precise over imprecise."""
    p, pd, pid = precise
    ip, ipd, ipid = imprecise
    if len(p) >= k:
        return p[:k], pd[:k], pid[:k]
    need = k - len(p)
    return p + ip[:need], pd + ipd[:need], pid + ipid[:need]


def execute_query_with_where(
    *,
    question: str,
    k: int,
    where: Dict[str, Any] | None,
    collection: Any,
    collection_name: str | None,
    retriever: Any,
    ranker: Any,
    enable_hybrid_rerank: bool,
    hybrid_vec_k: int,
    ranking_weights: Any,
) -> QueryWithWhereResult:
    """Execute query with optional article filtering and hybrid reranking.

    This is the core query logic extracted from RAGEngine.query_with_where().
    The caller (RAGEngine) handles state reset/sync around this function.

    Returns:
        QueryWithWhereResult with hits, distances, and metadata for state sync.
    """
    shortcut = _try_article_shortcut(
        question, k, where, collection, collection_name, retriever,
    )
    if shortcut is not None:
        return shortcut

    if enable_hybrid_rerank:
        return _hybrid_rerank_path(
            question=question, k=k, where=where, collection=collection,
            collection_name=collection_name, retriever=retriever,
            ranker=ranker, hybrid_vec_k=hybrid_vec_k,
            ranking_weights=ranking_weights,
        )

    return _simple_query_path(
        question=question, k=k, where=where, collection=collection,
        collection_name=collection_name, retriever=retriever,
    )


def _hybrid_rerank_path(
    *,
    question: str,
    k: int,
    where: Dict[str, Any] | None,
    collection: Any,
    collection_name: str | None,
    retriever: Any,
    ranker: Any,
    hybrid_vec_k: int,
    ranking_weights: Any,
) -> QueryWithWhereResult:
    """Hybrid reranking over expanded candidate set."""
    vec_k = max(int(hybrid_vec_k), int(k))
    ids, docs, metas, distances = retriever._query_collection_raw(
        collection=collection, question=question, k=vec_k, where=where,
    )

    ids, docs, metas, distances, sibling_expansion = _apply_sibling_expansion(
        retriever, collection, ids, docs, metas, distances,
    )

    hits, out_distances, out_ids = ranker._hybrid_rerank_hits(
        question=question, ids=ids, documents=docs, metadatas=metas,
        distances=distances, k=k, weights=ranking_weights,
    )

    p, pd, pid, ip, ipd, ipid = Retriever.split_precise(hits, out_distances, out_ids)
    selected, sel_d, sel_ids = _select_precise_hits((p, pd, pid), (ip, ipd, ipid), int(k))

    return QueryWithWhereResult(
        hits=selected, distances=list(sel_d), ids=list(sel_ids),
        metadatas=[dict(m or {}) for _, m in selected],
        query_where=where, collection_name=collection_name,
        sibling_expansion=sibling_expansion,
    )


def _simple_query_path(
    *,
    question: str,
    k: int,
    where: Dict[str, Any] | None,
    collection: Any,
    collection_name: str | None,
    retriever: Any,
) -> QueryWithWhereResult:
    """Fallback: simple distance-based query."""
    hits, distances = retriever._query_collection_with_distances(
        collection=collection, question=question, k=k, where=where,
    )

    p, pd, pid, ip, ipd, ipid = Retriever.split_precise_simple(hits, distances)
    selected, sel_d, sel_ids = _select_precise_hits((p, pd, pid), (ip, ipd, ipid), int(k))

    return QueryWithWhereResult(
        hits=selected, distances=list(sel_d), ids=list(sel_ids),
        metadatas=[dict(m or {}) for _, m in selected],
        query_where=where, collection_name=collection_name,
        sibling_expansion=_disabled_expansion(),
    )
