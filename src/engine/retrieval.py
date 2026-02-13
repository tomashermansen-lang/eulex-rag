import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
from copy import deepcopy
from dataclasses import dataclass, field

import yaml

from .llm_client import make_openai_client
from . import helpers
from .types import RAGEngineError
from ..common.config_loader import get_sibling_expansion_settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score floor / fusion utilities (consolidated from retrieval_fusion.py)
# ---------------------------------------------------------------------------


def _load_fusion_config() -> dict[str, Any]:
    """Load configuration from settings.yaml for fusion settings."""
    config_path = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_floor_config() -> Tuple[float, float]:
    """Get floor threshold and boost from config.

    Returns:
        Tuple of (floor_threshold, floor_boost)
    """
    config = _load_fusion_config()
    citation_cfg = config.get("citation_expansion", {})
    threshold = citation_cfg.get("floor_threshold", 0.85)
    boost = citation_cfg.get("floor_boost", 0.5)
    return float(threshold), float(boost)


def apply_score_floor(
    candidates: List[Dict[str, Any]],
    initial_hits: List[Dict[str, Any]],
    *,
    floor_threshold: float | None = None,
    floor_boost: float | None = None,
) -> List[Dict[str, Any]]:
    """Ensure high-confidence initial hits maintain minimum score.

    Prevents citation expansion from displacing semantically relevant chunks
    that scored well in initial embedding search.

    Args:
        candidates: All candidates after expansion + reranking.
                   Each dict should have 'chunk_id' and 'final_score' keys.
        initial_hits: Top hits from initial embedding search.
                     Each dict should have 'chunk_id' and 'distance' keys.
        floor_threshold: Distance threshold (lower = higher confidence).
                        Hits with distance below this get protection.
                        If None, reads from config.
        floor_boost: Minimum score boost for protected hits.
                    If None, reads from config.

    Returns:
        Candidates with score floor applied to protected initial hits.
    """
    if not candidates or not initial_hits:
        return candidates

    # Load defaults from config if not provided
    if floor_threshold is None or floor_boost is None:
        cfg_threshold, cfg_boost = get_floor_config()
        floor_threshold = floor_threshold if floor_threshold is not None else cfg_threshold
        floor_boost = floor_boost if floor_boost is not None else cfg_boost

    # Identify protected chunks: initial hits with high confidence
    protected_ids: Dict[str, float] = {}  # chunk_id -> original_distance
    for hit in initial_hits:
        chunk_id = hit.get("chunk_id") or hit.get("id")
        distance = hit.get("distance", 1.0)
        if chunk_id and distance < floor_threshold:
            protected_ids[chunk_id] = distance

    if not protected_ids:
        return candidates

    # Apply score floor to protected chunks
    result = []
    protected_applied = 0
    for candidate in candidates:
        chunk_id = candidate.get("chunk_id") or candidate.get("id")
        if chunk_id in protected_ids:
            original_distance = protected_ids[chunk_id]
            current_score = candidate.get("final_score", 0.0)

            # Calculate floor score based on original distance
            # Lower distance = higher floor score
            floor_score = floor_boost * (1.0 - original_distance)

            if current_score < floor_score:
                candidate = dict(candidate)  # Copy to avoid mutation
                candidate["final_score"] = floor_score
                candidate["score_floor_applied"] = True
                candidate["original_score"] = current_score
                protected_applied += 1

        result.append(candidate)

    if protected_applied > 0:
        logger.debug(
            "Score floor applied to %d/%d protected initial hits",
            protected_applied,
            len(protected_ids),
        )

    return result


def get_initial_protected_hits(
    hits: List[Tuple[float, Dict[str, Any]]],
    *,
    top_n: int = 3,
    threshold: float | None = None,
) -> List[Dict[str, Any]]:
    """Extract high-confidence initial hits for protection.

    Args:
        hits: List of (distance, metadata) tuples from initial retrieval.
        top_n: Consider only the top N hits.
        threshold: Only protect hits with distance below this.
                  If None, reads from config.

    Returns:
        List of hit dicts with 'chunk_id' and 'distance' for protection.
    """
    # Load threshold from config if not provided
    if threshold is None:
        threshold, _ = get_floor_config()

    protected = []
    for i, (distance, meta) in enumerate(hits[:top_n]):
        if distance < threshold:
            chunk_id = (meta or {}).get("chunk_id")
            if chunk_id:
                protected.append({
                    "chunk_id": chunk_id,
                    "distance": distance,
                    "initial_rank": i,
                    "metadata": meta,
                })

    if protected:
        logger.debug(
            "Protected %d initial hits with distance < %.2f",
            len(protected),
            threshold,
        )

    return protected


# ---------------------------------------------------------------------------
# Retrieval pass tracking helpers (extracted from RAGEngine.answer_structured)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalPassInfo:
    """Debug info for a single retrieval pass."""
    pass_name: str
    planned_where: Dict[str, Any] | None
    planned_collection_type: str
    effective_where: Dict[str, Any] | None = None
    effective_collection: str | None = None
    effective_collection_type: str | None = None
    retrieved_ids: List[str] = field(default_factory=list)
    distances_summary: Dict[str, Any] = field(default_factory=dict)


def distances_summary(d: List[float]) -> Dict[str, Any]:
    """Compute summary statistics for a list of distances.

    Args:
        d: List of distance values from retrieval.

    Returns:
        Dict with count, best_distance, worst_distance (or None if empty/error).
    """
    if not d:
        return {"count": 0, "best_distance": None}
    try:
        return {
            "count": int(len(d)),
            "best_distance": float(min(d)),
            "worst_distance": float(max(d)),
        }
    except Exception:  # noqa: BLE001
        return {"count": int(len(d)), "best_distance": None}


class RetrievalPassTracker:
    """Tracks retrieval passes for debugging and auditing.

    This encapsulates the _record_pass logic from answer_structured(),
    storing pass info and providing access to retriever state.
    """

    def __init__(self, retriever: "Retriever"):
        self.retriever = retriever
        self.passes: List[Dict[str, Any]] = []

    def record_pass(
        self,
        *,
        pass_name: str,
        planned_where: Dict[str, Any] | None,
        planned_collection_type: str,
    ) -> None:
        """Record a retrieval pass with current retriever state.

        Args:
            pass_name: Name identifying this retrieval pass.
            planned_where: The where clause that was planned.
            planned_collection_type: 'toc' or 'chunk'.
        """
        self.passes.append({
            "pass_name": str(pass_name),
            "planned_where": deepcopy(planned_where) if planned_where is not None else None,
            "planned_collection_type": str(planned_collection_type),
            "effective_where": deepcopy(getattr(self.retriever, "_last_effective_where", None)),
            "effective_collection": getattr(self.retriever, "_last_effective_collection_name", None),
            "effective_collection_type": getattr(self.retriever, "_last_effective_collection_type", None),
            "retrieved_ids": list(getattr(self.retriever, "_last_retrieved_ids", []) or []),
            "distances": distances_summary(list(getattr(self.retriever, "_last_distances", []) or [])),
        })

    def get_passes(self) -> List[Dict[str, Any]]:
        """Return all recorded passes."""
        return list(self.passes)

class Retriever:
    def __init__(self, collection: Any, embedding_model: str):
        self.collection = collection
        self.embedding_model = embedding_model
        
        # Audit state
        self._last_retrieved_ids: List[str] = []
        self._last_retrieved_metadatas: List[Dict[str, Any]] = []
        self._last_effective_where: Dict[str, Any] | None = None
        self._last_effective_collection_name: str | None = None
        self._last_effective_collection_type: str | None = None
        
        # Sibling expansion tracking
        self._last_sibling_expansion: Dict[str, Any] = {}

    def _embed(self, texts: List[str]) -> List[List[float]]:
        client = make_openai_client()
        try:
            response = client.embeddings.create(model=self.embedding_model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001
            raise RAGEngineError("OpenAI embedding request failed.") from exc

    @staticmethod
    def _normalize_chroma_where(where: Dict[str, Any] | None) -> Dict[str, Any] | None:
        """Normalize a where-clause to a single top-level operator for Chroma.

        Some Chroma versions require `where` to contain exactly one operator at the top level.
        This helper rewrites multi-key dicts into a deterministic `$and`.
        It also recursively normalizes items inside $or and $and arrays.

        Examples:
        - {"article": "6"} -> unchanged
        - {"corpus_id": "ai-act", "article": "6"} -> {"$and": [{"corpus_id": "ai-act"}, {"article": "6"}]}
        - {"corpus_id": "ai-act", "$or": [{"article": "2"}, {"article": "3"}]} ->
            {"$and": [{"corpus_id": "ai-act"}, {"$or": [...]}]}
        - {"$or": [{"chapter": "III", "section": "1"}, {"article": "6"}]} ->
            {"$or": [{"$and": [{"chapter": "III"}, {"section": "1"}]}, {"article": "6"}]}
        """

        if where is None:
            return None
        w = dict(where)
        if not w:
            return None

        # Recursively normalize items inside $or and $and arrays.
        for op_key in ("$or", "$and"):
            if op_key in w and isinstance(w[op_key], list):
                w[op_key] = [
                    Retriever._normalize_chroma_where(item) or item
                    for item in w[op_key]
                    if isinstance(item, dict)
                ]

        # Already a single top-level operator.
        if len(w) == 1:
            return w

        op_items: list[tuple[str, Any]] = []
        kv_items: list[tuple[str, Any]] = []
        for k, v in w.items():
            if isinstance(k, str) and k.startswith("$"):
                op_items.append((k, v))
            else:
                kv_items.append((k, v))

        clauses: list[dict[str, Any]] = []
        for k, v in sorted(kv_items, key=lambda t: str(t[0])):
            clauses.append({str(k): v})
        for k, v in sorted(op_items, key=lambda t: str(t[0])):
            clauses.append({str(k): v})

        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def _collection_name_best_effort(self, collection: Any) -> str | None:
        # Helper to extract collection name safely
        try:
            return str(getattr(collection, "name", None) or "")
        except Exception:
            return None

    def _collection_type(self, collection: Any) -> str | None:
        # Helper to extract collection type safely
        try:
            return str(type(collection).__name__)
        except Exception:
            return None

    def _query_collection_raw(
        self,
        *,
        collection: Any,
        question: str,
        k: int,
        where: Dict[str, Any] | None = None,
        track_state: bool = True,
    ) -> tuple[list[str], list[str], list[dict[str, Any]], list[float]]:
        # Normalize filters to Chroma's expected shape (single top-level operator).
        effective_where = self._normalize_chroma_where(where)

        # Capture the effective filter & collection for audit/debug (deep-copied).
        # Only track state for primary queries, not injection/expansion queries.
        if track_state:
            self._last_effective_where = deepcopy(effective_where) if effective_where is not None else None
            self._last_effective_collection_name = self._collection_name_best_effort(collection)
            self._last_effective_collection_type = self._collection_type(collection)

        query_embedding = self._embed([question])[0]
        kwargs: Dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": k}
        if effective_where is not None:
            kwargs["where"] = effective_where
        # Chroma always returns ids; 'ids' is not a valid `include` item.
        kwargs["include"] = ["documents", "metadatas", "distances"]
        results = collection.query(**kwargs)

        ids = results.get("ids", [[]])[0] if isinstance(results, dict) else []
        documents = results.get("documents", [[]])[0] if isinstance(results, dict) else []
        metadatas = results.get("metadatas", [[]])[0] if isinstance(results, dict) else []
        distances = results.get("distances", [[]])[0] if isinstance(results, dict) else []

        safe_ids = [str(item) for item in (ids or [])]
        safe_docs = [str(d or "") for d in (documents or [])]
        # Normalize metadata at read-time to eliminate case-sensitivity issues
        safe_metas = [helpers.normalize_metadata(m) for m in (metadatas or [])]

        safe_distances: list[float] = []
        for d in distances or []:
            try:
                safe_distances.append(float(d))
            except Exception:  # noqa: BLE001
                safe_distances.append(1.0)

        return safe_ids, safe_docs, safe_metas, safe_distances

    def _query_collection_with_distances(
        self,
        *,
        collection: Any,
        question: str,
        k: int,
        where: Dict[str, Any] | None = None,
        expand_siblings: bool | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[float]]:
        ids, documents, metadatas, safe_distances = self._query_collection_raw(
            collection=collection, question=question, k=k, where=where
        )
        
        original_count = len(ids)
        
        # Apply sibling expansion if enabled
        sibling_settings = get_sibling_expansion_settings()
        should_expand = expand_siblings if expand_siblings is not None else sibling_settings.get("enabled", False)
        
        if should_expand and ids:
            max_siblings = sibling_settings.get("max_siblings", 2)
            ids, documents, metadatas, safe_distances = self._expand_to_siblings(
                collection=collection,
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                distances=safe_distances,
                max_siblings=max_siblings,
            )
            # Track sibling expansion
            self._last_sibling_expansion = {
                "enabled": True,
                "original_count": original_count,
                "expanded_count": len(ids),
                "siblings_added": len(ids) - original_count,
                "max_siblings": max_siblings,
            }
        else:
            self._last_sibling_expansion = {
                "enabled": False,
                "original_count": original_count,
                "expanded_count": original_count,
                "siblings_added": 0,
            }
        
        self._last_retrieved_ids = [str(item) for item in (ids or [])]
        # Normalize metadata at read-time to eliminate case-sensitivity issues
        self._last_retrieved_metadatas = [helpers.normalize_metadata(meta) for meta in (metadatas or [])]
        return list(zip(documents, metadatas)), safe_distances

    def _query_collection(self, *, collection: Any, question: str, k: int, where: Dict[str, Any] | None = None):
        hits, _ = self._query_collection_with_distances(collection=collection, question=question, k=k, where=where)
        return hits

    def _expand_to_siblings(
        self,
        *,
        collection: Any,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        distances: List[float],
        max_siblings: int = 2,
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[float]]:
        """Expand retrieved chunks by fetching sibling chunks with the same location_id.

        For each unique location_id in the retrieved set, fetches additional chunks
        with the same location_id, ordered by chunk_index. This provides more complete
        context from the same structural unit (e.g., full article content).

        Args:
            collection: The Chroma collection to query.
            ids: Original retrieved chunk IDs.
            documents: Original retrieved documents.
            metadatas: Original retrieved metadata dicts.
            distances: Original distances.
            max_siblings: Maximum additional sibling chunks per location_id.

        Returns:
            Tuple of (expanded_ids, expanded_docs, expanded_metas, expanded_distances).
            Siblings are inserted adjacent to their original chunks, preserving relevance order.
        """
        if not ids or max_siblings <= 0:
            return ids, documents, metadatas, distances

        # Collect unique location_ids and their positions
        seen_ids: Set[str] = set(ids)
        location_to_first_idx: Dict[str, int] = {}
        location_to_distance: Dict[str, float] = {}

        for idx, meta in enumerate(metadatas):
            loc_id = (meta or {}).get("location_id", "")
            if loc_id and loc_id not in location_to_first_idx:
                location_to_first_idx[loc_id] = idx
                location_to_distance[loc_id] = distances[idx] if idx < len(distances) else 1.0

        if not location_to_first_idx:
            return ids, documents, metadatas, distances

        # Query for sibling chunks for each unique location_id
        siblings_by_location: Dict[str, List[Tuple[str, str, Dict[str, Any], float]]] = {}

        for loc_id, original_distance in location_to_distance.items():
            try:
                # Fetch chunks with the same location_id
                where_clause = {"location_id": loc_id}
                # Get a few more than max_siblings to account for the original chunk
                fetch_count = max_siblings + 3
                
                results = collection.get(
                    where=where_clause,
                    include=["documents", "metadatas"],
                    limit=fetch_count,
                )

                result_ids = results.get("ids", []) or []
                result_docs = results.get("documents", []) or []
                # Normalize sibling metadata at read-time
                result_metas = [helpers.normalize_metadata(m) for m in (results.get("metadatas", []) or [])]

                # Collect candidates excluding already-retrieved chunks
                candidates: List[Tuple[int, str, str, Dict[str, Any]]] = []
                for i, cid in enumerate(result_ids):
                    if cid not in seen_ids:
                        chunk_idx = result_metas[i].get("chunk_index", 0)
                        candidates.append((chunk_idx, cid, result_docs[i], result_metas[i]))

                # Sort by chunk_index and take max_siblings
                candidates.sort(key=lambda x: x[0])
                siblings_for_loc: List[Tuple[str, str, Dict[str, Any], float]] = []
                for chunk_idx, cid, doc, meta in candidates[:max_siblings]:
                    # Assign a slightly worse distance to siblings (they're contextual, not directly relevant)
                    sibling_distance = original_distance * 1.05  # 5% penalty
                    siblings_for_loc.append((cid, doc, meta, sibling_distance))
                    seen_ids.add(cid)

                if siblings_for_loc:
                    siblings_by_location[loc_id] = siblings_for_loc

            except Exception:  # noqa: BLE001
                # If sibling expansion fails for a location, skip it
                continue

        if not siblings_by_location:
            return ids, documents, metadatas, distances

        # Build expanded results, inserting siblings after their original chunks
        expanded_ids: List[str] = []
        expanded_docs: List[str] = []
        expanded_metas: List[Dict[str, Any]] = []
        expanded_distances: List[float] = []

        for idx, (cid, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
            # Add the original chunk
            expanded_ids.append(cid)
            expanded_docs.append(doc)
            expanded_metas.append(meta)
            expanded_distances.append(dist)

            # If this is the first chunk for its location, add siblings after it
            loc_id = (meta or {}).get("location_id", "")
            if loc_id in siblings_by_location and location_to_first_idx.get(loc_id) == idx:
                for sib_id, sib_doc, sib_meta, sib_dist in siblings_by_location[loc_id]:
                    expanded_ids.append(sib_id)
                    expanded_docs.append(sib_doc)
                    expanded_metas.append(sib_meta)
                    expanded_distances.append(sib_dist)

        return expanded_ids, expanded_docs, expanded_metas, expanded_distances

    def _retrieve_representative_chapter_chunks(
        self,
        *,
        question: str,
        where: Dict[str, Any],
        top_k: int,
        per_article: int = 2,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[float]]:
        # Pull a larger candidate set then diversify by article.
        candidate_k = max(20, top_k * 4)
        ids, docs, metas, distances = self._query_collection_raw(
            collection=self.collection,
            question=question,
            k=candidate_k,
            where=where,
        )
        if not docs:
            self._last_retrieved_ids = []
            self._last_retrieved_metadatas = []
            return [], []

        out_hits: list[tuple[str, dict[str, Any]]] = []
        out_distances: list[float] = []
        out_ids: list[str] = []
        out_metas: list[dict[str, Any]] = []
        by_article: dict[str, int] = {}
        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            art = str((meta or {}).get("article") or "").strip().upper()
            if art:
                used = by_article.get(art, 0)
                if used >= per_article:
                    continue
                by_article[art] = used + 1
            out_hits.append((doc, meta))
            out_distances.append(dist)
            out_ids.append(str(chunk_id))
            out_metas.append(dict(meta or {}))
            if len(out_hits) >= top_k:
                break

        # Align last retrieved IDs/metas with the selected set (not the candidate pool).
        self._last_retrieved_ids = out_ids
        self._last_retrieved_metadatas = out_metas
        return out_hits, out_distances

    @staticmethod
    def split_precise(
        hits_list: List[Tuple[str, Dict[str, Any]]],
        dists: List[float],
        ids_list: List[str] | None = None,
    ) -> Tuple[
        List[Tuple[str, Dict[str, Any]]],  # precise hits
        List[float],                        # precise distances
        List[str],                          # precise ids
        List[Tuple[str, Dict[str, Any]]],  # imprecise hits
        List[float],                        # imprecise distances
        List[str],                          # imprecise ids
    ]:
        """Split hits into precise (has article/annex/chapter) vs imprecise."""
        precise = []
        precise_d = []
        precise_ids = []
        imprecise = []
        imprecise_d = []
        imprecise_ids = []
        for i, (doc, meta) in enumerate(hits_list):
            m = dict(meta or {})
            is_precise = bool(m.get("article") or m.get("annex") or m.get("chapter"))
            did = dists[i] if i < len(dists) else None
            cid = None
            if ids_list is not None and i < len(ids_list):
                cid = ids_list[i]
            else:
                cid = m.get("chunk_id") or m.get("doc_id") or f"hit-{i+1}"
            if is_precise:
                precise.append((doc, meta))
                precise_d.append(did)
                precise_ids.append(cid)
            else:
                imprecise.append((doc, meta))
                imprecise_d.append(did)
                imprecise_ids.append(cid)
        return precise, precise_d, precise_ids, imprecise, imprecise_d, imprecise_ids

    @staticmethod
    def split_precise_simple(
        hits_list: List[Tuple[str, Dict[str, Any]]],
        dists: List[float],
    ) -> Tuple[
        List[Tuple[str, Dict[str, Any]]],  # precise hits
        List[float],                        # precise distances
        List[str],                          # precise ids
        List[Tuple[str, Dict[str, Any]]],  # imprecise hits
        List[float],                        # imprecise distances
        List[str],                          # imprecise ids
    ]:
        """Split hits into precise (has article/annex/chapter) vs imprecise (simplified version without ids_list)."""
        precise = []
        precise_d = []
        precise_ids = []
        imprecise = []
        imprecise_d = []
        imprecise_ids = []
        for i, (doc, meta) in enumerate(hits_list):
            m = dict(meta or {})
            is_precise = bool(m.get("article") or m.get("annex") or m.get("chapter"))
            did = dists[i] if i < len(dists) else None
            cid = m.get("chunk_id") or m.get("doc_id") or f"hit-{i+1}"
            if is_precise:
                precise.append((doc, meta))
                precise_d.append(did)
                precise_ids.append(cid)
            else:
                imprecise.append((doc, meta))
                imprecise_d.append(did)
                imprecise_ids.append(cid)
        return precise, precise_d, precise_ids, imprecise, imprecise_d, imprecise_ids


# ---------------------------------------------------------------------------
# Multi-anchor retrieval helper (extracted from RAGEngine.answer_structured)
# ---------------------------------------------------------------------------

@dataclass
class MultiAnchorResult:
    """Result of multi-anchor retrieval."""
    hits: List[Tuple[str, Dict[str, Any]]]
    distances: List[float]
    retrieved_ids: List[str]
    retrieved_metadatas: List[Dict[str, Any]]


def execute_multi_anchor_retrieval(
    *,
    query_fn,
    question: str,
    corpus_id: str,
    explicit_article_refs: List[str],
    explicit_annex_refs: List[str],
    top_k: int,
) -> MultiAnchorResult:
    """Execute multi-anchor retrieval for questions mentioning multiple articles/annexes.

    When the user explicitly asks to bind multiple Articles/Annexes, this does a deterministic
    multi-scope retrieval and merges results. This avoids brittle single-scope narrowing.

    Args:
        query_fn: A callable (question, k, where) -> (hits, distances, ids, metadatas)
                  that performs the query.
        question: The user's question.
        corpus_id: The corpus ID for filtering.
        explicit_article_refs: List of article references extracted from the question.
        explicit_annex_refs: List of annex references extracted from the question.
        top_k: Target number of hits to return.

    Returns:
        MultiAnchorResult with merged and deduplicated hits.
    """
    import math

    scopes: List[Dict[str, Any]] = []
    for a in list(explicit_article_refs or []):
        scopes.append({"corpus_id": corpus_id, "article": str(a)})
    for ax in list(explicit_annex_refs or []):
        scopes.append({"corpus_id": corpus_id, "annex": helpers.normalize_annex_for_chroma(ax)})

    if not scopes:
        return MultiAnchorResult(hits=[], distances=[], retrieved_ids=[], retrieved_metadatas=[])

    k_total = int(top_k)
    k_each = max(2, int(math.ceil(k_total / max(1, len(scopes)))))

    # Track chunks per scope to ensure diversity
    scope_chunks: Dict[int, List[Tuple[str, Dict[str, Any], float, str]]] = {}
    merged: Dict[str, Tuple[str, Dict[str, Any], float, str, int]] = {}
    # value: (doc, meta, distance, chroma_id, scope_index)

    for scope_idx, sc in enumerate(scopes):
        scope_chunks[scope_idx] = []
        h_sc, d_sc, ids_sc, metas_sc = query_fn(question=question, k=k_each, where=sc)

        for i, (doc, meta) in enumerate(list(h_sc or [])):
            m = dict(meta or {})
            chunk_id = str(m.get("chunk_id") or "").strip() or str(ids_sc[i] if i < len(ids_sc) else "").strip()
            if not chunk_id:
                continue
            dist = float(d_sc[i]) if i < len(d_sc) and d_sc[i] is not None else 1.0
            cid = str(ids_sc[i]) if i < len(ids_sc) else chunk_id
            chunk_data = (str(doc or ""), m, dist, cid)
            scope_chunks[scope_idx].append(chunk_data)
            prev = merged.get(chunk_id)
            if prev is None or dist < float(prev[2]):
                merged[chunk_id] = (str(doc or ""), m, dist, cid, scope_idx)

    # Diversity-preserving merge: ensure at least min_per_scope from each scope
    min_per_scope = max(1, k_total // (len(scopes) + 1))  # At least 1 per scope
    final_list: List[Tuple[str, Tuple[str, Dict[str, Any], float, str]]] = []
    scope_counts: Dict[int, int] = {i: 0 for i in range(len(scopes))}
    used_ids: Set[str] = set()

    # First pass: add min_per_scope from each scope (best distance per scope)
    for scope_idx in range(len(scopes)):
        for chunk_data in scope_chunks.get(scope_idx, []):
            if len(final_list) >= k_total:
                break
            chunk_id = chunk_data[3]  # cid
            if chunk_id in used_ids:
                continue
            if scope_counts[scope_idx] >= min_per_scope:
                break
            used_ids.add(chunk_id)
            final_list.append((chunk_id, chunk_data))
            scope_counts[scope_idx] += 1

    # Second pass: fill remaining slots with best overall
    remaining = sorted(merged.items(), key=lambda kv: (float(kv[1][2]), kv[0]))
    for chunk_id, data in remaining:
        if len(final_list) >= k_total:
            break
        if chunk_id in used_ids:
            continue
        used_ids.add(chunk_id)
        final_list.append((chunk_id, (data[0], data[1], data[2], data[3])))

    # Sort final list by distance
    final_list.sort(key=lambda kv: (float(kv[1][2]), kv[0]))

    hits = [(v[0], v[1]) for _k, v in final_list]
    distances = [float(v[2]) for _k, v in final_list]
    retrieved_ids = [str(v[3]) for _k, v in final_list]
    retrieved_metas = [dict(v[1] or {}) for _k, v in final_list]

    return MultiAnchorResult(
        hits=hits,
        distances=distances,
        retrieved_ids=retrieved_ids,
        retrieved_metadatas=retrieved_metas,
    )


# ---------------------------------------------------------------------------
# Retrieval Pipeline Result (extracted from RAGEngine.answer_structured)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalPipelineResult:
    """Complete result of the retrieval pipeline."""
    hits: List[Tuple[str, Dict[str, Any]]]
    distances: List[float]
    retrieved_ids: List[str]
    retrieved_metadatas: List[Dict[str, Any]]
    passes: List[Dict[str, Any]]
    debug: Dict[str, Any]
    # Planning info
    final_planned_where: Dict[str, Any] | None
    final_planned_collection_type: str


def execute_engineering_citable_prefer_pass(
    *,
    query_fn,
    question: str,
    where_for_retrieval: Dict[str, Any] | None,
    current_hits: List[Tuple[str, Dict[str, Any]]],
    top_k: int,
    is_citable_fn,
    pass_tracker: RetrievalPassTracker,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[float], Dict[str, Any]]:
    """Execute ENGINEERING citable-prefer second retrieval pass.

    When initial retrieval doesn't have enough citable chunks,
    this runs a second pass preferring citable=true chunks.

    Args:
        query_fn: Function (question, k, where) -> (hits, distances, ids, metas)
        question: The user's question.
        where_for_retrieval: Base where clause.
        current_hits: Current hits from first pass.
        top_k: Base top_k value.
        is_citable_fn: Function (meta, doc) -> (is_citable, reason)
        pass_tracker: Tracker for recording passes.

    Returns:
        Tuple of (new_hits, new_distances, debug_info)
    """
    debug: Dict[str, Any] = {"triggered": False}

    # Count current citable chunks
    def _count_citable(h: List[Tuple[str, Dict[str, Any]]]) -> int:
        c = 0
        for d, m in h:
            ok, _ = is_citable_fn(dict(m or {}), str(d or ""))
            if ok:
                c += 1
        return c

    initial_citable = _count_citable(current_hits)
    min_citable_required = 2  # Could be parameterized

    if initial_citable >= min_citable_required:
        debug["reason"] = "sufficient_citable"
        debug["initial_citable"] = initial_citable
        return current_hits, [], debug

    debug["triggered"] = True
    debug["initial_citable"] = initial_citable
    debug["min_required"] = min_citable_required

    k2 = max(top_k * 4, min_citable_required * 6, 12)
    where2: Dict[str, Any] | None = dict(where_for_retrieval or {})

    # Prefer citable-only chunks
    if where2 is not None:
        where2["citable"] = True
    else:
        where2 = {"citable": True}

    try:
        hits2, dists2, ids2, metas2 = query_fn(question=question, k=k2, where=where2)
        if hits2:
            pass_tracker.record_pass(
                pass_name="engineering_citable_prefer",
                planned_where=where2,
                planned_collection_type="chunk",
            )
            debug["pass_name"] = "engineering_citable_prefer"
            debug["new_hits_count"] = len(hits2)
            return hits2, dists2, debug
    except Exception:  # noqa: BLE001
        # Fallback to larger k without citable filter
        try:
            hits2, dists2, ids2, metas2 = query_fn(question=question, k=k2, where=where_for_retrieval)
            if hits2:
                pass_tracker.record_pass(
                    pass_name="engineering_citable_fallback",
                    planned_where=where_for_retrieval,
                    planned_collection_type="chunk",
                )
                debug["pass_name"] = "engineering_citable_fallback"
                debug["new_hits_count"] = len(hits2)
                return hits2, dists2, debug
        except Exception:  # noqa: BLE001
            pass

    debug["fallback_failed"] = True
    return current_hits, [], debug


def execute_anchor_hint_injection(
    *,
    query_raw_fn,
    question: str,
    corpus_id: str,
    hint_anchors: Set[str],
    current_hits: List[Tuple[str, Dict[str, Any]]],
    current_distances: List[float],
    current_ids: List[str],
    current_metas: List[Dict[str, Any]],
    top_k: int,
    max_anchors: int = 10,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[float], List[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Inject additional candidates by querying hinted anchors.

    This helps when TOC-scoped retrieval misses expected anchors entirely.

    Args:
        query_raw_fn: Function (collection, question, k, where) -> (ids, docs, metas, distances)
        question: The user's question.
        corpus_id: Corpus ID for filtering.
        hint_anchors: Set of anchor hints like "article:6", "annex:iii".
        current_hits: Current hits.
        current_distances: Current distances.
        current_ids: Current chunk IDs.
        current_metas: Current metadatas.
        top_k: Top k for each anchor query.
        max_anchors: Maximum anchors to process.

    Returns:
        Tuple of (hits, distances, ids, metas, debug_info)
    """
    debug: Dict[str, Any] = {
        "injected_queries": 0,
        "injected_added": 0,
        "injected_query_specs": [],
    }

    if not hint_anchors:
        return current_hits, current_distances, current_ids, current_metas, debug

    # Work with copies
    hits = list(current_hits)
    distances = list(current_distances)
    retrieved_ids = list(current_ids)
    retrieved_metas = list(current_metas)

    seen_ids = set(str(x) for x in retrieved_ids)
    anchors_sorted = sorted({a for a in hint_anchors if isinstance(a, str) and a.strip()})[:max_anchors]

    for anchor in anchors_sorted:
        if ":" not in anchor:
            continue
        kind, value = anchor.split(":", 1)
        kind = str(kind).strip().lower()
        value = str(value).strip()
        if kind not in {"article", "recital", "annex"} or not value:
            continue

        value_variants = sorted({value, value.lower(), value.upper()})
        debug["injected_query_specs"].append({
            "kind": str(kind),
            "value": str(value),
            "value_variants": list(value_variants),
        })

        for vv in value_variants:
            where_hint: Dict[str, Any] = {"corpus_id": corpus_id, kind: vv}
            debug["injected_queries"] += 1

            try:
                ids_h, docs_h, metas_h, dist_h = query_raw_fn(
                    question=question,
                    k=min(max(top_k, 3), 10),
                    where=where_hint,
                )

                for hid, hdoc, hmeta, hd in zip(ids_h, docs_h, metas_h, dist_h, strict=False):
                    sid = str(hid)
                    if not sid or sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    retrieved_ids.append(sid)
                    retrieved_metas.append(dict(hmeta or {}))
                    hits.append((str(hdoc or ""), dict(hmeta or {})))
                    distances.append(float(hd))
                    debug["injected_added"] += 1
            except Exception:  # noqa: BLE001
                pass

    return hits, distances, retrieved_ids, retrieved_metas, debug


@dataclass
class AnchorRescueResult:
    """Result of anchor rescue operation."""

    hits: List[Tuple[str, Dict[str, Any]]]
    distances: List[float]
    retrieved_ids: List[str]
    retrieved_metas: List[Dict[str, Any]]
    debug: Dict[str, Any]


def _norm_anchor(a: str) -> str:
    """Normalize anchor string for comparison."""
    import re
    return re.sub(r"\s+", "", str(a or "")).strip().lower()


def execute_anchor_rescue(
    *,
    required_anchors_payload: Dict[str, Any],
    hits: List[Tuple[str, Dict[str, Any]]],
    distances: List[float],
    retrieved_ids: List[str],
    retrieved_metas: List[Dict[str, Any]],
    question: str,
    corpus_id: str,
    top_k: int,
    planned_where: Dict[str, Any] | None,
    query_with_where_fn,
    query_collection_raw_fn,
    collection,
    normalize_anchor_list_fn,
    anchors_from_metadata_fn,
) -> AnchorRescueResult:
    """Execute anchor rescue to ensure required anchors are present in candidates.

    This expands the candidate pool and/or injects anchor-filtered hits for missing
    required anchors. Does NOT change ranking.

    Args:
        required_anchors_payload: Dict with must_include_any_of, must_include_any_of_2, must_include_all_of.
        hits: Current hits.
        distances: Current distances.
        retrieved_ids: Current chunk IDs.
        retrieved_metas: Current metadatas.
        question: User question.
        corpus_id: Corpus ID.
        top_k: Top k value.
        planned_where: WHERE clause for expansion query.
        query_with_where_fn: Function(question, k, where) -> hits.
        query_collection_raw_fn: Function(collection, question, k, where) -> (ids, docs, metas, distances).
        collection: Collection object.
        normalize_anchor_list_fn: Function to normalize anchor lists.
        anchors_from_metadata_fn: Function to extract anchors from metadata.

    Returns:
        AnchorRescueResult with updated hits, distances, ids, metas, and debug info.
    """
    debug: Dict[str, Any] = {
        "enabled": True,
        "candidate_pool_size": len(hits),
        "anchors_in_candidate_pool": {},
        "missing_required_anchor_any_of": [],
        "missing_required_anchor_any_of_2": [],
        "missing_required_anchor_all_of": [],
        "rescue_slots_added": 0,
        "rescue_injected_anchors": [],
        "rescue_injection_source": None,
        "rescue_added_refs_count": 0,
    }

    if not required_anchors_payload or not hits:
        return AnchorRescueResult(
            hits=hits,
            distances=distances,
            retrieved_ids=retrieved_ids,
            retrieved_metas=retrieved_metas,
            debug=debug,
        )

    # Work with copies
    hits = list(hits)
    distances = list(distances)
    retrieved_ids = list(retrieved_ids)
    retrieved_metas = list(retrieved_metas)

    try:
        required_any_1 = set(normalize_anchor_list_fn(required_anchors_payload.get("must_include_any_of"), require_colon=True))
        required_any_2 = set(normalize_anchor_list_fn(required_anchors_payload.get("must_include_any_of_2"), require_colon=True))
        required_all = set(normalize_anchor_list_fn(required_anchors_payload.get("must_include_all_of"), require_colon=True))
        required_all_union = set().union(required_any_1, required_any_2, required_all)

        def _compute_missing(anchors_present: set) -> tuple:
            missing_any_1: List[str] = []
            missing_any_2: List[str] = []
            missing_all: List[str] = []
            if required_any_1 and not (required_any_1 & anchors_present):
                missing_any_1 = sorted(required_any_1)
            if required_any_2 and not (required_any_2 & anchors_present):
                missing_any_2 = sorted(required_any_2)
            if required_all:
                missing_all = sorted(required_all - anchors_present)
            return missing_any_1, missing_any_2, missing_all

        # Candidate pool summary (before rescue).
        if not retrieved_metas:
            retrieved_metas = [dict(m or {}) for _d, m in hits]
        if not retrieved_ids:
            retrieved_ids = [
                str((m or {}).get("chunk_id") or f"hit-{i}") for i, (_d, m) in enumerate(hits)
            ]

        n0 = min(len(retrieved_metas), len(retrieved_ids), len(hits))
        anchors_present_0: set = set()
        positions_0: Dict[str, List[int]] = {}
        for pos, meta in enumerate(retrieved_metas[:n0], start=1):
            for a in (anchors_from_metadata_fn(meta) & required_all_union):
                anchors_present_0.add(a)
                positions_0.setdefault(a, []).append(int(pos))

        missing_any_1, missing_any_2, missing_all = _compute_missing(anchors_present_0)
        debug["candidate_pool_size"] = len(hits)
        debug["anchors_in_candidate_pool"] = {k: v for k, v in sorted(positions_0.items())}
        debug["missing_required_anchor_any_of"] = list(missing_any_1)
        debug["missing_required_anchor_any_of_2"] = list(missing_any_2)
        debug["missing_required_anchor_all_of"] = list(missing_all)

        # STEP 2: expand candidate pool deterministically when any required anchors are missing.
        rescue_slots_added = 0
        if missing_any_1 or missing_any_2 or missing_all:
            rescue_slots_added = 2
            try:
                k_expand = max(top_k * 10, len(hits), 30)
            except Exception:  # noqa: BLE001
                k_expand = max(30, len(hits))

            try:
                hits2 = query_with_where_fn(question, k=k_expand, where=planned_where)
            except Exception:  # noqa: BLE001
                hits2 = None

            if hits2 and isinstance(hits2, list) and len(hits2) > len(hits):
                # We need to get the new ids/metas/distances from the query
                # This is a simplified version - the caller should provide these
                seen_ids = set(str(x) for x in retrieved_ids)
                # For now, just extend with new hits (distances approximated)
                for i, (doc, meta) in enumerate(hits2):
                    sid = str((meta or {}).get("chunk_id") or f"hit2-{i}")
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)
                    hits.append((str(doc or ""), dict(meta or {})))
                    distances.append(1.0)  # Placeholder distance
                    retrieved_ids.append(sid)
                    retrieved_metas.append(dict(meta or {}))

        # Recompute missing after expansion.
        n1 = min(len(retrieved_metas), len(retrieved_ids), len(hits))
        anchors_present_1: set = set()
        positions_1: Dict[str, List[int]] = {}
        for pos, meta in enumerate(retrieved_metas[:n1], start=1):
            for a in (anchors_from_metadata_fn(meta) & required_all_union):
                anchors_present_1.add(a)
                positions_1.setdefault(a, []).append(int(pos))

        missing_any_1, missing_any_2, missing_all = _compute_missing(anchors_present_1)
        debug["candidate_pool_size"] = len(hits)
        debug["anchors_in_candidate_pool"] = {k: v for k, v in sorted(positions_1.items())}
        debug["missing_required_anchor_any_of"] = list(missing_any_1)
        debug["missing_required_anchor_any_of_2"] = list(missing_any_2)
        debug["missing_required_anchor_all_of"] = list(missing_all)
        debug["rescue_slots_added"] = rescue_slots_added

        # STEP 3: inject anchor-filtered chunks for any still-missing anchors.
        injected_anchors: List[str] = []
        injected_added = 0
        if (missing_any_1 or missing_any_2 or missing_all) and query_collection_raw_fn is not None:
            seen_ids = set(str(x) for x in retrieved_ids)

            # Choose concrete anchors to rescue deterministically.
            to_rescue: List[str] = []
            if missing_any_1:
                to_rescue.append(sorted(list(required_any_1))[0])
            if missing_any_2:
                to_rescue.append(sorted(list(required_any_2))[0])
            to_rescue.extend(list(missing_all))
            to_rescue = list(dict.fromkeys([_norm_anchor(a) for a in to_rescue if ":" in str(a)]))

            for anchor in to_rescue:
                try:
                    kind, value = anchor.split(":", 1)
                    kind = str(kind).strip().lower()
                    value = str(value).strip()
                    if kind not in {"article", "recital", "annex"} or not value:
                        continue

                    value_variants = sorted({value, value.lower(), value.upper()})
                    added_for_anchor = 0
                    for vv in value_variants:
                        where_hint: Dict[str, Any] = {"corpus_id": corpus_id, kind: vv}
                        ids_h, docs_h, metas_h, dist_h = query_collection_raw_fn(
                            collection=collection,
                            question=question,
                            k=min(max(top_k, 3), 10),
                            where=where_hint,
                        )
                        for hid, hdoc, hmeta, hd in zip(ids_h, docs_h, metas_h, dist_h, strict=False):
                            sid = str(hid)
                            if not sid or sid in seen_ids:
                                continue
                            seen_ids.add(sid)
                            hits.append((str(hdoc or ""), dict(hmeta or {})))
                            distances.append(float(hd))
                            retrieved_ids.append(sid)
                            retrieved_metas.append(dict(hmeta or {}))
                            injected_added += 1
                            added_for_anchor += 1
                            if added_for_anchor >= 1:
                                break
                        if added_for_anchor >= 1:
                            break

                    if added_for_anchor >= 1:
                        injected_anchors.append(anchor)
                except Exception:  # noqa: BLE001
                    continue

        debug["rescue_injected_anchors"] = list(sorted(set(injected_anchors)))
        debug["rescue_injection_source"] = "anchor_lookup" if injected_anchors else None
        debug["rescue_added_refs_count"] = injected_added

    except Exception:  # noqa: BLE001
        # Never fail the query due to optional rescue.
        pass

    return AnchorRescueResult(
        hits=hits,
        distances=distances,
        retrieved_ids=retrieved_ids,
        retrieved_metas=retrieved_metas,
        debug=debug,
    )
