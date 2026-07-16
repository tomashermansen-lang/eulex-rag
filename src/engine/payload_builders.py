"""Response payload assembly utilities.

Pure functions for building standardized answer response payloads,
retrieval state dicts, and hybrid rerank debug dicts.

Extracted from helpers.py (Phase 8b) — single responsibility.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

from .metadata_helpers import normalize_anchor


def compute_required_anchor_idxs(
    *,
    required_anchors_payload: Dict[str, Any],
    references_structured_all: List[Dict[str, Any]],
) -> Tuple[set, List[str]]:
    """Compute which reference indices are required based on required_anchors_payload.

    Returns:
        (required_idxs, missing_anchor_keys) - set of required idx integers and list of missing anchors.
    """
    anchor_to_idxs: Dict[str, List[int]] = {}
    for r in list(references_structured_all or []):
        if not isinstance(r, dict):
            continue
        try:
            ridx = int(r.get("idx") or 0)
        except Exception:  # noqa: BLE001
            continue
        if ridx <= 0:
            continue
        if r.get("article"):
            k = normalize_anchor(f"article:{str(r.get('article')).strip()}")
            anchor_to_idxs.setdefault(k, []).append(ridx)
        if r.get("recital"):
            k = normalize_anchor(f"recital:{str(r.get('recital')).strip()}")
            anchor_to_idxs.setdefault(k, []).append(ridx)
        if r.get("annex"):
            k = normalize_anchor(f"annex:{str(r.get('annex')).strip()}")
            anchor_to_idxs.setdefault(k, []).append(ridx)

    for k in list(anchor_to_idxs.keys()):
        anchor_to_idxs[k] = sorted(set(anchor_to_idxs[k]))

    req_any_1 = [
        normalize_anchor(a)
        for a in list(required_anchors_payload.get("must_include_any_of") or [])
        if isinstance(a, str) and a.strip() and ":" in a
    ]
    req_any_2 = [
        normalize_anchor(a)
        for a in list(required_anchors_payload.get("must_include_any_of_2") or [])
        if isinstance(a, str) and a.strip() and ":" in a
    ]
    req_all = [
        normalize_anchor(a)
        for a in list(required_anchors_payload.get("must_include_all_of") or [])
        if isinstance(a, str) and a.strip() and ":" in a
    ]

    required_idxs: set = set()
    missing_keys: List[str] = []

    # any-of: pick the lowest idx among present anchors deterministically.
    if req_any_1:
        present = [
            (min(anchor_to_idxs[a]), a) for a in req_any_1 if a in anchor_to_idxs
        ]
        if present:
            present.sort(key=lambda t: (t[0], t[1]))
            required_idxs.add(int(present[0][0]))
        else:
            missing_keys.extend(sorted(set(req_any_1)))

    if req_any_2:
        present = [
            (min(anchor_to_idxs[a]), a) for a in req_any_2 if a in anchor_to_idxs
        ]
        if present:
            present.sort(key=lambda t: (t[0], t[1]))
            required_idxs.add(int(present[0][0]))
        else:
            missing_keys.extend(sorted(set(req_any_2)))

    # all-of: require each anchor (use lowest idx if multiple).
    for a in req_all:
        if a in anchor_to_idxs:
            required_idxs.add(int(anchor_to_idxs[a][0]))
        else:
            missing_keys.append(a)

    return required_idxs, sorted(set(missing_keys))


def build_answer_response_payload(
    *,
    run_meta: dict[str, Any],
    user_profile_value: str,
    focus: Any | None,
    intent_value: str,
    answer_text: str,
    references: list[dict[str, Any]],
    reference_lines: list[str],
    distances: list[float],
    retrieval_state: dict[str, Any],
    effective_plan: Any,
    where_for_retrieval: dict[str, Any] | None,
    pass_tracker_passes: list[dict[str, Any]],
    # Optional extras
    dry_run: bool = False,
    prompt: str | None = None,
    planner: dict[str, Any] | None = None,
    references_structured_all: list[dict[str, Any]] | None = None,
    used_chunk_ids: list[str] | None = None,
    hybrid_rerank: dict[str, Any] | None = None,
    sibling_expansion: dict[str, Any] | None = None,
    ranking_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build standardized answer response payload.

    This consolidates the repeated payload construction pattern from answer_structured().
    """
    # Build focus dict if focus object provided
    focus_dict = None
    if focus is not None:
        focus_dict = {
            "type": getattr(focus, "type", None) and focus.type.value,
            "node_id": getattr(focus, "node_id", None),
            "title": getattr(focus, "title", None),
            "chapter": getattr(focus, "chapter", None),
            "section": getattr(focus, "section", None),
            "article": getattr(focus, "article", None),
            "annex": getattr(focus, "annex", None),
            "recital": getattr(focus, "recital", None),
        }

    # Build plan dict
    plan_dict = {
        "intent": getattr(effective_plan, "intent", None)
        and effective_plan.intent.value,
        "top_k": getattr(effective_plan, "top_k", None),
        "where": where_for_retrieval,
        "allow_low_evidence_answer": getattr(
            effective_plan, "allow_low_evidence_answer", True
        ),
    }

    # Build retrieval dict
    retrieval_dict: dict[str, Any] = {
        "distances": distances,
        "query_collection": retrieval_state.get("query_collection"),
        "query_where": retrieval_state.get("query_where"),
        "planned_where": deepcopy(retrieval_state.get("planned_where"))
        if retrieval_state.get("planned_where") is not None
        else None,
        "effective_where": deepcopy(retrieval_state.get("effective_where"))
        if retrieval_state.get("effective_where") is not None
        else None,
        "planned_collection_type": str(
            retrieval_state.get("planned_collection_type", "chunk")
        ),
        "effective_collection": retrieval_state.get("effective_collection"),
        "effective_collection_type": retrieval_state.get("effective_collection_type"),
        "passes": pass_tracker_passes,
        "retrieved_ids": list(retrieval_state.get("retrieved_ids") or []),
        "retrieved_metadatas": list(retrieval_state.get("retrieved_metadatas") or []),
        "plan": plan_dict,
    }

    # Add optional retrieval fields
    if ranking_debug is not None:
        retrieval_dict["ranking_debug"] = ranking_debug
    if planner is not None:
        retrieval_dict["planner"] = planner
    if references_structured_all is not None:
        retrieval_dict["references_structured_all"] = list(references_structured_all)
    if used_chunk_ids is not None:
        retrieval_dict["references_used_in_answer"] = list(used_chunk_ids)
    if hybrid_rerank is not None:
        retrieval_dict["hybrid_rerank"] = hybrid_rerank
    if sibling_expansion is not None:
        retrieval_dict["sibling_expansion"] = sibling_expansion

    # Build base response
    response: dict[str, Any] = {
        "run": run_meta,
        "user_profile": user_profile_value,
        "focus": focus_dict,
        "intent": intent_value,
        "answer": answer_text,
        "references": references,
        "reference_lines": reference_lines,
        "retrieval": retrieval_dict,
    }

    # Add dry_run specific fields
    if dry_run:
        response["dry_run"] = True
        if prompt is not None:
            response["prompt"] = prompt

    return response


def build_retrieval_state_dict(
    *,
    retriever: Any,
    query_collection_name: str | None,
    query_where: dict[str, Any] | None,
    planned_where: dict[str, Any] | None,
    planned_collection_type: str,
) -> dict[str, Any]:
    """Build retrieval state dict for answer response payload."""
    return {
        "query_collection": query_collection_name,
        "query_where": query_where,
        "planned_where": deepcopy(planned_where) if planned_where is not None else None,
        "effective_where": deepcopy(getattr(retriever, "_last_effective_where", None)),
        "planned_collection_type": str(planned_collection_type),
        "effective_collection": getattr(
            retriever, "_last_effective_collection_name", None
        ),
        "effective_collection_type": getattr(
            retriever, "_last_effective_collection_type", None
        ),
        "retrieved_ids": list(getattr(retriever, "_last_retrieved_ids", []) or []),
        "retrieved_metadatas": list(
            getattr(retriever, "_last_retrieved_metadatas", []) or []
        ),
    }


def build_hybrid_rerank_dict(
    *,
    enable_hybrid_rerank: bool,
    ranking_weights: Any,
    hybrid_vec_k: int,
) -> dict[str, Any]:
    """Build hybrid rerank debug dict for answer response payload."""
    return {
        "enabled": bool(enable_hybrid_rerank),
        "weights": {
            "alpha_vec": ranking_weights.alpha_vec,
            "beta_bm25": ranking_weights.beta_bm25,
            "gamma_cite": ranking_weights.gamma_cite,
            "delta_role": ranking_weights.delta_role,
        },
        "vec_k": int(hybrid_vec_k),
    }
