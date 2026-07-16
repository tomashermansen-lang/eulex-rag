"""Streaming preparation for RAG pipeline.

Single Responsibility: Prepare retrieval context and prompt for streaming
LLM responses. Performs retrieval, citation expansion, merging, and prompt
building — returning everything needed for the streaming endpoint.

Does NOT import rag.py. All external dependencies are injected.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from . import metadata_helpers
from .planning import QueryContext, UserProfile, build_retrieval_plan
from .prompt_builder import build_prompt
from .rag_config import _get_rag_settings

logger = logging.getLogger(__name__)


def prepare_for_streaming_stage(
    *,
    question: str,
    corpus_id: str,
    top_k: int,
    user_profile: UserProfile | str | None,
    query_fn: Callable[..., tuple[list[tuple[str, dict[str, Any]]], list[float]]],
    collection_name: str | None,
) -> dict[str, Any]:
    """Prepare retrieval context and prompt for streaming.

    Performs retrieval and prompt building, returning everything
    needed for streaming the LLM response.

    Args:
        question: User question.
        corpus_id: Corpus identifier.
        top_k: Number of top results to return.
        user_profile: User profile string or enum.
        query_fn: Callable that takes (question, k, where) kwargs and returns
                  (hits, distances).
        collection_name: Name of the active collection (for metadata).

    Returns:
        Dict with keys: prompt, references_structured, reference_lines, retrieval.

    Raises:
        ValueError: If question is empty.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Normalize user profile
    from .planning import normalize_user_profile

    resolved_profile = normalize_user_profile(user_profile)

    # Build query context
    ctx = QueryContext(
        corpus_id=str(corpus_id or "").strip(),
        user_profile=resolved_profile,
        focus=None,  # Simplified for streaming
        top_k=int(top_k),
        question=question,
    )
    plan = build_retrieval_plan(ctx)

    # Perform retrieval - fetch extra for citation expansion
    where_filter: dict[str, Any] = {"corpus_id": ctx.corpus_id}
    hits, distances = query_fn(
        question=question,
        k=ctx.top_k * 2,
        where=where_filter,
    )

    # Apply citation expansion to get scope articles/annexes
    expansion_hits: list[tuple[str, dict[str, Any]]] = []
    expansion_distances: list[float] = []
    try:
        from .citation_expansion import (
            get_citation_expansion_for_query,
            get_max_expansion,
            is_citation_expansion_enabled,
        )

        if is_citation_expansion_enabled():
            expansion_articles = get_citation_expansion_for_query(
                question=question,
                corpus_id=ctx.corpus_id,
                retrieved_metadatas=[h[1] for h in hits],
            )

            if expansion_articles:
                max_exp = get_max_expansion()
                for exp_ref in expansion_articles[:max_exp]:
                    exp_ref_str = str(exp_ref).strip()
                    if exp_ref_str.upper().startswith("ANNEX:"):
                        annex_val = (
                            exp_ref_str.split(":", 1)[1]
                            if ":" in exp_ref_str
                            else exp_ref_str
                        )
                        exp_where = {
                            "corpus_id": ctx.corpus_id,
                            "annex": metadata_helpers.normalize_annex_for_chroma(
                                annex_val
                            ),
                        }
                    else:
                        exp_where = {"corpus_id": ctx.corpus_id, "article": exp_ref_str}

                    try:
                        exp_hits_result, exp_dist = query_fn(
                            question=question,
                            k=1,
                            where=exp_where,
                        )
                        if exp_hits_result:
                            exp_meta = exp_hits_result[0][1]
                            exp_art = exp_meta.get("article")
                            exp_ann = exp_meta.get("annex")
                            already_present = False
                            for _, existing_meta in hits:
                                ex_art = existing_meta.get("article")
                                ex_ann = existing_meta.get("annex")
                                if exp_art and ex_art and exp_art == ex_art:
                                    already_present = True
                                    break
                                if exp_ann and ex_ann and exp_ann == ex_ann:
                                    already_present = True
                                    break
                            if not already_present:
                                expansion_hits.append(exp_hits_result[0])
                                expansion_distances.append(
                                    exp_dist[0] if exp_dist else 1.0
                                )
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "Citation expansion query failed for %s",
                            exp_ref_str,
                            exc_info=True,
                        )
    except Exception:  # noqa: BLE001
        logger.debug("Citation expansion unavailable (optional)", exc_info=True)

    # Merge: take first few original hits, then expansion hits, then remaining
    rag_cfg = _get_rag_settings()
    initial_hits_count = int(rag_cfg.get("streaming_initial_hits", 5))

    merged_hits: list[tuple[str, dict[str, Any]]] = []
    merged_distances: list[float] = []

    for i, h in enumerate(hits[:initial_hits_count]):
        merged_hits.append(h)
        merged_distances.append(distances[i] if i < len(distances) else 1.0)

    for i, h in enumerate(expansion_hits):
        merged_hits.append(h)
        merged_distances.append(
            expansion_distances[i] if i < len(expansion_distances) else 1.0
        )

    for i, h in enumerate(hits[initial_hits_count:]):
        if len(merged_hits) >= ctx.top_k:
            break
        merged_hits.append(h)
        merged_distances.append(
            distances[initial_hits_count + i]
            if (initial_hits_count + i) < len(distances)
            else 1.0
        )

    hits = merged_hits[: ctx.top_k]
    distances = merged_distances[: ctx.top_k]

    # Build context blocks and references
    context_blocks: list[str] = []
    references_structured: list[dict[str, Any]] = []
    reference_lines: list[str] = []
    kilder_lines: list[str] = ["KILDER:"]

    for idx, (doc, meta) in enumerate(hits, start=1):
        corpus = str(meta.get("corpus_id") or ctx.corpus_id).strip()
        article = meta.get("article")
        recital = meta.get("recital")
        annex = meta.get("annex")

        anchor_parts = []
        if article:
            anchor_parts.append(f"Artikel {article}")
        if recital:
            anchor_parts.append(f"Betragtning {recital}")
        if annex:
            anchor_parts.append(f"Bilag {annex}")
        anchor_label = ", ".join(anchor_parts) if anchor_parts else "Ukendt reference"

        display = f"{corpus} / {anchor_label}"
        excerpt_max = int(rag_cfg.get("streaming_excerpt_max_chars", 150))
        excerpt = str(doc or "")[:excerpt_max].strip().replace("\n", " ")

        ref = {
            "idx": idx,
            "display": display,
            "corpus_id": corpus,
            "article": article,
            "recital": recital,
            "annex": annex,
            "chunk_text": str(doc or ""),
        }
        references_structured.append(ref)
        reference_lines.append(f"[{idx}] {display}")
        kilder_lines.append(f"- [{idx}] {corpus} / {anchor_label} — {excerpt}")

        context_blocks.append(f"[{idx}] {anchor_label}:\n{doc}")

    kilder_block = "\n".join(kilder_lines).strip()
    if kilder_block and context_blocks:
        context = f"{kilder_block}\n\n" + "\n\n".join(context_blocks)
    else:
        context = "\n\n".join(context_blocks) if context_blocks else ""

    prompt = build_prompt(ctx=ctx, plan=plan, context=context, focus_block="")

    return {
        "prompt": prompt,
        "references_structured": references_structured,
        "reference_lines": reference_lines,
        "retrieval": {
            "distances": distances if distances else [],
            "retrieved_ids": [],
            "retrieved_metadatas": [h[1] for h in hits] if hits else [],
            "query_collection": collection_name,
            "query_where": where_filter,
        },
    }
