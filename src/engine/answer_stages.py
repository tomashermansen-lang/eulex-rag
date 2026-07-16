"""Stage functions extracted from RAGEngine.answer_structured().

Step 7.6+: Each stage is a standalone function that answer_structured() calls.
Engine modules MUST NOT import rag.py — these functions receive all deps as parameters.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from . import citations
from . import query_helpers
from . import text_transforms
from . import payload_builders
from . import instrumentation
from . import policy as policy_engine
from . import retrieval_orchestration as ro
from .generation_strategies import execute_generation_stage, build_engineering_answer
from .planning import UserProfile, QueryContext, RetrievalPlan, ClaimIntent
from .prompt_builder import build_prompt, build_disclaimer, focus_block_for_prompt
from .rag_config import _RetrievalResult, _get_rag_settings
from .synthesis_router import SynthesisContext, detect_synthesis_mode

# The answer_structured() path always uses "chunk" collection type.
# The "toc" variant only exists in the retrieval pipeline itself.
ANSWER_COLLECTION_TYPE = "chunk"


@dataclass
class RetrievalStageResult:
    """Result from the retrieval stage of answer_structured()."""

    retrieval_result: _RetrievalResult
    hits: List[Tuple[str, dict]]
    distances: List[float]
    synthesis_context: SynthesisContext | None
    use_multi_corpus: bool
    ranking_debug_payload: dict[str, Any] | None
    final_planned_where: dict[str, Any] | None


def execute_answer_retrieval_stage(
    *,
    question: str,
    corpus_scope: str,
    target_corpora: List[str] | None,
    resolved_profile: UserProfile,
    where_for_retrieval: dict[str, Any] | None,
    ctx: QueryContext,
    run_meta: dict[str, Any],
    effective_plan: RetrievalPlan,
    pass_tracker: Any,
    corpus_id: str,
    retriever: Any,
    chroma: Any,
    collection: Any,
    available_corpora_fn: Callable[[], List[str]],
    resolver_fn: Callable,
) -> RetrievalStageResult:
    """Execute the retrieval stage of answer_structured().

    Determines single vs multi-corpus, executes retrieval, merges run_meta
    updates, and records the retrieval pass.

    Side effects:
        - Mutates ``run_meta`` with retrieval metadata (laws_searched, corpus_scope, etc.)
        - Mutates ``retriever._last_retrieved_metadatas`` for multi-corpus sync.
        - Calls ``pass_tracker.record_pass()``.
    """
    effective_target_corpora = target_corpora or []
    use_multi_corpus = corpus_scope == "all" or (
        corpus_scope == "explicit" and len(effective_target_corpora) > 0
    )
    synthesis_context: SynthesisContext | None = None

    if use_multi_corpus:
        corpus_ids = (
            tuple(available_corpora_fn())
            if corpus_scope == "all"
            else tuple(effective_target_corpora)
        )
        synthesis_context = detect_synthesis_mode(
            question=question,
            corpus_scope="explicit",
            selected_corpora=list(corpus_ids),
            resolver=resolver_fn(),
        )
        run_meta["synthesis_mode"] = synthesis_context.mode.name
        multi_result = ro.execute_cross_law_retrieval(
            question=question,
            corpus_ids=corpus_ids,
            resolved_profile=resolved_profile,
            where_filter=where_for_retrieval,
            retriever=retriever,
            chroma=chroma,
            is_citable_fn=citations._is_citable_chunk,
        )
        d = ro.convert_multi_corpus_result(multi_result, resolved_profile)
        retrieval_result = _RetrievalResult(
            hits=d["hits"],
            distances=d["distances"],
            retrieved_ids=d["retrieved_ids"],
            retrieved_metas=d["retrieved_metas"],
            run_meta_updates=d["run_meta_updates"],
            selected_chunks=d["selected_chunks"],
            total_retrieved=d["total_retrieved"],
            citable_count=d["citable_count"],
        )
        retriever._last_retrieved_metadatas = list(retrieval_result.retrieved_metas)
        run_meta["laws_searched"] = list(
            retrieval_result.run_meta_updates.get("laws_searched", [])
        )
        run_meta["corpus_scope"] = corpus_scope
    else:
        result_dict = ro.modular_retrieval(
            question=question,
            resolved_profile=resolved_profile,
            where_for_retrieval=where_for_retrieval,
            corpus_id=corpus_id,
            retriever=retriever,
            collection=collection,
            is_citable_fn=citations._is_citable_chunk,
        )
        # Sync retriever state for backward compat
        retriever._last_retrieved_ids = list(result_dict["retrieved_ids"])
        retriever._last_retrieved_metadatas = list(result_dict["retrieved_metas"])
        retrieval_result = _RetrievalResult(
            hits=result_dict["hits"],
            distances=result_dict["distances"],
            retrieved_ids=result_dict["retrieved_ids"],
            retrieved_metas=result_dict["retrieved_metas"],
            run_meta_updates=result_dict["run_meta_updates"],
            selected_chunks=result_dict["selected_chunks"],
            total_retrieved=result_dict["total_retrieved"],
            citable_count=result_dict["citable_count"],
        )
        run_meta["laws_searched"] = [corpus_id]
        run_meta["corpus_scope"] = "single"

    hits = retrieval_result.hits
    distances = retrieval_result.distances
    final_planned_where = (
        deepcopy(effective_plan.where) if effective_plan.where is not None else None
    )
    pass_tracker.record_pass(
        pass_name="multi_corpus_pipeline" if use_multi_corpus else "modular_pipeline",
        planned_where=final_planned_where,
        planned_collection_type=ANSWER_COLLECTION_TYPE,
    )

    # Merge retrieval run_meta updates
    for key, value in retrieval_result.run_meta_updates.items():
        if isinstance(value, dict) and isinstance(run_meta.get(key), dict):
            run_meta[key].update(value)
        else:
            run_meta[key] = value

    ranking_debug_payload = retrieval_result.run_meta_updates.get("hybrid_rerank")

    return RetrievalStageResult(
        retrieval_result=retrieval_result,
        hits=hits,
        distances=distances,
        synthesis_context=synthesis_context,
        use_multi_corpus=use_multi_corpus,
        ranking_debug_payload=ranking_debug_payload,
        final_planned_where=final_planned_where,
    )


# ---------------------------------------------------------------------------
# Stage 3: Evidence gating + generation (Step 7.7)
# ---------------------------------------------------------------------------


@dataclass
class EvidenceGateResult:
    """Result from the evidence gating stage of answer_structured()."""

    answer_text: str
    did_abstain: bool
    bypass_required_support_gate: bool
    min_citable_required: int
    early_return: dict[str, Any] | None


def execute_evidence_gate_stage(
    *,
    question: str,
    hits: List[Tuple[str, dict]],
    distances: List[float],
    focus: str | None,
    history_context: str,
    dry_run: bool,
    corpus_scope: str,
    resolved_profile: UserProfile,
    effective_plan: Any,
    ctx: Any,
    plan: Any,
    references_structured_all: List[dict[str, Any]],
    citable_count_total: int,
    context: str,
    kilder_block: str,
    synthesis_context: SynthesisContext | None,
    effective_policy: Any,
    claim_intent_final: ClaimIntent,
    corpus_debug_on: bool,
    contract_min_citations: int | None,
    run_meta: dict[str, Any],
    payload_context: dict[str, Any],
    should_abstain_fn: Callable,
    llm_fn: Callable,
    resolver_fn: Callable,
) -> EvidenceGateResult:
    """Execute evidence gating: abstain check, dry-run handling, and LLM generation.

    Determines whether the pipeline should abstain, return early (dry_run),
    gate on insufficient evidence, or proceed to LLM generation.

    Args:
        payload_context: Pre-computed dict with keys: retrieval_state,
            where_for_retrieval, pass_tracker_passes, ranking_debug,
            sibling_expansion — used for building dry-run early-return payloads.

    Side effects:
        - Mutates ``run_meta`` with abstain info, dry_run flags, gate reasons.

    Returns:
        EvidenceGateResult with ``early_return`` set if the caller should
        return that payload immediately, otherwise ``answer_text`` for
        continued post-processing.
    """
    # Compute min_citable_required from config, lowered for scoped queries
    _rag_cfg = _get_rag_settings()
    min_citable_required = int(_rag_cfg.get("min_citable_required", 2))
    try:
        scoped = bool(
            focus
            or query_helpers._looks_like_structure_question(question)
            or query_helpers._extract_article_ref(question)
            or query_helpers._extract_chapter_ref(question)
        )
        if scoped:
            min_citable_required = 1
    except Exception:  # noqa: BLE001
        pass

    # Abstain check
    has_history = bool(history_context and history_context.strip())
    abstain_reason = should_abstain_fn(
        question,
        hits,
        distances,
        allow_low_evidence_answer=effective_plan.allow_low_evidence_answer
        or has_history,
        references_structured=references_structured_all,
        corpus_scope=corpus_scope,
    )

    did_abstain = False
    bypass_required_support_gate = False

    if abstain_reason:
        did_abstain = True
        run_meta.setdefault("abstain", {})
        run_meta["abstain"].update({"abstained": True, "reason": str(abstain_reason)})
        answer_text = str(abstain_reason)

        if dry_run:
            run_meta["dry_run"] = True
            run_meta["dry_run_stage"] = "abstained"
            ref_lines = [
                f"[{r['idx']}] {r.get('display')}" for r in references_structured_all
            ]
            early = payload_builders.build_answer_response_payload(
                run_meta=run_meta,
                user_profile_value=resolved_profile.value,
                focus=focus,
                intent_value=plan.intent.value,
                answer_text=answer_text,
                references=references_structured_all,
                reference_lines=ref_lines,
                distances=distances,
                retrieval_state=payload_context["retrieval_state"],
                effective_plan=effective_plan,
                where_for_retrieval=payload_context["where_for_retrieval"],
                pass_tracker_passes=payload_context["pass_tracker_passes"],
                dry_run=True,
                prompt="",
                ranking_debug=payload_context["ranking_debug"],
                sibling_expansion=payload_context["sibling_expansion"],
            )
            return EvidenceGateResult(
                answer_text=answer_text,
                did_abstain=True,
                bypass_required_support_gate=False,
                min_citable_required=min_citable_required,
                early_return=early,
            )
    else:
        run_meta.setdefault("abstain", {})
        run_meta["abstain"].update({"abstained": False})

        if (
            resolved_profile == UserProfile.ENGINEERING
            and citable_count_total < min_citable_required
        ):
            run_meta["final_gate_reason"] = "insufficient_citable_evidence_pre_llm"
            answer_text = "MISSING_REF"
        elif dry_run:
            run_meta["dry_run"] = True
            run_meta["dry_run_stage"] = "pre_llm_complete"
            focus_block_str = focus_block_for_prompt(focus)
            dry_run_prompt = build_prompt(
                ctx=ctx,
                plan=effective_plan,
                context=context,
                focus_block=focus_block_str,
                contract_min_citations=contract_min_citations,
                history_context=history_context,
            )
            ref_lines = [
                f"[{r['idx']}] {r.get('display')}" for r in references_structured_all
            ]
            early = payload_builders.build_answer_response_payload(
                run_meta=run_meta,
                user_profile_value=resolved_profile.value,
                focus=focus,
                intent_value=plan.intent.value,
                answer_text="[DRY_RUN - LLM not called]",
                references=references_structured_all,
                reference_lines=ref_lines,
                distances=distances,
                retrieval_state=payload_context["retrieval_state"],
                effective_plan=effective_plan,
                where_for_retrieval=payload_context["where_for_retrieval"],
                pass_tracker_passes=payload_context["pass_tracker_passes"],
                dry_run=True,
                prompt=dry_run_prompt,
                ranking_debug=payload_context["ranking_debug"],
                sibling_expansion=payload_context["sibling_expansion"],
            )
            return EvidenceGateResult(
                answer_text="[DRY_RUN - LLM not called]",
                did_abstain=False,
                bypass_required_support_gate=False,
                min_citable_required=min_citable_required,
                early_return=early,
            )
        else:
            answer_text = execute_generation_stage(
                question=question,
                context=context,
                kilder_block=kilder_block,
                ctx=ctx,
                effective_plan=effective_plan,
                synthesis_context=synthesis_context,
                resolved_profile=resolved_profile,
                effective_policy=effective_policy,
                claim_intent_final=claim_intent_final,
                references_structured_all=references_structured_all,
                contract_min_citations=contract_min_citations,
                history_context=history_context,
                corpus_debug_on=corpus_debug_on,
                run_meta=run_meta,
                llm_fn=llm_fn,
                resolver_fn=resolver_fn,
            )

    answer_text = text_transforms._normalize_abstain_text(answer_text)

    return EvidenceGateResult(
        answer_text=answer_text,
        did_abstain=did_abstain,
        bypass_required_support_gate=bypass_required_support_gate,
        min_citable_required=min_citable_required,
        early_return=None,
    )


# ---------------------------------------------------------------------------
# Stage 4: Post-generation pipeline (Step 7.9)
# ---------------------------------------------------------------------------


@dataclass
class PostGenerationResult:
    """Result from the post-generation pipeline stage."""

    answer_text: str
    references_structured: List[dict[str, Any]]
    reference_lines: List[str]
    used_chunk_ids: List[str]
    did_abstain: bool


def execute_post_generation_stage(
    *,
    answer_text: str,
    question: str,
    resolved_profile: UserProfile,
    run_meta: dict[str, Any],
    references_structured_all: List[dict[str, Any]],
    distances: List[float],
    effective_plan: Any,
    effective_policy: Any,
    claim_intent_final: ClaimIntent,
    did_abstain: bool,
    bypass_required_support_gate: bool,
    min_citable_required: int,
    ctx: Any,
    total_retrieved: int,
    citable_count_total: int,
    contract_min_citations: int | None,
    corpus_debug_on: bool,
    max_distance: float | None,
    corpus_id: str,
    project_root: "Path | None",
) -> PostGenerationResult:
    """Execute the post-generation pipeline: policy gates, citations, normalization.

    Processes the raw LLM answer through policy gates, citation processing,
    required-support guard, modal normalization, and SCOPE display rules.

    Side effects:
        - Mutates ``run_meta`` with policy/citation debug info.
        - Calls ``instrumentation.maybe_log_intent_event`` for telemetry.
    """
    # STAGE 4a-pre: Pre-engineering policy gates
    pre_policy_result = policy_engine.apply_pre_engineering_policy_gates(
        answer_text=answer_text,
        question=question,
        resolved_profile=resolved_profile,
        references_structured_all=list(references_structured_all or []),
        claim_intent_from_run_meta=(run_meta.get("claim_intent") or {}).get("final"),
        classify_intent_fn=policy_engine.classify_question_intent,
        run_meta=run_meta,
        corpus_debug_on=corpus_debug_on,
    )

    answer_text = pre_policy_result.answer_text
    references_structured_all = pre_policy_result.references_structured_all
    legal_allow_reference_fallback = pre_policy_result.legal_allow_reference_fallback
    intent_used = pre_policy_result.intent_used or claim_intent_final

    instrumentation.maybe_log_intent_event(
        question=question,
        intent=intent_used,
        profile=resolved_profile,
        corpus_id=corpus_id,
        project_root=project_root,
    )

    # Low evidence disclaimer
    low_evidence = False
    if max_distance is not None and distances:
        try:
            effective_max_distance = float(max_distance)
            if effective_plan.allow_low_evidence_answer:
                low_ev_dist = float(
                    _get_rag_settings().get("low_evidence_max_distance", 1.35)
                )
                effective_max_distance = max(effective_max_distance, low_ev_dist)
            low_evidence = min(distances) > effective_max_distance
        except Exception:  # noqa: BLE001
            low_evidence = False
    disclaimer = build_disclaimer(ctx=ctx, low_evidence=low_evidence)

    if str(answer_text or "").strip() != "MISSING_REF" and disclaimer:
        answer_text = f"{answer_text}\n\nBemærk: {disclaimer}"

    reference_lines_all = [
        f"[{r['idx']}] {r.get('precise_ref') or r.get('display')}"
        for r in references_structured_all
    ]

    # Engineering answer building
    if resolved_profile == UserProfile.ENGINEERING:
        answer_text = build_engineering_answer(
            raw_interpretation=answer_text,
            ctx=ctx,
            references_structured=references_structured_all,
            reference_lines=reference_lines_all,
            distances=distances,
            total_retrieved=total_retrieved,
            citable_count=citable_count_total,
            min_citable_required=min_citable_required,
        )

        try:
            ap = getattr(effective_policy, "answer_policy", None)
            if ap is not None:
                answer_text, ap_dbg = (
                    policy_engine._engineering_apply_answer_policy_requirements_enforcement(
                        answer_text=str(answer_text or ""),
                        answer_policy=ap,
                        engineering_json_mode=bool(
                            run_meta.get("engineering_json_mode") is True
                        ),
                    )
                )
                run_meta.setdefault("answer_policy", {})
                run_meta["answer_policy"].update(ap_dbg)
        except Exception:  # noqa: BLE001
            pass

    # STAGE 4a-post: Post-engineering policy gates
    post_policy_result = policy_engine.apply_post_engineering_policy_gates(
        answer_text=answer_text,
        question=question,
        resolved_profile=resolved_profile,
        references_structured_all=list(references_structured_all or []),
        intent_used=intent_used,
        classify_evidence_type_fn=policy_engine.classify_evidence_type_from_metadata,
        select_references_used_fn=citations.select_references_used_in_answer,
        inject_enforcement_citations_fn=citations._engineering_inject_neutral_hjemmel_citations_for_enforcement,
        run_meta=run_meta,
        corpus_debug_on=corpus_debug_on,
    )

    answer_text = post_policy_result.answer_text
    did_abstain = did_abstain or post_policy_result.did_abstain
    bypass_required_support_gate = (
        bypass_required_support_gate or post_policy_result.bypass_required_support_gate
    )

    # STAGE 4b: Citation processing
    citation_processing_result = citations.apply_all_citation_processing(
        answer_text=answer_text,
        question=question,
        references_structured_all=list(references_structured_all or []),
        resolved_profile=resolved_profile,
        intent_used=intent_used,
        did_abstain=did_abstain,
        required_anchors_payload=None,
        contract_min_citations=contract_min_citations,
        is_legal_profile=(resolved_profile == UserProfile.LEGAL),
        legal_allow_reference_fallback=bool(legal_allow_reference_fallback),
        run_meta=run_meta,
        corpus_debug_on=corpus_debug_on,
        is_debug_enabled_fn=instrumentation.is_debug_enabled,
    )

    answer_text = citation_processing_result.answer_text
    references_structured: list[dict[str, Any]] = (
        citation_processing_result.references_structured
    )
    reference_lines = citation_processing_result.reference_lines
    used_chunk_ids = citation_processing_result.used_chunk_ids

    instrumentation._debug_dump_run_meta(
        run_meta=run_meta,
        stage="after_citation_processing",
        extra={
            "answer_preview": str(answer_text or "")[:160],
            "references_structured_count": int(len(references_structured or [])),
        },
    )

    # Required-support guard
    answer_text, references_structured = policy_engine._apply_required_support_guard(
        resolved_profile=resolved_profile,
        claim_intent_final=intent_used,
        answer_text=str(answer_text or ""),
        references_structured=list(references_structured or []),
        bypass_required_support_gate=bool(bypass_required_support_gate),
        policy=effective_policy,
        run_meta=run_meta,
    )
    if str(answer_text or "").strip() == "MISSING_REF":
        references_structured, reference_lines, used_chunk_ids = [], [], []

    answer_text = text_transforms._normalize_modals_to_danish(answer_text)

    # SCOPE display-only post-processing
    if intent_used == ClaimIntent.SCOPE:
        if resolved_profile == UserProfile.ENGINEERING:
            answer_text = policy_engine._engineering_remove_normative_bullets_from_systemkrav_section_for_scope(
                answer_text
            )
        answer_text, reference_lines = (
            policy_engine._scope_apply_litra_consistency_to_display(
                answer_text=answer_text,
                reference_lines=list(reference_lines or []),
            )
        )

    if corpus_debug_on:
        instrumentation.collect_corpus_debug_telemetry(
            answer_text=str(answer_text or ""),
            references_structured_all=list(references_structured_all or []),
            contract_min_citations=contract_min_citations,
            run_meta=run_meta,
            total_retrieved=total_retrieved,
            citable_count=citable_count_total,
            strip_references_fn=text_transforms._strip_trailing_references_section,
            count_normative_fn=text_transforms._count_normative_sentences,
        )

    return PostGenerationResult(
        answer_text=answer_text,
        references_structured=references_structured,
        reference_lines=reference_lines,
        used_chunk_ids=used_chunk_ids,
        did_abstain=did_abstain,
    )
