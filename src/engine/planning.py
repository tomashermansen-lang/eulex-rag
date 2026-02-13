from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from . import helpers
from .types import ClaimIntent, UserProfile, FocusType, FocusSelection

if TYPE_CHECKING:
    from .policy import EffectivePolicy


@dataclass(frozen=True)
class QueryContext:
    corpus_id: str
    user_profile: UserProfile
    focus: FocusSelection | None
    top_k: int
    question: str


class Intent(str, Enum):
    STRUCTURE = "structure"
    CHAPTER_SUMMARY = "chapter_summary"
    ARTICLE_SUMMARY = "article_summary"
    FREEFORM = "freeform"


@dataclass(frozen=True)
class RetrievalPlan:
    intent: Intent
    top_k: int
    where: dict[str, Any] | None
    allow_low_evidence_answer: bool


_ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(value: str) -> int | None:
    v = (value or "").strip().upper()
    if not v or any(ch not in _ROMAN_MAP for ch in v):
        return None
    total = 0
    prev = 0
    for ch in reversed(v):
        cur = _ROMAN_MAP[ch]
        if cur < prev:
            total -= cur
        else:
            total += cur
            prev = cur
    return total


def ref_to_int(value: str | None) -> int | None:
    s = str(value or "").strip().upper()
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:  # noqa: BLE001
            return None
    return roman_to_int(s)


def detect_intent(*, question: str, focus: FocusSelection | None) -> Intent:
    q = (question or "").strip().lower()

    structure_markers = (
        "indholdsfortegnelse",
        "toc",
        "table of contents",
        "oversigt",
        "struktur",
        "hvilke kapitler",
        "hvilke artikler",
        "liste over",
        "oversigt over",
    )
    if any(m in q for m in structure_markers):
        return Intent.STRUCTURE

    if focus and focus.type == FocusType.CHAPTER and any(tok in q for tok in ("kapitlet", "dette kapitel")):
        # The user is referring to the selected chapter.
        if any(tok in q for tok in ("sammenfat", "opsummer", "hvad handler", "hvad står")):
            return Intent.CHAPTER_SUMMARY
        # Default: treat as chapter summary for vague "kapitlet" questions.
        if "kapitel" in q:
            return Intent.CHAPTER_SUMMARY

    # Explicit chapter reference in question.
    if re.search(r"(?i)k\s*a\s*p\s*i\s*t\s*e\s*l\s*([0-9]+|[ivxlcdm]+)", question or ""):
        if any(tok in q for tok in ("sammenfat", "opsummer", "hvad handler", "hvad står")):
            return Intent.CHAPTER_SUMMARY

    # Article summary.
    if re.search(r"(?i)a\s*r\s*t\s*i\s*k\s*e\s*l\s*(\d{1,3}[a-z]?)", question or ""):
        if any(tok in q for tok in ("sammenfat", "opsummer", "hvad siger", "hvad står")):
            return Intent.ARTICLE_SUMMARY

    return Intent.FREEFORM


def focus_to_where(*, corpus_id: str, focus: FocusSelection | None) -> dict[str, Any] | None:
    if not focus:
        return {"corpus_id": corpus_id} if corpus_id else None

    where: dict[str, Any] = {}
    if corpus_id:
        where["corpus_id"] = corpus_id

    if focus.type == FocusType.ARTICLE and focus.article:
        where["article"] = str(focus.article)
        return where

    if focus.type == FocusType.CHAPTER and focus.chapter:
        where["chapter"] = str(focus.chapter)
        return where

    if focus.type == FocusType.ANNEX and focus.annex:
        where["annex"] = helpers.normalize_annex_for_chroma(focus.annex)
        return where

    if focus.type == FocusType.SECTION and focus.section:
        where["section"] = str(focus.section)
        if focus.chapter:
            where["chapter"] = str(focus.chapter)
        if focus.annex:
            where["annex"] = helpers.normalize_annex_for_chroma(focus.annex)
        return where

    # Fallback: corpus-only
    return where or None


def default_top_k(*, user_profile: UserProfile, base_top_k: int) -> int:
    """Get default top_k from config based on user profile.
    
    Uses max_context_legal / max_context_engineering from settings.yaml.
    """
    from ..common.config_loader import load_settings
    cfg = load_settings()
    if user_profile == UserProfile.LEGAL:
        return int(cfg.max_context_legal)
    return int(cfg.max_context_engineering)


def build_retrieval_plan(ctx: QueryContext) -> RetrievalPlan:
    intent = detect_intent(question=ctx.question, focus=ctx.focus)

    planned_top_k = default_top_k(user_profile=ctx.user_profile, base_top_k=ctx.top_k)
    where = focus_to_where(corpus_id=ctx.corpus_id, focus=ctx.focus)

    # Guardrails: LEGAL abstains earlier; ENGINEERING can proceed with disclaimers.
    allow_low_evidence_answer = ctx.user_profile == UserProfile.ENGINEERING

    # For structure questions, tolerate weaker retrieval.
    if intent == Intent.STRUCTURE:
        allow_low_evidence_answer = True

    # For chapter summaries we want broader retrieval than default.
    if intent == Intent.CHAPTER_SUMMARY:
        if ctx.user_profile == UserProfile.LEGAL:
            planned_top_k = max(planned_top_k, 8)
        else:
            planned_top_k = max(planned_top_k, 12)

    return RetrievalPlan(
        intent=intent,
        top_k=planned_top_k,
        where=where,
        allow_low_evidence_answer=allow_low_evidence_answer,
    )


def refine_retrieval_plan(
    *,
    plan: RetrievalPlan,
    question: str,
    corpus_id: str,
    user_profile: UserProfile,
    claim_intent: ClaimIntent,
    requirements_cues_detected: bool,
) -> Tuple[Dict[str, Any] | None, int | None, Dict[str, Any]]:
    """
    Refine the retrieval plan by computing an effective 'where' filter.
    Handles explicit chapter/article references.

    Returns:
        (effective_where, effective_top_k, dbg_update)
    """
    # Compute an effective hard filter for retrieval. Focus selection is the primary source.
    # Additionally, if the question references a specific chapter/article, hard-scope retrieval to it.
    effective_where: dict[str, Any] = dict(plan.where or {})
    if corpus_id and "corpus_id" not in effective_where:
        effective_where["corpus_id"] = corpus_id

    if "chapter" not in effective_where:
        chapter_ref = helpers._extract_chapter_ref(question)
        if chapter_ref:
            canonical = str(chapter_ref).strip().upper()
            if canonical:
                effective_where["chapter"] = canonical

    # Hard-scope to an explicit article ONLY when the question references exactly one article.
    # Multi-article prompts must not be hard-scoped to the first match.
    explicit_article_refs = helpers._extract_article_refs(question)
    multi_part_question = helpers._looks_like_multi_part_question(question)
    explicit_annex_refs = helpers._extract_annex_refs(question)
    multi_anchor_question = (len(explicit_article_refs) >= 2) or (len(explicit_annex_refs) >= 1)
    if "article" not in effective_where:
        if (not multi_part_question) and len(explicit_article_refs) == 1:
            effective_where["article"] = explicit_article_refs[0]

    if "recital" not in effective_where:
        recital_ref = helpers._extract_recital_ref(question)
        if recital_ref:
            effective_where["recital"] = recital_ref

    dbg_update = {
        "explicit_article_refs": explicit_article_refs,
        "explicit_annex_refs": explicit_annex_refs,
        "multi_anchor_question": multi_anchor_question,
    }
    return effective_where or None, None, dbg_update


@dataclass
class AnswerContext:
    """Complete context for answer generation pipeline.

    Consolidates all setup data needed for the answer pipeline stages:
    - Query context (corpus, profile, focus, question)
    - Retrieval plans (initial and effective)
    - Policy configuration
    - Run metadata for tracing/debugging
    - Claim intent classification

    This is the output of the setup/planning stage (Stage 1) of answer_structured.
    """

    ctx: QueryContext
    plan: RetrievalPlan
    effective_plan: RetrievalPlan
    effective_policy: "EffectivePolicy"
    run_meta: Dict[str, Any]
    claim_intent_final: ClaimIntent
    corpus_debug_on: bool
    # Extracted refs for multi-anchor logic
    explicit_article_refs: List[str]
    explicit_annex_refs: List[str]
    multi_anchor_question: bool
    # Policy intent keys used
    policy_intent_keys: List[str]
    # Where filter for retrieval
    where_for_retrieval: Dict[str, Any] | None


def prepare_answer_context(
    *,
    question: str,
    corpus_id: str,
    resolved_profile: UserProfile,
    top_k: int,
    get_effective_policy_fn: Any,
    classify_intent_fn: Any,
    apply_policy_to_intent_fn: Any,
    is_debug_corpus_fn: Any,
    iso_utc_now_fn: Any,
    git_commit_fn: Any,
    resolver_fn: Any | None = None,
    required_anchors_payload: Dict[str, Any] | None = None,
    contract_min_citations: int | None = None,
    skip_intent_classification: bool = False,
    last_exchange: list | None = None,
    original_query: str | None = None,
) -> AnswerContext:
    """Stage 1: Prepare all context needed for answer generation.

    Consolidates:
    - Input validation
    - QueryContext creation
    - RetrievalPlan building
    - Policy resolution
    - run_meta initialization
    - Claim-intent classification
    - Corpus debug setup

    Args:
        question: The user's question
        corpus_id: The corpus to query
        resolved_profile: Normalized user profile (LEGAL/ENGINEERING)
        top_k: Base top_k value
        get_effective_policy_fn: Function to get effective policy
        classify_intent_fn: Function to classify question intent
        apply_policy_to_intent_fn: Function to apply policy to intent
        is_debug_corpus_fn: Function to check if corpus debug is enabled
        iso_utc_now_fn: Function to get current UTC timestamp
        git_commit_fn: Function to get git commit hash
        resolver_fn: Optional corpus resolver function
        required_anchors_payload: Optional anchor requirements
        contract_min_citations: Optional minimum citations requirement
        last_exchange: Optional last user+assistant exchange for context augmentation.
        original_query: Original query before rewriting (None for first turn).

    Returns:
        AnswerContext with all setup data for the pipeline
    """
    import uuid
    from pathlib import Path
    from ..common.corpus_registry import (
        normalize_alias,
        normalize_corpus_id as _normalize_corpus_id,
        default_registry_path,
        load_registry,
    )

    focus: FocusSelection | None = None
    ctx = QueryContext(
        corpus_id=corpus_id,
        user_profile=resolved_profile,
        focus=focus,
        top_k=top_k,
        question=question,
    )
    plan: RetrievalPlan = build_retrieval_plan(ctx)

    # Get effective policy with default intent
    policy_intent_keys = ["default", str(plan.intent.value)]
    effective_policy = get_effective_policy_fn(corpus_id=corpus_id, intent_keys=list(policy_intent_keys))

    # Initialize run_meta
    run_meta: Dict[str, Any] = {
        "run_id": str(uuid.uuid4()),
        "timestamp_utc": iso_utc_now_fn(),
        "git_commit": git_commit_fn(),
        "config_snapshot": {
            "law": ctx.corpus_id,
            "user_profile": resolved_profile.value,
            "top_k": top_k,
            "ui_debug_mode": None,
        },
    }

    # Surface policy inputs/outputs for auditability
    try:
        run_meta.setdefault("anchor_hints", {})
        run_meta["anchor_hints"].update(
            {
                "intent_used": list(policy_intent_keys),
                "intent_effective": list(getattr(effective_policy, "intent_keys_effective", ()) or ()),
                "policy_contributors": dict(getattr(effective_policy, "contributors", {}) or {}),
            }
        )

        ng = getattr(effective_policy, "normative_guard", None)
        if ng is not None:
            run_meta["anchor_hints"].setdefault("effective_policy", {})
            run_meta["anchor_hints"]["effective_policy"]["normative_guard"] = {
                "required_support": str(getattr(ng, "required_support", "") or ""),
                "profiles": list(getattr(ng, "profiles", ()) or ()),
            }

        ap = getattr(effective_policy, "answer_policy", None)
        if ap is not None:
            run_meta["anchor_hints"].setdefault("effective_policy", {})
            run_meta["anchor_hints"]["effective_policy"]["answer_policy"] = (
                ap.to_debug_dict() if hasattr(ap, "to_debug_dict") else {"intent_category": str(getattr(ap, "intent_category", "") or "")}
            )
    except Exception:  # noqa: BLE001
        pass

    # Determine claim-intent (or defer for pipeline overlap)
    query_was_rewritten = (
        original_query is not None and original_query != question
    )

    if skip_intent_classification:
        claim_intent_final = ClaimIntent.GENERAL
        claim_intent_dbg: Dict[str, Any] = {}
        run_meta.setdefault("claim_intent", {})
        run_meta["claim_intent"].update(
            {
                "classifier": "",
                "final": str(ClaimIntent.GENERAL.value),
                "router": {},
                "policy": {},
                "deferred": True,
            }
        )
    else:
        claim_intent_classifier, router_debug = classify_intent_fn(
            question,
            last_exchange=last_exchange,
            query_was_rewritten=query_was_rewritten,
        )
        claim_intent_final = claim_intent_classifier
        claim_intent_dbg: Dict[str, Any] = {"router": router_debug}
        try:
            claim_intent_final, policy_dbg = apply_policy_to_intent_fn(
                resolved_profile=resolved_profile,
                classifier_intent=claim_intent_classifier,
                policy=effective_policy,
                question=question,
            )
            claim_intent_dbg["policy"] = policy_dbg
        except Exception:  # noqa: BLE001
            claim_intent_final = claim_intent_classifier

        run_meta.setdefault("claim_intent", {})
        run_meta["claim_intent"].update(
            {
                "classifier": str(getattr(claim_intent_classifier, "value", claim_intent_classifier) or ""),
                "final": str(getattr(claim_intent_final, "value", claim_intent_final) or ""),
                "router": dict(router_debug or {}),
                "policy": dict(claim_intent_dbg.get("policy") or {}),
            }
        )

    # Contract / eval-relevant knobs
    run_meta["whether_contract_check_enabled"] = bool(contract_min_citations is not None)
    try:
        run_meta["contract_min_citations"] = int(contract_min_citations) if contract_min_citations is not None else 0
    except Exception:  # noqa: BLE001
        run_meta["contract_min_citations"] = 0

    # Placeholders for downstream citation parsing
    run_meta.setdefault("citations_source", None)
    run_meta.setdefault("parsed_citations_raw", [])
    run_meta.setdefault("answer_text_contains_brackets", None)
    run_meta.setdefault("final_gate_reason", None)

    # Required anchors (eval/contract-only)
    if required_anchors_payload:
        run_meta["required_anchors"] = dict(required_anchors_payload)
        run_meta["anchor_rescue"] = {
            "enabled": True,
            "candidate_pool_size": None,
            "anchors_in_candidate_pool": {},
            "anchors_in_top_k": [],
            "missing_required_anchor_any_of": [],
            "missing_required_anchor_any_of_2": [],
            "missing_required_anchor_all_of": [],
            "rescue_slots_added": 0,
            "rescue_injected_anchors": [],
            "rescue_injection_source": None,
            "rescue_added_refs_count": 0,
            "rescue_required_anchor_retry_performed": False,
            "rescue_required_anchor_retry_success": None,
        }
    else:
        run_meta.setdefault("anchor_rescue", {"enabled": False})

    # Corpus debug setup
    corpus_debug_on = is_debug_corpus_fn()
    if corpus_debug_on:
        q_norm = normalize_alias(question)
        q_tokens = re.findall(r"[0-9A-Za-zæøåÆØÅ]+", q_norm)
        selected_corpus_raw = str(ctx.corpus_id or "").strip()
        selected_corpus_norm = _normalize_corpus_id(selected_corpus_raw)

        corpus_candidates: List[str] = []
        if resolver_fn is not None:
            try:
                corpus_candidates = list(resolver_fn().mentioned_corpus_keys(q_norm) or [])
            except Exception:  # noqa: BLE001
                corpus_candidates = []

        matched_aliases: List[Dict[str, Any]] = []
        rejected_aliases: List[Dict[str, Any]] = []
        try:
            project_root = Path(__file__).resolve().parents[1]
            reg = load_registry(default_registry_path(project_root))
            for corpus_id_raw, entry in sorted(reg.items(), key=lambda t: str(t[0])):
                if not isinstance(entry, dict):
                    continue
                corpus_key = _normalize_corpus_id(str(corpus_id_raw or ""))
                display = str(entry.get("display_name") or "").strip()
                aliases_in = entry.get("aliases")
                aliases_raw: List[str] = []
                if isinstance(aliases_in, list):
                    for a in aliases_in:
                        if isinstance(a, str) and a.strip():
                            aliases_raw.append(a.strip())
                if display:
                    aliases_raw.append(display)

                for a in aliases_raw:
                    a_norm = normalize_alias(a)
                    if not a_norm:
                        continue
                    if len(a_norm) < 3:
                        rejected_aliases.append(
                            {
                                "corpus": corpus_key,
                                "alias": a,
                                "alias_norm": a_norm,
                                "rejected_reason": "alias_too_short_lt3",
                            }
                        )
                        continue

                    pat = re.compile(rf"(?<!\w){re.escape(a_norm)}(?!\w)")
                    if pat.search(q_norm):
                        matched_aliases.append(
                            {
                                "corpus": corpus_key,
                                "alias": a,
                                "alias_norm": a_norm,
                            }
                        )
        except Exception:  # noqa: BLE001
            matched_aliases = []
            rejected_aliases = []

        run_meta["corpus_debug"] = {
            "selected_corpus_raw": selected_corpus_raw,
            "selected_corpus_norm": selected_corpus_norm,
            "profile": str(resolved_profile.value),
            "contract_check": bool(contract_min_citations is not None),
            "expected_min_citations": (int(contract_min_citations) if contract_min_citations is not None else None),
            "query_norm": q_norm,
            "query_tokens": q_tokens,
            "corpus_candidates": corpus_candidates,
            "matched_aliases": matched_aliases,
            "rejected_aliases": rejected_aliases,
        }

    # Refine retrieval plan
    effective_where, _, dbg_update = refine_retrieval_plan(
        plan=plan,
        question=question,
        corpus_id=corpus_id,
        user_profile=resolved_profile,
        claim_intent=claim_intent_final,
        requirements_cues_detected=bool((claim_intent_dbg or {}).get("requirements_cues_detected")),
    )

    explicit_article_refs = list(dbg_update.get("explicit_article_refs") or [])
    explicit_annex_refs = list(dbg_update.get("explicit_annex_refs") or [])
    multi_anchor_question = bool(dbg_update.get("multi_anchor_question", False))

    where_for_retrieval: Dict[str, Any] | None = effective_where or None

    effective_plan = RetrievalPlan(
        intent=plan.intent,
        top_k=plan.top_k,
        where=where_for_retrieval,
        allow_low_evidence_answer=plan.allow_low_evidence_answer,
    )

    return AnswerContext(
        ctx=ctx,
        plan=plan,
        effective_plan=effective_plan,
        effective_policy=effective_policy,
        run_meta=run_meta,
        claim_intent_final=claim_intent_final,
        corpus_debug_on=corpus_debug_on,
        explicit_article_refs=explicit_article_refs,
        explicit_annex_refs=explicit_annex_refs,
        multi_anchor_question=multi_anchor_question,
        policy_intent_keys=policy_intent_keys,
        where_for_retrieval=where_for_retrieval,
    )


def apply_deferred_intent(
    answer_ctx: AnswerContext,
    classify_intent_fn: Callable[..., tuple[ClaimIntent, dict]],
    apply_policy_to_intent_fn: Callable[..., tuple[ClaimIntent, dict]],
    resolved_profile: UserProfile,
    last_exchange: list | None = None,
    original_query: str | None = None,
) -> AnswerContext:
    """Finalize intent classification after parallel execution.

    Called after prepare_answer_context(skip_intent_classification=True)
    to fill in the deferred intent. Mutates answer_ctx in place.

    Args:
        answer_ctx: Context with placeholder GENERAL intent
        classify_intent_fn: Intent classification function
        apply_policy_to_intent_fn: Policy application function
        resolved_profile: User profile for policy application
        last_exchange: Optional last user+assistant exchange for context augmentation.
        original_query: Original query before rewriting (None for first turn).

    Returns:
        The same AnswerContext with updated claim_intent_final and run_meta
    """
    question = answer_ctx.ctx.question
    query_was_rewritten = (
        original_query is not None and original_query != question
    )
    claim_intent_classifier, router_debug = classify_intent_fn(
        question,
        last_exchange=last_exchange,
        query_was_rewritten=query_was_rewritten,
    )
    claim_intent_final = claim_intent_classifier

    try:
        claim_intent_final, policy_dbg = apply_policy_to_intent_fn(
            resolved_profile=resolved_profile,
            classifier_intent=claim_intent_classifier,
            policy=answer_ctx.effective_policy,
            question=question,
        )
    except Exception:  # noqa: BLE001
        policy_dbg = {}

    answer_ctx.claim_intent_final = claim_intent_final
    answer_ctx.run_meta.setdefault("claim_intent", {})
    answer_ctx.run_meta["claim_intent"].update(
        {
            "classifier": str(getattr(claim_intent_classifier, "value", claim_intent_classifier) or ""),
            "final": str(getattr(claim_intent_final, "value", claim_intent_final) or ""),
            "router": dict(router_debug or {}),
            "policy": dict(policy_dbg or {}),
            "deferred": True,
        }
    )

    return answer_ctx


# ---------------------------------------------------------------------------
# Multi-Corpus Context Assembly (Cross-Law Synthesis)
# ---------------------------------------------------------------------------


@dataclass
class MultiCorpusContext:
    """Context for multi-corpus (cross-law) queries.

    Used when corpus_scope is "explicit" or "all" to track
    synthesis mode and per-corpus evidence.
    """

    question: str
    synthesis_mode: Any  # SynthesisMode enum
    target_corpora: Tuple[str, ...]
    per_corpus_evidence: Dict[str, List[Any]]  # corpus_id -> list of evidence
    resolved_profile: str
    top_k: int
    comparison_pairs: Tuple[Tuple[str, str], ...] | None = None
    routing_only: bool = False


def prepare_multi_corpus_context(
    *,
    question: str,
    synthesis_context: Any,  # SynthesisContext
    resolved_profile_str: str,
    top_k: int,
) -> MultiCorpusContext:
    """Prepare context for multi-corpus query.

    Args:
        question: User's question
        synthesis_context: SynthesisContext from synthesis_router
        resolved_profile_str: User profile as string ("LEGAL" or "ENGINEERING")
        top_k: Base top_k value

    Returns:
        MultiCorpusContext for cross-law query processing
    """
    # Initialize per_corpus_evidence dict with empty lists for each target corpus
    per_corpus_evidence: Dict[str, List[Any]] = {
        corpus_id: [] for corpus_id in synthesis_context.target_corpora
    }

    return MultiCorpusContext(
        question=question,
        synthesis_mode=synthesis_context.mode,
        target_corpora=synthesis_context.target_corpora,
        per_corpus_evidence=per_corpus_evidence,
        resolved_profile=resolved_profile_str,
        top_k=top_k,
        comparison_pairs=synthesis_context.comparison_pairs,
        routing_only=synthesis_context.routing_only,
    )

