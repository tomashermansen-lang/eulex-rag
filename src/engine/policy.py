from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

from .constants import (
    contains_normative_claim,
)
from .types import (
    ClaimIntent,
    EvidenceType,
    LegalClaimGateResult,
    UserProfile,
)
from .constants import _truthy_env
from . import query_helpers
from .concept_config import Policy as AnchorPolicy
from ..common.corpus_registry import normalize_corpus_id
from .corpus_resolver import CorpusResolver


def rescue_rules_feature_enabled(
    *, required_anchors_payload: dict[str, Any] | None
) -> bool:
    """Rescue rules are disabled in normal drift.

    Enable only when:
    - required_anchors is set (eval/contract), OR
    - explicit feature flag is enabled: ANCHOR_RESCUE_RULES=1
    """

    return bool(required_anchors_payload is not None) or _truthy_env(
        "ANCHOR_RESCUE_RULES"
    )


def _policy_required_support_for_profile(
    *,
    policy: AnchorPolicy | None,
    profile: UserProfile,
) -> str | None:
    """Return override required_support if policy applies to this profile."""

    if policy is None:
        return None
    ng = getattr(policy, "normative_guard", None)
    if ng is None:
        return None
    try:
        if not ng.applies_to_profile(str(profile.value)):
            return None
    except Exception:  # noqa: BLE001
        return None
    rs = str(getattr(ng, "required_support", "") or "").strip().lower()
    return rs if rs in {"article", "article_or_annex", "any"} else None


def _policy_normative_guard_match_info(
    *,
    policy: AnchorPolicy | None,
    profile: UserProfile,
) -> dict[str, Any]:
    """Return structured info about whether policy normative_guard matches this profile."""

    info: dict[str, Any] = {
        "policy_matched": False,
        "policy_matched_reason": "no_policy",
        "policy_required_support": None,
        "policy_profiles": None,
    }

    if policy is None:
        return info
    ng = getattr(policy, "normative_guard", None)
    if ng is None:
        info["policy_matched_reason"] = "no_normative_guard"
        return info

    try:
        info["policy_profiles"] = list(getattr(ng, "profiles", ()) or ())
    except Exception:  # noqa: BLE001
        info["policy_profiles"] = None

    try:
        applies = bool(ng.applies_to_profile(str(profile.value)))
    except Exception:  # noqa: BLE001
        info["policy_matched_reason"] = "profile_check_error"
        return info

    if not applies:
        info["policy_matched_reason"] = "profile_mismatch"
        return info

    rs = str(getattr(ng, "required_support", "") or "").strip().lower()
    if rs not in {"article", "article_or_annex", "any"}:
        info["policy_matched_reason"] = "invalid_required_support"
        return info

    info["policy_matched"] = True
    info["policy_matched_reason"] = "matched"
    info["policy_required_support"] = rs
    return info


# NOTE: _contains_normative_claim moved to constants.py as contains_normative_claim()


def _apply_required_support_guard(
    *,
    resolved_profile: UserProfile,
    claim_intent_final: ClaimIntent,
    answer_text: str,
    references_structured: list[dict[str, Any]],
    bypass_required_support_gate: bool,
    policy: AnchorPolicy | None,
    run_meta: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply required-support / normative guard with optional policy override.

    Default behavior: require article_or_annex support for normative claims.
    Both articles and annexes are legally binding parts of EU regulations, so either
    type of support is sufficient to ground a normative answer.

    If policy provides normative_guard.required_support for the active intents and profile,
    it overrides the default (last-wins in policy merge).
    """

    has_article_support = any(
        bool(r.get("article")) for r in list(references_structured or [])
    )
    has_annex_support = any(
        bool(r.get("annex")) for r in list(references_structured or [])
    )

    policy_info = _policy_normative_guard_match_info(
        policy=policy, profile=resolved_profile
    )
    override_required_support = policy_info.get("policy_required_support")

    # Determine whether we should enforce any guard at all.
    enforce_due_to_intent = (
        resolved_profile == UserProfile.ENGINEERING
        and claim_intent_final
        in {
            ClaimIntent.REQUIREMENTS,
            ClaimIntent.CLASSIFICATION,
        }
    )
    has_normative_claim = False
    try:
        has_normative_claim = bool(contains_normative_claim(answer_text))
    except Exception:  # noqa: BLE001
        has_normative_claim = False

    should_enforce = enforce_due_to_intent or has_normative_claim
    if not should_enforce:
        # Still surface deterministic policy matching info for debugging/eval.
        run_meta["normative_guard"] = {
            "policy_required_support": policy_info.get("policy_required_support"),
            "required_support_used": None,
            "policy_matched": bool(policy_info.get("policy_matched")),
            "policy_matched_reason": str(
                policy_info.get("policy_matched_reason") or ""
            ),
            "profile": str(resolved_profile.value),
            "intent": str(
                getattr(claim_intent_final, "value", claim_intent_final) or ""
            ),
            "should_enforce": False,
        }
        return answer_text, references_structured

    # Default required_support.
    # Both articles and annexes are legally binding parts of EU regulations.
    # Annexes often contain essential normative content (e.g., Annex III defines high-risk
    # categories in AI Act). Requiring article-only support would incorrectly reject valid
    # answers that are properly grounded in annex provisions.
    required_support_default: str = "article_or_annex"

    required_support = override_required_support or required_support_default
    required_support = str(required_support or "").strip().lower()
    if required_support not in {"article", "article_or_annex", "any"}:
        required_support = required_support_default

    if required_support == "any":
        return answer_text, references_structured

    has_required_support = (
        bool(has_article_support)
        if required_support == "article"
        else bool(has_article_support or has_annex_support)
    )

    # Always provide machine-readable context.
    run_meta["normative_guard"] = {
        "policy_required_support": policy_info.get("policy_required_support"),
        "required_support_used": str(required_support),
        "policy_matched": bool(policy_info.get("policy_matched")),
        "policy_matched_reason": str(policy_info.get("policy_matched_reason") or ""),
        "profile": str(resolved_profile.value),
        "intent": str(getattr(claim_intent_final, "value", claim_intent_final) or ""),
        "anchors_in_top_k": list(run_meta.get("anchors_in_top_k") or []),
        "has_article_support": bool(has_article_support),
        "has_annex_support": bool(has_annex_support),
    }

    if (not has_required_support) and (not bypass_required_support_gate):
        # Deterministic fail reason (not dependent on debug).
        if run_meta.get("final_fail_reason") is None:
            run_meta["final_fail_reason"] = "NORMATIVE_GUARD_NO_REQUIRED_SUPPORT"

        run_meta["normative_guard"]["triggered"] = True
        run_meta["normative_guard"]["trigger_reason"] = "no_required_support"
        run_meta["normative_guard"]["bypassed"] = False

        return "MISSING_REF", []

    run_meta["normative_guard"]["triggered"] = False
    run_meta["normative_guard"]["bypassed"] = bool(bypass_required_support_gate)

    return answer_text, references_structured


# Intent classification functions moved to intent_router.py (Phase 8a)
# Re-export for backward compatibility
from .intent_router import (  # noqa: E402, F401
    _intent_match_signals,
    _detect_intent_cues,
    classify_question_intent,
    classify_question_intent_with_router,
    _apply_answer_policy_to_claim_intent,
)


def _engineering_apply_answer_policy_requirements_enforcement(
    *,
    answer_text: str,
    answer_policy: Any,
    engineering_json_mode: bool,
) -> tuple[str, dict[str, Any]]:
    """Enforce minimal structured requirements output for ENGINEERING (text mode).

    - Ensures section 3 has at least N bullets when requested.
    - Optionally inserts an audit-evidence subsection (as neutral placeholders) when requested.
    """

    dbg: dict[str, Any] = {
        "applied": False,
        "skipped_reason": None,
        "min_section3_bullets": getattr(answer_policy, "min_section3_bullets", None),
        "include_audit_evidence": bool(
            getattr(answer_policy, "include_audit_evidence", False)
        ),
        "added_section3_bullets": 0,
        "added_audit_section": False,
    }

    if engineering_json_mode:
        dbg["skipped_reason"] = "json_mode"
        return str(answer_text or ""), dbg

    txt = str(answer_text or "")
    if not txt.strip() or txt.strip() == "MISSING_REF":
        dbg["skipped_reason"] = "empty_or_missing_ref"
        return txt, dbg

    intent_category = (
        str(getattr(answer_policy, "intent_category", "") or "").strip().upper()
    )
    if intent_category != "REQUIREMENTS":
        dbg["skipped_reason"] = "not_requirements_category"
        return txt, dbg

    # Locate section 3 range.
    lines = txt.splitlines()
    sec3_re = re.compile(r"^\s*3\.\s*Konkrete\s+systemkrav\s*$", flags=re.IGNORECASE)
    sec4_re = re.compile(r"^\s*4\.\s*", flags=re.IGNORECASE)

    sec3_idx = None
    for i, ln in enumerate(lines):
        if sec3_re.match(ln):
            sec3_idx = i
            break
    if sec3_idx is None:
        dbg["skipped_reason"] = "missing_section3"
        return txt, dbg

    end_idx = len(lines)
    for j in range(sec3_idx + 1, len(lines)):
        if sec4_re.match(lines[j]):
            end_idx = j
            break

    section3 = lines[sec3_idx + 1 : end_idx]
    bullet_lines = [ln for ln in section3 if ln.strip().startswith("-")]

    # Insert optional audit-evidence subsection if requested and not present.
    include_audit = bool(getattr(answer_policy, "include_audit_evidence", False))
    audit_heading = "Minimum ved tilsyn (evidens/artefakter)"
    has_audit = any(audit_heading.lower() in str(ln).lower() for ln in section3)
    inserts: list[str] = []
    if include_audit and not has_audit:
        inserts.extend(["", audit_heading, "- UTILSTRÆKKELIG_EVIDENS"])
        dbg["added_audit_section"] = True

    # Enforce minimum bullets in section 3.
    min_bullets = getattr(answer_policy, "min_section3_bullets", None)
    needed = 0
    if min_bullets is not None:
        try:
            mb = int(min_bullets)
        except Exception:  # noqa: BLE001
            mb = 0
        if mb > 0 and len(bullet_lines) < mb:
            needed = mb - len(bullet_lines)

    if needed > 0:
        inserts.extend(["- UTILSTRÆKKELIG_EVIDENS" for _ in range(needed)])
        dbg["added_section3_bullets"] = int(needed)

    if not inserts:
        dbg["skipped_reason"] = "no_changes_needed"
        return txt, dbg

    new_lines = lines[:end_idx] + inserts + lines[end_idx:]
    dbg["applied"] = True
    return "\n".join(new_lines), dbg


# Evidence classification moved to evidence_classifier.py (Phase 8a)
from .evidence_classifier import classify_evidence_type_from_metadata  # noqa: E402,F401


def _conditionalize_requirements_if_needed(answer_text: str) -> str:
    # Only deterministic text rewriting; keep it conservative.
    txt = str(answer_text or "")
    if not txt.strip():
        return txt

    # Only conditionalize when requirements appear tied to high-risk classification.
    if not re.search(r"(?i)\b(højrisiko|high[- ]risk)\b", txt):
        return txt

    # Trigger only when we see strong normative keywords.
    normative_re = re.compile(
        r"\b(MUST|SHALL|SHOULD)\b|\bskal\b|\bkræver\b", re.IGNORECASE
    )
    if not normative_re.search(txt):
        return txt

    lines = txt.splitlines()
    out: list[str] = []
    inserted_intro = False
    for line in lines:
        if not normative_re.search(line):
            out.append(line)
            continue

        if not inserted_intro:
            out.append(
                "Bemærk: Jeg kan ikke konkludere klassifikation (fx højrisiko) ud fra de fundne kilder alene. "
                "Krav afhænger af klassifikationen."
            )
            inserted_intro = True

        # Preserve bullet formatting/indentation.
        m = re.match(r"^(\s*(?:[-*]\s*)?)(.*)$", line)
        if m:
            prefix, rest = m.group(1), m.group(2)
            # Avoid double-conditionalizing.
            if re.match(r"\s*(Hvis|If)\b", rest, flags=re.IGNORECASE):
                out.append(line)
            else:
                out.append(
                    f"{prefix}Hvis systemet klassificeres som højrisiko (eller på anden måde omfattes af krav), så: {rest}"
                )
        else:
            out.append(
                f"Hvis systemet klassificeres som højrisiko (eller på anden måde omfattes af krav), så: {line}"
            )

    return "\n".join(out)


def _question_explicitly_assumes_classification(question: str) -> bool:
    q = str(question or "").lower()
    return any(k in q for k in ["højrisiko", "high-risk", "high risk"])


def _question_has_legal_assumption_bypass(question: str) -> bool:
    q = str(question or "").lower()
    return ("antag at" in q) or ("forudsat at" in q)


# Scope display transforms moved to scope_transforms.py (Phase 8a)
from .scope_transforms import (  # noqa: E402,F401
    _scope_extract_article_stk_litra_mentions,
    _scope_apply_litra_consistency_to_display,
    _engineering_remove_normative_bullets_from_systemkrav_section_for_scope,
    apply_scope_post_processing,
)


def _engineering_scaffold_answer_if_missing_anchors_and_citations(
    *,
    answer_text: str,
    references_structured_all: list[dict[str, Any]] | None,
    max_anchors: int = 5,
) -> str:
    """ENGINEERING safety scaffold.

    If the model fails to include both:
    - any bracket citations [n], and
    - any explicit anchor mentions (Artikel/Bilag/Betragtning),
    then we cannot safely keep the answer (it would fail closed later).

    This scaffold is conservative and only reports what sources were retrieved.
    It avoids introducing new requirements and binds every bullet to existing idx values.
    """

    txt = str(answer_text or "")
    if not txt.strip() or txt.strip() == "MISSING_REF":
        return txt

    if re.search(r"\[\d{1,3}\]", txt):
        return txt
    if re.search(r"(?i)\b(?:Artikel|Article|Art\.?)(?:\s+)\d{1,3}[a-z]?\b", txt):
        return txt
    if re.search(r"(?i)\b(?:bilag|annex)\s+(?:[ivxlcdm]+|\d{1,3})\b", txt):
        return txt
    if re.search(r"(?i)\b(?:betragtning(?:er)?|recital)\s+\d{1,4}\b", txt):
        return txt

    refs = [r for r in list(references_structured_all or []) if isinstance(r, dict)]
    if not refs:
        return txt

    # Build a neutral list of available sources.
    lines = [
        "Jeg har fundet følgende relevante kilder, men kan ikke give et specifikt svar ud fra dem:"
    ]
    count = 0
    for r in refs:
        if count >= max_anchors:
            break
        idx = r.get("idx")
        if not idx:
            continue

        # Try to construct a label.
        label = ""
        art = str(r.get("article") or "").strip()
        if art:
            label = f"Artikel {art}"
        else:
            rec = str(r.get("recital") or "").strip()
            if rec:
                label = f"Betragtning {rec}"
            else:
                ax = str(r.get("annex") or "").strip()
                if ax:
                    label = f"Bilag {ax}"

        if label:
            lines.append(f"- {label} [{idx}]")
            count += 1

    if count == 0:
        return txt

    return "\n".join(lines)


def apply_claim_stage_gate_for_legal(
    *,
    question: str,
    answer_text: str,
    references_structured_all: list[dict[str, Any]],
) -> LegalClaimGateResult:
    """Minimal claim-stage gate for LEGAL profile.

    - Does not change retrieval.
    - Does not affect ENGINEERING contract.
    - May conservatively rewrite answer + suppress irrelevant references.
    """

    intent = classify_question_intent(question)
    legal_assumption_bypass = _question_has_legal_assumption_bypass(question)
    evidence_types = {
        classify_evidence_type_from_metadata(r)
        for r in references_structured_all
        if isinstance(r, dict)
    }

    # Apply conditional rewriting first (it's conservative).
    answer_text = _conditionalize_requirements_if_needed(answer_text)

    allow_fallback = True
    final_refs = references_structured_all

    # 1. SCOPE mismatch guard.
    # If question is clearly SCOPE (applicability) but we only found ENFORCEMENT/CLASSIFICATION evidence,
    # we risk hallucinating applicability.
    if (
        intent == ClaimIntent.SCOPE
        and not legal_assumption_bypass
        and EvidenceType.SCOPE not in evidence_types
        and EvidenceType.DEFINITION not in evidence_types
    ):
        # If we have absolutely no scope/definition evidence, we should be careful.
        # But LEGAL profile prefers seeing the text over a hard block.
        # We inject a warning instead of blocking.
        # NOTE: Do NOT clear final_refs here - that breaks the downstream normative guard
        # which requires article support. The warning is sufficient for SCOPE mismatch.
        answer_text = f"Bemærk: Kilderne indeholder ikke direkte definitioner af anvendelsesområdet.\n\n{answer_text}"
        allow_fallback = False

    # 2. CLASSIFICATION mismatch guard.
    # If question is about CLASSIFICATION (high-risk) but we found NO classification evidence,
    # warn the user.
    elif (
        intent == ClaimIntent.CLASSIFICATION
        and not legal_assumption_bypass
        and EvidenceType.CLASSIFICATION not in evidence_types
    ):
        answer_text = f"Bemærk: Kilderne indeholder ikke specifikke klassifikationsregler (bilag/højrisiko).\n\n{answer_text}"
        # allow_fallback remains True

    # 3. ENFORCEMENT guard.
    # If question is about ENFORCEMENT (fines) but we found NO enforcement evidence,
    # warn the user.
    elif (
        intent == ClaimIntent.ENFORCEMENT
        and not legal_assumption_bypass
        and EvidenceType.ENFORCEMENT not in evidence_types
    ):
        # Often enforcement is general (GDPR Art 83), so missing specific evidence is common.
        # We just pass through but maybe add a generic disclaimer if answer is short.
        pass

    # 4. AI Act "High-Risk" assumption guard.
    # If the question assumes high-risk but the answer doesn't mention high-risk requirements,
    # we might be answering generically.
    elif _question_explicitly_assumes_classification(
        question
    ) and not _question_has_legal_assumption_bypass(question):
        # Check if answer actually addresses high-risk.
        if (
            "højrisiko" not in answer_text.lower()
            and "high-risk" not in answer_text.lower()
        ):
            answer_text = f"Bemærk: Du spurgte om højrisiko-systemer, men svaret er generelt.\n\n{answer_text}"
            # allow_fallback remains True

    # Default: pass through.
    # For LEGAL, we also enforce a "conclusion first" style if possible, but that's a prompt concern.
    # Here we just ensure we don't present misleading evidence.

    # Final check: If the answer is purely "I don't know" or empty, we allow fallback.
    if not answer_text.strip() or "jeg kan ikke" in answer_text.lower()[:50]:
        return LegalClaimGateResult(
            answer_text=answer_text,
            references_structured_all=final_refs,
            allow_reference_fallback=True,
        )

    # If we have a substantive answer, we try to enforce "conclusion first" by prepending a summary if missing.
    # (Skipped for now to avoid modifying text too much).

    # Inject a standard disclaimer for LEGAL profile if not present.
    lines = answer_text.splitlines()
    if not any("beslutningskontekst" in l for l in lines[-3:]):
        # Add standard legal disclaimer footer.
        k0 = [x for x in final_refs if isinstance(x, dict) and x.get("idx")]
        if k0:
            lines.append("")
            lines.append(
                f"- Afklar systemets formål, output og beslutningskontekst før endelig vurdering [{k0[0]['idx']}]."
            )
        else:
            lines.append("")
            lines.append(
                "- Afklar systemets formål, output og beslutningskontekst før endelig vurdering."
            )

    return LegalClaimGateResult(
        answer_text="\n".join(lines).strip(),
        references_structured_all=final_refs,
        allow_reference_fallback=allow_fallback,
    )


# ---------------------------------------------------------------------------
# Abstain logic (moved from rag.py Phase 3)
# ---------------------------------------------------------------------------


def _build_related_content_block(
    references_structured: list[dict[str, Any]] | None,
    question: str,
    *,
    max_items: int = 3,
) -> str:
    """Build a 'Related content' block from available references.

    Returns formatted string with related articles/topics and alternative questions.
    Returns empty string if no usable references.
    """
    refs = [r for r in list(references_structured or []) if isinstance(r, dict)]
    if not refs:
        return ""

    # Collect unique anchors with their titles
    seen_anchors: set[str] = set()
    related_items: list[str] = []

    for r in refs:
        if len(related_items) >= max_items:
            break

        # Try article
        art = str(r.get("article") or "").strip()
        art_title = str(r.get("article_title") or "").strip()
        if art and f"article:{art}" not in seen_anchors:
            seen_anchors.add(f"article:{art}")
            label = f"Artikel {art}"
            if art_title:
                label += f" - {art_title}"
            related_items.append(label)
            continue

        # Try annex
        annex = str(r.get("annex") or "").strip()
        annex_title = str(r.get("annex_title") or "").strip()
        if annex and f"annex:{annex}" not in seen_anchors:
            seen_anchors.add(f"annex:{annex}")
            label = f"Bilag {annex}"
            if annex_title:
                label += f" - {annex_title}"
            related_items.append(label)
            continue

        # Try recital
        rec = str(r.get("recital") or "").strip()
        if rec and f"recital:{rec}" not in seen_anchors:
            seen_anchors.add(f"recital:{rec}")
            related_items.append(f"Betragtning {rec}")
            continue

    if not related_items:
        return ""

    # Build alternative questions based on what we found
    alt_questions: list[str] = []
    q_lower = str(question or "").lower()

    # Suggest questions based on found articles
    for r in refs[:2]:
        art = str(r.get("article") or "").strip()
        art_title = str(r.get("article_title") or "").strip()
        if art:
            if art_title and len(alt_questions) < 2:
                # Use title for more specific suggestion
                title_lower = art_title.lower()
                if "definition" in title_lower or "begreb" in title_lower:
                    alt_questions.append(f"Hvad defineres i Artikel {art}?")
                elif "krav" in title_lower or "forpligt" in title_lower:
                    alt_questions.append(f"Hvilke krav stiller Artikel {art}?")
                elif "forbud" in title_lower:
                    alt_questions.append(f"Hvad forbyder Artikel {art}?")
                else:
                    alt_questions.append(f"Hvad handler Artikel {art} om?")
            elif len(alt_questions) < 2:
                alt_questions.append(f"Hvad siger Artikel {art}?")

    # Generic fallback suggestions if we don't have article-based ones
    if len(alt_questions) < 2:
        if "gdpr" in q_lower or "persondata" in q_lower or "databeskyttelse" in q_lower:
            if "Hvilke rettigheder har registrerede personer?" not in alt_questions:
                alt_questions.append("Hvilke rettigheder har registrerede personer?")
        elif "ai" in q_lower or "kunstig intelligens" in q_lower:
            if "Hvad er kravene til højrisiko AI-systemer?" not in alt_questions:
                alt_questions.append("Hvad er kravene til højrisiko AI-systemer?")

    # Build the block (Apple-style: bold headers, clean hierarchy)
    lines: list[str] = []
    lines.append("")
    lines.append("**Relateret indhold**")
    for item in related_items:
        lines.append(f"  · {item}")

    if alt_questions:
        lines.append("")
        lines.append("**Prøv at spørge**")
        for q in alt_questions[:2]:
            lines.append(f'  · "{q}"')

    return "\n".join(lines)


def should_abstain(
    *,
    question: str,
    hits: list[tuple[str, dict[str, Any]]],
    distances: list[float] | None,
    corpus_id: str,
    resolver: CorpusResolver | None,
    max_distance: float | None,
    hard_max_distance: float | None,
    allow_low_evidence_answer: bool,
    references_structured: list[dict[str, Any]] | None = None,
    corpus_scope: str = "single",
) -> str | None:
    """Determine if the system should abstain from answering.

    Returns an abstain message if the system should not answer, or None if it should proceed.

    This function checks multiple abstention conditions:
    1. Cross-law guardrail: question mentions different corpus than selected (single scope only)
    2. Empty hits: no evidence found
    3. Article/recital matching: explicit references must have matching chunks
    4. Distance thresholds: evidence must be relevant enough

    When abstaining due to distance thresholds, the message includes:
    - Related content that was retrieved (articles, annexes, recitals)
    - Alternative questions the user could ask based on available content
    """
    # Cross-law guardrail: if the user explicitly asks about a different law than the
    # selected corpus, abstain rather than answering from the wrong sources.
    # SKIP this check when corpus_scope is not "single" - the user has explicitly
    # selected multiple corpora, so mentioning multiple laws is expected.
    q_lower = str(question or "").lower()
    current_key = normalize_corpus_id(str(corpus_id or "").strip())

    mentioned: list[str] = []
    if resolver is not None and corpus_scope == "single":
        try:
            mentioned = resolver.mentioned_corpus_keys(q_lower)
        except Exception:  # noqa: BLE001
            mentioned = []

    # If the question mentions one or more known corpora/laws, require that it matches
    # the selected corpus. For multi-law questions, ask the user to pick one corpus.
    # This check is skipped when corpus_scope != "single" (handled above).
    if mentioned:
        if len(mentioned) != 1 or mentioned[0] != current_key:
            names: list[str] = list(mentioned)
            if resolver is not None:
                try:
                    names = [resolver.display_name_for(k) or k for k in mentioned]
                except Exception:  # noqa: BLE001
                    names = list(mentioned)

            if len(mentioned) == 1:
                target_key = mentioned[0]
                target_name = names[0]
                target_flag = target_key.replace("_", "-")
                return (
                    f"Jeg kan ikke svare om {target_name}, fordi dette corpus ikke er sat til {target_name}. "
                    f"Vælg `--law {target_flag}` (eller skift corpus) og prøv igen."
                )

            targets = ", ".join(names)
            return (
                f"Spørgsmålet nævner flere love/corpora ({targets}). "
                "Jeg kan kun svare ud fra ét corpus ad gangen. Vælg ét corpus (fx via `--law ...`) og stil spørgsmålet igen."
            )

    if not hits:
        return (
            "Jeg kan ikke finde tilstrækkelig evidens i de indlæste dokumenter til at svare sikkert. "
            "Prøv evt. at omformulere spørgsmålet, øge `rag.top_k`, eller indlæse flere relevante kilder."
        )

    # If the user references a specific article, ensure we have at least one hit tagged with that article.
    article = query_helpers._extract_article_ref(question)
    recital = query_helpers._extract_recital_ref(question)
    if article and query_helpers._looks_like_substantive_question(question):
        if not any(
            str((meta or {}).get("article", "")).upper() == article for _, meta in hits
        ):
            return (
                f"Jeg kan ikke finde chunks, der matcher Artikel {article}, i de hentede kilder. "
                "Jeg svarer ikke ved at gætte. Prøv evt. at spørge mere specifikt eller gen-indlæse kilderne."
            )

    # If the user references a specific recital, ensure we have at least one hit tagged with that recital.
    if recital and query_helpers._looks_like_substantive_question(question):
        if not any(
            str((meta or {}).get("recital", "")).strip() == recital for _, meta in hits
        ):
            return (
                f"Jeg kan ikke finde chunks, der matcher Betragtning {recital}, i de hentede kilder. "
                "Jeg svarer ikke ved at gætte. Prøv evt. at spørge mere specifikt eller gen-indlæse kilderne."
            )

    # Hard distance guardrail: even when a profile allows low-evidence answers, we should abstain
    # if retrieval relevance is extremely poor (typically indicates the question is outside corpus).
    used_distances = distances if distances is not None else []
    if used_distances:
        try:
            best = float(min(used_distances))
        except Exception:  # noqa: BLE001
            best = 0.0

        # Hard max distance check - use passed value, then settings (which reads env + config)
        hard_max = hard_max_distance
        if hard_max is None:
            try:
                from ..common.config_loader import load_settings

                settings = load_settings()
                hard_max = settings.rag_hard_max_distance
            except Exception:  # noqa: BLE001
                hard_max = None
        try:
            hard_max_f = float(hard_max) if hard_max is not None else 1.0
        except Exception:  # noqa: BLE001
            hard_max_f = 1.0

        if best > hard_max_f:
            if article and any(
                str((meta or {}).get("article", "")).upper() == article
                for _, meta in hits
            ):
                return None
            # Build related content block if we have references
            related_block = _build_related_content_block(
                references_structured, question
            )
            base_msg = "Jeg fandt relateret indhold, men ikke tilstrækkeligt grundlag for et sikkert svar."
            if related_block:
                return base_msg + "\n" + related_block
            return base_msg

    # Distance-based guardrail (optional): abstain if best hit is still too far.
    # Exception: if the user explicitly asks about an article and we actually retrieved chunks tagged
    # with that article, prefer answering rather than abstaining due to noisy distance signals.
    if max_distance is not None:
        if used_distances:
            best = min(used_distances)
            if best > max_distance:
                if article and any(
                    str((meta or {}).get("article", "")).upper() == article
                    for _, meta in hits
                ):
                    return None
                if recital and any(
                    str((meta or {}).get("recital", "")).strip() == recital
                    for _, meta in hits
                ):
                    return None
                if allow_low_evidence_answer:
                    return None
                # Build related content block if we have references
                related_block = _build_related_content_block(
                    references_structured, question
                )
                base_msg = "Jeg fandt relateret indhold, men ikke tilstrækkeligt grundlag for et sikkert svar."
                if related_block:
                    return base_msg + "\n" + related_block
                return base_msg

    return None


# ---------------------------------------------------------------------------
# Claim-stage gates (extracted from rag.py answer_structured)
# ---------------------------------------------------------------------------


@dataclass
class ClaimStageGateResult:
    """Result of applying claim-stage gates."""

    answer_text: str
    did_abstain: bool = False
    bypass_required_support_gate: bool = False
    debug: Dict[str, Any] = field(default_factory=dict)


def apply_claim_stage_gates(
    *,
    answer_text: str,
    question: str,
    user_profile: UserProfile,
    intent_used: ClaimIntent,
    has_used_scope_or_def: bool,
    has_used_classification: bool,
    references_structured_all: List[Dict[str, Any]],
    inject_enforcement_citations_fn: Optional[
        Callable[[str, List[Dict[str, Any]]], str]
    ] = None,
) -> ClaimStageGateResult:
    """Apply claim-stage gates that depend on evidence actually used in the answer.

    These gates may replace the answer text with a conservative response when
    the required evidence type is not present.

    Args:
        answer_text: The current answer text.
        question: The user's question.
        user_profile: User profile (LEGAL/ENGINEERING).
        intent_used: The classified intent for this question.
        has_used_scope_or_def: Whether scope/definition evidence was used.
        has_used_classification: Whether classification evidence was used.
        references_structured_all: All available structured references.
        inject_enforcement_citations_fn: Optional function to inject enforcement citations.

    Returns:
        ClaimStageGateResult with potentially modified answer and flags.
    """
    result = ClaimStageGateResult(answer_text=answer_text)
    debug: Dict[str, Any] = {}

    # Determine bypass conditions
    legal_assumption_bypass = (
        user_profile == UserProfile.LEGAL
    ) and _question_has_legal_assumption_bypass(question)
    legal_classification_assumed = (user_profile == UserProfile.LEGAL) and (
        _question_explicitly_assumes_classification(question) or legal_assumption_bypass
    )
    engineering_classification_assumed = (user_profile == UserProfile.ENGINEERING) and (
        _question_explicitly_assumes_classification(question)
        or _question_has_legal_assumption_bypass(question)
    )

    debug["legal_assumption_bypass"] = legal_assumption_bypass
    debug["legal_classification_assumed"] = legal_classification_assumed
    debug["engineering_classification_assumed"] = engineering_classification_assumed

    # ENGINEERING SCOPE gate: no categorical JA/NEJ without scope/definition evidence
    if (
        user_profile == UserProfile.ENGINEERING
        and intent_used == ClaimIntent.SCOPE
        and not has_used_scope_or_def
    ):
        result.answer_text = (
            "1. Klassifikation og betingelser\n"
            "- Kan ikke afgøres ud fra den foreliggende evidens.\n\n"
            "2. Relevante juridiske forpligtelser\n"
            "- Jeg kan ikke udlede anvendelsesområdet uden kilder om definitioner/anvendelsesområde.\n\n"
            "3. Konkrete systemkrav\n"
            "- (Ingen konkrete krav kan udledes, før scope er afklaret.)\n\n"
            "4. Åbne spørgsmål / risici\n"
            "- Hvad er systemets formål og konkrete outputs (score, anbefaling, beslutning)?\n"
            "- Bruges output til beslutninger med væsentlig påvirkning, og i hvilken kontekst?"
        )
        result.did_abstain = True
        result.bypass_required_support_gate = True
        debug["gate_applied"] = "engineering_scope_no_evidence"

    # ENGINEERING ENFORCEMENT gate: describe procedures but no system requirements
    if (
        user_profile == UserProfile.ENGINEERING
        and intent_used == ClaimIntent.ENFORCEMENT
    ):
        result.answer_text = (
            "1. Klassifikation og betingelser\n"
            "- AFHÆNGER AF (hvilken rolle/jurisdiktion og hvilket regelsæt der spørges til).\n\n"
            "2. Relevante juridiske forpligtelser\n"
            "- Jeg kan beskrive håndhævelse (kompetente myndigheder, procedurer, klageveje og sanktionsrammer) ud fra kilderne.\n\n"
            "3. Konkrete systemkrav\n"
            "- (Ingen konkrete systemkrav i ENGINEERING-profilen for håndhævelsesspørgsmål.)\n\n"
            "4. Åbne spørgsmål / risici\n"
            "- Hvilken rolle har I (udbyder/ibrugtager/distributør), og i hvilken jurisdiktion?"
        )
        debug["gate_applied"] = "engineering_enforcement"

    # CLASSIFICATION evidence restriction (both profiles)
    if intent_used == ClaimIntent.CLASSIFICATION and not (
        user_profile == UserProfile.LEGAL and legal_classification_assumed
    ):
        if not has_used_classification:
            if user_profile == UserProfile.ENGINEERING:
                result.answer_text = (
                    "1. Klassifikation og betingelser\n"
                    "- Kan ikke afgøres ud fra den foreliggende evidens, om systemet er 'højrisiko'.\n\n"
                    "2. Relevante juridiske forpligtelser\n"
                    "- Jeg kan ikke basere klassifikation på håndhævelse/tilsynsbestemmelser alene.\n\n"
                    "3. Konkrete systemkrav\n"
                    "- (Ingen krav kan udledes, før klassifikation er afklaret.)\n\n"
                    "4. Åbne spørgsmål / risici\n"
                    "- Hvad er systemets formål og brugssituation?\n"
                    "- Hvilke domæner/aktiviteter vedrører det, og hvem påvirkes?"
                )
                debug["gate_applied"] = "engineering_classification_no_evidence"
            elif user_profile == UserProfile.LEGAL:
                result.answer_text = (
                    "Kan ikke afgøres ud fra den foreliggende evidens, om systemet er 'højrisiko'. "
                    "Klassifikation afhænger af systemets anvendelse og relevante kategorier.\n\n"
                    "Afklar venligst kort:\n"
                    "- Hvad er formålet og beslutnings-/påvirkningskonteksten?\n"
                    "- Hvilken sektor/brugssituation og hvilke berørte personer/brugere?"
                )
                debug["gate_applied"] = "legal_classification_no_evidence"

    # CLASSIFICATION → REQUIREMENTS gate
    mentions_high_risk = bool(
        re.search(r"(?i)\b(højrisiko|high[- ]risk)\b", str(question or ""))
        or re.search(r"(?i)\b(højrisiko|high[- ]risk)\b", str(result.answer_text or ""))
    )
    if mentions_high_risk and not has_used_classification:
        if (
            user_profile == UserProfile.ENGINEERING
            and intent_used == ClaimIntent.REQUIREMENTS
        ):
            if not engineering_classification_assumed:
                result.answer_text = (
                    "Krav kan ikke fastlægges, før klassifikation er afklaret."
                )
                result.did_abstain = True
                result.bypass_required_support_gate = True
                debug["gate_applied"] = "engineering_requirements_needs_classification"
        elif user_profile == UserProfile.ENGINEERING and contains_normative_claim(
            result.answer_text
        ):
            if not engineering_classification_assumed:
                result.answer_text = (
                    "Krav kan ikke fastlægges, før klassifikation er afklaret."
                )
                result.did_abstain = True
                result.bypass_required_support_gate = True
                debug["gate_applied"] = "engineering_normative_needs_classification"
        elif (
            user_profile == UserProfile.LEGAL
            and contains_normative_claim(result.answer_text)
            and not legal_classification_assumed
        ):
            result.answer_text = _conditionalize_requirements_if_needed(
                result.answer_text
            )
            debug["gate_applied"] = "legal_conditionalize_requirements"

    # ENGINEERING + ENFORCEMENT: inject neutral hjemmel citations
    if (
        user_profile == UserProfile.ENGINEERING
        and intent_used == ClaimIntent.ENFORCEMENT
    ):
        if inject_enforcement_citations_fn is not None:
            result.answer_text = inject_enforcement_citations_fn(
                answer_text=result.answer_text,
                references_structured_all=references_structured_all,
            )
            debug["enforcement_citations_injected"] = True

    result.debug = debug
    return result


@dataclass
class PolicyStageResult:
    """Result from the policy stage (Stage 4a) of answer_structured.

    Contains all outputs from policy gate application:
    - Modified answer text
    - Flags for abstain/bypass decisions
    - Debug information for run_meta
    """

    answer_text: str
    references_structured_all: List[Dict[str, Any]]
    did_abstain: bool = False
    bypass_required_support_gate: bool = False
    legal_allow_reference_fallback: bool = True
    intent_used: ClaimIntent | None = None
    debug: Dict[str, Any] = field(default_factory=dict)


def apply_pre_engineering_policy_gates(
    *,
    answer_text: str,
    question: str,
    resolved_profile: UserProfile,
    references_structured_all: List[Dict[str, Any]],
    claim_intent_from_run_meta: str | None,
    classify_intent_fn: Callable[[str], ClaimIntent],
    run_meta: Dict[str, Any],
    corpus_debug_on: bool = False,
) -> PolicyStageResult:
    """Stage 4a-pre: Apply policy gates that run BEFORE engineering answer building.

    This includes:
    - LEGAL claim-stage gate (applies to LEGAL profile only)
    - Intent determination

    The claim-stage gates (SCOPE/CLASSIFICATION/REQUIREMENTS evidence checks)
    must run AFTER engineering answer building because they need to analyze
    the final answer text for mentions of high-risk/requirements.

    Args:
        answer_text: The answer text from generation stage
        question: The user's question
        resolved_profile: User profile (LEGAL/ENGINEERING)
        references_structured_all: All structured references
        claim_intent_from_run_meta: Intent from run_meta (from planning stage)
        classify_intent_fn: Function to classify question intent
        run_meta: Run metadata dict (mutated in-place)
        corpus_debug_on: Whether corpus debug is enabled

    Returns:
        PolicyStageResult with pre-engineering policy gate outputs
    """
    result = PolicyStageResult(
        answer_text=answer_text,
        references_structured_all=list(references_structured_all or []),
    )
    debug: Dict[str, Any] = {}

    # LEGAL claim-stage gate: enforce conservative conclusions when
    # evidence type is mismatched (e.g., enforcement-only evidence for scope claims)
    legal_allow_reference_fallback = True
    if resolved_profile == UserProfile.LEGAL:
        gate = apply_claim_stage_gate_for_legal(
            question=question,
            answer_text=result.answer_text,
            references_structured_all=list(result.references_structured_all or []),
        )
        result.answer_text = str(gate.answer_text or "")
        result.references_structured_all = list(gate.references_structured_all or [])
        legal_allow_reference_fallback = bool(gate.allow_reference_fallback)
        debug["legal_gate"] = {
            "applied": True,
            "allow_reference_fallback": legal_allow_reference_fallback,
        }

    result.legal_allow_reference_fallback = legal_allow_reference_fallback

    # Determine intent_used from run_meta or classify
    try:
        intent_used = ClaimIntent(str(claim_intent_from_run_meta or ""))
    except Exception:  # noqa: BLE001
        intent_used = classify_intent_fn(question)

    result.intent_used = intent_used
    debug["intent_used"] = str(getattr(intent_used, "value", intent_used) or "")

    # Update run_meta with intent
    if corpus_debug_on:
        run_meta.setdefault("corpus_debug", {})
        run_meta["corpus_debug"]["intent_used"] = str(
            getattr(intent_used, "value", intent_used) or ""
        )

    result.debug = debug
    return result


def apply_post_engineering_policy_gates(
    *,
    answer_text: str,
    question: str,
    resolved_profile: UserProfile,
    references_structured_all: List[Dict[str, Any]],
    intent_used: ClaimIntent,
    classify_evidence_type_fn: Callable[[Dict[str, Any]], "EvidenceType"],
    select_references_used_fn: Callable[..., List[str]],
    inject_enforcement_citations_fn: Callable[..., str] | None,
    run_meta: Dict[str, Any],
    corpus_debug_on: bool = False,
) -> PolicyStageResult:
    """Stage 4a-post: Apply policy gates that run AFTER engineering answer building.

    This includes:
    - Evidence type classification (based on final answer text)
    - Claim-stage gates (SCOPE/CLASSIFICATION/REQUIREMENTS checks)

    These gates analyze the final engineering answer text to determine if
    it mentions high-risk systems or normative claims without proper evidence.

    Args:
        answer_text: The answer text AFTER engineering answer building
        question: The user's question
        resolved_profile: User profile (LEGAL/ENGINEERING)
        references_structured_all: All structured references
        intent_used: The classified intent (from pre-engineering stage)
        classify_evidence_type_fn: Function to classify evidence type
        select_references_used_fn: Function to select references used in answer
        inject_enforcement_citations_fn: Function to inject enforcement citations
        run_meta: Run metadata dict (mutated in-place)
        corpus_debug_on: Whether corpus debug is enabled

    Returns:
        PolicyStageResult with post-engineering policy gate outputs
    """
    result = PolicyStageResult(
        answer_text=answer_text,
        references_structured_all=list(references_structured_all or []),
        intent_used=intent_used,
    )
    debug: Dict[str, Any] = {}

    # Determine evidence types used in the answer
    used_chunk_ids = select_references_used_fn(
        answer_text=result.answer_text,
        references_structured=result.references_structured_all,
    )
    used_id_set = set(
        str(x or "").strip() for x in (used_chunk_ids or []) if str(x or "").strip()
    )
    used_refs = [
        r
        for r in list(result.references_structured_all or [])
        if isinstance(r, dict) and str(r.get("chunk_id") or "").strip() in used_id_set
    ]
    used_evidence_types = {
        classify_evidence_type_fn(dict(r or {}))
        for r in used_refs
        if isinstance(r, dict)
    }

    # Import EvidenceType locally to avoid circular imports
    from .types import EvidenceType

    has_used_scope_or_def = bool(
        {EvidenceType.SCOPE, EvidenceType.DEFINITION} & used_evidence_types
    )
    has_used_classification = EvidenceType.CLASSIFICATION in used_evidence_types

    debug["evidence_types"] = {
        "used_chunk_ids_count": len(used_chunk_ids),
        "used_refs_count": len(used_refs),
        "has_used_scope_or_def": has_used_scope_or_def,
        "has_used_classification": has_used_classification,
    }

    # Apply claim-stage gates
    gate_result = apply_claim_stage_gates(
        answer_text=result.answer_text,
        question=question,
        user_profile=resolved_profile,
        intent_used=intent_used,
        has_used_scope_or_def=has_used_scope_or_def,
        has_used_classification=has_used_classification,
        references_structured_all=list(result.references_structured_all or []),
        inject_enforcement_citations_fn=inject_enforcement_citations_fn,
    )
    result.answer_text = gate_result.answer_text
    if gate_result.did_abstain:
        result.did_abstain = True
    if gate_result.bypass_required_support_gate:
        result.bypass_required_support_gate = True
    if gate_result.debug:
        debug["claim_stage_gates"] = gate_result.debug

    # Update run_meta
    if result.did_abstain:
        run_meta.setdefault("abstain", {})
        run_meta["abstain"].update(
            {
                "abstained": True,
                "reason": "classification_required_before_requirements",
            }
        )

    if gate_result.debug:
        run_meta.setdefault("claim_stage_gates", {})
        run_meta["claim_stage_gates"].update(gate_result.debug)

    result.debug = debug
    return result


# Keep apply_all_policy_gates for backward compatibility (calls both phases)
def apply_all_policy_gates(
    *,
    answer_text: str,
    question: str,
    resolved_profile: UserProfile,
    references_structured_all: List[Dict[str, Any]],
    claim_intent_from_run_meta: str | None,
    classify_intent_fn: Callable[[str], ClaimIntent],
    classify_evidence_type_fn: Callable[[Dict[str, Any]], "EvidenceType"],
    select_references_used_fn: Callable[..., List[str]],
    inject_enforcement_citations_fn: Callable[..., str] | None,
    run_meta: Dict[str, Any],
    corpus_debug_on: bool = False,
) -> PolicyStageResult:
    """Stage 4a: Apply all policy gates (combined pre+post for compatibility).

    NOTE: This function is for backward compatibility. For correct behavior
    with ENGINEERING profile, use apply_pre_engineering_policy_gates() before
    engineering answer building, then apply_post_engineering_policy_gates() after.

    This combined version runs both phases on the same answer_text, which is
    only correct when the answer_text is already the final engineering answer.
    """
    # Pre-engineering phase (LEGAL gate + intent)
    pre_result = apply_pre_engineering_policy_gates(
        answer_text=answer_text,
        question=question,
        resolved_profile=resolved_profile,
        references_structured_all=references_structured_all,
        claim_intent_from_run_meta=claim_intent_from_run_meta,
        classify_intent_fn=classify_intent_fn,
        run_meta=run_meta,
        corpus_debug_on=corpus_debug_on,
    )

    # Post-engineering phase (claim-stage gates)
    post_result = apply_post_engineering_policy_gates(
        answer_text=pre_result.answer_text,
        question=question,
        resolved_profile=resolved_profile,
        references_structured_all=list(pre_result.references_structured_all or []),
        intent_used=pre_result.intent_used or classify_intent_fn(question),
        classify_evidence_type_fn=classify_evidence_type_fn,
        select_references_used_fn=select_references_used_fn,
        inject_enforcement_citations_fn=inject_enforcement_citations_fn,
        run_meta=run_meta,
        corpus_debug_on=corpus_debug_on,
    )

    # Merge results
    return PolicyStageResult(
        answer_text=post_result.answer_text,
        references_structured_all=list(post_result.references_structured_all or []),
        did_abstain=post_result.did_abstain,
        bypass_required_support_gate=post_result.bypass_required_support_gate,
        legal_allow_reference_fallback=pre_result.legal_allow_reference_fallback,
        intent_used=pre_result.intent_used,
        debug={**pre_result.debug, **post_result.debug},
    )


# apply_scope_post_processing removed — now imported from scope_transforms.py above


# ---------------------------------------------------------------------------
# Step 6.4: Convenience gate for answer_structured (extracted from RAGEngine)
# ---------------------------------------------------------------------------


def should_abstain_gate(
    *,
    question: str,
    hits: list[tuple[str, dict[str, Any]]],
    distances: list[float] | None = None,
    fallback_distances: list[float] | None = None,
    corpus_id: str = "",
    max_distance: float | None = None,
    hard_max_distance: float | None = None,
    resolver_fn: Callable | None = None,
    allow_low_evidence_answer: bool = False,
    references_structured: list[dict[str, Any]] | None = None,
    corpus_scope: str = "single",
) -> str | None:
    """Convenience wrapper that resolves resolver_fn and defaults distances.

    This replaces RAGEngine._should_abstain — gathers context then calls
    should_abstain() with explicit params.
    """
    resolver = None
    if resolver_fn is not None:
        try:
            resolver = resolver_fn()
        except Exception:  # noqa: BLE001
            pass

    used_distances = distances if distances is not None else (fallback_distances or [])

    return should_abstain(
        question=question,
        hits=hits,
        distances=used_distances,
        corpus_id=corpus_id,
        resolver=resolver,
        max_distance=max_distance,
        hard_max_distance=hard_max_distance,
        allow_low_evidence_answer=allow_low_evidence_answer,
        references_structured=references_structured,
        corpus_scope=corpus_scope,
    )
