"""Generation strategies for different output modes.

Strategy Pattern: One function per generation mode instead of one
530-line function with 80 conditional branches.

Each strategy implements the GenerationStrategy protocol and handles
one specific output mode (prose, legal JSON, engineering JSON).
"""

from __future__ import annotations

import json
from typing import Any, Callable, TYPE_CHECKING

from .generation_types import GenerationConfig, StructuredGenerationResult

if TYPE_CHECKING:
    from .planning import QueryContext


# ---------------------------------------------------------------------------
# Metadata sync: JSON mode generation result → run_meta
# ---------------------------------------------------------------------------


def sync_json_mode_results_to_run_meta(
    gen_result: StructuredGenerationResult,
    run_meta: dict[str, Any],
    allowed_idxs: set[int],
    contract_min_citations: int | None,
    answer_policy: Any | None = None,
) -> None:
    """Sync JSON mode generation result to run_meta for backwards compatibility.

    This function consolidates the metadata bookkeeping that was previously
    inline in answer_structured().

    Args:
        gen_result: The structured generation result.
        run_meta: The run metadata dict to update (mutated in-place).
        allowed_idxs: Set of allowed citation indices.
        contract_min_citations: Minimum citations required by contract.
        answer_policy: Optional answer policy with min_section3_bullets etc.
    """
    # Common fields for all modes
    run_meta["llm_calls_count"] = gen_result.debug.get("llm_calls_count", 0)
    run_meta["citations_source"] = gen_result.debug.get(
        "citations_source", "text_parse"
    )

    # Initialize engineering_json block
    run_meta.setdefault(
        "engineering_json",
        {
            "enabled": True,
            "json_parse_ok": None,
            "repair_retry_performed": False,
            "enrich_retry_performed": False,
            "allowed_idxs": [],
            "cited_idxs": [],
            "valid_cited": [],
            "min_citations": contract_min_citations or 0,
            "fail_reason": None,
        },
    )

    # Sync allowed indices
    sorted_allowed = sorted(allowed_idxs)
    run_meta["allowed_idxs"] = sorted_allowed
    run_meta["engineering_json"]["allowed_idxs"] = sorted_allowed
    run_meta["allowed_idxs_count"] = len(sorted_allowed)
    run_meta["engineering_json"]["allowed_idxs_count"] = len(sorted_allowed)

    # Sync parse/repair/enrich status
    run_meta["json_parse_ok"] = gen_result.debug.get("json_parse_ok")
    run_meta["engineering_json"]["json_parse_ok"] = gen_result.debug.get(
        "json_parse_ok"
    )

    run_meta["repair_retry_performed"] = gen_result.repair_attempts > 0
    run_meta["engineering_json"]["repair_retry_performed"] = (
        gen_result.repair_attempts > 0
    )

    run_meta["enrich_retry_performed"] = gen_result.enrich_attempts > 0
    run_meta["engineering_json"]["enrich_retry_performed"] = (
        gen_result.enrich_attempts > 0
    )

    run_meta["enrich_success"] = gen_result.debug.get("enrich_success")

    # Sync citation indices
    run_meta["cited_idxs"] = gen_result.cited_idxs
    run_meta["engineering_json"]["cited_idxs"] = gen_result.cited_idxs
    run_meta["cited_idxs_json"] = list(gen_result.cited_idxs)

    run_meta["valid_cited"] = gen_result.valid_cited_idxs
    run_meta["engineering_json"]["valid_cited"] = gen_result.valid_cited_idxs

    # Sync failure reasons
    run_meta["min_citations"] = contract_min_citations or 0
    run_meta["fail_reason"] = gen_result.fail_reason
    run_meta["final_fail_reason"] = gen_result.debug.get("final_fail_reason")
    run_meta["engineering_json"]["fail_reason"] = gen_result.fail_reason

    run_meta["strict_validation_failed_code"] = gen_result.debug.get(
        "strict_validation_failed_code"
    )
    run_meta["schema_only_fallback_used"] = gen_result.debug.get(
        "schema_only_fallback_used", False
    )

    run_meta["json_policy_enforced"] = True
    run_meta["engineering_json"]["json_policy_enforced"] = True

    # Surface policy knobs
    try:
        run_meta["policy_min_section3_bullets"] = (
            getattr(answer_policy, "min_section3_bullets", None)
            if answer_policy is not None
            else None
        )
        run_meta["policy_include_audit_evidence"] = (
            bool(getattr(answer_policy, "include_audit_evidence", False))
            if answer_policy is not None
            else False
        )
    except Exception:  # noqa: BLE001
        run_meta["policy_min_section3_bullets"] = None
        run_meta["policy_include_audit_evidence"] = False

    # Sync bullet counts
    run_meta["requirements_bullet_count"] = gen_result.debug.get(
        "requirements_bullet_count", 0
    )
    run_meta["audit_evidence_bullet_count"] = gen_result.debug.get(
        "audit_evidence_bullet_count", 0
    )
    run_meta["engineering_json"]["requirements_bullet_count"] = gen_result.debug.get(
        "requirements_bullet_count", 0
    )
    run_meta["engineering_json"]["audit_evidence_bullet_count"] = gen_result.debug.get(
        "audit_evidence_bullet_count", 0
    )

    # Sync final gate reason if MISSING_REF
    if gen_result.is_missing_ref:
        run_meta["final_gate_reason"] = (
            gen_result.fail_reason or "json_parse_or_schema_fail"
        )

    # Sync rendered text
    if gen_result.parsed_json is not None:
        try:
            run_meta["engineering_json"]["rendered_text"] = str(
                gen_result.answer_text or ""
            )
        except Exception:  # noqa: BLE001
            pass


# Backward-compat alias
_sync_json_mode_results_to_run_meta = sync_json_mode_results_to_run_meta


# ---------------------------------------------------------------------------
# Strategy 1: Prose Generation (no JSON validation)
# ---------------------------------------------------------------------------


def execute_prose_generation(
    prompt: str,
    llm_fn: Callable[[str], str],
    config: GenerationConfig,
    allowed_idxs: set[int],
    **kwargs: Any,
) -> StructuredGenerationResult:
    """Generate prose answer without JSON validation.

    Used for:
    - LEGAL profile with prose output
    - Any profile with require_json_schema=False

    Args:
        prompt: The complete prompt to send to the LLM.
        llm_fn: Callable that takes a prompt and returns LLM response.
        config: GenerationConfig with profile-specific settings.
        allowed_idxs: Set of valid citation indices.

    Returns:
        StructuredGenerationResult with prose answer text.
    """
    result = StructuredGenerationResult(
        answer_text="",
        raw_llm_response="",
        debug={
            "config_profile": config.profile.value,
            "config_require_json_schema": config.require_json_schema,
            "config_citation_mode": config.citation_mode,
            "allowed_idxs": sorted(allowed_idxs),
            "min_citations": config.min_citations,
        },
    )

    raw_answer = llm_fn(prompt)
    result.raw_llm_response = raw_answer
    result.answer_text = raw_answer
    result.debug["llm_calls_count"] = 1
    result.debug["citations_source"] = "text_parse"

    # For soft citation mode, extract citations from text
    if config.citation_mode == "soft":
        try:
            from .contract_validation import parse_bracket_citations

            cited = parse_bracket_citations(raw_answer)
            result.cited_idxs = sorted(cited)
            result.valid_cited_idxs = sorted(set(cited) & allowed_idxs)
        except Exception:  # noqa: BLE001
            pass

    return result


# ---------------------------------------------------------------------------
# Strategy 2: Legal JSON Generation (soft validation with prose fallback)
# ---------------------------------------------------------------------------


def execute_legal_json_generation(
    prompt: str,
    llm_fn: Callable[[str], str],
    config: GenerationConfig,
    allowed_idxs: set[int],
    **kwargs: Any,
) -> StructuredGenerationResult:
    """Generate LEGAL JSON answer with soft validation.

    Uses a simpler schema than ENGINEERING:
    - Only 'summary' is required
    - Falls back to prose on JSON failure (soft_json_fallback)

    Args:
        prompt: The complete prompt to send to the LLM.
        llm_fn: Callable that takes a prompt and returns LLM response.
        config: GenerationConfig with profile-specific settings.
        allowed_idxs: Set of valid citation indices.

    Returns:
        StructuredGenerationResult with legal answer (JSON or prose fallback).
    """
    from .legal_json_mode import (
        extract_cited_idxs as _legal_json_extract_cited_idxs,
        render_legal_answer_text as _legal_json_render_text,
        validate_legal_answer_json_soft as _legal_json_validate_soft,
    )

    result = StructuredGenerationResult(
        answer_text="",
        raw_llm_response="",
        debug={
            "config_profile": config.profile.value,
            "config_require_json_schema": config.require_json_schema,
            "config_citation_mode": config.citation_mode,
            "allowed_idxs": sorted(allowed_idxs),
            "min_citations": config.min_citations,
            "json_schema_type": "legal",
        },
    )

    raw_answer = llm_fn(prompt)
    result.raw_llm_response = raw_answer

    # Try soft JSON validation (graceful degradation)
    validated_obj, error_reason = _legal_json_validate_soft(raw_answer)

    if validated_obj is not None:
        # JSON validation succeeded
        result.debug["json_parse_ok"] = True
        result.debug["legal_json_valid"] = True
        result.parsed_json = validated_obj

        # Extract citations from JSON
        cited_idxs = sorted(_legal_json_extract_cited_idxs(validated_obj))
        result.cited_idxs = cited_idxs
        result.valid_cited_idxs = sorted(set(cited_idxs) & allowed_idxs)
        result.debug["cited_idxs"] = cited_idxs
        result.debug["citations_source"] = "legal_json"

        # Render to text
        result.answer_text = _legal_json_render_text(validated_obj)

    elif config.soft_json_fallback:
        # JSON validation failed, fall back to prose
        result.debug["json_parse_ok"] = False
        result.debug["legal_json_valid"] = False
        result.debug["legal_json_fallback_to_prose"] = True
        result.debug["legal_json_error"] = error_reason
        result.debug["citations_source"] = "text_parse"

        # Use raw answer as prose
        result.answer_text = raw_answer

        # Extract citations from text (soft mode)
        try:
            from .contract_validation import parse_bracket_citations

            cited = parse_bracket_citations(raw_answer)
            result.cited_idxs = sorted(cited)
            result.valid_cited_idxs = sorted(set(cited) & allowed_idxs)
        except Exception:  # noqa: BLE001
            pass

    else:
        # JSON validation failed and no fallback allowed
        result.debug["json_parse_ok"] = False
        result.debug["legal_json_valid"] = False
        result.debug["legal_json_error"] = error_reason
        result.failed = True
        result.fail_reason = error_reason or "legal_json_fail"
        result.answer_text = "MISSING_REF"

    result.debug["llm_calls_count"] = 1
    return result


# ---------------------------------------------------------------------------
# Strategy 3: Engineering JSON Generation (strict validation with repair/enrich)
# ---------------------------------------------------------------------------


def execute_engineering_json_generation(
    prompt: str,
    llm_fn: Callable[[str], str],
    config: GenerationConfig,
    allowed_idxs: set[int],
    *,
    answer_policy: Any | None = None,
    claim_intent: Any | None = None,
    references_structured_all: list[dict] | None = None,
    **kwargs: Any,
) -> StructuredGenerationResult:
    """Generate ENGINEERING JSON answer with strict validation.

    Full pipeline with:
    1. LLM call with JSON schema prompt
    2. Parse + validate JSON
    3. Repair loop if schema invalid
    4. Enrich loop if citations insufficient
    5. Policy validation (min bullets etc)
    6. Render to text

    Args:
        prompt: The complete prompt to send to the LLM.
        llm_fn: Callable that takes a prompt and returns LLM response.
        config: GenerationConfig with profile-specific settings.
        allowed_idxs: Set of valid citation indices.
        answer_policy: Optional answer policy for bullet count requirements.
        claim_intent: Optional claim intent for policy routing.
        references_structured_all: List of reference dicts (unused but accepted for API compat).

    Returns:
        StructuredGenerationResult with engineering answer (JSON validated).
    """
    # Import engineering JSON validation functions
    from .engineering_json_mode import (
        EngineeringJSONValidationError,
        bullet_counts as _engineering_json_bullet_counts,
        extract_cited_idxs as _engineering_json_extract_cited_idxs,
        render_engineering_answer_text as _engineering_json_render_text,
        validate_engineering_answer_json as _engineering_json_validate,
        validate_engineering_answer_json_schema_only as _engineering_json_validate_schema_only,
        validate_engineering_answer_json_policy as _engineering_json_validate_policy,
    )

    # Import repair/enrich prompt builders from engineering_json_mode (SRP)
    from .engineering_json_mode import (
        build_repair_prompt,
        build_enrich_prompt,
        build_enrich_prompt_policy_completion,
        get_padding_citations,
    )

    result = StructuredGenerationResult(
        answer_text="",
        raw_llm_response="",
        debug={
            "config_profile": config.profile.value,
            "config_require_json_schema": config.require_json_schema,
            "config_citation_mode": config.citation_mode,
            "allowed_idxs": sorted(allowed_idxs),
            "min_citations": config.min_citations,
        },
    )

    sorted_allowed = sorted(allowed_idxs)
    min_cit = config.min_citations or 0

    # Track LLM calls for bounded retry budget
    llm_call_count = 0

    def _call_llm(p: str) -> str:
        nonlocal llm_call_count
        llm_call_count += 1
        return llm_fn(p)

    # State for repair/enrich loops
    repair_used = False
    enrich_used = False
    schema_only_fallback_used = False

    def _record_fail(reason: str) -> None:
        result.failed = True
        result.fail_reason = reason
        result.debug["fail_reason"] = reason
        result.debug["final_fail_reason"] = reason

    def _parse_and_validate(
        raw: str, *, allow_schema_only_fallback: bool = True
    ) -> dict[str, Any] | None:
        nonlocal repair_used, schema_only_fallback_used

        # Try to parse JSON
        try:
            parsed = json.loads(str(raw or ""))
            result.debug["json_parse_ok"] = True
        except Exception:  # noqa: BLE001
            result.debug["json_parse_ok"] = False
            if repair_used or llm_call_count >= config.max_repair_attempts:
                _record_fail("parse_fail")
                return None
            # Attempt repair
            repair_used = True
            result.repair_attempts += 1
            result.debug["repair_retry_performed"] = True
            raw2 = _call_llm(
                build_repair_prompt(
                    base_prompt=prompt,
                    raw_output=str(raw or ""),
                    why="json_parse_fail",
                )
            )
            try:
                parsed = json.loads(str(raw2 or ""))
                result.debug["json_parse_ok"] = True
            except Exception:  # noqa: BLE001
                _record_fail("parse_fail")
                return None

        # Validate against schema
        try:
            validated = _engineering_json_validate(parsed)
            return validated
        except EngineeringJSONValidationError as exc:
            # Handle normative_no_citation: allow schema-only fallback for enrich
            if exc.code == "normativ_no_citation":
                result.debug["strict_validation_failed_code"] = "normativ_no_citation"
                if allow_schema_only_fallback and config.allow_schema_fallback:
                    schema_only_fallback_used = True
                    result.debug["schema_only_fallback_used"] = True
                    try:
                        return _engineering_json_validate_schema_only(parsed)
                    except EngineeringJSONValidationError:
                        _record_fail("schema_fail")
                        return None
                _record_fail("normativ_no_citation")
                return None

            # Handle unknown fields: attempt repair
            if exc.code == "unknown_fields":
                if repair_used or llm_call_count >= config.max_repair_attempts:
                    _record_fail("schema_fail")
                    return None
                repair_used = True
                result.repair_attempts += 1
                result.debug["repair_retry_performed"] = True
                raw2 = _call_llm(
                    build_repair_prompt(
                        base_prompt=prompt,
                        raw_output=str(raw or ""),
                        why="unknown_fields",
                    )
                )
                try:
                    parsed2 = json.loads(str(raw2 or ""))
                except Exception:  # noqa: BLE001
                    _record_fail("parse_fail")
                    return None
                try:
                    return _engineering_json_validate(parsed2)
                except EngineeringJSONValidationError as exc2:
                    if exc2.code == "normativ_no_citation":
                        result.debug["strict_validation_failed_code"] = (
                            "normativ_no_citation"
                        )
                        if allow_schema_only_fallback and config.allow_schema_fallback:
                            schema_only_fallback_used = True
                            result.debug["schema_only_fallback_used"] = True
                            try:
                                return _engineering_json_validate_schema_only(parsed2)
                            except EngineeringJSONValidationError:
                                _record_fail("schema_fail")
                                return None
                    _record_fail(
                        "schema_fail"
                        if exc2.code != "normativ_no_citation"
                        else "normativ_no_citation"
                    )
                    return None

            _record_fail(
                "schema_fail"
                if exc.code != "normativ_no_citation"
                else "normativ_no_citation"
            )
            return None

    def _policy_validate_and_record(
        obj: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        ok, details = _engineering_json_validate_policy(
            obj,
            allowed_idxs=set(allowed_idxs),
            answer_policy=answer_policy,
            intent_selected=claim_intent,
        )
        result.debug["requirements_bullet_count"] = int(
            details.get("requirements_bullet_count") or 0
        )
        result.debug["audit_evidence_bullet_count"] = int(
            details.get("audit_evidence_bullet_count") or 0
        )
        return ok, details

    def _apply_padding(obj: dict[str, Any]) -> None:
        """Apply deterministic padding for policy-min-items issues."""
        pad_cits = get_padding_citations(sorted_allowed)
        if not pad_cits:
            return

        try:
            intent_sel = (
                str(getattr(claim_intent, "value", claim_intent) or "").strip().upper()
            )
        except Exception:  # noqa: BLE001
            intent_sel = ""

        # Requirements padding
        try:
            min_req = (
                getattr(answer_policy, "min_section3_bullets", None)
                if answer_policy is not None
                else None
            )
            min_req_i = int(min_req) if min_req is not None else 0
        except Exception:  # noqa: BLE001
            min_req_i = 0

        req_arr = obj.get("system_requirements")
        if not isinstance(req_arr, list):
            req_arr = []
            obj["system_requirements"] = req_arr

        if intent_sel == "REQUIREMENTS" and min_req_i > 0:
            while len(req_arr) < min_req_i:
                req_arr.append(
                    {
                        "level": "INFO",
                        "text": "UTILSTRÆKKELIG_EVIDENS",
                        "citations": list(pad_cits),
                    }
                )

        # Audit evidence padding
        include_audit = (
            bool(getattr(answer_policy, "include_audit_evidence", False))
            if answer_policy is not None
            else False
        )
        if include_audit:
            audit_arr = obj.get("audit_evidence_bullets")
            if not isinstance(audit_arr, list):
                audit_arr = []
                obj["audit_evidence_bullets"] = audit_arr
            while len(audit_arr) < 3:
                audit_arr.append(
                    {"text": "UTILSTRÆKKELIG_EVIDENS", "citations": list(pad_cits)}
                )

    # ----- Base LLM call -----
    raw_answer = _call_llm(prompt)
    result.raw_llm_response = raw_answer

    validated_obj = _parse_and_validate(str(raw_answer or ""))
    if not validated_obj:
        result.answer_text = "MISSING_REF"
        result.debug["llm_calls_count"] = llm_call_count
        result.debug["citations_source"] = "json_mode"
        return result

    # Extract and validate citations
    cited_idxs = sorted(_engineering_json_extract_cited_idxs(validated_obj))
    result.cited_idxs = cited_idxs
    result.debug["cited_idxs"] = cited_idxs

    # Check for hallucinated citations
    hallucinated = sorted(set(cited_idxs) - set(allowed_idxs))
    if hallucinated:
        _record_fail("hallucinated_idx")
        result.answer_text = "MISSING_REF"
        result.debug["llm_calls_count"] = llm_call_count
        result.debug["citations_source"] = "json_mode"
        return result

    valid_cited = sorted(set(cited_idxs) & set(allowed_idxs))
    result.valid_cited_idxs = valid_cited
    result.debug["valid_cited"] = valid_cited

    # Determine if enrich is needed
    enrich_required_min = max(1, min_cit) if schema_only_fallback_used else min_cit

    # Early fail if impossible to meet requirements
    if (
        schema_only_fallback_used
        and enrich_required_min > 0
        and len(allowed_idxs) < enrich_required_min
    ):
        _record_fail("INSUFFICIENT_ALLOWED_IDXS_FOR_REQUIRED_CITATIONS")
        result.answer_text = "MISSING_REF"
        result.debug["llm_calls_count"] = llm_call_count
        result.debug["citations_source"] = "json_mode"
        return result

    # Policy validation
    policy_ok, policy_details = _policy_validate_and_record(validated_obj)

    # Determine enrich type
    should_enrich = False
    enrich_kind = "citations_only"

    if config.enable_enrich and not enrich_used:
        if not policy_ok:
            should_enrich = True
            enrich_kind = "policy_completion"
        elif (
            schema_only_fallback_used
            and enrich_required_min > 0
            and len(allowed_idxs) >= enrich_required_min
        ):
            should_enrich = True
        elif (
            min_cit > 0 and len(allowed_idxs) >= min_cit and len(valid_cited) < min_cit
        ):
            should_enrich = True

    if should_enrich:
        if llm_call_count >= config.max_repair_attempts:
            # No budget left: apply padding fallback
            if enrich_kind == "policy_completion":
                pad_cits = get_padding_citations(sorted_allowed)
                if not pad_cits:
                    _record_fail("schema_fail")
                    result.answer_text = "MISSING_REF"
                else:
                    _apply_padding(validated_obj)
                    result.debug["final_fail_reason"] = (
                        "ANSWER_POLICY_PADDED_WITH_INSUFFICIENT_EVIDENCE"
                    )
                    _policy_validate_and_record(validated_obj)
                    result.answer_text = _engineering_json_render_text(validated_obj)
                    result.parsed_json = validated_obj
            else:
                _record_fail(
                    "NORMATIV_NO_CITATION_AFTER_ENRICH"
                    if schema_only_fallback_used
                    else "min_citations_not_met"
                )
                result.answer_text = "MISSING_REF"
        else:
            # Perform enrich
            enrich_used = True
            result.enrich_attempts += 1
            result.debug["enrich_retry_performed"] = True
            baseline_counts = dict(_engineering_json_bullet_counts(validated_obj))
            allowed_marks = " ".join(f"[{i}]" for i in sorted_allowed)

            # Calculate missing bullets for policy enrich
            missing_req = 0
            missing_audit = 0
            if enrich_kind == "policy_completion":
                for err in list((policy_details or {}).get("errors") or []):
                    if not isinstance(err, dict):
                        continue
                    if err.get("code") == "min_requirements_bullets_not_met":
                        missing_req = max(
                            0, int(err.get("need") or 0) - int(err.get("have") or 0)
                        )
                    if err.get("code") == "min_audit_evidence_bullets_not_met":
                        missing_audit = max(
                            0, int(err.get("need") or 0) - int(err.get("have") or 0)
                        )

            enrich_prompt = (
                build_enrich_prompt_policy_completion(
                    base_prompt=prompt,
                    raw_json=json.dumps(validated_obj, ensure_ascii=False),
                    allowed_marks=allowed_marks,
                    missing_requirements_bullets=missing_req,
                    missing_audit_bullets=missing_audit,
                    min_citations=enrich_required_min
                    if schema_only_fallback_used
                    else min_cit,
                )
                if enrich_kind == "policy_completion"
                else build_enrich_prompt(
                    base_prompt=prompt,
                    raw_json=json.dumps(validated_obj, ensure_ascii=False),
                    min_citations=enrich_required_min
                    if schema_only_fallback_used
                    else min_cit,
                    allowed_marks=allowed_marks,
                )
            )

            enrich_out = _call_llm(enrich_prompt)

            # After enrich, require strict validation (no schema-only fallback)
            validated_obj_2 = _parse_and_validate(
                str(enrich_out or ""), allow_schema_only_fallback=False
            )
            if not validated_obj_2:
                _record_fail(
                    "NORMATIV_NO_CITATION_AFTER_ENRICH"
                    if schema_only_fallback_used
                    else "schema_fail"
                )
                result.answer_text = "MISSING_REF"
                result.debug["enrich_success"] = False
            else:
                after_counts = dict(_engineering_json_bullet_counts(validated_obj_2))

                # Validate enrich didn't add unexpected bullets
                enrich_valid = True
                if enrich_kind == "citations_only":
                    if any(
                        int(after_counts.get(k) or 0) > int(baseline_counts.get(k) or 0)
                        for k in baseline_counts
                    ):
                        _record_fail("schema_fail")
                        result.answer_text = "MISSING_REF"
                        enrich_valid = False
                elif enrich_kind == "policy_completion":
                    # Only allow increasing system_requirements/audit_evidence_bullets
                    if int(after_counts.get("obligations") or 0) > int(
                        baseline_counts.get("obligations") or 0
                    ):
                        _record_fail("schema_fail")
                        result.answer_text = "MISSING_REF"
                        enrich_valid = False
                    elif int(after_counts.get("open_questions") or 0) > int(
                        baseline_counts.get("open_questions") or 0
                    ):
                        _record_fail("schema_fail")
                        result.answer_text = "MISSING_REF"
                        enrich_valid = False
                    else:
                        # Cap growth deterministically
                        sys_max = int(
                            baseline_counts.get("system_requirements") or 0
                        ) + max(0, missing_req)
                        aud_max = int(
                            baseline_counts.get("audit_evidence_bullets") or 0
                        ) + max(0, missing_audit)
                        if int(after_counts.get("system_requirements") or 0) > sys_max:
                            arr = validated_obj_2.get("system_requirements")
                            if isinstance(arr, list):
                                validated_obj_2["system_requirements"] = list(arr)[
                                    :sys_max
                                ]
                        if (
                            int(after_counts.get("audit_evidence_bullets") or 0)
                            > aud_max
                        ):
                            arr = validated_obj_2.get("audit_evidence_bullets")
                            if isinstance(arr, list):
                                validated_obj_2["audit_evidence_bullets"] = list(arr)[
                                    :aud_max
                                ]

                if enrich_valid and result.answer_text != "MISSING_REF":
                    # Re-extract citations after enrich
                    cited_idxs = sorted(
                        _engineering_json_extract_cited_idxs(validated_obj_2)
                    )
                    result.cited_idxs = cited_idxs
                    result.debug["cited_idxs"] = cited_idxs

                    hallucinated = sorted(set(cited_idxs) - set(allowed_idxs))
                    if hallucinated:
                        _record_fail("hallucinated_idx")
                        result.answer_text = "MISSING_REF"
                    else:
                        valid_cited = sorted(set(cited_idxs) & set(allowed_idxs))
                        result.valid_cited_idxs = valid_cited
                        result.debug["valid_cited"] = valid_cited
                        validated_obj = validated_obj_2

                        # For schema-only fallback: require meeting minimum after enrich
                        if (
                            schema_only_fallback_used
                            and enrich_required_min > 0
                            and len(valid_cited) < enrich_required_min
                        ):
                            _record_fail("NORMATIV_NO_CITATION_AFTER_ENRICH")
                            result.answer_text = "MISSING_REF"
                            result.debug["enrich_success"] = False
                        else:
                            # Re-check policy after enrich
                            policy_ok_2, _ = _policy_validate_and_record(validated_obj)
                            if enrich_kind == "policy_completion" and not policy_ok_2:
                                # Apply padding fallback
                                pad_cits = get_padding_citations(sorted_allowed)
                                if not pad_cits:
                                    _record_fail("schema_fail")
                                    result.answer_text = "MISSING_REF"
                                else:
                                    _apply_padding(validated_obj)
                                    result.debug["final_fail_reason"] = (
                                        "ANSWER_POLICY_PADDED_WITH_INSUFFICIENT_EVIDENCE"
                                    )
                                    _policy_validate_and_record(validated_obj)
                            result.debug["enrich_success"] = True

    # Final min-citations gate
    if result.answer_text != "MISSING_REF":
        if (
            min_cit > 0
            and len(allowed_idxs) >= min_cit
            and len(result.valid_cited_idxs) < min_cit
        ):
            _record_fail(
                "NORMATIV_NO_CITATION_AFTER_ENRICH"
                if schema_only_fallback_used
                else "min_citations_not_met"
            )
            result.answer_text = "MISSING_REF"
        elif min_cit > 0 and len(allowed_idxs) < min_cit:
            _record_fail("INSUFFICIENT_ALLOWED_IDXS_FOR_MIN_CITATIONS")
            result.answer_text = "MISSING_REF"
        else:
            # Final policy check
            policy_ok_final, _ = _policy_validate_and_record(validated_obj)
            if not policy_ok_final:
                pad_cits = get_padding_citations(sorted_allowed)
                if not pad_cits:
                    _record_fail("schema_fail")
                    result.answer_text = "MISSING_REF"
                else:
                    _apply_padding(validated_obj)
                    result.debug["final_fail_reason"] = (
                        "ANSWER_POLICY_PADDED_WITH_INSUFFICIENT_EVIDENCE"
                    )
                    _policy_validate_and_record(validated_obj)

    # Render final answer
    if result.answer_text != "MISSING_REF":
        result.answer_text = _engineering_json_render_text(validated_obj)
        result.parsed_json = validated_obj

    result.debug["llm_calls_count"] = llm_call_count
    result.debug["citations_source"] = "json_mode"
    result.debug["repair_retry_performed"] = result.repair_attempts > 0
    result.debug["enrich_retry_performed"] = result.enrich_attempts > 0

    return result


# ---------------------------------------------------------------------------
# Strategy Router (Dependency Inversion)
# ---------------------------------------------------------------------------


def get_strategy(config: GenerationConfig) -> Callable[..., StructuredGenerationResult]:
    """Select strategy based on config (Strategy Pattern + DI).

    Args:
        config: GenerationConfig with profile-specific settings.

    Returns:
        The appropriate generation strategy function.
    """
    if not config.require_json_schema:
        return execute_prose_generation
    if config.json_schema_type == "legal":
        return execute_legal_json_generation
    return execute_engineering_json_generation


def execute_structured_generation(
    *,
    prompt: str,
    llm_fn: Callable[[str], str],
    config: GenerationConfig,
    allowed_idxs: set[int],
    references_structured_all: list[dict] | None = None,
    answer_policy: Any | None = None,
    claim_intent: Any | None = None,
    strategy: Callable[..., StructuredGenerationResult] | None = None,
) -> StructuredGenerationResult:
    """Unified entry point - uses injected or auto-selected strategy.

    This function provides backward compatibility and enables dependency injection
    for testing. It routes to the appropriate strategy based on config.

    Args:
        prompt: The complete prompt to send to the LLM.
        llm_fn: Callable that takes a prompt and returns LLM response.
        config: GenerationConfig with profile-specific settings.
        allowed_idxs: Set of valid citation indices.
        references_structured_all: List of reference dicts with idx field.
        answer_policy: Optional answer policy for bullet count requirements.
        claim_intent: Optional claim intent for policy routing.
        strategy: Optional strategy to use (for DI/testing). Auto-selected if None.

    Returns:
        StructuredGenerationResult with answer text and metadata.
    """
    if strategy is None:
        strategy = get_strategy(config)

    return strategy(
        prompt,
        llm_fn,
        config,
        allowed_idxs,
        references_structured_all=references_structured_all,
        answer_policy=answer_policy,
        claim_intent=claim_intent,
    )


# ---------------------------------------------------------------------------
# Citation Retry (non-JSON prose mode)
# ---------------------------------------------------------------------------


def execute_citation_retry_if_needed(
    answer_text: str,
    prompt: str,
    llm_fn: Callable[[str], str],
    allowed_idxs: set[int],
    min_citations: int,
    run_meta: dict[str, Any],
    strip_references_fn: Callable[[str], str],
) -> str:
    """ENGINEERING: retry LLM if citations are insufficient (non-JSON mode only).

    This function consolidates the retry logic that was previously inline
    in rag.py answer_structured() (lines 2025-2100).

    Args:
        answer_text: The current answer text from LLM.
        prompt: The original prompt sent to LLM.
        llm_fn: Function to call LLM (prompt -> response).
        allowed_idxs: Set of allowed citation indices.
        min_citations: Minimum citations required.
        run_meta: The run metadata dict to update (mutated in-place).
        strip_references_fn: Function to strip trailing references section.

    Returns:
        The answer text (possibly from retry).
    """
    import re

    try:
        from .contract_validation import parse_bracket_citations
    except Exception:  # noqa: BLE001
        parse_bracket_citations = None  # type: ignore[assignment]

    def _extract_citations(txt: str) -> set[int]:
        body = strip_references_fn(str(txt or ""))
        if parse_bracket_citations:
            return set(parse_bracket_citations(body))
        return {int(m.group(1)) for m in re.finditer(r"\[(\d{1,3})\]", body)}

    def _valid_cited_count(txt: str) -> int:
        cited = _extract_citations(txt)
        return len(cited & allowed_idxs)

    valid_count_1 = _valid_cited_count(answer_text)
    do_retry = bool(
        (min_citations > 0)
        and (len(allowed_idxs) >= min_citations)
        and (valid_count_1 < min_citations)
    )

    body_1 = strip_references_fn(str(answer_text or ""))
    run_meta["llm_retry"] = {
        "profile": "ENGINEERING",
        "min_citations": int(min_citations),
        "allowed": int(len(allowed_idxs)),
        "retry_performed": False,
        "valid_cited_before": int(valid_count_1),
        "valid_cited_after": None,
        "cited_idxs_before": sorted(_extract_citations(body_1)),
        "cited_idxs_after": None,
    }

    if do_retry:
        allowed_marks = " ".join(f"[{i}]" for i in sorted(allowed_idxs))
        retry_hint = (
            "RETNING: Omskriv dit svar, så alle krav/konklusioner har bracket-citations [idx].\n"
            f"Brug mindst {int(min_citations)} unikke [idx].\n"
            f"Du må kun bruge [idx] der findes i KILDER: {allowed_marks}\n"
            "Citationsmarkører SKAL stå inline i svaret; tilføj IKKE en 'KILDER'/'REFERENCER' sektion.\n"
            "Tilføj ingen nye påstande — kun tilføj/ret citations.\n"
            "Bevar samme struktur og punktindhold.\n"
        )
        answer_text = llm_fn(f"{prompt}\n\n{retry_hint}")
        run_meta["llm_retry"]["retry_performed"] = True
        body_2 = strip_references_fn(str(answer_text or ""))
        run_meta["llm_retry"]["cited_idxs_after"] = sorted(_extract_citations(body_2))
        run_meta["llm_retry"]["valid_cited_after"] = int(
            _valid_cited_count(answer_text)
        )

    return answer_text


# ---------------------------------------------------------------------------
# Engineering Answer Post-Processing
# (Moved from generation.py as part of SOLID refactoring - Fase 12.1)
# ---------------------------------------------------------------------------


def build_engineering_answer(
    raw_interpretation: str,
    ctx: "QueryContext",
    references_structured: list[dict[str, Any]],
    reference_lines: list[str],
    distances: list[float],
    total_retrieved: int = 0,
    citable_count: int = 0,
    min_citable_required: int = 2,
) -> str:
    """Build a concise ENGINEERING answer with hard evidence gating.

    Requirements:
    - Do not include a references section in the answer body.
    - Keep engineering terms in English.
    - Avoid generic best practices unless supported by citations.

    Args:
        raw_interpretation: The raw LLM interpretation.
        ctx: Query context (used for type signature compatibility).
        references_structured: Structured references list.
        reference_lines: Reference lines for display.
        distances: Distance scores for references.
        total_retrieved: Total chunks retrieved.
        citable_count: Number of citable chunks.
        min_citable_required: Minimum citable chunks required.

    Returns:
        Formatted engineering answer string.
    """
    # Preserve internal fail-closed marker.
    ri = str(raw_interpretation or "").lstrip()
    if ri.startswith("MISSING_REF"):
        return "MISSING_REF"

    # Hard evidence gating.
    if citable_count < int(min_citable_required):
        missing_bits = "Der mangler citerbare uddrag med Artikel/Bilag/Kapitel (metadata/location_id/tekst)."
        fix_bits = "Forbedr ingestion (EUR-Lex strukturelle ids) eller afgræns spørgsmålet til en specifik Artikel/Bilag/Kapitel."
        out_parts: list[str] = []
        out_parts.append("1. Klassifikation og betingelser")
        out_parts.append("- UTILSTRÆKKELIG_EVIDENS")
        out_parts.append(
            f"- Hentet: {int(total_retrieved)}, citerbart: {int(citable_count)} (min. påkrævet: {int(min_citable_required)})."
        )

        out_parts.append("\n2. Relevante juridiske forpligtelser")
        out_parts.append(
            "- (utilstrækkelig citerbar evidens til at udlede forpligtelser)"
        )

        out_parts.append("\n3. Konkrete systemkrav")
        out_parts.append("- (utilstrækkelig citerbar evidens til at specificere krav)")

        out_parts.append("\n4. Åbne spørgsmål / risici")
        out_parts.append(f"- Mangler: {missing_bits}")
        out_parts.append(f"- Forslag: {fix_bits}")
        out_parts.append("- Manglende reference-præcision")
        return "\n".join(out_parts)

    # When evidence is sufficient, we trust the LLM's structured interpretation
    # because the prompt now enforces the Engineering Answer Contract structure.
    # We append a small evidence note for transparency.

    stats_line = f"\n\n(Evidens: {int(citable_count)} citerbart uddrag hentet)"
    return raw_interpretation + stats_line


# ---------------------------------------------------------------------------
# Step 6.6: Generation stage extracted from RAGEngine._execute_generation
# ---------------------------------------------------------------------------


def execute_generation_stage(
    *,
    question: str,
    context: str,
    kilder_block: str,
    ctx: "QueryContext",
    effective_plan: Any,
    synthesis_context: Any | None,
    resolved_profile: Any,
    effective_policy: Any,
    claim_intent_final: Any,
    references_structured_all: list[dict[str, Any]],
    contract_min_citations: int | None,
    history_context: str,
    corpus_debug_on: bool,
    run_meta: dict[str, Any],
    llm_fn: Callable[[str], str],
    resolver_fn: Callable,
) -> str:
    """Execute LLM generation stage: build prompt, call LLM, handle retries.

    Mutates run_meta with generation debug info.

    Returns:
        The generated answer text.
    """
    import os
    from . import instrumentation
    from . import text_transforms
    from .prompt_builder import (
        build_prompt,
        focus_block_for_prompt,
        build_answer_policy_suffix,
        build_citation_requirement_suffix,
        build_multi_corpus_prompt,
    )
    from .synthesis_router import SynthesisMode
    from .planning import UserProfile

    focus_block = focus_block_for_prompt(ctx.focus)

    # Determine JSON mode settings per profile
    engineering_json_mode = bool(
        resolved_profile == UserProfile.ENGINEERING
        and instrumentation.is_engineering_json_mode_enabled()
    )
    legal_json_mode = bool(
        resolved_profile == UserProfile.LEGAL
        and instrumentation.is_legal_json_mode_enabled()
    )

    # Build base prompt
    if synthesis_context is not None and synthesis_context.mode != SynthesisMode.SINGLE:
        prompt = build_multi_corpus_prompt(
            mode=synthesis_context.mode,
            question=question,
            context=context,
            kilder_block=kilder_block,
            references_structured=list(references_structured_all or []),
            user_profile=resolved_profile.name
            if hasattr(resolved_profile, "name")
            else str(resolved_profile),
            resolver=resolver_fn(),
        )
    else:
        prompt = build_prompt(
            ctx=ctx,
            plan=effective_plan,
            context=context,
            focus_block=focus_block,
            contract_min_citations=contract_min_citations,
            legal_json_mode=legal_json_mode,
            history_context=history_context,
        )

    # Policy-driven answer shaping
    answer_policy = getattr(effective_policy, "answer_policy", None)
    if os.getenv("DISABLE_ANSWER_POLICY", "").strip().lower() in ("1", "true", "yes"):
        answer_policy = None
    prompt += build_answer_policy_suffix(answer_policy, resolved_profile)

    # Track LLM calls
    if "llm_calls_count" not in run_meta:
        run_meta["llm_calls_count"] = 0

    def _call_llm_counted(p: str) -> str:
        run_meta["llm_calls_count"] = int(run_meta.get("llm_calls_count") or 0) + 1
        return llm_fn(p)

    # Citation requirement suffix
    prompt += build_citation_requirement_suffix(
        user_profile=resolved_profile,
        contract_min_citations=contract_min_citations,
        references_structured_all=references_structured_all,
        json_mode=engineering_json_mode,
    )

    # Build allowed_idxs set
    allowed_idxs: set[int] = set()
    for r in list(references_structured_all or []):
        if not isinstance(r, dict):
            continue
        try:
            allowed_idxs.add(int(r.get("idx")))
        except Exception:  # noqa: BLE001
            continue

    # Create generation config
    profile_json_mode = (
        engineering_json_mode
        if resolved_profile == UserProfile.ENGINEERING
        else legal_json_mode
    )
    gen_config = GenerationConfig.for_profile(
        resolved_profile,
        contract_min_citations=contract_min_citations,
        json_mode_enabled=profile_json_mode,
    )

    # Execute unified generation pipeline
    gen_result = execute_structured_generation(
        prompt=prompt,
        llm_fn=_call_llm_counted,
        config=gen_config,
        allowed_idxs=allowed_idxs,
        references_structured_all=list(references_structured_all or []),
        answer_policy=answer_policy,
        claim_intent=claim_intent_final,
    )

    answer_text = gen_result.answer_text

    # Update run_meta
    run_meta["engineering_json_mode"] = bool(engineering_json_mode)
    run_meta["legal_json_mode"] = bool(legal_json_mode)
    run_meta["llm_calls_count"] = int(run_meta.get("llm_calls_count") or 0)
    run_meta["citations_source"] = gen_result.debug.get(
        "citations_source", "text_parse"
    )

    if engineering_json_mode and gen_result is not None:
        sync_json_mode_results_to_run_meta(
            gen_result=gen_result,
            run_meta=run_meta,
            allowed_idxs=allowed_idxs,
            contract_min_citations=contract_min_citations,
            answer_policy=answer_policy,
        )

        instrumentation._debug_dump_run_meta(
            run_meta=run_meta,
            stage="after_engineering_json_mode",
            extra={
                "answer_preview": str(answer_text or "")[:160],
                "allowed_idxs_count": int(run_meta.get("allowed_idxs_count") or 0),
                "cited_idxs_json": list(run_meta.get("cited_idxs") or []),
                "valid_cited": list(run_meta.get("valid_cited") or []),
                "llm_calls_count": int(run_meta.get("llm_calls_count") or 0),
            },
        )
    else:
        run_meta.setdefault("allowed_idxs", sorted(allowed_idxs))
        run_meta.setdefault("allowed_idxs_count", len(allowed_idxs))

    # Debug-only: force a deterministic answer body
    forced_answer = None
    if not engineering_json_mode and corpus_debug_on:
        raw_forced = str(os.getenv("RAG_DEBUG_FORCE_ANSWER", "") or "").strip()
        if raw_forced:
            forced_answer = raw_forced
            answer_text = forced_answer
            run_meta.setdefault("corpus_debug", {})
            run_meta["corpus_debug"].update({"forced_answer_used": True})

    if not bool(run_meta.get("engineering_json_mode")):
        run_meta.setdefault("allowed_idxs", [])
        run_meta.setdefault(
            "allowed_idxs_count", int(len(run_meta.get("allowed_idxs") or []))
        )

    # ENGINEERING: deterministic 1x retry if citations are missing
    if (
        resolved_profile == UserProfile.ENGINEERING
        and contract_min_citations is not None
        and forced_answer is None
        and not bool(run_meta.get("engineering_json_mode"))
    ):
        try:
            min_cit = int(contract_min_citations)
        except Exception:  # noqa: BLE001
            min_cit = 0

        retry_allowed_idxs: set[int] = set()
        for r in list(references_structured_all or []):
            if not isinstance(r, dict):
                continue
            try:
                retry_allowed_idxs.add(int(r.get("idx")))
            except Exception:  # noqa: BLE001
                continue

        answer_text = execute_citation_retry_if_needed(
            answer_text=str(answer_text or ""),
            prompt=prompt,
            llm_fn=_call_llm_counted,
            allowed_idxs=retry_allowed_idxs,
            min_citations=min_cit,
            run_meta=run_meta,
            strip_references_fn=text_transforms._strip_trailing_references_section,
        )

    return str(answer_text or "")
