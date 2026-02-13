from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def env_flag_enabled(name: str) -> bool:
    return str((__import__("os").environ.get(name) or "")).strip().lower() in _TRUTHY_ENV_VALUES


_NORMATIVE_MARKERS = (
    "SKAL",
    "BØR",
    "MÅ IKKE",
    "KRÆVER",
    "FORPLIGT",
    "MUST",
    "SHALL",
    "REQUIRED",
    "OBLIG",
)


def _is_normative_text(text: str) -> bool:
    t = (text or "").upper()
    return any(m in t for m in _NORMATIVE_MARKERS)


@dataclass(frozen=True)
class EngineeringJSONValidationError(Exception):
    code: str
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.code}: {self.message}"


_TOP_KEYS_REQUIRED = {"classification", "obligations", "system_requirements", "open_questions"}
_TOP_KEYS_OPTIONAL = {"audit_evidence_bullets", "requirements_bullets"}


def _require_dict(obj: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise EngineeringJSONValidationError("schema_fail", f"Expected object at {path}.")
    return obj


def _require_list(obj: Any, *, path: str) -> list[Any]:
    if not isinstance(obj, list):
        raise EngineeringJSONValidationError("schema_fail", f"Expected array at {path}.")
    return obj


def _require_str(obj: Any, *, path: str) -> str:
    if not isinstance(obj, str) or not obj.strip():
        raise EngineeringJSONValidationError("schema_fail", f"Expected non-empty string at {path}.")
    return obj


def _require_int(obj: Any, *, path: str) -> int:
    if not isinstance(obj, int):
        raise EngineeringJSONValidationError("schema_fail", f"Expected integer at {path}.")
    return obj


def _require_no_unknown_fields(d: dict[str, Any], *, allowed: set[str], path: str) -> None:
    extra = sorted(set(d.keys()) - set(allowed))
    missing = sorted(set(allowed) - set(d.keys()))
    if extra or missing:
        raise EngineeringJSONValidationError(
            "unknown_fields",
            f"Invalid keys at {path}.",
            details={"path": path, "extra": extra, "missing": missing},
        )


def _require_no_unknown_fields_required_optional(
    d: dict[str, Any], *, required: set[str], optional: set[str], path: str
) -> None:
    allowed = set(required) | set(optional)
    extra = sorted(set(d.keys()) - allowed)
    missing_required = sorted(set(required) - set(d.keys()))
    if extra or missing_required:
        raise EngineeringJSONValidationError(
            "unknown_fields",
            f"Invalid keys at {path}.",
            details={"path": path, "extra": extra, "missing": missing_required},
        )


def _validate_citations(value: Any, *, path: str, require_nonempty: bool) -> list[int]:
    arr = _require_list(value, path=path)
    out: list[int] = []
    for i, v in enumerate(arr):
        idx = _require_int(v, path=f"{path}[{i}]")
        if idx <= 0:
            raise EngineeringJSONValidationError("schema_fail", f"Citation idx must be > 0 at {path}[{i}].")
        out.append(idx)
    if require_nonempty and len(out) == 0:
        raise EngineeringJSONValidationError("normativ_no_citation", f"Missing citations at {path}.")
    return out


def validate_engineering_answer_json(obj: Any) -> dict[str, Any]:
    root = _require_dict(obj, path="$")
    _require_no_unknown_fields_required_optional(
        root,
        required=_TOP_KEYS_REQUIRED,
        optional=_TOP_KEYS_OPTIONAL,
        path="$",
    )

    classification = _require_dict(root.get("classification"), path="$.classification")
    _require_no_unknown_fields(classification, allowed={"status", "text", "citations"}, path="$.classification")
    status = _require_str(classification.get("status"), path="$.classification.status")
    if status not in {"JA", "NEJ", "AFHÆNGER_AF"}:
        raise EngineeringJSONValidationError("schema_fail", "Invalid classification.status.")
    _require_str(classification.get("text"), path="$.classification.text")
    _validate_citations(classification.get("citations"), path="$.classification.citations", require_nonempty=True)

    obligations = _require_list(root.get("obligations"), path="$.obligations")
    for i, item in enumerate(obligations):
        d = _require_dict(item, path=f"$.obligations[{i}]")
        _require_no_unknown_fields(d, allowed={"title", "text", "citations"}, path=f"$.obligations[{i}]")
        title = _require_str(d.get("title"), path=f"$.obligations[{i}].title")
        text = _require_str(d.get("text"), path=f"$.obligations[{i}].text")
        require = _is_normative_text(f"{title} {text}")
        _validate_citations(d.get("citations"), path=f"$.obligations[{i}].citations", require_nonempty=require)

    system_requirements = _require_list(root.get("system_requirements"), path="$.system_requirements")
    for i, item in enumerate(system_requirements):
        d = _require_dict(item, path=f"$.system_requirements[{i}]")
        _require_no_unknown_fields(d, allowed={"level", "text", "citations"}, path=f"$.system_requirements[{i}]")
        level = _require_str(d.get("level"), path=f"$.system_requirements[{i}].level")
        if level not in {"SKAL", "BØR", "INFO"}:
            raise EngineeringJSONValidationError("schema_fail", f"Invalid system_requirements[{i}].level.")
        text = _require_str(d.get("text"), path=f"$.system_requirements[{i}].text")
        require = bool(level in {"SKAL", "BØR"}) or _is_normative_text(text)
        _validate_citations(d.get("citations"), path=f"$.system_requirements[{i}].citations", require_nonempty=require)

    open_questions = _require_list(root.get("open_questions"), path="$.open_questions")
    for i, item in enumerate(open_questions):
        d = _require_dict(item, path=f"$.open_questions[{i}]")
        _require_no_unknown_fields(d, allowed={"question", "why", "citations"}, path=f"$.open_questions[{i}]")
        q = _require_str(d.get("question"), path=f"$.open_questions[{i}].question")
        why = _require_str(d.get("why"), path=f"$.open_questions[{i}].why")
        require = _is_normative_text(f"{q} {why}")
        _validate_citations(d.get("citations"), path=f"$.open_questions[{i}].citations", require_nonempty=require)

    # Optional: audit evidence bullets for supervision/audit readiness.
    if "audit_evidence_bullets" in root:
        audit = _require_list(root.get("audit_evidence_bullets"), path="$.audit_evidence_bullets")
        for i, item in enumerate(audit):
            d = _require_dict(item, path=f"$.audit_evidence_bullets[{i}]")
            _require_no_unknown_fields(d, allowed={"text", "citations"}, path=f"$.audit_evidence_bullets[{i}]")
            _require_str(d.get("text"), path=f"$.audit_evidence_bullets[{i}].text")
            _validate_citations(d.get("citations"), path=f"$.audit_evidence_bullets[{i}].citations", require_nonempty=False)

    # Optional alias (additive): requirements bullets without SKAL/BØR levels.
    if "requirements_bullets" in root:
        reqs = _require_list(root.get("requirements_bullets"), path="$.requirements_bullets")
        for i, item in enumerate(reqs):
            d = _require_dict(item, path=f"$.requirements_bullets[{i}]")
            _require_no_unknown_fields(d, allowed={"text", "citations"}, path=f"$.requirements_bullets[{i}]")
            _require_str(d.get("text"), path=f"$.requirements_bullets[{i}].text")
            _validate_citations(d.get("citations"), path=f"$.requirements_bullets[{i}].citations", require_nonempty=False)

    return root


def validate_engineering_answer_json_schema_only(obj: Any) -> dict[str, Any]:
    """Schema-only validation.

    Accepts empty citations arrays. Intended for the enrich-retry workflow where
    we keep text unchanged and only repair citations.
    """

    root = _require_dict(obj, path="$")
    _require_no_unknown_fields_required_optional(
        root,
        required=_TOP_KEYS_REQUIRED,
        optional=_TOP_KEYS_OPTIONAL,
        path="$",
    )

    classification = _require_dict(root.get("classification"), path="$.classification")
    _require_no_unknown_fields(classification, allowed={"status", "text", "citations"}, path="$.classification")
    status = _require_str(classification.get("status"), path="$.classification.status")
    if status not in {"JA", "NEJ", "AFHÆNGER_AF"}:
        raise EngineeringJSONValidationError("schema_fail", "Invalid classification.status.")
    _require_str(classification.get("text"), path="$.classification.text")
    _validate_citations(classification.get("citations"), path="$.classification.citations", require_nonempty=False)

    obligations = _require_list(root.get("obligations"), path="$.obligations")
    for i, item in enumerate(obligations):
        d = _require_dict(item, path=f"$.obligations[{i}]")
        _require_no_unknown_fields(d, allowed={"title", "text", "citations"}, path=f"$.obligations[{i}]")
        _require_str(d.get("title"), path=f"$.obligations[{i}].title")
        _require_str(d.get("text"), path=f"$.obligations[{i}].text")
        _validate_citations(d.get("citations"), path=f"$.obligations[{i}].citations", require_nonempty=False)

    system_requirements = _require_list(root.get("system_requirements"), path="$.system_requirements")
    for i, item in enumerate(system_requirements):
        d = _require_dict(item, path=f"$.system_requirements[{i}]")
        _require_no_unknown_fields(d, allowed={"level", "text", "citations"}, path=f"$.system_requirements[{i}]")
        level = _require_str(d.get("level"), path=f"$.system_requirements[{i}].level")
        if level not in {"SKAL", "BØR", "INFO"}:
            raise EngineeringJSONValidationError("schema_fail", f"Invalid system_requirements[{i}].level.")
        _require_str(d.get("text"), path=f"$.system_requirements[{i}].text")
        _validate_citations(d.get("citations"), path=f"$.system_requirements[{i}].citations", require_nonempty=False)

    open_questions = _require_list(root.get("open_questions"), path="$.open_questions")
    for i, item in enumerate(open_questions):
        d = _require_dict(item, path=f"$.open_questions[{i}]")
        _require_no_unknown_fields(d, allowed={"question", "why", "citations"}, path=f"$.open_questions[{i}]")
        _require_str(d.get("question"), path=f"$.open_questions[{i}].question")
        _require_str(d.get("why"), path=f"$.open_questions[{i}].why")
        _validate_citations(d.get("citations"), path=f"$.open_questions[{i}].citations", require_nonempty=False)

    if "audit_evidence_bullets" in root:
        audit = _require_list(root.get("audit_evidence_bullets"), path="$.audit_evidence_bullets")
        for i, item in enumerate(audit):
            d = _require_dict(item, path=f"$.audit_evidence_bullets[{i}]")
            _require_no_unknown_fields(d, allowed={"text", "citations"}, path=f"$.audit_evidence_bullets[{i}]")
            _require_str(d.get("text"), path=f"$.audit_evidence_bullets[{i}].text")
            _validate_citations(d.get("citations"), path=f"$.audit_evidence_bullets[{i}].citations", require_nonempty=False)

    if "requirements_bullets" in root:
        reqs = _require_list(root.get("requirements_bullets"), path="$.requirements_bullets")
        for i, item in enumerate(reqs):
            d = _require_dict(item, path=f"$.requirements_bullets[{i}]")
            _require_no_unknown_fields(d, allowed={"text", "citations"}, path=f"$.requirements_bullets[{i}]")
            _require_str(d.get("text"), path=f"$.requirements_bullets[{i}].text")
            _validate_citations(d.get("citations"), path=f"$.requirements_bullets[{i}].citations", require_nonempty=False)

    return root


def extract_cited_idxs(obj: dict[str, Any]) -> set[int]:
    cited: set[int] = set()

    def add(arr: Any) -> None:
        if isinstance(arr, list):
            for v in arr:
                if isinstance(v, int):
                    cited.add(v)

    cls = obj.get("classification") or {}
    add((cls or {}).get("citations"))

    for item in obj.get("obligations") or []:
        add((item or {}).get("citations"))
    for item in obj.get("system_requirements") or []:
        add((item or {}).get("citations"))
    for item in obj.get("open_questions") or []:
        add((item or {}).get("citations"))

    for item in obj.get("audit_evidence_bullets") or []:
        add((item or {}).get("citations"))
    for item in obj.get("requirements_bullets") or []:
        add((item or {}).get("citations"))

    return cited


def bullet_counts(obj: dict[str, Any]) -> dict[str, int]:
    return {
        "obligations": int(len(obj.get("obligations") or [])),
        "system_requirements": int(len(obj.get("system_requirements") or [])),
        "open_questions": int(len(obj.get("open_questions") or [])),
        "audit_evidence_bullets": int(len(obj.get("audit_evidence_bullets") or [])),
        "requirements_bullets": int(len(obj.get("requirements_bullets") or [])),
    }


def validate_engineering_answer_json_policy(
    answer_json: dict[str, Any],
    allowed_idxs: set[int] | list[int],
    answer_policy: Any,
    intent_selected: Any,
) -> tuple[bool, dict[str, Any]]:
    """Policy-aware validation for ENGINEERING JSON output.

    This is intentionally separate from schema validation:
    - schema is static and additive
    - policy controls min bullet counts and citation completeness for specific sections
    """

    allowed: set[int] = set()
    try:
        if isinstance(allowed_idxs, set):
            allowed = set(allowed_idxs)
        else:
            allowed = {int(v) for v in list(allowed_idxs or [])}
    except Exception:  # noqa: BLE001
        allowed = set()

    intent = str(getattr(intent_selected, "value", intent_selected) or "").strip().upper()
    ap = answer_policy

    if ap is None:
        req_items = answer_json.get("system_requirements")
        if not isinstance(req_items, list) or not req_items:
            req_items = answer_json.get("requirements_bullets") or []
        audit_items = answer_json.get("audit_evidence_bullets") or []
        details = {
            "ok": True,
            "intent_selected": intent,
            "requirements_bullet_count": int(len(req_items or [])) if isinstance(req_items, list) else 0,
            "audit_evidence_bullet_count": int(len(audit_items or [])) if isinstance(audit_items, list) else 0,
            "policy_min_section3_bullets": None,
            "policy_include_audit_evidence": False,
            "errors": [],
        }
        return True, details

    try:
        min_req = getattr(ap, "min_section3_bullets", None)
    except Exception:  # noqa: BLE001
        min_req = None

    try:
        include_audit = bool(getattr(ap, "include_audit_evidence", False))
    except Exception:  # noqa: BLE001
        include_audit = False

    def _get_req_bullets() -> list[dict[str, Any]]:
        arr = answer_json.get("system_requirements")
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, dict)]
        arr2 = answer_json.get("requirements_bullets")
        if isinstance(arr2, list):
            return [x for x in arr2 if isinstance(x, dict)]
        return []

    def _get_audit_bullets() -> list[dict[str, Any]]:
        arr = answer_json.get("audit_evidence_bullets")
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, dict)]
        return []

    req_bullets = _get_req_bullets()
    audit_bullets = _get_audit_bullets()

    req_count = int(len(req_bullets))
    audit_count = int(len(audit_bullets))

    errors: list[dict[str, Any]] = []

    def _validate_bullets(arr: list[dict[str, Any]], *, path_prefix: str) -> None:
        for i, item in enumerate(arr):
            cit = item.get("citations")
            if not isinstance(cit, list) or not cit:
                errors.append({"code": "missing_citations", "path": f"{path_prefix}[{i}].citations"})
                continue
            bad_type = [v for v in cit if not isinstance(v, int)]
            if bad_type:
                errors.append({"code": "citations_not_int", "path": f"{path_prefix}[{i}].citations"})
                continue
            out_of_range = sorted({int(v) for v in cit if int(v) not in allowed})
            if out_of_range:
                errors.append(
                    {
                        "code": "citations_not_allowed",
                        "path": f"{path_prefix}[{i}].citations",
                        "not_allowed": out_of_range,
                    }
                )

    # Per spec: for all bullets in requirements/audit evidence, citations must be non-empty
    # and within allowed_idxs (even for UTILSTRÆKKELIG_EVIDENS).
    _validate_bullets(req_bullets, path_prefix="$.system_requirements")
    _validate_bullets(audit_bullets, path_prefix="$.audit_evidence_bullets")

    # Min bullet policies
    if intent == "REQUIREMENTS" and min_req is not None:
        try:
            min_req_i = int(min_req)
        except Exception:  # noqa: BLE001
            min_req_i = 0
        if min_req_i > 0 and req_count < min_req_i:
            errors.append(
                {
                    "code": "min_requirements_bullets_not_met",
                    "have": req_count,
                    "need": min_req_i,
                }
            )

    if include_audit:
        if audit_count < 3:
            errors.append({"code": "min_audit_evidence_bullets_not_met", "have": audit_count, "need": 3})

    ok = len(errors) == 0
    details = {
        "ok": bool(ok),
        "intent_selected": intent,
        "requirements_bullet_count": req_count,
        "audit_evidence_bullet_count": audit_count,
        "policy_min_section3_bullets": (int(min_req) if isinstance(min_req, int) else min_req),
        "policy_include_audit_evidence": bool(include_audit),
        "errors": errors,
    }
    return ok, details


def _fmt_citations(arr: Any) -> str:
    if not isinstance(arr, list):
        return ""
    uniq = sorted({v for v in arr if isinstance(v, int)})
    if not uniq:
        return ""
    return " " + " ".join(f"[{i}]" for i in uniq)


def render_engineering_answer_text(obj: dict[str, Any]) -> str:
    cls = obj.get("classification") or {}
    status = str((cls or {}).get("status") or "").strip()
    text = str((cls or {}).get("text") or "").strip()
    cls_line = f"- {status}: {text}{_fmt_citations((cls or {}).get('citations'))}".rstrip()

    lines: list[str] = []
    lines.append("1. Klassifikation og betingelser")
    lines.append(cls_line)
    lines.append("")

    lines.append("2. Relevante juridiske forpligtelser")
    for item in obj.get("obligations") or []:
        title = str((item or {}).get("title") or "").strip()
        txt = str((item or {}).get("text") or "").strip()
        lines.append(f"- {title}: {txt}{_fmt_citations((item or {}).get('citations'))}".rstrip())
    lines.append("")

    lines.append("3. Konkrete systemkrav")
    req_items = obj.get("system_requirements")
    if not isinstance(req_items, list) or not req_items:
        req_items = obj.get("requirements_bullets") or []

    for item in list(req_items or []):
        if not isinstance(item, dict):
            continue
        if "level" in item:
            level = str((item or {}).get("level") or "").strip()
            txt = str((item or {}).get("text") or "").strip()
            lines.append(f"- {level}: {txt}{_fmt_citations((item or {}).get('citations'))}".rstrip())
        else:
            txt = str((item or {}).get("text") or "").strip()
            lines.append(f"- {txt}{_fmt_citations((item or {}).get('citations'))}".rstrip())

    audit = obj.get("audit_evidence_bullets")
    if isinstance(audit, list) and audit:
        lines.append("")
        lines.append("Minimum ved tilsyn (evidens/artefakter)")
        for item in audit:
            txt = str((item or {}).get("text") or "").strip()
            lines.append(f"- {txt}{_fmt_citations((item or {}).get('citations'))}".rstrip())
    lines.append("")

    lines.append("4. Åbne spørgsmål / risici")
    for item in obj.get("open_questions") or []:
        q = str((item or {}).get("question") or "").strip()
        why = str((item or {}).get("why") or "").strip()
        joiner = " — " if why else ""
        lines.append(f"- {q}{joiner}{why}{_fmt_citations((item or {}).get('citations'))}".rstrip())

    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Repair/Enrich Prompt Builders (consolidated from generation.py)
# ---------------------------------------------------------------------------


def build_repair_prompt(*, base_prompt: str, raw_output: str, why: str) -> str:
    """Build a JSON repair prompt for engineering JSON mode.

    Used when the LLM output fails to parse as valid JSON or has schema errors.

    Args:
        base_prompt: The original prompt sent to the LLM.
        raw_output: The raw LLM output that needs repair.
        why: Reason for the repair (e.g., 'json_parse_fail', 'unknown_fields').

    Returns:
        A prompt asking the LLM to fix its JSON output.
    """
    return (
        f"{base_prompt}\n\n"
        "JSON-REPAIR (STRICT):\n"
        f"ÅRSAG: {why}\n"
        "Du SKAL returnere PRÆCIS én JSON object, der matcher schemaet.\n"
        "Fjern alle ukendte felter. Ret kun format/struktur/typer. Tilføj ingen ekstra tekst.\n\n"
        "DIN TIDLIGERE OUTPUT (kan være invalid JSON):\n"
        "-----BEGIN-----\n"
        f"{raw_output}\n"
        "-----END-----\n"
    )


def build_enrich_prompt(
    *, base_prompt: str, raw_json: str, min_citations: int, allowed_marks: str
) -> str:
    """Build an enrich prompt to add missing citations to engineering JSON.

    Used when the JSON is schema-valid but lacks sufficient citations.

    Args:
        base_prompt: The original prompt sent to the LLM.
        raw_json: The current JSON output (schema-valid).
        min_citations: Minimum required unique citations.
        allowed_marks: Comma-separated string of allowed citation indices.

    Returns:
        A prompt asking the LLM to add citations without changing text content.
    """
    return (
        f"{base_prompt}\n\n"
        "ENRICH (REPAIR-ONLY):\n"
        "Behold teksten uændret; ændr KUN citations-arrays.\n"
        "Tilføj ingen nye bullets og ret ikke tekstindhold.\n"
        f"Sørg for mindst {int(min_citations)} unikke citations-idx (union på tværs af alle citations-arrays).\n"
        f"Du må KUN bruge idx fra KILDER: {allowed_marks}\n"
        "Returnér KUN schema-konform JSON (ingen ekstra felter).\n\n"
        "DIN NUVÆRENDE JSON (schema-konform):\n"
        "-----BEGIN-----\n"
        f"{raw_json}\n"
        "-----END-----\n"
    )


def build_enrich_prompt_policy_completion(
    *,
    base_prompt: str,
    raw_json: str,
    allowed_marks: str,
    missing_requirements_bullets: int,
    missing_audit_bullets: int,
    min_citations: int,
) -> str:
    """Build an enrich prompt for policy completion (adding missing bullets).

    Used when the JSON needs additional bullets to meet policy requirements.

    Args:
        base_prompt: The original prompt sent to the LLM.
        raw_json: The current JSON output (schema-valid).
        allowed_marks: Comma-separated string of allowed citation indices.
        missing_requirements_bullets: Number of missing system_requirements bullets.
        missing_audit_bullets: Number of missing audit_evidence_bullets.
        min_citations: Minimum required unique citations.

    Returns:
        A prompt asking the LLM to add missing bullets while preserving existing content.
    """
    return (
        f"{base_prompt}\n\n"
        "ENRICH (POLICY-COMPLETION, STRICT):\n"
        "Tilføj KUN de manglende bullets og/eller manglende citations-arrays der kræves af policy.\n"
        "Tilføj IKKE nye top-level keys ud over schemaet, og tilføj IKKE ekstra tekst udenfor JSON.\n"
        "DU MÅ IKKE tilføje nye juridiske claims uden citations.\n"
        "Du må IKKE tilføje nye bullets i obligations eller open_questions.\n"
        "Du må KUN øge antallet af bullets i system_requirements og/eller audit_evidence_bullets.\n"
        "Hvis utilstrækkelig evidens til et punkt, brug text=UTILSTRÆKKELIG_EVIDENS og angiv stadig citations for de vurderede kilder.\n"
        f"Manglende requirements bullets: {int(max(0, missing_requirements_bullets))}\n"
        f"Manglende audit evidence bullets: {int(max(0, missing_audit_bullets))}\n"
        f"Sørg for mindst {int(min_citations)} unikke citations-idx (union på tværs af alle citations-arrays).\n"
        f"Du må KUN bruge idx fra KILDER: {allowed_marks}\n"
        "Returnér KUN schema-konform JSON (ingen ekstra felter).\n\n"
        "DIN NUVÆRENDE JSON (schema-konform):\n"
        "-----BEGIN-----\n"
        f"{raw_json}\n"
        "-----END-----\n"
    )


def get_padding_citations(sorted_allowed: list[int]) -> list[int]:
    """Get default padding citations from allowed indices.

    Used as fallback citations when none are specified.

    Args:
        sorted_allowed: Sorted list of allowed citation indices.

    Returns:
        List of up to 2 citation indices for padding.
    """
    if not sorted_allowed:
        return []
    if len(sorted_allowed) >= 2:
        return [int(sorted_allowed[0]), int(sorted_allowed[1])]
    return [int(sorted_allowed[0])]
