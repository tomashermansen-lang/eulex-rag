from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LegalJSONValidationError(Exception):
    """Validation error for LEGAL JSON output.

    Unlike ENGINEERING, LEGAL JSON validation is 'soft' - it doesn't fail-closed
    but allows graceful degradation to prose mode.
    """
    code: str
    message: str
    details: dict[str, Any] | None = None

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.code}: {self.message}"


# ---------------------------------------------------------------------------
# LEGAL JSON Schema
# ---------------------------------------------------------------------------
# Unlike ENGINEERING's strict 4-section schema, LEGAL uses a flexible prose-friendly
# schema with only 'summary' required. This enables machine-testability while
# maintaining the natural prose style expected for legal output.

_TOP_KEYS_REQUIRED = {"summary"}
_TOP_KEYS_OPTIONAL = {"key_points", "legal_basis", "caveats"}


def _require_dict(obj: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise LegalJSONValidationError("schema_fail", f"Expected object at {path}.")
    return obj


def _require_list(obj: Any, *, path: str) -> list[Any]:
    if not isinstance(obj, list):
        raise LegalJSONValidationError("schema_fail", f"Expected array at {path}.")
    return obj


def _require_str(obj: Any, *, path: str) -> str:
    if not isinstance(obj, str) or not obj.strip():
        raise LegalJSONValidationError("schema_fail", f"Expected non-empty string at {path}.")
    return obj


def _require_int(obj: Any, *, path: str) -> int:
    if not isinstance(obj, int):
        raise LegalJSONValidationError("schema_fail", f"Expected integer at {path}.")
    return obj


def _require_no_unknown_fields_required_optional(
    d: dict[str, Any], *, required: set[str], optional: set[str], path: str
) -> None:
    allowed = set(required) | set(optional)
    extra = sorted(set(d.keys()) - allowed)
    missing_required = sorted(set(required) - set(d.keys()))
    if extra or missing_required:
        raise LegalJSONValidationError(
            "unknown_fields",
            f"Invalid keys at {path}.",
            details={"path": path, "extra": extra, "missing": missing_required},
        )


def _validate_citations(value: Any, *, path: str) -> list[int]:
    """Validate citations array. Returns empty list if not present or invalid."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise LegalJSONValidationError("schema_fail", f"Expected array at {path}.")
    out: list[int] = []
    for i, v in enumerate(value):
        idx = _require_int(v, path=f"{path}[{i}]")
        if idx <= 0:
            raise LegalJSONValidationError("schema_fail", f"Citation idx must be > 0 at {path}[{i}].")
        out.append(idx)
    return out


def validate_legal_answer_json(obj: Any) -> dict[str, Any]:
    """Validate LEGAL answer JSON against the schema.

    Schema:
    {
        "summary": string (required) - Main answer prose with inline [n] citations
        "key_points": [               (optional) - Structured key points
            {"point": string, "citations": [int, ...]}
        ]
        "legal_basis": [string, ...]  (optional) - Referenced articles/annexes/recitals
        "caveats": [string, ...]      (optional) - Legal caveats and conditions
    }

    Args:
        obj: The parsed JSON object to validate.

    Returns:
        The validated object (possibly normalized).

    Raises:
        LegalJSONValidationError: If validation fails.
    """
    root = _require_dict(obj, path="$")
    _require_no_unknown_fields_required_optional(
        root,
        required=_TOP_KEYS_REQUIRED,
        optional=_TOP_KEYS_OPTIONAL,
        path="$",
    )

    # Summary is required
    _require_str(root.get("summary"), path="$.summary")

    # key_points is optional but must be valid if present
    if "key_points" in root:
        key_points = _require_list(root.get("key_points"), path="$.key_points")
        for i, item in enumerate(key_points):
            d = _require_dict(item, path=f"$.key_points[{i}]")
            # Allow only 'point' and 'citations' keys
            allowed_keys = {"point", "citations"}
            extra = sorted(set(d.keys()) - allowed_keys)
            if extra:
                raise LegalJSONValidationError(
                    "unknown_fields",
                    f"Invalid keys at $.key_points[{i}].",
                    details={"path": f"$.key_points[{i}]", "extra": extra, "missing": []},
                )
            _require_str(d.get("point"), path=f"$.key_points[{i}].point")
            if "citations" in d:
                _validate_citations(d.get("citations"), path=f"$.key_points[{i}].citations")

    # legal_basis is optional but must be valid if present
    if "legal_basis" in root:
        legal_basis = _require_list(root.get("legal_basis"), path="$.legal_basis")
        for i, item in enumerate(legal_basis):
            _require_str(item, path=f"$.legal_basis[{i}]")

    # caveats is optional but must be valid if present
    if "caveats" in root:
        caveats = _require_list(root.get("caveats"), path="$.caveats")
        for i, item in enumerate(caveats):
            _require_str(item, path=f"$.caveats[{i}]")

    return root


def extract_cited_idxs(obj: dict[str, Any]) -> set[int]:
    """Extract all citation indices from a LEGAL JSON answer.

    Citations can appear in:
    - key_points[].citations arrays
    - Inline [n] references in summary text

    Args:
        obj: Validated LEGAL JSON object.

    Returns:
        Set of unique citation indices.
    """
    import re

    cited: set[int] = set()

    # Extract from key_points citations arrays
    for item in obj.get("key_points") or []:
        cit = (item or {}).get("citations")
        if isinstance(cit, list):
            for v in cit:
                if isinstance(v, int) and v > 0:
                    cited.add(v)

    # Extract inline [n] from summary text
    summary = obj.get("summary") or ""
    if isinstance(summary, str):
        for match in re.finditer(r"\[(\d+)\]", summary):
            try:
                idx = int(match.group(1))
                if idx > 0:
                    cited.add(idx)
            except ValueError:
                pass

    return cited


def render_legal_answer_text(obj: dict[str, Any]) -> str:
    """Render LEGAL JSON to prose text.

    The output format follows the LEGAL prompt structure:
    1. Retsgrundlag (from legal_basis)
    2. Juridisk analyse (from summary)
    3. Konklusion (inline from summary or key_points)

    If the JSON only has summary (minimal schema), just returns the summary.

    Args:
        obj: Validated LEGAL JSON object.

    Returns:
        Rendered text suitable for LEGAL profile output.
    """
    lines: list[str] = []

    summary = str(obj.get("summary") or "").strip()
    legal_basis = obj.get("legal_basis") or []
    key_points = obj.get("key_points") or []
    caveats = obj.get("caveats") or []

    # If we have legal_basis, format as structured output
    if legal_basis:
        lines.append("**Retsgrundlag**")
        for basis in legal_basis:
            if isinstance(basis, str) and basis.strip():
                lines.append(f"- {basis.strip()}")
        lines.append("")

    # Main analysis/summary
    if legal_basis or key_points or caveats:
        lines.append("**Juridisk analyse**")
    lines.append(summary)

    # Key points if present
    if key_points:
        lines.append("")
        lines.append("**Hovedpunkter**")
        for item in key_points:
            if not isinstance(item, dict):
                continue
            point = str((item or {}).get("point") or "").strip()
            cit = (item or {}).get("citations") or []
            cit_str = ""
            if isinstance(cit, list) and cit:
                cit_str = " " + " ".join(f"[{i}]" for i in sorted(set(cit)) if isinstance(i, int))
            if point:
                lines.append(f"- {point}{cit_str}".rstrip())

    # Caveats if present
    if caveats:
        lines.append("")
        lines.append("**Forbehold**")
        for caveat in caveats:
            if isinstance(caveat, str) and caveat.strip():
                lines.append(f"- {caveat.strip()}")

    return "\n".join(lines).strip() + "\n"


def validate_legal_answer_json_soft(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Attempt soft JSON validation for LEGAL output.

    This function tries to parse and validate JSON from raw LLM output.
    Unlike ENGINEERING's fail-closed approach, this:
    - Returns (validated_obj, None) on success
    - Returns (None, error_reason) on failure - caller can fall back to prose

    Args:
        raw_text: Raw LLM output that may or may not be JSON.

    Returns:
        Tuple of (validated_object, error_reason).
        If successful, error_reason is None.
        If failed, validated_object is None and error_reason explains why.
    """
    import json

    text = (raw_text or "").strip()

    # Try to extract JSON from possible markdown code blocks
    if text.startswith("```"):
        # Remove markdown code block
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try to parse
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"json_parse_fail: {e}"

    # Try to validate
    try:
        validated = validate_legal_answer_json(parsed)
        return validated, None
    except LegalJSONValidationError as e:
        return None, f"{e.code}: {e.message}"
