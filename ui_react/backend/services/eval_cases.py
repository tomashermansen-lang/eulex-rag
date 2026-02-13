"""Eval case CRUD service.

Single Responsibility: Handle CRUD operations for eval cases stored in YAML files.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml


# Custom exceptions
class ValidationError(Exception):
    """Raised when case data fails validation."""
    pass


class NotFoundError(Exception):
    """Raised when a case is not found."""
    pass


# Valid test types
VALID_TEST_TYPES = frozenset([
    "retrieval",
    "faithfulness",
    "relevancy",
    "abstention",
    "robustness",
    "multi_hop",
])

# Valid anchor prefixes
VALID_ANCHOR_PREFIXES = frozenset([
    "article",
    "section",
    "annex",
    "recital",
    "paragraph",
    "chapter",
])


def _get_evals_dir() -> Path:
    """Get the evals directory path."""
    # Navigate from services/ to project root
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data" / "evals"


def _get_golden_cases_path(law: str) -> Path:
    """Get path to golden cases file for a law.

    Handles both underscore and hyphen naming conventions:
    - ai-act -> tries golden_cases_ai_act.yaml first, then golden_cases_ai-act.yaml
    """
    evals_dir = _get_evals_dir()

    # Try underscore version first (ai-act -> ai_act)
    law_underscore = law.replace("-", "_")
    candidates = [
        evals_dir / f"golden_cases_{law_underscore}.yaml",
        evals_dir / f"golden_cases_{law}.yaml",
    ]

    for path in candidates:
        if path.exists():
            return path

    # Return first candidate as default (for creating new files)
    return candidates[0]


def _load_yaml(path: Path) -> list[dict[str, Any]]:
    """Load YAML file, returning empty list if not found."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if data else []


def _save_yaml(path: Path, data: list[dict[str, Any]]) -> None:
    """Save data to YAML file with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )


def slugify(text: str, max_length: int = 30) -> str:
    """Convert text to URL-safe slug.

    Args:
        text: Input text to slugify
        max_length: Maximum length of the slug

    Returns:
        Lowercase, hyphenated slug
    """
    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Convert to lowercase
    text = text.lower()

    # Replace non-alphanumeric with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text)

    # Remove leading/trailing hyphens
    text = text.strip("-")

    # Truncate to max length (don't cut mid-word if possible)
    if len(text) > max_length:
        text = text[:max_length]
        # Try to cut at last hyphen
        last_hyphen = text.rfind("-")
        if last_hyphen > max_length // 2:
            text = text[:last_hyphen]

    return text


def generate_case_id(law: str, prompt: str, existing_ids: set[str]) -> str:
    """Generate unique case ID from law and prompt.

    Format: {law}-{nn}-{slug}
    Example: ai-act-07-risk-classification-of-ai

    Args:
        law: Law/corpus identifier
        prompt: The test prompt text
        existing_ids: Set of existing case IDs to avoid collisions

    Returns:
        Unique case ID string
    """
    slug = slugify(prompt)

    for n in range(1, 1000):
        candidate = f"{law}-{n:02d}-{slug}"
        if candidate not in existing_ids:
            return candidate

    raise ValueError(f"Could not generate unique ID for law '{law}'")


def validate_anchor(anchor: str) -> bool:
    """Validate anchor format.

    Valid formats:
        - article:N (e.g., article:5, article:14)
        - section:N
        - annex:N or annex:iii (roman numerals)
        - recital:N
        - paragraph:N
        - chapter:N

    Args:
        anchor: Anchor string to validate

    Returns:
        True if valid, False otherwise
    """
    if not anchor or ":" not in anchor:
        return False

    parts = anchor.split(":", 1)
    if len(parts) != 2:
        return False

    prefix, value = parts
    prefix = prefix.lower()

    if prefix not in VALID_ANCHOR_PREFIXES:
        return False

    if not value:
        return False

    # Value can be numeric or roman numeral
    return True


def _validate_case_data(data: dict[str, Any], is_create: bool = True) -> None:
    """Validate case data.

    Args:
        data: Case data dictionary
        is_create: Whether this is a create operation (stricter validation)

    Raises:
        ValidationError: If validation fails
    """
    if is_create:
        # Required fields for create
        if "prompt" not in data:
            raise ValidationError("Missing required field: prompt")

        if "profile" not in data:
            raise ValidationError("Missing required field: profile")

        if "test_types" not in data:
            raise ValidationError("Missing required field: test_types")

    # Validate prompt length
    if "prompt" in data:
        prompt = data["prompt"]
        if not isinstance(prompt, str) or len(prompt) < 10:
            raise ValidationError("Prompt must be at least 10 characters long")

    # Validate profile
    if "profile" in data:
        profile = data["profile"]
        if profile not in ("LEGAL", "ENGINEERING"):
            raise ValidationError("Profile must be 'LEGAL' or 'ENGINEERING'")

    # Validate test_types
    if "test_types" in data:
        test_types = data["test_types"]
        if not isinstance(test_types, list) or len(test_types) == 0:
            raise ValidationError("At least one test_type is required")

        for tt in test_types:
            if tt not in VALID_TEST_TYPES:
                raise ValidationError(f"Invalid test_type: {tt}. Must be one of: {', '.join(sorted(VALID_TEST_TYPES))}")

    # Validate anchors in expected
    if "expected" in data and isinstance(data["expected"], dict):
        expected = data["expected"]
        anchor_fields = [
            "must_include_any_of",
            "must_include_any_of_2",
            "must_include_all_of",
            "must_not_include_any_of",
        ]
        for field in anchor_fields:
            if field in expected:
                anchors = expected[field]
                if isinstance(anchors, list):
                    for anchor in anchors:
                        if anchor and not validate_anchor(anchor):
                            raise ValidationError(f"Invalid anchor format: {anchor}")


def load_cases_for_law(law: str) -> list[dict[str, Any]]:
    """Load all eval cases for a law.

    Args:
        law: Law/corpus identifier

    Returns:
        List of case dictionaries
    """
    path = _get_golden_cases_path(law)
    return _load_yaml(path)


def get_case_by_id(law: str, case_id: str) -> dict[str, Any] | None:
    """Get a single case by ID.

    Args:
        law: Law/corpus identifier
        case_id: Case identifier

    Returns:
        Case dictionary or None if not found
    """
    cases = load_cases_for_law(law)
    for case in cases:
        if case.get("id") == case_id:
            return case
    return None


def create_case(law: str, data: dict[str, Any]) -> dict[str, Any]:
    """Create a new eval case.

    Args:
        law: Law/corpus identifier
        data: Case data dictionary

    Returns:
        Created case dictionary

    Raises:
        ValidationError: If validation fails
    """
    _validate_case_data(data, is_create=True)

    cases = load_cases_for_law(law)
    existing_ids = {c.get("id") for c in cases if c.get("id")}

    # Generate or validate ID
    if "id" in data and data["id"]:
        case_id = data["id"]
        if case_id in existing_ids:
            raise ValidationError(f"Case ID already exists: {case_id}")
    else:
        case_id = generate_case_id(law, data["prompt"], existing_ids)

    # Build case with defaults
    new_case = {
        "id": case_id,
        "profile": data["profile"],
        "prompt": data["prompt"],
        "test_types": data["test_types"],
        "origin": "manual",  # Always manual for created cases
        "expected": _normalize_expected(data.get("expected", {})),
    }

    # Append and save
    cases.append(new_case)
    _save_yaml(_get_golden_cases_path(law), cases)

    return new_case


def _normalize_expected(expected: dict[str, Any]) -> dict[str, Any]:
    """Normalize expected behavior data with defaults."""
    return {
        "must_include_any_of": expected.get("must_include_any_of", []),
        "must_include_any_of_2": expected.get("must_include_any_of_2", []),
        "must_include_all_of": expected.get("must_include_all_of", []),
        "must_not_include_any_of": expected.get("must_not_include_any_of", []),
        "contract_check": expected.get("contract_check", False),
        "min_citations": expected.get("min_citations"),
        "max_citations": expected.get("max_citations"),
        "behavior": expected.get("behavior", "answer"),
        "allow_empty_references": expected.get("allow_empty_references", False),
        "must_have_article_support_for_normative": expected.get("must_have_article_support_for_normative", True),
        "notes": expected.get("notes", ""),
    }


def update_case(law: str, case_id: str, data: dict[str, Any]) -> dict[str, Any]:
    """Update an existing eval case.

    Args:
        law: Law/corpus identifier
        case_id: Case identifier
        data: Fields to update

    Returns:
        Updated case dictionary

    Raises:
        NotFoundError: If case not found
        ValidationError: If validation fails
    """
    _validate_case_data(data, is_create=False)

    cases = load_cases_for_law(law)
    case_index = None

    for i, case in enumerate(cases):
        if case.get("id") == case_id:
            case_index = i
            break

    if case_index is None:
        raise NotFoundError(f"Case not found: {case_id}")

    # Update fields (except ID which is immutable)
    existing = cases[case_index]

    if "profile" in data:
        existing["profile"] = data["profile"]

    if "prompt" in data:
        existing["prompt"] = data["prompt"]

    if "test_types" in data:
        existing["test_types"] = data["test_types"]

    if "expected" in data:
        existing["expected"] = _normalize_expected(data["expected"])

    # Always set origin to manual on update
    existing["origin"] = "manual"

    # Save
    _save_yaml(_get_golden_cases_path(law), cases)

    return existing


def delete_case(law: str, case_id: str) -> None:
    """Delete an eval case.

    Args:
        law: Law/corpus identifier
        case_id: Case identifier

    Raises:
        NotFoundError: If case not found
    """
    cases = load_cases_for_law(law)
    original_length = len(cases)

    cases = [c for c in cases if c.get("id") != case_id]

    if len(cases) == original_length:
        raise NotFoundError(f"Case not found: {case_id}")

    _save_yaml(_get_golden_cases_path(law), cases)


def duplicate_case(law: str, case_id: str) -> dict[str, Any]:
    """Duplicate an existing eval case with a new ID.

    Args:
        law: Law/corpus identifier
        case_id: Source case identifier

    Returns:
        New duplicated case dictionary

    Raises:
        NotFoundError: If source case not found
    """
    source = get_case_by_id(law, case_id)
    if source is None:
        raise NotFoundError(f"Case not found: {case_id}")

    cases = load_cases_for_law(law)
    existing_ids = {c.get("id") for c in cases if c.get("id")}

    # Generate new ID
    new_id = generate_case_id(law, source["prompt"], existing_ids)

    # Create duplicate
    new_case = {
        "id": new_id,
        "profile": source["profile"],
        "prompt": source["prompt"],
        "test_types": list(source.get("test_types", ["retrieval"])),
        "origin": "manual",  # Duplicates are always manual
        "expected": dict(source.get("expected", {})),
    }

    # Append and save
    cases.append(new_case)
    _save_yaml(_get_golden_cases_path(law), cases)

    return new_case
