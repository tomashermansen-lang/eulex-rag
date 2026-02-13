"""Cross-law evaluation suite manager.

This module manages cross-law eval suites, providing CRUD operations,
YAML import/export, and validation for cross-law test cases.

Single Responsibility: Suite persistence and validation.
Does NOT execute evaluations - that's eval_core.py's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
import yaml


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------


class SuiteValidationError(Exception):
    """Raised when suite validation fails."""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossLawGoldenCase:
    """A single cross-law evaluation test case.

    Immutable to prevent accidental mutation during eval runs.
    """

    id: str
    prompt: str
    corpus_scope: str  # "single" | "explicit" | "all"
    target_corpora: tuple[str, ...]
    synthesis_mode: str  # "aggregation" | "comparison" | "unified" | "routing" | "discovery"
    expected_anchors: tuple[str, ...]
    expected_corpora: tuple[str, ...]
    min_corpora_cited: int | None
    profile: str  # "LEGAL" | "ENGINEERING"
    disabled: bool
    origin: str  # "manual" | "auto-generated"
    # Extended fields for full eval support
    test_types: tuple[str, ...] = ()
    expected_behavior: str = "answer"  # "answer" | "abstain"
    must_include_any_of: tuple[str, ...] = ()
    must_include_any_of_2: tuple[str, ...] = ()
    must_include_all_of: tuple[str, ...] = ()
    must_not_include_any_of: tuple[str, ...] = ()
    contract_check: bool = False
    min_citations: int | None = None
    max_citations: int | None = None
    notes: str = ""
    # Quality evaluation fields (backward-compatible defaults)
    difficulty: str | None = None           # "easy" | "medium" | "hard" | None
    retrieval_confirmed: bool | None = None  # True/False/None (inverted generation)


@dataclass
class CrossLawEvalSuite:
    """A collection of cross-law evaluation test cases.

    Not frozen to allow timestamp updates on modification.
    """

    id: str
    name: str
    description: str
    target_corpora: tuple[str, ...]
    default_synthesis_mode: str
    cases: tuple[CrossLawGoldenCase, ...]
    created_at: str | None = field(default=None)
    modified_at: str | None = field(default=None)


# ---------------------------------------------------------------------------
# Suite Manager
# ---------------------------------------------------------------------------


class CrossLawSuiteManager:
    """Manager for cross-law eval suite CRUD and YAML operations.

    Responsibilities:
    - List/get/create/update/delete suites
    - YAML import/export
    - Suite validation (corpus IDs, duplicates, comparison rules)
    - Case-level CRUD operations
    """

    SUITE_PREFIX = "cross_law_"
    SUITE_EXTENSION = ".yaml"

    def __init__(
        self,
        evals_dir: Path,
        valid_corpus_ids: set[str],
    ):
        """Initialize the suite manager.

        Args:
            evals_dir: Directory for storing suite YAML files
            valid_corpus_ids: Set of valid corpus IDs for validation
        """
        self.evals_dir = evals_dir
        self.valid_corpus_ids = valid_corpus_ids
        self.evals_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Suite CRUD
    # -----------------------------------------------------------------------

    def list_suites(self) -> list[CrossLawEvalSuite]:
        """List all cross-law eval suites.

        Returns:
            List of all suites in the evals directory
        """
        suites = []
        pattern = f"{self.SUITE_PREFIX}*{self.SUITE_EXTENSION}"
        for path in self.evals_dir.glob(pattern):
            try:
                suite = self._load_suite_from_file(path)
                suites.append(suite)
            except Exception:
                # Skip malformed files
                continue
        return suites

    def get_suite(self, suite_id: str) -> CrossLawEvalSuite | None:
        """Get a specific suite by ID.

        Args:
            suite_id: The suite ID

        Returns:
            The suite, or None if not found
        """
        path = self._suite_path(suite_id)
        if not path.exists():
            return None
        return self._load_suite_from_file(path)

    def create_suite(self, suite: CrossLawEvalSuite) -> None:
        """Create a new suite (validate and save to YAML).

        Args:
            suite: The suite to create

        Raises:
            SuiteValidationError: If validation fails
        """
        self._validate_suite(suite)

        # Add timestamps
        now = datetime.now(timezone.utc).isoformat()
        suite_with_timestamps = CrossLawEvalSuite(
            id=suite.id,
            name=suite.name,
            description=suite.description,
            target_corpora=suite.target_corpora,
            default_synthesis_mode=suite.default_synthesis_mode,
            cases=suite.cases,
            created_at=now,
            modified_at=now,
        )

        self._save_suite_to_file(suite_with_timestamps)

    def update_suite(self, suite: CrossLawEvalSuite) -> None:
        """Update an existing suite.

        Args:
            suite: The suite with updated data

        Raises:
            FileNotFoundError: If suite doesn't exist
            SuiteValidationError: If validation fails
        """
        path = self._suite_path(suite.id)
        if not path.exists():
            raise FileNotFoundError(f"Suite not found: {suite.id}")

        # Preserve created_at, update modified_at
        existing = self._load_suite_from_file(path)
        self._validate_suite(suite)

        now = datetime.now(timezone.utc).isoformat()
        suite_with_timestamps = CrossLawEvalSuite(
            id=suite.id,
            name=suite.name,
            description=suite.description,
            target_corpora=suite.target_corpora,
            default_synthesis_mode=suite.default_synthesis_mode,
            cases=suite.cases,
            created_at=existing.created_at,
            modified_at=now,
        )

        self._save_suite_to_file(suite_with_timestamps)

    def delete_suite(self, suite_id: str) -> None:
        """Delete a suite.

        Args:
            suite_id: The suite ID to delete

        Raises:
            FileNotFoundError: If suite doesn't exist
        """
        path = self._suite_path(suite_id)
        if not path.exists():
            raise FileNotFoundError(f"Suite not found: {suite_id}")
        path.unlink()

    # -----------------------------------------------------------------------
    # Case Operations
    # -----------------------------------------------------------------------

    def add_case(self, suite_id: str, case: CrossLawGoldenCase) -> None:
        """Add a case to an existing suite.

        Args:
            suite_id: The suite ID
            case: The case to add

        Raises:
            FileNotFoundError: If suite doesn't exist
            SuiteValidationError: If case validation fails
        """
        suite = self.get_suite(suite_id)
        if suite is None:
            raise FileNotFoundError(f"Suite not found: {suite_id}")

        new_cases = suite.cases + (case,)
        updated = CrossLawEvalSuite(
            id=suite.id,
            name=suite.name,
            description=suite.description,
            target_corpora=suite.target_corpora,
            default_synthesis_mode=suite.default_synthesis_mode,
            cases=new_cases,
            created_at=suite.created_at,
            modified_at=suite.modified_at,
        )
        self.update_suite(updated)

    def update_case(self, suite_id: str, case: CrossLawGoldenCase) -> None:
        """Update a case in an existing suite.

        Args:
            suite_id: The suite ID
            case: The updated case (matched by ID)

        Raises:
            FileNotFoundError: If suite or case doesn't exist
        """
        suite = self.get_suite(suite_id)
        if suite is None:
            raise FileNotFoundError(f"Suite not found: {suite_id}")

        new_cases = tuple(
            case if c.id == case.id else c
            for c in suite.cases
        )

        updated = CrossLawEvalSuite(
            id=suite.id,
            name=suite.name,
            description=suite.description,
            target_corpora=suite.target_corpora,
            default_synthesis_mode=suite.default_synthesis_mode,
            cases=new_cases,
            created_at=suite.created_at,
            modified_at=suite.modified_at,
        )
        self.update_suite(updated)

    def delete_case(self, suite_id: str, case_id: str) -> None:
        """Delete a case from a suite.

        Args:
            suite_id: The suite ID
            case_id: The case ID to delete

        Raises:
            FileNotFoundError: If suite doesn't exist
        """
        suite = self.get_suite(suite_id)
        if suite is None:
            raise FileNotFoundError(f"Suite not found: {suite_id}")

        new_cases = tuple(c for c in suite.cases if c.id != case_id)

        updated = CrossLawEvalSuite(
            id=suite.id,
            name=suite.name,
            description=suite.description,
            target_corpora=suite.target_corpora,
            default_synthesis_mode=suite.default_synthesis_mode,
            cases=new_cases,
            created_at=suite.created_at,
            modified_at=suite.modified_at,
        )
        self.update_suite(updated)

    def duplicate_case(self, suite_id: str, case_id: str) -> str:
        """Duplicate a case in a suite.

        Args:
            suite_id: The suite ID
            case_id: The case ID to duplicate

        Returns:
            The new case ID

        Raises:
            FileNotFoundError: If suite or case doesn't exist
        """
        suite = self.get_suite(suite_id)
        if suite is None:
            raise FileNotFoundError(f"Suite not found: {suite_id}")

        original = next((c for c in suite.cases if c.id == case_id), None)
        if original is None:
            raise FileNotFoundError(f"Case not found: {case_id}")

        # Generate unique ID
        existing_ids = {c.id for c in suite.cases}
        new_id = f"{case_id}_copy"
        counter = 1
        while new_id in existing_ids:
            new_id = f"{case_id}_copy_{counter}"
            counter += 1

        new_case = CrossLawGoldenCase(
            id=new_id,
            prompt=original.prompt,
            corpus_scope=original.corpus_scope,
            target_corpora=original.target_corpora,
            synthesis_mode=original.synthesis_mode,
            expected_anchors=original.expected_anchors,
            expected_corpora=original.expected_corpora,
            min_corpora_cited=original.min_corpora_cited,
            profile=original.profile,
            disabled=original.disabled,
            origin=original.origin,
            test_types=original.test_types,
            expected_behavior=original.expected_behavior,
            must_include_any_of=original.must_include_any_of,
            must_include_any_of_2=original.must_include_any_of_2,
            must_include_all_of=original.must_include_all_of,
            must_not_include_any_of=original.must_not_include_any_of,
            contract_check=original.contract_check,
            min_citations=original.min_citations,
            max_citations=original.max_citations,
            notes=original.notes,
            difficulty=original.difficulty,
            retrieval_confirmed=original.retrieval_confirmed,
        )

        self.add_case(suite_id, new_case)
        return new_id

    # -----------------------------------------------------------------------
    # YAML Import/Export
    # -----------------------------------------------------------------------

    def import_yaml(self, yaml_content: str) -> CrossLawEvalSuite:
        """Import a suite from YAML string.

        Args:
            yaml_content: YAML content to parse

        Returns:
            The parsed suite

        Raises:
            SuiteValidationError: If YAML is malformed or validation fails
        """
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise SuiteValidationError(f"YAML parse error: {e}")

        if not isinstance(data, dict):
            raise SuiteValidationError("YAML must be a dictionary")

        suite = self._dict_to_suite(data)
        self._validate_suite(suite)
        return suite

    def export_yaml(self, suite_id: str) -> str:
        """Export a suite to YAML string.

        Args:
            suite_id: The suite ID to export

        Returns:
            YAML string representation

        Raises:
            FileNotFoundError: If suite doesn't exist
        """
        suite = self.get_suite(suite_id)
        if suite is None:
            raise FileNotFoundError(f"Suite not found: {suite_id}")

        data = self._suite_to_dict(suite)
        return yaml.dump(data, default_flow_style=False, allow_unicode=True)

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_suite(self, suite: CrossLawEvalSuite) -> None:
        """Validate a suite before saving.

        Args:
            suite: The suite to validate

        Raises:
            SuiteValidationError: If validation fails
        """
        # Check for duplicate case IDs
        case_ids = [c.id for c in suite.cases]
        if len(case_ids) != len(set(case_ids)):
            duplicates = [cid for cid in case_ids if case_ids.count(cid) > 1]
            raise SuiteValidationError(
                f"Duplicate case IDs: {', '.join(set(duplicates))}"
            )

        # Validate each case
        for case in suite.cases:
            self._validate_case(case)

    def _validate_case(self, case: CrossLawGoldenCase) -> None:
        """Validate a single case.

        Args:
            case: The case to validate

        Raises:
            SuiteValidationError: If validation fails
        """
        # Empty prompt check
        if not case.prompt or not case.prompt.strip():
            raise SuiteValidationError(f"Case '{case.id}' has empty prompt")

        # Validate corpus IDs in target_corpora
        for corpus_id in case.target_corpora:
            if corpus_id not in self.valid_corpus_ids:
                raise SuiteValidationError(
                    f"Unknown corpus ID: {corpus_id}. "
                    f"Valid: {', '.join(sorted(self.valid_corpus_ids))}"
                )

        # Validate corpus IDs in expected_corpora
        for corpus_id in case.expected_corpora:
            if corpus_id not in self.valid_corpus_ids:
                raise SuiteValidationError(
                    f"Unknown corpus ID: {corpus_id}. "
                    f"Valid: {', '.join(sorted(self.valid_corpus_ids))}"
                )

        # expected_corpora must be subset of target_corpora (when both non-empty)
        if case.expected_corpora and case.target_corpora:
            expected_set = set(case.expected_corpora)
            target_set = set(case.target_corpora)
            extra = expected_set - target_set
            if extra:
                raise SuiteValidationError(
                    f"Case '{case.id}': expected_corpora {sorted(extra)} "
                    f"not in target_corpora {sorted(target_set)}"
                )

        # Comparison mode requires 2+ corpora
        if case.synthesis_mode == "comparison":
            if len(case.target_corpora) < 2:
                raise SuiteValidationError(
                    f"Case '{case.id}': comparison mode requires at least 2 target corpora"
                )

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _suite_path(self, suite_id: str) -> Path:
        """Get file path for a suite."""
        return self.evals_dir / f"{self.SUITE_PREFIX}{suite_id}{self.SUITE_EXTENSION}"

    def _load_suite_from_file(self, path: Path) -> CrossLawEvalSuite:
        """Load a suite from YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return self._dict_to_suite(data)

    def _save_suite_to_file(self, suite: CrossLawEvalSuite) -> None:
        """Save a suite to YAML file."""
        path = self._suite_path(suite.id)
        data = self._suite_to_dict(suite)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def _dict_to_suite(self, data: dict) -> CrossLawEvalSuite:
        """Convert dict to CrossLawEvalSuite."""
        cases = []
        for case_data in data.get("cases", []):
            case = CrossLawGoldenCase(
                id=case_data["id"],
                prompt=case_data["prompt"],
                corpus_scope=case_data.get("corpus_scope", "all"),
                target_corpora=tuple(case_data.get("target_corpora", [])),
                synthesis_mode=case_data.get("synthesis_mode", "unified"),
                expected_anchors=tuple(case_data.get("expected_anchors", [])),
                expected_corpora=tuple(case_data.get("expected_corpora", [])),
                min_corpora_cited=case_data.get("min_corpora_cited", 1),
                profile=case_data.get("profile", "LEGAL"),
                disabled=case_data.get("disabled", False),
                origin=case_data.get("origin", "manual"),
                test_types=tuple(case_data.get("test_types", [])),
                expected_behavior=case_data.get("expected_behavior", "answer"),
                must_include_any_of=tuple(case_data.get("must_include_any_of", [])),
                must_include_any_of_2=tuple(case_data.get("must_include_any_of_2", [])),
                must_include_all_of=tuple(case_data.get("must_include_all_of", [])),
                must_not_include_any_of=tuple(case_data.get("must_not_include_any_of", [])),
                contract_check=case_data.get("contract_check", False),
                min_citations=case_data.get("min_citations"),
                max_citations=case_data.get("max_citations"),
                notes=case_data.get("notes", ""),
                difficulty=case_data.get("difficulty"),
                retrieval_confirmed=case_data.get("retrieval_confirmed"),
            )
            cases.append(case)

        # Generate ID from name if not provided
        suite_id = data.get("id")
        if not suite_id:
            name = data.get("name", "imported")
            base = name.lower().replace(" ", "_").replace("-", "_")
            base = "".join(c for c in base if c.isalnum() or c == "_")
            import uuid
            suite_id = f"{base}_{uuid.uuid4().hex[:8]}"

        return CrossLawEvalSuite(
            id=suite_id,
            name=data["name"],
            description=data.get("description", ""),
            target_corpora=tuple(data.get("target_corpora", [])),
            default_synthesis_mode=data.get("default_synthesis_mode", "unified"),
            cases=tuple(cases),
            created_at=data.get("created_at"),
            modified_at=data.get("modified_at"),
        )

    def _suite_to_dict(self, suite: CrossLawEvalSuite) -> dict:
        """Convert CrossLawEvalSuite to dict for YAML serialization."""
        return {
            "id": suite.id,
            "name": suite.name,
            "description": suite.description,
            "target_corpora": list(suite.target_corpora),
            "default_synthesis_mode": suite.default_synthesis_mode,
            "created_at": suite.created_at,
            "modified_at": suite.modified_at,
            "cases": [
                {
                    "id": c.id,
                    "prompt": c.prompt,
                    "corpus_scope": c.corpus_scope,
                    "target_corpora": list(c.target_corpora),
                    "synthesis_mode": c.synthesis_mode,
                    "expected_anchors": list(c.expected_anchors),
                    "expected_corpora": list(c.expected_corpora),
                    "min_corpora_cited": c.min_corpora_cited,
                    "profile": c.profile,
                    "disabled": c.disabled,
                    "origin": c.origin,
                    "test_types": list(c.test_types),
                    "expected_behavior": c.expected_behavior,
                    "must_include_any_of": list(c.must_include_any_of),
                    "must_include_any_of_2": list(c.must_include_any_of_2),
                    "must_include_all_of": list(c.must_include_all_of),
                    "must_not_include_any_of": list(c.must_not_include_any_of),
                    "contract_check": c.contract_check,
                    "min_citations": c.min_citations,
                    "max_citations": c.max_citations,
                    "notes": c.notes,
                    "difficulty": c.difficulty,
                    "retrieval_confirmed": c.retrieval_confirmed,
                }
                for c in suite.cases
            ],
        }
