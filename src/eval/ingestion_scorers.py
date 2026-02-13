"""Scorers for ingestion quality evaluation.

This module provides scorers for validating:
- Role classification accuracy (keyword vs LLM)
- Enrichment term quality
- Contextual description coverage

Used by ingestion_eval_runner.py to compare different role detection methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScorerResult:
    """Result from a scorer evaluation."""

    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoleClassificationScorer:
    """Validates role assignments against golden cases.

    Measures:
    - Precision: What fraction of assigned roles are correct?
    - Recall: What fraction of expected roles were found?
    - Pass criteria: All expected roles must be present (recall = 1.0)
    """

    name: str = "role_classification"

    def score(
        self,
        expected_roles: list[str],
        actual_roles: list[str],
        *,
        must_not_have_roles: list[str] | None = None,
    ) -> ScorerResult:
        """Score role assignment against expected roles.

        Args:
            expected_roles: Roles that MUST be present
            actual_roles: Roles that were assigned
            must_not_have_roles: Roles that MUST NOT be present

        Returns:
            ScorerResult with precision, recall, and pass/fail
        """
        expected_set = set(expected_roles)
        actual_set = set(actual_roles)
        must_not_set = set(must_not_have_roles or [])

        # Check for forbidden roles
        forbidden_found = actual_set & must_not_set
        if forbidden_found:
            return ScorerResult(
                name=self.name,
                passed=False,
                score=0.0,
                message=f"Found forbidden roles: {sorted(forbidden_found)}",
                details={
                    "expected": sorted(expected_set),
                    "actual": sorted(actual_set),
                    "forbidden_found": sorted(forbidden_found),
                    "precision": 0.0,
                    "recall": 0.0,
                },
            )

        # Calculate precision and recall
        if not actual_set:
            # No roles assigned
            precision = 1.0 if not expected_set else 0.0
            recall = 1.0 if not expected_set else 0.0
        else:
            true_positives = len(expected_set & actual_set)
            precision = true_positives / len(actual_set)
            recall = true_positives / len(expected_set) if expected_set else 1.0

        # Pass if all expected roles are found (recall = 1.0)
        passed = expected_set <= actual_set

        # Missing roles for error message
        missing = expected_set - actual_set
        extra = actual_set - expected_set

        message = ""
        if missing:
            message = f"Missing roles: {sorted(missing)}"
        if extra:
            extra_msg = f"Extra roles: {sorted(extra)}"
            message = f"{message}; {extra_msg}" if message else extra_msg

        return ScorerResult(
            name=self.name,
            passed=passed,
            score=recall,
            message=message,
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "expected": sorted(expected_set),
                "actual": sorted(actual_set),
                "missing": sorted(missing),
                "extra": sorted(extra),
                "true_positives": len(expected_set & actual_set),
            },
        )


@dataclass
class EnrichmentCoverageScorer:
    """Validates that chunks have enrichment terms and descriptions.

    Used to ensure LLM enrichment is working during ingestion.
    """

    name: str = "enrichment_coverage"

    def score(
        self,
        has_contextual_description: bool,
        has_search_terms: bool,
        search_terms_count: int = 0,
        min_terms: int = 2,
    ) -> ScorerResult:
        """Score enrichment coverage.

        Args:
            has_contextual_description: Whether chunk has contextual_description
            has_search_terms: Whether chunk has enrichment_terms
            search_terms_count: Number of search terms
            min_terms: Minimum required terms

        Returns:
            ScorerResult with coverage metrics
        """
        passed = has_contextual_description and has_search_terms and search_terms_count >= min_terms

        score = 0.0
        if has_contextual_description:
            score += 0.5
        if has_search_terms and search_terms_count >= min_terms:
            score += 0.5

        message = ""
        if not has_contextual_description:
            message = "Missing contextual_description"
        if not has_search_terms:
            message = f"{message}; Missing enrichment_terms" if message else "Missing enrichment_terms"
        elif search_terms_count < min_terms:
            message = f"{message}; Too few terms ({search_terms_count} < {min_terms})" if message else f"Too few terms ({search_terms_count} < {min_terms})"

        return ScorerResult(
            name=self.name,
            passed=passed,
            score=score,
            message=message,
            details={
                "has_contextual_description": has_contextual_description,
                "has_search_terms": has_search_terms,
                "search_terms_count": search_terms_count,
                "min_terms_required": min_terms,
            },
        )


@dataclass
class RoleConsistencyScorer:
    """Validates that LLM role assignment is consistent across runs.

    Runs the same chunk through role classification multiple times
    and measures how consistent the results are.
    """

    name: str = "role_consistency"

    def score(
        self,
        role_assignments: list[list[str]],
    ) -> ScorerResult:
        """Score consistency across multiple role assignments.

        Args:
            role_assignments: List of role lists from multiple runs

        Returns:
            ScorerResult with consistency score
        """
        if not role_assignments:
            return ScorerResult(
                name=self.name,
                passed=False,
                score=0.0,
                message="No role assignments provided",
                details={},
            )

        if len(role_assignments) == 1:
            return ScorerResult(
                name=self.name,
                passed=True,
                score=1.0,
                message="Only one assignment, cannot measure consistency",
                details={"runs": 1},
            )

        # Convert to sets for comparison
        role_sets = [frozenset(roles) for roles in role_assignments]

        # Count unique assignments
        unique_assignments = len(set(role_sets))
        total_runs = len(role_sets)

        # Consistency = 1 - (unique - 1) / (total - 1)
        # Perfect consistency: unique = 1, score = 1.0
        # All different: unique = total, score = 0.0
        if total_runs > 1:
            consistency = 1.0 - (unique_assignments - 1) / (total_runs - 1)
        else:
            consistency = 1.0

        passed = consistency >= 0.8  # 80% consistency threshold

        return ScorerResult(
            name=self.name,
            passed=passed,
            score=round(consistency, 4),
            message=f"{unique_assignments} unique assignments across {total_runs} runs",
            details={
                "total_runs": total_runs,
                "unique_assignments": unique_assignments,
                "consistency": round(consistency, 4),
                "assignments": [sorted(list(s)) for s in role_sets],
            },
        )
