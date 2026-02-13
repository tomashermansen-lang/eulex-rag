"""Cross-law evaluation scorers.

This module provides specialized scorers for evaluating cross-law synthesis
quality. These scorers assess how well the RAG system handles multi-corpus
queries involving multiple EU regulations.

Scorers:
    CorpusCoverageScorer: Checks that all expected corpora are cited
    SynthesisBalanceScorer: Checks citations are balanced across corpora
    CrossReferenceAccuracyScorer: Validates inter-law reference claims
    RoutingPrecisionScorer: Checks correct laws are identified
    ComparisonCompletenessScorer: Checks comparison targets are covered
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Result Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScorerResult:
    """Result from a cross-law scorer."""

    passed: bool
    score: float  # 0.0 to 1.0
    message: str


# ---------------------------------------------------------------------------
# CorpusCoverageScorer
# ---------------------------------------------------------------------------


class CorpusCoverageScorer:
    """Scorer that checks if all expected corpora are cited.

    R10.1: corpus_coverage test type - ensures cross-law queries cite
    all relevant regulations.
    """

    def score(
        self,
        answer_text: str,
        cited_corpora: set[str],
        expected_corpora: set[str],
        min_corpora_cited: int | None = None,
        threshold: float | None = None,
    ) -> ScorerResult:
        """Score corpus coverage.

        Args:
            answer_text: The answer text (not used for coverage, but may be used for context)
            cited_corpora: Set of corpus IDs that were actually cited
            expected_corpora: Set of corpus IDs expected to be cited
            min_corpora_cited: Minimum required corpora (takes precedence over threshold)
            threshold: Coverage ratio threshold (0.0-1.0). When set and min_corpora_cited
                is None, passes if coverage_ratio >= threshold. Default None = 100%.

        Returns:
            ScorerResult with pass/fail, score (ratio), and message
        """
        if not expected_corpora:
            return ScorerResult(
                passed=True,
                score=1.0,
                message="No expected corpora specified",
            )

        # Calculate coverage
        covered = cited_corpora & expected_corpora
        coverage_ratio = len(covered) / len(expected_corpora)

        # Determine pass threshold (min_corpora_cited > threshold > 100%)
        if min_corpora_cited is not None:
            passed = len(covered) >= min_corpora_cited
        elif threshold is not None:
            passed = coverage_ratio >= threshold
        else:
            passed = covered == expected_corpora

        # Build message
        missing = expected_corpora - cited_corpora
        if passed:
            message = f"Covered {len(covered)}/{len(expected_corpora)} expected corpora"
        else:
            message = f"Missing corpora: {', '.join(sorted(missing))}"

        return ScorerResult(
            passed=passed,
            score=coverage_ratio,
            message=message,
        )


# ---------------------------------------------------------------------------
# SynthesisBalanceScorer
# ---------------------------------------------------------------------------


class SynthesisBalanceScorer:
    """Scorer that checks citations are balanced across corpora.

    R10.1: synthesis_balance test type - ensures no single corpus
    dominates the answer (>70% threshold).

    N/A for unified and routing modes where imbalance is acceptable.
    """

    def __init__(self, dominance_threshold: float = 0.70):
        """Initialize scorer.

        Args:
            dominance_threshold: Max proportion for any single corpus (default 70%)
        """
        self.dominance_threshold = dominance_threshold

    def score(
        self,
        citation_counts: dict[str, int],
        synthesis_mode: str,
    ) -> ScorerResult:
        """Score synthesis balance.

        Args:
            citation_counts: Dict mapping corpus_id to citation count
            synthesis_mode: The synthesis mode used

        Returns:
            ScorerResult with pass/fail, score, and message
        """
        # N/A for unified and routing modes
        if synthesis_mode in ("unified", "routing"):
            return ScorerResult(
                passed=True,
                score=1.0,
                message="N/A - balance not required for unified/routing mode",
            )

        # Handle empty citations
        total = sum(citation_counts.values())
        if total == 0:
            return ScorerResult(
                passed=False,
                score=0.0,
                message="No citations found",
            )

        # Find dominant corpus
        dominant_corpus = max(citation_counts, key=citation_counts.get)
        dominant_count = citation_counts[dominant_corpus]
        dominant_ratio = dominant_count / total

        # Check balance
        if dominant_ratio > self.dominance_threshold:
            return ScorerResult(
                passed=False,
                score=1.0 - dominant_ratio,  # Lower score for more imbalance
                message=f"Dominant corpus: {dominant_corpus} ({dominant_ratio:.0%} of citations)",
            )

        # Calculate balance score (inverse of max dominance)
        balance_score = 1.0 - (dominant_ratio - (1.0 / len(citation_counts)))

        return ScorerResult(
            passed=True,
            score=min(1.0, balance_score),
            message=f"Balanced citations across {len(citation_counts)} corpora",
        )


# ---------------------------------------------------------------------------
# CrossReferenceAccuracyScorer
# ---------------------------------------------------------------------------


# Patterns that indicate cross-law references
CROSS_REF_PATTERNS = [
    r"\breferences?\b.*\b(gdpr|ai.?act|nis2|dora|cra)\b",
    r"\b(gdpr|ai.?act|nis2|dora|cra)\b.*\breferences?\b",
    r"\bin\s+accordance\s+with\b.*\b(gdpr|ai.?act|nis2|dora|cra)\b",
    r"\bpursuant\s+to\b.*\b(gdpr|ai.?act|nis2|dora|cra)\b",
    r"\bas\s+defined\s+in\b.*\b(gdpr|ai.?act|nis2|dora|cra)\b",
    r"\bunder\b.*\b(gdpr|ai.?act|nis2|dora|cra)\b",
    # Danish patterns
    r"\bhenviser\s+til\b.*\b(gdpr|ai.?act|nis2|dora|cra)\b",
    r"\bi\s+overensstemmelse\s+med\b.*\b(gdpr|ai.?act|nis2|dora|cra)\b",
]


class CrossReferenceAccuracyScorer:
    """Scorer that validates inter-law reference claims.

    R10.1: cross_reference_accuracy test type - ensures that when the
    answer claims Law A references Law B, both laws are actually cited.
    """

    def score(
        self,
        answer_text: str,
        cited_corpora: set[str],
    ) -> ScorerResult:
        """Score cross-reference accuracy.

        Args:
            answer_text: The answer text to analyze
            cited_corpora: Set of corpus IDs that were cited

        Returns:
            ScorerResult with pass/fail, score, and message
        """
        text_lower = answer_text.lower()

        # Check for cross-reference patterns
        has_cross_ref = False
        for pattern in CROSS_REF_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                has_cross_ref = True
                break

        # If no cross-reference claims, pass
        if not has_cross_ref:
            return ScorerResult(
                passed=True,
                score=1.0,
                message="No cross-reference claims detected",
            )

        # Cross-reference claims require multiple corpora cited
        if len(cited_corpora) >= 2:
            return ScorerResult(
                passed=True,
                score=1.0,
                message=f"Cross-reference claim supported by {len(cited_corpora)} cited corpora",
            )

        return ScorerResult(
            passed=False,
            score=0.0,
            message="Cross-reference claim with only single corpus cited",
        )


# ---------------------------------------------------------------------------
# RoutingPrecisionScorer
# ---------------------------------------------------------------------------


class RoutingPrecisionScorer:
    """Scorer that checks correct laws are identified in routing queries.

    R10.1: routing_precision test type - ensures "Which law covers X?"
    queries correctly identify the relevant regulations.

    Uses retrieval-level evidence (context_corpora from references_structured_all)
    rather than alias-based text matching. This follows the RAGAS IDBasedContextRecall
    pattern: check structured corpus_id fields, not fragile name matching.
    """

    def score(
        self,
        context_corpora: set[str],
        expected_laws: set[str],
        synthesis_mode: str,
    ) -> ScorerResult:
        """Score routing precision using retrieval-level corpus evidence.

        Args:
            context_corpora: Set of corpus IDs found by the retrieval pipeline
            expected_laws: Set of corpus IDs expected to be routed to
            synthesis_mode: The synthesis mode used

        Returns:
            ScorerResult with pass/fail, score, and message
        """
        # N/A for non-routing modes
        if synthesis_mode != "routing":
            return ScorerResult(
                passed=True,
                score=1.0,
                message="N/A - routing precision only for routing mode",
            )

        if not expected_laws:
            return ScorerResult(
                passed=True,
                score=1.0,
                message="No expected laws specified",
            )

        # Check retrieval-level corpus presence
        correctly_routed = context_corpora & expected_laws
        precision = len(correctly_routed) / len(expected_laws)

        # 80% threshold for pass
        passed = precision >= 0.80

        if passed:
            message = f"Routed to {len(correctly_routed)}/{len(expected_laws)} expected laws"
        else:
            missing = expected_laws - context_corpora
            message = f"Missing laws: {', '.join(sorted(missing))}"

        return ScorerResult(
            passed=passed,
            score=precision,
            message=message,
        )


# ---------------------------------------------------------------------------
# ComparisonCompletenessScorer
# ---------------------------------------------------------------------------


class ComparisonCompletenessScorer:
    """Scorer that checks comparison targets are covered.

    R10.1: comparison_completeness test type - ensures "Compare A and B"
    queries cite both comparison targets.
    """

    def score(
        self,
        cited_corpora: set[str],
        target_corpora: set[str],
        synthesis_mode: str,
    ) -> ScorerResult:
        """Score comparison completeness.

        Args:
            cited_corpora: Set of corpus IDs that were cited
            target_corpora: Set of corpus IDs being compared
            synthesis_mode: The synthesis mode used

        Returns:
            ScorerResult with pass/fail, score, and message
        """
        # N/A for non-comparison modes
        if synthesis_mode != "comparison":
            return ScorerResult(
                passed=True,
                score=1.0,
                message="N/A - completeness only for comparison mode",
            )

        if len(target_corpora) < 2:
            return ScorerResult(
                passed=False,
                score=0.0,
                message="Comparison requires at least 2 target corpora",
            )

        # Check coverage
        covered = cited_corpora & target_corpora
        missing = target_corpora - cited_corpora

        if not missing:
            return ScorerResult(
                passed=True,
                score=1.0,
                message=f"All {len(target_corpora)} comparison targets cited",
            )

        return ScorerResult(
            passed=False,
            score=len(covered) / len(target_corpora),
            message=f"Missing comparison targets: {', '.join(sorted(missing))}",
        )


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------


def build_cross_law_scorers() -> dict[str, Any]:
    """Build scorer registry.

    Returns:
        Dict mapping scorer name to scorer instance
    """
    return {
        "corpus_coverage": CorpusCoverageScorer(),
        "synthesis_balance": SynthesisBalanceScorer(),
        "cross_reference_accuracy": CrossReferenceAccuracyScorer(),
        "routing_precision": RoutingPrecisionScorer(),
        "comparison_completeness": ComparisonCompletenessScorer(),
    }
