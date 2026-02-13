"""Tests for cross-law evaluation scorers.

TDD: These tests are written BEFORE the implementation.
"""

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# CorpusCoverageScorer Tests
# ---------------------------------------------------------------------------


class TestCorpusCoverageScorer:
    """Tests for CorpusCoverageScorer."""

    def test_ccs_001_passes_when_all_expected_corpora_cited(self):
        """Passes when all expected corpora are cited."""
        from src.eval.cross_law_scorers import CorpusCoverageScorer

        scorer = CorpusCoverageScorer()
        result = scorer.score(
            answer_text="See AI-Act Article 5 [1] and GDPR Article 6 [2].",
            cited_corpora={"ai_act", "gdpr"},
            expected_corpora={"ai_act", "gdpr"},
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_ccs_002_passes_when_min_corpora_met(self):
        """Passes when min_corpora_cited threshold is met."""
        from src.eval.cross_law_scorers import CorpusCoverageScorer

        scorer = CorpusCoverageScorer()
        result = scorer.score(
            answer_text="See AI-Act [1] and GDPR [2].",
            cited_corpora={"ai_act", "gdpr"},
            expected_corpora={"ai_act", "gdpr", "nis2"},
            min_corpora_cited=2,
        )

        assert result.passed is True

    def test_ccs_003_fails_when_insufficient_corpora(self):
        """Fails when insufficient corpora are cited."""
        from src.eval.cross_law_scorers import CorpusCoverageScorer

        scorer = CorpusCoverageScorer()
        result = scorer.score(
            answer_text="See AI-Act [1].",
            cited_corpora={"ai_act"},
            expected_corpora={"ai_act", "gdpr", "nis2"},
            min_corpora_cited=2,
        )

        assert result.passed is False

    def test_ccs_005_score_reflects_coverage_ratio(self):
        """Score reflects the ratio of covered corpora."""
        from src.eval.cross_law_scorers import CorpusCoverageScorer

        scorer = CorpusCoverageScorer()
        result = scorer.score(
            answer_text="See AI-Act [1] and GDPR [2].",
            cited_corpora={"ai_act", "gdpr"},
            expected_corpora={"ai_act", "gdpr", "nis2", "dora"},
        )

        assert result.score == 0.5  # 2/4

    def test_ccs_006_passes_when_threshold_met(self):
        """Passes when coverage ratio meets threshold (discovery-style)."""
        from src.eval.cross_law_scorers import CorpusCoverageScorer

        scorer = CorpusCoverageScorer()
        result = scorer.score(
            answer_text="Topic answer",
            cited_corpora={"dora", "nis2"},
            expected_corpora={"dora", "nis2", "cra"},
            threshold=0.5,  # 2/3 = 0.67 >= 0.5
        )

        assert result.passed is True
        assert result.score == pytest.approx(2 / 3, rel=0.01)

    def test_ccs_007_fails_when_below_threshold(self):
        """Fails when coverage ratio is below threshold."""
        from src.eval.cross_law_scorers import CorpusCoverageScorer

        scorer = CorpusCoverageScorer()
        result = scorer.score(
            answer_text="Topic answer",
            cited_corpora={"dora"},
            expected_corpora={"dora", "nis2", "cra"},
            threshold=0.8,  # 1/3 = 0.33 < 0.8
        )

        assert result.passed is False
        assert result.score == pytest.approx(1 / 3, rel=0.01)

    def test_ccs_008_min_corpora_takes_precedence_over_threshold(self):
        """min_corpora_cited takes precedence over threshold when both set."""
        from src.eval.cross_law_scorers import CorpusCoverageScorer

        scorer = CorpusCoverageScorer()
        result = scorer.score(
            answer_text="Topic answer",
            cited_corpora={"dora", "nis2"},
            expected_corpora={"dora", "nis2", "cra"},
            min_corpora_cited=2,
            threshold=0.8,  # Would fail (0.67 < 0.8) but min_corpora_cited=2 passes
        )

        assert result.passed is True


# ---------------------------------------------------------------------------
# SynthesisBalanceScorer Tests
# ---------------------------------------------------------------------------


class TestSynthesisBalanceScorer:
    """Tests for SynthesisBalanceScorer."""

    def test_sbs_001_passes_when_balanced(self):
        """Passes when citations are balanced (no single corpus dominates)."""
        from src.eval.cross_law_scorers import SynthesisBalanceScorer

        scorer = SynthesisBalanceScorer()
        result = scorer.score(
            citation_counts={"ai_act": 5, "gdpr": 5},
            synthesis_mode="aggregation",
        )

        assert result.passed is True

    def test_sbs_002_fails_when_single_corpus_dominates(self):
        """Fails when single corpus has >70% of citations."""
        from src.eval.cross_law_scorers import SynthesisBalanceScorer

        scorer = SynthesisBalanceScorer()
        result = scorer.score(
            citation_counts={"ai_act": 8, "gdpr": 2},
            synthesis_mode="aggregation",
        )

        assert result.passed is False
        assert "ai_act" in result.message.lower() or "dominant" in result.message.lower()

    def test_sbs_003_returns_na_for_unified_mode(self):
        """Returns N/A (passes with message) for unified mode."""
        from src.eval.cross_law_scorers import SynthesisBalanceScorer

        scorer = SynthesisBalanceScorer()
        result = scorer.score(
            citation_counts={"ai_act": 10, "gdpr": 0},
            synthesis_mode="unified",
        )

        assert result.passed is True
        assert "n/a" in result.message.lower() or "not applicable" in result.message.lower()

    def test_sbs_004_returns_na_for_routing_mode(self):
        """Returns N/A for routing mode."""
        from src.eval.cross_law_scorers import SynthesisBalanceScorer

        scorer = SynthesisBalanceScorer()
        result = scorer.score(
            citation_counts={"ai_act": 10},
            synthesis_mode="routing",
        )

        assert result.passed is True
        assert "n/a" in result.message.lower() or "not applicable" in result.message.lower()

    def test_sbs_005_handles_zero_citations(self):
        """Fails gracefully when no citations exist."""
        from src.eval.cross_law_scorers import SynthesisBalanceScorer

        scorer = SynthesisBalanceScorer()
        result = scorer.score(
            citation_counts={},
            synthesis_mode="aggregation",
        )

        assert result.passed is False


# ---------------------------------------------------------------------------
# CrossReferenceAccuracyScorer Tests
# ---------------------------------------------------------------------------


class TestCrossReferenceAccuracyScorer:
    """Tests for CrossReferenceAccuracyScorer."""

    def test_cra_001_passes_when_multiple_corpora_support_claim(self):
        """Passes when cross-reference claim is supported by multiple corpora."""
        from src.eval.cross_law_scorers import CrossReferenceAccuracyScorer

        scorer = CrossReferenceAccuracyScorer()
        result = scorer.score(
            answer_text="AI-Act references GDPR for personal data definitions.",
            cited_corpora={"ai_act", "gdpr"},
        )

        assert result.passed is True

    def test_cra_002_warns_on_single_corpus_cross_ref(self):
        """Warns when cross-reference claim has only single corpus cited."""
        from src.eval.cross_law_scorers import CrossReferenceAccuracyScorer

        scorer = CrossReferenceAccuracyScorer()
        result = scorer.score(
            answer_text="AI-Act references GDPR for data protection.",
            cited_corpora={"ai_act"},  # Only one corpus
        )

        assert result.passed is False

    def test_cra_003_passes_when_no_cross_ref_claims(self):
        """Passes when no cross-reference language detected."""
        from src.eval.cross_law_scorers import CrossReferenceAccuracyScorer

        scorer = CrossReferenceAccuracyScorer()
        result = scorer.score(
            answer_text="Article 6 defines the requirements.",
            cited_corpora={"ai_act"},
        )

        assert result.passed is True


# ---------------------------------------------------------------------------
# RoutingPrecisionScorer Tests
# ---------------------------------------------------------------------------


class TestRoutingPrecisionScorer:
    """Tests for RoutingPrecisionScorer (retrieval-level)."""

    def test_rps_001_passes_when_retrieval_finds_all_expected_laws(self):
        """Passes when retrieval found all expected corpora."""
        from src.eval.cross_law_scorers import RoutingPrecisionScorer

        scorer = RoutingPrecisionScorer()
        result = scorer.score(
            context_corpora={"ai_act", "gdpr"},
            expected_laws={"ai_act", "gdpr"},
            synthesis_mode="routing",
        )

        assert result.passed is True

    def test_rps_003_partial_match_gives_partial_score(self):
        """Partial retrieval gives partial score."""
        from src.eval.cross_law_scorers import RoutingPrecisionScorer

        scorer = RoutingPrecisionScorer()
        result = scorer.score(
            context_corpora={"ai_act"},
            expected_laws={"ai_act", "gdpr", "nis2"},
            synthesis_mode="routing",
        )

        # 1/3 = 0.33
        assert result.score == pytest.approx(1/3, rel=0.01)

    def test_rps_004_returns_na_for_non_routing_modes(self):
        """Returns N/A for non-routing synthesis modes."""
        from src.eval.cross_law_scorers import RoutingPrecisionScorer

        scorer = RoutingPrecisionScorer()
        result = scorer.score(
            context_corpora={"ai_act"},
            expected_laws={"ai_act"},
            synthesis_mode="aggregation",
        )

        assert result.passed is True
        assert "n/a" in result.message.lower() or "not applicable" in result.message.lower()


# ---------------------------------------------------------------------------
# ComparisonCompletenessScorer Tests
# ---------------------------------------------------------------------------


class TestComparisonCompletenessScorer:
    """Tests for ComparisonCompletenessScorer."""

    def test_cpc_001_passes_when_both_targets_cited(self):
        """Passes when both comparison targets are cited."""
        from src.eval.cross_law_scorers import ComparisonCompletenessScorer

        scorer = ComparisonCompletenessScorer()
        result = scorer.score(
            cited_corpora={"ai_act", "gdpr"},
            target_corpora={"ai_act", "gdpr"},
            synthesis_mode="comparison",
        )

        assert result.passed is True

    def test_cpc_002_fails_when_one_target_missing(self):
        """Fails when one comparison target is not cited."""
        from src.eval.cross_law_scorers import ComparisonCompletenessScorer

        scorer = ComparisonCompletenessScorer()
        result = scorer.score(
            cited_corpora={"ai_act"},
            target_corpora={"ai_act", "gdpr"},
            synthesis_mode="comparison",
        )

        assert result.passed is False
        assert "gdpr" in result.message.lower() or "missing" in result.message.lower()

    def test_cpc_004_returns_na_for_non_comparison_modes(self):
        """Returns N/A for non-comparison synthesis modes."""
        from src.eval.cross_law_scorers import ComparisonCompletenessScorer

        scorer = ComparisonCompletenessScorer()
        result = scorer.score(
            cited_corpora={"ai_act"},
            target_corpora={"ai_act", "gdpr"},
            synthesis_mode="aggregation",
        )

        assert result.passed is True
        assert "n/a" in result.message.lower() or "not applicable" in result.message.lower()

    def test_cpc_005_message_lists_missing_corpora(self):
        """Message lists which corpora are missing."""
        from src.eval.cross_law_scorers import ComparisonCompletenessScorer

        scorer = ComparisonCompletenessScorer()
        result = scorer.score(
            cited_corpora={"ai_act"},
            target_corpora={"ai_act", "gdpr", "nis2"},
            synthesis_mode="comparison",
        )

        assert result.passed is False
        # Message should mention missing corpora
        assert "gdpr" in result.message.lower() or "nis2" in result.message.lower()


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


class TestBuildCrossLawScorers:
    """Tests for build_cross_law_scorers factory."""

    def test_factory_includes_all_active_scorers(self):
        """Factory includes all active scorers (corpus_recall merged into coverage)."""
        from src.eval.cross_law_scorers import build_cross_law_scorers

        scorers = build_cross_law_scorers()
        assert "corpus_coverage" in scorers
        assert "synthesis_balance" in scorers
        assert "routing_precision" in scorers
        assert "comparison_completeness" in scorers
        assert "cross_reference_accuracy" in scorers
        assert "corpus_recall" not in scorers
