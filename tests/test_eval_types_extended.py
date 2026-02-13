"""Tests for extended GoldenCase with cross-law fields.

TDD: These tests are written BEFORE the implementation.
"""

import pytest


class TestGoldenCaseCrossLaw:
    """Tests for cross-law fields in GoldenCase."""

    def test_ete_001_accepts_corpus_scope(self):
        """GoldenCase accepts corpus_scope field."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        case = GoldenCase(
            id="test_1",
            profile="LEGAL",
            prompt="What do all laws say?",
            expected=ExpectedBehavior(),
            corpus_scope="all",
        )

        assert case.corpus_scope == "all"

    def test_ete_002_defaults_corpus_scope_to_single(self):
        """GoldenCase defaults corpus_scope to 'single' for backward compatibility."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        case = GoldenCase(
            id="test_1",
            profile="LEGAL",
            prompt="What is Article 6?",
            expected=ExpectedBehavior(),
        )

        assert case.corpus_scope == "single"

    def test_ete_003_accepts_target_corpora(self):
        """GoldenCase accepts target_corpora list."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        case = GoldenCase(
            id="test_1",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR",
            expected=ExpectedBehavior(),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
        )

        assert case.target_corpora == ("ai_act", "gdpr")

    def test_ete_004_accepts_synthesis_mode(self):
        """GoldenCase accepts synthesis_mode field."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        case = GoldenCase(
            id="test_1",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR",
            expected=ExpectedBehavior(),
            synthesis_mode="comparison",
        )

        assert case.synthesis_mode == "comparison"

    def test_ete_005_old_format_still_works(self):
        """Existing golden cases (without cross-law fields) still parse."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        # Old format without any cross-law fields
        case = GoldenCase(
            id="old_test",
            profile="ENGINEERING",
            prompt="What are the requirements?",
            expected=ExpectedBehavior(must_include_any_of=["requirement"]),
        )

        assert case.id == "old_test"
        assert case.profile == "ENGINEERING"
        # Should have defaults for cross-law fields
        assert case.corpus_scope == "single"
        assert case.target_corpora == ()
        assert case.synthesis_mode is None


class TestExpectedBehaviorCrossLaw:
    """Tests for cross-law fields in ExpectedBehavior."""

    def test_expected_behavior_accepts_min_corpora_cited(self):
        """ExpectedBehavior accepts min_corpora_cited for cross-law."""
        from src.eval.types import ExpectedBehavior

        expected = ExpectedBehavior(
            min_corpora_cited=2,
        )

        assert expected.min_corpora_cited == 2

    def test_expected_behavior_defaults_min_corpora_cited_to_none(self):
        """ExpectedBehavior defaults min_corpora_cited to None."""
        from src.eval.types import ExpectedBehavior

        expected = ExpectedBehavior()

        assert expected.min_corpora_cited is None

    def test_expected_behavior_accepts_required_corpora(self):
        """ExpectedBehavior accepts required_corpora list."""
        from src.eval.types import ExpectedBehavior

        expected = ExpectedBehavior(
            required_corpora=("ai_act", "gdpr"),
        )

        assert expected.required_corpora == ("ai_act", "gdpr")
