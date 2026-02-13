"""Tests for src/eval/scorers.py - Evaluation scorers."""

import os
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from src.eval.scorers import (
    RateLimitTracker,
    Score,
    GoldenExpected,
    _normalize_anchor,
    get_rate_limit_tracker,
    AnchorScorer,
    ContractScorer,
    PipelineBreakdownScorer,
    FaithfulnessScorer,
    AnswerRelevancyScorer,
    AbstentionScorer,
    _get_llm_judge_settings,
    _get_judge_model,
    _get_faithfulness_threshold,
    _get_relevancy_threshold,
    _get_claim_batch_size,
    _get_max_context_chars,
    _get_cache_enabled,
    _get_citation_verification_threshold,
)


class TestRateLimitTracker:
    """Tests for RateLimitTracker class."""

    def test_initial_state(self):
        """Tracker starts with zero counts."""
        tracker = RateLimitTracker()
        stats = tracker.get_stats()
        assert stats["total_hits"] == 0
        assert stats["total_retries"] == 0
        assert stats["total_tokens"] == 0
        assert stats["request_count"] == 0

    def test_record_hit_increments_counts(self):
        """record_hit increments hit and retry counts."""
        tracker = RateLimitTracker()
        tracker.record_hit(attempt=1, wait_time=1.5, context="test")

        stats = tracker.get_stats()
        assert stats["total_hits"] == 1
        assert stats["total_retries"] == 1
        assert len(stats["events"]) == 1
        assert stats["events"][0]["attempt"] == 1
        assert stats["events"][0]["wait_time"] == 1.5

    def test_record_multiple_hits(self):
        """Multiple hits are tracked cumulatively."""
        tracker = RateLimitTracker()
        tracker.record_hit(attempt=1, wait_time=1.0)
        tracker.record_hit(attempt=2, wait_time=2.0)
        tracker.record_hit(attempt=1, wait_time=1.5)

        assert tracker.hit_count == 3
        stats = tracker.get_stats()
        assert stats["total_hits"] == 3

    def test_record_usage_tracks_tokens(self):
        """record_usage tracks token counts."""
        tracker = RateLimitTracker()
        tracker.record_usage(prompt_tokens=100, completion_tokens=50)
        tracker.record_usage(prompt_tokens=200, completion_tokens=100)

        stats = tracker.get_stats()
        assert stats["total_prompt_tokens"] == 300
        assert stats["total_completion_tokens"] == 150
        assert stats["total_tokens"] == 450
        assert stats["request_count"] == 2

    def test_reset_clears_all(self):
        """reset clears all counters and events."""
        tracker = RateLimitTracker()
        tracker.record_hit(attempt=1, wait_time=1.0)
        tracker.record_usage(prompt_tokens=100, completion_tokens=50)

        tracker.reset()

        stats = tracker.get_stats()
        assert stats["total_hits"] == 0
        assert stats["total_tokens"] == 0
        assert stats["request_count"] == 0
        assert len(stats["events"]) == 0

    def test_hit_count_property(self):
        """hit_count property returns current count."""
        tracker = RateLimitTracker()
        assert tracker.hit_count == 0

        tracker.record_hit(attempt=1, wait_time=1.0)
        assert tracker.hit_count == 1

    def test_set_verbose(self):
        """set_verbose enables verbose mode."""
        tracker = RateLimitTracker()
        # Just verify it doesn't raise
        tracker.set_verbose(True)
        tracker.set_verbose(False)

    def test_events_limited_to_20(self):
        """get_stats limits events to last 20."""
        tracker = RateLimitTracker()
        for i in range(25):
            tracker.record_hit(attempt=1, wait_time=1.0)

        stats = tracker.get_stats()
        assert len(stats["events"]) == 20


class TestGetRateLimitTracker:
    """Tests for get_rate_limit_tracker function."""

    def test_returns_tracker(self):
        """Returns a RateLimitTracker instance."""
        tracker = get_rate_limit_tracker()
        assert isinstance(tracker, RateLimitTracker)

    def test_returns_same_instance(self):
        """Returns the same global instance."""
        tracker1 = get_rate_limit_tracker()
        tracker2 = get_rate_limit_tracker()
        assert tracker1 is tracker2


class TestScore:
    """Tests for Score dataclass."""

    def test_create_passing_score(self):
        """Create a passing score."""
        score = Score(passed=True, score=1.0, message="ok")
        assert score.passed is True
        assert score.score == 1.0
        assert score.message == "ok"
        assert score.details == {}

    def test_create_failing_score_with_details(self):
        """Create a failing score with details."""
        score = Score(
            passed=False,
            score=0.5,
            message="partial match",
            details={"missing": ["article:6"]}
        )
        assert score.passed is False
        assert score.score == 0.5
        assert score.details["missing"] == ["article:6"]

    def test_score_is_frozen(self):
        """Score dataclass is frozen (immutable)."""
        score = Score(passed=True, score=1.0, message="ok")
        with pytest.raises(AttributeError):
            score.passed = False


class TestGoldenExpected:
    """Tests for GoldenExpected dataclass."""

    def test_default_values(self):
        """Default values are empty lists."""
        expected = GoldenExpected()
        assert expected.must_include_any_of == []
        assert expected.must_include_any_of_2 == []
        assert expected.must_include_all_of == []

    def test_with_values(self):
        """Can create with anchor lists."""
        expected = GoldenExpected(
            must_include_any_of=["article:6", "article:7"],
            must_include_all_of=["annex:III"],
        )
        assert "article:6" in expected.must_include_any_of
        assert "annex:III" in expected.must_include_all_of


class TestNormalizeAnchor:
    """Tests for _normalize_anchor function."""

    def test_lowercase(self):
        """Converts to lowercase."""
        assert _normalize_anchor("ARTICLE:6") == "article:6"

    def test_removes_spaces(self):
        """Removes all spaces."""
        assert _normalize_anchor("Article 6") == "article6"
        assert _normalize_anchor("  article : 6  ") == "article:6"

    def test_handles_none(self):
        """Handles None input."""
        assert _normalize_anchor(None) == ""

    def test_handles_empty(self):
        """Handles empty string."""
        assert _normalize_anchor("") == ""

    def test_handles_numbers(self):
        """Handles numeric input."""
        assert _normalize_anchor(6) == "6"


class TestAnchorScorer:
    """Tests for AnchorScorer class."""

    def test_name_property(self):
        """Returns correct name."""
        scorer = AnchorScorer()
        assert scorer.name == "anchor_presence"

    def test_passes_when_any_of_found(self):
        """Passes when at least one must_include_any_of is present."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["article:6", "article:7"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": ["article:6"]},
            "retrieved_metadatas": [],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True

    def test_fails_when_any_of_missing(self):
        """Fails when none of must_include_any_of is present."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["article:6", "article:7"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": ["article:99"]},
            "retrieved_metadatas": [],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is False
        assert "missing_any_of" in score.message

    def test_passes_when_all_of_found(self):
        """Passes when all must_include_all_of are present."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_all_of=["article:6", "article:7"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": ["article:6", "article:7", "article:8"]},
            "retrieved_metadatas": [],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True

    def test_fails_when_all_of_missing(self):
        """Fails when not all must_include_all_of are present."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_all_of=["article:6", "article:7"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": ["article:6"]},
            "retrieved_metadatas": [],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is False
        assert "missing_all_of" in score.message

    def test_extracts_anchors_from_retrieved_metadatas(self):
        """Extracts anchors from retrieved_metadatas."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["article:6"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": []},
            "retrieved_metadatas": [{"article": "6"}],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True

    def test_empty_expected_passes(self):
        """Empty expected lists always pass."""
        scorer = AnchorScorer()
        expected = GoldenExpected()

        score = scorer.score(
            expected=expected,
            retrieval_metrics={"run": {}},
            references_structured=[],
        )

        assert score.passed is True
        assert score.score == 1.0

    def test_includes_position_details(self):
        """Score details include anchor positions."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["article:6"])

        refs = [{"article": "5"}, {"article": "6"}, {"article": "7"}]

        score = scorer.score(
            expected=expected,
            retrieval_metrics={"run": {"anchors_in_top_k": ["article:6"]}},
            references_structured=refs,
        )

        assert "positions" in score.details
        assert score.details["positions"]["article:6"] == 2  # 1-based position


class TestContractScorer:
    """Tests for ContractScorer class."""

    def test_name_property(self):
        """Returns correct name."""
        scorer = ContractScorer()
        assert scorer.name == "contract_compliance"

    def test_passes_when_contract_passed(self):
        """Passes when contract_validation.passed is True."""
        scorer = ContractScorer()
        expected = GoldenExpected()

        retrieval_metrics = {
            "run": {
                "contract_validation": {"passed": True}
            },
            "references_used_in_answer": ["ref1", "ref2"],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True
        assert score.score == 1.0
        assert "satisfied" in score.message

    def test_fails_when_contract_failed(self):
        """Fails when contract_validation.passed is False."""
        scorer = ContractScorer()
        expected = GoldenExpected()

        retrieval_metrics = {
            "run": {
                "contract_validation": {
                    "passed": False,
                    "violations": ["min_citations_not_met"]
                }
            }
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is False
        assert score.score == 0.0
        assert "violations" in score.message

    def test_handles_missing_contract_validation(self):
        """Passes when contract_validation is missing (defaults to passed)."""
        scorer = ContractScorer()
        expected = GoldenExpected()

        score = scorer.score(
            expected=expected,
            retrieval_metrics={"run": {}},
            references_structured=[],
        )

        assert score.passed is True


class TestPipelineBreakdownScorer:
    """Tests for PipelineBreakdownScorer class."""

    def test_name_property(self):
        """Returns correct name."""
        scorer = PipelineBreakdownScorer()
        assert scorer.name == "pipeline_breakdown"

    def test_always_passes(self):
        """Scorer always passes (diagnostic only)."""
        scorer = PipelineBreakdownScorer()
        expected = GoldenExpected()

        score = scorer.score(
            expected=expected,
            retrieval_metrics={"run": {}},
            references_structured=[],
        )

        assert score.passed is True


class TestConfigHelpers:
    """Tests for configuration helper functions."""

    def test_get_llm_judge_settings(self):
        """Returns LLM judge settings dict."""
        settings = _get_llm_judge_settings()
        assert isinstance(settings, dict)

    def test_get_judge_model(self):
        """Returns judge model name."""
        model = _get_judge_model()
        assert model is None or isinstance(model, str)

    def test_get_faithfulness_threshold(self):
        """Returns faithfulness threshold."""
        threshold = _get_faithfulness_threshold()
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_get_relevancy_threshold(self):
        """Returns relevancy threshold."""
        threshold = _get_relevancy_threshold()
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_get_claim_batch_size(self):
        """Returns claim batch size."""
        batch_size = _get_claim_batch_size()
        assert isinstance(batch_size, int)
        assert batch_size > 0

    def test_get_max_context_chars(self):
        """Returns max context chars."""
        max_chars = _get_max_context_chars()
        assert isinstance(max_chars, int)
        assert max_chars > 0

    def test_get_cache_enabled(self):
        """Returns cache enabled flag."""
        enabled = _get_cache_enabled()
        assert isinstance(enabled, bool)

    def test_get_citation_verification_threshold(self):
        """Returns citation verification threshold."""
        threshold = _get_citation_verification_threshold()
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0


class TestAnchorScorerEdgeCases:
    """Edge case tests for AnchorScorer."""

    def test_handles_non_dict_refs(self):
        """Skips non-dict items in references_structured."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["article:6"])

        refs = [None, "invalid", {"article": "6"}, 123]

        score = scorer.score(
            expected=expected,
            retrieval_metrics={"run": {"anchors_in_top_k": ["article:6"]}},
            references_structured=refs,
        )

        # Should still find position 3 (1-based)
        assert score.details["positions"]["article:6"] == 3

    def test_handles_recital_and_annex(self):
        """Handles recital and annex anchors."""
        scorer = AnchorScorer()
        expected = GoldenExpected(
            must_include_any_of=["recital:42"],
            must_include_all_of=["annex:III"],
        )

        retrieval_metrics = {
            "run": {"anchors_in_top_k": []},
            "retrieved_metadatas": [
                {"recital": "42"},
                {"annex": "III"},
            ],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True

    def test_normalizes_anchor_comparison(self):
        """Anchors are normalized for comparison (case-insensitive)."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["ARTICLE:6"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": ["article:6"]},
            "retrieved_metadatas": [],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True


class TestAnchorScorerCorpusQualified:
    """Tests for corpus-qualified anchor matching in AnchorScorer.

    Cross-law inverted generation produces corpus-qualified anchors like
    'ai-act:article:6'. The scorer must match these against retrieved
    metadatas that carry corpus_id + article fields.
    """

    def test_matches_corpus_qualified_anchors_from_metadatas(self):
        """Corpus-qualified anchors match against retrieved_metadatas with corpus_id."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["ai-act:article:6"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": []},
            "retrieved_metadatas": [
                {"article": "6", "corpus_id": "ai-act"},
            ],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True

    def test_corpus_qualified_does_not_match_wrong_corpus(self):
        """ai-act:article:6 must NOT match when only gdpr has article 6."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["ai-act:article:6"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": []},
            "retrieved_metadatas": [
                {"article": "6", "corpus_id": "gdpr"},  # Wrong corpus!
            ],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is False

    def test_plain_anchors_still_work(self):
        """Plain anchors like article:6 still match (backward compat)."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["article:6"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": ["article:6"]},
            "retrieved_metadatas": [],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True

    def test_multiple_corpus_qualified_anchors(self):
        """Multiple corpus-qualified anchors from different corpora."""
        scorer = AnchorScorer()
        expected = GoldenExpected(
            must_include_all_of=["ai-act:article:6", "gdpr:article:35"]
        )

        retrieval_metrics = {
            "run": {"anchors_in_top_k": []},
            "retrieved_metadatas": [
                {"article": "6", "corpus_id": "ai-act"},
                {"article": "35", "corpus_id": "gdpr"},
            ],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True

    def test_corpus_qualified_annex_matches(self):
        """Corpus-qualified annex like ai-act:annex:III matches."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["ai-act:annex:III"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": []},
            "retrieved_metadatas": [
                {"annex": "III", "corpus_id": "ai-act"},
            ],
        }

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=[],
        )

        assert score.passed is True


class TestAnchorScorerRefsStructured:
    """Tests for corpus-qualified anchor matching via references_structured.

    In multi-corpus eval, retrieved_metadatas may be empty (rag.py bug),
    but references_structured has corpus_id. The scorer must build
    corpus-qualified anchors from references_structured as well.
    """

    def test_corpus_qualified_from_refs_structured(self):
        """Corpus-qualified anchors match via references_structured even with empty metadatas."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["data-act:article:1"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": ["article:1"]},
            "retrieved_metadatas": [],  # Empty! (multi-corpus bug)
        }

        refs_structured = [
            {"article": "1", "corpus_id": "data-act", "idx": 1},
        ]

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=refs_structured,
        )

        assert score.passed is True

    def test_refs_structured_wrong_corpus_no_match(self):
        """Corpus-qualified anchor from wrong corpus in refs_structured must not match."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["data-act:article:1"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": ["article:1"]},
            "retrieved_metadatas": [],
        }

        refs_structured = [
            {"article": "1", "corpus_id": "gdpr", "idx": 1},  # Wrong corpus
        ]

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=refs_structured,
        )

        assert score.passed is False

    def test_plain_anchors_still_match_from_refs_structured(self):
        """Plain anchors match when refs_structured provides articles."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["article:6"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": []},
            "retrieved_metadatas": [],
        }

        refs_structured = [
            {"article": "6", "corpus_id": "gdpr", "idx": 1},
        ]

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=refs_structured,
        )

        assert score.passed is True

    def test_annex_corpus_qualified_from_refs_structured(self):
        """Corpus-qualified annex anchors match via refs_structured."""
        scorer = AnchorScorer()
        expected = GoldenExpected(must_include_any_of=["nis2:annex:1"])

        retrieval_metrics = {
            "run": {"anchors_in_top_k": []},
            "retrieved_metadatas": [],
        }

        refs_structured = [
            {"annex": "1", "corpus_id": "nis2", "idx": 1},
        ]

        score = scorer.score(
            expected=expected,
            retrieval_metrics=retrieval_metrics,
            references_structured=refs_structured,
        )

        assert score.passed is True


class TestFaithfulnessScorer:
    """Tests for FaithfulnessScorer class."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache that returns None (no cached results)."""
        cache = MagicMock()
        cache.get.return_value = None
        cache.set.return_value = None
        return cache

    @pytest.fixture
    def mock_llm_client(self):
        """Create a mock OpenAI client."""
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = '["Claim 1", "Claim 2"]'
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        client.chat.completions.create.return_value = response
        return client

    def test_name_property(self):
        """Returns correct name."""
        with patch("src.eval.scorers.get_cache") as mock_get_cache:
            mock_get_cache.return_value = MagicMock()
            scorer = FaithfulnessScorer()
            assert scorer.name == "faithfulness"

    def test_default_threshold(self):
        """Uses config threshold by default."""
        with patch("src.eval.scorers.get_cache") as mock_get_cache:
            mock_get_cache.return_value = MagicMock()
            scorer = FaithfulnessScorer()
            # Threshold comes from config, typically 0.8
            assert 0.0 <= scorer.threshold <= 1.0

    def test_custom_threshold(self):
        """Accepts custom threshold."""
        with patch("src.eval.scorers.get_cache") as mock_get_cache:
            mock_get_cache.return_value = MagicMock()
            scorer = FaithfulnessScorer(threshold=0.9)
            assert scorer.threshold == 0.9

    def test_returns_cached_result(self, mock_cache):
        """Returns cached result if available."""
        cached_result = {
            "passed": True,
            "score": 0.95,
            "message": "Cached result",
            "details": {"cached": True},
        }
        mock_cache.get.return_value = cached_result

        with patch("src.eval.scorers.get_cache", return_value=mock_cache):
            scorer = FaithfulnessScorer()
            result = scorer.score(
                question="Test question?",
                answer="Test answer.",
                context="Test context.",
            )

            assert result.passed is True
            assert result.score == 0.95
            assert result.message == "Cached result"
            mock_cache.get.assert_called_once()

    def test_no_claims_returns_perfect_score(self, mock_cache, mock_llm_client):
        """Returns perfect score when no claims to verify."""
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = "[]"

        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._make_llm_client", return_value=mock_llm_client):
            scorer = FaithfulnessScorer()
            result = scorer.score(
                question="Test?",
                answer="",
                context="Context",
            )

            assert result.passed is True
            assert result.score == 1.0
            assert "No factual claims" in result.message

    def test_all_claims_supported(self, mock_cache, mock_llm_client):
        """Returns high score when all claims supported."""
        # First call extracts claims
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = '["Claim 1", "Claim 2"]'

        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._make_llm_client", return_value=mock_llm_client), \
             patch("src.eval.scorers._call_llm_json") as mock_call_json:
            # Mock verification response - all supported
            mock_call_json.return_value = {
                "verifications": [
                    {"verdict": "SUPPORTED", "explanation": "Found in context"},
                    {"verdict": "SUPPORTED", "explanation": "Found in context"},
                ]
            }

            scorer = FaithfulnessScorer(threshold=0.7)
            result = scorer.score(
                question="Test question?",
                answer="Claim 1. Claim 2.",
                context="Context supporting both claims.",
            )

            assert result.score == 1.0
            assert result.passed is True
            assert "All claims supported" in result.message

    def test_some_claims_unsupported(self, mock_cache, mock_llm_client):
        """Returns partial score when some claims unsupported."""
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = '["Claim 1", "Claim 2"]'

        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._make_llm_client", return_value=mock_llm_client), \
             patch("src.eval.scorers._call_llm_json") as mock_call_json:
            # One supported, one not
            mock_call_json.return_value = {
                "verifications": [
                    {"verdict": "SUPPORTED", "explanation": "Found"},
                    {"verdict": "NOT_SUPPORTED", "explanation": "Not found"},
                ]
            }

            scorer = FaithfulnessScorer(threshold=0.7)
            result = scorer.score(
                question="Test?",
                answer="Claim 1. Claim 2.",
                context="Context.",
            )

            assert result.score == 0.5  # 1/2 claims supported
            assert result.passed is False  # Below 0.7 threshold
            assert "1/2 claims supported" in result.message

    def test_caches_result(self, mock_cache, mock_llm_client):
        """Caches result after evaluation."""
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = "[]"

        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._make_llm_client", return_value=mock_llm_client):
            scorer = FaithfulnessScorer()
            scorer.score(
                question="Test?",
                answer="Answer.",
                context="Context.",
            )

            # Verify cache.set was called
            mock_cache.set.assert_called()

    def test_extract_claims_fallback_on_error(self, mock_cache, mock_llm_client):
        """Falls back to sentence splitting on LLM error."""
        mock_llm_client.chat.completions.create.side_effect = Exception("API Error")

        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._make_llm_client", return_value=mock_llm_client), \
             patch("src.eval.scorers._call_llm_json") as mock_call_json:
            mock_call_json.return_value = {
                "verifications": [
                    {"verdict": "SUPPORTED", "explanation": "ok"},
                ]
            }

            scorer = FaithfulnessScorer()
            result = scorer.score(
                question="Test?",
                answer="This is a longer sentence that should be treated as a claim.",
                context="Context.",
            )

            # Should still produce a result using fallback
            assert isinstance(result, Score)

    def test_progress_callback_called(self, mock_cache, mock_llm_client):
        """Calls progress callback during evaluation."""
        mock_llm_client.chat.completions.create.return_value.choices[0].message.content = '["Claim"]'

        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._make_llm_client", return_value=mock_llm_client), \
             patch("src.eval.scorers._call_llm_json") as mock_call_json:
            mock_call_json.return_value = {
                "verifications": [{"verdict": "SUPPORTED", "explanation": "ok"}]
            }

            callback = MagicMock()
            scorer = FaithfulnessScorer()
            scorer.score(
                question="Test?",
                answer="Claim.",
                context="Context.",
                progress_callback=callback,
            )

            # Progress callback should have been called at least once
            assert callback.call_count >= 1


class TestAnswerRelevancyScorer:
    """Tests for AnswerRelevancyScorer class."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache."""
        cache = MagicMock()
        cache.get.return_value = None
        cache.set.return_value = None
        return cache

    def test_name_property(self):
        """Returns correct name."""
        with patch("src.eval.scorers.get_cache") as mock_get_cache:
            mock_get_cache.return_value = MagicMock()
            scorer = AnswerRelevancyScorer()
            assert scorer.name == "answer_relevancy"

    def test_default_threshold(self):
        """Uses config threshold by default."""
        with patch("src.eval.scorers.get_cache") as mock_get_cache:
            mock_get_cache.return_value = MagicMock()
            scorer = AnswerRelevancyScorer()
            assert 0.0 <= scorer.threshold <= 1.0

    def test_custom_threshold(self):
        """Accepts custom threshold."""
        with patch("src.eval.scorers.get_cache") as mock_get_cache:
            mock_get_cache.return_value = MagicMock()
            scorer = AnswerRelevancyScorer(threshold=0.8)
            assert scorer.threshold == 0.8

    def test_returns_cached_result(self, mock_cache):
        """Returns cached result if available."""
        cached = {
            "passed": True,
            "score": 0.9,
            "message": "Cached",
            "details": {},
        }
        mock_cache.get.return_value = cached

        with patch("src.eval.scorers.get_cache", return_value=mock_cache):
            scorer = AnswerRelevancyScorer()
            result = scorer.score(
                question="Test?",
                answer="Answer.",
            )

            assert result.passed is True
            assert result.score == 0.9
            mock_cache.get.assert_called_once()

    def test_high_relevancy_score(self, mock_cache):
        """Returns passing score for relevant answer."""
        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._call_llm_json") as mock_call:
            mock_call.return_value = {
                "score": 9,
                "critique": "Very relevant answer",
            }

            scorer = AnswerRelevancyScorer(threshold=0.7)
            result = scorer.score(
                question="What is NIS2?",
                answer="NIS2 is a cybersecurity directive...",
            )

            assert result.score == 0.9  # 9/10
            assert result.passed is True
            assert "9/10" in result.message
            assert "✓" in result.message

    def test_low_relevancy_score(self, mock_cache):
        """Returns failing score for irrelevant answer."""
        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._call_llm_json") as mock_call:
            mock_call.return_value = {
                "score": 3,
                "critique": "Answer doesn't address the question",
            }

            scorer = AnswerRelevancyScorer(threshold=0.7)
            result = scorer.score(
                question="What is NIS2?",
                answer="The weather is nice today.",
            )

            assert result.score == 0.3  # 3/10
            assert result.passed is False
            assert "3/10" in result.message
            assert "✗" in result.message

    def test_normalizes_score_to_0_1(self, mock_cache):
        """Normalizes 0-10 score to 0-1 range."""
        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._call_llm_json") as mock_call:
            mock_call.return_value = {"score": 5, "critique": "Average"}

            scorer = AnswerRelevancyScorer()
            result = scorer.score(question="Q?", answer="A.")

            assert result.score == 0.5

    def test_handles_out_of_range_scores(self, mock_cache):
        """Clamps scores to 0-1 range."""
        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._call_llm_json") as mock_call:
            # Score > 10
            mock_call.return_value = {"score": 15, "critique": "..."}
            scorer = AnswerRelevancyScorer()
            result = scorer.score(question="Q?", answer="A.")
            assert result.score == 1.0

    def test_handles_llm_error(self, mock_cache):
        """Returns failing score on LLM error."""
        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._call_llm_json") as mock_call:
            mock_call.side_effect = Exception("API Error")

            scorer = AnswerRelevancyScorer()
            result = scorer.score(question="Q?", answer="A.")

            assert result.passed is False
            assert result.score == 0.0
            assert "failed" in result.message.lower()

    def test_caches_result(self, mock_cache):
        """Caches result after evaluation."""
        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._call_llm_json") as mock_call:
            mock_call.return_value = {"score": 8, "critique": "Good"}

            scorer = AnswerRelevancyScorer()
            scorer.score(question="Q?", answer="A.")

            mock_cache.set.assert_called_once()

    def test_progress_callback_called(self, mock_cache):
        """Calls progress callback during evaluation."""
        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._call_llm_json") as mock_call:
            mock_call.return_value = {"score": 7, "critique": "Ok"}

            callback = MagicMock()
            scorer = AnswerRelevancyScorer()
            scorer.score(
                question="Q?",
                answer="A.",
                progress_callback=callback,
            )

            callback.assert_called_once()

    def test_includes_critique_in_details(self, mock_cache):
        """Includes critique in score details."""
        with patch("src.eval.scorers.get_cache", return_value=mock_cache), \
             patch("src.eval.scorers._call_llm_json") as mock_call:
            mock_call.return_value = {
                "score": 8,
                "critique": "Good but could be more specific",
            }

            scorer = AnswerRelevancyScorer()
            result = scorer.score(question="Q?", answer="A.")

            assert result.details["critique"] == "Good but could be more specific"
            assert result.details["raw_score"] == 8


class TestAbstentionScorer:
    """Tests for AbstentionScorer class."""

    def test_name_property(self):
        """Returns correct name."""
        scorer = AbstentionScorer()
        assert scorer.name == "abstention"

    def test_correct_abstention_danish(self):
        """Passes when answer contains Danish abstention indicators."""
        scorer = AbstentionScorer()
        result = scorer.score(
            question="Er vi compliant med artikel 21?",
            answer="Det kan ikke vurderes uden nærmere information om jeres sikkerhedsforanstaltninger.",
        )

        assert result.passed is True
        assert result.score == 1.0
        assert "Correctly abstained" in result.message

    def test_correct_abstention_english(self):
        """Passes when answer contains English abstention indicators."""
        scorer = AbstentionScorer()
        result = scorer.score(
            question="Are we compliant with article 21?",
            answer="I cannot determine compliance without more specific information about your security measures.",
        )

        assert result.passed is True
        assert result.score == 1.0
        assert "Correctly abstained" in result.message

    def test_incorrect_definitive_answer_danish(self):
        """Fails when answer gives definitive answer (Danish)."""
        scorer = AbstentionScorer()
        result = scorer.score(
            question="Er vi compliant?",
            answer="Ja, i er compliant med alle krav.",
        )

        assert result.passed is False
        assert result.score == 0.0
        assert "definitive answer" in result.message.lower()

    def test_incorrect_definitive_answer_english(self):
        """Fails when answer gives definitive answer (English)."""
        scorer = AbstentionScorer()
        result = scorer.score(
            question="Are we compliant?",
            answer="Yes, you are compliant with all requirements.",
        )

        assert result.passed is False
        assert result.score == 0.0

    def test_implicit_abstention(self):
        """Soft pass when no clear abstention or definitive answer."""
        scorer = AbstentionScorer()
        result = scorer.score(
            question="Are we compliant?",
            answer="The requirements in article 21 cover various security measures.",
        )

        # No definitive yes/no, but also no explicit abstention
        assert result.passed is True
        assert result.score == 0.7  # Soft pass
        assert "implicit abstention" in result.message.lower()

    def test_multiple_abstention_indicators(self):
        """Counts multiple abstention indicators."""
        scorer = AbstentionScorer()
        result = scorer.score(
            question="Er vi compliant?",
            answer="Det afhænger af mange faktorer og kræver yderligere information.",
        )

        assert result.passed is True
        assert result.score == 1.0
        # Should mention multiple indicators found
        assert "indicator" in result.message.lower()

    def test_case_insensitive_matching(self):
        """Matches indicators case-insensitively."""
        scorer = AbstentionScorer()
        result = scorer.score(
            question="Are we compliant?",
            answer="CANNOT DETERMINE without more information.",
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_definitive_overrides_abstention(self):
        """Fails if both abstention and definitive indicators present."""
        scorer = AbstentionScorer()
        result = scorer.score(
            question="Er vi compliant?",
            answer="Ja, i er compliant, men det afhænger af yderligere faktorer.",
        )

        # Definitive answer should cause failure even with abstention indicators
        assert result.passed is False
        assert result.score == 0.0
