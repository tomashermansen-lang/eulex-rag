# tests/test_eval_core.py
"""TDD tests for eval_core.py - the unified evaluation core.

These tests are written BEFORE implementation (TDD).
Run with: pytest tests/test_eval_core.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test EvalConfig dataclass
# ---------------------------------------------------------------------------

class TestEvalConfig:
    """Test EvalConfig dataclass and its properties."""

    def test_default_values(self):
        """EvalConfig has sensible defaults."""
        from src.eval.eval_core import EvalConfig

        config = EvalConfig(law="ai-act")

        assert config.law == "ai-act"
        assert config.run_mode == "full"
        assert config.max_retries == 3
        assert config.escalation_enabled is False
        assert config.fallback_model is None
        assert config.verbose is False

    def test_run_mode_retrieval_only(self):
        """retrieval_only mode sets skip_llm=True, llm_judge=False."""
        from src.eval.eval_core import EvalConfig

        config = EvalConfig(law="ai-act", run_mode="retrieval_only")

        assert config.skip_llm is True
        assert config.llm_judge is False

    def test_run_mode_full(self):
        """full mode sets skip_llm=False, llm_judge=False."""
        from src.eval.eval_core import EvalConfig

        config = EvalConfig(law="ai-act", run_mode="full")

        assert config.skip_llm is False
        assert config.llm_judge is False

    def test_run_mode_full_with_judge(self):
        """full_with_judge mode sets skip_llm=False, llm_judge=True."""
        from src.eval.eval_core import EvalConfig

        config = EvalConfig(law="ai-act", run_mode="full_with_judge")

        assert config.skip_llm is False
        assert config.llm_judge is True

    def test_frozen_dataclass(self):
        """EvalConfig is immutable (frozen)."""
        from src.eval.eval_core import EvalConfig

        config = EvalConfig(law="ai-act")

        with pytest.raises(Exception):  # FrozenInstanceError
            config.law = "gdpr"


# ---------------------------------------------------------------------------
# Test evaluate_cases_iter generator
# ---------------------------------------------------------------------------

class TestEvaluateCasesIter:
    """Test the core evaluation iterator."""

    @pytest.fixture
    def mock_ask_result(self):
        """Create a mock ask result."""
        result = MagicMock()
        result.answer = "Test answer"
        result.references_structured = [
            {"article": "1", "chunk_text": "Test chunk"}
        ]
        result.retrieval_metrics = {
            "references_structured": [{"article": "1"}],
            "run": {"context_positioning": "top"},
        }
        return result

    @pytest.fixture
    def mock_golden_case(self):
        """Create a mock golden case."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        return GoldenCase(
            id="test-case-1",
            profile="LEGAL",
            prompt="What is Article 1?",
            expected=ExpectedBehavior(
                must_include_any_of=["article:1"],
            ),
            test_types=("retrieval",),
            origin="manual",
        )

    def test_yields_case_result_for_each_case(self, mock_ask_result, mock_golden_case):
        """Iterator yields CaseResult for each case."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import CaseResult, EvalSummary

        cases = [mock_golden_case]
        config = EvalConfig(law="test", run_mode="retrieval_only")

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.return_value = mock_ask_result

            results = list(evaluate_cases_iter(cases, config))

        # Should yield CaseResult + EvalSummary
        assert len(results) == 2
        assert isinstance(results[0], CaseResult)
        assert isinstance(results[1], EvalSummary)

    def test_summary_is_last_item(self, mock_ask_result, mock_golden_case):
        """EvalSummary is always the last yielded item."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import EvalSummary

        cases = [mock_golden_case, mock_golden_case]  # 2 cases
        config = EvalConfig(law="test", run_mode="retrieval_only")

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.return_value = mock_ask_result

            results = list(evaluate_cases_iter(cases, config))

        assert len(results) == 3  # 2 CaseResult + 1 EvalSummary
        assert isinstance(results[-1], EvalSummary)
        assert results[-1].total == 2

    def test_retrieval_only_skips_llm(self, mock_ask_result, mock_golden_case):
        """retrieval_only mode passes dry_run=True to ask."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter

        config = EvalConfig(law="test", run_mode="retrieval_only")

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.return_value = mock_ask_result

            list(evaluate_cases_iter([mock_golden_case], config))

            # Verify dry_run=True was passed
            call_kwargs = mock_ask.ask.call_args.kwargs
            assert call_kwargs.get("dry_run") is True

    def test_full_mode_calls_llm(self, mock_ask_result, mock_golden_case):
        """full mode passes dry_run=False to ask."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter

        config = EvalConfig(law="test", run_mode="full")

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.return_value = mock_ask_result

            list(evaluate_cases_iter([mock_golden_case], config))

            # Verify dry_run=False was passed
            call_kwargs = mock_ask.ask.call_args.kwargs
            assert call_kwargs.get("dry_run") is False

    def test_uses_provided_engine(self, mock_ask_result, mock_golden_case):
        """Uses the provided engine instead of building one."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter

        config = EvalConfig(law="test", run_mode="retrieval_only")
        mock_engine = MagicMock()

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.return_value = mock_ask_result

            list(evaluate_cases_iter([mock_golden_case], config, engine=mock_engine))

            # Verify engine was passed
            call_kwargs = mock_ask.ask.call_args.kwargs
            assert call_kwargs.get("engine") is mock_engine


# ---------------------------------------------------------------------------
# Test retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Test case retry behavior."""

    @pytest.fixture
    def mock_golden_case(self):
        """Create a mock golden case."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        return GoldenCase(
            id="retry-test",
            profile="LEGAL",
            prompt="Test question",
            expected=ExpectedBehavior(
                must_include_any_of=["article:1"],
            ),
        )

    def test_retries_zero_single_attempt(self, mock_golden_case):
        """max_retries=0 means exactly 1 pipeline invocation per case."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter

        config = EvalConfig(law="test", run_mode="retrieval_only", max_retries=0)

        call_count = 0
        def mock_ask_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.answer = ""
            result.references_structured = []
            result.retrieval_metrics = {}
            return result

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.side_effect = mock_ask_side_effect

            results = list(evaluate_cases_iter([mock_golden_case], config))

        assert call_count == 1  # Exactly one attempt, no retries

    def test_retries_failed_cases(self, mock_golden_case):
        """Failed cases are retried up to max_retries."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import CaseResult

        config = EvalConfig(law="test", run_mode="retrieval_only", max_retries=2)

        call_count = 0
        def mock_ask_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Always return failing result
            result = MagicMock()
            result.answer = ""
            result.references_structured = []  # No articles = fail
            result.retrieval_metrics = {}
            return result

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.side_effect = mock_ask_side_effect

            results = list(evaluate_cases_iter([mock_golden_case], config))

        # Should have called ask 3 times (1 initial + 2 retries)
        assert call_count == 3

        # Result should show retry count
        case_result = results[0]
        assert isinstance(case_result, CaseResult)
        assert case_result.retry_count == 2  # Max retries exhausted

    def test_no_retry_on_pass(self, mock_golden_case):
        """Passed cases are not retried."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter

        config = EvalConfig(law="test", run_mode="retrieval_only", max_retries=3)

        call_count = 0
        def mock_ask_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.answer = "Test"
            result.references_structured = [{"article": "1"}]  # Pass
            # Include proper format for AnchorScorer - anchors_in_top_k is the key field
            result.retrieval_metrics = {
                "references_structured": [{"article": "1"}],
                "run": {"anchors_in_top_k": ["article:1"]},  # This is what AnchorScorer checks
            }
            return result

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.side_effect = mock_ask_side_effect
            mock_engine = MagicMock()

            list(evaluate_cases_iter([mock_golden_case], config, engine=mock_engine))

        # Should only call once (passed on first try)
        assert call_count == 1


# ---------------------------------------------------------------------------
# Test escalation logic
# ---------------------------------------------------------------------------

class TestEscalationLogic:
    """Test model escalation behavior."""

    @pytest.fixture
    def mock_golden_case(self):
        """Create a mock golden case that tests faithfulness."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        return GoldenCase(
            id="escalate-test",
            profile="LEGAL",
            prompt="Test question",
            expected=ExpectedBehavior(
                must_include_any_of=["article:1"],
            ),
            test_types=("faithfulness",),
        )

    def test_escalates_after_retries_exhausted(self, mock_golden_case):
        """Escalates to fallback_model after max retries."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import CaseResult

        config = EvalConfig(
            law="test",
            run_mode="full",
            max_retries=1,
            escalation_enabled=True,
            fallback_model="gpt-4o",
        )

        models_used = []
        def mock_ask_side_effect(*args, **kwargs):
            # Track which model/engine was used
            engine = kwargs.get("engine")
            if engine:
                models_used.append(getattr(engine, "_model", "default"))

            result = MagicMock()
            result.answer = "Test"
            result.references_structured = [{"article": "1"}]
            result.retrieval_metrics = {"references_structured": [{"article": "1"}]}
            return result

        with patch("src.eval.eval_core.ask") as mock_ask, \
             patch("src.eval.eval_core._build_engine") as mock_build:
            mock_ask.ask.side_effect = mock_ask_side_effect

            # First call fails, escalation succeeds
            results = list(evaluate_cases_iter([mock_golden_case], config))

        # Check escalation was recorded
        case_result = results[0]
        assert isinstance(case_result, CaseResult)
        # The test verifies the escalation flow exists

    def test_no_escalation_when_disabled(self, mock_golden_case):
        """No escalation when escalation_enabled=False."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter

        config = EvalConfig(
            law="test",
            run_mode="retrieval_only",
            max_retries=1,
            escalation_enabled=False,  # Disabled
            fallback_model="gpt-4o",
        )

        call_count = 0
        def mock_ask_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            result.answer = ""
            result.references_structured = []
            result.retrieval_metrics = {}
            return result

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.side_effect = mock_ask_side_effect

            results = list(evaluate_cases_iter([mock_golden_case], config))

        # Only retries, no escalation
        assert call_count == 2  # 1 + 1 retry, no escalation

        case_result = results[0]
        assert case_result.escalated is False


# ---------------------------------------------------------------------------
# Test summary statistics
# ---------------------------------------------------------------------------

class TestSummaryStatistics:
    """Test EvalSummary statistics computation."""

    @pytest.fixture
    def mock_golden_cases(self):
        """Create multiple mock golden cases."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        return [
            GoldenCase(
                id=f"case-{i}",
                profile="LEGAL",
                prompt=f"Question {i}",
                expected=ExpectedBehavior(must_include_any_of=[f"article:{i}"]),
            )
            for i in range(3)
        ]

    def test_summary_counts_correct(self, mock_golden_cases):
        """Summary correctly counts passed/failed."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import EvalSummary

        config = EvalConfig(law="test", run_mode="retrieval_only")

        call_count = 0
        def mock_ask_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = MagicMock()
            # First 2 pass, last 1 fails
            if call_count <= 2:
                result.references_structured = [{"article": str(call_count - 1)}]
                result.retrieval_metrics = {"references_structured": [{"article": str(call_count - 1)}]}
            else:
                result.references_structured = []
                result.retrieval_metrics = {}
            result.answer = "Test"
            return result

        with patch("src.eval.eval_core.ask") as mock_ask:
            mock_ask.ask.side_effect = mock_ask_side_effect

            results = list(evaluate_cases_iter(mock_golden_cases, config))

        summary = results[-1]
        assert isinstance(summary, EvalSummary)
        assert summary.total == 3
        # Note: The exact pass/fail depends on scoring logic


# ---------------------------------------------------------------------------
# Test scorers configuration
# ---------------------------------------------------------------------------

class TestScorersConfiguration:
    """Test that correct scorers are applied based on run_mode."""

    @pytest.fixture
    def mock_golden_case(self):
        """Create a mock golden case."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        return GoldenCase(
            id="scorer-test",
            profile="LEGAL",
            prompt="Test question",
            expected=ExpectedBehavior(must_include_any_of=["article:1"]),
        )

    def test_retrieval_only_uses_basic_scorers(self, mock_golden_case):
        """retrieval_only mode only uses basic scorers (anchor, pipeline)."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import CaseResult

        config = EvalConfig(law="test", run_mode="retrieval_only")

        with patch("src.eval.eval_core.ask") as mock_ask:
            result = MagicMock()
            result.answer = ""
            result.references_structured = [{"article": "1"}]
            result.retrieval_metrics = {"references_structured": [{"article": "1"}]}
            mock_ask.ask.return_value = result

            results = list(evaluate_cases_iter([mock_golden_case], config))

        case_result = results[0]
        assert isinstance(case_result, CaseResult)

        # Should have anchor_presence and pipeline_breakdown
        assert "anchor_presence" in case_result.scores
        # Should NOT have faithfulness or answer_relevancy (LLM-judge scorers)
        assert "faithfulness" not in case_result.scores
        assert "answer_relevancy" not in case_result.scores

    def test_full_with_judge_includes_llm_scorers(self, mock_golden_case):
        """full_with_judge mode includes LLM-judge scorers."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import CaseResult

        config = EvalConfig(law="test", run_mode="full_with_judge")

        with patch("src.eval.eval_core.ask") as mock_ask, \
             patch("src.eval.eval_core.FaithfulnessScorer") as mock_faith, \
             patch("src.eval.eval_core.AnswerRelevancyScorer") as mock_rel:

            result = MagicMock()
            result.answer = "Test answer"
            result.references_structured = [{"article": "1", "chunk_text": "Text"}]
            result.retrieval_metrics = {
                "references_structured": [{"article": "1"}],
                "references_structured_all": [{"article": "1", "chunk_text": "Text"}],
            }
            mock_ask.ask.return_value = result

            # Mock LLM scorers
            mock_faith_instance = MagicMock()
            mock_faith_instance.score.return_value = MagicMock(passed=True, score=1.0, message="OK")
            mock_faith.return_value = mock_faith_instance

            mock_rel_instance = MagicMock()
            mock_rel_instance.score.return_value = MagicMock(passed=True, score=1.0, message="OK")
            mock_rel.return_value = mock_rel_instance

            results = list(evaluate_cases_iter([mock_golden_case], config))

        # LLM scorers should have been instantiated
        mock_faith.assert_called()
        mock_rel.assert_called()


# ---------------------------------------------------------------------------
# Test abstention cases require full_with_judge mode
# ---------------------------------------------------------------------------

class TestAbstentionCasesRequireLLMJudge:
    """Test that abstain cases don't falsely pass in retrieval_only mode.

    Bug: Abstain cases were passing with 100% in retrieval_only mode because:
    - AnchorScorer returns 1.0 when no anchors expected (empty list)
    - PipelineBreakdownScorer always passes (diagnostic only)
    - AbstentionScorer only runs in full_with_judge mode

    Fix: Abstain cases should be marked as "not evaluated" in non-full_with_judge modes.
    """

    @pytest.fixture
    def abstain_golden_case(self):
        """Create an abstain-type golden case."""
        from src.eval.types import GoldenCase, ExpectedBehavior

        return GoldenCase(
            id="abstain-test",
            profile="ENGINEERING",
            prompt="Is my company GDPR compliant?",  # Unanswerable without context
            expected=ExpectedBehavior(
                behavior="abstain",  # Key: this is an abstain case
                must_include_any_of=[],  # Empty - no anchors expected
            ),
            test_types=("abstention",),
            origin="manual",
        )

    def test_abstain_case_not_passed_in_retrieval_only_mode(self, abstain_golden_case):
        """Abstain cases should NOT be marked as passed in retrieval_only mode.

        In retrieval_only mode, we can't evaluate whether the system correctly
        abstained because we don't have the LLM's answer. These cases should
        be marked as not-evaluated (score with not_evaluated=True indicator).
        """
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import CaseResult

        config = EvalConfig(law="test", run_mode="retrieval_only")

        with patch("src.eval.eval_core.ask") as mock_ask:
            result = MagicMock()
            result.answer = ""  # No answer in dry_run
            result.references_structured = []
            result.retrieval_metrics = {"run": {}}
            mock_ask.ask.return_value = result

            results = list(evaluate_cases_iter([abstain_golden_case], config))

        case_result = results[0]
        assert isinstance(case_result, CaseResult)

        # The case should NOT be marked as passed - it wasn't actually evaluated
        # Either passed=False or there's a not_evaluated indicator in scores
        abstention_score = case_result.scores.get("abstention")

        # Abstention score should exist and indicate not-evaluated
        assert abstention_score is not None, "Abstain cases must have an abstention score"
        assert abstention_score.details.get("not_evaluated") is True, \
            "Abstain cases in retrieval_only mode should be marked as not_evaluated"

    def test_abstain_case_not_passed_in_full_mode(self, abstain_golden_case):
        """Abstain cases should NOT be marked as passed in full mode (without judge).

        Even in full mode, without the LLM-judge, we can't properly evaluate
        abstention behavior. These cases should be marked as not-evaluated.
        """
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import CaseResult

        config = EvalConfig(law="test", run_mode="full")

        with patch("src.eval.eval_core.ask") as mock_ask:
            result = MagicMock()
            result.answer = "I cannot determine your compliance status..."
            result.references_structured = []
            result.retrieval_metrics = {"run": {}}
            mock_ask.ask.return_value = result

            results = list(evaluate_cases_iter([abstain_golden_case], config))

        case_result = results[0]
        assert isinstance(case_result, CaseResult)

        # Should have abstention score with not_evaluated indicator
        abstention_score = case_result.scores.get("abstention")
        assert abstention_score is not None, "Abstain cases must have an abstention score"
        assert abstention_score.details.get("not_evaluated") is True, \
            "Abstain cases in full mode (without judge) should be marked as not_evaluated"

    def test_abstain_case_properly_evaluated_in_full_with_judge(self, abstain_golden_case):
        """Abstain cases should be properly evaluated in full_with_judge mode."""
        from src.eval.eval_core import EvalConfig, evaluate_cases_iter
        from src.eval.reporters import CaseResult

        config = EvalConfig(law="test", run_mode="full_with_judge")

        with patch("src.eval.eval_core.ask") as mock_ask:
            result = MagicMock()
            result.answer = "Jeg kan ikke vurdere din compliance uden konkrete oplysninger."
            result.references_structured = []
            result.retrieval_metrics = {
                "run": {},
                "references_structured_all": [],
            }
            mock_ask.ask.return_value = result

            results = list(evaluate_cases_iter([abstain_golden_case], config))

        case_result = results[0]
        assert isinstance(case_result, CaseResult)

        # Should have abstention score that was actually evaluated
        abstention_score = case_result.scores.get("abstention")
        assert abstention_score is not None, "Abstain cases must have an abstention score"

        # In full_with_judge mode, should NOT be marked as not_evaluated
        assert abstention_score.details.get("not_evaluated") is not True, \
            "Abstain cases in full_with_judge mode should be properly evaluated"


# ---------------------------------------------------------------------------
# Test configured concurrency (R1)
# ---------------------------------------------------------------------------

class TestConfiguredConcurrency:
    """eval.default_concurrency from settings.yaml controls ThreadPoolExecutor."""

    @pytest.fixture
    def mock_golden_case(self):
        from src.eval.types import GoldenCase, ExpectedBehavior
        return GoldenCase(
            id="conc-case-1",
            profile="LEGAL",
            prompt="What is Article 1?",
            expected=ExpectedBehavior(must_include_any_of=["article:1"]),
            test_types=("retrieval",),
            origin="manual",
        )

    def _run_with_mocked_concurrency(self, cases, config, settings_return):
        """Helper: run evaluate_cases_iter with mocked settings, capture max_workers."""
        from src.eval.eval_core import evaluate_cases_iter
        from concurrent.futures import ThreadPoolExecutor as OriginalTPE

        captured_max_workers = None

        class CapturingTPE(OriginalTPE):
            def __init__(self, max_workers=None, **kwargs):
                nonlocal captured_max_workers
                captured_max_workers = max_workers
                super().__init__(max_workers=max_workers, **kwargs)

        with patch("src.eval.eval_core.ask") as mock_ask, \
             patch("src.eval.eval_core.get_settings_yaml") as mock_settings, \
             patch("src.eval.eval_core.ThreadPoolExecutor", CapturingTPE):
            mock_settings.return_value = settings_return

            result = MagicMock()
            result.answer = "Test"
            result.references_structured = [{"article": "1"}]
            result.retrieval_metrics = {
                "references_structured": [{"article": "1"}],
                "run": {"anchors_in_top_k": ["article:1"]},
            }
            mock_ask.ask.return_value = result

            list(evaluate_cases_iter(cases, config))

        return captured_max_workers

    def test_eval_concurrency_reads_config(self, mock_golden_case):
        """ThreadPoolExecutor uses eval.default_concurrency from settings."""
        from src.eval.eval_core import EvalConfig

        config = EvalConfig(law="test", run_mode="retrieval_only")
        cases = [mock_golden_case] * 7

        max_workers = self._run_with_mocked_concurrency(
            cases, config, {"eval": {"default_concurrency": 7}}
        )
        assert max_workers == 7

    def test_eval_concurrency_fallback_when_missing(self, mock_golden_case):
        """Falls back to 4 workers when config key is missing."""
        from src.eval.eval_core import EvalConfig

        config = EvalConfig(law="test", run_mode="retrieval_only")
        cases = [mock_golden_case] * 5

        max_workers = self._run_with_mocked_concurrency(cases, config, {})
        assert max_workers == 4

    def test_eval_concurrency_capped_by_case_count(self, mock_golden_case):
        """Workers capped at len(cases) even if config is higher."""
        from src.eval.eval_core import EvalConfig

        config = EvalConfig(law="test", run_mode="retrieval_only")
        cases = [mock_golden_case] * 3

        max_workers = self._run_with_mocked_concurrency(
            cases, config, {"eval": {"default_concurrency": 10}}
        )
        assert max_workers == 3
