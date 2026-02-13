"""Tests for src/eval/reporters.py - Evaluation reporters."""

import io
import json
import tempfile
from pathlib import Path

import pytest

from src.eval.reporters import (
    CaseResult,
    EscalationStats,
    EvalSummary,
    FailureReporter,
    JsonReporter,
    PipelineStageStats,
    ProgressionTracker,
    ProgressReporter,
    RetryStats,
)
from src.eval.scorers import Score


class TestEscalationStats:
    """Tests for EscalationStats dataclass."""

    def test_escalation_success_rate_no_escalations(self):
        """Zero escalations returns 0.0 success rate."""
        stats = EscalationStats(cases_escalated=0)
        assert stats.escalation_success_rate == 0.0

    def test_escalation_success_rate_all_passed(self):
        """All escalated cases passed returns 1.0."""
        stats = EscalationStats(
            cases_escalated=5,
            cases_passed_on_escalation=5,
        )
        assert stats.escalation_success_rate == 1.0

    def test_escalation_success_rate_partial(self):
        """Partial success returns correct rate."""
        stats = EscalationStats(
            cases_escalated=10,
            cases_passed_on_escalation=7,
        )
        assert stats.escalation_success_rate == 0.7


class TestPipelineStageStats:
    """Tests for PipelineStageStats dataclass."""

    def test_retrieval_total(self):
        """Retrieval total sums passed and failed."""
        stats = PipelineStageStats(retrieval_passed=5, retrieval_failed=3)
        assert stats.retrieval_total == 8

    def test_augmentation_total(self):
        """Augmentation total sums passed and failed."""
        stats = PipelineStageStats(augmentation_passed=4, augmentation_failed=2)
        assert stats.augmentation_total == 6

    def test_generation_total(self):
        """Generation total sums passed and failed."""
        stats = PipelineStageStats(generation_passed=7, generation_failed=1)
        assert stats.generation_total == 8

    def test_retrieval_rate_empty(self):
        """Retrieval rate returns 0.0 when no cases."""
        stats = PipelineStageStats()
        assert stats.retrieval_rate == 0.0

    def test_retrieval_rate_computed(self):
        """Retrieval rate computed correctly."""
        stats = PipelineStageStats(retrieval_passed=8, retrieval_failed=2)
        assert stats.retrieval_rate == 0.8

    def test_augmentation_rate_computed(self):
        """Augmentation rate computed correctly."""
        stats = PipelineStageStats(augmentation_passed=6, augmentation_failed=4)
        assert stats.augmentation_rate == 0.6

    def test_generation_rate_computed(self):
        """Generation rate computed correctly."""
        stats = PipelineStageStats(generation_passed=9, generation_failed=1)
        assert stats.generation_rate == 0.9

    def test_from_results_empty(self):
        """from_results with empty list returns zero stats."""
        stats = PipelineStageStats.from_results([])
        assert stats.retrieval_total == 0
        assert stats.augmentation_total == 0
        assert stats.generation_total == 0

    def test_from_results_counts_anchor_presence(self):
        """from_results counts anchor_presence scores."""
        results = [
            CaseResult(
                case_id="case1",
                profile="LEGAL",
                passed=True,
                scores={"anchor_presence": Score(passed=True, score=1.0, message="ok")},
                duration_ms=100,
            ),
            CaseResult(
                case_id="case2",
                profile="LEGAL",
                passed=False,
                scores={"anchor_presence": Score(passed=False, score=0.0, message="fail")},
                duration_ms=100,
            ),
        ]
        stats = PipelineStageStats.from_results(results)
        assert stats.retrieval_passed == 1
        assert stats.retrieval_failed == 1

    def test_from_results_counts_contract_compliance(self):
        """from_results counts contract_compliance for augmentation."""
        results = [
            CaseResult(
                case_id="case1",
                profile="ENGINEERING",
                passed=True,
                scores={"contract_compliance": Score(passed=True, score=1.0, message="ok")},
                duration_ms=100,
            ),
        ]
        stats = PipelineStageStats.from_results(results)
        assert stats.augmentation_passed == 1
        assert stats.augmentation_failed == 0

    def test_from_results_counts_faithfulness_for_generation(self):
        """from_results counts faithfulness for generation stage."""
        results = [
            CaseResult(
                case_id="case1",
                profile="LEGAL",
                passed=True,
                scores={"faithfulness": Score(passed=True, score=0.9, message="ok")},
                duration_ms=100,
            ),
            CaseResult(
                case_id="case2",
                profile="LEGAL",
                passed=False,
                scores={"faithfulness": Score(passed=False, score=0.5, message="fail")},
                duration_ms=100,
            ),
        ]
        stats = PipelineStageStats.from_results(results)
        assert stats.generation_passed == 1
        assert stats.generation_failed == 1


class TestEvalSummary:
    """Tests for EvalSummary dataclass."""

    def test_passed_primary_no_escalations(self):
        """passed_primary counts non-escalated passes."""
        results = [
            CaseResult(case_id="case1", profile="LEGAL", passed=True, scores={}, duration_ms=100, escalated=False),
            CaseResult(case_id="case2", profile="LEGAL", passed=True, scores={}, duration_ms=100, escalated=False),
        ]
        summary = EvalSummary(
            law="ai-act",
            total=2,
            passed=2,
            failed=0,
            skipped=0,
            duration_seconds=1.0,
            results=results,
        )
        assert summary.passed_primary == 2

    def test_passed_escalated_counts_escalated_passes(self):
        """passed_escalated counts escalated passes only."""
        results = [
            CaseResult(case_id="case1", profile="LEGAL", passed=True, scores={}, duration_ms=100, escalated=False),
            CaseResult(case_id="case2", profile="LEGAL", passed=True, scores={}, duration_ms=100, escalated=True),
        ]
        summary = EvalSummary(
            law="ai-act",
            total=2,
            passed=2,
            failed=0,
            skipped=0,
            duration_seconds=1.0,
            results=results,
        )
        assert summary.passed_escalated == 1
        assert summary.passed_primary == 1

    def test_post_init_computes_stage_stats(self):
        """__post_init__ computes stage_stats from results."""
        results = [
            CaseResult(
                case_id="case1",
                profile="LEGAL",
                passed=True,
                scores={"anchor_presence": Score(passed=True, score=1.0, message="ok")},
                duration_ms=100,
            ),
        ]
        summary = EvalSummary(
            law="ai-act",
            total=1,
            passed=1,
            failed=0,
            skipped=0,
            duration_seconds=1.0,
            results=results,
        )
        assert summary.stage_stats.retrieval_passed == 1


class TestProgressReporter:
    """Tests for ProgressReporter class."""

    def test_start_writes_header(self):
        """start() writes evaluation header."""
        output = io.StringIO()
        reporter = ProgressReporter(total=10, show_progress=True, output=output)
        reporter.start("ai-act")

        result = output.getvalue()
        assert "Evaluating ai-act" in result
        assert "10 cases" in result

    def test_start_silent_when_disabled(self):
        """start() is silent when show_progress=False."""
        output = io.StringIO()
        reporter = ProgressReporter(total=10, show_progress=False, output=output)
        reporter.start("ai-act")

        assert output.getvalue() == ""

    def test_update_increments_current(self):
        """update() increments current counter."""
        output = io.StringIO()
        reporter = ProgressReporter(total=10, show_progress=True, output=output)

        reporter.update("case1", passed=True)
        assert reporter.current == 1

        reporter.update("case2", passed=False)
        assert reporter.current == 2

    def test_update_shows_pass_icon(self):
        """update() shows checkmark for passed cases."""
        output = io.StringIO()
        reporter = ProgressReporter(total=2, show_progress=True, output=output)
        reporter.update("case1", passed=True)

        result = output.getvalue()
        assert "✓" in result
        assert "case1" in result

    def test_update_shows_fail_icon(self):
        """update() shows X for failed cases."""
        output = io.StringIO()
        reporter = ProgressReporter(total=2, show_progress=True, output=output)
        reporter.update("case1", passed=False)

        result = output.getvalue()
        assert "✗" in result

    def test_update_shows_message_on_failure(self):
        """update() shows message on failure."""
        output = io.StringIO()
        reporter = ProgressReporter(total=2, show_progress=True, output=output)
        reporter.update("case1", passed=False, message="missing anchors")

        result = output.getvalue()
        assert "missing anchors" in result

    def test_finish_shows_summary(self):
        """finish() shows evaluation summary."""
        output = io.StringIO()
        reporter = ProgressReporter(total=10, show_progress=True, output=output)

        summary = EvalSummary(
            law="ai-act",
            total=10,
            passed=8,
            failed=2,
            skipped=0,
            duration_seconds=5.0,
            results=[],
        )
        reporter.finish(summary)

        result = output.getvalue()
        assert "Results for ai-act" in result
        assert "Passed: 8/10" in result
        assert "Failed: 2/10" in result


class TestFailureReporter:
    """Tests for FailureReporter class."""

    def test_report_failure_writes_case_info(self):
        """report_failure() writes case ID and profile."""
        output = io.StringIO()
        reporter = FailureReporter(output=output)

        result = CaseResult(
            case_id="test_case",
            profile="LEGAL",
            passed=False,
            scores={"anchor_presence": Score(passed=False, score=0.5, message="missing article 6")},
            duration_ms=100,
        )
        reporter.report_failure(result)

        output_text = output.getvalue()
        assert "FAILED: test_case" in output_text
        assert "LEGAL" in output_text
        assert "missing article 6" in output_text

    def test_report_summary_all_passed(self):
        """report_summary() shows success message when no failures."""
        output = io.StringIO()
        reporter = FailureReporter(output=output)
        reporter.report_summary([])

        assert "All cases passed" in output.getvalue()

    def test_report_summary_shows_failures(self):
        """report_summary() lists failed cases."""
        output = io.StringIO()
        reporter = FailureReporter(output=output)

        failures = [
            CaseResult(
                case_id="case1",
                profile="LEGAL",
                passed=False,
                scores={"anchor_presence": Score(passed=False, score=0.0, message="missing article")},
                duration_ms=100,
            ),
            CaseResult(
                case_id="case2",
                profile="ENGINEERING",
                passed=False,
                scores={"error": Score(passed=False, score=0.0, message="connection timeout")},
                duration_ms=100,
            ),
        ]
        reporter.report_summary(failures)

        output_text = output.getvalue()
        assert "2 FAILED CASES" in output_text
        assert "case1" in output_text
        assert "case2" in output_text
        assert "ERROR" in output_text


class TestJsonReporter:
    """Tests for JsonReporter class."""

    def test_write_creates_file(self):
        """write() creates JSON file at output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            reporter = JsonReporter(output_path)

            summary = EvalSummary(
                law="ai-act",
                total=5,
                passed=4,
                failed=1,
                skipped=0,
                duration_seconds=10.0,
                results=[],
            )
            reporter.write(summary)

            assert output_path.exists()

    def test_write_includes_meta(self):
        """write() includes meta section with law and timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            reporter = JsonReporter(output_path)

            summary = EvalSummary(
                law="ai-act",
                total=5,
                passed=4,
                failed=1,
                skipped=0,
                duration_seconds=10.0,
                results=[],
                timestamp="2026-02-02T10:00:00Z",
            )
            reporter.write(summary)

            with open(output_path, "r") as f:
                data = json.load(f)

            assert data["meta"]["law"] == "ai-act"
            assert data["meta"]["timestamp"] == "2026-02-02T10:00:00Z"

    def test_write_includes_summary_stats(self):
        """write() includes summary section with counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            reporter = JsonReporter(output_path)

            summary = EvalSummary(
                law="ai-act",
                total=10,
                passed=8,
                failed=2,
                skipped=1,
                duration_seconds=10.0,
                results=[],
            )
            reporter.write(summary)

            with open(output_path, "r") as f:
                data = json.load(f)

            assert data["summary"]["total"] == 10
            assert data["summary"]["passed"] == 8
            assert data["summary"]["failed"] == 2
            assert data["summary"]["pass_rate"] == 0.8

    def test_write_serializes_results(self):
        """write() serializes case results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            reporter = JsonReporter(output_path)

            results = [
                CaseResult(
                    case_id="case1",
                    profile="LEGAL",
                    passed=True,
                    scores={"anchor_presence": Score(passed=True, score=1.0, message="ok")},
                    duration_ms=100,
                ),
            ]
            summary = EvalSummary(
                law="ai-act",
                total=1,
                passed=1,
                failed=0,
                skipped=0,
                duration_seconds=1.0,
                results=results,
            )
            reporter.write(summary)

            with open(output_path, "r") as f:
                data = json.load(f)

            assert len(data["results"]) == 1
            assert data["results"][0]["case_id"] == "case1"
            assert data["results"][0]["passed"] is True

    def test_serialize_result_includes_escalation_info(self):
        """_serialize_result() includes escalation fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            reporter = JsonReporter(output_path)

            result = CaseResult(
                case_id="case1",
                profile="LEGAL",
                passed=True,
                scores={},
                duration_ms=100,
                escalated=True,
                escalation_model="gpt-4o",
            )

            serialized = reporter._serialize_result(result)
            assert serialized["escalated"] is True
            assert serialized["escalation_model"] == "gpt-4o"


class TestProgressionTracker:
    """Tests for ProgressionTracker class."""

    def test_record_creates_file(self):
        """record() creates tracking file if not exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "subdir" / "progression.json"
            tracker = ProgressionTracker(tracking_file)

            summary = EvalSummary(
                law="ai-act",
                total=10,
                passed=8,
                failed=2,
                skipped=0,
                duration_seconds=5.0,
                results=[],
            )
            tracker.record(summary)

            assert tracking_file.exists()

    def test_record_appends_entries(self):
        """record() appends to existing entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "progression.json"
            tracker = ProgressionTracker(tracking_file)

            summary1 = EvalSummary(
                law="ai-act",
                total=10,
                passed=8,
                failed=2,
                skipped=0,
                duration_seconds=5.0,
                results=[],
                timestamp="2026-02-01T10:00:00Z",
            )
            summary2 = EvalSummary(
                law="ai-act",
                total=10,
                passed=9,
                failed=1,
                skipped=0,
                duration_seconds=4.0,
                results=[],
                timestamp="2026-02-02T10:00:00Z",
            )

            tracker.record(summary1)
            tracker.record(summary2)

            with open(tracking_file, "r") as f:
                entries = json.load(f)

            assert len(entries) == 2
            assert entries[0]["passed"] == 8
            assert entries[1]["passed"] == 9

    def test_get_recent_history_empty(self):
        """get_recent_history() returns empty list when no file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "nonexistent.json"
            tracker = ProgressionTracker(tracking_file)

            history = tracker.get_recent_history()
            assert history == []

    def test_get_recent_history_returns_newest_first(self):
        """get_recent_history() returns entries newest first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "progression.json"
            tracker = ProgressionTracker(tracking_file)

            # Record two summaries
            for i in range(3):
                summary = EvalSummary(
                    law="ai-act",
                    total=10,
                    passed=i,
                    failed=10-i,
                    skipped=0,
                    duration_seconds=1.0,
                    results=[],
                    timestamp=f"2026-02-0{i+1}T10:00:00Z",
                )
                tracker.record(summary)

            history = tracker.get_recent_history()
            # Most recent (passed=2) should be first
            assert history[0]["passed"] == 2
            assert history[-1]["passed"] == 0

    def test_get_recent_history_filters_by_law(self):
        """get_recent_history() filters by law when specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "progression.json"
            tracker = ProgressionTracker(tracking_file)

            for law in ["ai-act", "gdpr", "ai-act"]:
                summary = EvalSummary(
                    law=law,
                    total=10,
                    passed=8,
                    failed=2,
                    skipped=0,
                    duration_seconds=1.0,
                    results=[],
                )
                tracker.record(summary)

            history = tracker.get_recent_history(law="ai-act")
            assert len(history) == 2
            assert all(e["law"] == "ai-act" for e in history)

    def test_get_recent_history_respects_limit(self):
        """get_recent_history() respects limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "progression.json"
            tracker = ProgressionTracker(tracking_file)

            for i in range(10):
                summary = EvalSummary(
                    law="ai-act",
                    total=10,
                    passed=i,
                    failed=10-i,
                    skipped=0,
                    duration_seconds=1.0,
                    results=[],
                )
                tracker.record(summary)

            history = tracker.get_recent_history(limit=5)
            assert len(history) == 5

    def test_print_history_no_entries(self):
        """print_history() shows message when no history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "nonexistent.json"
            tracker = ProgressionTracker(tracking_file)

            output = io.StringIO()
            tracker.print_history(output=output)

            assert "No progression history found" in output.getvalue()

    def test_print_history_shows_entries(self):
        """print_history() shows formatted entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "progression.json"
            tracker = ProgressionTracker(tracking_file)

            summary = EvalSummary(
                law="ai-act",
                total=10,
                passed=8,
                failed=2,
                skipped=0,
                duration_seconds=5.0,
                results=[],
                run_mode="full_with_judge",
            )
            tracker.record(summary)

            output = io.StringIO()
            tracker.print_history(output=output)

            result = output.getvalue()
            assert "Progression History" in result
            assert "ai-act" in result
            assert "full_with_judge" in result

    def test_record_handles_corrupt_file(self):
        """record() handles corrupt JSON file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "progression.json"

            # Write corrupt JSON
            with open(tracking_file, "w") as f:
                f.write("{invalid json")

            tracker = ProgressionTracker(tracking_file)
            summary = EvalSummary(
                law="ai-act",
                total=10,
                passed=8,
                failed=2,
                skipped=0,
                duration_seconds=5.0,
                results=[],
            )

            # Should not raise, should start fresh
            tracker.record(summary)

            with open(tracking_file, "r") as f:
                entries = json.load(f)
            assert len(entries) == 1


class TestRetryStats:
    """Tests for RetryStats dataclass."""

    def test_default_values(self):
        """RetryStats has sensible defaults."""
        stats = RetryStats()
        assert stats.cases_with_retries == 0
        assert stats.total_retries == 0
        assert stats.cases_passed_on_retry == 0
        assert stats.cases_failed_after_retries == 0


class TestCaseResult:
    """Tests for CaseResult dataclass."""

    def test_default_values(self):
        """CaseResult has sensible defaults."""
        result = CaseResult(
            case_id="test",
            profile="LEGAL",
            passed=True,
            scores={},
            duration_ms=100,
        )
        assert result.retry_count == 0
        assert result.escalated is False
        assert result.escalation_model is None
        assert result.retrieval_metrics == {}
