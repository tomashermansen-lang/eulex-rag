"""Tests for src/eval/baseline.py — regression baseline capture and comparison."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from src.eval.reporters import CaseResult, EvalSummary
from src.eval.scorers import Score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score(passed: bool, score_val: float = 1.0) -> Score:
    return Score(passed=passed, score=score_val, message="", details={})


def _make_case(
    case_id: str,
    anchor_pass: bool = True,
    faith_pass: bool = True,
    relevancy_pass: bool = True,
    is_abstention: bool = False,
) -> CaseResult:
    """Build a CaseResult with configurable scorer outcomes."""
    scores: dict[str, Score] = {
        "anchor_presence": _make_score(anchor_pass),
    }
    if is_abstention:
        scores["abstention"] = _make_score(True)
        # Abstention cases do NOT get faithfulness/relevancy scorers
    else:
        scores["faithfulness"] = _make_score(faith_pass)
        scores["answer_relevancy"] = _make_score(relevancy_pass)
    return CaseResult(
        case_id=case_id,
        profile="LEGAL",
        passed=anchor_pass and (is_abstention or (faith_pass and relevancy_pass)),
        scores=scores,
        duration_ms=100.0,
    )


def _make_summary(
    law: str,
    cases: list[CaseResult],
) -> EvalSummary:
    passed = sum(1 for c in cases if c.passed)
    return EvalSummary(
        law=law,
        total=len(cases),
        passed=passed,
        failed=len(cases) - passed,
        skipped=0,
        duration_seconds=1.0,
        results=cases,
        run_mode="full_with_judge",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def baseline_metrics() -> dict[str, float]:
    return {
        "retrieval_precision": 0.85,
        "faithfulness": 0.90,
        "relevancy": 0.88,
    }


@pytest.fixture()
def baseline_file(tmp_path: Path, baseline_metrics: dict[str, float]) -> Path:
    """Write a valid baseline JSON and return its path."""
    path = tmp_path / "baseline.json"
    data = {
        "timestamp": "2026-03-24T12:00:00+00:00",
        "eval_config": {
            "run_mode": "full_with_judge",
            "corpora": ["gdpr", "ai-act"],
            "case_count": 10,
        },
        "metrics": baseline_metrics,
        "regression_threshold_pp": 2,
        "per_corpus": {
            "gdpr": {
                "retrieval_precision": 0.80,
                "faithfulness": 0.90,
                "relevancy": 0.85,
            },
            "ai-act": {
                "retrieval_precision": 0.90,
                "faithfulness": 0.90,
                "relevancy": 0.90,
            },
        },
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


DEFAULT_EVAL_SETTINGS: dict = {
    "regression_threshold_pp": 2,
    "baseline_path": "data/evals/runs/baseline_pre_case_law.json",
    "scorer_metric_mapping": {
        "anchor_presence": "retrieval_precision",
        "faithfulness": "faithfulness",
        "answer_relevancy": "relevancy",
    },
}


@pytest.fixture()
def mock_eval_settings():
    with patch(
        "src.eval.baseline.get_eval_settings", return_value=dict(DEFAULT_EVAL_SETTINGS)
    ) as m:
        yield m


# ---------------------------------------------------------------------------
# Component 1: capture_baseline
# ---------------------------------------------------------------------------


class TestCaptureBaseline:
    """Tests #1-#8 from TESTPLAN."""

    def test_capture_baseline_writes_valid_schema(self, tmp_path, mock_eval_settings):
        """#1 — AS-1, AS-4, REQ-2: File has all required keys with correct types."""
        from src.eval.baseline import capture_baseline

        cases = [
            _make_case("c1", anchor_pass=True, faith_pass=True, relevancy_pass=True),
            _make_case("c2", anchor_pass=True, faith_pass=False, relevancy_pass=True),
        ]
        summary_gdpr = _make_summary("gdpr", cases)
        summary_ai = _make_summary("ai-act", [_make_case("c3")])

        out = tmp_path / "baseline.json"
        with (
            patch(
                "src.eval.baseline._discover_corpora", return_value=["ai-act", "gdpr"]
            ),
            patch(
                "src.eval.baseline._run_corpus_eval",
                side_effect=[summary_ai, summary_gdpr],
            ),
        ):
            capture_baseline(output_path=out)

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))

        # Required top-level keys
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["eval_config"], dict)
        assert "run_mode" in data["eval_config"]
        assert "corpora" in data["eval_config"]
        assert "case_count" in data["eval_config"]
        assert isinstance(data["metrics"], dict)
        for key in ("retrieval_precision", "faithfulness", "relevancy"):
            assert key in data["metrics"]
            assert 0.0 <= data["metrics"][key] <= 1.0
        assert isinstance(data["regression_threshold_pp"], int)
        assert isinstance(data["per_corpus"], dict)

    def test_capture_baseline_returns_snapshot_dict(self, tmp_path, mock_eval_settings):
        """#2 — REQ-1: Return value matches file contents."""
        from src.eval.baseline import capture_baseline

        summary = _make_summary("gdpr", [_make_case("c1")])
        out = tmp_path / "baseline.json"

        with (
            patch("src.eval.baseline._discover_corpora", return_value=["gdpr"]),
            patch("src.eval.baseline._run_corpus_eval", return_value=summary),
        ):
            result = capture_baseline(output_path=out)

        file_data = json.loads(out.read_text(encoding="utf-8"))
        assert result["timestamp"] == file_data["timestamp"]
        assert result["metrics"] == file_data["metrics"]

    def test_capture_baseline_idempotent_overwrite(self, tmp_path, mock_eval_settings):
        """#3 — AS-6, REQ-7: Second capture overwrites (new timestamp)."""
        from src.eval.baseline import capture_baseline

        summary = _make_summary("gdpr", [_make_case("c1")])
        out = tmp_path / "baseline.json"

        with (
            patch("src.eval.baseline._discover_corpora", return_value=["gdpr"]),
            patch("src.eval.baseline._run_corpus_eval", return_value=summary),
        ):
            r1 = capture_baseline(output_path=out)
            r2 = capture_baseline(output_path=out)

        assert r1["timestamp"] != r2["timestamp"]

    def test_capture_baseline_no_golden_cases_raises(
        self, tmp_path, mock_eval_settings
    ):
        """#4 — Edge case 1: Empty corpus list raises ValueError."""
        from src.eval.baseline import capture_baseline

        with (
            patch("src.eval.baseline._discover_corpora", return_value=[]),
            pytest.raises(ValueError, match="No golden cases found"),
        ):
            capture_baseline(output_path=tmp_path / "b.json")

    def test_capture_baseline_partial_corpus_failure(
        self, tmp_path, mock_eval_settings, caplog
    ):
        """#5 — Edge case 2: Failed corpus excluded, not zero-scored."""
        from src.eval.baseline import capture_baseline

        good_summary = _make_summary("gdpr", [_make_case("c1", anchor_pass=True)])

        def side_effect(corpus_id):
            if corpus_id == "ai-act":
                raise RuntimeError("vector store missing")
            return good_summary

        out = tmp_path / "baseline.json"
        with (
            patch(
                "src.eval.baseline._discover_corpora", return_value=["ai-act", "gdpr"]
            ),
            patch("src.eval.baseline._run_corpus_eval", side_effect=side_effect),
            caplog.at_level(logging.WARNING),
        ):
            result = capture_baseline(output_path=out)

        # Only gdpr should be in results
        assert "ai-act" not in result["per_corpus"]
        assert "gdpr" in result["per_corpus"]
        assert "ai-act" in caplog.text

    def test_capture_baseline_negative_threshold_raises(self, tmp_path):
        """#6 — Edge case 8: Negative threshold rejected."""
        from src.eval.baseline import capture_baseline

        settings = dict(DEFAULT_EVAL_SETTINGS)
        settings["regression_threshold_pp"] = -1

        with (
            patch("src.eval.baseline.get_eval_settings", return_value=settings),
            pytest.raises(ValueError),
        ):
            capture_baseline(output_path=tmp_path / "b.json")

    def test_capture_baseline_float_threshold_raises_type_error(self, tmp_path):
        """Float threshold rejected with TypeError (not silently truncated)."""
        from src.eval.baseline import capture_baseline

        settings = dict(DEFAULT_EVAL_SETTINGS)
        settings["regression_threshold_pp"] = 2.5

        with (
            patch("src.eval.baseline.get_eval_settings", return_value=settings),
            pytest.raises(TypeError, match="integer"),
        ):
            capture_baseline(output_path=tmp_path / "b.json")

    def test_capture_baseline_uses_config_path(self, tmp_path, mock_eval_settings):
        """#7 — REQ-5: Default path from config when no arg given."""
        from src.eval.baseline import capture_baseline

        # Override baseline_path to point to tmp_path
        mock_eval_settings.return_value = {
            **DEFAULT_EVAL_SETTINGS,
            "baseline_path": str(tmp_path / "default_baseline.json"),
        }

        summary = _make_summary("gdpr", [_make_case("c1")])
        with (
            patch("src.eval.baseline._discover_corpora", return_value=["gdpr"]),
            patch("src.eval.baseline._run_corpus_eval", return_value=summary),
            patch("src.eval.baseline._get_project_root", return_value=tmp_path),
        ):
            capture_baseline()  # No output_path arg

        assert (tmp_path / "default_baseline.json").exists()

    def test_capture_baseline_embeds_threshold(self, tmp_path):
        """#8 — REQ-6: Threshold from config embedded in output."""
        from src.eval.baseline import capture_baseline

        settings = {**DEFAULT_EVAL_SETTINGS, "regression_threshold_pp": 5}
        summary = _make_summary("gdpr", [_make_case("c1")])
        out = tmp_path / "b.json"

        with (
            patch("src.eval.baseline.get_eval_settings", return_value=settings),
            patch("src.eval.baseline._discover_corpora", return_value=["gdpr"]),
            patch("src.eval.baseline._run_corpus_eval", return_value=summary),
        ):
            result = capture_baseline(output_path=out)

        assert result["regression_threshold_pp"] == 5


# ---------------------------------------------------------------------------
# Component 2: _extract_metrics
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    """Tests #9-#11 from TESTPLAN."""

    def test_metric_derivation_from_scorers(self):
        """#9 — REQ-3: Scorer names mapped to metric keys."""
        from src.eval.baseline import _extract_metrics

        # 2 cases: c1 all pass, c2 anchor fails
        cases = [
            _make_case("c1", anchor_pass=True, faith_pass=True, relevancy_pass=True),
            _make_case("c2", anchor_pass=False, faith_pass=True, relevancy_pass=False),
        ]
        summaries = {"gdpr": _make_summary("gdpr", cases)}
        mapping = DEFAULT_EVAL_SETTINGS["scorer_metric_mapping"]

        aggregate, per_corpus = _extract_metrics(summaries, mapping)

        # anchor: 1 pass / 2 total = 0.5
        assert aggregate["retrieval_precision"] == pytest.approx(0.5)
        # faithfulness: 2 pass / 2 total = 1.0
        assert aggregate["faithfulness"] == pytest.approx(1.0)
        # relevancy: 1 pass / 2 total = 0.5
        assert aggregate["relevancy"] == pytest.approx(0.5)

    def test_metric_derivation_skips_abstention_cases(self):
        """#10 — REQ-3: Abstention cases excluded from faithfulness/relevancy."""
        from src.eval.baseline import _extract_metrics

        cases = [
            _make_case("c1", anchor_pass=True, faith_pass=True, relevancy_pass=True),
            _make_case("c2", anchor_pass=True, is_abstention=True),
        ]
        summaries = {"gdpr": _make_summary("gdpr", cases)}
        mapping = DEFAULT_EVAL_SETTINGS["scorer_metric_mapping"]

        aggregate, _ = _extract_metrics(summaries, mapping)

        # anchor: 2/2 (both have it)
        assert aggregate["retrieval_precision"] == pytest.approx(1.0)
        # faithfulness: 1/1 (abstention excluded)
        assert aggregate["faithfulness"] == pytest.approx(1.0)
        # relevancy: 1/1 (abstention excluded)
        assert aggregate["relevancy"] == pytest.approx(1.0)

    def test_extract_metrics_per_corpus(self):
        """#11 — REQ-2: Per-corpus breakdown correct."""
        from src.eval.baseline import _extract_metrics

        gdpr_cases = [_make_case("g1", anchor_pass=True, faith_pass=False)]
        ai_cases = [_make_case("a1", anchor_pass=True, faith_pass=True)]
        summaries = {
            "gdpr": _make_summary("gdpr", gdpr_cases),
            "ai-act": _make_summary("ai-act", ai_cases),
        }
        mapping = DEFAULT_EVAL_SETTINGS["scorer_metric_mapping"]

        _, per_corpus = _extract_metrics(summaries, mapping)

        assert per_corpus["gdpr"]["faithfulness"] == pytest.approx(0.0)
        assert per_corpus["ai-act"]["faithfulness"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Component 3: compare_to_baseline
# ---------------------------------------------------------------------------


class TestCompareToBaseline:
    """Tests #12-#21 from TESTPLAN."""

    def test_compare_no_regression(self, baseline_file):
        """#12 — AS-2: Deltas within threshold."""
        from src.eval.baseline import compare_to_baseline

        current = {"retrieval_precision": 0.84, "faithfulness": 0.91, "relevancy": 0.87}
        result = compare_to_baseline(current, baseline_path=baseline_file)

        assert result["regression_detected"] is False
        assert result["deltas"]["retrieval_precision"] == pytest.approx(-0.01)
        assert result["deltas"]["faithfulness"] == pytest.approx(0.01)
        assert result["deltas"]["relevancy"] == pytest.approx(-0.01)

    def test_compare_regression_detected(self, baseline_file):
        """#13 — AS-3: 3pp drop exceeds 2pp threshold."""
        from src.eval.baseline import compare_to_baseline

        current = {"retrieval_precision": 0.82, "faithfulness": 0.90, "relevancy": 0.88}
        result = compare_to_baseline(current, baseline_path=baseline_file)

        assert result["regression_detected"] is True
        assert result["deltas"]["retrieval_precision"] == pytest.approx(-0.03)

    def test_compare_boundary_exactly_at_threshold(self, baseline_file):
        """#14 — Exactly 2pp drop is NOT regression (threshold exclusive)."""
        from src.eval.baseline import compare_to_baseline

        current = {"retrieval_precision": 0.83, "faithfulness": 0.90, "relevancy": 0.88}
        result = compare_to_baseline(current, baseline_path=baseline_file)

        assert result["regression_detected"] is False

    def test_compare_missing_baseline_file_raises(self, tmp_path):
        """#15 — AS-5: FileNotFoundError for missing file."""
        from src.eval.baseline import compare_to_baseline

        missing = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            compare_to_baseline({"retrieval_precision": 0.8}, baseline_path=missing)

    def test_compare_baseline_missing_metric(self, tmp_path):
        """#16 — Edge case 4: Missing metric in baseline skipped with warning."""
        from src.eval.baseline import compare_to_baseline

        data = {
            "timestamp": "2026-03-24T12:00:00+00:00",
            "eval_config": {
                "run_mode": "full_with_judge",
                "corpora": ["gdpr"],
                "case_count": 5,
            },
            "metrics": {"retrieval_precision": 0.85, "faithfulness": 0.90},
            "regression_threshold_pp": 2,
            "per_corpus": {},
        }
        path = tmp_path / "b.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        current = {"retrieval_precision": 0.84, "faithfulness": 0.89, "relevancy": 0.80}
        result = compare_to_baseline(current, baseline_path=path)

        # relevancy not in baseline — should be in deltas but NOT trigger regression
        assert "relevancy" in result["deltas"]
        assert any("relevancy" in w for w in result["warnings"])

    def test_compare_current_missing_metric(self, tmp_path, baseline_file):
        """#17 — Edge case 5: Missing metric in current treated as 0.0."""
        from src.eval.baseline import compare_to_baseline

        current = {"retrieval_precision": 0.85, "relevancy": 0.88}
        # faithfulness missing → treated as 0.0, delta = 0.0 - 0.90 = -0.90
        result = compare_to_baseline(current, baseline_path=baseline_file)

        assert result["deltas"]["faithfulness"] == pytest.approx(-0.90)
        assert result["regression_detected"] is True

    def test_compare_threshold_zero(self, tmp_path):
        """#18 — Edge case 6: Zero threshold, any negative triggers regression."""
        from src.eval.baseline import compare_to_baseline

        data = {
            "timestamp": "2026-03-24T12:00:00+00:00",
            "eval_config": {
                "run_mode": "full_with_judge",
                "corpora": ["gdpr"],
                "case_count": 5,
            },
            "metrics": {"retrieval_precision": 0.85},
            "regression_threshold_pp": 0,
            "per_corpus": {},
        }
        path = tmp_path / "b.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        current = {"retrieval_precision": 0.849}
        result = compare_to_baseline(current, baseline_path=path)

        assert result["regression_detected"] is True

    def test_compare_uses_baseline_threshold_not_config(self, tmp_path):
        """#19 — Edge case 9: Threshold from baseline file, not current config."""
        from src.eval.baseline import compare_to_baseline

        # Baseline has threshold 5 (lenient)
        data = {
            "timestamp": "2026-03-24T12:00:00+00:00",
            "eval_config": {
                "run_mode": "full_with_judge",
                "corpora": ["gdpr"],
                "case_count": 5,
            },
            "metrics": {"retrieval_precision": 0.85},
            "regression_threshold_pp": 5,
            "per_corpus": {},
        }
        path = tmp_path / "b.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        # 3pp drop: within 5pp threshold → no regression
        current = {"retrieval_precision": 0.82}
        result = compare_to_baseline(current, baseline_path=path)

        assert result["regression_detected"] is False
        assert result["threshold_pp"] == 5

    def test_compare_new_metric_excluded_from_regression(self, baseline_file):
        """#20 — New-metric policy: in current but not baseline → no regression."""
        from src.eval.baseline import compare_to_baseline

        current = {
            "retrieval_precision": 0.85,
            "faithfulness": 0.90,
            "relevancy": 0.88,
            "new_metric": 0.5,
        }
        result = compare_to_baseline(current, baseline_path=baseline_file)

        assert result["regression_detected"] is False
        assert "new_metric" in result["deltas"]
        assert any("new_metric" in w for w in result["warnings"])

    def test_compare_uses_config_path(
        self, tmp_path, baseline_file, mock_eval_settings
    ):
        """#21 — REQ-5: Default path from config when no arg."""
        from src.eval.baseline import compare_to_baseline

        mock_eval_settings.return_value = {
            **DEFAULT_EVAL_SETTINGS,
            "baseline_path": str(baseline_file),
        }

        with patch(
            "src.eval.baseline._get_project_root", return_value=baseline_file.parent
        ):
            result = compare_to_baseline(
                {"retrieval_precision": 0.85, "faithfulness": 0.90, "relevancy": 0.88}
            )

        assert result["regression_detected"] is False


# ---------------------------------------------------------------------------
# Component 4: Config + discover
# ---------------------------------------------------------------------------


class TestConfigAndDiscover:
    """Tests #22-#23 from TESTPLAN."""

    def test_config_default_threshold(self):
        """#22 — Edge case 7: Missing config defaults to 2."""
        from src.eval.baseline import _get_regression_threshold

        with patch("src.eval.baseline.get_eval_settings", return_value={}):
            assert _get_regression_threshold() == 2

    def test_discover_corpora_from_golden_files(self, tmp_path):
        """#23 — Glob finds golden case files, extracts corpus IDs sorted."""
        from src.eval.baseline import _discover_corpora

        (tmp_path / "golden_cases_gdpr.yaml").touch()
        (tmp_path / "golden_cases_ai-act.yaml").touch()
        (tmp_path / "cross_law_synthesis.yaml").touch()  # should be ignored

        with patch("src.eval.baseline._get_evals_dir", return_value=tmp_path):
            result = _discover_corpora()

        assert result == ["ai-act", "gdpr"]
