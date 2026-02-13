"""Tests for eval metrics aggregation pure functions (C2a, C2b).

Covers: compute_unified_pass_rate, compute_trend, compute_percentiles,
compute_histogram_bins, compute_health_status, aggregate_by_mode,
aggregate_by_difficulty, aggregate_scorers, compute_mode_latency,
compute_difficulty_latency.

All functions are pure — no I/O, no fixtures needed.
"""

import pytest

from ui_react.backend.routes.eval_metrics import (
    compute_unified_pass_rate,
    compute_trend,
    compute_percentiles,
    compute_histogram_bins,
    compute_health_status,
    aggregate_by_mode,
    aggregate_by_difficulty,
    aggregate_scorers,
    compute_mode_latency,
    compute_difficulty_latency,
)


# ─────────────────────────────────────────────────────────────────────────────
# T-B1 – T-B3: compute_unified_pass_rate
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeUnifiedPassRate:
    """Weighted average pass rate across single-law and cross-law runs."""

    def test_weighted_average(self):
        """T-B1: Weighted by case count, not simple average."""
        single_law = [{"total": 100, "passed": 95}]
        cross_law = [{"total": 20, "passed": 16}]
        result = compute_unified_pass_rate(single_law, cross_law)
        # (95 + 16) / (100 + 20) = 111/120 = 92.5
        assert result == pytest.approx(92.5)

    def test_single_category_only(self):
        """T-B2: Only single-law data, cross-law empty."""
        single_law = [{"total": 50, "passed": 45}]
        result = compute_unified_pass_rate(single_law, [])
        assert result == pytest.approx(90.0)

    def test_no_data(self):
        """T-B3: No runs at all → 0.0."""
        result = compute_unified_pass_rate([], [])
        assert result == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# T-B4 – T-B7: compute_trend
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeTrend:
    """Trend direction based on historical pass rates."""

    def test_improving(self):
        """T-B4: Latest above mean by more than threshold → improving."""
        history = [90.0, 92.0, 91.0, 93.0, 94.0]
        result = compute_trend(history, latest=96.0, window=5, threshold=2.0)
        assert result["direction"] == "improving"
        assert result["delta_pp"] > 0

    def test_stable(self):
        """T-B5: Latest within threshold of mean → stable."""
        history = [93.0, 94.0, 93.0, 94.0, 93.0]
        result = compute_trend(history, latest=94.0, window=5, threshold=2.0)
        assert result["direction"] == "stable"

    def test_declining(self):
        """T-B6: Latest below mean by more than threshold → declining."""
        history = [96.0, 95.0, 94.0, 93.0, 92.0]
        result = compute_trend(history, latest=88.0, window=5, threshold=2.0)
        assert result["direction"] == "declining"
        assert result["delta_pp"] < 0

    def test_insufficient_data(self):
        """T-B7: Only 1 run → insufficient_data."""
        result = compute_trend([], latest=95.0, window=5, threshold=2.0)
        assert result["direction"] == "insufficient_data"


# ─────────────────────────────────────────────────────────────────────────────
# T-B8 – T-B9: compute_percentiles
# ─────────────────────────────────────────────────────────────────────────────


class TestComputePercentiles:
    """Latency percentile computation."""

    def test_standard_case(self):
        """T-B8: 20 durations → P50, P95, P99 computed correctly."""
        durations_ms = [i * 100 for i in range(1, 21)]  # 100..2000
        result = compute_percentiles(durations_ms)
        assert "p50" in result
        assert "p95" in result
        assert "p99" in result
        # P50 of 100..2000 should be around 1000-1100ms → ~1.0-1.1s
        assert 0.9 <= result["p50"] <= 1.2
        # P95 should be high
        assert result["p95"] > result["p50"]
        assert result["p99"] >= result["p95"]

    def test_single_duration(self):
        """T-B9: Single duration → all percentiles equal."""
        result = compute_percentiles([500])
        assert result["p50"] == pytest.approx(0.5)
        assert result["p95"] == pytest.approx(0.5)
        assert result["p99"] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# T-B10: compute_histogram_bins
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeHistogramBins:
    """Duration distribution binning."""

    def test_distributes_correctly(self):
        """T-B10: Bins are contiguous and counts sum to total."""
        durations_ms = [i * 100 for i in range(1, 51)]  # 100..5000
        result = compute_histogram_bins(durations_ms, num_bins=10)
        assert len(result) == 10
        total_count = sum(b["count"] for b in result)
        assert total_count == 50
        # Bins must be contiguous
        for i in range(1, len(result)):
            assert result[i]["range_start"] == pytest.approx(
                result[i - 1]["range_end"]
            )


# ─────────────────────────────────────────────────────────────────────────────
# T-B11: compute_health_status
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeHealthStatus:
    """4-tier health status from pass rate."""

    @pytest.mark.parametrize(
        "rate, expected",
        [
            (96.0, "green"),
            (95.0, "green"),
            (85.0, "yellow"),
            (80.0, "yellow"),
            (65.0, "orange"),
            (60.0, "orange"),
            (50.0, "red"),
            (0.0, "red"),
        ],
    )
    def test_all_tiers(self, rate: float, expected: str):
        """T-B11: Each tier mapped correctly at boundaries."""
        thresholds = [95, 80, 60]
        assert compute_health_status(rate, thresholds) == expected


# ─────────────────────────────────────────────────────────────────────────────
# T-B12 – T-B13: aggregate_by_mode
# ─────────────────────────────────────────────────────────────────────────────


class TestAggregateByMode:
    """Group cases by synthesis_mode and compute pass rates."""

    def test_groups_correctly(self):
        """T-B12: Mixed modes → correct per-mode pass rates."""
        cases = [
            {"synthesis_mode": "comparison", "passed": True},
            {"synthesis_mode": "comparison", "passed": True},
            {"synthesis_mode": "comparison", "passed": True},
            {"synthesis_mode": "comparison", "passed": False},
            {"synthesis_mode": "comparison", "passed": True},
            {"synthesis_mode": "discovery", "passed": True},
            {"synthesis_mode": "discovery", "passed": False},
            {"synthesis_mode": "discovery", "passed": False},
            {"synthesis_mode": "aggregation", "passed": True},
            {"synthesis_mode": "aggregation", "passed": True},
        ]
        result = aggregate_by_mode(cases)
        by_mode = {r["mode"]: r for r in result}

        assert by_mode["comparison"]["pass_rate"] == pytest.approx(80.0)
        assert by_mode["comparison"]["total"] == 5
        assert by_mode["discovery"]["pass_rate"] == pytest.approx(100 / 3)
        assert by_mode["discovery"]["total"] == 3
        assert by_mode["aggregation"]["pass_rate"] == pytest.approx(100.0)
        assert by_mode["aggregation"]["total"] == 2

    def test_single_mode(self):
        """T-B13: All cases are same mode → single entry."""
        cases = [
            {"synthesis_mode": "comparison", "passed": True},
            {"synthesis_mode": "comparison", "passed": False},
        ]
        result = aggregate_by_mode(cases)
        assert len(result) == 1
        assert result[0]["mode"] == "comparison"


# ─────────────────────────────────────────────────────────────────────────────
# T-B14 – T-B15: aggregate_by_difficulty
# ─────────────────────────────────────────────────────────────────────────────


class TestAggregateByDifficulty:
    """Group cases by difficulty and compute pass rates."""

    def test_groups_correctly(self):
        """T-B14: Mixed difficulties → correct per-difficulty pass rates."""
        cases = [
            {"difficulty": "easy", "passed": True},
            {"difficulty": "easy", "passed": True},
            {"difficulty": "easy", "passed": True},
            {"difficulty": "medium", "passed": True},
            {"difficulty": "medium", "passed": True},
            {"difficulty": "medium", "passed": True},
            {"difficulty": "medium", "passed": False},
            {"difficulty": "hard", "passed": True},
            {"difficulty": "hard", "passed": False},
            {"difficulty": "hard", "passed": False},
        ]
        result = aggregate_by_difficulty(cases)
        by_diff = {r["difficulty"]: r for r in result}

        assert by_diff["easy"]["pass_rate"] == pytest.approx(100.0)
        assert by_diff["medium"]["pass_rate"] == pytest.approx(75.0)
        assert by_diff["hard"]["pass_rate"] == pytest.approx(100 / 3)

    def test_null_difficulty_excluded(self):
        """T-B15: Cases with null difficulty grouped as 'Uklassificeret'."""
        cases = [
            {"difficulty": "easy", "passed": True},
            {"difficulty": None, "passed": True},
            {"difficulty": None, "passed": False},
        ]
        result = aggregate_by_difficulty(cases)
        by_diff = {r["difficulty"]: r for r in result}

        assert "easy" in by_diff
        assert "Uklassificeret" in by_diff
        assert by_diff["Uklassificeret"]["total"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# T-B16 – T-B17: aggregate_scorers
# ─────────────────────────────────────────────────────────────────────────────


class TestAggregateScorers:
    """Scorer aggregation with mode-specific denominator logic."""

    def test_mode_specific_denominator(self):
        """T-B16: Mode-specific scorer uses only applicable cases."""
        cases = [
            {
                "synthesis_mode": "comparison",
                "passed": True,
                "scores": {"comparison_completeness": True, "corpus_coverage": True},
            },
            {
                "synthesis_mode": "comparison",
                "passed": True,
                "scores": {"comparison_completeness": True, "corpus_coverage": True},
            },
            {
                "synthesis_mode": "comparison",
                "passed": False,
                "scores": {"comparison_completeness": False, "corpus_coverage": True},
            },
            {
                "synthesis_mode": "discovery",
                "passed": True,
                "scores": {"corpus_coverage": True},
            },
            {
                "synthesis_mode": "discovery",
                "passed": False,
                "scores": {"corpus_coverage": False},
            },
        ]
        result = aggregate_scorers(cases)
        by_scorer = {r["scorer"]: r for r in result}

        # comparison_completeness: 2/3 = 66.7% (denom = 3 comparison cases)
        assert by_scorer["comparison_completeness"]["total"] == 3
        assert by_scorer["comparison_completeness"]["passed"] == 2

    def test_generic_scorer_uses_all_cases(self):
        """T-B17: Generic scorer denominator is total case count."""
        cases = [
            {
                "synthesis_mode": "comparison",
                "passed": True,
                "scores": {"corpus_coverage": True},
            },
            {
                "synthesis_mode": "comparison",
                "passed": True,
                "scores": {"corpus_coverage": True},
            },
            {
                "synthesis_mode": "comparison",
                "passed": False,
                "scores": {"corpus_coverage": False},
            },
            {
                "synthesis_mode": "discovery",
                "passed": True,
                "scores": {"corpus_coverage": True},
            },
            {
                "synthesis_mode": "discovery",
                "passed": False,
                "scores": {"corpus_coverage": False},
            },
        ]
        result = aggregate_scorers(cases)
        by_scorer = {r["scorer"]: r for r in result}

        # corpus_coverage: 3/5 = 60% (denom = all 5 cases)
        assert by_scorer["corpus_coverage"]["total"] == 5
        assert by_scorer["corpus_coverage"]["passed"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# T-B18 – T-B19: compute_mode_latency, compute_difficulty_latency
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeModeLatency:
    """P50 latency per synthesis mode."""

    def test_p50_per_mode(self):
        """T-B18: Correct P50 per mode."""
        cases = [
            {"synthesis_mode": "comparison", "duration_ms": 4000},
            {"synthesis_mode": "comparison", "duration_ms": 5000},
            {"synthesis_mode": "comparison", "duration_ms": 6000},
            {"synthesis_mode": "comparison", "duration_ms": 5000},
            {"synthesis_mode": "comparison", "duration_ms": 5000},
            {"synthesis_mode": "discovery", "duration_ms": 10000},
            {"synthesis_mode": "discovery", "duration_ms": 12000},
            {"synthesis_mode": "discovery", "duration_ms": 14000},
        ]
        result = compute_mode_latency(cases)
        by_mode = {r["mode"]: r for r in result}

        assert by_mode["comparison"]["p50_seconds"] == pytest.approx(5.0)
        assert by_mode["comparison"]["case_count"] == 5
        assert by_mode["discovery"]["p50_seconds"] == pytest.approx(12.0)
        assert by_mode["discovery"]["case_count"] == 3


class TestComputeDifficultyLatency:
    """P50 latency per difficulty level."""

    def test_p50_per_difficulty(self):
        """T-B19: Correct P50 per difficulty."""
        cases = [
            {"difficulty": "easy", "duration_ms": 800},
            {"difficulty": "easy", "duration_ms": 1000},
            {"difficulty": "easy", "duration_ms": 1200},
            {"difficulty": "hard", "duration_ms": 10000},
            {"difficulty": "hard", "duration_ms": 12000},
        ]
        result = compute_difficulty_latency(cases)
        by_diff = {r["difficulty"]: r for r in result}

        assert by_diff["easy"]["p50_seconds"] == pytest.approx(1.0)
        assert by_diff["hard"]["p50_seconds"] == pytest.approx(11.0)
