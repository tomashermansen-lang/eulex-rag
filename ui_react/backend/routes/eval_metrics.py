"""API routes for eval metrics dashboard.

Single Responsibility: Aggregate evaluation and ingestion data
into metrics-specific response shapes.

Endpoints (to be added in later steps):
- GET /overview — Trust overview (Level 1)
- GET /quality — Eval quality breakdown (Level 2)
- GET /performance — Processing performance (Level 2)
- GET /ingestion — Ingestion health (Level 2)
- GET /detail/law/{law} — Per-law drill-down (Level 3)
- GET /detail/suite/{suite_id} — Per-suite drill-down (Level 3)
- GET /detail/scorer/{scorer} — Per-scorer drill-down (Level 3)
- GET /detail/mode/{mode} — Per-mode drill-down (Level 3)
- GET /detail/difficulty/{difficulty} — Per-difficulty drill-down (Level 3)
- POST /analyse — AI analysis (SSE stream)
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

# Add backend directory and project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas import (
    MetricsOverviewResponse,
    TrendInfo,
    MetricsSummary,
    CategorySummary,
    MetricsQualityResponse,
    LawPassRate,
    SuitePassRate,
    ModePassRate,
    DifficultyPassRate,
    ScorerPassRate,
    TrendPoint,
    StageBreakdown,
    MetricsPerformanceResponse,
    Percentiles,
    ModeLatency,
    DifficultyLatency,
    HistogramBin,
    RunModeLatency,
    RunModeTrend,
    LatencyTrendPoint,
    RateTrendPoint,
    SLPerformance,
    CLPerformance,
    MetricsIngestionResponse,
    CorpusHealth,
    CaseResultSummary,
    CrossLawCaseResultSummary,
    AppliedFilters,
    MetricsLawDetailResponse,
    MetricsSuiteDetailResponse,
    MetricsScorerDetailResponse,
    MetricsModeDetailResponse,
    MetricsDifficultyDetailResponse,
)
from src.common.config_loader import get_settings_yaml
from src.engine.metrics_analysis import analyse_metrics_stream

logger = logging.getLogger(__name__)

router = APIRouter(tags=["eval-metrics"])


# ─────────────────────────────────────────────────────────────────────────────
# Pure aggregation functions (C2a)
# ─────────────────────────────────────────────────────────────────────────────


def compute_unified_pass_rate(
    single_law_runs: list[dict[str, Any]],
    cross_law_runs: list[dict[str, Any]],
) -> float:
    """Weighted average pass rate across single-law and cross-law runs.

    Args:
        single_law_runs: List of run dicts with 'total' and 'passed' keys.
        cross_law_runs: List of run dicts with 'total' and 'passed' keys.

    Returns:
        Weighted pass rate as percentage (0.0–100.0), or 0.0 if no data.
    """
    total = sum(r["total"] for r in single_law_runs + cross_law_runs)
    if total == 0:
        return 0.0
    passed = sum(r["passed"] for r in single_law_runs + cross_law_runs)
    return (passed / total) * 100


def compute_trend(
    history: list[float],
    latest: float,
    window: int,
    threshold: float,
) -> dict[str, Any]:
    """Compute trend direction from historical pass rates.

    Args:
        history: Historical pass rates (oldest first), NOT including latest.
        latest: Most recent pass rate.
        window: Number of historical values to consider.
        threshold: Percentage-point delta required for improving/declining.

    Returns:
        Dict with 'direction', 'delta_pp', 'window', 'history' keys.
    """
    if not history:
        return {
            "direction": "insufficient_data",
            "delta_pp": None,
            "window": 0,
            "history": [latest],
        }

    windowed = history[-window:]
    mean = statistics.mean(windowed)
    delta = latest - mean

    if delta > threshold:
        direction = "improving"
    elif delta < -threshold:
        direction = "declining"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "delta_pp": round(delta, 1),
        "window": len(windowed),
        "history": windowed + [latest],
    }


def compute_percentiles(durations_ms: list[float]) -> dict[str, float]:
    """Compute P50, P95, P99 from durations in milliseconds.

    Args:
        durations_ms: List of durations in milliseconds.

    Returns:
        Dict with 'p50', 'p95', 'p99' in seconds.  Returns all zeros if empty.
    """
    if not durations_ms:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    sorted_d = sorted(durations_ms)

    def _percentile(data: list[float], pct: float) -> float:
        idx = (pct / 100) * (len(data) - 1)
        lower = int(idx)
        upper = min(lower + 1, len(data) - 1)
        frac = idx - lower
        return data[lower] + frac * (data[upper] - data[lower])

    return {
        "p50": round(_percentile(sorted_d, 50) / 1000, 3),
        "p95": round(_percentile(sorted_d, 95) / 1000, 3),
        "p99": round(_percentile(sorted_d, 99) / 1000, 3),
    }


def compute_histogram_bins(
    durations_ms: list[float],
    num_bins: int = 10,
) -> list[dict[str, Any]]:
    """Distribute durations into equal-width bins.

    Args:
        durations_ms: List of durations in milliseconds.
        num_bins: Number of bins to create.

    Returns:
        List of dicts with 'range_start', 'range_end', 'count' (in seconds).
    """
    if not durations_ms:
        return []

    seconds = [d / 1000 for d in durations_ms]
    min_val = min(seconds)
    max_val = max(seconds)

    # Edge case: all durations identical — return single bin
    if max_val == min_val:
        return [{"range_start": round(min_val, 3), "range_end": round(max_val, 3), "count": len(seconds)}]

    bin_width = (max_val - min_val) / num_bins

    bins: list[dict[str, Any]] = []
    for i in range(num_bins):
        start = min_val + i * bin_width
        end = min_val + (i + 1) * bin_width
        count = sum(1 for s in seconds if start <= s < end or (i == num_bins - 1 and s == end))
        bins.append({
            "range_start": round(start, 3),
            "range_end": round(end, 3),
            "count": count,
        })

    return bins


def compute_health_status(rate: float, thresholds: list[int]) -> str:
    """Map pass rate to 4-tier health status.

    Args:
        rate: Pass rate percentage (0.0–100.0).
        thresholds: Three descending thresholds [green, yellow, orange].

    Returns:
        One of 'green', 'yellow', 'orange', 'red'.
    """
    if rate >= thresholds[0]:
        return "green"
    elif rate >= thresholds[1]:
        return "yellow"
    elif rate >= thresholds[2]:
        return "orange"
    else:
        return "red"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-law aggregation functions (C2b)
# ─────────────────────────────────────────────────────────────────────────────

# Mode-specific scorers: only counted against cases with the applicable mode
MODE_SPECIFIC_SCORERS: dict[str, set[str]] = {
    "comparison_completeness": {"comparison"},
    "routing_precision": {"routing"},
}


def aggregate_by_mode(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cases by synthesis_mode and compute pass rates.

    Args:
        cases: List of case dicts with 'synthesis_mode' and 'passed' keys.

    Returns:
        List of dicts with 'mode', 'pass_rate', 'total', 'passed'.
    """
    groups: dict[str, list[bool]] = defaultdict(list)
    for case in cases:
        mode = case.get("synthesis_mode")
        if mode:
            groups[mode].append(bool(case.get("passed", False)))

    return [
        {
            "mode": mode,
            "pass_rate": (sum(passed_list) / len(passed_list)) * 100 if passed_list else 0.0,
            "total": len(passed_list),
            "passed": sum(passed_list),
        }
        for mode, passed_list in sorted(groups.items())
    ]


def aggregate_by_difficulty(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cases by difficulty and compute pass rates.

    Cases with null/missing difficulty are grouped as 'Uklassificeret'.

    Args:
        cases: List of case dicts with 'difficulty' and 'passed' keys.

    Returns:
        List of dicts with 'difficulty', 'pass_rate', 'total', 'passed'.
    """
    groups: dict[str, list[bool]] = defaultdict(list)
    for case in cases:
        difficulty = case.get("difficulty") or "Uklassificeret"
        groups[difficulty].append(bool(case.get("passed", False)))

    return [
        {
            "difficulty": diff,
            "pass_rate": (sum(passed_list) / len(passed_list)) * 100 if passed_list else 0.0,
            "total": len(passed_list),
            "passed": sum(passed_list),
        }
        for diff, passed_list in sorted(groups.items())
    ]


def aggregate_scorers(
    cases: list[dict[str, Any]],
    mode_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate scorer pass rates with mode-specific denominator logic.

    Mode-specific scorers (e.g. comparison_completeness) only count cases
    where the scorer is applicable (i.e. the case's synthesis_mode matches).

    Args:
        cases: List of case dicts with 'synthesis_mode', 'scores' keys.
        mode_filter: Optional mode to filter applicable scorers.

    Returns:
        List of dicts with 'scorer', 'pass_rate', 'total', 'passed'.
    """
    scorer_results: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})

    for case in cases:
        scores = case.get("scores", {})
        case_mode = case.get("synthesis_mode")

        # If mode_filter is set, skip cases that don't match
        if mode_filter is not None and case_mode != mode_filter:
            continue

        for scorer, score_passed in scores.items():
            # Check if this scorer is mode-specific
            applicable_modes = MODE_SPECIFIC_SCORERS.get(scorer)
            if applicable_modes is not None and case_mode not in applicable_modes:
                continue  # Skip — scorer doesn't apply to this case's mode

            scorer_results[scorer]["total"] += 1
            if score_passed:
                scorer_results[scorer]["passed"] += 1

    return [
        {
            "scorer": scorer,
            "pass_rate": (data["passed"] / data["total"]) * 100 if data["total"] > 0 else 0.0,
            "total": data["total"],
            "passed": data["passed"],
        }
        for scorer, data in sorted(scorer_results.items())
    ]


def compute_mode_latency(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute P50 latency per synthesis mode.

    Args:
        cases: List of case dicts with 'synthesis_mode' and 'duration_ms'.

    Returns:
        List of dicts with 'mode', 'p50_seconds', 'case_count'.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        mode = case.get("synthesis_mode")
        duration = case.get("duration_ms")
        if mode and duration is not None and duration > 0:
            groups[mode].append(duration)

    return [
        {
            "mode": mode,
            "p50_seconds": round(statistics.median(durations) / 1000, 3),
            "case_count": len(durations),
        }
        for mode, durations in sorted(groups.items())
    ]


def compute_difficulty_latency(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute P50 latency per difficulty level.

    Args:
        cases: List of case dicts with 'difficulty' and 'duration_ms'.

    Returns:
        List of dicts with 'difficulty', 'p50_seconds', 'case_count'.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        difficulty = case.get("difficulty")
        duration = case.get("duration_ms")
        if difficulty and duration is not None and duration > 0:
            groups[difficulty].append(duration)

    return [
        {
            "difficulty": diff,
            "p50_seconds": round(statistics.median(durations) / 1000, 3),
            "case_count": len(durations),
        }
        for diff, durations in sorted(groups.items())
    ]


def compute_run_mode_latency(
    sl_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute P50 and P95 latency per run mode from single-law runs.

    Groups case durations by the run's ``meta.run_mode`` and computes
    percentile latency for each group.

    Args:
        sl_runs: List of single-law run dicts with ``meta.run_mode``.

    Returns:
        List of dicts with ``run_mode``, ``p50_seconds``, ``p95_seconds``,
        ``case_count``.
    """
    groups: dict[str, list[float]] = defaultdict(list)
    for run in sl_runs:
        mode = run.get("meta", {}).get("run_mode", "unknown")
        for result in run.get("results", []):
            d = result.get("duration_ms")
            if d is not None and d > 0:
                groups[mode].append(d)

    mode_order = ["retrieval_only", "full", "full_with_judge"]
    result_list = []
    for mode in mode_order:
        if mode in groups:
            durations = sorted(groups[mode])
            n = len(durations)
            p50_idx = int(0.5 * (n - 1))
            p95_idx = int(0.95 * (n - 1))
            result_list.append({
                "run_mode": mode,
                "p50_seconds": round(durations[p50_idx] / 1000, 3),
                "p95_seconds": round(durations[min(p95_idx, n - 1)] / 1000, 3),
                "case_count": n,
            })
    # Any other modes not in the canonical order
    for mode, durations in sorted(groups.items()):
        if mode not in mode_order:
            durations_sorted = sorted(durations)
            n = len(durations_sorted)
            p50_idx = int(0.5 * (n - 1))
            p95_idx = int(0.95 * (n - 1))
            result_list.append({
                "run_mode": mode,
                "p50_seconds": round(durations_sorted[p50_idx] / 1000, 3),
                "p95_seconds": round(durations_sorted[min(p95_idx, n - 1)] / 1000, 3),
                "case_count": n,
            })
    return result_list


def compute_latency_trend(
    runs: list[dict[str, Any]],
    is_cross_law: bool = False,
) -> list[dict[str, Any]]:
    """Compute median latency trend from a list of runs.

    Each run produces one trend point: timestamp + median case latency.

    Args:
        runs: List of run dicts.
        is_cross_law: If True, uses cross-law structure (timestamp at
            top level, results directly).  Otherwise uses single-law
            structure (meta.timestamp).

    Returns:
        List of dicts with ``timestamp``, ``median_ms``, ``case_count``,
        sorted oldest-first.
    """
    points: list[dict[str, Any]] = []
    for run in runs:
        if is_cross_law:
            ts = run.get("timestamp", "")
            results = run.get("results", [])
        else:
            ts = run.get("meta", {}).get("timestamp", "")
            results = run.get("results", [])

        durations = [
            r.get("duration_ms")
            for r in results
            if r.get("duration_ms") is not None and r.get("duration_ms", 0) > 0
        ]
        if not durations or not ts:
            continue

        points.append({
            "timestamp": ts,
            "median_ms": round(statistics.median(durations), 1),
            "case_count": len(durations),
        })

    points.sort(key=lambda p: p["timestamp"])
    return points


def compute_latency_trend_by_run_mode(
    sl_runs: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Compute latency trend per run mode from historical SL runs.

    Groups runs by ``meta.run_mode``, then computes median case latency
    per run within each group.

    Args:
        sl_runs: List of single-law run dicts with ``meta.run_mode``.

    Returns:
        Dict mapping run_mode → list of trend points (timestamp, median_ms,
        case_count), sorted oldest-first.
    """
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for run in sl_runs:
        mode = run.get("meta", {}).get("run_mode", "unknown")
        ts = run.get("meta", {}).get("timestamp", "")
        results = run.get("results", [])
        durations = [
            r.get("duration_ms")
            for r in results
            if r.get("duration_ms") is not None and r.get("duration_ms", 0) > 0
        ]
        if not durations or not ts:
            continue

        by_mode[mode].append({
            "timestamp": ts,
            "median_ms": round(statistics.median(durations), 1),
            "case_count": len(durations),
        })

    # Sort each mode's points oldest-first
    for mode in by_mode:
        by_mode[mode].sort(key=lambda p: p["timestamp"])

    return dict(by_mode)


def compute_rate_trends(
    sl_runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute escalation and retry rate trends from historical SL runs.

    Aggregates runs by date (day), computing the escalation rate and
    retry rate for each day.

    Args:
        sl_runs: List of single-law run dicts with escalation_stats
            and retry_stats.

    Returns:
        Tuple of (escalation_trend, retry_trend), each a list of dicts
        with ``timestamp``, ``rate``, ``count``, ``total``.
    """
    # Group by date to avoid too many points
    by_date: dict[str, dict[str, int]] = {}
    for run in sl_runs:
        ts = run.get("meta", {}).get("timestamp", "")
        if not ts:
            continue
        date_key = ts[:10]  # YYYY-MM-DD
        total = run.get("summary", {}).get("total", 0)
        if total == 0:
            continue
        esc = run.get("escalation_stats", {}).get("cases_escalated", 0)
        retry = run.get("retry_stats", {}).get("cases_with_retries", 0)

        if date_key not in by_date:
            by_date[date_key] = {"total": 0, "escalated": 0, "retried": 0}
        by_date[date_key]["total"] += total
        by_date[date_key]["escalated"] += esc
        by_date[date_key]["retried"] += retry

    esc_trend: list[dict[str, Any]] = []
    retry_trend: list[dict[str, Any]] = []
    for date_key in sorted(by_date):
        d = by_date[date_key]
        total = d["total"]
        esc_trend.append({
            "timestamp": f"{date_key}T00:00:00Z",
            "rate": round(d["escalated"] / total * 100, 2) if total else 0.0,
            "count": d["escalated"],
            "total": total,
        })
        retry_trend.append({
            "timestamp": f"{date_key}T00:00:00Z",
            "rate": round(d["retried"] / total * 100, 2) if total else 0.0,
            "count": d["retried"],
            "total": total,
        })

    return esc_trend, retry_trend


def _load_all_single_law_runs(
    max_scan: int,
    active_laws: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load ALL single-law runs including historical timestamped files.

    Reads both ``eval_{law}.json`` (latest) and
    ``{timestamp}_eval_{law}.json`` (historical) from the runs directory.

    Args:
        max_scan: Maximum number of files to read.
        active_laws: Optional set of law IDs to keep.

    Returns:
        List of run dicts (full structure with meta, summary, results).
    """
    runs_dir = _get_single_law_runs_dir()
    if not runs_dir.exists():
        return []

    seen_timestamps: set[str] = set()
    runs: list[dict[str, Any]] = []

    # Load timestamped historical runs: {ts}_eval_{law}.json
    for path in sorted(runs_dir.glob("*_eval_*.json"))[-max_scan:]:
        if active_laws is not None:
            name = path.stem
            law_id = name.split("_eval_", 1)[-1] if "_eval_" in name else ""
            if law_id not in active_laws:
                continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("meta", {}).get("timestamp", "")
            if ts and ts not in seen_timestamps:
                seen_timestamps.add(ts)
                runs.append(data)
        except (json.JSONDecodeError, IOError):
            continue

    # Also load the latest eval_{law}.json files (deduplicate by timestamp)
    for path in sorted(runs_dir.glob("eval_*.json"))[:max_scan]:
        if active_laws is not None:
            law_id = path.stem.removeprefix("eval_")
            if law_id not in active_laws:
                continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("meta", {}).get("timestamp", "")
            if ts and ts not in seen_timestamps:
                seen_timestamps.add(ts)
                runs.append(data)
        except (json.JSONDecodeError, IOError):
            continue

    return runs


def _load_all_cross_law_runs(max_scan: int) -> list[dict[str, Any]]:
    """Load ALL cross-law runs (not just latest per suite).

    Args:
        max_scan: Maximum number of files to read.

    Returns:
        List of all run dicts, for trend computation.
    """
    runs_dir = _get_cross_law_runs_dir()
    if not runs_dir.exists():
        return []

    runs: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*.json"))[-max_scan:]:
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution (monkeypatchable for tests)
# ─────────────────────────────────────────────────────────────────────────────


def _get_single_law_runs_dir() -> Path:
    """Get path to single-law runs directory."""
    return PROJECT_ROOT / "runs"


def _get_cross_law_runs_dir() -> Path:
    """Get path to cross-law runs directory."""
    return PROJECT_ROOT / "data" / "evals" / "runs"


def _get_suites_dir() -> Path:
    """Get path to cross-law suites directory."""
    return PROJECT_ROOT / "data" / "evals"


def _get_corpora_path() -> Path:
    """Get path to corpora metadata file."""
    return PROJECT_ROOT / "data" / "processed" / "corpora.json"


# ─────────────────────────────────────────────────────────────────────────────
# Data loader functions (C2c)
# ─────────────────────────────────────────────────────────────────────────────


def _load_latest_single_law_runs(
    max_scan: int,
    active_laws: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load the latest stable run for each ingested law.

    Reads ``eval_{law}.json`` files from the runs directory.  When
    *active_laws* is provided, runs for laws not in the set are skipped
    (e.g. laws that have been deleted from the corpus).

    Args:
        max_scan: Maximum number of files to read.
        active_laws: Optional set of law IDs to keep.  If ``None``,
            all runs are included (backward-compatible).

    Returns:
        List of run dicts (full structure with meta, summary, results).
    """
    runs_dir = _get_single_law_runs_dir()
    if not runs_dir.exists():
        return []

    runs: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("eval_*.json"))[:max_scan]:
        if active_laws is not None:
            # Parse law from filename: eval_LAW.json → LAW
            law_id = path.stem.removeprefix("eval_")
            if law_id not in active_laws:
                continue
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to read single-law run: %s", path)
            continue
    return runs


def _load_latest_cross_law_runs(max_scan: int) -> list[dict[str, Any]]:
    """Load the latest run for each cross-law suite.

    Groups by suite_id and keeps only the newest run per suite.

    Args:
        max_scan: Maximum number of files to read.

    Returns:
        List of run dicts (one per suite, latest only).
    """
    runs_dir = _get_cross_law_runs_dir()
    if not runs_dir.exists():
        return []

    latest_by_suite: dict[str, dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_scan]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            suite_id = data.get("suite_id")
            if not suite_id:
                continue
            existing = latest_by_suite.get(suite_id)
            if existing is None or data.get("timestamp", "") > existing.get("timestamp", ""):
                latest_by_suite[suite_id] = data
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to read cross-law run: %s", path)
            continue

    return list(latest_by_suite.values())


def _load_historical_pass_rates(max_scan: int) -> list[float]:
    """Load historical pass rates for trend calculation.

    Collects pass rates from cross-law runs sorted by timestamp.

    Args:
        max_scan: Maximum number of run files to scan.

    Returns:
        List of pass rate percentages, oldest first.
    """
    runs_dir = _get_cross_law_runs_dir()
    if not runs_dir.exists():
        return []

    runs_with_ts: list[tuple[str, float]] = []
    for path in sorted(runs_dir.glob("*.json"))[:max_scan]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            ts = data.get("timestamp", "")
            rate = data.get("pass_rate", 0.0)
            runs_with_ts.append((ts, rate * 100))
        except (json.JSONDecodeError, IOError):
            continue

    runs_with_ts.sort(key=lambda x: x[0])
    return [rate for _, rate in runs_with_ts]


def _load_suite_names() -> dict[str, str]:
    """Load suite_id → display name mapping from suite YAML files.

    Returns:
        Dict mapping suite_id to suite name.
    """
    suites_dir = _get_suites_dir()
    if not suites_dir.exists():
        return {}

    names: dict[str, str] = {}
    for path in suites_dir.glob("cross_law_*.yaml"):
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                suite_id = data.get("id") or path.stem.removeprefix("cross_law_")
                name = data.get("name", suite_id)
                names[suite_id] = name
        except (yaml.YAMLError, IOError):
            logger.warning("Failed to read suite YAML: %s", path)
            continue
    return names


def _load_corpora() -> dict[str, Any]:
    """Load corpora metadata from corpora.json.

    Returns:
        Dict with 'corpora' key mapping corpus_id → metadata.
    """
    corpora_path = _get_corpora_path()
    if not corpora_path.exists():
        return {"corpora": {}}
    try:
        with open(corpora_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.warning("Failed to read corpora: %s", corpora_path)
        return {"corpora": {}}


def _avg_pass_rate_per_group(
    runs: list[dict[str, Any]],
    key: str,
) -> float | None:
    """Average pass rate treating each group (law/suite) equally.

    For single-law runs: each run IS one law, so average their pass rates.
    For cross-law runs: each run IS one suite, so average their pass rates.

    Args:
        runs: List of run dicts.
        key: ``"single_law"`` or ``"cross_law"``.

    Returns:
        Average pass rate percentage, or ``None`` if no data.
    """
    if not runs:
        return None

    rates: list[float] = []
    for run in runs:
        if key == "single_law":
            summary = run.get("summary", {})
            total = summary.get("total", 0)
            passed = summary.get("passed", 0)
        else:
            total = run.get("total", 0)
            passed = run.get("passed", 0)
        if total > 0:
            rates.append(passed / total * 100)

    if not rates:
        return None
    return round(statistics.mean(rates), 2)


def _get_active_law_ids() -> set[str]:
    """Return set of law IDs that have been ingested (present in corpora.json)."""
    corpora = _load_corpora()
    return set(corpora.get("corpora", {}).keys())


def _normalize_scores(scores: dict[str, Any]) -> dict[str, bool]:
    """Normalize scorer results to plain booleans.

    Run files store scores as ``{passed: bool, score: float, message: str}``
    dicts.  Pure aggregation functions expect plain booleans.

    Args:
        scores: Raw scores dict from a run result.

    Returns:
        Dict mapping scorer name → passed boolean.
    """
    normalized: dict[str, bool] = {}
    for scorer, value in scores.items():
        if isinstance(value, dict):
            normalized[scorer] = bool(value.get("passed", False))
        else:
            normalized[scorer] = bool(value)
    return normalized


def _extract_score_messages(scores: dict[str, Any]) -> dict[str, str]:
    """Extract human-readable messages from scorer results.

    Args:
        scores: Raw scores dict from a run result.

    Returns:
        Dict mapping scorer name → message string (empty if absent).
    """
    messages: dict[str, str] = {}
    for scorer, value in scores.items():
        if isinstance(value, dict):
            msg = value.get("message", "")
            if msg:
                messages[scorer] = str(msg)
    return messages


def _extract_anchors(result: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Extract actual and expected anchors from a single-law case result.

    Args:
        result: Raw case result dict.

    Returns:
        Tuple of (actual_anchors, expected_anchors).
    """
    actual: list[str] = []
    expected: list[str] = []

    # From run_meta.anchors_in_top_k
    run_meta = result.get("run_meta", {})
    if isinstance(run_meta, dict):
        actual = run_meta.get("anchors_in_top_k", [])

    # From scores.anchor_presence.details
    scores = result.get("scores", {})
    anchor_score = scores.get("anchor_presence", {})
    if isinstance(anchor_score, dict):
        details = anchor_score.get("details", {})
        if isinstance(details, dict):
            expected = details.get("expected_any_of", []) + details.get("expected_all_of", [])

    return actual, expected


def _collect_single_law_cases(
    single_law_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract normalized case dicts from single-law runs.

    Args:
        single_law_runs: List of single-law run dicts.

    Returns:
        List of case dicts with normalized scores.
    """
    cases: list[dict[str, Any]] = []
    for run in single_law_runs:
        for result in run.get("results", []):
            cases.append({
                "case_id": result.get("case_id", ""),
                "passed": result.get("passed", False),
                "duration_ms": result.get("duration_ms", 0),
                "scores": _normalize_scores(result.get("scores", {})),
            })
    return cases


def _collect_cross_law_cases(
    cross_law_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract normalized case dicts from cross-law runs.

    Args:
        cross_law_runs: List of cross-law run dicts.

    Returns:
        List of case dicts with normalized scores.
    """
    cases: list[dict[str, Any]] = []
    for run in cross_law_runs:
        for result in run.get("results", []):
            cases.append({
                "case_id": result.get("case_id", ""),
                "synthesis_mode": result.get("synthesis_mode"),
                "difficulty": result.get("difficulty"),
                "passed": result.get("passed", False),
                "duration_ms": result.get("duration_ms", 0),
                "scores": _normalize_scores(result.get("scores", {})),
                "score_messages": _extract_score_messages(result.get("scores", {})),
                "prompt": result.get("prompt", ""),
                "target_corpora": result.get("target_corpora", []),
            })
    return cases


def _count_case_origins() -> tuple[dict[str, int], dict[str, int]]:
    """Count eval case definitions by origin (manual/auto/auto-generated).

    Scans golden_cases_*.yaml and cross_law_*.yaml in the evals directory.

    Returns:
        Tuple of (sl_counts, cl_counts) — each mapping origin label → case count.
    """
    evals_dir = _get_suites_dir()
    sl_counts: dict[str, int] = defaultdict(int)
    cl_counts: dict[str, int] = defaultdict(int)

    # Single-law golden cases
    for path in sorted(evals_dir.glob("golden_cases_*.yaml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            cases = data if isinstance(data, list) else data.get("cases", [])
            for case in cases:
                origin = case.get("origin", "unknown")
                sl_counts[origin] += 1
        except Exception:
            continue

    # Cross-law suites
    for path in sorted(evals_dir.glob("cross_law_*.yaml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            cases = data.get("cases", []) if isinstance(data, dict) else data
            for case in cases:
                origin = case.get("origin", "unknown")
                cl_counts[origin] += 1
        except Exception:
            continue

    return dict(sl_counts), dict(cl_counts)


def _build_case_prompt_lookup() -> dict[str, str]:
    """Build case_id → prompt mapping from golden_cases YAML definitions.

    Returns:
        Dict mapping case_id to its prompt string.
    """
    evals_dir = _get_suites_dir()
    lookup: dict[str, str] = {}

    for path in sorted(evals_dir.glob("golden_cases_*.yaml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            cases = data if isinstance(data, list) else data.get("cases", [])
            for case in cases:
                case_id = case.get("id", "")
                prompt = case.get("prompt", "")
                if case_id and prompt:
                    lookup[case_id] = prompt
        except Exception:
            continue

    return lookup


def _get_display_name(law: str, corpora: dict[str, Any]) -> str:
    """Resolve human-readable display name for a law.

    Args:
        law: Law identifier.
        corpora: Full corpora dict from corpora.json.

    Returns:
        Display name, or formatted law id as fallback.
    """
    corpus = corpora.get("corpora", {}).get(law, {})
    return corpus.get("display_name", law.replace("-", " ").replace("_", " ").title())


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 endpoints (C2c)
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/overview", response_model=MetricsOverviewResponse)
async def get_metrics_overview() -> MetricsOverviewResponse:
    """Level 1 trust overview — unified pass rate, health, trend."""
    config = get_settings_yaml()["dashboard"]

    sl_runs = _load_latest_single_law_runs(config["max_runs_scan"], _get_active_law_ids())
    cl_runs = _load_latest_cross_law_runs(config["max_runs_scan"])

    # Build summary dicts for compute_unified_pass_rate
    sl_summaries = [
        {"total": r["summary"]["total"], "passed": r["summary"]["passed"]}
        for r in sl_runs
        if "summary" in r
    ]
    cl_summaries = [
        {"total": r.get("total", 0), "passed": r.get("passed", 0)}
        for r in cl_runs
    ]

    unified_rate = compute_unified_pass_rate(sl_summaries, cl_summaries)
    health = compute_health_status(unified_rate, config["health_thresholds"])

    # Trend from historical cross-law pass rates
    history = _load_historical_pass_rates(config["max_runs_scan"])
    # Remove the latest entry to separate history from current
    if history:
        trend_history = history[:-1]
        trend_latest = history[-1]
    else:
        trend_history = []
        trend_latest = unified_rate
    trend = compute_trend(
        trend_history, trend_latest, config["trend_window"], config["trend_threshold"]
    )

    # Totals
    sl_total = sum(s["total"] for s in sl_summaries)
    sl_passed = sum(s["passed"] for s in sl_summaries)
    cl_total = sum(s["total"] for s in cl_summaries)
    cl_passed = sum(s["passed"] for s in cl_summaries)

    has_data = bool(sl_runs or cl_runs)

    # Summary stats
    timestamps = []
    for r in sl_runs:
        ts = r.get("meta", {}).get("timestamp")
        if ts:
            timestamps.append(ts)
    for r in cl_runs:
        ts = r.get("timestamp")
        if ts:
            timestamps.append(ts)

    # Run mode distribution — split into SL and CL
    sl_run_mode_dist: dict[str, int] = defaultdict(int)
    cl_run_mode_dist: dict[str, int] = defaultdict(int)
    for r in sl_runs:
        mode = r.get("meta", {}).get("run_mode", "unknown")
        case_count = r.get("summary", {}).get("total", 0)
        sl_run_mode_dist[mode] += case_count
    for r in cl_runs:
        mode = r.get("run_mode", "unknown")
        case_count = r.get("total", 0)
        cl_run_mode_dist[mode] += case_count

    # Case origin distribution — split into SL and CL
    sl_origin_dist, cl_origin_dist = _count_case_origins()

    # Ingestion health summary (mirrors /ingestion logic)
    corpora_data = _load_corpora()
    coverages: list[float] = []
    for _cid, meta in corpora_data.get("corpora", {}).items():
        coverage = meta.get("quality", {}).get("structure_coverage_pct", 0.0)
        coverages.append(coverage)
    ingested_coverages = [c for c in coverages if c > 0]
    na_count = len(coverages) - len(ingested_coverages)
    ingestion_overall = statistics.mean(ingested_coverages) if ingested_coverages else 0.0
    ingestion_health = compute_health_status(ingestion_overall, config["health_thresholds"])

    return MetricsOverviewResponse(
        unified_pass_rate=round(unified_rate, 2),
        health_status=health,
        trend=TrendInfo(**trend),
        summary=MetricsSummary(
            total_cases=sl_total + cl_total,
            law_count=len(sl_runs),
            suite_count=len(cl_runs),
            last_run_timestamp=max(timestamps) if timestamps else None,
        ),
        single_law=CategorySummary(
            total=sl_total,
            passed=sl_passed,
            pass_rate=round((sl_passed / sl_total * 100) if sl_total else 0.0, 2),
            group_pass_rate=_avg_pass_rate_per_group(sl_runs, key="single_law"),
        ),
        cross_law=CategorySummary(
            total=cl_total,
            passed=cl_passed,
            pass_rate=round((cl_passed / cl_total * 100) if cl_total else 0.0, 2),
            group_pass_rate=_avg_pass_rate_per_group(cl_runs, key="cross_law"),
        ),
        has_data=has_data,
        sl_run_mode_distribution=dict(sl_run_mode_dist),
        cl_run_mode_distribution=dict(cl_run_mode_dist),
        sl_case_origin_distribution=sl_origin_dist,
        cl_case_origin_distribution=cl_origin_dist,
        ingestion_overall_coverage=round(ingestion_overall, 2),
        ingestion_health_status=ingestion_health,
        ingestion_na_count=na_count,
    )


@router.get("/quality", response_model=MetricsQualityResponse)
async def get_metrics_quality() -> MetricsQualityResponse:
    """Level 2 eval quality — per-law, per-suite, per-mode, per-scorer breakdowns."""
    config = get_settings_yaml()["dashboard"]
    corpora = _load_corpora()

    sl_runs = _load_latest_single_law_runs(config["max_runs_scan"], _get_active_law_ids())
    cl_runs = _load_latest_cross_law_runs(config["max_runs_scan"])
    suite_names = _load_suite_names()

    # Per-law pass rates (single-law)
    per_law: list[LawPassRate] = []
    for run in sl_runs:
        law = run.get("meta", {}).get("law", "unknown")
        summary = run.get("summary", {})
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        per_law.append(LawPassRate(
            law=law,
            display_name=_get_display_name(law, corpora),
            pass_rate=round((passed / total * 100) if total else 0.0, 2),
            total=total,
            passed=passed,
        ))

    # Per-suite pass rates (cross-law)
    per_suite: list[SuitePassRate] = []
    for run in cl_runs:
        suite_id = run.get("suite_id", "unknown")
        total = run.get("total", 0)
        passed = run.get("passed", 0)
        per_suite.append(SuitePassRate(
            suite_id=suite_id,
            name=suite_names.get(suite_id, suite_id),
            pass_rate=round((passed / total * 100) if total else 0.0, 2),
            total=total,
            passed=passed,
        ))

    # Single-law case-level scorer aggregation
    sl_cases = _collect_single_law_cases(sl_runs)
    sl_scorer_raw = aggregate_scorers(sl_cases)
    per_scorer_single_law = [
        ScorerPassRate(scorer=s["scorer"], pass_rate=round(s["pass_rate"], 2),
                       total=s["total"], passed=s["passed"], category="single_law")
        for s in sl_scorer_raw
    ]

    # Cross-law case-level aggregations
    cl_cases = _collect_cross_law_cases(cl_runs)

    per_mode_raw = aggregate_by_mode(cl_cases)
    per_mode = [ModePassRate(**m) for m in per_mode_raw]

    per_diff_raw = aggregate_by_difficulty(cl_cases)
    per_difficulty = [DifficultyPassRate(**d) for d in per_diff_raw]

    scorer_raw = aggregate_scorers(cl_cases)
    per_scorer = [
        ScorerPassRate(scorer=s["scorer"], pass_rate=round(s["pass_rate"], 2),
                       total=s["total"], passed=s["passed"], category="cross_law")
        for s in scorer_raw
    ]

    # Stage stats from single-law runs
    retrieval_total = retrieval_passed = 0
    augmentation_total = augmentation_passed = 0
    generation_total = generation_passed = 0
    for run in sl_runs:
        ss = run.get("stage_stats", {})
        retrieval_total += ss.get("retrieval_total", 0)
        retrieval_passed += ss.get("retrieval_passed", 0)
        augmentation_total += ss.get("augmentation_total", 0)
        augmentation_passed += ss.get("augmentation_passed", 0)
        generation_total += ss.get("generation_total", 0)
        generation_passed += ss.get("generation_passed", 0)

    stage_stats = StageBreakdown(
        retrieval=round((retrieval_passed / retrieval_total * 100) if retrieval_total else 0.0, 2),
        augmentation=round((augmentation_passed / augmentation_total * 100) if augmentation_total else 0.0, 2),
        generation=round((generation_passed / generation_total * 100) if generation_total else 0.0, 2),
    )

    # Escalation and retry rates
    sl_total_cases = sum(r.get("summary", {}).get("total", 0) for r in sl_runs)
    escalated = sum(
        r.get("escalation_stats", {}).get("cases_escalated", 0) for r in sl_runs
    )
    retried = sum(
        r.get("retry_stats", {}).get("cases_with_retries", 0) for r in sl_runs
    )

    return MetricsQualityResponse(
        per_law=per_law,
        per_suite=per_suite,
        per_mode=per_mode,
        per_difficulty=per_difficulty,
        per_scorer=per_scorer,
        per_scorer_single_law=per_scorer_single_law,
        stage_stats=stage_stats,
        escalation_rate=round((escalated / sl_total_cases * 100) if sl_total_cases else 0.0, 2),
        retry_rate=round((retried / sl_total_cases * 100) if sl_total_cases else 0.0, 2),
    )


@router.get("/performance", response_model=MetricsPerformanceResponse)
async def get_metrics_performance() -> MetricsPerformanceResponse:
    """Level 2 processing performance — latency, percentiles, histogram."""
    config = get_settings_yaml()["dashboard"]
    active_laws = _get_active_law_ids()

    sl_runs = _load_latest_single_law_runs(config["max_runs_scan"], active_laws)
    cl_runs = _load_latest_cross_law_runs(config["max_runs_scan"])

    # ── Combined (legacy) ──────────────────────────────────────────────
    all_durations_ms: list[float] = []
    sl_durations_ms: list[float] = []
    for run in sl_runs:
        for result in run.get("results", []):
            d = result.get("duration_ms")
            if d is not None and d > 0:
                all_durations_ms.append(d)
                sl_durations_ms.append(d)

    cl_cases = _collect_cross_law_cases(cl_runs)
    cl_durations_ms: list[float] = []
    for case in cl_cases:
        d = case.get("duration_ms")
        if d is not None and d > 0:
            all_durations_ms.append(d)
            cl_durations_ms.append(d)

    total_cases = len(all_durations_ms)
    percentiles_raw = compute_percentiles(all_durations_ms) if all_durations_ms else {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    histogram_raw = compute_histogram_bins(all_durations_ms)
    mode_lat_raw = compute_mode_latency(cl_cases)
    diff_lat_raw = compute_difficulty_latency(cl_cases)

    # ── Single-law performance ─────────────────────────────────────────
    sl_percentiles_raw = compute_percentiles(sl_durations_ms) if sl_durations_ms else {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    sl_histogram_raw = compute_histogram_bins(sl_durations_ms)

    # Escalation and retry rates (same computation as quality endpoint)
    sl_total_case_count = sum(r.get("summary", {}).get("total", 0) for r in sl_runs)
    escalated = sum(r.get("escalation_stats", {}).get("cases_escalated", 0) for r in sl_runs)
    retried = sum(r.get("retry_stats", {}).get("cases_with_retries", 0) for r in sl_runs)

    # Load all historical SL runs (needed for run-mode latency + trends)
    all_sl_runs = _load_all_single_law_runs(config["max_runs_scan"], active_laws)

    # Latency by run mode (from ALL runs, including historical retrieval_only/full)
    run_mode_lat_raw = compute_run_mode_latency(all_sl_runs)
    sl_trend_raw = compute_latency_trend(all_sl_runs, is_cross_law=False)

    # Escalation and retry rate trends from historical SL runs
    esc_trend_raw, retry_trend_raw = compute_rate_trends(all_sl_runs)

    # Per-run-mode latency trends
    mode_trends_raw = compute_latency_trend_by_run_mode(all_sl_runs)
    mode_order = ["retrieval_only", "full", "full_with_judge"]
    mode_trends = [
        RunModeTrend(
            run_mode=mode,
            points=[LatencyTrendPoint(**p) for p in mode_trends_raw[mode][-20:]],
        )
        for mode in mode_order
        if mode in mode_trends_raw and len(mode_trends_raw[mode]) > 1
    ]

    sl_perf = SLPerformance(
        percentiles=Percentiles(**sl_percentiles_raw),
        total_cases=len(sl_durations_ms),
        escalation_rate=round((escalated / sl_total_case_count * 100) if sl_total_case_count else 0.0, 2),
        retry_rate=round((retried / sl_total_case_count * 100) if sl_total_case_count else 0.0, 2),
        histogram_bins=[HistogramBin(**b) for b in sl_histogram_raw],
        latency_by_run_mode=[RunModeLatency(**m) for m in run_mode_lat_raw],
        trend=[LatencyTrendPoint(**p) for p in sl_trend_raw[-20:]],
        trend_by_run_mode=mode_trends,
        escalation_trend=[RateTrendPoint(**p) for p in esc_trend_raw[-20:]],
        retry_trend=[RateTrendPoint(**p) for p in retry_trend_raw[-20:]],
    )

    # ── Cross-law performance ──────────────────────────────────────────
    cl_percentiles_raw = compute_percentiles(cl_durations_ms) if cl_durations_ms else {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    cl_histogram_raw = compute_histogram_bins(cl_durations_ms)

    # CL trend: load all runs (not just latest per suite)
    all_cl_runs = _load_all_cross_law_runs(config["max_runs_scan"])
    cl_trend_raw = compute_latency_trend(all_cl_runs, is_cross_law=True)

    cl_perf = CLPerformance(
        percentiles=Percentiles(**cl_percentiles_raw),
        total_cases=len(cl_durations_ms),
        histogram_bins=[HistogramBin(**b) for b in cl_histogram_raw],
        latency_by_synthesis_mode=[ModeLatency(**m) for m in mode_lat_raw],
        latency_by_difficulty=[DifficultyLatency(**d) for d in diff_lat_raw],
        trend=[LatencyTrendPoint(**p) for p in cl_trend_raw[-20:]],
    )

    return MetricsPerformanceResponse(
        percentiles=Percentiles(**percentiles_raw),
        total_cases=total_cases,
        excluded_zero_duration=0,
        histogram_bins=[HistogramBin(**b) for b in histogram_raw],
        latency_by_synthesis_mode=[ModeLatency(**m) for m in mode_lat_raw],
        latency_by_difficulty=[DifficultyLatency(**d) for d in diff_lat_raw],
        single_law=sl_perf,
        cross_law=cl_perf,
    )


@router.get("/ingestion", response_model=MetricsIngestionResponse)
async def get_metrics_ingestion() -> MetricsIngestionResponse:
    """Level 2 ingestion health — corpus coverage and status."""
    config = get_settings_yaml()["dashboard"]
    corpora_data = _load_corpora()

    corpora_list: list[CorpusHealth] = []
    coverages: list[float] = []

    for corpus_id, meta in corpora_data.get("corpora", {}).items():
        quality = meta.get("quality", {})
        coverage = quality.get("structure_coverage_pct", 0.0)
        coverages.append(coverage)
        corpora_list.append(CorpusHealth(
            corpus_id=corpus_id,
            display_name=meta.get("display_name", corpus_id),
            coverage=coverage,
            unhandled=quality.get("unhandled_count", 0),
            chunks=quality.get("chunk_count", 0),
            is_ingested=meta.get("enabled", True),
        ))

    # Exclude 0% coverage corpora (not yet ingested) from the overall calculation
    ingested_coverages = [c for c in coverages if c > 0]
    overall = statistics.mean(ingested_coverages) if ingested_coverages else 0.0
    health = compute_health_status(overall, config["health_thresholds"])

    return MetricsIngestionResponse(
        overall_coverage=round(overall, 2),
        health_status=health,
        corpora=corpora_list,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Additional loader functions (C2d)
# ─────────────────────────────────────────────────────────────────────────────


def _load_all_cross_law_runs_for_suite(
    suite_id: str, max_scan: int,
) -> list[dict[str, Any]]:
    """Load all runs for a specific cross-law suite, sorted newest first.

    Args:
        suite_id: The suite identifier.
        max_scan: Maximum run files to scan.

    Returns:
        List of run dicts for this suite, newest first.
    """
    runs_dir = _get_cross_law_runs_dir()
    if not runs_dir.exists():
        return []

    suite_runs: list[dict[str, Any]] = []
    for path in runs_dir.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("suite_id") == suite_id:
                suite_runs.append(data)
        except (json.JSONDecodeError, IOError):
            continue

    suite_runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return suite_runs[:max_scan]


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 endpoints (C2d)
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/detail/law/{law}", response_model=MetricsLawDetailResponse)
async def get_law_detail(law: str) -> MetricsLawDetailResponse:
    """Level 3 per-law drill-down — trend, scorer breakdown, case results."""
    config = get_settings_yaml()["dashboard"]
    corpora = _load_corpora()

    # Load all single-law runs for this law
    sl_runs = _load_latest_single_law_runs(config["max_runs_scan"], _get_active_law_ids())
    law_runs = [r for r in sl_runs if r.get("meta", {}).get("law") == law]

    # Latest run for case results and scorer breakdown
    latest_run = law_runs[0] if law_runs else None

    # Scorer breakdown from latest run
    scorer_breakdown: list[ScorerPassRate] = []
    latest_results: list[CaseResultSummary] = []

    if latest_run:
        # Build case_id → prompt lookup from golden_cases YAML definitions
        prompt_lookup = _build_case_prompt_lookup()

        # Aggregate scorers from case-level results
        scorer_totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"passed": 0, "total": 0}
        )
        for result in latest_run.get("results", []):
            scores = result.get("scores", {})
            for scorer, value in scores.items():
                passed = value.get("passed", False) if isinstance(value, dict) else bool(value)
                scorer_totals[scorer]["total"] += 1
                if passed:
                    scorer_totals[scorer]["passed"] += 1

            actual_anchors, expected_anchors = _extract_anchors(result)
            case_id = result.get("case_id", "")
            latest_results.append(CaseResultSummary(
                case_id=case_id,
                passed=result.get("passed", False),
                duration_ms=result.get("duration_ms", 0),
                scores=_normalize_scores(result.get("scores", {})),
                score_messages=_extract_score_messages(result.get("scores", {})),
                anchors=actual_anchors,
                expected_anchors=expected_anchors,
                profile=result.get("profile", ""),
                prompt=prompt_lookup.get(case_id, ""),
            ))

        for scorer, data in sorted(scorer_totals.items()):
            rate = (data["passed"] / data["total"] * 100) if data["total"] else 0.0
            scorer_breakdown.append(ScorerPassRate(
                scorer=scorer,
                pass_rate=round(rate, 2),
                total=data["total"],
                passed=data["passed"],
                category="single_law",
            ))

    return MetricsLawDetailResponse(
        law=law,
        display_name=_get_display_name(law, corpora),
        scorer_breakdown=scorer_breakdown,
        latest_results=latest_results,
    )


@router.get("/detail/suite/{suite_id}", response_model=MetricsSuiteDetailResponse)
async def get_suite_detail(
    suite_id: str,
    mode: str | None = Query(default=None, description="Filter by synthesis mode"),
    difficulty: str | None = Query(default=None, description="Filter by difficulty"),
) -> MetricsSuiteDetailResponse:
    """Level 3 per-suite drill-down — trend, filtered results, mode/difficulty counts."""
    config = get_settings_yaml()["dashboard"]
    suite_names = _load_suite_names()

    suite_runs = _load_all_cross_law_runs_for_suite(suite_id, config["max_runs_scan"])

    # Trend from all runs for this suite
    trend: list[TrendPoint] = []
    for run in reversed(suite_runs):  # oldest first
        trend.append(TrendPoint(
            run_id=run.get("run_id", ""),
            timestamp=run.get("timestamp", ""),
            pass_rate=round(run.get("pass_rate", 0.0) * 100, 2),
        ))

    # Latest run for detail
    latest = suite_runs[0] if suite_runs else None
    all_cases = _collect_cross_law_cases([latest]) if latest else []

    # Mode/difficulty counts (always unfiltered)
    mode_counts: dict[str, int] = defaultdict(int)
    difficulty_counts: dict[str, int] = defaultdict(int)
    for case in all_cases:
        m = case.get("synthesis_mode")
        if m:
            mode_counts[m] += 1
        d = case.get("difficulty")
        if d:
            difficulty_counts[d] += 1

    # Apply filters
    filtered_cases = all_cases
    if mode:
        filtered_cases = [c for c in filtered_cases if c.get("synthesis_mode") == mode]
    if difficulty:
        filtered_cases = [c for c in filtered_cases if c.get("difficulty") == difficulty]

    # Scorer breakdown from filtered cases
    scorer_raw = aggregate_scorers(filtered_cases, mode_filter=mode)
    scorer_breakdown = [
        ScorerPassRate(scorer=s["scorer"], pass_rate=s["pass_rate"],
                       total=s["total"], passed=s["passed"], category="cross_law")
        for s in scorer_raw
    ]

    # Case results
    latest_results = [
        CrossLawCaseResultSummary(
            case_id=c["case_id"],
            passed=c["passed"],
            duration_ms=c["duration_ms"],
            synthesis_mode=c.get("synthesis_mode", ""),
            difficulty=c.get("difficulty"),
            scores=c.get("scores", {}),
            score_messages=c.get("score_messages", {}),
            prompt=c.get("prompt", ""),
            target_corpora=c.get("target_corpora", []),
        )
        for c in filtered_cases
    ]

    return MetricsSuiteDetailResponse(
        suite_id=suite_id,
        name=suite_names.get(suite_id, suite_id),
        trend=trend,
        scorer_breakdown=scorer_breakdown,
        latest_results=latest_results,
        mode_counts=dict(mode_counts),
        difficulty_counts=dict(difficulty_counts),
        applied_filters=AppliedFilters(mode=mode, difficulty=difficulty),
    )


@router.get("/detail/scorer/{scorer}", response_model=MetricsScorerDetailResponse)
async def get_scorer_detail(scorer: str) -> MetricsScorerDetailResponse:
    """Level 3 per-scorer drill-down — per-law rates for this scorer."""
    config = get_settings_yaml()["dashboard"]
    corpora = _load_corpora()

    sl_runs = _load_latest_single_law_runs(config["max_runs_scan"], _get_active_law_ids())

    # Per-law pass rates for this specific scorer
    per_law_rates: list[LawPassRate] = []
    for run in sl_runs:
        law = run.get("meta", {}).get("law", "unknown")
        total = 0
        passed = 0
        for result in run.get("results", []):
            scores = result.get("scores", {})
            if scorer in scores:
                total += 1
                value = scores[scorer]
                if isinstance(value, dict):
                    if value.get("passed", False):
                        passed += 1
                elif bool(value):
                    passed += 1

        if total > 0:
            per_law_rates.append(LawPassRate(
                law=law,
                display_name=_get_display_name(law, corpora),
                pass_rate=round((passed / total * 100), 2),
                total=total,
                passed=passed,
            ))

    return MetricsScorerDetailResponse(
        scorer=scorer,
        label=scorer.replace("_", " ").title(),
        per_law_rates=per_law_rates,
    )


@router.get("/detail/mode/{mode}", response_model=MetricsModeDetailResponse)
async def get_mode_detail(mode: str) -> MetricsModeDetailResponse:
    """Level 3 per-mode drill-down — cases and applicable scorers for this mode."""
    config = get_settings_yaml()["dashboard"]

    cl_runs = _load_latest_cross_law_runs(config["max_runs_scan"])
    all_cases = _collect_cross_law_cases(cl_runs)

    # Filter cases by mode
    mode_cases = [c for c in all_cases if c.get("synthesis_mode") == mode]

    total = len(mode_cases)
    passed = sum(1 for c in mode_cases if c.get("passed"))
    pass_rate = round((passed / total * 100) if total else 0.0, 2)

    # Applicable scorers — only scorers that appear in cases of this mode
    scorer_raw = aggregate_scorers(mode_cases, mode_filter=mode)
    applicable_scorers = [
        ScorerPassRate(scorer=s["scorer"], pass_rate=s["pass_rate"],
                       total=s["total"], passed=s["passed"], category="cross_law")
        for s in scorer_raw
    ]

    cases = [
        CrossLawCaseResultSummary(
            case_id=c["case_id"],
            passed=c["passed"],
            duration_ms=c["duration_ms"],
            synthesis_mode=c.get("synthesis_mode", ""),
            difficulty=c.get("difficulty"),
            scores=c.get("scores", {}),
            score_messages=c.get("score_messages", {}),
            prompt=c.get("prompt", ""),
            target_corpora=c.get("target_corpora", []),
        )
        for c in mode_cases
    ]

    return MetricsModeDetailResponse(
        mode=mode,
        pass_rate=pass_rate,
        total=total,
        passed=passed,
        applicable_scorers=applicable_scorers,
        cases=cases,
    )


@router.get("/detail/difficulty/{difficulty}", response_model=MetricsDifficultyDetailResponse)
async def get_difficulty_detail(difficulty: str) -> MetricsDifficultyDetailResponse:
    """Level 3 per-difficulty drill-down — cases with this difficulty level."""
    config = get_settings_yaml()["dashboard"]

    cl_runs = _load_latest_cross_law_runs(config["max_runs_scan"])
    all_cases = _collect_cross_law_cases(cl_runs)

    # Filter cases by difficulty
    diff_cases = [c for c in all_cases if c.get("difficulty") == difficulty]

    total = len(diff_cases)
    passed = sum(1 for c in diff_cases if c.get("passed"))
    pass_rate = round((passed / total * 100) if total else 0.0, 2)

    cases = [
        CrossLawCaseResultSummary(
            case_id=c["case_id"],
            passed=c["passed"],
            duration_ms=c["duration_ms"],
            synthesis_mode=c.get("synthesis_mode", ""),
            difficulty=c.get("difficulty"),
            scores=c.get("scores", {}),
            score_messages=c.get("score_messages", {}),
            prompt=c.get("prompt", ""),
            target_corpora=c.get("target_corpora", []),
        )
        for c in diff_cases
    ]

    return MetricsDifficultyDetailResponse(
        difficulty=difficulty,
        pass_rate=pass_rate,
        total=total,
        passed=passed,
        cases=cases,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyse — AI analysis (SSE stream) (C2e)
# ─────────────────────────────────────────────────────────────────────────────


def _build_metrics_snapshot(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build a comprehensive metrics snapshot dict for the AI analysis prompt.

    Collects data from the same loaders used by the overview/quality/performance
    endpoints, then packs it into a flat dict matching what the engine expects.
    """
    active_laws = _get_active_law_ids()
    sl_runs = _load_latest_single_law_runs(config["max_runs_scan"], active_laws)
    cl_runs = _load_latest_cross_law_runs(config["max_runs_scan"])
    corpora = _load_corpora()

    # Overview numbers
    unified = compute_unified_pass_rate(
        [r.get("summary", {}) for r in sl_runs],
        [{"total": r.get("total", 0), "passed": r.get("passed", 0)} for r in cl_runs],
    )
    health = compute_health_status(unified, config["health_thresholds"])

    # Trend
    history = _load_historical_pass_rates(config["max_runs_scan"])
    if history:
        trend_history = history[:-1]
        trend_latest = history[-1]
    else:
        trend_history = []
        trend_latest = unified
    trend = compute_trend(
        trend_history, trend_latest, config["trend_window"], config["trend_threshold"]
    )

    sl_total = sum(r.get("summary", {}).get("total", 0) for r in sl_runs)
    sl_passed = sum(r.get("summary", {}).get("passed", 0) for r in sl_runs)
    sl_rate = round((sl_passed / sl_total * 100) if sl_total else 0.0, 2)

    cl_total = sum(r.get("total", 0) for r in cl_runs)
    cl_passed = sum(r.get("passed", 0) for r in cl_runs)
    cl_rate = round((cl_passed / cl_total * 100) if cl_total else 0.0, 2)

    # Quality: per-mode and per-difficulty
    all_cases = _collect_cross_law_cases(cl_runs)
    per_mode = aggregate_by_mode(all_cases)
    per_difficulty = aggregate_by_difficulty(all_cases)
    per_scorer = aggregate_scorers(all_cases)

    # Performance: collect durations (same logic as performance endpoint)
    sl_durations_ms: list[float] = []
    for run in sl_runs:
        for result in run.get("results", []):
            d = result.get("duration_ms")
            if d is not None and d > 0:
                sl_durations_ms.append(d)
    cl_durations_ms: list[float] = []
    for case in all_cases:
        d = case.get("duration_ms")
        if d is not None and d > 0:
            cl_durations_ms.append(d)
    all_durations_ms = sl_durations_ms + cl_durations_ms
    percentiles = compute_percentiles(all_durations_ms) if all_durations_ms else {
        "p50": 0.0, "p95": 0.0, "p99": 0.0,
    }
    sl_percentiles = compute_percentiles(sl_durations_ms) if sl_durations_ms else {
        "p50": 0.0, "p95": 0.0, "p99": 0.0,
    }
    cl_percentiles = compute_percentiles(cl_durations_ms) if cl_durations_ms else {
        "p50": 0.0, "p95": 0.0, "p99": 0.0,
    }

    # Escalation & retry rates
    sl_total_case_count = sum(r.get("summary", {}).get("total", 0) for r in sl_runs)
    escalated = sum(r.get("escalation_stats", {}).get("cases_escalated", 0) for r in sl_runs)
    retried = sum(r.get("retry_stats", {}).get("cases_with_retries", 0) for r in sl_runs)
    escalation_rate = round((escalated / sl_total_case_count * 100) if sl_total_case_count else 0.0, 2)
    retry_rate = round((retried / sl_total_case_count * 100) if sl_total_case_count else 0.0, 2)

    # Per-run-mode latency (SL)
    all_sl_runs = _load_all_single_law_runs(config["max_runs_scan"], active_laws)
    run_mode_lat = compute_run_mode_latency(all_sl_runs)

    # Escalation & retry trends
    esc_trend, retry_trend = compute_rate_trends(all_sl_runs)

    # HTML-tjek per lov (structure coverage, excluding N/A)
    html_tjek_details: list[dict[str, Any]] = []
    checked_coverages: list[float] = []
    na_count = 0
    for corpus_id, meta in corpora.get("corpora", {}).items():
        quality = meta.get("quality", {})
        coverage = quality.get("structure_coverage_pct", 0.0)
        entry: dict[str, Any] = {
            "name": meta.get("display_name", corpus_id),
            "chunks": quality.get("chunk_count", 0),
            "unhandled": quality.get("unhandled_count", 0),
        }
        if coverage > 0:
            entry["html_tjek_pct"] = coverage
            checked_coverages.append(coverage)
        else:
            entry["html_tjek_pct"] = None
            na_count += 1
        html_tjek_details.append(entry)
    html_tjek_overall = round(
        statistics.mean(checked_coverages), 2
    ) if checked_coverages else 0.0

    # Per-law pass rates (SL)
    per_law: list[dict[str, Any]] = []
    for run in sl_runs:
        meta = run.get("meta", {})
        summary = run.get("summary", {})
        law_total = summary.get("total", 0)
        law_passed = summary.get("passed", 0)
        per_law.append({
            "law": meta.get("law", "unknown"),
            "pass_rate": round((law_passed / law_total * 100) if law_total else 0.0, 1),
            "passed": law_passed,
            "total": law_total,
        })

    # Failing cases with scorer failure reasons (SL + CL, capped at 15)
    failing_cases: list[dict[str, Any]] = []
    for run in sl_runs:
        law_name = run.get("meta", {}).get("law", "unknown")
        for result in run.get("results", []):
            if not result.get("passed", True):
                failed_scorers: dict[str, str] = {}
                for scorer, value in result.get("scores", {}).items():
                    if isinstance(value, dict) and not value.get("passed", True):
                        failed_scorers[scorer] = value.get("message", "")[:120]
                failing_cases.append({
                    "law": law_name,
                    "case_id": result.get("case_id", ""),
                    "failed_scorers": failed_scorers,
                })
    for case in all_cases:
        if not case.get("passed", False):
            cl_failed: dict[str, str] = {}
            for scorer, value in case.get("score_messages", {}).items():
                if value:
                    cl_failed[scorer] = str(value)[:120]
            failing_cases.append({
                "law": "cross-law",
                "case_id": case.get("case_id", ""),
                "failed_scorers": cl_failed,
            })

    return {
        "unified_pass_rate": unified,
        "health_status": health,
        "trend": trend,
        "single_law": {"total": sl_total, "passed": sl_passed, "pass_rate": sl_rate},
        "cross_law": {"total": cl_total, "passed": cl_passed, "pass_rate": cl_rate},
        "per_mode": per_mode,
        "per_difficulty": per_difficulty,
        "per_scorer": per_scorer,
        "per_law": per_law,
        "failing_cases": failing_cases[:15],
        "percentiles": percentiles,
        "sl_percentiles": sl_percentiles,
        "cl_percentiles": cl_percentiles,
        "escalation_rate": escalation_rate,
        "retry_rate": retry_rate,
        "run_mode_latency": run_mode_lat,
        "escalation_trend": esc_trend[-5:] if esc_trend else [],
        "retry_trend": retry_trend[-5:] if retry_trend else [],
        "html_tjek_overall": html_tjek_overall,
        "html_tjek_checked": len(checked_coverages),
        "html_tjek_na": na_count,
        "html_tjek_corpora": html_tjek_details,
    }


@router.post("/analyse")
async def post_analyse() -> StreamingResponse:
    """Stream AI analysis of current metrics as Server-Sent Events."""
    config = get_settings_yaml()["dashboard"]
    snapshot = _build_metrics_snapshot(config)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        try:
            async for chunk in analyse_metrics_stream(snapshot):
                yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        except Exception as exc:
            logger.exception("AI analysis stream error")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Analyse kunne ikke gennemføres'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
