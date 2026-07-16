"""Regression baseline capture and comparison.

Captures eval suite performance as a frozen snapshot and compares future
eval runs against it to detect per-metric regressions.  Consumes existing
eval infrastructure (run_eval, EvalSummary, scorers) without modification.

Dependencies:
    src.eval.eval_runner.run_eval
    src.eval.reporters.EvalSummary, CaseResult
    src.common.config_loader.get_eval_settings
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

from src.common.config_loader import get_eval_settings
from src.eval.reporters import CaseResult, EvalSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

BASELINE_METRICS_KEYS: tuple[str, ...] = (
    "retrieval_precision",
    "faithfulness",
    "relevancy",
)


class BaselineMetrics(TypedDict):
    retrieval_precision: float
    faithfulness: float
    relevancy: float


class BaselineSnapshot(TypedDict):
    timestamp: str
    eval_config: dict[str, Any]
    metrics: BaselineMetrics
    regression_threshold_pp: int
    per_corpus: dict[str, BaselineMetrics]


class ComparisonResult(TypedDict):
    deltas: dict[str, float]
    regression_detected: bool
    threshold_pp: int
    baseline_timestamp: str
    warnings: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_project_root() -> Path:
    """Return the project root (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def _get_evals_dir() -> Path:
    """Return the evals directory."""
    return _get_project_root() / "data" / "evals"


def _discover_corpora() -> list[str]:
    """Glob golden_cases_*.yaml, extract corpus IDs, return sorted."""
    evals_dir = _get_evals_dir()
    files = sorted(evals_dir.glob("golden_cases_*.yaml"))
    corpora = []
    for f in files:
        # golden_cases_gdpr.yaml → gdpr
        name = f.stem  # golden_cases_gdpr
        corpus_id = name.removeprefix("golden_cases_")
        corpora.append(corpus_id)
    return sorted(corpora)


def _run_corpus_eval(corpus_id: str) -> EvalSummary:
    """Run eval for a single corpus. Thin wrapper around run_eval."""
    from src.eval.eval_runner import run_eval

    return run_eval(
        law=corpus_id,
        llm_judge=True,
        progress=False,
        max_retries=3,
    )


def _get_regression_threshold() -> int:
    """Read regression_threshold_pp from config, default 2."""
    settings = get_eval_settings()
    threshold = settings.get("regression_threshold_pp", 2)
    if not isinstance(threshold, int):
        msg = f"regression_threshold_pp must be an integer, got {type(threshold).__name__}: {threshold}"
        raise TypeError(msg)
    if threshold < 0:
        msg = f"regression_threshold_pp must be non-negative, got {threshold}"
        raise ValueError(msg)
    return threshold


def _get_scorer_metric_mapping() -> dict[str, str]:
    """Read scorer→metric mapping from config with defaults."""
    settings = get_eval_settings()
    return settings.get(
        "scorer_metric_mapping",
        {
            "anchor_presence": "retrieval_precision",
            "faithfulness": "faithfulness",
            "answer_relevancy": "relevancy",
        },
    )


def _get_default_baseline_path() -> Path:
    """Read eval.baseline_path from config, default data/evals/runs/baseline_pre_case_law.json."""
    settings = get_eval_settings()
    raw = settings.get("baseline_path", "data/evals/runs/baseline_pre_case_law.json")
    path = Path(raw)
    if not path.is_absolute():
        path = _get_project_root() / path
    return path


def _tally_case(
    case_result: CaseResult,
    scorer_metric_mapping: dict[str, str],
    valid_metrics: set[str],
    agg: dict[str, list[int]],
    corpus_counts: dict[str, list[int]],
) -> None:
    """Tally a single case's scores into aggregate and per-corpus counters."""
    is_abstention = "abstention" in case_result.scores
    for scorer_name, metric_key in scorer_metric_mapping.items():
        if metric_key not in valid_metrics:
            continue
        if is_abstention and metric_key in ("faithfulness", "relevancy"):
            continue
        score = case_result.scores.get(scorer_name)
        if score is None:
            continue
        val = 1 if score.passed else 0
        agg[metric_key][0] += val
        agg[metric_key][1] += 1
        corpus_counts[metric_key][0] += val
        corpus_counts[metric_key][1] += 1


def _counts_to_metrics(counts: dict[str, list[int]]) -> BaselineMetrics:
    """Convert {metric: [passed, total]} counters to BaselineMetrics."""
    return cast(
        BaselineMetrics,
        {k: _safe_rate(v[0], v[1]) for k, v in counts.items()},
    )


def _extract_metrics(
    summaries: dict[str, EvalSummary],
    scorer_metric_mapping: dict[str, str],
) -> tuple[BaselineMetrics, dict[str, BaselineMetrics]]:
    """Aggregate pass rates from EvalSummary results using scorer→metric mapping.

    Abstention cases (those with an "abstention" key in scores) are excluded
    from faithfulness and relevancy counts.

    Returns (aggregate_metrics, per_corpus_metrics).
    """
    agg: dict[str, list[int]] = {m: [0, 0] for m in BASELINE_METRICS_KEYS}
    valid_metrics = set(agg)
    per_corpus_metrics: dict[str, BaselineMetrics] = {}

    for corpus_id, summary in summaries.items():
        corpus_counts: dict[str, list[int]] = {m: [0, 0] for m in BASELINE_METRICS_KEYS}
        for case_result in summary.results:
            _tally_case(
                case_result, scorer_metric_mapping, valid_metrics, agg, corpus_counts
            )
        per_corpus_metrics[corpus_id] = _counts_to_metrics(corpus_counts)

    return _counts_to_metrics(agg), per_corpus_metrics


def _safe_rate(passed: int, total: int) -> float:
    return passed / total if total > 0 else 0.0


def _load_baseline(path: Path) -> dict[str, Any]:
    """Load and parse a baseline JSON file."""
    if not path.exists():
        msg = f"Baseline file not found: {path}"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def capture_baseline(output_path: Path | None = None) -> BaselineSnapshot:
    """Run eval suite across all corpora, save aggregate metrics as baseline.

    Args:
        output_path: Where to write the JSON. Uses config default if None.

    Returns:
        The baseline snapshot dict.
    """
    threshold = _get_regression_threshold()
    mapping = _get_scorer_metric_mapping()

    if output_path is None:
        output_path = _get_default_baseline_path()

    corpora = _discover_corpora()
    if not corpora:
        raise ValueError("No golden cases found")

    # Run eval per corpus, collecting successes
    summaries: dict[str, EvalSummary] = {}
    for corpus_id in corpora:
        try:
            summaries[corpus_id] = _run_corpus_eval(corpus_id)
        except Exception:
            logger.warning(
                "Eval failed for corpus %s, excluding from baseline",
                corpus_id,
                exc_info=True,
            )

    if not summaries:
        raise ValueError("All corpus evaluations failed; no baseline data to capture")

    aggregate, per_corpus = _extract_metrics(summaries, mapping)

    total_cases = sum(s.total for s in summaries.values())
    snapshot: BaselineSnapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_config": {
            "run_mode": "full_with_judge",
            "corpora": sorted(summaries.keys()),
            "case_count": total_cases,
        },
        "metrics": aggregate,
        "regression_threshold_pp": threshold,
        "per_corpus": per_corpus,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return snapshot


def compare_to_baseline(
    current_metrics: BaselineMetrics,
    baseline_path: Path | None = None,
) -> ComparisonResult:
    """Compare current metrics against saved baseline, detect regressions.

    Args:
        current_metrics: Dict with metric keys and float values.
        baseline_path: Path to baseline JSON. Uses config default if None.

    Returns:
        ComparisonResult with deltas, regression flag, threshold, and warnings.
    """
    if baseline_path is None:
        baseline_path = _get_default_baseline_path()

    baseline = _load_baseline(baseline_path)
    baseline_metrics: dict[str, float] = baseline["metrics"]
    threshold_pp: int = baseline["regression_threshold_pp"]
    threshold_frac = threshold_pp / 100

    deltas: dict[str, float] = {}
    warnings: list[str] = []

    # Compare metrics present in baseline
    current_dict: dict[str, float] = cast(dict[str, float], current_metrics)
    for metric, baseline_val in baseline_metrics.items():
        current_val = current_dict.get(metric, 0.0)
        if metric not in current_dict:
            warnings.append(
                f"Metric '{metric}' missing from current metrics, treated as 0.0"
            )
        deltas[metric] = current_val - float(baseline_val)

    # Metrics in current but not in baseline
    for metric, current_val in current_dict.items():
        if metric not in baseline_metrics:
            deltas[metric] = current_val
            warnings.append(
                f"Metric '{metric}' not in baseline, excluded from regression check"
            )

    # Regression: any baseline metric dropped beyond threshold (exclusive).
    # Round to 6 decimal places to avoid floating-point boundary errors.
    regression_detected = any(
        round(deltas[metric], 6) < round(-threshold_frac, 6)
        for metric in baseline_metrics
        if metric in deltas
    )

    return ComparisonResult(
        deltas=deltas,
        regression_detected=regression_detected,
        threshold_pp=threshold_pp,
        baseline_timestamp=baseline["timestamp"],
        warnings=warnings,
    )
