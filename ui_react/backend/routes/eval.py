"""API routes for eval dashboard functionality.

Single Responsibility: Handle HTTP requests for eval operations.

Endpoints:
- GET /eval/overview - Matrix view data (laws × test_types with pass rates)
- GET /eval/runs - List historical runs
- GET /eval/runs/{run_id} - Detailed run results
- POST /eval/trigger - Trigger new eval run (SSE stream)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import AsyncGenerator

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Add backend directory and project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas import (
    EvalOverviewResponse,
    EvalLawStats,
    EvalTestTypeStats,
    EvalRunListResponse,
    EvalRunSummary,
    EvalRunDetailResponse,
    EvalCaseResult,
    TriggerEvalRequest,
    EvalCaseCreate,
    EvalCaseUpdate,
    EvalCaseResponse,
    EvalCaseListResponse,
    ExpectedBehaviorSchema,
    RunSingleCaseRequest,
    SingleCaseResultResponse,
    Reference,
)
from services.eval_cases import (
    load_cases_for_law,
    get_case_by_id,
    create_case,
    update_case,
    delete_case,
    duplicate_case,
    ValidationError as CaseValidationError,
    NotFoundError as CaseNotFoundError,
)

router = APIRouter(prefix="/eval", tags=["eval"])

# Valid test types
EVAL_TEST_TYPES = [
    "retrieval",
    "faithfulness",
    "relevancy",
    "abstention",
    "robustness",
    "multi_hop",
]

# Track running evals (in-memory, lost on server restart)
# Key: law, Value: progress dict
_running_evals: dict[str, dict] = {}


def _get_runs_dir() -> Path:
    """Get path to runs directory."""
    return PROJECT_ROOT / "runs"


def _get_evals_dir() -> Path:
    """Get path to evals directory."""
    return PROJECT_ROOT / "data" / "evals"


def _load_golden_cases(law: str) -> list[dict]:
    """Load golden cases for a law."""
    evals_dir = _get_evals_dir()

    # Try different naming conventions
    law_underscore = law.replace("-", "_")
    candidates = [
        evals_dir / f"golden_cases_{law_underscore}.yaml",
        evals_dir / f"golden_cases_{law}.yaml",
    ]

    for path in candidates:
        if path.exists():
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or []
            # Support both formats: plain list or {"cases": [...]}
            if isinstance(data, dict):
                return data.get("cases", [])
            return data

    return []


def _load_latest_run(law: str) -> dict | None:
    """Load the latest eval run for a law."""
    runs_dir = _get_runs_dir()

    # First try stable name (latest)
    stable_path = runs_dir / f"eval_{law}.json"
    if stable_path.exists():
        with open(stable_path, encoding="utf-8") as f:
            return json.load(f)

    return None


def _load_all_runs() -> list[tuple[str, dict]]:
    """Load all eval runs.

    Returns list of (run_id, data) tuples.
    """
    runs_dir = _get_runs_dir()
    runs = []

    for path in sorted(runs_dir.glob("*_eval_*.json"), reverse=True):
        # Extract run_id from filename (timestamp part)
        # Format: 20260126_145328Z_eval_ai-act.json
        run_id = path.stem  # e.g., "20260126_145328Z_eval_ai-act"

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                runs.append((run_id, data))
        except (json.JSONDecodeError, IOError):
            continue

    return runs


def _get_display_name(law: str) -> str:
    """Get display name for a law."""
    # Load from corpora.json if available
    # Try both possible locations
    corpora_paths = [
        PROJECT_ROOT / "data" / "processed" / "corpora.json",
        PROJECT_ROOT / "config" / "corpora.json",
    ]
    for corpora_path in corpora_paths:
        try:
            if corpora_path.exists():
                with open(corpora_path, encoding="utf-8") as f:
                    corpora = json.load(f)
                    if law in corpora.get("corpora", {}):
                        return corpora["corpora"][law].get("display_name", law)
        except (json.JSONDecodeError, IOError):
            continue

    # Fallback to formatted name
    return law.replace("-", " ").replace("_", " ").title()


# Map scorer names to test type columns
SCORER_TO_TEST_TYPE = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "relevancy",
    "retrieval": "retrieval",
    "article_coverage": "retrieval",
}


@router.get("/running")
async def get_running_evals():
    """Get currently running evals.

    Returns dict of law -> progress info for any evals currently in progress.
    """
    return {"running": _running_evals}


@router.get("/overview", response_model=EvalOverviewResponse)
async def get_eval_overview() -> EvalOverviewResponse:
    """Get eval overview with matrix view data.

    Returns stats for each law broken down by test type.
    Stats come from:
    - Golden case test_types (for retrieval, abstention, robustness, multi_hop)
    - Actual scorer results (for faithfulness, relevancy when run_mode includes judge)
    """
    evals_dir = _get_evals_dir()

    # Find all golden case files
    golden_files = list(evals_dir.glob("golden_cases_*.yaml"))

    laws_stats: list[EvalLawStats] = []
    total_cases_all = 0
    total_passed_all = 0

    for golden_path in sorted(golden_files):
        # Extract law name from filename
        # Format: golden_cases_ai_act.yaml -> ai-act
        law = golden_path.stem.replace("golden_cases_", "").replace("_", "-")

        # Load golden cases
        cases = _load_golden_cases(law)
        if not cases:
            continue

        # Load latest run results
        latest_run = _load_latest_run(law)

        # Build case_id -> result mapping
        results_map = {}
        if latest_run and "results" in latest_run:
            for result in latest_run["results"]:
                results_map[result.get("case_id")] = result

        # Calculate stats by test type
        # Initialize with all test types
        test_type_counts: dict[str, dict[str, int]] = {tt: {"total": 0, "passed": 0} for tt in EVAL_TEST_TYPES}

        total_cases = len(cases)
        passed_count = 0

        for case in cases:
            case_id = case.get("id", "")
            case_test_types = case.get("test_types", ["retrieval"])

            # Get result if available
            result = results_map.get(case_id, {})
            case_passed = result.get("passed", False) if result else False

            if case_passed:
                passed_count += 1

            # Update test type stats from golden case test_types
            # (for retrieval, abstention, robustness, multi_hop)
            for tt in case_test_types:
                if tt in test_type_counts and tt not in ("faithfulness", "relevancy"):
                    test_type_counts[tt]["total"] += 1
                    if case_passed:
                        test_type_counts[tt]["passed"] += 1

            # For faithfulness and relevancy, use actual scorer results
            # These scorers only run in full_with_judge mode
            if result and "scores" in result:
                scores = result.get("scores", {})
                for scorer_name, score_data in scores.items():
                    test_type = SCORER_TO_TEST_TYPE.get(scorer_name)
                    if test_type and test_type in ("faithfulness", "relevancy"):
                        test_type_counts[test_type]["total"] += 1
                        if isinstance(score_data, dict) and score_data.get("passed", False):
                            test_type_counts[test_type]["passed"] += 1

        # Build test type stats list
        by_test_type = []
        for tt in EVAL_TEST_TYPES:
            counts = test_type_counts[tt]
            if counts["total"] > 0:
                by_test_type.append(EvalTestTypeStats(
                    test_type=tt,
                    total=counts["total"],
                    passed=counts["passed"],
                    failed=counts["total"] - counts["passed"],
                    pass_rate=counts["passed"] / counts["total"] if counts["total"] > 0 else 0.0,
                ))

        # Get last run timestamp and mode
        last_run = None
        last_run_mode = None
        if latest_run and "meta" in latest_run:
            last_run = latest_run["meta"].get("timestamp")
            last_run_mode = latest_run["meta"].get("run_mode")

        pass_rate = passed_count / total_cases if total_cases > 0 else 0.0

        laws_stats.append(EvalLawStats(
            law=law,
            display_name=_get_display_name(law),
            total_cases=total_cases,
            passed=passed_count,
            failed=total_cases - passed_count,
            pass_rate=pass_rate,
            last_run=last_run,
            last_run_mode=last_run_mode,
            by_test_type=by_test_type,
        ))

        total_cases_all += total_cases
        total_passed_all += passed_count

    overall_pass_rate = total_passed_all / total_cases_all if total_cases_all > 0 else 0.0

    return EvalOverviewResponse(
        laws=laws_stats,
        test_types=EVAL_TEST_TYPES,
        total_cases=total_cases_all,
        overall_pass_rate=overall_pass_rate,
    )


@router.get("/runs", response_model=EvalRunListResponse)
async def list_eval_runs(
    law: str | None = Query(default=None, description="Filter by law"),
    limit: int = Query(default=50, description="Maximum runs to return"),
) -> EvalRunListResponse:
    """List historical eval runs."""
    all_runs = _load_all_runs()

    # Filter by law if specified
    if law:
        all_runs = [(run_id, data) for run_id, data in all_runs if data.get("meta", {}).get("law") == law]

    # Apply limit
    all_runs = all_runs[:limit]

    runs: list[EvalRunSummary] = []
    for run_id, data in all_runs:
        meta = data.get("meta", {})
        summary = data.get("summary", {})

        runs.append(EvalRunSummary(
            run_id=run_id,
            law=meta.get("law", "unknown"),
            timestamp=meta.get("timestamp", ""),
            total=summary.get("total", 0),
            passed=summary.get("passed", 0),
            failed=summary.get("failed", 0),
            pass_rate=summary.get("pass_rate", 0.0),
            duration_seconds=meta.get("duration_seconds", 0.0),
            trigger_source=meta.get("trigger_source", "cli"),
            run_mode=meta.get("run_mode", "full"),
        ))

    return EvalRunListResponse(runs=runs, total=len(runs))


@router.get("/runs/{run_id}", response_model=EvalRunDetailResponse)
async def get_eval_run_detail(run_id: str) -> EvalRunDetailResponse:
    """Get detailed results for a specific eval run."""
    runs_dir = _get_runs_dir()

    # Try to find the run file
    run_path = runs_dir / f"{run_id}.json"
    if not run_path.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    try:
        with open(run_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to load run: {e}")

    meta = data.get("meta", {})
    law = meta.get("law", "unknown")

    # Load golden cases to get test_types and prompts
    golden_cases = _load_golden_cases(law)
    cases_map = {c.get("id"): c for c in golden_cases}

    # Build case results
    results: list[EvalCaseResult] = []
    for result in data.get("results", []):
        case_id = result.get("case_id", "")
        golden_case = cases_map.get(case_id, {})

        # Determine failure reason
        failure_reason = None
        if not result.get("passed", True):
            for scorer_name, score_data in result.get("scores", {}).items():
                if isinstance(score_data, dict) and not score_data.get("passed", True):
                    failure_reason = f"{scorer_name}: {score_data.get('message', 'failed')}"
                    break

        results.append(EvalCaseResult(
            case_id=case_id,
            profile=result.get("profile", "LEGAL"),
            prompt=golden_case.get("prompt", ""),
            passed=result.get("passed", False),
            test_types=golden_case.get("test_types", ["retrieval"]),
            origin=golden_case.get("origin", "auto"),
            duration_ms=result.get("duration_ms", 0.0),
            scores=result.get("scores", {}),
            failure_reason=failure_reason,
            retry_count=result.get("retry_count", 0),
            escalated=result.get("escalated", False),
            escalation_model=result.get("escalation_model"),
        ))

    return EvalRunDetailResponse(
        run_id=run_id,
        law=law,
        timestamp=meta.get("timestamp", ""),
        duration_seconds=meta.get("duration_seconds", 0.0),
        summary=data.get("summary", {}),
        results=results,
        stage_stats=data.get("stage_stats", {}),
        retry_stats=data.get("retry_stats", {}),
        escalation_stats=data.get("escalation_stats", {}),
    )


@router.get("/definition/{law}/{case_id}")
async def get_test_definition(law: str, case_id: str) -> dict:
    """Get the definition of a specific test case.

    Returns the golden case definition from YAML.
    """
    golden_cases = _load_golden_cases(law)
    if not golden_cases:
        raise HTTPException(status_code=404, detail=f"No eval cases found for law: {law}")

    # Find the specific case
    for case in golden_cases:
        if case.get("id") == case_id:
            return {
                "id": case.get("id", ""),
                "profile": case.get("profile", "LEGAL"),
                "prompt": case.get("prompt", ""),
                "test_types": case.get("test_types", ["retrieval"]),
                "origin": case.get("origin", "auto"),
                "expected": {
                    "must_include_any_of": case.get("expected", {}).get("must_include_any_of", []),
                    "must_include_all_of": case.get("expected", {}).get("must_include_all_of", []),
                    "must_not_include_any_of": case.get("expected", {}).get("must_not_include_any_of", []),
                    "behavior": case.get("expected", {}).get("behavior"),
                },
            }

    raise HTTPException(status_code=404, detail=f"Test case not found: {case_id}")


def _run_eval_in_thread(request: TriggerEvalRequest, queue: Queue) -> None:
    """Run eval in a separate thread with parallel execution, reporting progress via queue.

    Uses same execution logic as CLI:
    - Concurrency from settings (default_concurrency: 5)
    - Phase 1: Parallel execution with primary model
    - Phase 2: Sequential escalation with fallback model (concurrency 1)
    """
    # Track running state for reconnection
    _running_evals[request.law] = {
        "law": request.law,
        "run_mode": request.run_mode,
        "stage": "starting",
        "progress": "Starter...",
        "passed": 0,
        "failed": 0,
        "total": 0,
    }

    try:
        # Import here to avoid circular imports
        from src.eval.eval_runner import evaluate_case, load_golden_cases
        from src.eval.reporters import PipelineStageStats, CaseResult
        from src.eval.scorers import Score
        from src.common.config_loader import get_settings_yaml
        import time

        # Determine skip_llm and llm_judge based on run_mode
        skip_llm = request.run_mode == "retrieval_only"
        llm_judge = request.run_mode == "full_with_judge"

        # Load concurrency and escalation settings from config
        settings = get_settings_yaml()
        eval_settings = settings.get("eval", {})
        max_workers = eval_settings.get("default_concurrency", 5)

        # Model escalation settings
        escalation_settings = eval_settings.get("model_escalation", {})
        escalation_enabled = escalation_settings.get("enabled", False) and not skip_llm
        fallback_model = escalation_settings.get("fallback_model")
        max_primary_retries = escalation_settings.get("max_primary_retries", 3)

        # Load golden cases
        queue.put(("event", {
            "type": "stage",
            "stage": "loading",
            "message": "Indlæser test cases...",
            "completed": False,
        }))

        # Find golden cases file path
        evals_dir = _get_evals_dir()
        law_underscore = request.law.replace("-", "_")
        golden_path = evals_dir / f"golden_cases_{law_underscore}.yaml"
        if not golden_path.exists():
            golden_path = evals_dir / f"golden_cases_{request.law}.yaml"

        cases = load_golden_cases(golden_path)
        if request.case_ids:
            cases = [c for c in cases if c.id in set(request.case_ids)]
        if request.limit:
            cases = cases[:request.limit]

        total = len(cases)
        queue.put(("event", {
            "type": "stage",
            "stage": "loading",
            "message": f"Indlæst {total} test cases",
            "completed": True,
        }))

        # Send start event with total and concurrency info
        queue.put(("event", {
            "type": "start",
            "law": request.law,
            "run_mode": request.run_mode,
            "total": total,
            "concurrency": max_workers,
            "escalation_enabled": escalation_enabled,
            "message": f"Kører eval for {request.law} (concurrency: {max_workers})",
        }))

        # Update running state
        _running_evals[request.law].update({
            "stage": "running",
            "total": total,
            "progress": f"Kører {total} test cases...",
        })

        # Thread-safe tracking
        results: list = []
        failures: list = []
        results_lock = threading.Lock()
        case_counter = [0]  # Mutable counter for thread-safe increment
        passed_counter = [0]
        failed_counter = [0]

        start_time = time.time()

        def process_case(case):
            """Process a single case with retry logic (thread-safe)."""
            case_start = time.time()
            attempt = 0
            last_result = None

            while attempt <= max_primary_retries:
                try:
                    result = evaluate_case(
                        case,
                        law=request.law,
                        skip_llm=skip_llm,
                        llm_judge=llm_judge,
                        llm_judge_progress_callback=None,
                    )

                    if result.passed:
                        return CaseResult(
                            case_id=result.case_id,
                            profile=result.profile,
                            passed=result.passed,
                            scores=result.scores,
                            duration_ms=(time.time() - case_start) * 1000,
                            retrieval_metrics=result.retrieval_metrics,
                            retry_count=attempt,
                            escalated=False,
                        )

                    last_result = result
                    attempt += 1

                except Exception as e:
                    last_result = CaseResult(
                        case_id=case.id,
                        profile=case.profile,
                        passed=False,
                        scores={"error": Score(passed=False, score=0.0, message=str(e))},
                        duration_ms=(time.time() - case_start) * 1000,
                        retry_count=attempt,
                    )
                    attempt += 1

            # All retries exhausted
            return CaseResult(
                case_id=last_result.case_id,
                profile=last_result.profile,
                passed=last_result.passed,
                scores=last_result.scores,
                duration_ms=(time.time() - case_start) * 1000,
                retrieval_metrics=getattr(last_result, 'retrieval_metrics', {}),
                retry_count=max_primary_retries,
                escalated=False,
            )

        def process_escalation_case(case):
            """Process a single case with fallback model (Phase 2 escalation)."""
            case_start = time.time()
            try:
                result = evaluate_case(
                    case,
                    law=request.law,
                    skip_llm=False,
                    llm_judge=llm_judge,
                    llm_judge_progress_callback=None,
                    model_override=fallback_model,
                )

                return CaseResult(
                    case_id=result.case_id,
                    profile=result.profile,
                    passed=result.passed,
                    scores=result.scores,
                    duration_ms=(time.time() - case_start) * 1000,
                    retrieval_metrics=result.retrieval_metrics,
                    retry_count=max_primary_retries,
                    escalated=True,
                    escalation_model=fallback_model,
                )

            except Exception as e:
                return CaseResult(
                    case_id=case.id,
                    profile=case.profile,
                    passed=False,
                    scores={"error": Score(passed=False, score=0.0, message=f"Escalation failed: {e}")},
                    duration_ms=(time.time() - case_start) * 1000,
                    retry_count=max_primary_retries,
                    escalated=True,
                    escalation_model=fallback_model,
                )

        # =====================================================================
        # PHASE 1: Run all cases with primary model (parallel)
        # =====================================================================
        queue.put(("event", {
            "type": "phase",
            "phase": 1,
            "message": f"Phase 1: Parallel execution ({max_workers} workers)",
        }))

        use_parallel = max_workers > 1

        if use_parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_case = {executor.submit(process_case, case): case for case in cases}

                for future in as_completed(future_to_case):
                    case = future_to_case[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = CaseResult(
                            case_id=case.id,
                            profile=case.profile,
                            passed=False,
                            scores={"error": Score(passed=False, score=0.0, message=str(e))},
                            duration_ms=0,
                        )

                    with results_lock:
                        results.append(result)
                        case_counter[0] += 1

                        if result.passed:
                            passed_counter[0] += 1
                        else:
                            failed_counter[0] += 1
                            failures.append(result)

                        # Report case result
                        queue.put(("event", {
                            "type": "case_result",
                            "case_id": case.id,
                            "case_num": case_counter[0],
                            "total": total,
                            "passed": result.passed,
                            "duration_ms": result.duration_ms,
                            "scores": {
                                name: {
                                    "passed": score.passed,
                                    "score": score.score,
                                    "message": score.message,
                                }
                                for name, score in result.scores.items()
                            },
                            "failure_reason": None if result.passed else next(
                                (f"{n}: {s.message}" for n, s in result.scores.items() if not s.passed),
                                "Unknown"
                            ),
                            "running_passed": passed_counter[0],
                            "running_failed": failed_counter[0],
                            "retry_count": getattr(result, 'retry_count', 0),
                        }))
        else:
            # Sequential execution
            for case in cases:
                result = process_case(case)
                results.append(result)
                case_counter[0] += 1

                if result.passed:
                    passed_counter[0] += 1
                else:
                    failed_counter[0] += 1
                    failures.append(result)

                queue.put(("event", {
                    "type": "case_result",
                    "case_id": case.id,
                    "case_num": case_counter[0],
                    "total": total,
                    "passed": result.passed,
                    "duration_ms": result.duration_ms,
                    "scores": {
                        name: {
                            "passed": score.passed,
                            "score": score.score,
                            "message": score.message,
                        }
                        for name, score in result.scores.items()
                    },
                    "failure_reason": None if result.passed else next(
                        (f"{n}: {s.message}" for n, s in result.scores.items() if not s.passed),
                        "Unknown"
                    ),
                    "running_passed": passed_counter[0],
                    "running_failed": failed_counter[0],
                    "retry_count": getattr(result, 'retry_count', 0),
                }))

        # Update running state after Phase 1
        _running_evals[request.law].update({
            "passed": passed_counter[0],
            "failed": failed_counter[0],
            "progress": f"{passed_counter[0]}/{total} bestået, {failed_counter[0]} fejlet",
        })

        # =====================================================================
        # PHASE 2: Escalate failed cases to fallback model (sequential)
        # =====================================================================
        if escalation_enabled and fallback_model and not skip_llm and failures:
            # Identify cases eligible for escalation (generation failures only)
            cases_to_escalate = []
            case_lookup = {c.id: c for c in cases}

            for failed_result in failures:
                faith_score = failed_result.scores.get("faithfulness")
                relevancy_score = failed_result.scores.get("answer_relevancy")
                # Only escalate generation failures, not retrieval failures
                if (faith_score and not faith_score.passed) or (relevancy_score and not relevancy_score.passed):
                    if failed_result.case_id in case_lookup:
                        cases_to_escalate.append(case_lookup[failed_result.case_id])

            if cases_to_escalate:
                queue.put(("event", {
                    "type": "phase",
                    "phase": 2,
                    "message": f"Phase 2: Escalating {len(cases_to_escalate)} failed cases to {fallback_model} (concurrency: 1)",
                }))

                for case in cases_to_escalate:
                    escalated_result = process_escalation_case(case)

                    # Find and replace the original failed result
                    for i, r in enumerate(results):
                        if r.case_id == escalated_result.case_id:
                            # Update counters
                            if not r.passed and escalated_result.passed:
                                passed_counter[0] += 1
                                failed_counter[0] -= 1
                            results[i] = escalated_result
                            break

                    # Update failures list
                    failures = [f for f in failures if f.case_id != escalated_result.case_id]
                    if not escalated_result.passed:
                        failures.append(escalated_result)

                    # Report escalation result
                    queue.put(("event", {
                        "type": "escalation_result",
                        "case_id": case.id,
                        "passed": escalated_result.passed,
                        "duration_ms": escalated_result.duration_ms,
                        "escalation_model": fallback_model,
                        "running_passed": passed_counter[0],
                        "running_failed": failed_counter[0],
                    }))

        duration = time.time() - start_time

        # Build summary and save results
        stage_stats = PipelineStageStats.from_results(results) if results else PipelineStageStats()

        # Save results to JSON file
        from src.eval.reporters import EvalSummary, JsonReporter, RetryStats, EscalationStats
        from datetime import timezone

        # Calculate retry and escalation statistics
        retry_stats = RetryStats()
        escalation_stats = EscalationStats()

        for r in results:
            retry_count = getattr(r, 'retry_count', 0)
            escalated = getattr(r, 'escalated', False)

            if retry_count > 0:
                retry_stats.cases_with_retries += 1
                retry_stats.total_retries += retry_count
                if r.passed and not escalated:
                    retry_stats.cases_passed_on_retry += 1
                elif not r.passed and not escalated:
                    retry_stats.cases_failed_after_retries += 1

            if escalated:
                escalation_stats.cases_escalated += 1
                escalation_stats.escalated_case_ids.append(r.case_id)
                if r.passed:
                    escalation_stats.cases_passed_on_escalation += 1
                else:
                    escalation_stats.cases_failed_after_escalation += 1

        summary = EvalSummary(
            law=request.law,
            total=total,
            passed=passed_counter[0],
            failed=failed_counter[0],
            skipped=0,
            duration_seconds=duration,
            results=results,
            stage_stats=stage_stats,
            escalation_stats=escalation_stats,
            retry_stats=retry_stats,
        )

        # Save to runs directory
        runs_dir = PROJECT_ROOT / "runs"
        runs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

        # Helper to save and add run_mode
        def save_with_run_mode(path: Path):
            JsonReporter(path).write(summary)
            # Add run_mode to meta
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["meta"]["run_mode"] = request.run_mode
            data["meta"]["trigger_source"] = "api"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        json_path = runs_dir / f"{timestamp}_eval_{request.law}.json"
        save_with_run_mode(json_path)

        # Also save to stable name
        stable_path = runs_dir / f"eval_{request.law}.json"
        save_with_run_mode(stable_path)

        # Send completion event with stage stats
        queue.put(("event", {
            "type": "complete",
            "law": request.law,
            "total": total,
            "passed": passed_counter[0],
            "failed": failed_counter[0],
            "pass_rate": passed_counter[0] / total if total > 0 else 0.0,
            "duration_seconds": duration,
            "concurrency": max_workers,
            "stage_stats": {
                "retrieval": {
                    "total": stage_stats.retrieval_total,
                    "passed": stage_stats.retrieval_passed,
                },
                "augmentation": {
                    "total": stage_stats.augmentation_total,
                    "passed": stage_stats.augmentation_passed,
                },
                "generation": {
                    "total": stage_stats.generation_total,
                    "passed": stage_stats.generation_passed,
                },
            },
            "retry_stats": {
                "cases_with_retries": retry_stats.cases_with_retries,
                "total_retries": retry_stats.total_retries,
                "cases_passed_on_retry": retry_stats.cases_passed_on_retry,
                "cases_failed_after_retries": retry_stats.cases_failed_after_retries,
            },
            "escalation_stats": {
                "cases_escalated": escalation_stats.cases_escalated,
                "cases_passed_on_escalation": escalation_stats.cases_passed_on_escalation,
                "cases_failed_after_escalation": escalation_stats.cases_failed_after_escalation,
            },
        }))

        queue.put(("done", None))

        # Clear running state
        _running_evals.pop(request.law, None)

    except Exception as e:
        logger.exception("Eval run failed for %s", request.law)
        queue.put(("error", str(e)))
        # Clear running state on error
        _running_evals.pop(request.law, None)


async def _generate_eval_events(request: TriggerEvalRequest) -> AsyncGenerator[str, None]:
    """Generate SSE events for eval progress."""
    queue: Queue = Queue()

    # Start eval in background thread
    thread = Thread(target=_run_eval_in_thread, args=(request, queue), daemon=True)
    thread.start()

    try:
        while True:
            try:
                msg_type, data = queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.1)
                continue

            if msg_type == "done":
                break
            elif msg_type == "error":
                error_event = {"type": "error", "error": str(data)}
                yield f"data: {json.dumps(error_event)}\n\n"
                break
            elif msg_type == "event":
                yield f"data: {json.dumps(data)}\n\n"

    except Exception as e:
        error_event = {"type": "error", "error": f"Unexpected error: {str(e)}"}
        yield f"data: {json.dumps(error_event)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/trigger")
async def trigger_eval(request: TriggerEvalRequest) -> StreamingResponse:
    """Trigger a new eval run with SSE progress streaming.

    Run modes:
    - retrieval_only: Fast mode, only tests retrieval (--skip-llm)
    - full: Standard mode with LLM generation
    - full_with_judge: Thorough mode with LLM-as-judge scoring
    """
    # Validate law exists
    golden_cases = _load_golden_cases(request.law)
    if not golden_cases:
        raise HTTPException(
            status_code=404,
            detail=f"No eval cases found for law: {request.law}"
        )

    return StreamingResponse(
        _generate_eval_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# Eval Case CRUD Endpoints
# ============================================================================


def _case_dict_to_response(case: dict) -> EvalCaseResponse:
    """Convert case dict to response model."""
    expected = case.get("expected", {})
    return EvalCaseResponse(
        id=case.get("id", ""),
        profile=case.get("profile", "LEGAL"),
        prompt=case.get("prompt", ""),
        test_types=case.get("test_types", ["retrieval"]),
        origin=case.get("origin", "auto"),
        expected=ExpectedBehaviorSchema(
            must_include_any_of=expected.get("must_include_any_of", []),
            must_include_any_of_2=expected.get("must_include_any_of_2", []),
            must_include_all_of=expected.get("must_include_all_of", []),
            must_not_include_any_of=expected.get("must_not_include_any_of", []),
            contract_check=expected.get("contract_check", False),
            min_citations=expected.get("min_citations"),
            max_citations=expected.get("max_citations"),
            behavior=expected.get("behavior", "answer"),
            allow_empty_references=expected.get("allow_empty_references", False),
            must_have_article_support_for_normative=expected.get("must_have_article_support_for_normative", True),
            notes=expected.get("notes", ""),
        ),
    )


@router.get("/cases/{law}", response_model=EvalCaseListResponse)
async def list_eval_cases(law: str) -> EvalCaseListResponse:
    """List all eval cases for a law."""
    cases = load_cases_for_law(law)
    return EvalCaseListResponse(
        cases=[_case_dict_to_response(c) for c in cases],
        total=len(cases),
    )


@router.get("/cases/{law}/{case_id}", response_model=EvalCaseResponse)
async def get_eval_case(law: str, case_id: str) -> EvalCaseResponse:
    """Get a single eval case by ID."""
    case = get_case_by_id(law, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    return _case_dict_to_response(case)


@router.post("/cases/{law}", response_model=EvalCaseResponse, status_code=201)
async def create_eval_case(law: str, case_data: EvalCaseCreate) -> EvalCaseResponse:
    """Create a new eval case."""
    try:
        # Convert Pydantic model to dict
        data = {
            "profile": case_data.profile,
            "prompt": case_data.prompt,
            "test_types": case_data.test_types,
            "expected": case_data.expected.model_dump() if case_data.expected else {},
        }
        if case_data.id:
            data["id"] = case_data.id

        created = create_case(law, data)
        return _case_dict_to_response(created)

    except CaseValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.put("/cases/{law}/{case_id}", response_model=EvalCaseResponse)
async def update_eval_case(law: str, case_id: str, case_data: EvalCaseUpdate) -> EvalCaseResponse:
    """Update an existing eval case."""
    try:
        # Build update dict with only provided fields
        data = {}
        if case_data.profile is not None:
            data["profile"] = case_data.profile
        if case_data.prompt is not None:
            data["prompt"] = case_data.prompt
        if case_data.test_types is not None:
            data["test_types"] = case_data.test_types
        if case_data.expected is not None:
            data["expected"] = case_data.expected.model_dump()

        updated = update_case(law, case_id, data)
        return _case_dict_to_response(updated)

    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    except CaseValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.delete("/cases/{law}/{case_id}", status_code=204)
async def delete_eval_case(law: str, case_id: str) -> None:
    """Delete an eval case."""
    try:
        delete_case(law, case_id)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")


@router.post("/cases/{law}/{case_id}/duplicate", response_model=EvalCaseResponse, status_code=201)
async def duplicate_eval_case(law: str, case_id: str) -> EvalCaseResponse:
    """Duplicate an eval case with a new ID."""
    try:
        duplicated = duplicate_case(law, case_id)
        return _case_dict_to_response(duplicated)
    except CaseNotFoundError:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")


# ============================================================================
# Single Case Validation (Quick Test)
# ============================================================================


@router.post("/run-single", response_model=SingleCaseResultResponse)
async def run_single_case(request: RunSingleCaseRequest) -> SingleCaseResultResponse:
    """Run a single eval case for quick validation.

    This endpoint runs ONE test case and returns detailed results including
    the actual RAG response. Results are NOT saved to the eval history.

    Use cases:
    - Validate a test case before saving
    - Debug why a test case is failing
    - Test changes to expected values

    Can use either:
    - case_id: Run an existing saved case
    - inline definition: Run with prompt/profile/test_types/expected (for unsaved cases)
    """
    import time
    from src.eval.eval_core import EvalConfig, _build_engine, _evaluate_single_case
    from src.eval.types import GoldenCase, ExpectedBehavior
    from src.eval.scorers import (
        AnchorScorer,
        PipelineBreakdownScorer,
        ContractScorer,
    )
    from src.services import ask

    start_time = time.perf_counter()

    # Build the test case
    if request.case_id:
        # Load existing case
        case_dict = get_case_by_id(request.law, request.case_id)
        if case_dict is None:
            raise HTTPException(status_code=404, detail=f"Case not found: {request.case_id}")

        expected_dict = case_dict.get("expected", {})
        golden_case = GoldenCase(
            id=case_dict.get("id", ""),
            profile=case_dict.get("profile", "LEGAL"),
            prompt=case_dict.get("prompt", ""),
            test_types=case_dict.get("test_types", ["retrieval"]),
            expected=ExpectedBehavior(
                must_include_any_of=expected_dict.get("must_include_any_of", []),
                must_include_any_of_2=expected_dict.get("must_include_any_of_2", []),
                must_include_all_of=expected_dict.get("must_include_all_of", []),
                must_not_include_any_of=expected_dict.get("must_not_include_any_of", []),
                behavior=expected_dict.get("behavior", "answer"),
                min_citations=expected_dict.get("min_citations"),
                max_citations=expected_dict.get("max_citations"),
            ),
        )
    elif request.prompt and request.profile:
        # Inline definition
        expected_dict = request.expected.model_dump() if request.expected else {}
        golden_case = GoldenCase(
            id="inline-validation",
            profile=request.profile,
            prompt=request.prompt,
            test_types=request.test_types or ["retrieval"],
            expected=ExpectedBehavior(
                must_include_any_of=expected_dict.get("must_include_any_of", []),
                must_include_any_of_2=expected_dict.get("must_include_any_of_2", []),
                must_include_all_of=expected_dict.get("must_include_all_of", []),
                must_not_include_any_of=expected_dict.get("must_not_include_any_of", []),
                behavior=expected_dict.get("behavior", "answer"),
                min_citations=expected_dict.get("min_citations"),
                max_citations=expected_dict.get("max_citations"),
            ),
        )
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either case_id or inline definition (prompt + profile)"
        )

    # Build config
    config = EvalConfig(
        law=request.law,
        run_mode=request.run_mode,
        max_retries=0,  # No retries for quick validation
        escalation_enabled=False,
    )

    try:
        # Build engine and scorers
        engine = _build_engine(config.law)

        scorers = [AnchorScorer(), PipelineBreakdownScorer()]
        if not config.skip_llm:
            scorers.append(ContractScorer())

        # Run the case
        result = _evaluate_single_case(golden_case, config, engine, scorers)

        # Now get the actual RAG response for display
        # We need to call ask() again to get the answer text
        from src.engine.planning import UserProfile
        profile = UserProfile.ENGINEERING if golden_case.profile == "ENGINEERING" else UserProfile.LEGAL

        rag_result = ask.ask(
            question=golden_case.prompt,
            law=config.law,
            user_profile=profile,
            engine=engine,
            contract_min_citations=golden_case.expected.min_citations,
            dry_run=config.skip_llm,
        )

        # Convert references to schema format
        references = []
        for i, ref in enumerate(rag_result.references_structured):
            if isinstance(ref, dict):
                references.append(Reference(
                    idx=i + 1,
                    display=ref.get("display", ""),
                    chunk_text=ref.get("chunk_text", ref.get("text", "")),
                    corpus_id=ref.get("corpus_id"),
                    article=ref.get("article"),
                    recital=ref.get("recital"),
                    annex=ref.get("annex"),
                    paragraph=ref.get("paragraph"),
                    litra=ref.get("litra"),
                ))

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Build test definition for display
        test_definition = {
            "id": golden_case.id,
            "profile": golden_case.profile,
            "prompt": golden_case.prompt,
            "test_types": golden_case.test_types,
            "expected": {
                "must_include_any_of": golden_case.expected.must_include_any_of,
                "must_include_any_of_2": golden_case.expected.must_include_any_of_2,
                "must_include_all_of": golden_case.expected.must_include_all_of,
                "must_not_include_any_of": golden_case.expected.must_not_include_any_of,
                "behavior": golden_case.expected.behavior,
                "min_citations": golden_case.expected.min_citations,
                "max_citations": golden_case.expected.max_citations,
            },
        }

        # Convert scores to serializable format
        scores_dict = {
            name: {
                "passed": score.passed,
                "score": score.score,
                "message": score.message,
            }
            for name, score in result.scores.items()
        }

        return SingleCaseResultResponse(
            passed=result.passed,
            duration_ms=duration_ms,
            scores=scores_dict,
            answer=rag_result.answer or "",
            references=references,
            test_definition=test_definition,
            retrieval_metrics=result.retrieval_metrics,
        )

    except Exception as e:
        logger.exception("Single case validation failed")
        duration_ms = (time.perf_counter() - start_time) * 1000
        return SingleCaseResultResponse(
            passed=False,
            duration_ms=duration_ms,
            scores={},
            answer="",
            references=[],
            test_definition={},
            retrieval_metrics={},
            error=str(e),
        )
