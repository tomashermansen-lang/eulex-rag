"""API routes for cross-law evaluation management.

Single Responsibility: Handle HTTP requests for cross-law eval suite CRUD.
Delegates business logic to CrossLawSuiteManager.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from queue import Queue, Empty
from threading import Thread
from typing import Any, AsyncGenerator
import uuid

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Add project root to path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.cross_law_suite_manager import (
    CrossLawSuiteManager,
    CrossLawEvalSuite,
    CrossLawGoldenCase,
    SuiteValidationError,
)
from src.eval.eval_case_generator import (
    GenerationRequest,
    GeneratedCase,
    generate_cross_law_cases,
    assign_test_types,
    assign_difficulty,
    CaseGenerationError,
    MAX_CASES_LIMIT,
    suggest_suite_text,
)
from src.eval.eval_core import EvalConfig, evaluate_cases_iter, _build_engine, _evaluate_single_case
from src.eval.types import GoldenCase, ExpectedBehavior
from src.eval.reporters import CaseResult, EvalSummary
from src.eval.scorers import AnchorScorer, PipelineBreakdownScorer, ContractScorer
from src.services.ask import ask as ask_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class CreateSuiteRequest(BaseModel):
    """Request to create a new cross-law eval suite."""

    name: str
    description: str = ""
    target_corpora: list[str]
    default_synthesis_mode: str = "comparison"


class SuiteResponse(BaseModel):
    """Response for a cross-law eval suite."""

    id: str
    name: str
    description: str
    target_corpora: list[str]
    default_synthesis_mode: str
    case_count: int
    created_at: str | None
    modified_at: str | None


class SuiteDetailResponse(SuiteResponse):
    """Detailed response for a cross-law eval suite including cases."""

    cases: list["CaseResponse"] = []


class CaseRequest(BaseModel):
    """Request to create/update a cross-law test case."""

    prompt: str
    corpus_scope: str = "explicit"
    target_corpora: list[str] = []
    synthesis_mode: str = "comparison"
    expected_anchors: list[str] = []
    expected_corpora: list[str] = []
    min_corpora_cited: int = 2
    profile: str = "LEGAL"
    disabled: bool = False
    # Extended fields for full eval support
    test_types: list[str] = []
    expected_behavior: str = "answer"
    must_include_any_of: list[str] = []
    must_include_any_of_2: list[str] = []
    must_include_all_of: list[str] = []
    must_not_include_any_of: list[str] = []
    contract_check: bool = False
    min_citations: int | None = None
    max_citations: int | None = None
    notes: str = ""
    # Quality metadata
    difficulty: str | None = None


class CaseResponse(BaseModel):
    """Response for a cross-law test case."""

    id: str
    prompt: str
    corpus_scope: str
    target_corpora: list[str]
    synthesis_mode: str
    expected_anchors: list[str]
    expected_corpora: list[str]
    min_corpora_cited: int | None = None
    profile: str
    disabled: bool
    origin: str
    # Extended fields for full eval support
    test_types: list[str] = []
    expected_behavior: str = "answer"
    must_include_any_of: list[str] = []
    must_include_any_of_2: list[str] = []
    must_include_all_of: list[str] = []
    must_not_include_any_of: list[str] = []
    contract_check: bool = False
    min_citations: int | None = None
    max_citations: int | None = None
    notes: str = ""
    # Quality metadata
    difficulty: str | None = None
    retrieval_confirmed: bool | None = None


class ImportRequest(BaseModel):
    """Request to import a suite from YAML."""

    yaml_content: str


class GenerateCasesRequest(BaseModel):
    """Request to generate cross-law test cases."""

    target_corpora: list[str]
    synthesis_mode: str = "comparison"
    max_cases: int = 15
    suite_id: str | None = None
    suite_name: str | None = None
    suite_description: str = ""
    synthesis_distribution: dict[str, float] | None = None
    difficulty_distribution: dict[str, float] | None = None
    generation_strategy: str = "standard"   # "standard" | "inverted"
    calibrate_anchors: bool = True          # Run retrieval probe on inverted cases


class GenerateCasesResponse(BaseModel):
    """Response for case generation."""

    suite_id: str
    case_count: int
    cases: list[CaseResponse]


class AiSuggestRequest(BaseModel):
    """Request for AI-powered name/description suggestions."""

    type: str  # "name" | "description"
    corpora: list[str]
    corpora_names: list[str] = []
    synthesis_mode: str = "comparison"


class AiSuggestResponse(BaseModel):
    """Response for AI suggestion."""

    suggestion: str


# ---------------------------------------------------------------------------
# Overview Response Models (R-UI-02: drill-down navigation)
# ---------------------------------------------------------------------------


class CrossLawSuiteStats(BaseModel):
    """Stats for a single suite in overview."""

    id: str
    name: str
    case_count: int
    passed: int
    failed: int
    pass_rate: float
    last_run: str | None = None
    last_run_mode: str | None = None
    scorer_pass_rates: dict[str, float] = {}
    default_synthesis_mode: str = "comparison"
    mode_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}


class CrossLawOverviewResponse(BaseModel):
    """Response for cross-law overview (suite matrix stats)."""

    suites: list[CrossLawSuiteStats]
    total_cases: int
    overall_pass_rate: float


class CrossLawRunSummary(BaseModel):
    """Summary of a cross-law eval run (R-UI-04)."""

    run_id: str
    suite_id: str
    timestamp: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    duration_seconds: float
    run_mode: str


class CrossLawRunListResponse(BaseModel):
    """Response for listing runs for a suite."""

    runs: list[CrossLawRunSummary]
    total: int


class CrossLawCaseResult(BaseModel):
    """Result for a single test case in a run (R-UI-05)."""

    case_id: str
    prompt: str
    synthesis_mode: str
    target_corpora: list[str]
    passed: bool
    duration_ms: float
    scores: dict[str, Any]
    error: str | None = None
    difficulty: str | None = None


class CrossLawRunDetailResponse(BaseModel):
    """Detailed response for a specific run."""

    run_id: str
    suite_id: str
    timestamp: str
    duration_seconds: float
    total: int
    passed: int
    failed: int
    pass_rate: float
    run_mode: str
    results: list[CrossLawCaseResult]


class TriggerCrossLawEvalRequest(BaseModel):
    """Request to trigger a cross-law evaluation run (R-UI-06)."""

    run_mode: str = "retrieval_only"
    case_ids: list[str] | None = None
    max_retries: int = Field(default=0, ge=0, le=5)


class RunSingleCrossLawRequest(BaseModel):
    """Request to run a single cross-law case for validation (R-ED-15)."""

    prompt: str
    profile: str = "LEGAL"
    test_types: list[str] = []
    run_mode: str = "retrieval_only"
    expected_behavior: str = "answer"
    expected_corpora: list[str] = []
    min_corpora_cited: int = 2
    must_include_any_of: list[str] = []
    must_include_any_of_2: list[str] = []
    must_include_all_of: list[str] = []
    must_not_include_any_of: list[str] = []
    contract_check: bool = False
    min_citations: int | None = None
    max_citations: int | None = None


class CrossLawReference(BaseModel):
    """A source reference from cross-law RAG response."""

    idx: int
    display: str
    chunk_text: str = ""
    corpus_id: str | None = None
    article: str | None = None
    recital: str | None = None
    annex: str | None = None
    paragraph: str | None = None
    litra: str | None = None


class RunSingleCrossLawResponse(BaseModel):
    """Response from running a single cross-law case (R-ED-14)."""

    passed: bool
    duration_ms: float
    scores: dict[str, Any] = {}
    answer: str = ""
    references: list[CrossLawReference] = []
    error: str | None = None


# ---------------------------------------------------------------------------
# Service Layer
# ---------------------------------------------------------------------------


class CrossLawEvalService:
    """Service layer for cross-law eval management.

    Wraps CrossLawSuiteManager with HTTP-friendly interface.
    """

    def __init__(
        self,
        evals_dir: Path,
        valid_corpus_ids: set[str],
    ):
        """Initialize service.

        Args:
            evals_dir: Directory for storing suite YAML files
            valid_corpus_ids: Set of valid corpus IDs for validation
        """
        self.manager = CrossLawSuiteManager(
            evals_dir=evals_dir,
            valid_corpus_ids=valid_corpus_ids,
        )
        self.valid_corpus_ids = valid_corpus_ids
        self.runs_dir = evals_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def list_suites(self) -> list[SuiteResponse]:
        """List all cross-law eval suites."""
        suites = self.manager.list_suites()
        return [self._suite_to_response(s) for s in suites]

    def get_overview(self) -> CrossLawOverviewResponse:
        """Get overview stats for all suites (R-UI-02: drill-down navigation)."""
        suites = self.manager.list_suites()

        # Single pass: read all run files once, group by suite_id
        runs_by_suite: dict[str, list[dict]] = {}
        for run_file in self.runs_dir.glob("*.json"):
            try:
                run_data = json.loads(run_file.read_text())
                sid = run_data.get("suite_id")
                if sid:
                    runs_by_suite.setdefault(sid, []).append(run_data)
            except (json.JSONDecodeError, OSError):
                continue

        suite_stats = []
        total_cases = 0

        overall_passed = 0
        overall_failed = 0

        for suite in suites:
            case_count = len(suite.cases)
            total_cases += case_count

            suite_runs = runs_by_suite.get(suite.id, [])

            if suite_runs:
                latest = max(suite_runs, key=lambda r: r.get("timestamp", ""))
                passed = latest.get("passed", 0)
                failed = latest.get("failed", 0)
                pass_rate = latest.get("pass_rate", 0.0)
                last_run = latest.get("timestamp")
                last_run_mode = latest.get("run_mode")
                overall_passed += passed
                overall_failed += failed

                # Aggregate per-scorer pass rates from latest run results
                scorer_pass_rates = self._compute_scorer_pass_rates(
                    latest.get("results", [])
                )
            else:
                passed = 0
                failed = 0
                pass_rate = 0.0
                last_run = None
                last_run_mode = None
                scorer_pass_rates = {}

            # Compute mode and difficulty counts from cases
            mode_counts: dict[str, int] = {}
            difficulty_counts: dict[str, int] = {}
            for c in suite.cases:
                mode_counts[c.synthesis_mode] = mode_counts.get(c.synthesis_mode, 0) + 1
                if c.difficulty:
                    difficulty_counts[c.difficulty] = difficulty_counts.get(c.difficulty, 0) + 1

            stats = CrossLawSuiteStats(
                id=suite.id,
                name=suite.name,
                case_count=case_count,
                passed=passed,
                failed=failed,
                pass_rate=pass_rate,
                last_run=last_run,
                last_run_mode=last_run_mode,
                scorer_pass_rates=scorer_pass_rates,
                default_synthesis_mode=suite.default_synthesis_mode,
                mode_counts=mode_counts,
                difficulty_counts=difficulty_counts,
            )
            suite_stats.append(stats)

        overall_total = overall_passed + overall_failed
        overall_pass_rate = (overall_passed / overall_total) if overall_total > 0 else 0.0

        return CrossLawOverviewResponse(
            suites=suite_stats,
            total_cases=total_cases,
            overall_pass_rate=overall_pass_rate,
        )

    @staticmethod
    def _compute_scorer_pass_rates(results: list[dict]) -> dict[str, float]:
        """Compute per-scorer pass rates from run results.

        Args:
            results: List of case result dicts with scores

        Returns:
            Dict mapping scorer name to pass rate (0.0-1.0)
        """
        scorer_counts: dict[str, int] = {}
        scorer_passed: dict[str, int] = {}

        for case_result in results:
            for scorer_name, score_data in case_result.get("scores", {}).items():
                scorer_counts[scorer_name] = scorer_counts.get(scorer_name, 0) + 1
                if isinstance(score_data, dict) and score_data.get("passed"):
                    scorer_passed[scorer_name] = scorer_passed.get(scorer_name, 0) + 1

        return {
            name: scorer_passed.get(name, 0) / count
            for name, count in scorer_counts.items()
            if count > 0
        }

    def list_runs(self, suite_id: str) -> CrossLawRunListResponse:
        """List runs for a suite (R-UI-04: run history)."""
        suite = self.manager.get_suite(suite_id)
        if suite is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

        runs = []
        for run_file in self.runs_dir.glob("*.json"):
            try:
                run_data = json.loads(run_file.read_text())
                if run_data.get("suite_id") == suite_id:
                    runs.append(
                        CrossLawRunSummary(
                            run_id=run_data["run_id"],
                            suite_id=run_data["suite_id"],
                            timestamp=run_data["timestamp"],
                            total=run_data["total"],
                            passed=run_data["passed"],
                            failed=run_data["failed"],
                            pass_rate=run_data["pass_rate"],
                            duration_seconds=run_data["duration_seconds"],
                            run_mode=run_data["run_mode"],
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                continue

        runs.sort(key=lambda r: r.timestamp, reverse=True)
        return CrossLawRunListResponse(runs=runs, total=len(runs))

    def get_run(self, run_id: str) -> CrossLawRunDetailResponse:
        """Get detailed results for a specific run (R-UI-05: test case results)."""
        run_file = self.runs_dir / f"{run_id}.json"
        if not run_file.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        try:
            run_data = json.loads(run_file.read_text())
        except (json.JSONDecodeError, KeyError):
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        return CrossLawRunDetailResponse(
            run_id=run_data["run_id"],
            suite_id=run_data["suite_id"],
            timestamp=run_data["timestamp"],
            duration_seconds=run_data["duration_seconds"],
            total=run_data["total"],
            passed=run_data["passed"],
            failed=run_data["failed"],
            pass_rate=run_data["pass_rate"],
            run_mode=run_data["run_mode"],
            results=[
                CrossLawCaseResult(**r) for r in run_data.get("results", [])
            ],
        )

    def _persist_run(self, run_data: dict) -> None:
        """Persist run data to JSON file."""
        run_file = self.runs_dir / f"{run_data['run_id']}.json"
        run_file.write_text(json.dumps(run_data, ensure_ascii=False, indent=2))

    def _convert_case_to_golden(self, case: CrossLawGoldenCase) -> GoldenCase:
        """Convert CrossLawGoldenCase to eval_core GoldenCase."""
        # Fall back to default test_types for backward compat with old cases
        if case.test_types:
            test_types = case.test_types
        elif case.corpus_scope != "single":
            # Cross-law cases: standard scorers + cross-law specific scorers
            cross_law_types = ["retrieval", "faithfulness", "relevancy", "corpus_coverage"]
            if case.synthesis_mode == "comparison":
                cross_law_types.append("comparison_completeness")
            elif case.synthesis_mode == "routing":
                cross_law_types.append("routing_precision")
            elif case.synthesis_mode == "discovery":
                pass  # corpus_coverage (already in base list) handles discovery via threshold
            else:
                cross_law_types.append("synthesis_balance")
            test_types = tuple(cross_law_types)
        else:
            test_types = ("retrieval", "faithfulness", "relevancy")

        # Fall back to expected_anchors when must_include_any_of is empty (backward compat)
        raw_anchors = (
            list(case.must_include_any_of) if case.must_include_any_of
            else list(case.expected_anchors)
        )
        must_include_any_of = raw_anchors

        return GoldenCase(
            id=case.id,
            profile=case.profile,
            prompt=case.prompt,
            expected=ExpectedBehavior(
                behavior=case.expected_behavior,
                must_include_any_of=must_include_any_of,
                must_include_any_of_2=list(case.must_include_any_of_2),
                must_include_all_of=list(case.must_include_all_of),
                must_not_include_any_of=list(case.must_not_include_any_of),
                contract_check=case.contract_check,
                min_citations=case.min_citations,
                max_citations=case.max_citations,
                min_corpora_cited=case.min_corpora_cited,
                required_corpora=tuple(case.expected_corpora),
            ),
            test_types=test_types,
            corpus_scope=case.corpus_scope,
            target_corpora=tuple(case.target_corpora),
            synthesis_mode=case.synthesis_mode,
        )

    async def trigger_eval(
        self, suite_id: str, request: TriggerCrossLawEvalRequest
    ) -> AsyncGenerator[str, None]:
        """Trigger evaluation run with SSE progress (R-UI-06: run triggering)."""
        suite = self.manager.get_suite(suite_id)
        if suite is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now(timezone.utc).isoformat()
        case_count = len(suite.cases)

        # Emit start event
        start_event = {
            "type": "start",
            "suite_id": suite_id,
            "total": case_count,
            "run_mode": request.run_mode,
        }
        yield f"data: {json.dumps(start_event)}\n\n"

        if case_count == 0:
            # No cases to evaluate
            run_data = {
                "run_id": run_id,
                "suite_id": suite_id,
                "timestamp": timestamp,
                "total": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "duration_seconds": 0.0,
                "run_mode": request.run_mode,
                "max_retries": request.max_retries,
                "results": [],
            }
            self._persist_run(run_data)
            yield f"data: {json.dumps({'type': 'complete', **{k: v for k, v in run_data.items() if k != 'results'}})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Convert cases to GoldenCase for eval_core
        golden_cases = [self._convert_case_to_golden(c) for c in suite.cases]

        # Use first corpus as the "law" for EvalConfig
        primary_law = suite.target_corpora[0] if suite.target_corpora else "unknown"
        config = EvalConfig(
            law=primary_law,
            run_mode=request.run_mode,
            max_retries=request.max_retries,
        )

        # Pre-build case lookup dict for O(1) access in result handler
        case_map = {c.id: c for c in suite.cases}

        # Run evaluation in a background thread so SSE chunks can flush
        # (evaluate_cases_iter is sync and would block the event loop)
        # Uses same proven pattern as single-law eval: threading.Queue + polling
        q: Queue = Queue()

        def _run_eval() -> None:
            try:
                for item in evaluate_cases_iter(golden_cases, config):
                    q.put(("item", item))
                q.put(("done", None))
            except Exception as exc:
                q.put(("error", exc))

        thread = Thread(target=_run_eval, daemon=True)
        thread.start()

        case_results: list[dict] = []
        passed_count = 0
        failed_count = 0

        while True:
            try:
                msg_type, item = q.get_nowait()
            except Empty:
                await asyncio.sleep(0.1)
                continue

            if msg_type == "done":
                break
            if msg_type == "error":
                logger.error("Cross-law eval error: %s", item)
                yield f"data: {json.dumps({'type': 'error', 'message': str(item)})}\n\n"
                break

            if isinstance(item, CaseResult):
                # Stream case result (O(1) lookup via case_map)
                matched_case = case_map.get(item.case_id)
                result_data = {
                    "case_id": item.case_id,
                    "prompt": matched_case.prompt if matched_case else "",
                    "synthesis_mode": matched_case.synthesis_mode if matched_case else "",
                    "target_corpora": list(matched_case.target_corpora) if matched_case else [],
                    "passed": item.passed,
                    "duration_ms": item.duration_ms,
                    "scores": {
                        k: {"passed": v.passed, "score": v.score, "message": v.message}
                        for k, v in item.scores.items()
                    },
                    "difficulty": matched_case.difficulty if matched_case else None,
                }
                case_results.append(result_data)

                if item.passed:
                    passed_count += 1
                else:
                    failed_count += 1

                event = {"type": "case_result", **result_data}
                yield f"data: {json.dumps(event)}\n\n"

            elif isinstance(item, EvalSummary):
                # Persist run
                pass_rate = (
                    passed_count / (passed_count + failed_count)
                    if (passed_count + failed_count) > 0
                    else 0.0
                )
                run_data = {
                    "run_id": run_id,
                    "suite_id": suite_id,
                    "timestamp": timestamp,
                    "total": passed_count + failed_count,
                    "passed": passed_count,
                    "failed": failed_count,
                    "pass_rate": pass_rate,
                    "duration_seconds": item.duration_seconds,
                    "run_mode": request.run_mode,
                    "max_retries": request.max_retries,
                    "results": case_results,
                }
                self._persist_run(run_data)

                complete_event = {
                    "type": "complete",
                    "suite_id": suite_id,
                    "run_id": run_id,
                    "total": passed_count + failed_count,
                    "passed": passed_count,
                    "failed": failed_count,
                    "pass_rate": pass_rate,
                    "duration_seconds": item.duration_seconds,
                }
                yield f"data: {json.dumps(complete_event)}\n\n"

        yield "data: [DONE]\n\n"

    def run_single(
        self, suite_id: str, request: RunSingleCrossLawRequest
    ) -> RunSingleCrossLawResponse:
        """Run a single cross-law case for in-memory validation (R-ED-15)."""
        import time

        suite = self.manager.get_suite(suite_id)
        if suite is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

        start_time = time.perf_counter()

        # Build GoldenCase from request
        test_types = tuple(request.test_types) if request.test_types else ("retrieval", "faithfulness", "relevancy")
        golden_case = GoldenCase(
            id="inline-validation",
            profile=request.profile,
            prompt=request.prompt,
            expected=ExpectedBehavior(
                behavior=request.expected_behavior,
                must_include_any_of=request.must_include_any_of,
                must_include_any_of_2=request.must_include_any_of_2,
                must_include_all_of=request.must_include_all_of,
                must_not_include_any_of=request.must_not_include_any_of,
                contract_check=request.contract_check,
                min_citations=request.min_citations,
                max_citations=request.max_citations,
                min_corpora_cited=request.min_corpora_cited,
                required_corpora=tuple(request.expected_corpora),
            ),
            test_types=test_types,
            corpus_scope="explicit",
            target_corpora=suite.target_corpora,
            synthesis_mode=suite.default_synthesis_mode,
        )

        # Build config
        primary_law = suite.target_corpora[0] if suite.target_corpora else "unknown"
        config = EvalConfig(
            law=primary_law,
            run_mode=request.run_mode,
            max_retries=0,
            escalation_enabled=False,
        )

        try:
            engine = _build_engine(config.law)
            scorers = [AnchorScorer(), PipelineBreakdownScorer()]
            if not config.skip_llm:
                scorers.append(ContractScorer())

            result = _evaluate_single_case(golden_case, config, engine, scorers)

            # Convert references from the scored result (no second ask.ask() call)
            references = []
            for i, ref in enumerate(result.references_structured):
                if isinstance(ref, dict):
                    references.append(CrossLawReference(
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

            scores_dict = {
                name: {"passed": score.passed, "score": score.score, "message": score.message}
                for name, score in result.scores.items()
            }

            return RunSingleCrossLawResponse(
                passed=result.passed,
                duration_ms=duration_ms,
                scores=scores_dict,
                answer=result.answer,
                references=references,
            )

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return RunSingleCrossLawResponse(
                passed=False,
                duration_ms=duration_ms,
                error=str(e),
            )

    def get_suite(self, suite_id: str) -> SuiteDetailResponse | None:
        """Get a specific suite with cases."""
        suite = self.manager.get_suite(suite_id)
        if suite is None:
            return None
        return self._suite_to_detail_response(suite)

    def create_suite(self, request: CreateSuiteRequest) -> SuiteResponse:
        """Create a new suite."""
        # Validate corpus IDs
        self._validate_corpora(request.target_corpora)

        suite_id = self._generate_suite_id(request.name)
        suite = CrossLawEvalSuite(
            id=suite_id,
            name=request.name,
            description=request.description,
            target_corpora=tuple(request.target_corpora),
            default_synthesis_mode=request.default_synthesis_mode,
            cases=(),
        )

        try:
            self.manager.create_suite(suite)
        except SuiteValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return self.get_suite(suite_id)

    def update_suite(self, suite_id: str, request: CreateSuiteRequest) -> SuiteResponse:
        """Update an existing suite."""
        existing = self.manager.get_suite(suite_id)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

        # Validate corpus IDs
        self._validate_corpora(request.target_corpora)

        updated = CrossLawEvalSuite(
            id=suite_id,
            name=request.name,
            description=request.description,
            target_corpora=tuple(request.target_corpora),
            default_synthesis_mode=request.default_synthesis_mode,
            cases=existing.cases,
            created_at=existing.created_at,
            modified_at=existing.modified_at,
        )

        try:
            self.manager.update_suite(updated)
        except SuiteValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return self.get_suite(suite_id)

    def delete_suite(self, suite_id: str) -> None:
        """Delete a suite."""
        try:
            self.manager.delete_suite(suite_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

    def add_case(self, suite_id: str, request: CaseRequest) -> CaseResponse:
        """Add a case to a suite."""
        suite = self.manager.get_suite(suite_id)
        if suite is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

        case_id = self._generate_case_id(request.prompt)
        case = self._request_to_case(case_id, request)

        try:
            self.manager.add_case(suite_id, case)
        except SuiteValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return self._case_to_response(case)

    def update_case(
        self, suite_id: str, case_id: str, request: CaseRequest
    ) -> CaseResponse:
        """Update a case in a suite."""
        suite = self.manager.get_suite(suite_id)
        if suite is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

        # Check case exists
        existing = next((c for c in suite.cases if c.id == case_id), None)
        if existing is None:
            raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

        case = self._request_to_case(case_id, request, origin="manual")

        try:
            self.manager.update_case(suite_id, case)
        except SuiteValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return self._case_to_response(case)

    def delete_case(self, suite_id: str, case_id: str) -> None:
        """Delete a case from a suite."""
        suite = self.manager.get_suite(suite_id)
        if suite is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

        self.manager.delete_case(suite_id, case_id)

    def duplicate_case(self, suite_id: str, case_id: str) -> CaseResponse:
        """Duplicate a case in a suite."""
        suite = self.manager.get_suite(suite_id)
        if suite is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

        try:
            new_id = self.manager.duplicate_case(suite_id, case_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

        # Get the duplicated case
        updated_suite = self.manager.get_suite(suite_id)
        new_case = next((c for c in updated_suite.cases if c.id == new_id), None)
        return self._case_to_response(new_case)

    def export_yaml(self, suite_id: str) -> str:
        """Export a suite to YAML."""
        try:
            return self.manager.export_yaml(suite_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

    def import_yaml(self, yaml_content: str) -> SuiteResponse:
        """Import a suite from YAML."""
        try:
            suite = self.manager.import_yaml(yaml_content)
        except SuiteValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Save the imported suite
        try:
            self.manager.create_suite(suite)
        except SuiteValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

        return self._suite_to_response(suite)

    async def generate_cases(
        self, request: GenerateCasesRequest
    ) -> GenerateCasesResponse:
        """Generate cross-law test cases and persist to suite.

        Creates a new suite or adds to existing one.
        """
        # Validate corpora
        self._validate_corpora(request.target_corpora)

        if len(request.target_corpora) < 2:
            raise HTTPException(
                status_code=400,
                detail="Cross-law generation requires at least 2 corpora",
            )

        # Cap max_cases
        max_cases = min(request.max_cases, MAX_CASES_LIMIT)

        # Build generation request
        gen_request = GenerationRequest(
            target_corpora=tuple(request.target_corpora),
            synthesis_mode=request.synthesis_mode,
            max_cases=max_cases,
            generation_strategy=request.generation_strategy,
        )

        # Build corpus metadata from valid IDs
        corpus_metadata = {
            cid: {"name": cid, "fullname": cid}
            for cid in request.target_corpora
        }

        # Generate cases
        try:
            generated = await generate_cross_law_cases(gen_request, corpus_metadata)
        except CaseGenerationError as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Assign test types (mode-based routing)
        generated = assign_test_types(generated)

        # Create or get existing suite
        if request.suite_id:
            suite = self.manager.get_suite(request.suite_id)
            if suite is None:
                raise HTTPException(
                    status_code=404, detail=f"Suite not found: {request.suite_id}"
                )
            suite_id = request.suite_id
        else:
            suite_name = request.suite_name or f"Cross-Law {request.synthesis_mode}"
            suite_id = self._generate_suite_id(suite_name)
            suite = CrossLawEvalSuite(
                id=suite_id,
                name=suite_name,
                description=request.suite_description,
                target_corpora=tuple(request.target_corpora),
                default_synthesis_mode=request.synthesis_mode,
                cases=(),
            )
            try:
                self.manager.create_suite(suite)
            except SuiteValidationError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Add generated cases to suite
        is_inverted = request.generation_strategy == "inverted"
        case_responses = []
        for gen_case in generated:
            # Discovery: don't set min_corpora_cited — let corpus_coverage_threshold handle it.
            # Other modes: allow one miss (len - 1, min 1).
            is_discovery = gen_case.synthesis_mode == "discovery"
            n_expected = len(gen_case.expected_corpora)
            min_corpora = None if is_discovery else max(1, n_expected - 1)

            # Inverted generation: anchors are reliable (from seed articles).
            # Standard generation: strip unreliable LLM-guessed anchors.
            anchors = gen_case.expected_anchors if is_inverted else ()

            case = CrossLawGoldenCase(
                id=gen_case.id,
                prompt=gen_case.prompt,
                corpus_scope="explicit",
                target_corpora=tuple(request.target_corpora),
                synthesis_mode=gen_case.synthesis_mode,
                expected_anchors=anchors,
                expected_corpora=gen_case.expected_corpora,
                min_corpora_cited=min_corpora,
                profile="LEGAL",
                disabled=False,
                origin="auto-generated",
                difficulty=assign_difficulty(gen_case),
                test_types=gen_case.test_types,
                must_include_any_of=anchors,
            )

            # Retrieval calibration: verify anchors appear in actual retrieval
            if is_inverted and anchors and request.calibrate_anchors:
                case = await self._calibrate_case(case)

            try:
                self.manager.add_case(suite_id, case)
            except SuiteValidationError:
                pass  # Skip invalid cases
            case_responses.append(self._case_to_response(case))

        return GenerateCasesResponse(
            suite_id=suite_id,
            case_count=len(case_responses),
            cases=case_responses,
        )

    def ai_suggest(self, request: AiSuggestRequest) -> AiSuggestResponse:
        """Generate AI-powered name or description suggestion."""
        suggestion = suggest_suite_text(
            suggest_type=request.type,
            corpora_ids=request.corpora,
            corpora_names=request.corpora_names or None,
            synthesis_mode=request.synthesis_mode,
        )
        return AiSuggestResponse(suggestion=suggestion)

    # -----------------------------------------------------------------------
    # Retrieval Calibration
    # -----------------------------------------------------------------------

    async def _calibrate_case(
        self, case: CrossLawGoldenCase,
    ) -> CrossLawGoldenCase:
        """Verify expected anchors appear in actual retrieval results.

        Runs ask(dry_run=True) — same retrieval path as eval (EVAL = PROD).
        If anchors are found in top_k, marks retrieval_confirmed=True.
        If not found, clears anchors and marks retrieval_confirmed=False.
        On exception, keeps anchors optimistically with retrieval_confirmed=None.

        Args:
            case: Golden case with expected_anchors to verify

        Returns:
            Updated case with retrieval_confirmed set
        """
        from dataclasses import replace

        if not case.expected_anchors:
            return case

        try:
            result = ask_service(
                question=case.prompt,
                law=case.target_corpora[0] if case.target_corpora else "",
                dry_run=True,
                corpus_scope="explicit",
                target_corpora=list(case.target_corpora),
            )

            # Extract anchors found in retrieval
            run_meta = result.retrieval_metrics.get("run") or {}
            found_anchors = set(run_meta.get("anchors_in_top_k") or [])

            # Also check retrieved_metadatas for article mentions
            for meta in (result.retrieval_metrics.get("retrieved_metadatas") or []):
                if not isinstance(meta, dict):
                    continue
                corpus_id = meta.get("corpus_id", "")
                if meta.get("article"):
                    found_anchors.add(f"article:{meta['article']}")
                    if corpus_id:
                        found_anchors.add(f"{corpus_id}:article:{meta['article']}")
                if meta.get("annex"):
                    found_anchors.add(f"annex:{meta['annex']}")
                    if corpus_id:
                        found_anchors.add(f"{corpus_id}:annex:{meta['annex']}")

            # Check if ANY expected anchor was found
            expected_set = set(case.expected_anchors)
            matched = expected_set & found_anchors

            if matched:
                return replace(case, retrieval_confirmed=True)
            else:
                logger.info(
                    "Calibration: no anchors found for case %s (expected %s, found %s)",
                    case.id, expected_set, found_anchors,
                )
                return replace(case, retrieval_confirmed=False)

        except Exception as e:
            logger.warning("Calibration failed for case %s: %s", case.id, e)
            return replace(case, retrieval_confirmed=None)

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    def _validate_corpora(self, corpora: list[str]) -> None:
        """Validate corpus IDs."""
        invalid = set(corpora) - self.valid_corpus_ids
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown corpus IDs: {', '.join(sorted(invalid))}. "
                f"Valid: {', '.join(sorted(self.valid_corpus_ids))}",
            )

    def _suite_to_response(self, suite: CrossLawEvalSuite) -> SuiteResponse:
        """Convert suite to response model."""
        return SuiteResponse(
            id=suite.id,
            name=suite.name,
            description=suite.description,
            target_corpora=list(suite.target_corpora),
            default_synthesis_mode=suite.default_synthesis_mode,
            case_count=len(suite.cases),
            created_at=suite.created_at,
            modified_at=suite.modified_at,
        )

    def _suite_to_detail_response(self, suite: CrossLawEvalSuite) -> SuiteDetailResponse:
        """Convert suite to detail response model including cases."""
        return SuiteDetailResponse(
            id=suite.id,
            name=suite.name,
            description=suite.description,
            target_corpora=list(suite.target_corpora),
            default_synthesis_mode=suite.default_synthesis_mode,
            case_count=len(suite.cases),
            created_at=suite.created_at,
            modified_at=suite.modified_at,
            cases=[self._case_to_response(c) for c in suite.cases],
        )

    def _case_to_response(self, case: CrossLawGoldenCase) -> CaseResponse:
        """Convert case to response model."""
        return CaseResponse(
            id=case.id,
            prompt=case.prompt,
            corpus_scope=case.corpus_scope,
            target_corpora=list(case.target_corpora),
            synthesis_mode=case.synthesis_mode,
            expected_anchors=list(case.expected_anchors),
            expected_corpora=list(case.expected_corpora),
            min_corpora_cited=case.min_corpora_cited,
            profile=case.profile,
            disabled=case.disabled,
            origin=case.origin,
            test_types=list(case.test_types),
            expected_behavior=case.expected_behavior,
            must_include_any_of=list(case.must_include_any_of),
            must_include_any_of_2=list(case.must_include_any_of_2),
            must_include_all_of=list(case.must_include_all_of),
            must_not_include_any_of=list(case.must_not_include_any_of),
            contract_check=case.contract_check,
            min_citations=case.min_citations,
            max_citations=case.max_citations,
            notes=case.notes,
            difficulty=case.difficulty,
            retrieval_confirmed=case.retrieval_confirmed,
        )

    def _request_to_case(
        self, case_id: str, request: CaseRequest, origin: str = "manual"
    ) -> CrossLawGoldenCase:
        """Convert request to case."""
        return CrossLawGoldenCase(
            id=case_id,
            prompt=request.prompt,
            corpus_scope=request.corpus_scope,
            target_corpora=tuple(request.target_corpora),
            synthesis_mode=request.synthesis_mode,
            expected_anchors=tuple(request.expected_anchors),
            expected_corpora=tuple(request.expected_corpora),
            min_corpora_cited=request.min_corpora_cited,
            profile=request.profile,
            disabled=request.disabled,
            origin=origin,
            test_types=tuple(request.test_types),
            expected_behavior=request.expected_behavior,
            must_include_any_of=tuple(request.must_include_any_of),
            must_include_any_of_2=tuple(request.must_include_any_of_2),
            must_include_all_of=tuple(request.must_include_all_of),
            must_not_include_any_of=tuple(request.must_not_include_any_of),
            contract_check=request.contract_check,
            min_citations=request.min_citations,
            max_citations=request.max_citations,
            notes=request.notes,
            difficulty=request.difficulty,
        )

    def _generate_suite_id(self, name: str) -> str:
        """Generate a unique suite ID from name."""
        base = name.lower().replace(" ", "_").replace("-", "_")
        base = "".join(c for c in base if c.isalnum() or c == "_")
        return f"{base}_{uuid.uuid4().hex[:8]}"

    def _generate_case_id(self, prompt: str) -> str:
        """Generate a unique case ID from prompt."""
        words = prompt.lower().split()[:5]
        base = "_".join(w for w in words if w.isalnum())
        return f"case_{base}_{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Router Factory
# ---------------------------------------------------------------------------


def create_router(service: CrossLawEvalService) -> APIRouter:
    """Create router with injected service."""
    router = APIRouter(tags=["eval-cross-law"])

    # --- Overview (R-UI-02: drill-down navigation) ---

    @router.get("/overview", response_model=CrossLawOverviewResponse)
    def get_overview():
        """Get overview stats for suite matrix."""
        return service.get_overview()

    # --- Suite CRUD ---

    @router.post("/suites", response_model=SuiteResponse, status_code=201)
    def create_suite(request: CreateSuiteRequest):
        """Create a new cross-law eval suite (R6.1)."""
        return service.create_suite(request)

    @router.get("/suites", response_model=list[SuiteResponse])
    def list_suites():
        """List all cross-law eval suites."""
        return service.list_suites()

    @router.get("/suites/{suite_id}", response_model=SuiteDetailResponse)
    def get_suite(suite_id: str):
        """Get a specific suite with cases."""
        result = service.get_suite(suite_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")
        return result

    @router.put("/suites/{suite_id}", response_model=SuiteResponse)
    def update_suite(suite_id: str, request: CreateSuiteRequest):
        """Update a suite's metadata (R8.1)."""
        return service.update_suite(suite_id, request)

    @router.delete("/suites/{suite_id}", status_code=204)
    def delete_suite(suite_id: str):
        """Delete a suite (R8.3)."""
        service.delete_suite(suite_id)
        return Response(status_code=204)

    # --- Runs (R-UI-04: run history) ---

    @router.get("/suites/{suite_id}/runs", response_model=CrossLawRunListResponse)
    def list_runs(suite_id: str):
        """List run history for a suite."""
        return service.list_runs(suite_id)

    @router.get("/runs/{run_id}", response_model=CrossLawRunDetailResponse)
    def get_run(run_id: str):
        """Get detailed results for a specific run."""
        return service.get_run(run_id)

    @router.post("/suites/{suite_id}/run")
    async def trigger_eval(suite_id: str, request: TriggerCrossLawEvalRequest):
        """Trigger evaluation run with SSE progress (R-UI-06)."""
        # Check suite exists first (before starting stream)
        if service.manager.get_suite(suite_id) is None:
            raise HTTPException(status_code=404, detail=f"Suite not found: {suite_id}")

        return StreamingResponse(
            service.trigger_eval(suite_id, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Run Single (R-ED-15: in-memory validation) ---

    @router.post(
        "/suites/{suite_id}/run-single",
        response_model=RunSingleCrossLawResponse,
    )
    def run_single(suite_id: str, request: RunSingleCrossLawRequest):
        """Run a single cross-law case for in-memory validation (R-ED-15)."""
        return service.run_single(suite_id, request)

    # --- Case CRUD ---

    @router.post(
        "/suites/{suite_id}/cases", response_model=CaseResponse, status_code=201
    )
    def add_case(suite_id: str, request: CaseRequest):
        """Add a test case to a suite (R6.2)."""
        return service.add_case(suite_id, request)

    @router.put("/suites/{suite_id}/cases/{case_id}", response_model=CaseResponse)
    def update_case(suite_id: str, case_id: str, request: CaseRequest):
        """Update a test case (R8.1)."""
        return service.update_case(suite_id, case_id, request)

    @router.delete("/suites/{suite_id}/cases/{case_id}", status_code=204)
    def delete_case(suite_id: str, case_id: str):
        """Delete a test case (R8.3)."""
        service.delete_case(suite_id, case_id)
        return Response(status_code=204)

    @router.post(
        "/suites/{suite_id}/cases/{case_id}/duplicate",
        response_model=CaseResponse,
        status_code=201,
    )
    def duplicate_case(suite_id: str, case_id: str):
        """Duplicate a test case (R8.2)."""
        return service.duplicate_case(suite_id, case_id)

    # --- Generation + AI Suggest ---

    @router.post("/generate", response_model=GenerateCasesResponse, status_code=201)
    async def generate_cases(request: GenerateCasesRequest):
        """Generate cross-law test cases using LLM (R1.1, R3.1)."""
        return await service.generate_cases(request)

    @router.post("/ai-suggest", response_model=AiSuggestResponse)
    def ai_suggest(request: AiSuggestRequest):
        """Get AI-powered name/description suggestion (R8.1, R8.2)."""
        return service.ai_suggest(request)

    # --- YAML Import/Export ---

    @router.get("/suites/{suite_id}/export")
    def export_yaml(suite_id: str):
        """Export a suite to YAML (R6.4)."""
        yaml_str = service.export_yaml(suite_id)
        return Response(
            content=yaml_str,
            media_type="application/x-yaml",
            headers={"Content-Disposition": f"attachment; filename={suite_id}.yaml"},
        )

    @router.post("/import", response_model=SuiteResponse, status_code=201)
    def import_yaml(request: ImportRequest):
        """Import a suite from YAML (R6.3)."""
        return service.import_yaml(request.yaml_content)

    return router
