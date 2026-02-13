# src/eval/eval_runner.py
"""
Minimalist evaluation runner following EVAL = PROD principle.

Architecture:
- Loads golden test cases from YAML/JSON
- Calls production code (ask.ask) for each case
- Applies scorers to production results
- Reports via reporters

Key principle: This file NEVER reimplements production logic.
All retrieval, ranking, and context selection is done by ask.ask().

Usage:
    python -m src.eval.eval_runner --law ai-act
    python -m src.eval.eval_runner --law ai-act --skip-llm
    python -m src.eval.eval_runner --law ai-act --pipeline-analysis
    python -m src.eval.eval_runner --law ai-act --case "case_id"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..services import ask
from ..engine.planning import UserProfile
from ..common.config_loader import get_settings_yaml
from .scorers import (
    AbstentionScorer,
    AnchorScorer,
    AnswerRelevancyScorer,
    ContractScorer,
    FaithfulnessScorer,
    GoldenExpected,
    PipelineBreakdownScorer,
    Score,
)
from .reporters import (
    CaseResult,
    EvalSummary,
    RetryStats,
    EscalationStats,
    ProgressReporter,
    FailureReporter,
    PipelineAnalysisReporter,
    LLMJudgeProgressReporter,
    FaithfulnessDebugReporter,
    JsonReporter,
    ProgressionTracker,
)
from .eval_core import EvalConfig
from .types import GoldenCase, ExpectedBehavior, RunMode


# ---------------------------------------------------------------------------
# Run mode helpers (maps legacy flags to unified run modes)
# ---------------------------------------------------------------------------

def _derive_run_mode(skip_llm: bool, llm_judge: bool | None) -> RunMode:
    """Derive RunMode from legacy CLI flags.

    Mapping:
        skip_llm=True            → retrieval_only
        skip_llm=False, llm_judge=False → full
        skip_llm=False, llm_judge=True  → full_with_judge

    Args:
        skip_llm: Legacy --skip-llm flag
        llm_judge: Legacy --llm-judge flag (None = auto)

    Returns:
        Unified RunMode
    """
    if skip_llm:
        return "retrieval_only"
    # Auto: enable LLM judge by default when not skipping LLM
    if llm_judge is None or llm_judge:
        return "full_with_judge"
    return "full"


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def _normalize_anchor(anchor: str) -> str:
    """Normalize anchor for comparison."""
    raw = str(anchor or "").strip().lower()
    return re.sub(r"\s+", "", raw)


def _load_yaml_or_json(path: Path) -> Any:
    """Load YAML or JSON file."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def load_golden_cases(path: Path) -> list[GoldenCase]:
    """Load golden cases from file."""
    data = _load_yaml_or_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of cases")

    cases: list[GoldenCase] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: case[{index}] must be an object")

        case_id = str(item.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"{path}: case[{index}] missing 'id'")

        profile = str(item.get("profile") or "").strip().upper()
        if profile not in {"LEGAL", "ENGINEERING"}:
            raise ValueError(f"{path}: case[{index}] invalid profile '{profile}'")

        prompt = str(item.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"{path}: case[{index}] missing 'prompt'")

        expected_raw = item.get("expected") or {}
        
        # Parse expected behavior
        contract_default = profile == "ENGINEERING"
        contract_check = expected_raw.get("contract_check", contract_default)
        
        min_citations = expected_raw.get("min_citations")
        max_citations = expected_raw.get("max_citations")
        if contract_check:
            if min_citations is None:
                min_citations = 2
            if max_citations is None:
                max_citations = 8

        # Parse behavior field (default: "answer")
        behavior = str(expected_raw.get("behavior") or "answer").strip().lower()
        if behavior not in {"answer", "abstain"}:
            raise ValueError(f"{path}: case[{index}] invalid behavior '{behavior}' (must be 'answer' or 'abstain')")

        # Parse cross-law expected behavior fields
        min_corpora_cited = expected_raw.get("min_corpora_cited")
        required_corpora_raw = expected_raw.get("required_corpora") or []
        required_corpora = tuple(str(c).strip() for c in required_corpora_raw)

        expected = ExpectedBehavior(
            must_include_any_of=[_normalize_anchor(a) for a in expected_raw.get("must_include_any_of") or []],
            must_include_any_of_2=[_normalize_anchor(a) for a in expected_raw.get("must_include_any_of_2") or []],
            must_include_all_of=[_normalize_anchor(a) for a in expected_raw.get("must_include_all_of") or []],
            must_not_include_any_of=[_normalize_anchor(a) for a in expected_raw.get("must_not_include_any_of") or []],
            contract_check=bool(contract_check),
            min_citations=min_citations,
            max_citations=max_citations,
            behavior=behavior,
            min_corpora_cited=min_corpora_cited,
            required_corpora=required_corpora,
        )

        # Parse new fields: test_types and origin
        test_types_raw = item.get("test_types") or ["retrieval"]
        test_types = tuple(str(t).strip().lower() for t in test_types_raw)

        origin = str(item.get("origin") or "auto").strip().lower()

        # Parse cross-law fields
        corpus_scope = str(item.get("corpus_scope") or "single").strip().lower()
        target_corpora_raw = item.get("target_corpora") or []
        target_corpora = tuple(str(c).strip() for c in target_corpora_raw)
        synthesis_mode = item.get("synthesis_mode")
        if synthesis_mode is not None:
            synthesis_mode = str(synthesis_mode).strip().lower()
            if synthesis_mode == "null":
                synthesis_mode = None

        cases.append(GoldenCase(
            id=case_id,
            profile=profile,
            prompt=prompt,
            expected=expected,
            test_types=test_types,
            origin=origin,
            corpus_scope=corpus_scope,
            target_corpora=target_corpora,
            synthesis_mode=synthesis_mode,
        ))

    # Deterministic ordering
    cases.sort(key=lambda c: c.id)
    return cases


def _find_cases_file(law: str, cases_file: str | None = None) -> Path:
    """Find the golden cases file for a law."""
    if cases_file:
        path = Path(cases_file)
        if path.exists():
            return path
        raise ValueError(f"Cases file not found: {cases_file}")

    # Standard locations
    repo_root = Path(__file__).resolve().parents[2]
    # Try different naming conventions
    law_underscore = law.replace("-", "_")
    candidates = [
        repo_root / "data" / "evals" / f"golden_cases_{law_underscore}.yaml",
        repo_root / "data" / "evals" / f"golden_cases_{law_underscore}.json",
        repo_root / "data" / "evals" / f"golden_cases_{law}.yaml",
        repo_root / "data" / "evals" / f"golden_cases_{law}.json",
        repo_root / "data" / "evals" / f"{law}.yaml",
        repo_root / "data" / "evals" / f"{law}.json",
        repo_root / "data" / "evals" / f"golden_{law}.yaml",
        repo_root / "data" / "evals" / f"golden_{law}.json",
    ]
    
    for path in candidates:
        if path.exists():
            return path
    
    raise ValueError(f"No golden cases file found for law '{law}'. Tried: {[str(p) for p in candidates]}")


# ---------------------------------------------------------------------------
# Thread-local engine cache
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _get_engine(law: str, model_override: str | None = None) -> ask.RAGEngine:
    """Get or create engine for current thread.

    Args:
        law: The corpus/law to use
        model_override: Optional model to use instead of default (for escalation)
    """
    if not hasattr(_thread_local, "engines"):
        _thread_local.engines = {}

    # Use model as part of cache key for escalation support
    cache_key = f"{law}:{model_override or 'default'}"
    if cache_key not in _thread_local.engines:
        if model_override:
            # Build engine with specific model for escalation
            from ..common.config_loader import load_settings
            settings = load_settings()
            # Create a modified settings object with the fallback model
            _thread_local.engines[cache_key] = ask.RAGEngine(
                docs_path=str(settings.docs_path),
                corpus_id=law,
                chunks_collection=settings.corpora[law].chunks_collection,
                embedding_model=settings.embedding_model,
                chat_model=model_override,  # Use fallback model
                vector_store_path=str(settings.vector_store_path),
                max_distance=(settings.corpora[law].max_distance
                              if settings.corpora[law].max_distance is not None
                              else settings.rag_max_distance),
                hybrid_vec_k=settings.hybrid_vec_k,
                ranking_weights=settings.ranking_weights,
            )
        else:
            _thread_local.engines[cache_key] = ask.build_engine(law=law)
    return _thread_local.engines[cache_key]


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

def _retry_with_backoff(func, *args, max_attempts: int = 5, **kwargs):
    """Retry function with exponential backoff for transient errors."""
    base_delay = 0.5
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            is_retryable = any(x in msg for x in ["429", "500", "502", "503", "504", "rate limit", "timeout"])
            
            if not is_retryable or attempt == max_attempts - 1:
                raise
            
            delay = min(base_delay * (2 ** attempt), 10.0)
            time.sleep(delay)


def _build_context_text(references_structured: list[dict]) -> str:
    """Build context text from references for LLM-judge scorers."""
    parts = []
    for ref in references_structured:
        if not isinstance(ref, dict):
            continue
        # Check multiple possible field names for the text content
        text = ref.get("chunk_text") or ref.get("text") or ref.get("content") or ""
        if text:
            # Add anchor info if available
            anchor_parts = []
            if ref.get("article"):
                anchor_parts.append(f"Article {ref['article']}")
            if ref.get("recital"):
                anchor_parts.append(f"Recital {ref['recital']}")
            if ref.get("annex"):
                anchor_parts.append(f"Annex {ref['annex']}")
            
            prefix = f"[{', '.join(anchor_parts)}] " if anchor_parts else ""
            parts.append(f"{prefix}{text}")
    
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def evaluate_case(
    case: GoldenCase,
    *,
    law: str,
    skip_llm: bool = False,
    scorers: list | None = None,
    llm_judge: bool = False,
    llm_judge_progress_callback=None,
    model_override: str | None = None,
) -> CaseResult:
    """
    Evaluate a single case using production code.

    CRITICAL: This function ONLY calls ask.ask() and applies scorers.
    No reimplementation of retrieval, ranking, or context selection.

    Args:
        case: The golden test case
        law: Law/corpus being evaluated
        skip_llm: If True, skip LLM generation (dry_run mode)
        scorers: List of scorers to apply (default: anchor, contract, pipeline)
        llm_judge: If True, also apply LLM-as-judge scorers (Faithfulness, Relevancy)
        llm_judge_progress_callback: Optional callback for LLM-judge progress
        model_override: Optional model to use instead of default (for escalation)
    """
    start = time.perf_counter()

    # Resolve profile
    profile = UserProfile.ENGINEERING if case.profile == "ENGINEERING" else UserProfile.LEGAL

    # EVAL = PROD: No special treatment. We don't send required_anchors to ask.ask.
    # The expected anchors are ONLY used for SCORING after retrieval, not to influence retrieval.
    # This ensures eval tests the exact same code path as production.

    # Call production code - THE ONLY PLACE WE CALL PRODUCTION
    engine = _get_engine(law, model_override=model_override)
    result = _retry_with_backoff(
        ask.ask,
        question=case.prompt,
        law=law,
        user_profile=profile,
        engine=engine,
        # NOTE: required_anchors intentionally NOT passed - eval must not influence retrieval
        contract_min_citations=case.expected.min_citations,
        dry_run=skip_llm,
    )
    
    # Convert expected to scorer format
    expected = GoldenExpected(
        must_include_any_of=case.expected.must_include_any_of,
        must_include_any_of_2=case.expected.must_include_any_of_2,
        must_include_all_of=case.expected.must_include_all_of,
    )
    
    # Apply scorers
    if scorers is None:
        scorers = [AnchorScorer(), ContractScorer(), PipelineBreakdownScorer()]
    
    scores: dict[str, Score] = {}
    for scorer in scorers:
        scores[scorer.name] = scorer.score(
            expected=expected,
            retrieval_metrics=result.retrieval_metrics,
            references_structured=result.references_structured,
        )
    
    # Apply LLM-as-judge scorers if enabled (and not in dry_run/skip_llm mode)
    if llm_judge and not skip_llm:
        # Build context text from ALL references sent to LLM (not just the ones cited)
        # This is crucial for faithfulness scoring - we need to check if claims are
        # supported by the full context the LLM received, not just what it cited.
        all_refs = result.retrieval_metrics.get("references_structured_all") or result.references_structured
        context_text = _build_context_text(all_refs)
        answer_text = result.answer or ""

        # Check if this is an abstain case (per UAEval4RAG best practices)
        # Abstain cases test that the system correctly refuses to answer ambiguous questions.
        # Standard faithfulness scoring is skipped because:
        # 1. RAGAS gives 0 to correct refusals (false negative)
        # 2. The goal is to test abstention behavior, not answer quality
        is_abstain_case = case.expected.behavior == "abstain"

        if is_abstain_case:
            # For abstain cases: Apply abstention-specific scoring
            abstention_scorer = AbstentionScorer()
            abstention_result = abstention_scorer.score(
                question=case.prompt,
                answer=answer_text,
            )
            scores["abstention"] = abstention_result

            # Skip faithfulness - per research, it produces false negatives for correct refusals
            # Instead, mark as N/A with explanation
            scores["faithfulness"] = Score(
                passed=True,  # Don't fail abstain cases on faithfulness
                score=1.0,
                message="Skipped for abstain case (per UAEval4RAG best practices)",
            )
            scores["answer_relevancy"] = Score(
                passed=True,  # Don't fail abstain cases on relevancy
                score=1.0,
                message="Skipped for abstain case (per UAEval4RAG best practices)",
            )
        else:
            # For answer cases: Apply standard faithfulness and relevancy scoring
            faithfulness_scorer = FaithfulnessScorer()
            faithfulness_result = faithfulness_scorer.score(
                question=case.prompt,
                answer=answer_text,
                context=context_text,
                progress_callback=llm_judge_progress_callback,
            )
            scores["faithfulness"] = faithfulness_result

            # Answer Relevancy scorer
            relevancy_scorer = AnswerRelevancyScorer()
            relevancy_result = relevancy_scorer.score(
                question=case.prompt,
                answer=answer_text,
                progress_callback=llm_judge_progress_callback,
            )
            scores["answer_relevancy"] = relevancy_result
    
    # Determine overall pass/fail
    passed = all(s.passed for s in scores.values())
    
    duration_ms = (time.perf_counter() - start) * 1000
    
    return CaseResult(
        case_id=case.id,
        profile=case.profile,
        passed=passed,
        scores=scores,
        duration_ms=duration_ms,
        retrieval_metrics=result.retrieval_metrics,
    )


def run_eval(
    *,
    law: str,
    cases_file: str | None = None,
    case_ids: set[str] | None = None,
    profiles: set[str] | None = None,
    limit: int | None = None,
    skip_llm: bool = False,
    llm_judge: bool | None = None,  # None = auto (True unless skip_llm)
    debug_faithfulness: bool = False,
    pipeline_analysis: bool = False,
    progress: bool = True,
    output_dir: str | None = None,
    dump_failures: bool = False,
    verbose: bool = False,
    max_retries: int = 3,
) -> EvalSummary:
    """
    Run evaluation on golden test cases.
    
    Args:
        law: Which law/corpus to evaluate (ai-act, gdpr, dora)
        cases_file: Optional path to golden cases file
        case_ids: Optional set of specific case IDs to run
        profiles: Optional set of profiles to filter (LEGAL, ENGINEERING)
        limit: Maximum number of cases to run
        skip_llm: If True, run pipeline but skip LLM generation (--skip-llm)
        llm_judge: If True, apply LLM-as-judge scorers. Default: auto (True unless skip_llm)
                   EVAL = PROD principle: always run LLM-judge by default
        debug_faithfulness: If True, show detailed per-claim faithfulness output
        pipeline_analysis: If True, show detailed pipeline breakdown
        progress: Show progress bar
        output_dir: Directory for output files (default: runs/)
        dump_failures: If True, write detailed failure info
        verbose: Verbose output
        max_retries: Maximum number of retry attempts for flaky tests (default: 3)
    
    Returns:
        EvalSummary with all results
    """
    start_time = time.perf_counter()
    
    # Load cases
    cases_path = _find_cases_file(law, cases_file)
    all_cases = load_golden_cases(cases_path)
    
    # Filter cases
    cases = all_cases
    if case_ids:
        cases = [c for c in cases if c.id in case_ids]
    if profiles:
        profiles_upper = {p.upper() for p in profiles}
        cases = [c for c in cases if c.profile in profiles_upper]
    if limit and limit > 0:
        cases = cases[:limit]
    
    if not cases:
        print(f"No cases found for law={law}", file=sys.stderr)
        return EvalSummary(
            law=law,
            total=0,
            passed=0,
            failed=0,
            skipped=len(all_cases) - len(cases),
            duration_seconds=0,
            results=[],
        )
    
    # Setup reporters
    progress_reporter = ProgressReporter(len(cases), show_progress=progress)
    failure_reporter = FailureReporter(verbose=verbose)
    pipeline_reporter = PipelineAnalysisReporter() if pipeline_analysis else None
    llm_judge_reporter = LLMJudgeProgressReporter(verbose=verbose) if llm_judge else None
    faithfulness_debug_reporter = FaithfulnessDebugReporter() if debug_faithfulness else None
    
    # EVAL = PROD: LLM-judge is enabled by default unless explicitly disabled or skip_llm
    if llm_judge is None:
        llm_judge = not skip_llm

    # Setup scorers - always include PipelineBreakdownScorer for diagnostics
    scorers = [AnchorScorer(), PipelineBreakdownScorer()]
    if not skip_llm:
        scorers.append(ContractScorer())
    
    # Run evaluation
    progress_reporter.start(law)
    
    if llm_judge and not skip_llm:
        print("  📊 LLM-as-judge enabled (Faithfulness + Relevancy)")
        # Enable live rate limit tracking in verbose mode
        from .scorers import get_rate_limit_tracker
        tracker = get_rate_limit_tracker()
        tracker.reset()
        tracker.set_verbose(verbose)
    
    # Get concurrency setting from config
    settings = get_settings_yaml()
    eval_settings = settings.get("eval", {})
    max_workers = eval_settings.get("default_concurrency", 10)

    # Model escalation settings
    escalation_settings = eval_settings.get("model_escalation", {})
    escalation_enabled = escalation_settings.get("enabled", False) and not skip_llm
    fallback_model = escalation_settings.get("fallback_model")
    max_primary_retries = escalation_settings.get("max_primary_retries", 3)

    if escalation_enabled and fallback_model:
        print(f"  🚀 Model escalation enabled (fallback: {fallback_model})")
        print(f"     Phase 1: Primary model @ concurrency {max_workers}")
        print(f"     Phase 2: Fallback model @ concurrency 1 (failed cases only)")

    # Use parallel processing when multiple workers configured
    use_parallel = max_workers > 1

    results: list[CaseResult] = []
    failures: list[CaseResult] = []
    results_lock = threading.Lock()

    # Retry statistics tracking
    retry_stats = RetryStats()
    escalation_stats = EscalationStats()

    def process_case(case: GoldenCase) -> CaseResult:
        """Process a single case with retry logic (thread-safe).

        Strategy: Try with primary model up to max_primary_retries times.
        Escalation is handled separately in Phase 2 after all primary runs complete.
        """
        attempt = 0
        last_result = None

        while attempt <= max_primary_retries:
            try:
                result = evaluate_case(
                    case,
                    law=law,
                    skip_llm=skip_llm,
                    scorers=scorers,
                    llm_judge=llm_judge,
                    llm_judge_progress_callback=None,  # Disable per-case callbacks in parallel
                )

                if result.passed:
                    # Success with primary model
                    return CaseResult(
                        case_id=result.case_id,
                        profile=result.profile,
                        passed=result.passed,
                        scores=result.scores,
                        duration_ms=result.duration_ms,
                        retrieval_metrics=result.retrieval_metrics,
                        retry_count=attempt,
                        escalated=False,
                    )

                # Test failed - save result for potential retry
                last_result = result
                attempt += 1

                if attempt <= max_primary_retries:
                    if verbose:
                        print(f"  🔄 Retrying {case.id} (attempt {attempt}/{max_primary_retries})")

            except Exception as e:
                last_result = CaseResult(
                    case_id=case.id,
                    profile=case.profile,
                    passed=False,
                    scores={"error": Score(passed=False, score=0.0, message=str(e))},
                    duration_ms=0,
                    retry_count=attempt,
                )
                attempt += 1

        # All retries exhausted - return last failed result
        return CaseResult(
            case_id=last_result.case_id,
            profile=last_result.profile,
            passed=last_result.passed,
            scores=last_result.scores,
            duration_ms=last_result.duration_ms,
            retrieval_metrics=getattr(last_result, 'retrieval_metrics', {}),
            retry_count=max_primary_retries,
            escalated=False,
        )

    def process_escalation_case(case: GoldenCase) -> CaseResult:
        """Process a single case with fallback model (Phase 2 escalation)."""
        if verbose:
            print(f"  🚀 Escalating {case.id} to {fallback_model}")

        try:
            result = evaluate_case(
                case,
                law=law,
                skip_llm=False,
                scorers=scorers,
                llm_judge=llm_judge,
                llm_judge_progress_callback=None,
                model_override=fallback_model,
            )

            return CaseResult(
                case_id=result.case_id,
                profile=result.profile,
                passed=result.passed,
                scores=result.scores,
                duration_ms=result.duration_ms,
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
                duration_ms=0,
                retry_count=max_primary_retries,
                escalated=True,
                escalation_model=fallback_model,
            )
    
    # =========================================================================
    # PHASE 1: Run all cases with primary model
    # =========================================================================
    if use_parallel:
        print(f"  ⚡ Phase 1: Parallel execution ({max_workers} workers)")
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

                    # Report progress
                    anchor_score = result.scores.get("anchor_presence")
                    message = anchor_score.message if anchor_score and not anchor_score.passed else ""
                    progress_reporter.update(case.id, result.passed, message)

                    # Track failures
                    if not result.passed:
                        failures.append(result)
                        if verbose:
                            failure_reporter.report_failure(result)

                    # Pipeline analysis
                    if pipeline_reporter:
                        pipeline_reporter.report(result)

                    # Faithfulness debug
                    if faithfulness_debug_reporter:
                        faithfulness_debug_reporter.report(result)
    else:
        # Sequential execution (skip_llm mode or single worker)
        print(f"  ⚡ Phase 1: Sequential execution")
        for case in cases:
            result = process_case(case)
            results.append(result)

            # Report progress
            anchor_score = result.scores.get("anchor_presence")
            message = anchor_score.message if anchor_score and not anchor_score.passed else ""
            progress_reporter.update(case.id, result.passed, message)

            # Track failures
            if not result.passed:
                failures.append(result)
                if verbose:
                    failure_reporter.report_failure(result)

            # Pipeline analysis
            if pipeline_reporter:
                pipeline_reporter.report(result)

            # Faithfulness debug
            if faithfulness_debug_reporter:
                faithfulness_debug_reporter.report(result)

    # =========================================================================
    # PHASE 2: Escalate failed cases to fallback model (sequential, concurrency 1)
    # =========================================================================
    if escalation_enabled and fallback_model and not skip_llm and failures:
        # Identify cases eligible for escalation (faithfulness/relevancy failures only)
        cases_to_escalate: list[GoldenCase] = []
        case_lookup = {c.id: c for c in cases}

        for failed_result in failures:
            faith_score = failed_result.scores.get("faithfulness")
            relevancy_score = failed_result.scores.get("answer_relevancy")
            # Only escalate generation failures (faithfulness/relevancy), not retrieval failures
            if (faith_score and not faith_score.passed) or (relevancy_score and not relevancy_score.passed):
                if failed_result.case_id in case_lookup:
                    cases_to_escalate.append(case_lookup[failed_result.case_id])

        if cases_to_escalate:
            print(f"\n  🚀 Phase 2: Escalating {len(cases_to_escalate)} failed cases to {fallback_model} (concurrency 1)")

            for case in cases_to_escalate:
                escalated_result = process_escalation_case(case)

                # Find and replace the original failed result
                for i, r in enumerate(results):
                    if r.case_id == escalated_result.case_id:
                        results[i] = escalated_result
                        break

                # Update failures list
                failures = [f for f in failures if f.case_id != escalated_result.case_id]
                if not escalated_result.passed:
                    failures.append(escalated_result)

                # Report escalation progress
                status = "✓" if escalated_result.passed else "✗"
                print(f"    {status} {case.id} (escalated)")

                # Pipeline analysis for escalated
                if pipeline_reporter:
                    pipeline_reporter.report(escalated_result)

                # Faithfulness debug for escalated
                if faithfulness_debug_reporter:
                    faithfulness_debug_reporter.report(escalated_result)
        else:
            print(f"\n  ℹ️  No cases eligible for escalation (failures are retrieval-related, not generation)")
    elif escalation_enabled and fallback_model and not failures:
        print(f"\n  ✅ All cases passed - no escalation needed")
    
    # Build summary
    duration_seconds = time.perf_counter() - start_time

    # Calculate retry and escalation statistics from results
    for r in results:
        if r.retry_count > 0:
            retry_stats.cases_with_retries += 1
            retry_stats.total_retries += r.retry_count
            if r.passed and not r.escalated:
                retry_stats.cases_passed_on_retry += 1
            elif not r.passed and not r.escalated:
                retry_stats.cases_failed_after_retries += 1

        if r.escalated:
            escalation_stats.cases_escalated += 1
            escalation_stats.escalated_case_ids.append(r.case_id)
            if r.passed:
                escalation_stats.cases_passed_on_escalation += 1
            else:
                escalation_stats.cases_failed_after_escalation += 1

    summary = EvalSummary(
        law=law,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        skipped=len(all_cases) - len(cases),
        duration_seconds=duration_seconds,
        results=results,
        retry_stats=retry_stats,
        escalation_stats=escalation_stats,
    )
    
    # Report summary
    progress_reporter.finish(summary)
    failure_reporter.report_summary(failures)
    
    # Show rate limit stats if LLM judge was used
    if llm_judge and not skip_llm:
        from .scorers import get_rate_limit_tracker
        tracker = get_rate_limit_tracker()
        tracker.print_summary()
        tracker.reset()  # Reset for next eval run
    
    # Write output
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(__file__).resolve().parents[2] / "runs"
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    json_path = output_path / f"{timestamp}_eval_{law}.json"
    JsonReporter(json_path).write(summary)
    print(f"\n📄 Results written to: {json_path}")
    
    # Optionally copy to stable name
    stable_path = output_path / f"eval_{law}.json"
    JsonReporter(stable_path).write(summary)
    
    # Record to progression tracking file
    progression_path = output_path / "progression.json"
    progression_tracker = ProgressionTracker(progression_path)
    progression_tracker.record(summary)
    print(f"📈 Progression recorded to: {progression_path}")
    
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_case_ids(raw: str) -> set[str]:
    """Parse comma-separated case IDs."""
    if not raw:
        return set()
    return {c.strip() for c in raw.split(",") if c.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Runner (EVAL = PROD architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EVAL = PROD Principle:
  - LLM-judge (Faithfulness + Relevancy) is enabled by default
  - Citation verification is always included
  - Pipeline breakdown is always included
  - All production features are tested

Examples:
  python -m src.eval.eval_runner --law ai-act              # Full eval with LLM-judge
  python -m src.eval.eval_runner --law ai-act --skip-llm   # Skip LLM (retrieval only)
  python -m src.eval.eval_runner --law ai-act --no-llm-judge  # Disable LLM-judge
  python -m src.eval.eval_runner --law ai-act --case "case_id"
  python -m src.eval.eval_runner --law ai-act --limit 10
  python -m src.eval.eval_runner --law ai-act --history
        """,
    )
    
    parser.add_argument("--law", required=True, help="Law/corpus to evaluate (ai-act, gdpr, dora)")
    parser.add_argument("--cases-file", help="Path to golden cases file (default: auto-detect)")
    parser.add_argument("--case", dest="case_ids", help="Comma-separated case IDs to run")
    parser.add_argument("--profile", dest="profiles", help="Comma-separated profiles (LEGAL, ENGINEERING)")
    parser.add_argument("--limit", type=int, help="Maximum number of cases to run")
    parser.add_argument("--skip-llm", action="store_true", help="Run pipeline but skip LLM generation")
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM-as-judge scorers (enabled by default)")
    parser.add_argument("--debug-faithfulness", action="store_true", help="Show detailed faithfulness debug output per claim")
    parser.add_argument("--pipeline-analysis", action="store_true", help="Show detailed pipeline breakdown (always included in output)")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument("--out", dest="output_dir", help="Output directory for results")
    parser.add_argument("--dump-failures", action="store_true", help="Write detailed failure info")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts for flaky tests (default: 3)")
    parser.add_argument("--history", action="store_true", help="Show progression history instead of running eval")
    parser.add_argument("--history-limit", type=int, default=10, help="Number of history entries to show (default: 10)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Handle history command
    if args.history:
        output_path = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[2] / "runs"
        progression_path = output_path / "progression.json"
        tracker = ProgressionTracker(progression_path)
        tracker.print_history(law=args.law, limit=args.history_limit)
        return 0
    
    case_ids = _parse_case_ids(args.case_ids) if args.case_ids else None
    profiles = {p.strip().upper() for p in args.profiles.split(",")} if args.profiles else None
    
    # EVAL = PROD: llm_judge is True by default, --no-llm-judge disables it
    llm_judge = None if not args.no_llm_judge else False

    try:
        summary = run_eval(
            law=args.law,
            cases_file=args.cases_file,
            case_ids=case_ids,
            profiles=profiles,
            limit=args.limit,
            skip_llm=args.skip_llm,
            llm_judge=llm_judge,
            debug_faithfulness=args.debug_faithfulness,
            pipeline_analysis=args.pipeline_analysis,
            progress=not args.no_progress,
            output_dir=args.output_dir,
            dump_failures=args.dump_failures,
            verbose=args.verbose,
            max_retries=args.max_retries,
        )
        
        return 0 if summary.failed == 0 else 1
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
