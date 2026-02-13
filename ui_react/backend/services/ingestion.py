"""Ingestion orchestrator service.

Single Responsibility: Orchestrate the full ingestion pipeline for adding
new legislation to the system, with progress reporting via generator.

Pipeline stages:
1. Download HTML from EUR-Lex
2. Chunking + LLM enrichment
3. Vector store indexing
4. Citation graph building
5. (Optional) Eval case generation
6. Corpora inventory update
"""

from __future__ import annotations

import logging
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Iterator

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.eurlex_listing import (
    download_legislation_html,
    validate_celex,
    build_html_url,
    EurLexSecurityError,
    EurLexNetworkError,
)
from src.ingestion.eurlex_engine import run_ingestion_for_file
from src.ingestion.html_chunks import preflight_check_html
from src.ingestion.citation_graph import CitationGraph
from src.common.config_loader import load_settings, clear_config_cache
from src.common.corpora_inventory import (
    load_corpora_inventory,
    save_corpora_inventory,
    upsert_corpus_inventory,
    default_corpora_path,
)
from src.common.corpus_registry import (
    load_registry,
    save_registry,
    upsert_corpus,
    derive_aliases,
    default_registry_path,
)

logger = logging.getLogger(__name__)

# Stage names for progress reporting
STAGE_DOWNLOAD = "download"
STAGE_CHUNKING = "chunking"
STAGE_INDEXING = "indexing"
STAGE_CITATION_GRAPH = "citation_graph"
STAGE_EVAL_GENERATION = "eval_generation"
STAGE_EVAL_RUN = "eval_run"
STAGE_CONFIG_UPDATE = "config_update"


def _emit_stage(stage: str, message: str, completed: bool = False) -> dict[str, Any]:
    """Create a stage event."""
    return {
        "type": "stage",
        "stage": stage,
        "message": message,
        "completed": completed,
    }


def _emit_progress(stage: str, progress_pct: float, current: int = 0, total: int = 0) -> dict[str, Any]:
    """Create a progress event."""
    return {
        "type": "progress",
        "stage": stage,
        "progress_pct": round(progress_pct, 1),
        "current": current,
        "total": total,
    }


def _emit_complete(corpus_id: str) -> dict[str, Any]:
    """Create a completion event."""
    return {
        "type": "complete",
        "corpus_id": corpus_id,
    }


def _emit_error(message: str) -> dict[str, Any]:
    """Create an error event."""
    return {
        "type": "error",
        "error": message,
    }


def _emit_eval_result(
    case_id: str,
    question: str,
    answer: str,
    expected_articles: list[str],
    actual_articles: list[str],
    passed: bool,
    notes: str = "",
) -> dict[str, Any]:
    """Create an eval result event."""
    return {
        "type": "eval_result",
        "case_id": case_id,
        "question": question,
        "answer": answer,
        "expected_articles": expected_articles,
        "actual_articles": actual_articles,
        "passed": passed,
        "notes": notes,
    }


def _emit_preflight(
    handled: dict[str, int],
    unhandled: dict[str, int],
    warnings: list[dict[str, str]],
) -> dict[str, Any]:
    """Create a preflight analysis event."""
    return {
        "type": "preflight",
        "handled": handled,
        "unhandled": unhandled,
        "warnings": warnings,
    }


def _emit_eval_summary(
    total: int,
    passed: int,
    failed: int,
) -> dict[str, Any]:
    """Create an eval summary event."""
    return {
        "type": "eval_summary",
        "total": total,
        "passed": passed,
        "failed": failed,
    }


def _retry_with_backoff(func, *args, max_attempts: int = 3, **kwargs):
    """Retry function with exponential backoff for transient errors."""
    import time as time_module

    base_delay = 0.5
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            msg = str(e).lower()
            is_retryable = any(x in msg for x in ["429", "500", "502", "503", "504", "rate limit", "timeout"])

            if not is_retryable or attempt == max_attempts - 1:
                raise

            delay = base_delay * (2 ** attempt)
            time_module.sleep(delay)

    raise RuntimeError("Max retries exceeded")


def _run_verification_eval(
    corpus_id: str,
    run_mode: str = "full",
) -> Iterator[dict[str, Any]]:
    """Run verification eval on generated eval cases with parallel execution.

    EVAL = PROD: Uses the exact same execution logic as CLI and Dashboard.
    All settings are read from config/settings.yaml:
    - Concurrency from eval.default_concurrency
    - Retry count from eval.model_escalation.max_primary_retries
    - Model escalation from eval.model_escalation.enabled + fallback_model

    Yields eval_result events for each case and eval_summary at the end.
    After completion, results are persisted to runs/eval_{corpus_id}.json.

    Args:
        corpus_id: The corpus to evaluate
        run_mode: Eval run mode (retrieval_only, full, full_with_judge)
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.engine.rag import RAGEngine
    from src.eval.eval_runner import evaluate_case, load_golden_cases
    from src.eval.reporters import CaseResult, EvalSummary, JsonReporter, RetryStats, EscalationStats, PipelineStageStats
    from src.eval.scorers import Score, AnchorScorer, PipelineBreakdownScorer, ContractScorer
    from src.common.config_loader import get_settings_yaml

    # Load ALL settings from config (same as CLI and Dashboard)
    settings = get_settings_yaml()
    eval_settings = settings.get("eval", {})
    max_workers = eval_settings.get("default_concurrency", 5)

    # Model escalation settings (same as CLI and Dashboard)
    escalation_settings = eval_settings.get("model_escalation", {})
    max_primary_retries = escalation_settings.get("max_primary_retries", 3)
    escalation_enabled = escalation_settings.get("enabled", False)
    fallback_model = escalation_settings.get("fallback_model")

    skip_llm = run_mode == "retrieval_only"
    llm_judge = run_mode == "full_with_judge"

    # Disable escalation for retrieval_only mode
    if skip_llm:
        escalation_enabled = False

    # Load eval cases
    evals_path = PROJECT_ROOT / "data" / "evals" / f"golden_cases_{corpus_id}.yaml"
    if not evals_path.exists():
        logger.warning("No eval cases found at %s", evals_path)
        yield _emit_eval_result(
            case_id="error",
            question="Kunne ikke finde eval cases",
            answer=f"Fil ikke fundet: {evals_path}",
            expected_articles=[],
            actual_articles=[],
            passed=False,
            notes="Systemfejl",
        )
        yield _emit_eval_summary(total=0, passed=0, failed=0)
        return

    try:
        cases = load_golden_cases(evals_path)
    except Exception as e:
        logger.error("Failed to load eval cases: %s", e)
        yield _emit_eval_result(
            case_id="error",
            question="Kunne ikke indlæse eval cases",
            answer=f"Fejl ved indlæsning: {str(e)}",
            expected_articles=[],
            actual_articles=[],
            passed=False,
            notes="Systemfejl",
        )
        yield _emit_eval_summary(total=0, passed=0, failed=0)
        return

    if not cases:
        logger.warning("No valid eval cases in %s", evals_path)
        yield _emit_eval_result(
            case_id="error",
            question="Ingen valide eval cases",
            answer="Filen indeholder ingen gyldige testcases",
            expected_articles=[],
            actual_articles=[],
            passed=False,
            notes="Systemfejl",
        )
        yield _emit_eval_summary(total=0, passed=0, failed=0)
        return

    # Build case lookup for extracting expected articles
    case_lookup = {c.id: c for c in cases}

    total = len(cases)
    results: list[CaseResult] = []
    failures: list[CaseResult] = []
    passed_count = 0
    failed_count = 0
    case_index = 0

    # Setup scorers
    scorers = [AnchorScorer(), PipelineBreakdownScorer()]
    if not skip_llm:
        scorers.append(ContractScorer())

    def process_case(case):
        """Process a single case with retry logic (same as CLI and Dashboard)."""
        attempt = 0
        last_result = None

        while attempt <= max_primary_retries:
            try:
                result = evaluate_case(
                    case,
                    law=corpus_id,
                    skip_llm=skip_llm,
                    scorers=scorers,
                    llm_judge=llm_judge,
                    llm_judge_progress_callback=None,
                )

                if result.passed:
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

                last_result = result
                attempt += 1

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

        # All retries exhausted
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

    def process_escalation_case(case):
        """Process a single case with fallback model (Phase 2 escalation, same as CLI and Dashboard)."""
        try:
            result = evaluate_case(
                case,
                law=corpus_id,
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
    # PHASE 1: Run all cases with primary model (parallel) - same as CLI/Dashboard
    # =========================================================================
    use_parallel = max_workers > 1

    if use_parallel:
        logger.info("Phase 1: Running verification eval with %d workers", max_workers)
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

                results.append(result)
                case_index += 1

                if result.passed:
                    passed_count += 1
                else:
                    failed_count += 1
                    failures.append(result)

                # Extract actual articles from retrieval metrics
                actual_articles = []
                run_meta = result.retrieval_metrics.get("run") or {}
                for anchor in run_meta.get("anchors_in_top_k") or []:
                    actual_articles.append(anchor)

                # Get expected articles from the case
                original_case = case_lookup.get(result.case_id)
                expected_articles = list(original_case.expected.must_include_any_of) if original_case else []

                # Emit eval result event for UI
                yield _emit_eval_result(
                    case_id=result.case_id,
                    question=original_case.prompt if original_case else "",
                    answer="",
                    expected_articles=expected_articles,
                    actual_articles=actual_articles,
                    passed=result.passed,
                    notes="",
                )

                # Emit progress (Phase 1 uses 10-70%)
                pct = 10 + (60 * case_index / total)
                yield _emit_progress(STAGE_EVAL_RUN, pct, case_index, total)
    else:
        # Sequential execution (fallback)
        logger.info("Phase 1: Running verification eval sequentially")
        for case in cases:
            result = process_case(case)
            results.append(result)
            case_index += 1

            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
                failures.append(result)

            # Extract actual articles from retrieval metrics
            actual_articles = []
            run_meta = result.retrieval_metrics.get("run") or {}
            for anchor in run_meta.get("anchors_in_top_k") or []:
                actual_articles.append(anchor)

            expected_articles = list(case.expected.must_include_any_of)

            yield _emit_eval_result(
                case_id=result.case_id,
                question=case.prompt,
                answer="",
                expected_articles=expected_articles,
                actual_articles=actual_articles,
                passed=result.passed,
                notes="",
            )

            pct = 10 + (60 * case_index / total)
            yield _emit_progress(STAGE_EVAL_RUN, pct, case_index, total)

    # =========================================================================
    # PHASE 2: Escalate failed cases to fallback model (sequential) - same as CLI/Dashboard
    # =========================================================================
    escalation_stats = EscalationStats()

    if escalation_enabled and fallback_model and failures:
        # Identify cases eligible for escalation (generation failures only)
        cases_to_escalate = []

        for failed_result in failures:
            faith_score = failed_result.scores.get("faithfulness")
            relevancy_score = failed_result.scores.get("answer_relevancy")
            # Only escalate generation failures, not retrieval failures
            if (faith_score and not faith_score.passed) or (relevancy_score and not relevancy_score.passed):
                if failed_result.case_id in case_lookup:
                    cases_to_escalate.append(case_lookup[failed_result.case_id])

        if cases_to_escalate:
            logger.info("Phase 2: Escalating %d failed cases to %s", len(cases_to_escalate), fallback_model)

            for i, case in enumerate(cases_to_escalate):
                escalated_result = process_escalation_case(case)

                # Find and replace the original failed result
                for j, r in enumerate(results):
                    if r.case_id == escalated_result.case_id:
                        # Update counters
                        if not r.passed and escalated_result.passed:
                            passed_count += 1
                            failed_count -= 1
                        results[j] = escalated_result
                        break

                # Update failures list
                failures = [f for f in failures if f.case_id != escalated_result.case_id]
                if not escalated_result.passed:
                    failures.append(escalated_result)

                # Track escalation stats
                escalation_stats.cases_escalated += 1
                escalation_stats.escalated_case_ids.append(escalated_result.case_id)
                if escalated_result.passed:
                    escalation_stats.cases_passed_on_escalation += 1
                else:
                    escalation_stats.cases_failed_after_escalation += 1

                # Emit progress (Phase 2 uses 70-90%)
                pct = 70 + (20 * (i + 1) / len(cases_to_escalate))
                yield _emit_progress(STAGE_EVAL_RUN, pct)

    # Calculate retry statistics
    retry_stats = RetryStats()
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

    # Build and save summary
    stage_stats = PipelineStageStats.from_results(results) if results else PipelineStageStats()

    summary = EvalSummary(
        law=corpus_id,
        total=total,
        passed=passed_count,
        failed=failed_count,
        skipped=0,
        duration_seconds=0,  # Not tracked in verification
        results=results,
        stage_stats=stage_stats,
        retry_stats=retry_stats,
        escalation_stats=escalation_stats,
    )

    # Persist results using JsonReporter format
    runs_dir = PROJECT_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts_filename = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    timestamped_path = runs_dir / f"{ts_filename}_eval_{corpus_id}.json"
    stable_path = runs_dir / f"eval_{corpus_id}.json"

    try:
        JsonReporter(timestamped_path).write(summary)
        JsonReporter(stable_path).write(summary)
        logger.info("Saved eval results to %s and %s", timestamped_path, stable_path)
    except Exception as e:
        logger.error("Failed to save eval results: %s", e)

    # Emit summary event for UI
    yield _emit_eval_summary(
        total=summary.total,
        passed=summary.passed,
        failed=summary.failed,
    )
    yield _emit_progress(STAGE_EVAL_RUN, 100)


def run_full_ingestion(
    *,
    celex_number: str,
    corpus_id: str,
    display_name: str,
    fullname: str | None = None,
    eurovoc_labels: list[str] | None = None,
    generate_eval: bool = False,
    entry_into_force: str | None = None,
    last_modified: str | None = None,
    eval_run_mode: str | None = "full",
) -> Iterator[dict[str, Any]]:
    """Run the full ingestion pipeline with progress reporting.

    Yields SSE-compatible event dictionaries for progress tracking.

    Args:
        celex_number: CELEX number of the legislation (e.g., "32022L2555")
        corpus_id: Short ID for the corpus (e.g., "nis2")
        display_name: Display name for the UI
        fullname: Full official legal title for citations (optional)
        eurovoc_labels: EuroVoc subject keywords from EUR-Lex (optional)
        generate_eval: Whether to generate eval cases (optional)
        entry_into_force: Entry into force date (ISO format, optional)
        last_modified: Document adoption date (ISO format, optional)
        eval_run_mode: Run mode for verification eval (retrieval_only, full, full_with_judge)

    Yields:
        Event dictionaries with type, stage, progress, etc.
    """
    settings = load_settings()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Stage 1: Download HTML
        yield _emit_stage(STAGE_DOWNLOAD, "Downloader HTML fra EUR-Lex...")
        yield _emit_progress(STAGE_DOWNLOAD, 10)

        try:
            html_path = download_legislation_html(celex_number, raw_dir)
            # Rename to match corpus_id for consistency
            final_html_path = raw_dir / f"{corpus_id}.html"
            if html_path != final_html_path:
                html_path.rename(final_html_path)
                html_path = final_html_path

            yield _emit_progress(STAGE_DOWNLOAD, 100)
            yield _emit_stage(STAGE_DOWNLOAD, "HTML downloadet", completed=True)

            # Run preflight check immediately after download
            html_content = html_path.read_text(encoding="utf-8", errors="replace")
            preflight = preflight_check_html(html_content, enable_eurlex_structural_ids=True)
            warnings_dicts = [
                {
                    "category": w.category,
                    "message": w.message,
                    "location": w.location,
                    "severity": w.severity,
                    "suggestion": w.suggestion,
                }
                for w in preflight.warnings
            ]
            yield _emit_preflight(
                handled=preflight.handled,
                unhandled=preflight.unhandled,
                warnings=warnings_dicts,
            )

        except (EurLexSecurityError, EurLexNetworkError) as e:
            yield _emit_error(f"Download fejlede: {str(e)}")
            return

        # Stage 2: Chunking + LLM enrichment
        yield _emit_stage(STAGE_CHUNKING, "Chunker og beriger med LLM...")
        yield _emit_progress(STAGE_CHUNKING, 5)

        try:
            chunk_tokens = int(getattr(settings, "eurlex_chunk_tokens", 500))
            overlap = int(getattr(settings, "eurlex_overlap", 100))

            # Use queue for progress reporting from background thread
            progress_queue: Queue[tuple[int, int] | None] = Queue()
            result_holder: dict[str, Any] = {"path": None, "error": None}

            def progress_callback(current: int, total: int) -> None:
                progress_queue.put((current, total))

            def run_chunking() -> None:
                try:
                    path = run_ingestion_for_file(
                        corpus_id=corpus_id,
                        html_path=html_path,
                        out_dir=processed_dir,
                        chunk_tokens=chunk_tokens,
                        overlap=overlap,
                        language="da",
                        progress_callback=progress_callback,
                    )
                    result_holder["path"] = path
                except Exception as e:
                    result_holder["error"] = e
                finally:
                    progress_queue.put(None)  # Signal completion

            # Start chunking in background thread
            thread = threading.Thread(target=run_chunking, daemon=True)
            thread.start()

            # Yield progress events while chunking runs
            while True:
                try:
                    item = progress_queue.get(timeout=0.5)
                    if item is None:
                        break  # Chunking complete
                    current, total = item
                    # Scale progress from 5% to 95% during enrichment
                    pct = 5 + (90 * current / total) if total > 0 else 5
                    yield _emit_progress(STAGE_CHUNKING, pct, current, total)
                except Empty:
                    continue  # Keep waiting

            thread.join()

            # Check for errors
            if result_holder["error"]:
                raise result_holder["error"]

            # result_holder["path"] now holds an IngestionResult, extract the path
            ingestion_result = result_holder["path"]
            jsonl_path = ingestion_result.output_path

            yield _emit_progress(STAGE_CHUNKING, 100)
            yield _emit_stage(STAGE_CHUNKING, "Chunking færdig", completed=True)

        except Exception as e:
            yield _emit_error(f"Chunking fejlede: {str(e)}")
            return

        # Stage 3: Vector store indexing
        yield _emit_stage(STAGE_INDEXING, "Indekserer i vector store...")
        yield _emit_progress(STAGE_INDEXING, 10)

        try:
            # Import and use RAGEngine for indexing
            from src.engine.rag import RAGEngine

            # Create engine for this corpus
            # RAGEngine requires docs_path and optionally vector_store_path
            vector_store_path = PROJECT_ROOT / "data" / "vector_store"
            engine = RAGEngine(
                docs_path=str(processed_dir),
                corpus_id=corpus_id,
                vector_store_path=str(vector_store_path),
            )
            from src.engine.indexing import index_jsonl

            index_jsonl(engine, str(jsonl_path))

            yield _emit_progress(STAGE_INDEXING, 100)
            yield _emit_stage(STAGE_INDEXING, "Indeksering færdig", completed=True)

        except Exception as e:
            yield _emit_error(f"Indeksering fejlede: {str(e)}")
            return

        # Stage 4: Citation graph
        yield _emit_stage(STAGE_CITATION_GRAPH, "Bygger citation graph...")
        yield _emit_progress(STAGE_CITATION_GRAPH, 10)

        try:
            graph = CitationGraph.from_corpus(corpus_id)
            graph.save()

            yield _emit_progress(STAGE_CITATION_GRAPH, 100)
            yield _emit_stage(STAGE_CITATION_GRAPH, "Citation graph bygget", completed=True)

        except Exception as e:
            # Citation graph is non-critical - log but continue
            logger.warning("Citation graph build failed for %s: %s", corpus_id, e)
            yield _emit_stage(STAGE_CITATION_GRAPH, f"Citation graph sprunget over: {str(e)}", completed=True)

        # Stage 5: Generate example questions + optional eval cases
        yield _emit_stage(STAGE_EVAL_GENERATION, "Genererer eksempelspørgsmål...")
        yield _emit_progress(STAGE_EVAL_GENERATION, 10)

        try:
            from src.ingestion.example_generator import add_corpus_examples

            success = add_corpus_examples(corpus_id, display_name, celex_number)
            if success:
                yield _emit_progress(STAGE_EVAL_GENERATION, 30)
                logger.info("Generated example questions for %s", corpus_id)
            else:
                logger.warning("Could not generate example questions for %s", corpus_id)

        except Exception as e:
            # Non-critical - log but continue
            logger.warning("Example question generation failed for %s: %s", corpus_id, e)

        # Optional eval cases
        eval_cases_generated = False
        if generate_eval:
            yield _emit_stage(STAGE_EVAL_GENERATION, "Genererer eval cases...")
            yield _emit_progress(STAGE_EVAL_GENERATION, 50)

            try:
                from src.ingestion.eval_generator import add_corpus_eval_cases

                eval_success = add_corpus_eval_cases(
                    corpus_id,
                    display_name,
                    celex_number,
                    num_cases=15,
                )
                if eval_success:
                    eval_cases_generated = True
                    yield _emit_progress(STAGE_EVAL_GENERATION, 100)
                    yield _emit_stage(STAGE_EVAL_GENERATION, "Eval cases genereret", completed=True)
                    logger.info("Generated eval cases for %s", corpus_id)
                else:
                    yield _emit_stage(STAGE_EVAL_GENERATION, "Kunne ikke generere eval cases (fortsætter)", completed=True)
                    logger.warning("Could not generate eval cases for %s", corpus_id)

            except Exception as e:
                # Non-critical - log but continue
                logger.warning("Eval case generation failed for %s: %s", corpus_id, e)
                yield _emit_stage(STAGE_EVAL_GENERATION, f"Eval cases sprunget over: {str(e)}", completed=True)
        else:
            yield _emit_progress(STAGE_EVAL_GENERATION, 100)
            yield _emit_stage(STAGE_EVAL_GENERATION, "Eksempelspørgsmål genereret", completed=True)

        # Stage 6: Update configuration (must happen BEFORE verification eval so corpus is registered)
        yield _emit_stage(STAGE_CONFIG_UPDATE, "Opdaterer konfiguration...")
        yield _emit_progress(STAGE_CONFIG_UPDATE, 10)

        try:
            # Update corpus registry
            registry_path = default_registry_path(PROJECT_ROOT)
            registry = load_registry(registry_path)
            aliases = derive_aliases(corpus_id, display_name)
            upsert_corpus(registry, corpus_id, display_name, aliases)
            save_registry(registry_path, registry)

            # Update corpora inventory with extended metadata
            corpora_path = default_corpora_path(PROJECT_ROOT)
            inv = load_corpora_inventory(corpora_path)

            # Calculate quality metrics from preflight and ingestion result
            total_handled = sum(preflight.handled.values()) if preflight.handled else 0
            total_unhandled = sum(preflight.unhandled.values()) if preflight.unhandled else 0
            total_patterns = total_handled + total_unhandled
            unhandled_pct = (100.0 * total_unhandled / total_patterns) if total_patterns > 0 else 0.0

            # Get structure coverage from ingestion result
            structure_coverage_pct = ingestion_result.structure_coverage_pct if ingestion_result else 0.0

            # Build extra metadata, only including dates if provided
            extra_metadata: dict[str, Any] = {
                "chunks_collection": f"{corpus_id}_documents",
                "max_distance": None,
                "source_url": build_html_url(celex_number),
                "celex_number": celex_number,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                # Quality metrics
                "quality": {
                    "unhandled_patterns": preflight.unhandled if preflight.unhandled else {},
                    "unhandled_count": total_unhandled,
                    "unhandled_pct": round(unhandled_pct, 1),
                    "structure_coverage_pct": round(structure_coverage_pct, 1),
                    "chunk_count": ingestion_result.chunk_count if ingestion_result else 0,
                },
            }
            if fullname:
                extra_metadata["fullname"] = fullname
            if eurovoc_labels:
                extra_metadata["eurovoc_labels"] = eurovoc_labels
            if entry_into_force:
                extra_metadata["entry_into_force"] = entry_into_force
            if last_modified:
                extra_metadata["last_modified"] = last_modified

            upsert_corpus_inventory(
                inv,
                corpus_id,
                display_name=display_name,
                enabled=True,
                extra=extra_metadata,
            )
            save_corpora_inventory(corpora_path, inv)

            # Clear settings cache so new corpus is immediately visible
            clear_config_cache()
            logger.info("Cleared config cache after ingestion of %s", corpus_id)

            yield _emit_progress(STAGE_CONFIG_UPDATE, 100)
            yield _emit_stage(STAGE_CONFIG_UPDATE, "Konfiguration opdateret", completed=True)

        except Exception as e:
            yield _emit_error(f"Konfigurationsopdatering fejlede: {str(e)}")
            return

        # Stage 7: Run verification eval (if eval cases were generated)
        if eval_cases_generated:
            yield _emit_stage(STAGE_EVAL_RUN, "Kører pipeline-verifikation (med retry og model-eskalering fra settings)...")
            yield _emit_progress(STAGE_EVAL_RUN, 5)

            try:
                # Run verification eval and yield results
                event_count = 0
                for event in _run_verification_eval(corpus_id, run_mode=eval_run_mode or "full"):
                    event_count += 1
                    logger.debug("Verification event %d: %s", event_count, event.get("type"))
                    yield event

                logger.info("Verification yielded %d events for %s", event_count, corpus_id)
                yield _emit_stage(STAGE_EVAL_RUN, "Verifikation gennemført", completed=True)

            except Exception as e:
                # Non-critical - log but continue
                logger.warning("Verification eval failed for %s: %s", corpus_id, e)
                yield _emit_stage(STAGE_EVAL_RUN, f"Verifikation sprunget over: {str(e)}", completed=True)
        else:
            logger.info("Skipping verification - eval_cases_generated=%s", eval_cases_generated)

        # All done!
        yield _emit_complete(corpus_id)

    except Exception as e:
        logger.exception("Unexpected error during ingestion of %s", celex_number)
        yield _emit_error(f"Uventet fejl: {str(e)}")
