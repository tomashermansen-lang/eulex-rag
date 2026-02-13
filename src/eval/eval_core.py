# src/eval/eval_core.py
"""Core evaluation logic - unified evaluation for CLI, Dashboard, and Ingestion.

EVAL = PROD Principle:
- All evaluation uses the exact same code path as production (ask.ask())
- Scorers are applied to production results
- Retries and escalation are handled uniformly

Architecture:
- EvalConfig: Immutable configuration for eval runs
- evaluate_cases_iter(): Generator that yields results for each case

Run modes (same as Dashboard):
- retrieval_only: Fast, skip LLM generation (--skip-llm)
- full: Standard with LLM generation
- full_with_judge: Thorough with LLM-as-judge scoring
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterator, Literal, Any

from ..common.config_loader import get_settings_yaml
from ..services import ask
from ..engine.planning import UserProfile
from .reporters import (
    CaseResult,
    EvalSummary,
    RetryStats,
    EscalationStats,
)
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
from .cross_law_scorers import (
    CorpusCoverageScorer,
    SynthesisBalanceScorer,
    CrossReferenceAccuracyScorer,
    ComparisonCompletenessScorer,
    RoutingPrecisionScorer,
)
from .types import GoldenCase, RunMode

@dataclass(frozen=True)
class EvalConfig:
    """Configuration for eval execution.

    Run modes (same as Dashboard):
    - retrieval_only: Fast, skip LLM generation (--skip-llm)
    - full: Standard with LLM generation
    - full_with_judge: Thorough with LLM-as-judge scoring

    Attributes:
        law: Corpus/law to evaluate
        run_mode: One of retrieval_only, full, full_with_judge
        max_retries: Maximum retry attempts for failed cases
        escalation_enabled: Whether to escalate to fallback model
        fallback_model: Model to use for escalation
        verbose: Enable verbose logging
    """
    law: str
    run_mode: RunMode = "full"
    max_retries: int = 3
    escalation_enabled: bool = False
    fallback_model: str | None = None
    verbose: bool = False
    corpus_coverage_threshold: float = 0.8

    @property
    def skip_llm(self) -> bool:
        """Whether to skip LLM generation (dry_run mode)."""
        return self.run_mode == "retrieval_only"

    @property
    def llm_judge(self) -> bool:
        """Whether to apply LLM-as-judge scorers."""
        return self.run_mode == "full_with_judge"


def _build_engine(
    law: str,
    model_override: str | None = None,
) -> ask.RAGEngine:
    """Build RAG engine for evaluation.

    Args:
        law: The corpus/law to use
        model_override: Optional model to use instead of default (for escalation)

    Returns:
        Configured RAGEngine instance
    """
    if model_override:
        from ..common.config_loader import load_settings
        settings = load_settings()
        return ask.RAGEngine(
            docs_path=str(settings.docs_path),
            corpus_id=law,
            chunks_collection=settings.corpora[law].chunks_collection,
            embedding_model=settings.embedding_model,
            chat_model=model_override,
            vector_store_path=str(settings.vector_store_path),
            max_distance=(settings.corpora[law].max_distance
                          if settings.corpora[law].max_distance is not None
                          else settings.rag_max_distance),
            hybrid_vec_k=settings.hybrid_vec_k,
            ranking_weights=settings.ranking_weights,
        )
    else:
        return ask.build_engine(law=law)


def _extract_cited_corpora(references_structured: list[dict]) -> set[str]:
    """Extract unique corpus IDs from references.

    Args:
        references_structured: List of reference dicts with corpus_id field

    Returns:
        Set of corpus IDs that were cited
    """
    corpora = set()
    for ref in references_structured:
        if isinstance(ref, dict):
            corpus_id = ref.get("corpus_id") or ""
            if corpus_id:
                corpora.add(corpus_id)
    return corpora


def _derive_citation_counts(references_structured: list[dict]) -> dict[str, int]:
    """Count references per corpus_id from production output.

    Args:
        references_structured: List of reference dicts with corpus_id field

    Returns:
        Dict mapping corpus_id to reference count
    """
    counts: dict[str, int] = {}
    for ref in references_structured:
        if isinstance(ref, dict):
            corpus_id = ref.get("corpus_id") or ""
            if corpus_id:
                counts[corpus_id] = counts.get(corpus_id, 0) + 1
    return counts


def _build_context_text(references_structured: list[dict]) -> str:
    """Build context text from references for LLM-judge scorers."""
    parts = []
    for ref in references_structured:
        if not isinstance(ref, dict):
            continue
        text = ref.get("chunk_text") or ref.get("text") or ref.get("content") or ""
        if text:
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


def _evaluate_single_case(
    case: GoldenCase,
    config: EvalConfig,
    engine: ask.RAGEngine,
    scorers: list,
) -> CaseResult:
    """Evaluate a single case using production code.

    CRITICAL: This function ONLY calls ask.ask() and applies scorers.
    No reimplementation of retrieval, ranking, or context selection.
    """
    start = time.perf_counter()

    # Resolve profile
    profile = UserProfile.ENGINEERING if case.profile == "ENGINEERING" else UserProfile.LEGAL

    # Call production code - EVAL = PROD
    result = _retry_with_backoff(
        ask.ask,
        question=case.prompt,
        law=config.law,
        user_profile=profile,
        engine=engine,
        contract_min_citations=case.expected.min_citations,
        dry_run=config.skip_llm,
        corpus_scope=case.corpus_scope,
        target_corpora=list(case.target_corpora) if case.target_corpora else None,
    )

    # Convert expected to scorer format
    expected = GoldenExpected(
        must_include_any_of=case.expected.must_include_any_of,
        must_include_any_of_2=case.expected.must_include_any_of_2,
        must_include_all_of=case.expected.must_include_all_of,
    )

    # Apply scorers
    scores: dict[str, Score] = {}
    for scorer in scorers:
        scores[scorer.name] = scorer.score(
            expected=expected,
            retrieval_metrics=result.retrieval_metrics,
            references_structured=result.references_structured,
        )

    # Apply cross-law scorers for non-single corpus_scope cases
    is_cross_law = case.corpus_scope != "single"
    if is_cross_law:
        # Generation-level: what the LLM actually cited
        cited_corpora = _extract_cited_corpora(result.references_structured)
        citation_counts = _derive_citation_counts(result.references_structured)

        # Retrieval-level: what the retrieval pipeline found (all context refs)
        # Used for corpus_coverage, comparison_completeness, routing_precision
        all_refs = (
            result.retrieval_metrics.get("references_structured_all")
            or result.references_structured
        )
        context_corpora = _extract_cited_corpora(all_refs)

        answer_text = result.answer or ""
        synthesis_mode = case.synthesis_mode or "aggregation"

        # corpus_coverage scorer (retrieval-level)
        # Discovery mode uses configurable threshold instead of 100%
        if "corpus_coverage" in case.test_types:
            expected_corpora = set(case.expected.required_corpora) or set(case.target_corpora)
            coverage_threshold = (
                config.corpus_coverage_threshold
                if synthesis_mode == "discovery"
                else None
            )
            scorer = CorpusCoverageScorer()
            scorer_result = scorer.score(
                answer_text=answer_text,
                cited_corpora=context_corpora,
                expected_corpora=expected_corpora,
                min_corpora_cited=case.expected.min_corpora_cited,
                threshold=coverage_threshold,
            )
            scores["corpus_coverage"] = Score(
                passed=scorer_result.passed,
                score=scorer_result.score,
                message=scorer_result.message,
            )

        # synthesis_balance scorer
        if "synthesis_balance" in case.test_types:
            scorer = SynthesisBalanceScorer()
            scorer_result = scorer.score(
                citation_counts=citation_counts,
                synthesis_mode=synthesis_mode,
            )
            scores["synthesis_balance"] = Score(
                passed=scorer_result.passed,
                score=scorer_result.score,
                message=scorer_result.message,
            )

        # cross_reference_accuracy scorer
        if "cross_reference_accuracy" in case.test_types:
            scorer = CrossReferenceAccuracyScorer()
            scorer_result = scorer.score(
                answer_text=answer_text,
                cited_corpora=cited_corpora,
            )
            scores["cross_reference_accuracy"] = Score(
                passed=scorer_result.passed,
                score=scorer_result.score,
                message=scorer_result.message,
            )

        # comparison_completeness scorer (retrieval-level)
        if "comparison_completeness" in case.test_types:
            scorer = ComparisonCompletenessScorer()
            scorer_result = scorer.score(
                cited_corpora=context_corpora,
                target_corpora=set(case.expected.required_corpora),
                synthesis_mode=synthesis_mode,
            )
            scores["comparison_completeness"] = Score(
                passed=scorer_result.passed,
                score=scorer_result.score,
                message=scorer_result.message,
            )

        # routing_precision scorer (retrieval-level, routing mode)
        if "routing_precision" in case.test_types:
            scorer = RoutingPrecisionScorer()
            scorer_result = scorer.score(
                context_corpora=context_corpora,
                expected_laws=set(case.expected.required_corpora),
                synthesis_mode=synthesis_mode,
            )
            scores["routing_precision"] = Score(
                passed=scorer_result.passed,
                score=scorer_result.score,
                message=scorer_result.message,
            )

    # Handle abstain cases in non-full_with_judge modes
    # These cases require LLM-judge to properly evaluate abstention behavior
    is_abstain_case = case.expected.behavior == "abstain"
    if is_abstain_case and not config.llm_judge:
        scores["abstention"] = Score(
            passed=False,
            score=0.0,
            message="Requires full_with_judge mode for proper evaluation",
            details={"not_evaluated": True, "reason": "abstain cases require LLM-judge"},
        )

    # Apply LLM-as-judge scorers if enabled
    if config.llm_judge and not config.skip_llm:
        all_refs = result.retrieval_metrics.get("references_structured_all") or result.references_structured
        context_text = _build_context_text(all_refs)
        answer_text = result.answer or ""

        is_abstain_case = case.expected.behavior == "abstain"

        if is_abstain_case:
            abstention_scorer = AbstentionScorer()
            scores["abstention"] = abstention_scorer.score(
                question=case.prompt,
                answer=answer_text,
            )
            scores["faithfulness"] = Score(
                passed=True,
                score=1.0,
                message="Skipped for abstain case (per UAEval4RAG best practices)",
            )
            scores["answer_relevancy"] = Score(
                passed=True,
                score=1.0,
                message="Skipped for abstain case (per UAEval4RAG best practices)",
            )
        else:
            faithfulness_scorer = FaithfulnessScorer()
            scores["faithfulness"] = faithfulness_scorer.score(
                question=case.prompt,
                answer=answer_text,
                context=context_text,
            )

            relevancy_scorer = AnswerRelevancyScorer()
            scores["answer_relevancy"] = relevancy_scorer.score(
                question=case.prompt,
                answer=answer_text,
            )

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
        answer=result.answer or "",
        references_structured=list(result.references_structured),
    )


def _evaluate_case_with_retries(
    case: GoldenCase,
    config: EvalConfig,
    engine: ask.RAGEngine,
    scorers: list,
) -> CaseResult:
    """Evaluate a single case with retries and optional escalation.

    Encapsulates the full retry + escalation logic for one case.
    Thread-safe: uses only its arguments (no shared mutable state).
    """
    attempt = 0
    last_result: CaseResult | None = None

    # Retry loop with primary model
    while attempt <= config.max_retries:
        try:
            result = _evaluate_single_case(case, config, engine, scorers)

            if result.passed:
                return CaseResult(
                    case_id=result.case_id,
                    profile=result.profile,
                    passed=result.passed,
                    scores=result.scores,
                    duration_ms=result.duration_ms,
                    retrieval_metrics=result.retrieval_metrics,
                    answer=result.answer,
                    references_structured=result.references_structured,
                    retry_count=attempt,
                    escalated=False,
                )

            last_result = result
            attempt += 1

            if attempt <= config.max_retries and config.verbose:
                print(f"  Retrying {case.id} (attempt {attempt}/{config.max_retries})")

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

    # Retries exhausted — check for escalation
    should_escalate = (
        config.escalation_enabled
        and config.fallback_model
        and not config.skip_llm
        and last_result is not None
    )

    if should_escalate:
        faith_score = last_result.scores.get("faithfulness")
        relevancy_score = last_result.scores.get("answer_relevancy")
        is_generation_failure = (
            (faith_score and not faith_score.passed)
            or (relevancy_score and not relevancy_score.passed)
        )

        if is_generation_failure:
            if config.verbose:
                print(f"  Escalating {case.id} to {config.fallback_model}")

            try:
                escalation_engine = _build_engine(
                    config.law, model_override=config.fallback_model
                )
                escalated_result = _evaluate_single_case(
                    case, config, escalation_engine, scorers
                )
                return CaseResult(
                    case_id=escalated_result.case_id,
                    profile=escalated_result.profile,
                    passed=escalated_result.passed,
                    scores=escalated_result.scores,
                    duration_ms=escalated_result.duration_ms,
                    retrieval_metrics=escalated_result.retrieval_metrics,
                    answer=escalated_result.answer,
                    references_structured=escalated_result.references_structured,
                    retry_count=config.max_retries,
                    escalated=True,
                    escalation_model=config.fallback_model,
                )
            except Exception as e:
                return CaseResult(
                    case_id=case.id,
                    profile=case.profile,
                    passed=False,
                    scores={
                        "error": Score(
                            passed=False,
                            score=0.0,
                            message=f"Escalation failed: {e}",
                        )
                    },
                    duration_ms=0,
                    retry_count=config.max_retries,
                    escalated=True,
                    escalation_model=config.fallback_model,
                )

    # No escalation — return failed result
    return CaseResult(
        case_id=last_result.case_id,
        profile=last_result.profile,
        passed=last_result.passed,
        scores=last_result.scores,
        duration_ms=last_result.duration_ms,
        retrieval_metrics=getattr(last_result, "retrieval_metrics", {}),
        answer=last_result.answer,
        references_structured=last_result.references_structured,
        retry_count=config.max_retries,
        escalated=False,
    )


def evaluate_cases_iter(
    cases: list[GoldenCase],
    config: EvalConfig,
    *,
    scorers: list | None = None,
    engine: ask.RAGEngine | None = None,
) -> Iterator[CaseResult | EvalSummary]:
    """Evaluate cases in parallel, yielding results as they complete.

    EVAL = PROD: Uses ask.ask() internally (same as production).

    Args:
        cases: List of golden test cases to evaluate
        config: Evaluation configuration
        scorers: Optional custom scorers (default: standard scorers based on run_mode)
        engine: Optional pre-built engine (will build one if not provided)

    Yields:
        CaseResult for each evaluated case (order may vary)
        EvalSummary as the final item

    Features:
        - Parallel execution via ThreadPoolExecutor
        - Same 3 run modes as Dashboard
        - Retries failed cases up to max_retries
        - Escalates to fallback_model after retries exhausted
        - Applies all configured scorers
    """
    start_time = time.perf_counter()

    # Build engine if not provided
    if engine is None:
        engine = _build_engine(config.law)

    # Setup scorers based on run_mode
    if scorers is None:
        scorers = [AnchorScorer(), PipelineBreakdownScorer()]
        if not config.skip_llm:
            scorers.append(ContractScorer())

    # Track statistics
    results: list[CaseResult] = []
    retry_stats = RetryStats()
    escalation_stats = EscalationStats()

    # Evaluate cases in parallel (bounded by case count and configured concurrency)
    settings = get_settings_yaml()
    configured_concurrency = settings.get("eval", {}).get("default_concurrency", 4)
    max_workers = min(len(cases), configured_concurrency) if cases else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(
                _evaluate_case_with_retries, case, config, engine, scorers
            ): case
            for case in cases
        }

        for future in as_completed(future_to_case):
            final_result = future.result()
            results.append(final_result)
            yield final_result

    # Update retry statistics
    for r in results:
        if r.retry_count > 0:
            retry_stats.cases_with_retries += 1
            retry_stats.total_retries += r.retry_count
            if r.passed and not r.escalated:
                retry_stats.cases_passed_on_retry += 1
            elif not r.passed and not r.escalated:
                retry_stats.cases_failed_after_retries += 1

    # Update escalation statistics
    for r in results:
        if r.escalated:
            escalation_stats.cases_escalated += 1
            escalation_stats.escalated_case_ids.append(r.case_id)
            if r.passed:
                escalation_stats.cases_passed_on_escalation += 1
            else:
                escalation_stats.cases_failed_after_escalation += 1

    # Build and yield summary
    duration_seconds = time.perf_counter() - start_time

    summary = EvalSummary(
        law=config.law,
        total=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        skipped=0,
        duration_seconds=duration_seconds,
        results=results,
        run_mode=config.run_mode,
        retry_stats=retry_stats,
        escalation_stats=escalation_stats,
    )

    yield summary
