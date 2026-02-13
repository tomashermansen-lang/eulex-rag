# src/eval/reporters.py
"""
Reporters for RAG evaluation output.

Handles progress display, failure reporting, and pipeline analysis output.
Separates presentation from evaluation logic.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .scorers import Score


@dataclass
class CaseResult:
    """Result from evaluating a single test case."""
    case_id: str
    profile: str
    passed: bool
    scores: dict[str, Score]  # scorer_name -> Score
    duration_ms: float
    retrieval_metrics: dict[str, Any] = field(default_factory=dict)
    answer: str = ""  # Raw answer text from ask.ask()
    references_structured: list[dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0  # Number of retries (0 = passed on first attempt)
    escalated: bool = False  # True if case was escalated to fallback model
    escalation_model: str | None = None  # Model used for escalation (if any)


@dataclass
class RetryStats:
    """Statistics about retries during evaluation."""
    cases_with_retries: int = 0  # Cases that needed at least one retry
    total_retries: int = 0  # Sum of all retries across all cases
    cases_passed_on_retry: int = 0  # Cases that failed first but passed on retry
    cases_failed_after_retries: int = 0  # Cases that failed even after all retries


@dataclass
class EscalationStats:
    """Statistics about model escalation during evaluation."""
    cases_escalated: int = 0  # Cases that were escalated to fallback model
    cases_passed_on_escalation: int = 0  # Cases that passed after escalation
    cases_failed_after_escalation: int = 0  # Cases that failed even with fallback model
    escalated_case_ids: list[str] = field(default_factory=list)  # IDs of escalated cases

    @property
    def escalation_success_rate(self) -> float:
        """Rate of escalated cases that passed."""
        if self.cases_escalated == 0:
            return 0.0
        return self.cases_passed_on_escalation / self.cases_escalated


@dataclass
class PipelineStageStats:
    """Statistics per pipeline stage - shows where failures occur.

    Maps each scorer to a pipeline stage to help identify bottlenecks.
    No performance overhead - uses existing score data.
    """
    # Retrieval stage: anchor_presence (did we find relevant chunks?)
    retrieval_passed: int = 0
    retrieval_failed: int = 0

    # Augmentation stage: contract_compliance + pipeline_breakdown
    # (did enrichment stages work correctly?)
    augmentation_passed: int = 0
    augmentation_failed: int = 0

    # Generation stage: faithfulness + answer_relevancy
    # (did LLM produce correct output?)
    generation_passed: int = 0
    generation_failed: int = 0

    @property
    def retrieval_total(self) -> int:
        return self.retrieval_passed + self.retrieval_failed

    @property
    def augmentation_total(self) -> int:
        return self.augmentation_passed + self.augmentation_failed

    @property
    def generation_total(self) -> int:
        return self.generation_passed + self.generation_failed

    @property
    def retrieval_rate(self) -> float:
        return self.retrieval_passed / self.retrieval_total if self.retrieval_total > 0 else 0.0

    @property
    def augmentation_rate(self) -> float:
        return self.augmentation_passed / self.augmentation_total if self.augmentation_total > 0 else 0.0

    @property
    def generation_rate(self) -> float:
        return self.generation_passed / self.generation_total if self.generation_total > 0 else 0.0
    
    @classmethod
    def from_results(cls, results: list["CaseResult"]) -> "PipelineStageStats":
        """Aggregate stage stats from case results."""
        stats = cls()
        
        for result in results:
            scores = result.scores
            
            # Retrieval stage: anchor_presence
            anchor = scores.get("anchor_presence")
            if anchor:
                if anchor.passed:
                    stats.retrieval_passed += 1
                else:
                    stats.retrieval_failed += 1
            
            # Augmentation stage: contract_compliance + pipeline_breakdown
            # Both must pass for augmentation to be considered successful
            contract = scores.get("contract_compliance")
            pipeline = scores.get("pipeline_breakdown")
            if contract or pipeline:
                aug_passed = True
                if contract and not contract.passed:
                    aug_passed = False
                if pipeline and not pipeline.passed:
                    aug_passed = False
                if aug_passed:
                    stats.augmentation_passed += 1
                else:
                    stats.augmentation_failed += 1
            
            # Generation stage: faithfulness + answer_relevancy
            # Both must pass for generation to be considered successful
            faith = scores.get("faithfulness")
            relevancy = scores.get("answer_relevancy")
            if faith or relevancy:
                gen_passed = True
                if faith and not faith.passed:
                    gen_passed = False
                if relevancy and not relevancy.passed:
                    gen_passed = False
                if gen_passed:
                    stats.generation_passed += 1
                else:
                    stats.generation_failed += 1

        return stats


@dataclass
class EvalSummary:
    """Summary of an evaluation run."""
    law: str
    total: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    results: list[CaseResult]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    run_mode: str = "full"  # retrieval_only, full, or full_with_judge
    retry_stats: RetryStats = field(default_factory=RetryStats)
    stage_stats: PipelineStageStats = field(default_factory=PipelineStageStats)
    escalation_stats: EscalationStats = field(default_factory=EscalationStats)

    @property
    def passed_primary(self) -> int:
        """Cases that passed with primary model only."""
        return sum(1 for r in self.results if r.passed and not r.escalated)

    @property
    def passed_escalated(self) -> int:
        """Cases that passed only after escalation."""
        return sum(1 for r in self.results if r.passed and r.escalated)

    def __post_init__(self) -> None:
        """Compute stage stats from results if not provided."""
        # Only compute if stage_stats is default (no data)
        if self.stage_stats.retrieval_total == 0 and self.results:
            self.stage_stats = PipelineStageStats.from_results(self.results)


class ProgressReporter:
    """Reports progress during evaluation with optional progress bar."""
    
    def __init__(self, total: int, show_progress: bool = True, output: TextIO = sys.stdout):
        self.total = total
        self.show_progress = show_progress
        self.output = output
        self.current = 0
        self._start_time = datetime.now(timezone.utc)
    
    def start(self, law: str) -> None:
        """Called when evaluation starts."""
        if self.show_progress:
            self.output.write(f"\n🔍 Evaluating {law} ({self.total} cases)\n")
            self.output.flush()
    
    def update(self, case_id: str, passed: bool, message: str = "") -> None:
        """Called after each case is evaluated."""
        self.current += 1
        
        if self.show_progress:
            icon = "✓" if passed else "✗"
            pct = (self.current / self.total) * 100 if self.total > 0 else 100
            status = f"[{self.current}/{self.total}] {pct:5.1f}% {icon} {case_id}"
            if message and not passed:
                status += f" — {message}"
            self.output.write(f"\r{status:<80}\n")
            self.output.flush()
    
    def finish(self, summary: EvalSummary) -> None:
        """Called when evaluation completes."""
        if self.show_progress:
            elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
            self.output.write(f"\n{'='*60}\n")
            self.output.write(f"📊 Results for {summary.law}:\n")
            self.output.write(f"   Passed: {summary.passed}/{summary.total}\n")
            self.output.write(f"   Failed: {summary.failed}/{summary.total}\n")
            if summary.skipped > 0:
                self.output.write(f"   Skipped: {summary.skipped}\n")
            self.output.write(f"   Duration: {elapsed:.1f}s\n")
            
            # Show pipeline stage pass rates
            ss = summary.stage_stats
            if ss.retrieval_total > 0 or ss.generation_total > 0:
                self.output.write(f"\n📈 Pipeline Stage Pass Rates:\n")
                if ss.retrieval_total > 0:
                    ret_pct = ss.retrieval_rate * 100
                    ret_icon = "✓" if ss.retrieval_failed == 0 else "⚠"
                    self.output.write(f"   {ret_icon} Retrieval:    {ss.retrieval_passed}/{ss.retrieval_total} ({ret_pct:.0f}%)\n")
                if ss.augmentation_total > 0:
                    aug_pct = ss.augmentation_rate * 100
                    aug_icon = "✓" if ss.augmentation_failed == 0 else "⚠"
                    self.output.write(f"   {aug_icon} Augmentation: {ss.augmentation_passed}/{ss.augmentation_total} ({aug_pct:.0f}%)\n")
                if ss.generation_total > 0:
                    gen_pct = ss.generation_rate * 100
                    gen_icon = "✓" if ss.generation_failed == 0 else "⚠"
                    self.output.write(f"   {gen_icon} Generation:   {ss.generation_passed}/{ss.generation_total} ({gen_pct:.0f}%)\n")
            
            # Show retry stats if any retries occurred
            if summary.retry_stats.total_retries > 0:
                self.output.write(f"\n🔄 Retry Statistics:\n")
                self.output.write(f"   Cases needing retries: {summary.retry_stats.cases_with_retries}\n")
                self.output.write(f"   Total retry attempts: {summary.retry_stats.total_retries}\n")
                if summary.retry_stats.cases_passed_on_retry > 0:
                    self.output.write(f"   ✓ Passed on retry: {summary.retry_stats.cases_passed_on_retry}\n")
                if summary.retry_stats.cases_failed_after_retries > 0:
                    self.output.write(f"   ✗ Failed after retries: {summary.retry_stats.cases_failed_after_retries}\n")

            # Show escalation stats if any cases were escalated
            es = summary.escalation_stats
            if es.cases_escalated > 0:
                escalation_rate = es.cases_escalated / summary.total * 100 if summary.total > 0 else 0
                self.output.write(f"\n🚀 Model Escalation:\n")
                self.output.write(f"   Cases escalated: {es.cases_escalated}/{summary.total} ({escalation_rate:.1f}%)\n")
                self.output.write(f"   ✓ Passed on escalation: {es.cases_passed_on_escalation}\n")
                if es.cases_failed_after_escalation > 0:
                    self.output.write(f"   ✗ Failed after escalation: {es.cases_failed_after_escalation}\n")
                # Show breakdown: passed_primary vs passed_escalated
                self.output.write(f"\n   Pass breakdown:\n")
                self.output.write(f"      Primary model: {summary.passed_primary}\n")
                self.output.write(f"      After escalation: {summary.passed_escalated}\n")
                # Warn if escalation rate is high
                from ..common.config_loader import get_settings_yaml
                settings = get_settings_yaml()
                threshold = settings.get("eval", {}).get("model_escalation", {}).get("escalation_alert_threshold", 0.10)
                if escalation_rate / 100 > threshold:
                    self.output.write(f"\n   ⚠️  Escalation rate ({escalation_rate:.1f}%) exceeds threshold ({threshold*100:.0f}%)\n")
                    self.output.write(f"      Consider investigating prompt/retrieval issues.\n")

            self.output.write(f"{'='*60}\n\n")
            self.output.flush()


class FailureReporter:
    """Reports details about failed cases."""
    
    def __init__(self, output: TextIO = sys.stdout, verbose: bool = False):
        self.output = output
        self.verbose = verbose
    
    def report_failure(self, result: CaseResult) -> None:
        """Report a single failure."""
        self.output.write(f"\n❌ FAILED: {result.case_id} ({result.profile})\n")
        
        for scorer_name, score in result.scores.items():
            if not score.passed:
                self.output.write(f"   {scorer_name}: {score.message}\n")
                if self.verbose and score.details:
                    # Show positions for anchor failures
                    positions = score.details.get("positions")
                    if positions:
                        pos_str = " ".join(f"{k}={v}" for k, v in positions.items())
                        self.output.write(f"   Positions: {pos_str}\n")
                    
                    # Show present anchors
                    anchors = score.details.get("anchors_in_context")
                    if anchors:
                        self.output.write(f"   Found: {', '.join(anchors[:10])}")
                        if len(anchors) > 10:
                            self.output.write(f" (+{len(anchors)-10} more)")
                        self.output.write("\n")
        
        self.output.flush()
    
    def report_summary(self, failures: list[CaseResult]) -> None:
        """Report summary of all failures."""
        if not failures:
            self.output.write("\n✅ All cases passed!\n")
            return

        self.output.write(f"\n{'='*60}\n")
        self.output.write(f"❌ {len(failures)} FAILED CASES:\n")
        self.output.write(f"{'='*60}\n")

        for result in failures:
            # Check for error score first (exception during evaluation)
            error_score = result.scores.get("error")
            if error_score:
                self.output.write(f"  • {result.case_id}: ERROR - {error_score.message}\n")
                continue

            # Check anchor presence failure
            anchor_score = result.scores.get("anchor_presence")
            if anchor_score and not anchor_score.passed:
                self.output.write(f"  • {result.case_id}: {anchor_score.message}\n")
                continue

            # Show all failed scorers with their scores and messages
            failed_scores = [(name, s) for name, s in result.scores.items() if not s.passed]
            if failed_scores:
                self.output.write(f"  • {result.case_id}:\n")
                for name, score in failed_scores:
                    # Format score value
                    score_str = f"(score: {score.score:.2f})" if score.score is not None else ""
                    # Truncate long messages
                    msg = score.message or "(no message)"
                    if len(msg) > 80:
                        msg = msg[:77] + "..."
                    self.output.write(f"      - {name} {score_str}: {msg}\n")
            else:
                self.output.write(f"  • {result.case_id}: (unknown failure)\n")

        self.output.flush()


class PipelineAnalysisReporter:
    """Reports detailed pipeline analysis for debugging."""
    
    def __init__(self, output: TextIO = sys.stdout):
        self.output = output
    
    def report(self, result: CaseResult, include_full: bool = False) -> None:
        """Report pipeline analysis for a case."""
        breakdown_score = result.scores.get("pipeline_breakdown")
        if not breakdown_score:
            return

        details = breakdown_score.details
        stages = details.get("stages") or {}

        self.output.write(f"\n📊 Pipeline Analysis: {result.case_id}\n")
        self.output.write(f"{'─'*50}\n")

        # Step 1: Vector Retrieval
        vec = stages.get("1_vector_retrieval") or {}
        vec_count = vec.get("count", "N/A")
        self.output.write(f"  1️⃣  Vector Retrieval: {vec_count} chunks\n")
        if vec.get("best_distance") is not None:
            self.output.write(f"      └─ Best distance: {vec['best_distance']:.4f}\n")

        # Step 2: Sibling Expansion (not in modular pipeline yet)
        self.output.write(f"  2️⃣  Sibling Expansion: N/A (modular pipeline)\n")

        # Step 3: Anchor hints (from citation expansion stage)
        cite_stage = stages.get("2_citation_expansion") or {}
        hint_anchors = cite_stage.get("hint_anchors") or []
        if hint_anchors:
            self.output.write(f"  3️⃣  Anchor Hints: {', '.join(hint_anchors[:5])}\n")
        else:
            self.output.write(f"  3️⃣  Anchor Hints: none detected\n")

        # Step 4: Citation expansion
        expanded = cite_stage.get("articles") or []
        chunks_injected = cite_stage.get("chunks_injected", 0)
        if expanded or chunks_injected > 0:
            self.output.write(f"  4️⃣  Citation Expansion: {len(expanded)} anchors, +{chunks_injected} chunks\n")
            if expanded:
                self.output.write(f"      └─ Anchors: {', '.join(expanded[:5])}\n")
        else:
            self.output.write(f"  4️⃣  Citation Expansion: none\n")

        # Step 5: Hybrid Rerank
        hybrid = stages.get("3_hybrid_rerank") or {}
        if hybrid.get("enabled", True):  # enabled by default in modular
            top_scores = hybrid.get("top_scores") or []
            if top_scores:
                self.output.write(f"  5️⃣  Hybrid Rerank: ✓ (4-factor scoring)\n")
                top = top_scores[0] if top_scores else {}
                if top:
                    self.output.write(f"      └─ Top: {top.get('anchor', 'N/A')} (vec={top.get('vec', 0):.2f}, bm25={top.get('bm25', 0):.2f}, cite={top.get('cite', 0):.2f}, role={top.get('role', 0):.2f})\n")
            else:
                self.output.write(f"  5️⃣  Hybrid Rerank: ✓ (enabled)\n")
        else:
            self.output.write(f"  5️⃣  Hybrid Rerank: disabled\n")

        # Step 6: Final context
        self.output.write(f"  6️⃣  Final Context: {details.get('final_context_count', 'N/A')} chunks\n")
        anchors = details.get("anchors_in_context") or []
        if anchors:
            self.output.write(f"      └─ Anchors: {', '.join(anchors[:10])}\n")

        # Step 7: LLM call
        llm_called = details.get("llm_called", True)
        profile = details.get("user_profile", "?")
        if llm_called:
            self.output.write(f"  7️⃣  LLM Called: ✓ (profile: {profile})\n")
        else:
            self.output.write(f"  7️⃣  LLM Called: ✗ (dry_run)\n")

        # Step 8: LLM-Judge scores (if available)
        faith = result.scores.get("faithfulness")
        relevancy = result.scores.get("answer_relevancy")
        if faith or relevancy:
            self.output.write(f"  8️⃣  LLM-Judge Scores:\n")
            if faith:
                icon = "✓" if faith.passed else "✗"
                self.output.write(f"      └─ Faithfulness: {faith.score:.1%} {icon}\n")
            if relevancy:
                icon = "✓" if relevancy.passed else "✗"
                self.output.write(f"      └─ Answer Relevancy: {relevancy.score:.1%} {icon}\n")


class LLMJudgeProgressReporter:
    """Reports progress for LLM-as-judge scorers with detailed step visibility."""
    
    def __init__(self, output: TextIO = sys.stdout, verbose: bool = False):
        self.output = output
        self.verbose = verbose
    
    def on_scorer_start(self, scorer_name: str, case_id: str) -> None:
        """Called when an LLM-judge scorer starts."""
        if self.verbose:
            self.output.write(f"  ├─ {scorer_name}:\n")
            self.output.flush()
    
    def on_step(self, scorer_name: str, step: str, current: int = 0, total: int = 0) -> None:
        """Called for each step in the scoring process."""
        if self.verbose:
            if total > 0:
                self.output.write(f"  │    ├─ {step} {current}/{total}\n")
            else:
                self.output.write(f"  │    ├─ {step}\n")
            self.output.flush()
    
    def on_claim_verified(
        self, claim_index: int, total_claims: int, supported: bool, claim_preview: str = ""
    ) -> None:
        """Called when a claim is verified (for Faithfulness scorer)."""
        if self.verbose:
            icon = "✓" if supported else "✗"
            preview = f" ({claim_preview[:40]}...)" if claim_preview else ""
            self.output.write(f"  │    ├─ Verifying claim {claim_index}/{total_claims}... {icon}{preview}\n")
            self.output.flush()
    
    def on_scorer_complete(
        self, scorer_name: str, score: float, passed: bool, details: str = ""
    ) -> None:
        """Called when an LLM-judge scorer completes."""
        if self.verbose:
            icon = "✓" if passed else "✗"
            self.output.write(f"  │    └─ Score: {score:.2f} {icon}")
            if details:
                self.output.write(f" {details}")
            self.output.write("\n")
            self.output.flush()


class JsonReporter:
    """Writes evaluation results to JSON file."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
    
    def write(self, summary: EvalSummary) -> None:
        """Write summary to JSON file."""
        ss = summary.stage_stats
        output = {
            "meta": {
                "law": summary.law,
                "timestamp": summary.timestamp,
                "duration_seconds": summary.duration_seconds,
                "run_mode": summary.run_mode,
            },
            "summary": {
                "total": summary.total,
                "passed": summary.passed,
                "failed": summary.failed,
                "skipped": summary.skipped,
                "pass_rate": summary.passed / summary.total if summary.total > 0 else 0,
            },
            "stage_stats": {
                "retrieval": {
                    "passed": ss.retrieval_passed,
                    "failed": ss.retrieval_failed,
                    "total": ss.retrieval_total,
                    "pass_rate": ss.retrieval_rate,
                },
                "augmentation": {
                    "passed": ss.augmentation_passed,
                    "failed": ss.augmentation_failed,
                    "total": ss.augmentation_total,
                    "pass_rate": ss.augmentation_rate,
                },
                "generation": {
                    "passed": ss.generation_passed,
                    "failed": ss.generation_failed,
                    "total": ss.generation_total,
                    "pass_rate": ss.generation_rate,
                },
            },
            "retry_stats": {
                "cases_with_retries": summary.retry_stats.cases_with_retries,
                "total_retries": summary.retry_stats.total_retries,
                "cases_passed_on_retry": summary.retry_stats.cases_passed_on_retry,
                "cases_failed_after_retries": summary.retry_stats.cases_failed_after_retries,
            },
            "escalation_stats": {
                "cases_escalated": summary.escalation_stats.cases_escalated,
                "cases_passed_on_escalation": summary.escalation_stats.cases_passed_on_escalation,
                "cases_failed_after_escalation": summary.escalation_stats.cases_failed_after_escalation,
                "escalated_case_ids": summary.escalation_stats.escalated_case_ids,
            },
            "results": [
                self._serialize_result(r)
                for r in summary.results
            ],
        }

        # Write to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    def _serialize_result(self, r: CaseResult) -> dict:
        """Serialize a single case result with run_meta for EVAL = PROD parity."""
        run_meta = r.retrieval_metrics.get("run") or {}

        result = {
            "case_id": r.case_id,
            "profile": r.profile,
            "passed": r.passed,
            "duration_ms": r.duration_ms,
            "retry_count": r.retry_count,
            "escalated": r.escalated,
            "escalation_model": r.escalation_model,
            "scores": {
                name: {
                    "passed": s.passed,
                    "score": s.score,
                    "message": s.message,
                    "details": s.details,
                }
                for name, s in r.scores.items()
            },
            # EVAL = PROD: Include production run_meta for full visibility
            "run_meta": {
                "context_positioning": run_meta.get("context_positioning"),
                "warnings": run_meta.get("warnings"),
                "abstain": run_meta.get("abstain", False),
                "anchors_in_top_k": run_meta.get("anchors_in_top_k"),
            },
        }
        return result


class FaithfulnessDebugReporter:
    """Reports detailed faithfulness debug output per claim."""
    
    def __init__(self, output: TextIO = sys.stdout):
        self.output = output
    
    def report(self, result: CaseResult) -> None:
        """Report faithfulness debug info for a case."""
        faithfulness = result.scores.get("faithfulness")
        if not faithfulness:
            return
        
        details = faithfulness.details
        claims = details.get("claims", [])
        verifications = details.get("verifications", [])
        
        if not claims:
            return
        
        self.output.write(f"\n🔍 Faithfulness Debug: {result.case_id}\n")
        self.output.write(f"{'─'*60}\n")
        self.output.write(f"  Score: {faithfulness.score:.1%} ({details.get('supported_count', 0)}/{details.get('total_claims', 0)} claims)\n")
        self.output.write(f"  Threshold: {details.get('threshold', 0.8):.0%}\n")
        self.output.write(f"  Status: {'✓ PASS' if faithfulness.passed else '✗ FAIL'}\n\n")
        
        self.output.write(f"  Claims Analysis:\n")
        self.output.write(f"  {'─'*56}\n")
        
        for i, verification in enumerate(verifications, 1):
            claim = verification.get("claim", "")
            supported = verification.get("supported", False)
            explanation = verification.get("explanation", "")
            
            icon = "✓" if supported else "✗"
            status = "SUPPORTED" if supported else "NOT SUPPORTED"
            
            # Wrap claim text
            claim_short = claim[:100] + "..." if len(claim) > 100 else claim
            
            self.output.write(f"  {i}. [{icon}] {status}\n")
            self.output.write(f"     Claim: \"{claim_short}\"\n")
            self.output.write(f"     Reason: {explanation}\n\n")
        
        # Show unsupported claims summary
        unsupported = details.get("unsupported_claims", [])
        if unsupported:
            self.output.write(f"  ⚠️  Unsupported Claims Summary:\n")
            for claim in unsupported:
                self.output.write(f"     • {claim[:80]}{'...' if len(claim) > 80 else ''}\n")
        
        self.output.write(f"{'─'*60}\n")
        self.output.flush()


class ProgressionTracker:
    """Tracks evaluation progression over time by appending to a central JSON file.
    
    This enables tracking how pass rates improve/regress over development iterations.
    Each run appends a new entry with summary stats, enabling trend analysis.
    """
    
    def __init__(self, tracking_file: Path):
        """Initialize tracker with path to tracking file.
        
        Args:
            tracking_file: Path to the progression tracking JSON file
        """
        self.tracking_file = tracking_file
    
    def record(self, summary: EvalSummary) -> None:
        """Record evaluation summary to progression tracking file.
        
        Args:
            summary: The evaluation summary to record
        """
        # Load existing entries
        entries = []
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, "r", encoding="utf-8") as f:
                    entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                entries = []
        
        # Create new entry
        ss = summary.stage_stats
        entry = {
            "timestamp": summary.timestamp,
            "law": summary.law,
            "run_mode": summary.run_mode,
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "pass_rate": summary.passed / summary.total if summary.total > 0 else 0.0,
            "duration_seconds": summary.duration_seconds,
            "stage_stats": {
                "retrieval": {"passed": ss.retrieval_passed, "failed": ss.retrieval_failed, "rate": ss.retrieval_rate},
                "augmentation": {"passed": ss.augmentation_passed, "failed": ss.augmentation_failed, "rate": ss.augmentation_rate},
                "generation": {"passed": ss.generation_passed, "failed": ss.generation_failed, "rate": ss.generation_rate},
            },
            "retry_stats": {
                "cases_with_retries": summary.retry_stats.cases_with_retries,
                "total_retries": summary.retry_stats.total_retries,
                "cases_passed_on_retry": summary.retry_stats.cases_passed_on_retry,
                "cases_failed_after_retries": summary.retry_stats.cases_failed_after_retries,
            },
            "escalation_stats": {
                "cases_escalated": summary.escalation_stats.cases_escalated,
                "cases_passed_on_escalation": summary.escalation_stats.cases_passed_on_escalation,
                "cases_failed_after_escalation": summary.escalation_stats.cases_failed_after_escalation,
                "escalated_case_ids": summary.escalation_stats.escalated_case_ids,
            },
            # Track which cases failed (for regression detection)
            "failed_cases": [r.case_id for r in summary.results if not r.passed],
        }
        
        # Append to entries
        entries.append(entry)
        
        # Write back
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tracking_file, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    
    def get_recent_history(self, law: str | None = None, limit: int = 10) -> list[dict]:
        """Get recent evaluation history.
        
        Args:
            law: Optional filter by law (None = all laws)
            limit: Maximum number of entries to return
            
        Returns:
            List of recent entries, newest first
        """
        if not self.tracking_file.exists():
            return []
        
        try:
            with open(self.tracking_file, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
        
        # Filter by law if specified
        if law:
            entries = [e for e in entries if e.get("law") == law]
        
        # Return most recent first
        return list(reversed(entries[-limit:]))
    
    def print_history(self, law: str | None = None, limit: int = 10, output=None) -> None:
        """Print recent evaluation history.
        
        Args:
            law: Optional filter by law (None = all laws)
            limit: Maximum number of entries to show
            output: Output stream (default: stdout)
        """
        import sys
        if output is None:
            output = sys.stdout
        
        entries = self.get_recent_history(law=law, limit=limit)
        
        if not entries:
            output.write(f"📈 No progression history found{f' for {law}' if law else ''}\n")
            return
        
        output.write(f"\n📈 Progression History{f' for {law}' if law else ' (all laws)'}:\n")
        output.write(f"{'─'*85}\n")
        output.write(f"{'Timestamp':<25} {'Law':<10} {'Mode':<15} {'Passed':>8} {'Rate':>8} {'Retries':>8}\n")
        output.write(f"{'─'*85}\n")

        for entry in entries:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            law_name = entry.get("law", "")
            run_mode = entry.get("run_mode", "full")
            passed = f"{entry.get('passed', 0)}/{entry.get('total', 0)}"
            rate = f"{entry.get('pass_rate', 0):.1%}"
            retries = entry.get("retry_stats", {}).get("total_retries", 0)

            output.write(f"{ts:<25} {law_name:<10} {run_mode:<15} {passed:>8} {rate:>8} {retries:>8}\n")
        
        output.write(f"{'─'*70}\n")
        output.flush()
