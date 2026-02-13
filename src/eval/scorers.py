# src/eval/scorers.py
"""
Scorers for RAG evaluation.

Each scorer reads from ask.AskResult.retrieval_metrics which contains
all pipeline data from production code. No reimplementation of production logic.

Following Braintrust/DeepEval model: Scorer = f(expected, actual) -> Score
"""
from __future__ import annotations

import json
import os
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Rate Limit Tracking
# ---------------------------------------------------------------------------
class RateLimitTracker:
    """Thread-safe tracker for OpenAI rate limit events and token usage."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._events: list[dict] = []
        self._total_hits = 0
        self._total_retries = 0
        self._verbose = False
        # Token tracking
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._request_count = 0
    
    def set_verbose(self, verbose: bool):
        """Enable/disable live rate limit printing."""
        with self._lock:
            self._verbose = verbose
    
    @property
    def hit_count(self) -> int:
        """Get current hit count (thread-safe)."""
        with self._lock:
            return self._total_hits
    
    def record_hit(self, attempt: int, wait_time: float, context: str = ""):
        """Record a rate limit hit and optionally print live progress."""
        with self._lock:
            self._total_hits += 1
            self._total_retries += 1
            self._events.append({
                "time": datetime.now().isoformat(),
                "attempt": attempt,
                "wait_time": wait_time,
                "context": context,
            })
            hit_num = self._total_hits
            verbose = self._verbose
        
        # Print outside lock to avoid blocking
        if verbose:
            print(f"  ⚠️  Rate limit #{hit_num}: retry {attempt}, waiting {wait_time:.1f}s{' (' + context + ')' if context else ''}")
    
    def record_usage(self, prompt_tokens: int, completion_tokens: int):
        """Record token usage from an API call."""
        with self._lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._request_count += 1
    
    def get_stats(self) -> dict:
        """Get rate limit statistics."""
        with self._lock:
            return {
                "total_hits": self._total_hits,
                "total_retries": self._total_retries,
                "events": self._events[-20:],  # Last 20 events
                "total_prompt_tokens": self._total_prompt_tokens,
                "total_completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
                "request_count": self._request_count,
            }
    
    def reset(self):
        """Reset the tracker."""
        with self._lock:
            self._events.clear()
            self._total_hits = 0
            self._total_retries = 0
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._request_count = 0
    
    def print_summary(self):
        """Print a summary of rate limit events and token usage."""
        with self._lock:
            total_tokens = self._total_prompt_tokens + self._total_completion_tokens
            
            if self._total_hits == 0:
                print(f"  📊 Rate limits: 0 hits")
            else:
                print(f"  ⚠️  Rate limits: {self._total_hits} hits, {self._total_retries} total retries")
                if self._events:
                    total_wait = sum(e["wait_time"] for e in self._events)
                    print(f"      └─ Total wait time: {total_wait:.1f}s")
            
            if self._request_count > 0:
                print(f"  📊 Token usage: {total_tokens:,} total ({self._total_prompt_tokens:,} prompt, {self._total_completion_tokens:,} completion)")
                print(f"      └─ {self._request_count} API calls, avg {total_tokens // self._request_count:,} tokens/call")


# Global tracker instance
_rate_limit_tracker = RateLimitTracker()


def get_rate_limit_tracker() -> RateLimitTracker:
    """Get the global rate limit tracker."""
    return _rate_limit_tracker

from .cache import get_cache
from .prompts import (
    ANSWER_RELEVANCY_PROMPT,
    EXTRACT_CLAIMS_PROMPT,
    VERIFY_CLAIMS_PROMPT,
)


@dataclass(frozen=True)
class Score:
    """Result from a scorer."""
    passed: bool
    score: float  # 0.0 to 1.0
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoldenExpected:
    """Expected values from a golden test case."""
    must_include_any_of: list[str] = field(default_factory=list)
    must_include_any_of_2: list[str] = field(default_factory=list)
    must_include_all_of: list[str] = field(default_factory=list)


def _normalize_anchor(anchor: str) -> str:
    """Normalize anchor for comparison (lowercase, no spaces)."""
    return re.sub(r"\s+", "", str(anchor or "")).strip().lower()


class Scorer(ABC):
    """Base class for all scorers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this scorer."""
        pass
    
    @abstractmethod
    def score(
        self,
        *,
        expected: GoldenExpected,
        retrieval_metrics: dict[str, Any],
        references_structured: list[dict[str, Any]],
    ) -> Score:
        """
        Evaluate the result against expected values.
        
        Args:
            expected: Expected anchors from golden test case
            retrieval_metrics: From ask.AskResult.retrieval_metrics (includes run_meta)
            references_structured: From ask.AskResult.references_structured (final context)
        
        Returns:
            Score with pass/fail, 0-1 score, and details
        """
        pass


class AnchorScorer(Scorer):
    """
    Scores anchor presence in final context.
    
    Uses production data:
    - run_meta.anchors_in_top_k: anchors actually in final LLM context
    - references_structured: the actual final context chunks
    
    CRITICAL: We check what's in the FINAL context, not raw retrieval.
    This is what the LLM actually sees.
    """
    
    @property
    def name(self) -> str:
        return "anchor_presence"
    
    def score(
        self,
        *,
        expected: GoldenExpected,
        retrieval_metrics: dict[str, Any],
        references_structured: list[dict[str, Any]],
    ) -> Score:
        # Get anchors from production's final context calculation
        run_meta = retrieval_metrics.get("run") or {}
        anchors_in_context = set(
            _normalize_anchor(a) for a in (run_meta.get("anchors_in_top_k") or [])
        )
        
        # ALSO add anchors from retrieved_metadatas (what LLM actually sees in context)
        # This catches annexes/articles in context that LLM didn't explicitly cite
        for meta in (retrieval_metrics.get("retrieved_metadatas") or []):
            if not isinstance(meta, dict):
                continue
            corpus_id = meta.get("corpus_id", "")
            if meta.get("article"):
                anchors_in_context.add(_normalize_anchor(f"article:{meta['article']}"))
                # Corpus-qualified: enables cross-law anchor disambiguation
                if corpus_id:
                    anchors_in_context.add(_normalize_anchor(f"{corpus_id}:article:{meta['article']}"))
            if meta.get("recital"):
                anchors_in_context.add(_normalize_anchor(f"recital:{meta['recital']}"))
                if corpus_id:
                    anchors_in_context.add(_normalize_anchor(f"{corpus_id}:recital:{meta['recital']}"))
            if meta.get("annex"):
                anchors_in_context.add(_normalize_anchor(f"annex:{meta['annex']}"))
                if corpus_id:
                    anchors_in_context.add(_normalize_anchor(f"{corpus_id}:annex:{meta['annex']}"))
        
        # Also add anchors from references_structured (has corpus_id for cross-law)
        for ref in references_structured:
            if not isinstance(ref, dict):
                continue
            ref_corpus = ref.get("corpus_id", "")
            if ref.get("article"):
                anchors_in_context.add(_normalize_anchor(f"article:{ref['article']}"))
                if ref_corpus:
                    anchors_in_context.add(_normalize_anchor(f"{ref_corpus}:article:{ref['article']}"))
            if ref.get("recital"):
                anchors_in_context.add(_normalize_anchor(f"recital:{ref['recital']}"))
                if ref_corpus:
                    anchors_in_context.add(_normalize_anchor(f"{ref_corpus}:recital:{ref['recital']}"))
            if ref.get("annex"):
                anchors_in_context.add(_normalize_anchor(f"annex:{ref['annex']}"))
                if ref_corpus:
                    anchors_in_context.add(_normalize_anchor(f"{ref_corpus}:annex:{ref['annex']}"))

        # Use combined anchors as source of truth
        present_anchors = anchors_in_context
        
        # Evaluate constraints
        expected_any_of = [_normalize_anchor(a) for a in expected.must_include_any_of]
        expected_any_of_2 = [_normalize_anchor(a) for a in expected.must_include_any_of_2]
        expected_all_of = [_normalize_anchor(a) for a in expected.must_include_all_of]
        
        any_of_ok = (not expected_any_of) or any(a in present_anchors for a in expected_any_of)
        any_of_2_ok = (not expected_any_of_2) or any(a in present_anchors for a in expected_any_of_2)
        all_of_ok = (not expected_all_of) or all(a in present_anchors for a in expected_all_of)
        
        passed = any_of_ok and any_of_2_ok and all_of_ok
        
        # Compute missing anchors for diagnostics
        missing: list[str] = []
        if expected_any_of and not any_of_ok:
            missing.append(f"missing_any_of={expected.must_include_any_of}")
        if expected_any_of_2 and not any_of_2_ok:
            missing.append(f"missing_any_of_2={expected.must_include_any_of_2}")
        if expected_all_of:
            missing_all = [a for a, norm in zip(expected.must_include_all_of, expected_all_of) 
                          if norm not in present_anchors]
            if missing_all:
                missing.append(f"missing_all_of={missing_all}")
        
        # Compute position info for diagnostics
        positions: dict[str, int | None] = {}
        all_expected = list(dict.fromkeys([*expected_any_of, *expected_any_of_2, *expected_all_of]))
        for anchor in all_expected:
            positions[anchor] = self._find_position(anchor, references_structured)
        
        # Score = fraction of expected anchors found
        total_expected = len(all_expected)
        found = sum(1 for a in all_expected if a in present_anchors)
        score_val = found / total_expected if total_expected > 0 else 1.0
        
        return Score(
            passed=passed,
            score=score_val,
            message=" | ".join(missing) if missing else "all anchors present",
            details={
                "anchors_in_context": sorted(present_anchors),
                "anchors_from_refs": sorted(present_anchors),
                "positions": positions,
                "expected_any_of": expected.must_include_any_of,
                "expected_any_of_2": expected.must_include_any_of_2,
                "expected_all_of": expected.must_include_all_of,
                "any_of_ok": any_of_ok,
                "any_of_2_ok": any_of_2_ok,
                "all_of_ok": all_of_ok,
            }
        )
    
    def _find_position(self, anchor: str, refs: list[dict[str, Any]]) -> int | None:
        """Find 1-based position of anchor in references_structured."""
        want = _normalize_anchor(anchor)
        for i, ref in enumerate(refs, start=1):
            if not isinstance(ref, dict):
                continue
            found: set[str] = set()
            if ref.get("article"):
                found.add(_normalize_anchor(f"article:{ref['article']}"))
            if ref.get("recital"):
                found.add(_normalize_anchor(f"recital:{ref['recital']}"))
            if ref.get("annex"):
                found.add(_normalize_anchor(f"annex:{ref['annex']}"))
            if want in found:
                return i
        return None


class ContractScorer(Scorer):
    """
    Scores ENGINEERING profile contract compliance.
    
    Uses production data:
    - run_meta.contract_validation: results from contract validation
    - retrieval_metrics.references_used_in_answer: citation tracking
    
    Checks:
    - Minimum citations met
    - Contract violations detected
    """
    
    @property
    def name(self) -> str:
        return "contract_compliance"
    
    def score(
        self,
        *,
        expected: GoldenExpected,
        retrieval_metrics: dict[str, Any],
        references_structured: list[dict[str, Any]],
    ) -> Score:
        run_meta = retrieval_metrics.get("run") or {}
        
        # Get contract validation from production
        contract_validation = run_meta.get("contract_validation") or {}
        contract_passed = contract_validation.get("passed", True)
        violations = contract_validation.get("violations") or []
        
        # Get citation info
        refs_used = retrieval_metrics.get("references_used_in_answer") or []
        citation_count = len(refs_used)
        
        message = "contract satisfied" if contract_passed else f"violations: {violations}"
        
        return Score(
            passed=contract_passed,
            score=1.0 if contract_passed else 0.0,
            message=message,
            details={
                "contract_validation": contract_validation,
                "citation_count": citation_count,
                "violations": violations,
            }
        )


class PipelineBreakdownScorer(Scorer):
    """
    Provides pipeline analysis without pass/fail.

    Uses production data from run_meta to trace the modular retrieval pipeline:
    - Vector retrieval count
    - Citation expansion / anchor hints
    - Hybrid rerank scores
    - Context selection

    This scorer always "passes" - it's for diagnostics, not evaluation.
    """

    @property
    def name(self) -> str:
        return "pipeline_breakdown"

    def score(
        self,
        *,
        expected: GoldenExpected,
        retrieval_metrics: dict[str, Any],
        references_structured: list[dict[str, Any]],
    ) -> Score:
        run_meta = retrieval_metrics.get("run") or {}
        modular_pipeline = run_meta.get("modular_pipeline") or {}

        # Extract breakdown from modular pipeline
        vec = modular_pipeline.get("vector_retrieval") or {}
        vec_enhanced = dict(vec)

        # Extract best distance from hybrid rerank top scores if available
        hybrid = modular_pipeline.get("hybrid_rerank") or {}
        top_scores = hybrid.get("top_10_scores") or hybrid.get("top_scores") or []
        if top_scores:
            vec_enhanced["best_distance"] = top_scores[0].get("vec", None)

        # Get unique anchors from context selection
        ctx_sel = modular_pipeline.get("context_selection") or {}
        unique_anchors = ctx_sel.get("unique_anchors") or []

        # Also check run_meta for anchors_in_top_k (set by rag.py)
        anchors_in_context = run_meta.get("anchors_in_top_k") or unique_anchors

        breakdown = {
            "stages": {
                # Stage 1: Vector Retrieval
                "1_vector_retrieval": vec_enhanced,

                # Stage 2: Citation Expansion
                "2_citation_expansion": {
                    "articles": modular_pipeline.get("citation_expansion_articles") or [],
                    "chunks_injected": modular_pipeline.get("chunks_injected", 0),
                    "hint_anchors": list((modular_pipeline.get("anchor_hints") or {}).get("hint_anchors") or []),
                },

                # Stage 3: Hybrid Rerank
                "3_hybrid_rerank": {
                    "enabled": hybrid.get("enabled", True),
                    "query_intent": hybrid.get("query_intent"),
                    "top_scores": top_scores,
                    "duration_ms": hybrid.get("duration_ms"),
                },

                # Stage 4: Context Selection
                "4_context_selection": ctx_sel,
            },

            # Timing
            "total_duration_ms": modular_pipeline.get("total_duration_ms"),

            # Final results (common fields for reporter)
            "final_context_count": len(references_structured),
            "anchors_in_context": anchors_in_context,
            "user_profile": retrieval_metrics.get("user_profile"),
            "llm_called": not run_meta.get("dry_run", False),
        }

        return Score(
            passed=True,  # Always passes - diagnostic only
            score=1.0,
            message="pipeline analysis complete",
            details=breakdown
        )


def _get_citation_verification_threshold() -> float:
    """Get citation verification threshold from config."""
    from ..common.config_loader import get_settings_yaml
    settings = get_settings_yaml()
    citation_settings = settings.get("citation_improvement", {})
    return float(citation_settings.get("similarity_threshold", 0.75))


class CitationVerificationScorer(Scorer):
    """
    Scores citation accuracy using post-hoc verification from production.

    Uses production data from run_meta.citation_verification which contains:
    - overall_score: Average similarity between citations and their chunks
    - verified_count: Citations that passed similarity threshold
    - suspicious_count: Citations with low similarity (potential mis-attribution)
    - suspicious: List of suspicious citation details

    This scorer reads verification results computed by production code,
    following the EVAL = PROD principle.

    Threshold is read from config/settings.yaml -> citation_improvement.similarity_threshold
    """

    def __init__(self, threshold: float | None = None):
        """Initialize with similarity threshold.

        Args:
            threshold: Minimum overall_score to pass (default: from config)
        """
        self.threshold = threshold if threshold is not None else _get_citation_verification_threshold()

    @property
    def name(self) -> str:
        return "citation_verification"

    def score(
        self,
        *,
        expected: GoldenExpected,
        retrieval_metrics: dict[str, Any],
        references_structured: list[dict[str, Any]],
    ) -> Score:
        run_meta = retrieval_metrics.get("run") or {}
        citation_verification = run_meta.get("citation_verification")

        # Check if verification was run
        if not citation_verification:
            return Score(
                passed=True,  # Pass if not run (e.g., abstain, no citations)
                score=1.0,
                message="citation verification not run",
                details={"reason": "no_verification_data"},
            )

        overall_score = citation_verification.get("overall_score", 1.0)
        verified_count = citation_verification.get("verified_count", 0)
        suspicious_count = citation_verification.get("suspicious_count", 0)
        suspicious = citation_verification.get("suspicious", [])

        passed = overall_score >= self.threshold

        if suspicious_count > 0:
            message = f"{verified_count} verified, {suspicious_count} suspicious (score: {overall_score:.2f})"
        else:
            message = f"all {verified_count} citations verified (score: {overall_score:.2f})"

        return Score(
            passed=passed,
            score=overall_score,
            message=message,
            details={
                "overall_score": overall_score,
                "verified_count": verified_count,
                "suspicious_count": suspicious_count,
                "suspicious": suspicious,
                "threshold": self.threshold,
            },
        )


# =============================================================================
# LLM-AS-JUDGE SCORERS
# =============================================================================

# Type alias for progress callbacks
ProgressCallback = Callable[[str, int, int], None]


def _get_llm_judge_settings() -> dict:
    """Get LLM-judge settings from config."""
    from ..common.config_loader import get_settings_yaml
    settings = get_settings_yaml()
    eval_settings = settings.get("eval", {})
    return eval_settings.get("llm_judge", {})


def _get_judge_model() -> str:
    """Get the model to use for LLM-as-judge from config."""
    llm_judge = _get_llm_judge_settings()
    return os.getenv("EVAL_JUDGE_MODEL", llm_judge.get("model", "gpt-4o"))


def _get_faithfulness_threshold() -> float:
    """Get faithfulness threshold from config."""
    llm_judge = _get_llm_judge_settings()
    return float(os.getenv("EVAL_FAITHFULNESS_THRESHOLD", llm_judge.get("faithfulness_threshold", 0.8)))


def _get_relevancy_threshold() -> float:
    """Get relevancy threshold from config."""
    llm_judge = _get_llm_judge_settings()
    return float(os.getenv("EVAL_RELEVANCY_THRESHOLD", llm_judge.get("relevancy_threshold", 0.7)))


def _get_claim_batch_size() -> int:
    """Get claim batch size from config."""
    llm_judge = _get_llm_judge_settings()
    return int(llm_judge.get("claim_batch_size", 3))


def _get_max_context_chars() -> int:
    """Get max context chars from config."""
    llm_judge = _get_llm_judge_settings()
    return int(llm_judge.get("max_context_chars", 8000))


def _get_cache_enabled() -> bool:
    """Get cache enabled setting from config."""
    llm_judge = _get_llm_judge_settings()
    return bool(llm_judge.get("cache_enabled", True))


def _make_llm_client():
    """Create OpenAI client for LLM-as-judge."""
    from ..engine.llm_client import make_openai_client
    return make_openai_client()


def _call_llm_json(prompt: str, model: Optional[str] = None, max_retries: int = 3, context: str = "") -> dict[str, Any]:
    """Call LLM and parse JSON response with retry for rate limits."""
    import time
    
    client = _make_llm_client()
    model = model or _get_judge_model()
    tracker = get_rate_limit_tracker()
    
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Deterministic for evaluation
                response_format={"type": "json_object"},
            )
            
            # Track token usage
            if response.usage:
                tracker.record_usage(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            content = response.choices[0].message.content or "{}"
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    return json.loads(match.group())
                return {"error": "Failed to parse JSON", "raw": content}
                
        except Exception as e:
            last_error = e
            error_str = str(e)
            # Check for rate limit error
            if "429" in error_str or "rate_limit" in error_str.lower():
                wait_time = min(2 ** attempt * 2, 30)  # Exponential backoff: 2s, 4s, 8s...
                tracker.record_hit(attempt + 1, wait_time, context)
                time.sleep(wait_time)
                continue
            raise  # Re-raise non-rate-limit errors
    
    # All retries exhausted
    raise last_error if last_error else RuntimeError("LLM call failed after retries")


class FaithfulnessScorer:
    """
    Scores if the answer is faithful to the retrieved context.
    
    Algorithm (Ragas-inspired):
    1. Extract claims from the answer
    2. Verify each claim against the context
    3. Score = supported_claims / total_claims
    
    All settings read from config/settings.yaml -> eval.llm_judge
    Environment variables can override: EVAL_JUDGE_MODEL, EVAL_FAITHFULNESS_THRESHOLD
    """
    
    name: str = "faithfulness"
    
    def __init__(
        self,
        threshold: float | None = None,
        use_cache: bool | None = None,
        claim_batch_size: int | None = None,
    ):
        # Read from config with optional overrides
        self.threshold = threshold if threshold is not None else _get_faithfulness_threshold()
        self.use_cache = use_cache if use_cache is not None else _get_cache_enabled()
        self.claim_batch_size = claim_batch_size if claim_batch_size is not None else _get_claim_batch_size()
        self.max_context_chars = _get_max_context_chars()
        self.cache = get_cache(enabled=self.use_cache)
    
    def score(
        self,
        *,
        question: str,
        answer: str,
        context: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Score:
        """
        Evaluate faithfulness of answer to context.
        
        Args:
            question: The user's question
            answer: The LLM's answer
            context: The retrieved context (full text)
            progress_callback: Optional callback for progress updates
        
        Returns:
            Score with faithfulness evaluation
        """
        # Check cache first
        cached = self.cache.get(
            self.name,
            question=question,
            answer=answer,
            context=context[:1000],  # Truncate for cache key
        )
        if cached:
            return Score(
                passed=cached["passed"],
                score=cached["score"],
                message=cached["message"],
                details=cached.get("details", {}),
            )
        
        # Step 1: Extract claims from answer
        if progress_callback:
            progress_callback("Extracting claims...", 0, 0)
        
        claims = self._extract_claims(question, answer)
        
        if not claims:
            result = Score(
                passed=True,
                score=1.0,
                message="No factual claims to verify",
                details={"claims": [], "verifications": []},
            )
            self._cache_result(result, question, answer, context)
            return result
        
        if progress_callback:
            progress_callback(f"Found {len(claims)} claims", len(claims), len(claims))
        
        # Step 2: Verify each claim against context
        verifications = self._verify_claims(claims, context, progress_callback)
        
        # Step 3: Calculate score
        supported_count = sum(1 for v in verifications if v["supported"])
        total_claims = len(claims)
        score_val = supported_count / total_claims if total_claims > 0 else 1.0
        passed = score_val >= self.threshold
        
        unsupported = [v["claim"] for v in verifications if not v["supported"]]
        if unsupported:
            message = f"{supported_count}/{total_claims} claims supported"
        else:
            message = "All claims supported by context"
        
        result = Score(
            passed=passed,
            score=score_val,
            message=message,
            details={
                "claims": claims,
                "verifications": verifications,
                "supported_count": supported_count,
                "total_claims": total_claims,
                "unsupported_claims": unsupported,
                "threshold": self.threshold,
            },
        )
        
        self._cache_result(result, question, answer, context)
        return result
    
    def _extract_claims(self, question: str, answer: str) -> list[str]:
        """Extract factual claims from the answer."""
        prompt = EXTRACT_CLAIMS_PROMPT.format(question=question, answer=answer)
        
        try:
            # Call LLM to extract claims
            client = _make_llm_client()
            model = _get_judge_model()
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            
            content = response.choices[0].message.content or "[]"
            
            # Parse JSON array
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                claims = json.loads(match.group())
                return [str(c) for c in claims if c]
            
            return []
        except Exception:
            # Fallback: split answer into sentences as claims
            sentences = re.split(r"[.!?]+", answer)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    
    def _verify_claims(
        self,
        claims: list[str],
        context: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[dict[str, Any]]:
        """Verify each claim against the context."""
        verifications = []
        
        # Process claims in batches
        for i in range(0, len(claims), self.claim_batch_size):
            batch = claims[i : i + self.claim_batch_size]
            batch_start = i + 1
            
            if progress_callback:
                progress_callback(
                    f"Verifying claims {batch_start}-{min(i + self.claim_batch_size, len(claims))}",
                    min(i + self.claim_batch_size, len(claims)),
                    len(claims),
                )
            
            # Format claims for prompt
            claims_text = "\n".join(f"{j+1}. {c}" for j, c in enumerate(batch))
            prompt = VERIFY_CLAIMS_PROMPT.format(context=context[:self.max_context_chars], claims=claims_text)
            
            try:
                result = _call_llm_json(prompt)
                batch_verifications = result.get("verifications", [])
                
                for j, claim in enumerate(batch):
                    if j < len(batch_verifications):
                        v = batch_verifications[j]
                        verifications.append({
                            "claim": claim,
                            "supported": v.get("verdict", "").upper() == "SUPPORTED",
                            "explanation": v.get("explanation", ""),
                        })
                    else:
                        verifications.append({
                            "claim": claim,
                            "supported": False,
                            "explanation": "Verification missing from response",
                        })
            except Exception as e:
                # Mark all claims in batch as unverified
                for claim in batch:
                    verifications.append({
                        "claim": claim,
                        "supported": False,
                        "explanation": f"Verification failed: {e}",
                    })
        
        return verifications
    
    def _cache_result(self, result: Score, question: str, answer: str, context: str) -> None:
        """Cache the result."""
        self.cache.set(
            self.name,
            {
                "passed": result.passed,
                "score": result.score,
                "message": result.message,
                "details": result.details,
            },
            question=question,
            answer=answer,
            context=context[:1000],
        )


class AnswerRelevancyScorer:
    """
    Scores if the answer is relevant to the question.
    
    Uses a single LLM call to evaluate relevancy on a 0-10 scale.
    
    All settings read from config/settings.yaml -> eval.llm_judge
    Environment variables can override: EVAL_JUDGE_MODEL, EVAL_RELEVANCY_THRESHOLD
    """
    
    name: str = "answer_relevancy"
    
    def __init__(
        self,
        threshold: float | None = None,
        use_cache: bool | None = None,
    ):
        # Read from config with optional overrides
        self.threshold = threshold if threshold is not None else _get_relevancy_threshold()
        self.use_cache = use_cache if use_cache is not None else _get_cache_enabled()
        self.cache = get_cache(enabled=self.use_cache)
    
    def score(
        self,
        *,
        question: str,
        answer: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Score:
        """
        Evaluate if the answer addresses the question.
        
        Args:
            question: The user's question
            answer: The LLM's answer
            progress_callback: Optional callback for progress updates
        
        Returns:
            Score with relevancy evaluation
        """
        # Check cache first
        cached = self.cache.get(
            self.name,
            question=question,
            answer=answer,
        )
        if cached:
            return Score(
                passed=cached["passed"],
                score=cached["score"],
                message=cached["message"],
                details=cached.get("details", {}),
            )
        
        if progress_callback:
            progress_callback("Evaluating answer relevance...", 0, 0)
        
        prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
        
        try:
            result = _call_llm_json(prompt)
            
            # Normalize score from 0-10 to 0-1
            raw_score = result.get("score", 5)
            score_val = max(0.0, min(1.0, raw_score / 10.0))
            critique = result.get("critique", "No critique provided")
            
            passed = score_val >= self.threshold
            message = f"Relevancy: {raw_score}/10" + (" ✓" if passed else " ✗")
            
            score_result = Score(
                passed=passed,
                score=score_val,
                message=message,
                details={
                    "raw_score": raw_score,
                    "critique": critique,
                    "threshold": self.threshold,
                },
            )
            
        except Exception as e:
            score_result = Score(
                passed=False,
                score=0.0,
                message=f"Relevancy evaluation failed: {e}",
                details={"error": str(e)},
            )
        
        # Cache result
        self.cache.set(
            self.name,
            {
                "passed": score_result.passed,
                "score": score_result.score,
                "message": score_result.message,
                "details": score_result.details,
            },
            question=question,
            answer=answer,
        )

        return score_result


class AbstentionScorer:
    """
    Scores if the system correctly abstained from answering an unanswerable question.

    Per UAEval4RAG and AbstentionBench best practices, this scorer evaluates:
    1. Whether the response contains abstention indicators (refusal to give definitive answer)
    2. Whether the abstention is appropriate (doesn't provide false certainty)

    This replaces faithfulness scoring for abstain cases, as standard faithfulness
    metrics (e.g., RAGAS) incorrectly score correct refusals as failures.

    References:
    - UAEval4RAG: https://arxiv.org/abs/2412.12300
    - AbstentionBench: https://github.com/facebookresearch/AbstentionBench
    - Know Your Limits Survey: https://arxiv.org/abs/2407.18418
    """

    name: str = "abstention"

    # Abstention indicators - phrases that suggest the system is refusing/clarifying
    # rather than providing a definitive answer
    ABSTENTION_INDICATORS_DA = [
        # Danish abstention phrases
        "kan ikke vurdere",
        "kan ikke afgøre",
        "kan ikke besvare",
        "ikke muligt at vurdere",
        "ikke muligt at afgøre",
        "ikke tilstrækkelig information",
        "utilstrækkelig information",
        "mangler oplysninger",
        "mangler information",
        "afhænger af",
        "kræver yderligere",
        "kræver konkret",
        "kræver specifik",
        "uden nærmere",
        "uden konkret",
        "uden specifik",
        "generelt set",
        "overordnet",
        "ikke entydigt",
        "kan ikke konkludere",
        "umuligt at sige",
    ]

    ABSTENTION_INDICATORS_EN = [
        # English abstention phrases
        "cannot determine",
        "cannot assess",
        "cannot evaluate",
        "cannot answer",
        "insufficient information",
        "depends on",
        "requires further",
        "requires specific",
        "without more",
        "without specific",
        "generally speaking",
        "in general",
        "not possible to",
        "unable to",
    ]

    # Definitive answer indicators - phrases that suggest a definitive yes/no answer
    # which should NOT appear in a correct abstention
    DEFINITIVE_INDICATORS_DA = [
        "ja, i er",
        "nej, i er ikke",
        "i er compliant",
        "i er ikke compliant",
        "i overholder",
        "i overholder ikke",
    ]

    DEFINITIVE_INDICATORS_EN = [
        "yes, you are",
        "no, you are not",
        "you are compliant",
        "you are not compliant",
        "you comply",
        "you do not comply",
    ]

    def score(
        self,
        *,
        question: str,
        answer: str,
    ) -> Score:
        """
        Evaluate if the system correctly abstained from answering.

        Args:
            question: The user's question (expected to be unanswerable/ambiguous)
            answer: The system's response

        Returns:
            Score indicating whether abstention was correct
        """
        answer_lower = answer.lower()

        # Check for abstention indicators
        all_abstention = self.ABSTENTION_INDICATORS_DA + self.ABSTENTION_INDICATORS_EN
        abstention_found = [ind for ind in all_abstention if ind in answer_lower]

        # Check for definitive answers (should NOT be present)
        all_definitive = self.DEFINITIVE_INDICATORS_DA + self.DEFINITIVE_INDICATORS_EN
        definitive_found = [ind for ind in all_definitive if ind in answer_lower]

        # Scoring logic:
        # - Pass if: abstention indicators present AND no definitive answers
        # - Fail if: definitive answer present OR no abstention indicators
        has_abstention = len(abstention_found) > 0
        has_definitive = len(definitive_found) > 0

        if has_definitive:
            # System gave a definitive answer when it should have abstained
            passed = False
            score_val = 0.0
            message = f"Incorrectly gave definitive answer: {definitive_found[0]}"
        elif has_abstention:
            # Correct: System abstained appropriately
            passed = True
            score_val = 1.0
            message = f"Correctly abstained ({len(abstention_found)} indicators found)"
        else:
            # Unclear: No clear abstention or definitive answer
            # This is a soft pass - the system didn't hallucinate a definitive answer
            passed = True
            score_val = 0.7
            message = "No definitive answer given (implicit abstention)"

        return Score(
            passed=passed,
            score=score_val,
            message=message,
            details={
                "abstention_indicators_found": abstention_found,
                "definitive_indicators_found": definitive_found,
                "has_abstention": has_abstention,
                "has_definitive": has_definitive,
            },
        )
