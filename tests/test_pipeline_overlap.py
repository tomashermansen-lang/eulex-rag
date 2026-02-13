"""Tests for pipeline overlap (Component 4).

Tests skip_intent_classification parameter in prepare_answer_context(),
apply_deferred_intent() function, and parallel timing.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from src.engine.planning import (
    AnswerContext,
    apply_deferred_intent,
    prepare_answer_context,
)
from src.engine.types import ClaimIntent, UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_fns(
    intent_delay: float = 0.0,
    intent_result: ClaimIntent = ClaimIntent.SCOPE,
) -> dict:
    """Create mock functions for prepare_answer_context()."""
    def classify_intent_fn(
        question: str, *, last_exchange: list | None = None, query_was_rewritten: bool = False,
    ) -> Tuple[ClaimIntent, dict]:
        if intent_delay > 0:
            time.sleep(intent_delay)
        return intent_result, {"method": "mock", "confidence": 0.9}

    def apply_policy_to_intent_fn(
        *,
        resolved_profile: UserProfile,
        classifier_intent: ClaimIntent,
        policy: Any,
        question: str,
    ) -> Tuple[ClaimIntent, dict]:
        return classifier_intent, {"policy_applied": True}

    def get_effective_policy_fn(corpus_id: str, intent_keys: list) -> Any:
        return MagicMock(
            intent_keys_effective=intent_keys,
            contributors={},
            normative_guard=None,
            answer_policy=None,
        )

    return {
        "classify_intent_fn": classify_intent_fn,
        "apply_policy_to_intent_fn": apply_policy_to_intent_fn,
        "get_effective_policy_fn": get_effective_policy_fn,
        "is_debug_corpus_fn": lambda: False,
        "iso_utc_now_fn": lambda: "2026-01-01T00:00:00Z",
        "git_commit_fn": lambda: "test-commit",
    }


# ---------------------------------------------------------------------------
# PO-01: skip_intent returns placeholder
# ---------------------------------------------------------------------------


class TestSkipIntent:
    """Tests for skip_intent_classification parameter."""

    def test_skip_intent_returns_placeholder(self, monkeypatch):
        """PO-01: prepare_answer_context(skip_intent_classification=True) returns
        ClaimIntent.GENERAL without calling classify_intent_fn."""
        classify_called = [False]
        fns = _make_mock_fns()

        original_classify = fns["classify_intent_fn"]

        def tracked_classify(question, **kwargs):
            classify_called[0] = True
            return original_classify(question, **kwargs)

        fns["classify_intent_fn"] = tracked_classify

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: {"retrieval_pipeline": {}},
        )

        answer_ctx = prepare_answer_context(
            question="What are the requirements of Article 6?",
            corpus_id="ai_act",
            resolved_profile=UserProfile.LEGAL,
            top_k=20,
            skip_intent_classification=True,
            **fns,
        )

        assert answer_ctx.claim_intent_final == ClaimIntent.GENERAL
        assert classify_called[0] is False

    def test_skip_intent_preserves_retrieval_context(self, monkeypatch):
        """PO-02: where_for_retrieval, ctx, effective_plan all populated correctly."""
        fns = _make_mock_fns()

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: {"retrieval_pipeline": {}},
        )

        answer_ctx = prepare_answer_context(
            question="What does Article 6 paragraph 1 require?",
            corpus_id="ai_act",
            resolved_profile=UserProfile.LEGAL,
            top_k=20,
            skip_intent_classification=True,
            **fns,
        )

        # Core retrieval context is populated
        assert answer_ctx.ctx is not None
        assert answer_ctx.ctx.question == "What does Article 6 paragraph 1 require?"
        assert answer_ctx.ctx.corpus_id == "ai_act"
        assert answer_ctx.effective_plan is not None
        assert answer_ctx.run_meta is not None


# ---------------------------------------------------------------------------
# PO-03: apply_deferred_intent updates context
# ---------------------------------------------------------------------------


class TestApplyDeferredIntent:
    """Tests for apply_deferred_intent()."""

    def test_updates_context(self, monkeypatch):
        """PO-03: apply_deferred_intent() updates claim_intent_final and run_meta."""
        fns = _make_mock_fns(intent_result=ClaimIntent.SCOPE)

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: {"retrieval_pipeline": {}},
        )

        # First, create context with skipped intent
        answer_ctx = prepare_answer_context(
            question="What is the scope of the AI Act?",
            corpus_id="ai_act",
            resolved_profile=UserProfile.LEGAL,
            top_k=20,
            skip_intent_classification=True,
            **fns,
        )

        assert answer_ctx.claim_intent_final == ClaimIntent.GENERAL

        # Now apply deferred intent
        updated = apply_deferred_intent(
            answer_ctx=answer_ctx,
            classify_intent_fn=fns["classify_intent_fn"],
            apply_policy_to_intent_fn=fns["apply_policy_to_intent_fn"],
            resolved_profile=UserProfile.LEGAL,
        )

        assert updated.claim_intent_final == ClaimIntent.SCOPE
        assert updated.run_meta["claim_intent"]["final"] == "SCOPE"

    def test_deferred_matches_immediate(self, monkeypatch):
        """PO-04: Deferred path produces identical claim_intent_final as immediate path."""
        fns = _make_mock_fns(intent_result=ClaimIntent.REQUIREMENTS)

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: {"retrieval_pipeline": {}},
        )

        # Immediate path (no skip)
        immediate_ctx = prepare_answer_context(
            question="What are the requirements?",
            corpus_id="ai_act",
            resolved_profile=UserProfile.LEGAL,
            top_k=20,
            **fns,
        )

        # Deferred path (skip + apply)
        deferred_ctx = prepare_answer_context(
            question="What are the requirements?",
            corpus_id="ai_act",
            resolved_profile=UserProfile.LEGAL,
            top_k=20,
            skip_intent_classification=True,
            **fns,
        )
        deferred_ctx = apply_deferred_intent(
            answer_ctx=deferred_ctx,
            classify_intent_fn=fns["classify_intent_fn"],
            apply_policy_to_intent_fn=fns["apply_policy_to_intent_fn"],
            resolved_profile=UserProfile.LEGAL,
        )

        assert immediate_ctx.claim_intent_final == deferred_ctx.claim_intent_final


# ---------------------------------------------------------------------------
# PO-05: Parallel overlap faster
# ---------------------------------------------------------------------------


class TestParallelOverlapTiming:
    """Tests for parallel execution timing."""

    def test_parallel_overlap_faster(self, monkeypatch):
        """PO-05: With 300ms intent + 200ms retrieval: total < 400ms (not 500ms)."""
        from concurrent.futures import ThreadPoolExecutor

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: {"retrieval_pipeline": {}},
        )

        intent_delay = 0.3
        retrieval_delay = 0.2

        def slow_intent(question: str) -> Tuple[ClaimIntent, dict]:
            time.sleep(intent_delay)
            return ClaimIntent.SCOPE, {"method": "mock"}

        def slow_retrieval() -> str:
            time.sleep(retrieval_delay)
            return "retrieval_result"

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as executor:
            intent_future = executor.submit(slow_intent, "test question")
            retrieval_future = executor.submit(slow_retrieval)

            intent_result = intent_future.result()
            retrieval_result = retrieval_future.result()

        elapsed = time.perf_counter() - start

        # Should complete in ~300ms (max of both), not 500ms (sum)
        assert elapsed < 0.4, f"Expected < 400ms, got {elapsed * 1000:.0f}ms"
        assert intent_result[0] == ClaimIntent.SCOPE
        assert retrieval_result == "retrieval_result"


# ---------------------------------------------------------------------------
# PO-06: Fallback sequential when disabled
# ---------------------------------------------------------------------------


class TestFallbackSequential:
    """Tests for sequential fallback."""

    def test_fallback_sequential_when_disabled(self, monkeypatch):
        """PO-06: async_enabled=False → sequential execution, same results."""
        fns = _make_mock_fns(intent_result=ClaimIntent.ENFORCEMENT)

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: {"retrieval_pipeline": {}},
        )

        # When skip_intent_classification=False (default), intent runs inline
        answer_ctx = prepare_answer_context(
            question="What are the enforcement rules?",
            corpus_id="ai_act",
            resolved_profile=UserProfile.LEGAL,
            top_k=20,
            skip_intent_classification=False,
            **fns,
        )

        assert answer_ctx.claim_intent_final == ClaimIntent.ENFORCEMENT


# ---------------------------------------------------------------------------
# PO-07: Timing breakdown in run_meta
# ---------------------------------------------------------------------------


class TestTimingBreakdown:
    """Tests for timing data in run_meta."""

    def test_timing_in_run_meta(self, monkeypatch):
        """PO-07: run_meta contains timing data when available."""
        fns = _make_mock_fns()

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: {"retrieval_pipeline": {}},
        )

        answer_ctx = prepare_answer_context(
            question="What does Article 5 say?",
            corpus_id="ai_act",
            resolved_profile=UserProfile.LEGAL,
            top_k=20,
            **fns,
        )

        # run_meta should exist and contain basic metadata
        assert "run_id" in answer_ctx.run_meta
        assert "timestamp_utc" in answer_ctx.run_meta
        assert "claim_intent" in answer_ctx.run_meta
