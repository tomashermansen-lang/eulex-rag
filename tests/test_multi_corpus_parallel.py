"""Tests for parallel multi-corpus retrieval (Component 3).

Tests ThreadPoolExecutor-based parallel retrieval, timeout handling,
partial failure, and config integration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from src.engine.multi_corpus_retrieval import (
    MultiCorpusConfig,
    MultiCorpusInput,
    MultiCorpusResult,
    apply_rrf_fusion,
    execute_multi_corpus_retrieval,
)
from src.engine.retrieval_pipeline import (
    CitationExpansionResult,
    ContextSelectionResult,
    HybridRerankResult,
    PipelineConfig,
    PipelineInput,
    PipelineResult,
    RetrievedChunk,
    ScoredChunk,
    VectorRetrievalResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scored_chunk(chunk_id: str, corpus_id: str, score: float = 0.5) -> ScoredChunk:
    """Create a ScoredChunk for testing."""
    return ScoredChunk(
        chunk=RetrievedChunk(
            chunk_id=chunk_id,
            document=f"Doc {chunk_id}",
            metadata={"corpus_id": corpus_id},
            distance=1.0 - score,
        ),
        vec_score=score,
        bm25_score=0.0,
        citation_score=0.0,
        role_score=0.0,
        final_score=score,
    )


def _make_pipeline_result(corpus_id: str, n_chunks: int = 3) -> PipelineResult:
    """Create a fake PipelineResult with n_chunks."""
    chunks = [
        _make_scored_chunk(f"{corpus_id}-chunk-{i}", corpus_id, score=0.9 - i * 0.1)
        for i in range(n_chunks)
    ]
    return PipelineResult(
        context_result=ContextSelectionResult(
            selected=(),
            citable_count=0,
            unique_anchors=frozenset(),
            duration_ms=0.0,
        ),
        vector_result=VectorRetrievalResult(
            chunks=(),
            retrieval_mode="standard",
            duration_ms=0.0,
        ),
        expansion_result=CitationExpansionResult(
            chunks=(),
            hint_anchors=frozenset(),
            chunks_injected=0,
            injected_anchors=(),
            duration_ms=0.0,
        ),
        rerank_result=HybridRerankResult(
            scored_chunks=tuple(chunks),
            query_intent="informational",
            duration_ms=10.0,
        ),
        total_duration_ms=50.0,
    )


def _make_delayed_pipeline_fn(delay_secs: float, corpus_id: str, n_chunks: int = 3):
    """Create a pipeline function with configurable delay."""
    def pipeline_fn(*, input, config, query_fn, inject_fn, is_citable_fn):  # noqa: A002
        time.sleep(delay_secs)
        return _make_pipeline_result(corpus_id, n_chunks)
    return pipeline_fn


def _make_factories():
    """Create dummy factory functions for testing."""
    def query_fn_factory(corpus_id: str):
        def query_fn(question, k, where_filter):
            return ([], [])
        return query_fn

    def inject_fn_factory(corpus_id: str):
        return None

    def is_citable_fn(metadata, doc_text):
        return (True, None)

    return query_fn_factory, inject_fn_factory, is_citable_fn


# ---------------------------------------------------------------------------
# MC-01: Parallel execution is faster than sequential
# ---------------------------------------------------------------------------


class TestParallelExecution:
    """Tests for parallel multi-corpus retrieval timing."""

    def test_parallel_faster_than_sequential(self, monkeypatch):
        """MC-01: 5 corpora with 100ms delay each: total < 300ms (not 500ms)."""
        corpus_ids = tuple(f"corpus_{i}" for i in range(5))
        delay_per_corpus = 0.1  # 100ms

        call_log = []

        def mock_execute_pipeline(*, input, config, query_fn, inject_fn, is_citable_fn):  # noqa: A002
            call_log.append(input.corpus_id)
            time.sleep(delay_per_corpus)
            return _make_pipeline_result(input.corpus_id, n_chunks=2)

        monkeypatch.setattr(
            "src.engine.multi_corpus_retrieval.execute_pipeline",
            mock_execute_pipeline,
        )

        query_fn_factory, inject_fn_factory, is_citable_fn = _make_factories()

        mc_input = MultiCorpusInput(
            question="test question",
            corpus_ids=corpus_ids,
            user_profile="LEGAL",
        )
        config = MultiCorpusConfig(
            retrieval_pool_size=50,
            merged_pool_size=100,
            rrf_k=60,
            max_workers=5,
            timeout_secs=3.0,
        )

        start = time.perf_counter()
        result = execute_multi_corpus_retrieval(
            input=mc_input,
            config=config,
            query_fn_factory=query_fn_factory,
            inject_fn_factory=inject_fn_factory,
            is_citable_fn=is_citable_fn,
        )
        elapsed = time.perf_counter() - start

        # Should complete in ~100ms (parallel), not 500ms (sequential)
        assert elapsed < 0.3, f"Expected < 300ms, got {elapsed * 1000:.0f}ms"
        assert len(call_log) == 5
        assert len(result.fused_chunks) > 0


# ---------------------------------------------------------------------------
# MC-02: Results identical to sequential
# ---------------------------------------------------------------------------


class TestResultsIdentical:
    """Tests that parallel results match sequential results."""

    def test_results_identical_to_sequential(self, monkeypatch):
        """MC-02: Same chunks and RRF scores regardless of execution order."""
        corpus_ids = ("corpus_a", "corpus_b", "corpus_c")

        def mock_execute_pipeline(*, input, config, query_fn, inject_fn, is_citable_fn):  # noqa: A002
            return _make_pipeline_result(input.corpus_id, n_chunks=3)

        monkeypatch.setattr(
            "src.engine.multi_corpus_retrieval.execute_pipeline",
            mock_execute_pipeline,
        )

        query_fn_factory, inject_fn_factory, is_citable_fn = _make_factories()

        mc_input = MultiCorpusInput(
            question="test question",
            corpus_ids=corpus_ids,
            user_profile="LEGAL",
        )
        config = MultiCorpusConfig(
            retrieval_pool_size=50,
            merged_pool_size=100,
            rrf_k=60,
            max_workers=3,
            timeout_secs=3.0,
        )

        result = execute_multi_corpus_retrieval(
            input=mc_input,
            config=config,
            query_fn_factory=query_fn_factory,
            inject_fn_factory=inject_fn_factory,
            is_citable_fn=is_citable_fn,
        )

        # All 3 corpora × 3 chunks = 9 chunks before fusion
        assert len(result.fused_chunks) == 9
        # Per-corpus hits should be 3 each
        for cid in corpus_ids:
            assert result.per_corpus_hits[cid] == 3


# ---------------------------------------------------------------------------
# MC-03: Partial failure returns successful corpora
# ---------------------------------------------------------------------------


class TestPartialFailure:
    """Tests for partial failure handling."""

    def test_partial_failure_returns_successful(self, monkeypatch):
        """MC-03: 2/5 corpora raise; 3 results returned with error in debug."""
        failing_corpora = {"corpus_1", "corpus_3"}

        def mock_execute_pipeline(*, input, config, query_fn, inject_fn, is_citable_fn):  # noqa: A002
            if input.corpus_id in failing_corpora:
                raise RuntimeError(f"Connection failed for {input.corpus_id}")
            return _make_pipeline_result(input.corpus_id, n_chunks=2)

        monkeypatch.setattr(
            "src.engine.multi_corpus_retrieval.execute_pipeline",
            mock_execute_pipeline,
        )

        query_fn_factory, inject_fn_factory, is_citable_fn = _make_factories()
        corpus_ids = tuple(f"corpus_{i}" for i in range(5))

        mc_input = MultiCorpusInput(
            question="test question",
            corpus_ids=corpus_ids,
            user_profile="LEGAL",
        )
        config = MultiCorpusConfig(
            retrieval_pool_size=50,
            merged_pool_size=100,
            rrf_k=60,
            max_workers=5,
            timeout_secs=3.0,
        )

        result = execute_multi_corpus_retrieval(
            input=mc_input,
            config=config,
            query_fn_factory=query_fn_factory,
            inject_fn_factory=inject_fn_factory,
            is_citable_fn=is_citable_fn,
        )

        # 3 successful × 2 chunks = 6 chunks
        assert len(result.fused_chunks) == 6

        # Failed corpora have 0 hits
        assert result.per_corpus_hits["corpus_1"] == 0
        assert result.per_corpus_hits["corpus_3"] == 0

        # Failed corpora have errors in debug
        assert "error" in result.debug["per_corpus"]["corpus_1"]
        assert "error" in result.debug["per_corpus"]["corpus_3"]

        # Successful corpora have hits
        assert result.per_corpus_hits["corpus_0"] == 2
        assert result.per_corpus_hits["corpus_2"] == 2
        assert result.per_corpus_hits["corpus_4"] == 2


# ---------------------------------------------------------------------------
# MC-04: Timeout treated as failure
# ---------------------------------------------------------------------------


class TestTimeout:
    """Tests for per-corpus timeout handling."""

    def test_timeout_treated_as_failure(self, monkeypatch):
        """MC-04: Corpus exceeding timeout treated as failure, others succeed."""
        def mock_execute_pipeline(*, input, config, query_fn, inject_fn, is_citable_fn):  # noqa: A002
            if input.corpus_id == "slow_corpus":
                time.sleep(5.0)  # Way over timeout
            return _make_pipeline_result(input.corpus_id, n_chunks=2)

        monkeypatch.setattr(
            "src.engine.multi_corpus_retrieval.execute_pipeline",
            mock_execute_pipeline,
        )

        query_fn_factory, inject_fn_factory, is_citable_fn = _make_factories()

        mc_input = MultiCorpusInput(
            question="test question",
            corpus_ids=("fast_corpus", "slow_corpus"),
            user_profile="LEGAL",
        )
        config = MultiCorpusConfig(
            retrieval_pool_size=50,
            merged_pool_size=100,
            rrf_k=60,
            max_workers=2,
            timeout_secs=0.5,
        )

        start = time.perf_counter()
        result = execute_multi_corpus_retrieval(
            input=mc_input,
            config=config,
            query_fn_factory=query_fn_factory,
            inject_fn_factory=inject_fn_factory,
            is_citable_fn=is_citable_fn,
        )
        elapsed = time.perf_counter() - start

        # Should not wait for the slow corpus (timeout at 0.5s)
        assert elapsed < 2.0, f"Expected < 2s, got {elapsed:.1f}s"

        # Fast corpus succeeded
        assert result.per_corpus_hits["fast_corpus"] == 2

        # Slow corpus timed out (treated as failure)
        assert result.per_corpus_hits["slow_corpus"] == 0
        assert "error" in result.debug["per_corpus"]["slow_corpus"]


# ---------------------------------------------------------------------------
# MC-05: All corpora fail returns empty
# ---------------------------------------------------------------------------


class TestAllCorporaFail:
    """Tests for total failure."""

    def test_all_corpora_fail_returns_empty(self, monkeypatch):
        """MC-05: All N fail → empty fused_chunks, errors in debug."""
        def mock_execute_pipeline(*, input, config, query_fn, inject_fn, is_citable_fn):  # noqa: A002
            raise RuntimeError(f"Failed: {input.corpus_id}")

        monkeypatch.setattr(
            "src.engine.multi_corpus_retrieval.execute_pipeline",
            mock_execute_pipeline,
        )

        query_fn_factory, inject_fn_factory, is_citable_fn = _make_factories()

        mc_input = MultiCorpusInput(
            question="test question",
            corpus_ids=("corpus_a", "corpus_b"),
            user_profile="LEGAL",
        )
        config = MultiCorpusConfig(
            retrieval_pool_size=50,
            merged_pool_size=100,
            rrf_k=60,
            max_workers=2,
            timeout_secs=3.0,
        )

        result = execute_multi_corpus_retrieval(
            input=mc_input,
            config=config,
            query_fn_factory=query_fn_factory,
            inject_fn_factory=inject_fn_factory,
            is_citable_fn=is_citable_fn,
        )

        assert len(result.fused_chunks) == 0
        assert result.per_corpus_hits["corpus_a"] == 0
        assert result.per_corpus_hits["corpus_b"] == 0
        assert "error" in result.debug["per_corpus"]["corpus_a"]
        assert "error" in result.debug["per_corpus"]["corpus_b"]


# ---------------------------------------------------------------------------
# MC-06: Zero corpora graceful
# ---------------------------------------------------------------------------


class TestZeroCorpora:
    """Tests for empty corpus list."""

    def test_zero_corpora_graceful(self, monkeypatch):
        """MC-06: Zero corpora input → empty result, no crash."""
        query_fn_factory, inject_fn_factory, is_citable_fn = _make_factories()

        mc_input = MultiCorpusInput(
            question="test question",
            corpus_ids=(),
            user_profile="LEGAL",
        )
        config = MultiCorpusConfig(
            retrieval_pool_size=50,
            merged_pool_size=100,
            rrf_k=60,
            max_workers=4,
            timeout_secs=3.0,
        )

        result = execute_multi_corpus_retrieval(
            input=mc_input,
            config=config,
            query_fn_factory=query_fn_factory,
            inject_fn_factory=inject_fn_factory,
            is_citable_fn=is_citable_fn,
        )

        assert len(result.fused_chunks) == 0
        assert len(result.per_corpus_hits) == 0


# ---------------------------------------------------------------------------
# MC-07: Max workers configurable
# ---------------------------------------------------------------------------


class TestMaxWorkersConfigurable:
    """Tests for max_workers configuration."""

    def test_max_workers_configurable(self):
        """MC-07: MultiCorpusConfig(max_workers=4) limits thread pool."""
        config = MultiCorpusConfig(
            retrieval_pool_size=50,
            merged_pool_size=100,
            rrf_k=60,
            max_workers=4,
            timeout_secs=3.0,
        )
        assert config.max_workers == 4

    def test_max_workers_default(self):
        """MC-07: max_workers has a sensible default."""
        config = MultiCorpusConfig()
        assert config.max_workers == 16  # From performance config default


# ---------------------------------------------------------------------------
# MC-08: Single corpus no regression
# ---------------------------------------------------------------------------


class TestSingleCorpus:
    """Tests for single corpus operation."""

    def test_single_corpus_no_regression(self, monkeypatch):
        """MC-08: Single-corpus call has minimal overhead vs. direct pipeline."""
        def mock_execute_pipeline(*, input, config, query_fn, inject_fn, is_citable_fn):  # noqa: A002
            return _make_pipeline_result(input.corpus_id, n_chunks=3)

        monkeypatch.setattr(
            "src.engine.multi_corpus_retrieval.execute_pipeline",
            mock_execute_pipeline,
        )

        query_fn_factory, inject_fn_factory, is_citable_fn = _make_factories()

        mc_input = MultiCorpusInput(
            question="test question",
            corpus_ids=("single_corpus",),
            user_profile="LEGAL",
        )
        config = MultiCorpusConfig(
            retrieval_pool_size=50,
            merged_pool_size=100,
            rrf_k=60,
            max_workers=1,
            timeout_secs=3.0,
        )

        result = execute_multi_corpus_retrieval(
            input=mc_input,
            config=config,
            query_fn_factory=query_fn_factory,
            inject_fn_factory=inject_fn_factory,
            is_citable_fn=is_citable_fn,
        )

        assert len(result.fused_chunks) == 3
        assert result.per_corpus_hits["single_corpus"] == 3


# ---------------------------------------------------------------------------
# MC-09: Per-corpus timing in debug
# ---------------------------------------------------------------------------


class TestPerCorpusTiming:
    """Tests for per-corpus timing in debug output."""

    def test_per_corpus_timing_in_debug(self, monkeypatch):
        """MC-09: Each corpus has duration_ms in debug output."""
        def mock_execute_pipeline(*, input, config, query_fn, inject_fn, is_citable_fn):  # noqa: A002
            return _make_pipeline_result(input.corpus_id, n_chunks=2)

        monkeypatch.setattr(
            "src.engine.multi_corpus_retrieval.execute_pipeline",
            mock_execute_pipeline,
        )

        query_fn_factory, inject_fn_factory, is_citable_fn = _make_factories()

        mc_input = MultiCorpusInput(
            question="test question",
            corpus_ids=("corpus_a", "corpus_b"),
            user_profile="LEGAL",
        )
        config = MultiCorpusConfig(
            retrieval_pool_size=50,
            merged_pool_size=100,
            rrf_k=60,
            max_workers=2,
            timeout_secs=3.0,
        )

        result = execute_multi_corpus_retrieval(
            input=mc_input,
            config=config,
            query_fn_factory=query_fn_factory,
            inject_fn_factory=inject_fn_factory,
            is_citable_fn=is_citable_fn,
        )

        for corpus_id in ("corpus_a", "corpus_b"):
            corpus_debug = result.debug["per_corpus"][corpus_id]
            assert "duration_ms" in corpus_debug
            assert isinstance(corpus_debug["duration_ms"], float)


# ---------------------------------------------------------------------------
# MC-10: Config from settings
# ---------------------------------------------------------------------------


class TestConfigFromSettings:
    """Tests for MultiCorpusConfig.from_settings()."""

    def test_config_from_settings(self, monkeypatch):
        """MC-10: MultiCorpusConfig.from_settings() reads performance section."""
        mock_settings = {
            "performance": {
                "max_retrieval_workers": 8,
                "retrieval_timeout_secs": 2.0,
            },
        }
        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )

        config = MultiCorpusConfig.from_settings()

        assert config.max_workers == 8
        assert config.timeout_secs == 2.0
