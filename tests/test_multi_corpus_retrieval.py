"""Tests for multi-corpus retrieval with RRF fusion.

TDD: These tests are written BEFORE the implementation.
"""

import pytest
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import MagicMock


# Test imports - will fail until implementation exists
@pytest.fixture
def import_module():
    """Import the module under test."""
    from src.engine.multi_corpus_retrieval import (
        MultiCorpusInput,
        MultiCorpusConfig,
        MultiCorpusResult,
        apply_rrf_fusion,
        execute_multi_corpus_retrieval,
    )
    return {
        "MultiCorpusInput": MultiCorpusInput,
        "MultiCorpusConfig": MultiCorpusConfig,
        "MultiCorpusResult": MultiCorpusResult,
        "apply_rrf_fusion": apply_rrf_fusion,
        "execute_multi_corpus_retrieval": execute_multi_corpus_retrieval,
    }


# ---------------------------------------------------------------------------
# RRF Fusion Tests (MCR-001 to MCR-010)
# ---------------------------------------------------------------------------


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion algorithm."""

    def test_mcr_001_rrf_calculates_scores_correctly(self, import_module):
        """RRF formula: score = 1 / (k + rank) per corpus.

        In cross-law synthesis, each (corpus_id, chunk_id) is unique because
        the same chunk_id in different corpora represents different legal content.
        """
        apply_rrf_fusion = import_module["apply_rrf_fusion"]

        # Three corpora, each with ranked results
        # Each chunk is unique per-corpus (different laws have different content)
        per_corpus_results = {
            "corpus_1": [
                _make_scored_chunk("chunk_c1_a", 0.9, "corpus_1"),  # rank 1
                _make_scored_chunk("chunk_c1_b", 0.8, "corpus_1"),  # rank 2
            ],
            "corpus_2": [
                _make_scored_chunk("chunk_c2_a", 0.95, "corpus_2"),  # rank 1
            ],
        }

        result = apply_rrf_fusion(per_corpus_results, k=60, max_results=10)

        # All chunks from all corpora should be present
        # Rank 1 chunks get RRF score = 1/(60+1) = 0.01639
        # Rank 2 chunks get RRF score = 1/(60+2) = 0.01613
        assert len(result) == 3

        # Rank 1 chunks should have higher scores than rank 2
        rank1_scores = [r.final_score for r in result if "c1_a" in r.chunk.chunk_id or "c2_a" in r.chunk.chunk_id]
        rank2_scores = [r.final_score for r in result if "c1_b" in r.chunk.chunk_id]
        assert min(rank1_scores) > max(rank2_scores)

        # Verify RRF formula: 1/(k+rank), rank is 1-indexed
        expected_rank1_score = 1.0 / (60 + 1)  # ~0.01639
        assert abs(result[0].final_score - expected_rank1_score) < 0.0001

    def test_mcr_002_rrf_handles_missing_documents(self, import_module):
        """Chunks only in some corpora get score from those corpora only."""
        apply_rrf_fusion = import_module["apply_rrf_fusion"]

        per_corpus_results = {
            "corpus_1": [
                _make_scored_chunk("chunk_a", 0.9, "corpus_1"),
                _make_scored_chunk("chunk_b", 0.8, "corpus_1"),
            ],
            "corpus_2": [
                _make_scored_chunk("chunk_c", 0.95, "corpus_2"),  # Only in corpus_2
            ],
        }

        result = apply_rrf_fusion(per_corpus_results, k=60, max_results=10)

        # chunk_c only in corpus_2, rank 1: RRF = 1/(60+1) = 0.0164
        # chunk_a only in corpus_1, rank 1: RRF = 1/(60+1) = 0.0164
        # chunk_b only in corpus_1, rank 2: RRF = 1/(60+2) = 0.0161
        assert len(result) == 3
        # Order may vary for tied scores, but chunk_b should be last
        chunk_ids = [r.chunk.chunk_id for r in result]
        assert "chunk_a" in chunk_ids
        assert "chunk_b" in chunk_ids
        assert "chunk_c" in chunk_ids

    def test_mcr_003_rrf_respects_max_results(self, import_module):
        """merged_pool_size limit is respected."""
        apply_rrf_fusion = import_module["apply_rrf_fusion"]

        # Create 10 chunks per corpus
        per_corpus_results = {
            "corpus_1": [
                _make_scored_chunk(f"chunk_{i}", 1.0 - i * 0.1, "corpus_1")
                for i in range(10)
            ],
            "corpus_2": [
                _make_scored_chunk(f"chunk_{i+10}", 1.0 - i * 0.1, "corpus_2")
                for i in range(10)
            ],
        }

        result = apply_rrf_fusion(per_corpus_results, k=60, max_results=5)

        assert len(result) == 5

    def test_mcr_006_empty_corpus_handled(self, import_module):
        """One empty corpus doesn't break fusion."""
        apply_rrf_fusion = import_module["apply_rrf_fusion"]

        per_corpus_results = {
            "corpus_1": [
                _make_scored_chunk("chunk_a", 0.9, "corpus_1"),
            ],
            "corpus_2": [],  # Empty
            "corpus_3": [
                _make_scored_chunk("chunk_b", 0.8, "corpus_3"),
            ],
        }

        result = apply_rrf_fusion(per_corpus_results, k=60, max_results=10)

        assert len(result) == 2
        chunk_ids = [r.chunk.chunk_id for r in result]
        assert "chunk_a" in chunk_ids
        assert "chunk_b" in chunk_ids

    def test_mcr_007_all_corpora_empty(self, import_module):
        """All empty corpora returns empty result."""
        apply_rrf_fusion = import_module["apply_rrf_fusion"]

        per_corpus_results = {
            "corpus_1": [],
            "corpus_2": [],
        }

        result = apply_rrf_fusion(per_corpus_results, k=60, max_results=10)

        assert len(result) == 0

    def test_mcr_008_k_parameter_affects_distribution(self, import_module):
        """Different k values produce different score distributions."""
        apply_rrf_fusion = import_module["apply_rrf_fusion"]

        per_corpus_results = {
            "corpus_1": [
                _make_scored_chunk("chunk_a", 0.9, "corpus_1"),
                _make_scored_chunk("chunk_b", 0.8, "corpus_1"),
            ],
        }

        result_k30 = apply_rrf_fusion(per_corpus_results, k=30, max_results=10)
        result_k60 = apply_rrf_fusion(per_corpus_results, k=60, max_results=10)

        # Lower k means higher scores overall
        assert result_k30[0].final_score > result_k60[0].final_score

    def test_mcr_009_duplicate_chunk_ids_different_corpora(self, import_module):
        """Same chunk_id from different corpora both included (different corpus_id)."""
        apply_rrf_fusion = import_module["apply_rrf_fusion"]

        # Same chunk_id but different corpus_id
        per_corpus_results = {
            "corpus_1": [
                _make_scored_chunk("chunk_a", 0.9, "corpus_1"),
            ],
            "corpus_2": [
                _make_scored_chunk("chunk_a", 0.8, "corpus_2"),  # Same ID, different corpus
            ],
        }

        result = apply_rrf_fusion(per_corpus_results, k=60, max_results=10)

        # Both should be included since they have different corpus_id
        assert len(result) == 2
        corpus_ids = [r.chunk.metadata.get("corpus_id") for r in result]
        assert "corpus_1" in corpus_ids
        assert "corpus_2" in corpus_ids


class TestMultiCorpusRetrieval:
    """Tests for the multi-corpus retrieval coordinator."""

    def test_mcr_004_parallel_retrieval_calls_all_corpora(self, import_module):
        """Parallel retrieval calls query_fn for each specified corpus."""
        MultiCorpusInput = import_module["MultiCorpusInput"]
        MultiCorpusConfig = import_module["MultiCorpusConfig"]
        execute_multi_corpus_retrieval = import_module["execute_multi_corpus_retrieval"]

        input_data = MultiCorpusInput(
            question="Test question",
            corpus_ids=("corpus_a", "corpus_b", "corpus_c"),
            user_profile="LEGAL",
            where_filter=None,
        )
        config = MultiCorpusConfig(retrieval_pool_size=50, merged_pool_size=100, rrf_k=60)

        # Track which corpora were queried
        queried_corpora = []

        def mock_query_fn_factory(corpus_id: str):
            def query_fn(question, k, where):
                queried_corpora.append(corpus_id)
                return ([], [])  # Empty results
            return query_fn

        def mock_inject_fn_factory(corpus_id: str):
            return None

        def mock_is_citable(meta, doc):
            return (True, None)

        result = execute_multi_corpus_retrieval(
            input=input_data,
            config=config,
            query_fn_factory=mock_query_fn_factory,
            inject_fn_factory=mock_inject_fn_factory,
            is_citable_fn=mock_is_citable,
        )

        assert set(queried_corpora) == {"corpus_a", "corpus_b", "corpus_c"}

    def test_mcr_005_chunks_preserve_corpus_id(self, import_module):
        """Retrieved chunks have corpus_id metadata set."""
        MultiCorpusInput = import_module["MultiCorpusInput"]
        MultiCorpusConfig = import_module["MultiCorpusConfig"]
        execute_multi_corpus_retrieval = import_module["execute_multi_corpus_retrieval"]

        input_data = MultiCorpusInput(
            question="Test question",
            corpus_ids=("ai_act", "gdpr"),
            user_profile="LEGAL",
            where_filter=None,
        )
        config = MultiCorpusConfig(retrieval_pool_size=50, merged_pool_size=100, rrf_k=60)

        def mock_query_fn_factory(corpus_id: str):
            def query_fn(question, k, where):
                return (
                    [(f"chunk_{corpus_id}", f"Doc from {corpus_id}", {"article": "1"})],
                    [0.1],
                )
            return query_fn

        def mock_inject_fn_factory(corpus_id: str):
            return None

        def mock_is_citable(meta, doc):
            return (True, None)

        result = execute_multi_corpus_retrieval(
            input=input_data,
            config=config,
            query_fn_factory=mock_query_fn_factory,
            inject_fn_factory=mock_inject_fn_factory,
            is_citable_fn=mock_is_citable,
        )

        # Check that chunks have corpus_id in metadata
        for chunk in result.fused_chunks:
            assert "corpus_id" in chunk.chunk.metadata
            assert chunk.chunk.metadata["corpus_id"] in ("ai_act", "gdpr")

    def test_mcr_010_duration_recorded(self, import_module):
        """duration_ms is recorded in result."""
        MultiCorpusInput = import_module["MultiCorpusInput"]
        MultiCorpusConfig = import_module["MultiCorpusConfig"]
        execute_multi_corpus_retrieval = import_module["execute_multi_corpus_retrieval"]

        input_data = MultiCorpusInput(
            question="Test question",
            corpus_ids=("corpus_a",),
            user_profile="LEGAL",
            where_filter=None,
        )
        config = MultiCorpusConfig()

        def mock_query_fn_factory(corpus_id: str):
            def query_fn(question, k, where):
                return ([], [])
            return query_fn

        def mock_inject_fn_factory(corpus_id: str):
            return None

        def mock_is_citable(meta, doc):
            return (True, None)

        result = execute_multi_corpus_retrieval(
            input=input_data,
            config=config,
            query_fn_factory=mock_query_fn_factory,
            inject_fn_factory=mock_inject_fn_factory,
            is_citable_fn=mock_is_citable,
        )

        assert result.duration_ms >= 0

    def test_per_corpus_hits_tracked(self, import_module):
        """per_corpus_hits dict tracks hit count per corpus."""
        MultiCorpusInput = import_module["MultiCorpusInput"]
        MultiCorpusConfig = import_module["MultiCorpusConfig"]
        execute_multi_corpus_retrieval = import_module["execute_multi_corpus_retrieval"]

        input_data = MultiCorpusInput(
            question="Test question",
            corpus_ids=("corpus_a", "corpus_b"),
            user_profile="LEGAL",
            where_filter=None,
        )
        config = MultiCorpusConfig()

        def mock_query_fn_factory(corpus_id: str):
            def query_fn(question, k, where):
                if corpus_id == "corpus_a":
                    return (
                        [("c1", "doc1", {}), ("c2", "doc2", {})],
                        [0.1, 0.2],
                    )
                else:
                    return (
                        [("c3", "doc3", {})],
                        [0.15],
                    )
            return query_fn

        def mock_inject_fn_factory(corpus_id: str):
            return None

        def mock_is_citable(meta, doc):
            return (True, None)

        result = execute_multi_corpus_retrieval(
            input=input_data,
            config=config,
            query_fn_factory=mock_query_fn_factory,
            inject_fn_factory=mock_inject_fn_factory,
            is_citable_fn=mock_is_citable,
        )

        assert result.per_corpus_hits["corpus_a"] == 2
        assert result.per_corpus_hits["corpus_b"] == 1


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _make_scored_chunk(chunk_id: str, final_score: float, corpus_id: str):
    """Create a mock ScoredChunk for testing."""
    from src.engine.retrieval_pipeline import RetrievedChunk, ScoredChunk

    chunk = RetrievedChunk(
        chunk_id=chunk_id,
        document=f"Document for {chunk_id}",
        metadata={"corpus_id": corpus_id},
        distance=1.0 - final_score,
    )
    return ScoredChunk(
        chunk=chunk,
        vec_score=final_score,
        bm25_score=0.0,
        citation_score=0.0,
        role_score=0.0,
        final_score=final_score,
    )
