"""Tests for the new modular retrieval pipeline.

This module tests the clean, testable retrieval pipeline with immutable
dataclasses and pure functions.

Pipeline Stages:
    1. VectorRetrieval   → Initial embedding search
    2. CitationExpansion → Graph-based article discovery + chunk injection
    3. HybridRerank      → 4-factor scoring (vec + bm25 + citation + role)
    4. ContextSelection  → Citable filter + diversity + top-k cap
"""

from __future__ import annotations

import pytest
from typing import Any, Dict, List, Tuple

from src.engine.retrieval_pipeline import (
    PipelineConfig,
    PipelineInput,
    PipelineResult,
    RetrievedChunk,
    ScoredChunk,
    SelectedChunk,
    VectorRetrievalInput,
    VectorRetrievalResult,
    CitationExpansionInput,
    CitationExpansionResult,
    HybridRerankInput,
    HybridRerankResult,
    ContextSelectionInput,
    ContextSelectionResult,
    execute_vector_retrieval,
    execute_citation_expansion,
    execute_hybrid_rerank,
    execute_context_selection,
    execute_pipeline,
)
from src.common.config_loader import RankingWeights


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chunks() -> List[RetrievedChunk]:
    """Create sample chunks for testing."""
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            document="Article 6 defines high-risk AI systems.",
            metadata={"article": "6", "corpus_id": "ai-act"},
            distance=0.2,
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document="Annex III lists high-risk use cases.",
            metadata={"annex": "III", "corpus_id": "ai-act"},
            distance=0.3,
        ),
        RetrievedChunk(
            chunk_id="chunk-3",
            document="Point 5 covers credit assessment.",
            metadata={"annex": "III", "annex_point": "5", "corpus_id": "ai-act"},
            distance=0.4,
        ),
        RetrievedChunk(
            chunk_id="chunk-4",
            document="Article 2 defines the scope.",
            metadata={"article": "2", "corpus_id": "ai-act"},
            distance=0.5,
        ),
    ]


@pytest.fixture
def mock_query_fn():
    """Create a mock query function that returns predictable results."""
    def query_fn(
        question: str, k: int, where: Dict[str, Any] | None
    ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[float]]:
        # Return mock hits
        hits = [
            ("chunk-1", "Article 6 defines high-risk AI.", {"article": "6"}),
            ("chunk-2", "Annex III lists use cases.", {"annex": "III"}),
            ("chunk-3", "Point 5 covers credit assessment.", {"annex": "III", "annex_point": "5"}),
        ]
        distances = [0.2, 0.3, 0.4]
        return hits[:k], distances[:k]
    return query_fn


@pytest.fixture
def mock_is_citable_fn():
    """Create a mock is_citable function."""
    def is_citable_fn(meta: Dict[str, Any], doc: str) -> Tuple[bool, str | None]:
        # Citable if has article or annex
        if meta.get("article") or meta.get("annex"):
            return True, None
        return False, None
    return is_citable_fn


# ---------------------------------------------------------------------------
# Test RetrievedChunk
# ---------------------------------------------------------------------------

class TestRetrievedChunk:
    """Test RetrievedChunk dataclass."""

    def test_anchor_key_article(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="1",
            document="test",
            metadata={"article": "6"},
            distance=0.1,
        )
        assert chunk.anchor_key() == "article:6"

    def test_anchor_key_annex(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="1",
            document="test",
            metadata={"annex": "III"},
            distance=0.1,
        )
        assert chunk.anchor_key() == "annex:iii"

    def test_anchor_key_annex_with_point(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="1",
            document="test",
            metadata={"annex": "III", "annex_point": "5"},
            distance=0.1,
        )
        assert chunk.anchor_key() == "annex:iii:5"

    def test_anchor_key_recital(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="1",
            document="test",
            metadata={"recital": "42"},
            distance=0.1,
        )
        assert chunk.anchor_key() == "recital:42"

    def test_anchor_key_none(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="1",
            document="test",
            metadata={},
            distance=0.1,
        )
        assert chunk.anchor_key() is None

    def test_to_hit_tuple(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="1",
            document="test doc",
            metadata={"article": "6"},
            distance=0.1,
        )
        doc, meta = chunk.to_hit_tuple()
        assert doc == "test doc"
        assert meta == {"article": "6"}


# ---------------------------------------------------------------------------
# Test PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_values(self) -> None:
        config = PipelineConfig()
        assert config.retrieval_pool_size == 50
        assert config.max_context_legal == 20
        assert config.max_context_engineering == 15
        assert config.alpha_vec == 0.25
        assert config.beta_bm25 == 0.25
        assert config.gamma_cite == 0.35
        assert config.delta_role == 0.15

    def test_frozen(self) -> None:
        config = PipelineConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.retrieval_pool_size = 100  # type: ignore


# ---------------------------------------------------------------------------
# Test Stage 1: Vector Retrieval
# ---------------------------------------------------------------------------

class TestVectorRetrieval:
    """Test execute_vector_retrieval function."""

    def test_basic_retrieval(self, mock_query_fn) -> None:
        result = execute_vector_retrieval(
            VectorRetrievalInput(question="test query", k=3),
            query_fn=mock_query_fn,
        )
        assert len(result.chunks) == 3
        assert result.retrieval_mode == "standard"
        assert result.duration_ms >= 0

    def test_filtered_retrieval(self, mock_query_fn) -> None:
        result = execute_vector_retrieval(
            VectorRetrievalInput(
                question="test query",
                k=3,
                where_filter={"corpus_id": "ai-act"},
            ),
            query_fn=mock_query_fn,
        )
        assert result.retrieval_mode == "filtered"

    def test_chunks_are_immutable(self, mock_query_fn) -> None:
        result = execute_vector_retrieval(
            VectorRetrievalInput(question="test", k=3),
            query_fn=mock_query_fn,
        )
        # Tuple is immutable
        assert isinstance(result.chunks, tuple)


# ---------------------------------------------------------------------------
# Test Stage 2: Citation Expansion
# ---------------------------------------------------------------------------

class TestCitationExpansion:
    """Test execute_citation_expansion function."""

    def test_disabled_expansion(self, sample_chunks) -> None:
        result = execute_citation_expansion(
            CitationExpansionInput(
                chunks=tuple(sample_chunks),
                question="test",
                corpus_id="ai-act",
            ),
            expansion_enabled=False,
        )
        # Should return original chunks unchanged
        assert len(result.chunks) == len(sample_chunks)
        assert len(result.hint_anchors) == 0
        assert result.chunks_injected == 0

    def test_empty_chunks(self) -> None:
        result = execute_citation_expansion(
            CitationExpansionInput(
                chunks=(),
                question="test",
                corpus_id="ai-act",
            ),
            expansion_enabled=True,
        )
        assert len(result.chunks) == 0


# ---------------------------------------------------------------------------
# Test Stage 3: Hybrid Rerank
# ---------------------------------------------------------------------------

class TestHybridRerank:
    """Test execute_hybrid_rerank function."""

    def test_empty_chunks(self) -> None:
        result = execute_hybrid_rerank(
            HybridRerankInput(
                chunks=(),
                question="test",
                hint_anchors=frozenset(),
                user_profile="LEGAL",
            ),
        )
        assert len(result.scored_chunks) == 0
        assert result.query_intent == "UNKNOWN"

    def test_rerank_with_citation_boost(self, sample_chunks) -> None:
        # Chunk 3 (annex:iii:5) should get boosted when in hint_anchors
        result = execute_hybrid_rerank(
            HybridRerankInput(
                chunks=tuple(sample_chunks),
                question="pension AI classification",
                hint_anchors=frozenset(["annex:iii:5", "article:6"]),
                user_profile="LEGAL",
            ),
        )
        assert len(result.scored_chunks) == len(sample_chunks)

        # Check that citation signal is 1.0 for matched anchors
        for sc in result.scored_chunks:
            anchor = sc.chunk.anchor_key()
            if anchor in ["article:6", "annex:iii:5"]:
                assert sc.citation_score == 1.0
            else:
                assert sc.citation_score == 0.0

    def test_sorted_by_final_score(self, sample_chunks) -> None:
        result = execute_hybrid_rerank(
            HybridRerankInput(
                chunks=tuple(sample_chunks),
                question="test",
                hint_anchors=frozenset(),
                user_profile="LEGAL",
            ),
        )
        scores = [sc.final_score for sc in result.scored_chunks]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Test Stage 4: Context Selection
# ---------------------------------------------------------------------------

class TestContextSelection:
    """Test execute_context_selection function."""

    def test_filters_to_citable(self, sample_chunks, mock_is_citable_fn) -> None:
        # Create scored chunks
        scored = tuple(
            ScoredChunk(
                chunk=c,
                vec_score=0.5,
                bm25_score=0.5,
                citation_score=0.0,
                role_score=0.5,
                final_score=0.5,
            )
            for c in sample_chunks
        )

        result = execute_context_selection(
            ContextSelectionInput(
                scored_chunks=scored,
                max_context=10,
                user_profile="LEGAL",
            ),
            is_citable_fn=mock_is_citable_fn,
        )

        # All sample chunks have article or annex, so all should be citable
        assert result.citable_count == len(sample_chunks)
        assert len(result.selected) == len(sample_chunks)

    def test_respects_max_context(self, sample_chunks, mock_is_citable_fn) -> None:
        scored = tuple(
            ScoredChunk(
                chunk=c,
                vec_score=0.5,
                bm25_score=0.5,
                citation_score=0.0,
                role_score=0.5,
                final_score=0.5,
            )
            for c in sample_chunks
        )

        result = execute_context_selection(
            ContextSelectionInput(
                scored_chunks=scored,
                max_context=2,  # Only allow 2
                user_profile="LEGAL",
            ),
            is_citable_fn=mock_is_citable_fn,
        )

        assert len(result.selected) == 2


# ---------------------------------------------------------------------------
# Test Full Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Test execute_pipeline function."""

    def test_full_pipeline_execution(self, mock_query_fn, mock_is_citable_fn) -> None:
        config = PipelineConfig(
            retrieval_pool_size=10,
            max_context_legal=5,
            citation_expansion_enabled=False,  # Disable for test
        )

        result = execute_pipeline(
            input=PipelineInput(
                question="What are high-risk AI systems?",
                corpus_id="ai-act",
                user_profile="LEGAL",
            ),
            config=config,
            query_fn=mock_query_fn,
            inject_fn=None,
            is_citable_fn=mock_is_citable_fn,
        )

        # Verify result structure
        assert isinstance(result, PipelineResult)
        assert result.total_duration_ms >= 0

        # Check all stages ran
        assert len(result.vector_result.chunks) > 0
        assert result.rerank_result.query_intent != "UNKNOWN"
        assert len(result.context_result.selected) > 0

    def test_debug_summary_format(self, mock_query_fn, mock_is_citable_fn) -> None:
        config = PipelineConfig(citation_expansion_enabled=False)

        result = execute_pipeline(
            input=PipelineInput(
                question="test",
                corpus_id="ai-act",
                user_profile="LEGAL",
            ),
            config=config,
            query_fn=mock_query_fn,
            inject_fn=None,
            is_citable_fn=mock_is_citable_fn,
        )

        debug = result.debug_summary

        # Verify debug structure (compatible with eval scorer)
        assert "vector_retrieval" in debug
        assert "citation_expansion_articles" in debug
        assert "hybrid_rerank" in debug
        assert "context_selection" in debug
        assert "total_duration_ms" in debug

    def test_get_hits_format(self, mock_query_fn, mock_is_citable_fn) -> None:
        config = PipelineConfig(citation_expansion_enabled=False)

        result = execute_pipeline(
            input=PipelineInput(
                question="test",
                corpus_id="ai-act",
                user_profile="LEGAL",
            ),
            config=config,
            query_fn=mock_query_fn,
            inject_fn=None,
            is_citable_fn=mock_is_citable_fn,
        )

        hits = result.get_hits()
        distances = result.get_distances()
        ids = result.get_ids()

        # Verify format matches legacy expectations
        assert len(hits) == len(distances) == len(ids)
        for doc, meta in hits:
            assert isinstance(doc, str)
            assert isinstance(meta, dict)


# ---------------------------------------------------------------------------
# Test Ranking Weight Application
# ---------------------------------------------------------------------------

class TestRankingWeights:
    """Test that ranking weights are correctly applied."""

    def test_high_citation_weight_boosts_matched_anchors(self, sample_chunks) -> None:
        """When gamma_cite is high, matched anchors should rank higher."""
        high_cite_weights = RankingWeights(
            alpha_vec=0.1,
            beta_bm25=0.1,
            gamma_cite=0.7,  # High citation weight
            delta_role=0.1,
        )

        result = execute_hybrid_rerank(
            HybridRerankInput(
                chunks=tuple(sample_chunks),
                question="pension AI classification",
                hint_anchors=frozenset(["annex:iii:5"]),  # Only boost chunk 3
                user_profile="LEGAL",
                weights=high_cite_weights,
            ),
        )

        # Chunk 3 (annex:iii:5) should be near the top due to citation boost
        top_anchors = [sc.chunk.anchor_key() for sc in result.scored_chunks[:2]]
        assert "annex:iii:5" in top_anchors
