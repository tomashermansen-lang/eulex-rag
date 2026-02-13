"""Tests for RAGEngine cross-law orchestration.

TDD: These tests verify that rag.py correctly routes to multi_corpus_retrieval
when corpus_scope is not 'single'.
"""

import pytest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.rag import RAGEngine, _RetrievalResult
from src.engine.retrieval_pipeline import RetrievedChunk, SelectedChunk
from src.common.config_loader import RankingWeights


def _make_mock_retrieval_result(hits=None):
    """Create a mock _RetrievalResult for testing."""
    if hits is None:
        hits = []

    selected_chunks = []
    for i, (doc, meta) in enumerate(hits):
        chunk = RetrievedChunk(
            chunk_id=f"chunk-{i}",
            document=doc,
            metadata=dict(meta),
            distance=0.1,
        )
        selected_chunks.append(
            SelectedChunk(
                chunk=chunk,
                is_citable=True,
                precise_ref=None,
                rank=i,
            )
        )

    return _RetrievalResult(
        hits=hits,
        distances=[0.1] * len(hits),
        retrieved_ids=[f"chunk-{i}" for i in range(len(hits))],
        retrieved_metas=[meta for _, meta in hits],
        run_meta_updates={},
        selected_chunks=tuple(selected_chunks),
        total_retrieved=len(hits),
        citable_count=len(hits),
    )


def _make_mock_multi_corpus_result(fused_chunks=None, per_corpus_hits=None):
    """Create a mock MultiCorpusResult for testing."""
    from src.engine.multi_corpus_retrieval import MultiCorpusResult

    if fused_chunks is None:
        fused_chunks = ()
    if per_corpus_hits is None:
        per_corpus_hits = {}

    return MultiCorpusResult(
        fused_chunks=tuple(fused_chunks),
        per_corpus_hits=per_corpus_hits,
        duration_ms=10.0,
        debug={},
    )


def _setup_mock_engine():
    """Set up a mock RAGEngine with required attributes."""
    engine = RAGEngine.__new__(RAGEngine)
    engine.ranking_weights = RankingWeights()
    engine.enable_hybrid_rerank = True
    engine.corpus_id = "ai-act"
    engine.top_k = 3

    # Set up mock retriever
    engine._retriever = SimpleNamespace(
        _embed=lambda texts: [[0.0] * 1536] * len(texts),
        _query_collection_raw=lambda **k: ([], [], [], []),
        _query_collection_with_distances=lambda **k: ([], []),
        _last_retrieved_ids=[],
        _last_retrieved_metadatas=[],
        _last_distances=[],
    )

    # Mock collection
    engine._collection = MagicMock()

    # Mock LLM client
    engine._llm_client = MagicMock()
    engine._llm_client.chat_completion_with_cot.return_value = (
        "Test answer [1].",
        {"reasoning": "test"},
    )

    return engine


class TestCrossLawOrchestration:
    """Tests for cross-law routing in answer_structured."""

    def test_rcl_001_answer_structured_accepts_corpus_scope_parameter(self, monkeypatch):
        """answer_structured should accept corpus_scope parameter."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        mock_result = _make_mock_retrieval_result([])
        engine._modular_retrieval = MagicMock(return_value=mock_result)

        # Should not raise when passing corpus_scope
        try:
            engine.answer_structured(
                question="What is Article 5?",
                user_profile="LEGAL",
                corpus_scope="single",  # New parameter
                dry_run=True,
            )
        except TypeError as e:
            if "corpus_scope" in str(e):
                pytest.fail("answer_structured should accept corpus_scope parameter")
            raise

    def test_rcl_002_answer_structured_accepts_target_corpora_parameter(self, monkeypatch):
        """answer_structured should accept target_corpora parameter."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        mock_result = _make_mock_retrieval_result([])
        engine._modular_retrieval = MagicMock(return_value=mock_result)
        # Mock cross-law retrieval since we're testing with explicit scope
        engine._execute_cross_law_retrieval = MagicMock(return_value=mock_result)

        # Should not raise when passing target_corpora
        try:
            engine.answer_structured(
                question="What is Article 5?",
                user_profile="LEGAL",
                corpus_scope="explicit",  # Use explicit scope
                target_corpora=["ai-act", "gdpr"],  # New parameter
                dry_run=True,
            )
        except TypeError as e:
            if "target_corpora" in str(e):
                pytest.fail("answer_structured should accept target_corpora parameter")
            raise

    def test_rcl_003_single_scope_uses_modular_retrieval(self, monkeypatch):
        """corpus_scope='single' should use _modular_retrieval."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        mock_result = _make_mock_retrieval_result([])
        engine._modular_retrieval = MagicMock(return_value=mock_result)

        with patch("src.engine.rag.execute_multi_corpus_retrieval") as mock_multi:
            engine.answer_structured(
                question="What is Article 5?",
                user_profile="LEGAL",
                corpus_scope="single",
                dry_run=True,
            )

            # Should use _modular_retrieval
            assert engine._modular_retrieval.called
            # Should NOT use multi_corpus_retrieval
            assert not mock_multi.called

    def test_rcl_004_all_scope_uses_multi_corpus_retrieval(self, monkeypatch):
        """corpus_scope='all' should use execute_multi_corpus_retrieval."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        engine._available_corpora = MagicMock(return_value=["ai-act", "gdpr", "nis2"])

        mock_result = _make_mock_retrieval_result([])
        engine._modular_retrieval = MagicMock(return_value=mock_result)

        # Mock multi-corpus retrieval to return compatible result
        mock_multi_result = _make_mock_multi_corpus_result(
            fused_chunks=[],
            per_corpus_hits={"ai-act": 0, "gdpr": 0, "nis2": 0},
        )
        with patch(
            "src.engine.rag.execute_multi_corpus_retrieval",
            return_value=mock_multi_result,
        ) as mock_multi:
            engine.answer_structured(
                question="What is Article 5?",
                user_profile="LEGAL",
                corpus_scope="all",
                dry_run=True,
            )

            # Should use multi_corpus_retrieval
            assert mock_multi.called
            # Should query all available corpora
            call_args = mock_multi.call_args
            input_obj = call_args.kwargs.get("input")
            assert set(input_obj.corpus_ids) == {"ai-act", "gdpr", "nis2"}

    def test_rcl_005_explicit_scope_uses_target_corpora(self, monkeypatch):
        """corpus_scope='explicit' should use specified target_corpora."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        mock_result = _make_mock_retrieval_result([])
        engine._modular_retrieval = MagicMock(return_value=mock_result)

        mock_multi_result = _make_mock_multi_corpus_result(
            fused_chunks=[],
            per_corpus_hits={"gdpr": 0, "nis2": 0},
        )
        with patch(
            "src.engine.rag.execute_multi_corpus_retrieval",
            return_value=mock_multi_result,
        ) as mock_multi:
            engine.answer_structured(
                question="What is Article 5?",
                user_profile="LEGAL",
                corpus_scope="explicit",
                target_corpora=["gdpr", "nis2"],  # Specific selection
                dry_run=True,
            )

            # Should use multi_corpus_retrieval with specified corpora
            assert mock_multi.called
            call_args = mock_multi.call_args
            input_obj = call_args.kwargs.get("input")
            assert set(input_obj.corpus_ids) == {"gdpr", "nis2"}

    def test_rcl_006_explicit_scope_without_target_uses_current_corpus(self, monkeypatch):
        """corpus_scope='explicit' without target_corpora falls back to current corpus."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        mock_result = _make_mock_retrieval_result([])
        engine._modular_retrieval = MagicMock(return_value=mock_result)

        with patch("src.engine.rag.execute_multi_corpus_retrieval") as mock_multi:
            # With empty target_corpora, should fall back to single corpus mode
            engine.answer_structured(
                question="What is Article 5?",
                user_profile="LEGAL",
                corpus_scope="explicit",
                target_corpora=[],  # Empty
                dry_run=True,
            )

            # Should use _modular_retrieval (single corpus fallback)
            assert engine._modular_retrieval.called
            # Should NOT use multi_corpus_retrieval
            assert not mock_multi.called

    def test_rcl_007_response_includes_laws_searched(self, monkeypatch):
        """Response should include laws_searched field for cross-law queries."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        engine._available_corpora = MagicMock(return_value=["ai-act", "gdpr"])

        mock_multi_result = _make_mock_multi_corpus_result(
            fused_chunks=[],
            per_corpus_hits={"ai-act": 5, "gdpr": 3},
        )

        with patch(
            "src.engine.rag.execute_multi_corpus_retrieval",
            return_value=mock_multi_result,
        ):
            result = engine.answer_structured(
                question="What is Article 5?",
                user_profile="LEGAL",
                corpus_scope="all",
                dry_run=True,
            )

            # Should have laws_searched in run (run_meta is stored as "run" in response)
            assert "laws_searched" in result.get("run", {})
            assert set(result["run"]["laws_searched"]) == {"ai-act", "gdpr"}

    def test_rcl_008_default_corpus_scope_is_single(self, monkeypatch):
        """Default corpus_scope should be 'single' for backwards compatibility."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        mock_result = _make_mock_retrieval_result([])
        engine._modular_retrieval = MagicMock(return_value=mock_result)

        with patch("src.engine.rag.execute_multi_corpus_retrieval") as mock_multi:
            # Call without corpus_scope - should default to single
            engine.answer_structured(
                question="What is Article 5?",
                user_profile="LEGAL",
                dry_run=True,
            )

            # Should use _modular_retrieval (single corpus)
            assert engine._modular_retrieval.called
            assert not mock_multi.called


class TestGetCollectionForCorpus:
    """Tests for _get_collection_for_corpus method.

    TDD: This test verifies the bug fix where _get_collection_for_corpus
    was always returning self.collection instead of the correct collection
    for each corpus_id.
    """

    def test_rcl_009_get_collection_returns_correct_collection_for_corpus_id(self):
        """_get_collection_for_corpus should return collection named {corpus_id}_documents."""
        engine = RAGEngine.__new__(RAGEngine)
        engine.corpus_id = "ai-act"

        # Mock the chroma client
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "cyberrobusthed_documents"
        mock_chroma.get_or_create_collection.return_value = mock_collection
        engine.chroma = mock_chroma

        # Set up the engine's own collection (the primary one)
        engine.collection = MagicMock()
        engine.collection.name = "ai-act_documents"

        # Call with a DIFFERENT corpus_id
        result = engine._get_collection_for_corpus("cyberrobusthed")

        # Should call chroma.get_or_create_collection with correct name
        mock_chroma.get_or_create_collection.assert_called_once_with("cyberrobusthed_documents")

        # Should return the looked-up collection, NOT self.collection
        assert result == mock_collection
        assert result.name == "cyberrobusthed_documents"
        assert result != engine.collection

    def test_rcl_010_get_collection_returns_different_collections_for_different_ids(self):
        """_get_collection_for_corpus should return different collections for different corpus_ids."""
        engine = RAGEngine.__new__(RAGEngine)
        engine.corpus_id = "ai-act"

        # Track which collections are created
        created_collections = {}

        def mock_get_or_create(name):
            if name not in created_collections:
                mock_col = MagicMock()
                mock_col.name = name
                created_collections[name] = mock_col
            return created_collections[name]

        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.side_effect = mock_get_or_create
        engine.chroma = mock_chroma

        engine.collection = MagicMock()
        engine.collection.name = "ai-act_documents"

        # Get collections for different corpus_ids
        col_cyber = engine._get_collection_for_corpus("cyberrobusthed")
        col_gdpr = engine._get_collection_for_corpus("gdpr")
        col_aiact = engine._get_collection_for_corpus("ai-act")

        # Each should have correct name
        assert col_cyber.name == "cyberrobusthed_documents"
        assert col_gdpr.name == "gdpr_documents"
        assert col_aiact.name == "ai-act_documents"

        # They should be different objects (unless same corpus_id)
        assert col_cyber != col_gdpr
        assert col_cyber != col_aiact


def _make_fused_chunks_for_multi_corpus(hits):
    """Create ScoredChunk objects suitable for MultiCorpusResult.fused_chunks."""
    from src.engine.retrieval_pipeline import RetrievedChunk, ScoredChunk

    fused_chunks = []
    for i, (doc, meta) in enumerate(hits):
        chunk = RetrievedChunk(
            chunk_id=f"chunk-{i}",
            document=doc,
            metadata=dict(meta),
            distance=0.1,
        )
        scored = ScoredChunk(
            chunk=chunk,
            vec_score=0.9,
            bm25_score=0.0,
            citation_score=0.0,
            role_score=0.0,
            final_score=0.9 - i * 0.01,
        )
        fused_chunks.append(scored)
    return fused_chunks


class TestCrossLawAbstainGuardrail:
    """Tests for abstain guardrail behavior in multi-corpus mode.

    Bug: When corpus_scope='explicit' with target_corpora=['ai-act', 'gdpr'],
    and the question mentions both laws (e.g., "AI-forordningen og GDPR"),
    the abstain logic incorrectly rejects the question saying
    "Jeg kan kun svare ud fra ét corpus ad gangen."

    Expected: The abstain guardrail should be bypassed when the user has
    explicitly selected multiple corpora.
    """

    def test_rcl_011_should_not_abstain_when_explicit_scope_matches_mentioned_laws(self, monkeypatch):
        """When corpus_scope='explicit' and question mentions laws in target_corpora,
        the system should NOT abstain.

        This tests the bug where asking "Hvad siger AI-forordningen og GDPR om X?"
        with explicit scope selecting both laws causes an incorrect abstain.
        """
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()

        # Set up mock retrieval that returns hits from both corpora
        hits = [
            ("AI transparency rules about notifying users", {"corpus_id": "ai-act", "article": "50"}),
            ("GDPR transparency requirements for data processing", {"corpus_id": "gdpr", "article": "12"}),
        ]
        fused_chunks = _make_fused_chunks_for_multi_corpus(hits)

        mock_multi_result = _make_mock_multi_corpus_result(
            fused_chunks=fused_chunks,
            per_corpus_hits={"ai-act": 10, "gdpr": 10},
        )

        with patch(
            "src.engine.rag.execute_multi_corpus_retrieval",
            return_value=mock_multi_result,
        ):
            result = engine.answer_structured(
                question="Hvad siger AI-forordningen og GDPR om gennemsigtighed?",
                user_profile="LEGAL",
                corpus_scope="explicit",
                target_corpora=["ai-act", "gdpr"],
                dry_run=True,
            )

        # Should NOT abstain - the user explicitly selected these corpora
        abstain_info = result.get("run", {}).get("abstain", {})
        did_abstain = abstain_info.get("abstained", False)
        abstain_reason = abstain_info.get("reason", "")

        # This assertion currently FAILS due to the bug
        assert not did_abstain, (
            f"Should not abstain when corpus_scope='explicit' matches mentioned laws. "
            f"Got abstain reason: {abstain_reason}"
        )


class TestSynthesisModeIntegration:
    """Tests for synthesis mode detection and routing in answer_structured."""

    def test_rcl_012_detects_synthesis_mode_for_multi_corpus(self, monkeypatch):
        """answer_structured should detect synthesis mode when corpus_scope != 'single'."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        engine._available_corpora = MagicMock(return_value=["ai-act", "gdpr"])

        # Create mock chunks with corpus_id
        hits = [
            ("AI Act chunk", {"corpus_id": "ai-act", "article": "5"}),
            ("GDPR chunk", {"corpus_id": "gdpr", "article": "6"}),
        ]
        fused_chunks = _make_fused_chunks_for_multi_corpus(hits)
        mock_multi_result = _make_mock_multi_corpus_result(
            fused_chunks=fused_chunks,
            per_corpus_hits={"ai-act": 1, "gdpr": 1},
        )

        with patch(
            "src.engine.rag.execute_multi_corpus_retrieval",
            return_value=mock_multi_result,
        ):
            result = engine.answer_structured(
                question="What are the requirements?",
                user_profile="LEGAL",
                corpus_scope="all",
                dry_run=True,
            )

        # Should have synthesis_mode in run_meta
        run_meta = result.get("run", {})
        assert "synthesis_mode" in run_meta, "run_meta should contain synthesis_mode"

    def test_rcl_013_unified_mode_for_generic_query(self, monkeypatch):
        """Generic query without comparison/aggregation keywords should use UNIFIED mode."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        engine._available_corpora = MagicMock(return_value=["ai-act", "gdpr"])

        hits = [
            ("AI Act chunk", {"corpus_id": "ai-act", "article": "5"}),
        ]
        fused_chunks = _make_fused_chunks_for_multi_corpus(hits)
        mock_multi_result = _make_mock_multi_corpus_result(
            fused_chunks=fused_chunks,
            per_corpus_hits={"ai-act": 1},
        )

        with patch(
            "src.engine.rag.execute_multi_corpus_retrieval",
            return_value=mock_multi_result,
        ):
            result = engine.answer_structured(
                question="What are the penalties?",  # No aggregation/comparison keywords
                user_profile="LEGAL",
                corpus_scope="all",
                dry_run=True,
            )

        run_meta = result.get("run", {})
        synthesis_mode = run_meta.get("synthesis_mode", "")
        # Should be UNIFIED (default for generic queries)
        assert synthesis_mode == "UNIFIED" or synthesis_mode == "unified", (
            f"Expected UNIFIED mode for generic query, got: {synthesis_mode}"
        )

    def test_rcl_014_aggregation_mode_for_all_laws_query(self, monkeypatch):
        """Query with 'alle love' should detect AGGREGATION mode."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        engine._available_corpora = MagicMock(return_value=["ai-act", "gdpr"])

        hits = [
            ("AI Act chunk", {"corpus_id": "ai-act", "article": "5"}),
            ("GDPR chunk", {"corpus_id": "gdpr", "article": "6"}),
        ]
        fused_chunks = _make_fused_chunks_for_multi_corpus(hits)
        mock_multi_result = _make_mock_multi_corpus_result(
            fused_chunks=fused_chunks,
            per_corpus_hits={"ai-act": 1, "gdpr": 1},
        )

        with patch(
            "src.engine.rag.execute_multi_corpus_retrieval",
            return_value=mock_multi_result,
        ):
            result = engine.answer_structured(
                question="Hvad siger alle love om transparens?",  # "alle love" triggers aggregation
                user_profile="LEGAL",
                corpus_scope="all",
                dry_run=True,
            )

        run_meta = result.get("run", {})
        synthesis_mode = run_meta.get("synthesis_mode", "")
        assert synthesis_mode == "AGGREGATION" or synthesis_mode == "aggregation", (
            f"Expected AGGREGATION mode for 'alle love' query, got: {synthesis_mode}"
        )

    def test_rcl_015_single_corpus_does_not_detect_synthesis_mode(self, monkeypatch):
        """corpus_scope='single' should not trigger synthesis mode detection."""
        monkeypatch.setenv("INTENT_ROUTER_DISABLED", "1")

        engine = _setup_mock_engine()
        mock_result = _make_mock_retrieval_result([])
        engine._modular_retrieval = MagicMock(return_value=mock_result)

        result = engine.answer_structured(
            question="What is Article 5?",
            user_profile="LEGAL",
            corpus_scope="single",
            dry_run=True,
        )

        run_meta = result.get("run", {})
        # Should NOT have synthesis_mode for single corpus
        assert run_meta.get("synthesis_mode") is None or run_meta.get("corpus_scope") == "single"
