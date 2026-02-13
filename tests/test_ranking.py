import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.rag import RAGEngine
from src.engine.ranking import Ranker, RankingPipelineResult, execute_ranking_pipeline
from src.engine.planning import UserProfile
from src.common.config_loader import RankingWeights


# =============================================================================
# _tokenize_for_lexical tests
# =============================================================================

class TestTokenizeForLexical:
    """Tests for Ranker._tokenize_for_lexical()."""

    def test_tokenize_empty_string(self):
        """Empty string returns empty list."""
        assert Ranker._tokenize_for_lexical("") == []

    def test_tokenize_none_returns_empty(self):
        """None-like falsy input returns empty list."""
        assert Ranker._tokenize_for_lexical(None) == []  # type: ignore[arg-type]

    def test_tokenize_simple_text(self):
        """Simple text tokenized to lowercase words."""
        result = Ranker._tokenize_for_lexical("Hello World")
        assert result == ["hello", "world"]

    def test_tokenize_danish_characters(self):
        """Danish characters (æ, ø, å) are preserved."""
        result = Ranker._tokenize_for_lexical("Æble Ørred Århus")
        assert result == ["æble", "ørred", "århus"]

    def test_tokenize_with_numbers(self):
        """Numbers are included in tokens."""
        result = Ranker._tokenize_for_lexical("Article 10 paragraph 2")
        assert result == ["article", "10", "paragraph", "2"]

    def test_tokenize_strips_punctuation(self):
        """Punctuation is removed."""
        result = Ranker._tokenize_for_lexical("Hello, world! How are you?")
        assert result == ["hello", "world", "how", "are", "you"]


# =============================================================================
# _bm25_scores tests
# =============================================================================

class TestBm25Scores:
    """Tests for Ranker._bm25_scores()."""

    def test_bm25_empty_query(self):
        """Empty query returns zeros for all documents."""
        result = Ranker._bm25_scores(query="", documents=["doc1", "doc2"])
        assert result == [0.0, 0.0]

    def test_bm25_empty_documents(self):
        """Empty document list returns empty list."""
        result = Ranker._bm25_scores(query="test query", documents=[])
        assert result == []

    def test_bm25_no_match(self):
        """Documents with no matching terms get low scores."""
        result = Ranker._bm25_scores(
            query="artificial intelligence",
            documents=["the quick brown fox", "jumped over lazy dog"]
        )
        assert all(score == 0.0 for score in result)

    def test_bm25_exact_match_scores_higher(self):
        """Document with exact query terms scores higher."""
        result = Ranker._bm25_scores(
            query="artificial intelligence",
            documents=[
                "artificial intelligence is important",
                "the quick brown fox"
            ]
        )
        assert result[0] > result[1]
        assert result[1] == 0.0

    def test_bm25_partial_match(self):
        """Partial match gets non-zero score."""
        result = Ranker._bm25_scores(
            query="artificial intelligence system",
            documents=["artificial learning system", "random text"]
        )
        assert result[0] > 0.0
        assert result[1] == 0.0

    def test_bm25_repeated_query_terms_not_overweighted(self):
        """Repeated query terms are deduplicated."""
        result1 = Ranker._bm25_scores(
            query="test test test",
            documents=["test document"]
        )
        result2 = Ranker._bm25_scores(
            query="test",
            documents=["test document"]
        )
        # Scores should be equal since repeated terms are deduplicated
        assert result1[0] == result2[0]


# =============================================================================
# _normalize_scores tests
# =============================================================================

class TestNormalizeScores:
    """Tests for Ranker._normalize_scores()."""

    def test_normalize_empty_list(self):
        """Empty list returns empty list."""
        assert Ranker._normalize_scores([]) == []

    def test_normalize_single_value(self):
        """Single value normalizes to 0.0 (no range)."""
        assert Ranker._normalize_scores([5.0]) == [0.0]

    def test_normalize_identical_values(self):
        """Identical values all normalize to 0.0."""
        assert Ranker._normalize_scores([3.0, 3.0, 3.0]) == [0.0, 0.0, 0.0]

    def test_normalize_min_max(self):
        """Values normalized to [0, 1] range."""
        result = Ranker._normalize_scores([0.0, 5.0, 10.0])
        assert result == [0.0, 0.5, 1.0]

    def test_normalize_preserves_order(self):
        """Relative order preserved after normalization."""
        result = Ranker._normalize_scores([2.0, 8.0, 4.0, 10.0])
        assert result[0] < result[2] < result[1] < result[3]
        assert result[0] == 0.0
        assert result[3] == 1.0


# =============================================================================
# classify_role tests
# =============================================================================

class TestClassifyRole:
    """Tests for Ranker.classify_role()."""

    def test_classify_role_recital_from_metadata(self):
        """Recital in metadata → RECITAL_CONTEXT role."""
        meta = {"recital": "12"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_RECITAL_CONTEXT
        assert result["confidence"] == 0.95
        assert "meta.recital_present" in result["signals"]

    def test_classify_role_recital_from_doc_type(self):
        """Recital in doc_type → RECITAL_CONTEXT role."""
        meta = {"doc_type": "recital"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_RECITAL_CONTEXT

    def test_classify_role_recital_from_location_id(self):
        """Recital in location_id → RECITAL_CONTEXT role."""
        meta = {"location_id": "recital:12"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_RECITAL_CONTEXT

    def test_classify_role_preamble_from_location_id(self):
        """Preamble in location_id → RECITAL_CONTEXT role."""
        meta = {"location_id": "preamble"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_RECITAL_CONTEXT

    def test_classify_role_definition_from_heading(self):
        """Definition in heading → DEFINITION_SCOPE role."""
        meta = {"heading_path_display": "Definitions"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_DEFINITION_SCOPE
        assert "heading:definition_or_scope" in result["signals"]

    def test_classify_role_prohibition_from_heading(self):
        """Prohibited in heading → PROHIBITIONS role."""
        meta = {"title": "Prohibited AI Practices"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_PROHIBITIONS

    def test_classify_role_transparency_from_heading(self):
        """Transparency in heading → TRANSPARENCY role."""
        meta = {"heading_path_display": "Transparency obligations"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_TRANSPARENCY

    def test_classify_role_enforcement_from_heading(self):
        """Penalty in heading → ENFORCEMENT_SUPERVISION role."""
        meta = {"title": "Penalties"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_ENFORCEMENT_SUPERVISION

    def test_classify_role_sandbox_from_heading(self):
        """Sandbox in heading → EXCEPTIONS_SANDBOX role."""
        meta = {"heading_path_display": "AI Sandbox"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_EXCEPTIONS_SANDBOX

    def test_classify_role_governance_from_heading(self):
        """Risk management in heading → GOVERNANCE_PROCESS role."""
        meta = {"title": "Risk Management System"}
        result = Ranker.classify_role(meta, "Some text")
        assert result["role"] == Ranker.ROLE_GOVERNANCE_PROCESS

    def test_classify_role_obligations_from_text(self):
        """Modal verbs in text → OBLIGATIONS_NORMATIVE signals."""
        meta = {}
        result = Ranker.classify_role(meta, "The provider shall ensure compliance.")
        assert "text:modal_must_shall_required" in result["signals"]

    def test_classify_role_danish_modal_skal(self):
        """Danish modal 'skal' detected."""
        meta = {}
        result = Ranker.classify_role(meta, "Udbyderen skal sikre overholdelse.")
        assert "text:modal_skal" in result["signals"]

    def test_classify_role_transparency_from_text(self):
        """Transparency keywords in text → signals added."""
        meta = {}
        result = Ranker.classify_role(meta, "We must inform users about AI usage.")
        assert "text:inform_user" in result["signals"]

    def test_classify_role_fallback_no_signals(self):
        """No signals → fallback to OBLIGATIONS_NORMATIVE with low confidence."""
        meta = {}
        result = Ranker.classify_role(meta, "Generic text without keywords.")
        assert result["role"] == Ranker.ROLE_OBLIGATIONS_NORMATIVE
        assert result["confidence"] == 0.20
        assert "fallback:no_signals" in result["signals"]

    def test_classify_role_none_metadata(self):
        """None metadata handled gracefully."""
        result = Ranker.classify_role(None, "Some text")
        assert "role" in result
        assert "confidence" in result
        assert "signals" in result


# =============================================================================
# _rerank_delta_for_role tests
# =============================================================================

class TestRerankDeltaForRole:
    """Tests for Ranker._rerank_delta_for_role()."""

    def test_delta_transparency_intent_matches_transparency_role(self):
        """TRANSPARENCY intent + TRANSPARENCY role → positive delta."""
        delta = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_TRANSPARENCY,
            role_confidence=1.0,
            intent="TRANSPARENCY",
            user_profile=UserProfile.LEGAL,
        )
        assert delta > 0.0

    def test_delta_transparency_intent_penalizes_recital(self):
        """TRANSPARENCY intent + RECITAL_CONTEXT role → negative delta."""
        delta = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_RECITAL_CONTEXT,
            role_confidence=1.0,
            intent="TRANSPARENCY",
            user_profile=UserProfile.LEGAL,
        )
        assert delta < 0.0

    def test_delta_obligations_intent_boosts_normative(self):
        """OBLIGATIONS intent + OBLIGATIONS_NORMATIVE role → positive delta."""
        delta = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_OBLIGATIONS_NORMATIVE,
            role_confidence=1.0,
            intent="OBLIGATIONS",
            user_profile=UserProfile.ENGINEERING,
        )
        assert delta > 0.0

    def test_delta_enforcement_intent_boosts_enforcement(self):
        """ENFORCEMENT intent + ENFORCEMENT_SUPERVISION role → positive delta."""
        delta = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_ENFORCEMENT_SUPERVISION,
            role_confidence=1.0,
            intent="ENFORCEMENT",
            user_profile=UserProfile.LEGAL,
        )
        assert delta > 0.0

    def test_delta_engineering_profile_penalizes_recital(self):
        """ENGINEERING profile penalizes recitals heavily."""
        delta_eng = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_RECITAL_CONTEXT,
            role_confidence=1.0,
            intent="GENERAL",
            user_profile=UserProfile.ENGINEERING,
        )
        delta_legal = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_RECITAL_CONTEXT,
            role_confidence=1.0,
            intent="GENERAL",
            user_profile=UserProfile.LEGAL,
        )
        assert delta_eng < delta_legal

    def test_delta_scaled_by_confidence(self):
        """Delta is scaled by role confidence."""
        delta_high = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_TRANSPARENCY,
            role_confidence=1.0,
            intent="TRANSPARENCY",
            user_profile=UserProfile.LEGAL,
        )
        delta_low = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_TRANSPARENCY,
            role_confidence=0.2,
            intent="TRANSPARENCY",
            user_profile=UserProfile.LEGAL,
        )
        assert abs(delta_high) > abs(delta_low)

    def test_delta_string_profile_engineering(self):
        """String profile 'ENGINEERING' works."""
        delta = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_RECITAL_CONTEXT,
            role_confidence=1.0,
            intent="GENERAL",
            user_profile="engineering",
        )
        assert delta < 0.0

    def test_delta_confidence_clamped(self):
        """Confidence values are clamped to [0, 1]."""
        delta_over = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_TRANSPARENCY,
            role_confidence=2.0,  # Over 1.0
            intent="TRANSPARENCY",
            user_profile=UserProfile.LEGAL,
        )
        delta_normal = Ranker._rerank_delta_for_role(
            role=Ranker.ROLE_TRANSPARENCY,
            role_confidence=1.0,
            intent="TRANSPARENCY",
            user_profile=UserProfile.LEGAL,
        )
        assert delta_over == delta_normal


# =============================================================================
# anchor_score tests (extending existing)
# =============================================================================

class TestAnchorScore:
    """Tests for Ranker.anchor_score()."""

    def test_anchor_score_hint_match_highest(self):
        """Matching hint anchor gets score 5."""
        meta = {"article": "10"}
        hints = {"article:10"}
        assert Ranker.anchor_score(meta, hints) == 5

    def test_anchor_score_hint_match_annex(self):
        """Matching hint anchor for annex gets score 5."""
        meta = {"annex": "iii"}
        hints = {"annex:iii"}
        assert Ranker.anchor_score(meta, hints) == 5

    def test_anchor_score_hint_match_recital(self):
        """Matching hint anchor for recital gets score 5."""
        meta = {"recital": "12"}
        hints = {"recital:12"}
        assert Ranker.anchor_score(meta, hints) == 5

    def test_anchor_score_article_paragraph(self):
        """Article + paragraph without hint → score 4."""
        meta = {"article": "10", "paragraph": "2"}
        assert Ranker.anchor_score(meta) == 4

    def test_anchor_score_article_only(self):
        """Article only → score 3."""
        meta = {"article": "10"}
        assert Ranker.anchor_score(meta) == 3

    def test_anchor_score_recital_only(self):
        """Recital only → score 2."""
        meta = {"recital": "12"}
        assert Ranker.anchor_score(meta) == 2

    def test_anchor_score_annex_only(self):
        """Annex only → score 1."""
        meta = {"annex": "III"}
        assert Ranker.anchor_score(meta) == 1

    def test_anchor_score_chapter_only(self):
        """Chapter only → score 0."""
        meta = {"chapter": "I"}
        assert Ranker.anchor_score(meta) == 0

    def test_anchor_score_none_metadata(self):
        """None metadata → score 0."""
        assert Ranker.anchor_score(None) == 0

    def test_anchor_score_empty_metadata(self):
        """Empty metadata → score 0."""
        assert Ranker.anchor_score({}) == 0


# =============================================================================
# rerank_retrieved_chunks tests
# =============================================================================

class TestRerankRetrievedChunks:
    """Tests for Ranker.rerank_retrieved_chunks()."""

    @patch("src.engine.ranking.classify_query_intent", return_value="GENERAL")
    def test_rerank_returns_indices(self, mock_intent):
        """Returns list of reordered indices."""
        metadatas = [{"article": "1"}, {"article": "2"}]
        documents = ["Doc 1", "Doc 2"]
        distances = [0.5, 0.3]

        indices = Ranker.rerank_retrieved_chunks(
            metadatas=metadatas,
            documents=documents,
            distances=distances,
            question="Test question",
            user_profile=UserProfile.LEGAL,
        )

        assert len(indices) == 2
        assert set(indices) == {0, 1}

    @patch("src.engine.ranking.classify_query_intent", return_value="GENERAL")
    def test_rerank_lower_distance_ranks_higher(self, mock_intent):
        """Lower distance (more similar) ranks higher."""
        metadatas = [{"article": "1"}, {"article": "2"}]
        documents = ["Doc 1", "Doc 2"]
        distances = [0.9, 0.1]  # Second doc is much closer

        indices = Ranker.rerank_retrieved_chunks(
            metadatas=metadatas,
            documents=documents,
            distances=distances,
            question="Test question",
            user_profile=UserProfile.LEGAL,
        )

        # Second doc (index 1) should rank first due to lower distance
        assert indices[0] == 1

    @patch("src.engine.ranking.classify_query_intent", return_value="GENERAL")
    def test_rerank_mismatched_lengths(self, mock_intent):
        """Handles mismatched list lengths gracefully."""
        metadatas = [{"article": "1"}, {"article": "2"}, {"article": "3"}]
        documents = ["Doc 1", "Doc 2"]
        distances = [0.5, 0.3, 0.7, 0.2]  # More distances than docs

        indices = Ranker.rerank_retrieved_chunks(
            metadatas=metadatas,
            documents=documents,
            distances=distances,
            question="Test question",
            user_profile=UserProfile.LEGAL,
        )

        # Should only return 2 indices (min of all lengths)
        assert len(indices) == 2


# =============================================================================
# execute_ranking_pipeline tests
# =============================================================================

class TestExecuteRankingPipeline:
    """Tests for execute_ranking_pipeline()."""

    def test_pipeline_empty_hits(self):
        """Empty hits returns empty result."""
        result = execute_ranking_pipeline(
            hits=[],
            distances=[],
            retrieved_ids=[],
            retrieved_metas=[],
            question="Test",
        )

        assert result.ranked_hits == []
        assert result.ranked_distances == []
        assert result.ranked_ids == []
        assert result.ranked_metas == []

    def test_pipeline_preserves_order(self):
        """Pipeline preserves the input order."""
        hits = [("doc1", {"a": 1}), ("doc2", {"a": 2})]
        distances = [0.5, 0.3]
        ids = ["id1", "id2"]
        metas = [{"a": 1}, {"a": 2}]

        result = execute_ranking_pipeline(
            hits=hits,
            distances=distances,
            retrieved_ids=ids,
            retrieved_metas=metas,
            question="Test",
        )

        assert result.ranked_hits == hits
        assert result.ranked_distances == distances
        assert result.ranked_ids == ids
        assert result.ranked_metas == metas

    @patch("src.engine.ranking.classify_query_intent", return_value="GENERAL")
    def test_pipeline_with_user_profile_generates_debug(self, mock_intent):
        """With user_profile, debug_info is generated."""
        hits = [("doc1", {"article": "1"})]
        distances = [0.5]
        ids = ["id1"]
        metas = [{"article": "1"}]

        result = execute_ranking_pipeline(
            hits=hits,
            distances=distances,
            retrieved_ids=ids,
            retrieved_metas=metas,
            question="Test",
            user_profile=UserProfile.LEGAL,
        )

        assert result.debug_info is not None
        assert "query_intent" in result.debug_info
        assert "items" in result.debug_info
        assert len(result.debug_info["items"]) == 1

    def test_pipeline_without_user_profile_no_debug(self):
        """Without user_profile, no debug_info."""
        hits = [("doc1", {"article": "1"})]
        distances = [0.5]
        ids = ["id1"]
        metas = [{"article": "1"}]

        result = execute_ranking_pipeline(
            hits=hits,
            distances=distances,
            retrieved_ids=ids,
            retrieved_metas=metas,
            question="Test",
            user_profile=None,
        )

        assert result.debug_info is None

    def test_pipeline_truncates_to_min_length(self):
        """Pipeline truncates to minimum list length."""
        hits = [("doc1", {}), ("doc2", {}), ("doc3", {})]
        distances = [0.5, 0.3]  # Only 2
        ids = ["id1", "id2", "id3"]
        metas = [{"a": 1}, {"a": 2}]  # Only 2

        result = execute_ranking_pipeline(
            hits=hits,
            distances=distances,
            retrieved_ids=ids,
            retrieved_metas=metas,
            question="Test",
        )

        assert len(result.ranked_hits) == 2
        assert len(result.ranked_distances) == 2
        assert len(result.ranked_ids) == 2
        assert len(result.ranked_metas) == 2


# =============================================================================
# RankingPipelineResult tests
# =============================================================================

class TestRankingPipelineResult:
    """Tests for RankingPipelineResult dataclass."""

    def test_dataclass_creation(self):
        """Dataclass can be created with required fields."""
        result = RankingPipelineResult(
            ranked_hits=[("doc", {})],
            ranked_distances=[0.5],
            ranked_ids=["id1"],
            ranked_metas=[{}],
        )
        assert result.ranked_hits == [("doc", {})]
        assert result.debug_info is None

    def test_dataclass_with_debug_info(self):
        """Dataclass accepts optional debug_info."""
        debug = {"key": "value"}
        result = RankingPipelineResult(
            ranked_hits=[],
            ranked_distances=[],
            ranked_ids=[],
            ranked_metas=[],
            debug_info=debug,
        )
        assert result.debug_info == debug


# =============================================================================
# Original integration tests (unchanged)
# =============================================================================


def test_anchor_score_prefers_article_paragraph_over_chapter_and_annex():
    """Test that anchor_score assigns higher scores to more specific metadata."""
    metadatas = [
        {"source": "AI Act", "chapter": "I"},
        {"source": "AI Act", "annex": "I"},
        {"source": "AI Act", "article": "10", "paragraph": "2"},
    ]

    scores = [Ranker.anchor_score(m) for m in metadatas]

    # article+paragraph (4) > annex (1) > chapter only (0)
    assert scores[2] > scores[1] > scores[0]
    assert scores[0] == 0  # chapter only
    assert scores[1] == 1  # annex
    assert scores[2] == 4  # article+paragraph


def test_chapter_only_chunks_never_appear_in_references():
    engine = RAGEngine.__new__(RAGEngine)
    engine.top_k = 10
    engine.max_distance = None
    engine.ranking_weights = RankingWeights()

    def fake_query(question: str):  # noqa: ARG001
        return [
            ("Kapitel tekst", {"source": "AI Act", "chapter": "I"}),
            ("Artikel tekst", {"source": "AI Act", "article": "10", "paragraph": "2"}),
        ]

    def fake_call(prompt: str) -> str:  # noqa: ARG001
        return "Kravene fremgår af Artikel 10, stk. 2."

    engine.query = fake_query  # type: ignore[attr-defined]
    engine._call_openai = fake_call  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(engine, "Hvad er kravene?", user_profile=UserProfile.LEGAL)
    refs = list(payload.get("references") or [])

    assert len(refs) == 1
    assert refs[0].get("article") == "10"
    assert not refs[0].get("chapter")


def test_missing_ref_triggered_for_normative_claim_without_article_support():
    engine = RAGEngine.__new__(RAGEngine)
    engine.top_k = 10
    engine.max_distance = None
    engine.ranking_weights = RankingWeights()

    def fake_query(question: str):  # noqa: ARG001
        return [
            ("Recital tekst", {"source": "AI Act", "recital": "12"}),
        ]

    def fake_call(prompt: str) -> str:  # noqa: ARG001
        return "Systemet MUST implementere logging. (Betragtning 12)."

    engine.query = fake_query  # type: ignore[attr-defined]
    engine._call_openai = fake_call  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(engine, "Hvad skal vi gøre?", user_profile=UserProfile.ENGINEERING)

    assert str(payload.get("answer") or "") == "MISSING_REF"
    assert list(payload.get("reference_lines") or []) == []
    assert list(payload.get("references") or []) == []


def test_duplicate_recitals_removed_from_references():
    engine = RAGEngine.__new__(RAGEngine)
    engine.top_k = 10
    engine.max_distance = None
    engine.ranking_weights = RankingWeights()

    def fake_query(question: str):  # noqa: ARG001
        return [
            ("Recital tekst A", {"source": "AI Act", "recital": "12"}),
            ("Recital tekst B", {"source": "AI Act", "recital": "12"}),
        ]

    def fake_call(prompt: str) -> str:  # noqa: ARG001
        return "Dette følger af Betragtning 12."

    engine.query = fake_query  # type: ignore[attr-defined]
    engine._call_openai = fake_call  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(engine, "Hvad betyder det?", user_profile=UserProfile.LEGAL)
    refs = list(payload.get("references") or [])

    assert len(refs) == 1
    assert refs[0].get("recital") == "12"


