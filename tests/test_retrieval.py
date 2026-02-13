"""Tests for src/engine/retrieval.py - Core retrieval logic."""

import pytest
from unittest.mock import MagicMock, patch

from src.engine.retrieval import (
    apply_score_floor,
    get_initial_protected_hits,
    get_floor_config,
    distances_summary,
    Retriever,
    RetrievalPassInfo,
    RetrievalPassTracker,
    execute_multi_anchor_retrieval,
    MultiAnchorResult,
)


# -----------------------------------------------------------------------------
# Score floor tests
# -----------------------------------------------------------------------------


class TestApplyScoreFloor:
    """Tests for apply_score_floor function."""

    def test_empty_candidates_returns_empty(self):
        """Empty candidates list returns empty list."""
        result = apply_score_floor([], [], floor_threshold=0.85, floor_boost=0.5)
        assert result == []

    def test_empty_initial_hits_returns_unchanged(self):
        """When no initial hits, candidates are returned unchanged."""
        candidates = [
            {"chunk_id": "c1", "final_score": 0.3},
            {"chunk_id": "c2", "final_score": 0.5},
        ]
        result = apply_score_floor(candidates, [], floor_threshold=0.85, floor_boost=0.5)
        assert result == candidates

    def test_no_protected_hits_returns_unchanged(self):
        """When no hits below threshold, candidates are unchanged."""
        candidates = [
            {"chunk_id": "c1", "final_score": 0.3},
        ]
        initial_hits = [
            {"chunk_id": "c1", "distance": 0.9},  # Above threshold
        ]
        result = apply_score_floor(
            candidates, initial_hits, floor_threshold=0.85, floor_boost=0.5
        )
        assert result[0]["final_score"] == 0.3
        assert "score_floor_applied" not in result[0]

    def test_applies_floor_to_protected_hits(self):
        """Protected hits get floor score applied when below current score."""
        candidates = [
            {"chunk_id": "c1", "final_score": 0.01},  # Very low score
        ]
        initial_hits = [
            {"chunk_id": "c1", "distance": 0.5},  # Below threshold (high confidence)
        ]
        result = apply_score_floor(
            candidates, initial_hits, floor_threshold=0.85, floor_boost=0.5
        )
        # Floor score = 0.5 * (1.0 - 0.5) = 0.25
        assert result[0]["final_score"] == 0.25
        assert result[0]["score_floor_applied"] is True
        assert result[0]["original_score"] == 0.01

    def test_does_not_lower_score(self):
        """Floor only boosts, never lowers existing score."""
        candidates = [
            {"chunk_id": "c1", "final_score": 0.9},  # High score
        ]
        initial_hits = [
            {"chunk_id": "c1", "distance": 0.5},
        ]
        result = apply_score_floor(
            candidates, initial_hits, floor_threshold=0.85, floor_boost=0.5
        )
        # Score stays at 0.9 since floor would be 0.25
        assert result[0]["final_score"] == 0.9
        assert "score_floor_applied" not in result[0]

    def test_handles_id_key_variants(self):
        """Handles both 'chunk_id' and 'id' keys."""
        candidates = [{"id": "c1", "final_score": 0.01}]
        initial_hits = [{"id": "c1", "distance": 0.5}]
        result = apply_score_floor(
            candidates, initial_hits, floor_threshold=0.85, floor_boost=0.5
        )
        assert result[0]["score_floor_applied"] is True


class TestGetInitialProtectedHits:
    """Tests for get_initial_protected_hits function."""

    def test_empty_hits_returns_empty(self):
        """Empty hits list returns empty list."""
        result = get_initial_protected_hits([], top_n=3, threshold=0.85)
        assert result == []

    def test_filters_by_threshold(self):
        """Only hits with distance below threshold are protected."""
        hits = [
            (0.5, {"chunk_id": "c1"}),  # Below threshold
            (0.9, {"chunk_id": "c2"}),  # Above threshold
            (0.7, {"chunk_id": "c3"}),  # Below threshold
        ]
        result = get_initial_protected_hits(hits, top_n=10, threshold=0.85)
        assert len(result) == 2
        assert result[0]["chunk_id"] == "c1"
        assert result[1]["chunk_id"] == "c3"

    def test_respects_top_n(self):
        """Only considers first top_n hits."""
        hits = [
            (0.5, {"chunk_id": "c1"}),
            (0.6, {"chunk_id": "c2"}),
            (0.4, {"chunk_id": "c3"}),  # Would be protected but beyond top_n=2
        ]
        result = get_initial_protected_hits(hits, top_n=2, threshold=0.85)
        assert len(result) == 2
        chunk_ids = [r["chunk_id"] for r in result]
        assert "c3" not in chunk_ids

    def test_includes_initial_rank(self):
        """Results include initial_rank field."""
        hits = [(0.5, {"chunk_id": "c1"})]
        result = get_initial_protected_hits(hits, top_n=3, threshold=0.85)
        assert result[0]["initial_rank"] == 0

    def test_skips_hits_without_chunk_id(self):
        """Hits without chunk_id are skipped."""
        hits = [
            (0.5, {}),  # No chunk_id
            (0.6, {"chunk_id": "c2"}),
        ]
        result = get_initial_protected_hits(hits, top_n=10, threshold=0.85)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c2"


class TestGetFloorConfig:
    """Tests for get_floor_config function."""

    @patch("src.engine.retrieval._load_fusion_config")
    def test_returns_config_values(self, mock_load):
        """Returns threshold and boost from config."""
        mock_load.return_value = {
            "citation_expansion": {
                "floor_threshold": 0.75,
                "floor_boost": 0.6,
            }
        }
        threshold, boost = get_floor_config()
        assert threshold == 0.75
        assert boost == 0.6

    @patch("src.engine.retrieval._load_fusion_config")
    def test_returns_defaults_when_missing(self, mock_load):
        """Returns default values when config is empty."""
        mock_load.return_value = {}
        threshold, boost = get_floor_config()
        assert threshold == 0.85
        assert boost == 0.5


# -----------------------------------------------------------------------------
# Distances summary tests
# -----------------------------------------------------------------------------


class TestDistancesSummary:
    """Tests for distances_summary function."""

    def test_empty_list_returns_zero_count(self):
        """Empty list returns count=0, best_distance=None."""
        result = distances_summary([])
        assert result["count"] == 0
        assert result["best_distance"] is None

    def test_computes_min_max(self):
        """Computes correct min and max distances."""
        result = distances_summary([0.3, 0.5, 0.1, 0.8])
        assert result["count"] == 4
        assert result["best_distance"] == 0.1
        assert result["worst_distance"] == 0.8

    def test_single_element(self):
        """Single element has same best and worst."""
        result = distances_summary([0.5])
        assert result["count"] == 1
        assert result["best_distance"] == 0.5
        assert result["worst_distance"] == 0.5


# -----------------------------------------------------------------------------
# Retriever tests
# -----------------------------------------------------------------------------


class TestRetrieverNormalizeChromaWhere:
    """Tests for Retriever._normalize_chroma_where static method."""

    def test_none_returns_none(self):
        """None input returns None."""
        result = Retriever._normalize_chroma_where(None)
        assert result is None

    def test_empty_dict_returns_none(self):
        """Empty dict returns None."""
        result = Retriever._normalize_chroma_where({})
        assert result is None

    def test_single_key_unchanged(self):
        """Single key dict is unchanged."""
        where = {"article": "6"}
        result = Retriever._normalize_chroma_where(where)
        assert result == {"article": "6"}

    def test_multi_key_wrapped_in_and(self):
        """Multiple keys wrapped in $and."""
        where = {"corpus_id": "ai-act", "article": "6"}
        result = Retriever._normalize_chroma_where(where)
        assert "$and" in result
        # Keys sorted alphabetically
        assert {"article": "6"} in result["$and"]
        assert {"corpus_id": "ai-act"} in result["$and"]

    def test_existing_operator_preserved(self):
        """Existing $or operator is preserved."""
        where = {"$or": [{"article": "2"}, {"article": "3"}]}
        result = Retriever._normalize_chroma_where(where)
        assert result == where

    def test_mixed_keys_and_operators(self):
        """Mix of regular keys and operators handled."""
        where = {"corpus_id": "ai-act", "$or": [{"article": "2"}, {"article": "3"}]}
        result = Retriever._normalize_chroma_where(where)
        assert "$and" in result
        # Should contain corpus_id clause and the $or clause
        clauses = result["$and"]
        assert {"corpus_id": "ai-act"} in clauses

    def test_nested_or_items_normalized(self):
        """Items inside $or are also normalized."""
        where = {"$or": [{"chapter": "III", "section": "1"}, {"article": "6"}]}
        result = Retriever._normalize_chroma_where(where)
        # First item should be wrapped in $and
        assert "$or" in result
        first_item = result["$or"][0]
        assert "$and" in first_item or len(first_item) == 1


class TestRetrieverSplitPrecise:
    """Tests for Retriever.split_precise static method."""

    def test_separates_by_citability(self):
        """Separates hits by presence of article/annex/chapter."""
        hits = [
            ("doc1", {"article": "6"}),  # Precise
            ("doc2", {}),  # Imprecise
            ("doc3", {"annex": "III"}),  # Precise
        ]
        dists = [0.1, 0.2, 0.3]
        precise, precise_d, precise_ids, imprecise, imprecise_d, imprecise_ids = (
            Retriever.split_precise(hits, dists)
        )
        assert len(precise) == 2
        assert len(imprecise) == 1
        assert precise[0][1]["article"] == "6"
        assert imprecise[0][0] == "doc2"

    def test_handles_empty_hits(self):
        """Empty hits returns all empty lists."""
        precise, precise_d, precise_ids, imprecise, imprecise_d, imprecise_ids = (
            Retriever.split_precise([], [])
        )
        assert precise == []
        assert imprecise == []

    def test_chapter_is_precise(self):
        """Chapter metadata makes hit precise."""
        hits = [("doc1", {"chapter": "IV"})]
        dists = [0.1]
        precise, _, _, imprecise, _, _ = Retriever.split_precise(hits, dists)
        assert len(precise) == 1
        assert len(imprecise) == 0

    def test_uses_provided_ids(self):
        """Uses provided IDs when available."""
        hits = [("doc1", {"article": "6"})]
        dists = [0.1]
        ids = ["custom-id-1"]
        precise, _, precise_ids, _, _, _ = Retriever.split_precise(hits, dists, ids)
        assert precise_ids[0] == "custom-id-1"


# -----------------------------------------------------------------------------
# Retrieval pass tracking tests
# -----------------------------------------------------------------------------


class TestRetrievalPassInfo:
    """Tests for RetrievalPassInfo dataclass."""

    def test_creates_with_required_fields(self):
        """Creates instance with required fields."""
        info = RetrievalPassInfo(
            pass_name="initial",
            planned_where={"article": "6"},
            planned_collection_type="chunk",
        )
        assert info.pass_name == "initial"
        assert info.planned_where == {"article": "6"}
        assert info.planned_collection_type == "chunk"

    def test_optional_fields_have_defaults(self):
        """Optional fields have sensible defaults."""
        info = RetrievalPassInfo(
            pass_name="test",
            planned_where=None,
            planned_collection_type="toc",
        )
        assert info.effective_where is None
        assert info.retrieved_ids == []
        assert info.distances_summary == {}


class TestRetrievalPassTracker:
    """Tests for RetrievalPassTracker class."""

    def test_records_pass(self):
        """Records a retrieval pass."""
        mock_retriever = MagicMock()
        mock_retriever._last_effective_where = {"article": "6"}
        mock_retriever._last_effective_collection_name = "test_collection"
        mock_retriever._last_effective_collection_type = "Collection"
        mock_retriever._last_retrieved_ids = ["id1", "id2"]
        mock_retriever._last_distances = [0.1, 0.2]

        tracker = RetrievalPassTracker(mock_retriever)
        tracker.record_pass(
            pass_name="initial",
            planned_where={"corpus_id": "ai-act"},
            planned_collection_type="chunk",
        )

        passes = tracker.get_passes()
        assert len(passes) == 1
        assert passes[0]["pass_name"] == "initial"
        assert passes[0]["planned_where"] == {"corpus_id": "ai-act"}
        assert passes[0]["effective_where"] == {"article": "6"}

    def test_get_passes_returns_list(self):
        """get_passes returns list of recorded passes."""
        mock_retriever = MagicMock()
        tracker = RetrievalPassTracker(mock_retriever)
        assert tracker.get_passes() == []


# -----------------------------------------------------------------------------
# Multi-anchor retrieval tests
# -----------------------------------------------------------------------------


class TestExecuteMultiAnchorRetrieval:
    """Tests for execute_multi_anchor_retrieval function."""

    def test_empty_refs_returns_empty_result(self):
        """Empty article/annex refs returns empty result."""
        result = execute_multi_anchor_retrieval(
            query_fn=MagicMock(),
            question="test",
            corpus_id="ai-act",
            explicit_article_refs=[],
            explicit_annex_refs=[],
            top_k=10,
        )
        assert isinstance(result, MultiAnchorResult)
        assert result.hits == []
        assert result.distances == []

    def test_merges_results_from_multiple_scopes(self):
        """Merges results from multiple article scopes."""

        def mock_query_fn(question, k, where):
            article = where.get("article", "")
            if article == "6":
                return (
                    [("doc6", {"chunk_id": "c6", "article": "6"})],
                    [0.1],
                    ["c6"],
                    [{"chunk_id": "c6", "article": "6"}],
                )
            elif article == "7":
                return (
                    [("doc7", {"chunk_id": "c7", "article": "7"})],
                    [0.2],
                    ["c7"],
                    [{"chunk_id": "c7", "article": "7"}],
                )
            return ([], [], [], [])

        result = execute_multi_anchor_retrieval(
            query_fn=mock_query_fn,
            question="test",
            corpus_id="ai-act",
            explicit_article_refs=["6", "7"],
            explicit_annex_refs=[],
            top_k=10,
        )
        assert len(result.hits) == 2
        chunk_ids = result.retrieved_ids
        assert "c6" in chunk_ids
        assert "c7" in chunk_ids

    def test_deduplicates_by_chunk_id(self):
        """Same chunk_id from different scopes is deduplicated."""

        def mock_query_fn(question, k, where):
            # Both scopes return same chunk
            return (
                [("doc1", {"chunk_id": "shared", "article": where.get("article")})],
                [0.1],
                ["shared"],
                [{"chunk_id": "shared"}],
            )

        result = execute_multi_anchor_retrieval(
            query_fn=mock_query_fn,
            question="test",
            corpus_id="ai-act",
            explicit_article_refs=["6", "7"],
            explicit_annex_refs=[],
            top_k=10,
        )
        # Should be deduplicated to just 1 hit
        assert len(result.hits) == 1
        assert result.retrieved_ids == ["shared"]

    def test_respects_top_k(self):
        """Result respects top_k limit."""

        def mock_query_fn(question, k, where):
            return (
                [
                    ("doc1", {"chunk_id": f"c{i}", "article": "6"})
                    for i in range(5)
                ],
                [0.1 * i for i in range(5)],
                [f"c{i}" for i in range(5)],
                [{"chunk_id": f"c{i}"} for i in range(5)],
            )

        result = execute_multi_anchor_retrieval(
            query_fn=mock_query_fn,
            question="test",
            corpus_id="ai-act",
            explicit_article_refs=["6"],
            explicit_annex_refs=[],
            top_k=3,
        )
        assert len(result.hits) <= 3
