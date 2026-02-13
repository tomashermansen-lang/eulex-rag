"""Tests for sibling chunk expansion in retrieval."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.engine.retrieval import Retriever


class TestSiblingExpansion:
    """Tests for _expand_to_siblings method."""

    def _make_retriever(self) -> Retriever:
        """Create a Retriever with a mock collection."""
        mock_collection = MagicMock()
        return Retriever(collection=mock_collection, embedding_model="text-embedding-3-large")

    def test_expand_to_siblings_empty_input(self):
        """Should return empty results for empty input."""
        retriever = self._make_retriever()
        mock_collection = MagicMock()

        ids, docs, metas, dists = retriever._expand_to_siblings(
            collection=mock_collection,
            ids=[],
            documents=[],
            metadatas=[],
            distances=[],
            max_siblings=2,
        )

        assert ids == []
        assert docs == []
        assert metas == []
        assert dists == []

    def test_expand_to_siblings_no_location_id(self):
        """Should return original results if no location_id in metadata."""
        retriever = self._make_retriever()
        mock_collection = MagicMock()

        original_ids = ["chunk-1", "chunk-2"]
        original_docs = ["doc 1", "doc 2"]
        original_metas = [{"article": "1"}, {"article": "2"}]  # No location_id
        original_dists = [0.5, 0.6]

        ids, docs, metas, dists = retriever._expand_to_siblings(
            collection=mock_collection,
            ids=original_ids,
            documents=original_docs,
            metadatas=original_metas,
            distances=original_dists,
            max_siblings=2,
        )

        assert ids == original_ids
        assert docs == original_docs
        assert metas == original_metas
        assert dists == original_dists

    def test_expand_to_siblings_with_siblings_found(self):
        """Should expand with sibling chunks when found."""
        retriever = self._make_retriever()
        mock_collection = MagicMock()

        # Original retrieved chunk
        original_ids = ["chunk-1"]
        original_docs = ["First chunk of article 5"]
        original_metas = [{"article": "5", "location_id": "loc:v1/article:5", "chunk_index": 0}]
        original_dists = [0.5]

        # Mock collection.get to return siblings
        mock_collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2", "chunk-3"],
            "documents": ["First chunk of article 5", "Second chunk of article 5", "Third chunk of article 5"],
            "metadatas": [
                {"article": "5", "location_id": "loc:v1/article:5", "chunk_index": 0},
                {"article": "5", "location_id": "loc:v1/article:5", "chunk_index": 1},
                {"article": "5", "location_id": "loc:v1/article:5", "chunk_index": 2},
            ],
        }

        ids, docs, metas, dists = retriever._expand_to_siblings(
            collection=mock_collection,
            ids=original_ids,
            documents=original_docs,
            metadatas=original_metas,
            distances=original_dists,
            max_siblings=2,
        )

        # Should include original + 2 siblings
        assert len(ids) == 3
        assert "chunk-1" in ids
        assert "chunk-2" in ids
        assert "chunk-3" in ids

        # Siblings should be inserted after original
        assert ids[0] == "chunk-1"

        # Sibling distances should be slightly higher (5% penalty)
        assert dists[0] == 0.5  # Original
        assert dists[1] == pytest.approx(0.5 * 1.05)  # Sibling

    def test_expand_to_siblings_respects_max_siblings(self):
        """Should respect max_siblings limit."""
        retriever = self._make_retriever()
        mock_collection = MagicMock()

        original_ids = ["chunk-1"]
        original_docs = ["Content"]
        original_metas = [{"location_id": "loc:v1/article:10", "chunk_index": 0}]
        original_dists = [0.5]

        # Mock returns 5 sibling chunks
        mock_collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"],
            "documents": ["c1", "c2", "c3", "c4", "c5"],
            "metadatas": [
                {"location_id": "loc:v1/article:10", "chunk_index": i} for i in range(5)
            ],
        }

        ids, docs, metas, dists = retriever._expand_to_siblings(
            collection=mock_collection,
            ids=original_ids,
            documents=original_docs,
            metadatas=original_metas,
            distances=original_dists,
            max_siblings=2,  # Only allow 2 siblings
        )

        # Original + max 2 siblings = 3 total
        assert len(ids) == 3

    def test_expand_to_siblings_no_duplicates(self):
        """Should not add chunks that are already in retrieved set."""
        retriever = self._make_retriever()
        mock_collection = MagicMock()

        # Two chunks from same location already retrieved
        original_ids = ["chunk-1", "chunk-2"]
        original_docs = ["Content 1", "Content 2"]
        original_metas = [
            {"location_id": "loc:v1/article:5", "chunk_index": 0},
            {"location_id": "loc:v1/article:5", "chunk_index": 1},
        ]
        original_dists = [0.5, 0.55]

        # Mock returns the same chunks (already retrieved)
        mock_collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2", "chunk-3"],
            "documents": ["c1", "c2", "c3"],
            "metadatas": [
                {"location_id": "loc:v1/article:5", "chunk_index": 0},
                {"location_id": "loc:v1/article:5", "chunk_index": 1},
                {"location_id": "loc:v1/article:5", "chunk_index": 2},
            ],
        }

        ids, docs, metas, dists = retriever._expand_to_siblings(
            collection=mock_collection,
            ids=original_ids,
            documents=original_docs,
            metadatas=original_metas,
            distances=original_dists,
            max_siblings=2,
        )

        # Should only add chunk-3 (chunk-1 and chunk-2 already exist)
        assert len(ids) == 3
        assert ids.count("chunk-1") == 1
        assert ids.count("chunk-2") == 1
        assert ids.count("chunk-3") == 1

    def test_expand_to_siblings_multiple_locations(self):
        """Should expand siblings for each unique location_id."""
        retriever = self._make_retriever()
        mock_collection = MagicMock()

        # Two chunks from different locations
        original_ids = ["art5-chunk", "art10-chunk"]
        original_docs = ["Article 5 content", "Article 10 content"]
        original_metas = [
            {"location_id": "loc:v1/article:5", "chunk_index": 0},
            {"location_id": "loc:v1/article:10", "chunk_index": 0},
        ]
        original_dists = [0.5, 0.6]

        def mock_get(where, include, limit):
            loc = where.get("location_id", "")
            if "article:5" in loc:
                return {
                    "ids": ["art5-chunk", "art5-sibling"],
                    "documents": ["Article 5 content", "Article 5 sibling"],
                    "metadatas": [
                        {"location_id": "loc:v1/article:5", "chunk_index": 0},
                        {"location_id": "loc:v1/article:5", "chunk_index": 1},
                    ],
                }
            else:
                return {
                    "ids": ["art10-chunk", "art10-sibling"],
                    "documents": ["Article 10 content", "Article 10 sibling"],
                    "metadatas": [
                        {"location_id": "loc:v1/article:10", "chunk_index": 0},
                        {"location_id": "loc:v1/article:10", "chunk_index": 1},
                    ],
                }

        mock_collection.get.side_effect = mock_get

        ids, docs, metas, dists = retriever._expand_to_siblings(
            collection=mock_collection,
            ids=original_ids,
            documents=original_docs,
            metadatas=original_metas,
            distances=original_dists,
            max_siblings=1,
        )

        # 2 original + 1 sibling each = 4 total
        assert len(ids) == 4
        assert "art5-sibling" in ids
        assert "art10-sibling" in ids

    def test_expand_to_siblings_orders_by_chunk_index(self):
        """Should order siblings by chunk_index."""
        retriever = self._make_retriever()
        mock_collection = MagicMock()

        original_ids = ["chunk-middle"]
        original_docs = ["Middle chunk"]
        original_metas = [{"location_id": "loc:v1/article:1", "chunk_index": 2}]
        original_dists = [0.5]

        # Return siblings in random order
        mock_collection.get.return_value = {
            "ids": ["chunk-last", "chunk-middle", "chunk-first"],
            "documents": ["Last", "Middle", "First"],
            "metadatas": [
                {"location_id": "loc:v1/article:1", "chunk_index": 4},
                {"location_id": "loc:v1/article:1", "chunk_index": 2},
                {"location_id": "loc:v1/article:1", "chunk_index": 0},
            ],
        }

        ids, docs, metas, dists = retriever._expand_to_siblings(
            collection=mock_collection,
            ids=original_ids,
            documents=original_docs,
            metadatas=original_metas,
            distances=original_dists,
            max_siblings=2,
        )

        # Original first, then siblings sorted by chunk_index
        assert ids[0] == "chunk-middle"
        # Siblings should be chunk-first (idx 0) and chunk-last (idx 4), sorted
        sibling_ids = ids[1:]
        assert "chunk-first" in sibling_ids
        # chunk-first should come before chunk-last (lower chunk_index)
        assert sibling_ids.index("chunk-first") < sibling_ids.index("chunk-last")


class TestSiblingExpansionIntegration:
    """Integration tests for sibling expansion with config."""

    @patch("src.engine.retrieval.get_sibling_expansion_settings")
    def test_query_collection_with_distances_calls_expansion_when_enabled(self, mock_settings):
        """Should call _expand_to_siblings when enabled in config."""
        mock_settings.return_value = {"enabled": True, "max_siblings": 2}

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Content"]],
            "metadatas": [[{"location_id": "loc:v1/article:1", "chunk_index": 0}]],
            "distances": [[0.5]],
        }
        mock_collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["Content", "Sibling"],
            "metadatas": [
                {"location_id": "loc:v1/article:1", "chunk_index": 0},
                {"location_id": "loc:v1/article:1", "chunk_index": 1},
            ],
        }

        retriever = Retriever(collection=mock_collection, embedding_model="text-embedding-3-large")

        with patch.object(retriever, "_embed", return_value=[[0.1] * 1536]):
            hits, distances = retriever._query_collection_with_distances(
                collection=mock_collection,
                question="test query",
                k=5,
            )

        # Should have expanded
        assert len(hits) == 2
        assert len(distances) == 2

    @patch("src.engine.retrieval.get_sibling_expansion_settings")
    def test_query_collection_skips_expansion_when_disabled(self, mock_settings):
        """Should not expand when disabled in config."""
        mock_settings.return_value = {"enabled": False, "max_siblings": 2}

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Content"]],
            "metadatas": [[{"location_id": "loc:v1/article:1"}]],
            "distances": [[0.5]],
        }

        retriever = Retriever(collection=mock_collection, embedding_model="text-embedding-3-large")

        with patch.object(retriever, "_embed", return_value=[[0.1] * 1536]):
            hits, distances = retriever._query_collection_with_distances(
                collection=mock_collection,
                question="test query",
                k=5,
            )

        # Should not expand - only original chunk
        assert len(hits) == 1
        # collection.get should not have been called for sibling expansion
        mock_collection.get.assert_not_called()

    @patch("src.engine.retrieval.get_sibling_expansion_settings")
    def test_explicit_expand_siblings_override(self, mock_settings):
        """Should respect explicit expand_siblings parameter over config."""
        mock_settings.return_value = {"enabled": False, "max_siblings": 2}

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Content"]],
            "metadatas": [[{"location_id": "loc:v1/article:1", "chunk_index": 0}]],
            "distances": [[0.5]],
        }
        mock_collection.get.return_value = {
            "ids": ["chunk-1", "chunk-2"],
            "documents": ["Content", "Sibling"],
            "metadatas": [
                {"location_id": "loc:v1/article:1", "chunk_index": 0},
                {"location_id": "loc:v1/article:1", "chunk_index": 1},
            ],
        }

        retriever = Retriever(collection=mock_collection, embedding_model="text-embedding-3-large")

        with patch.object(retriever, "_embed", return_value=[[0.1] * 1536]):
            # Explicitly enable expansion even though config says disabled
            hits, distances = retriever._query_collection_with_distances(
                collection=mock_collection,
                question="test query",
                k=5,
                expand_siblings=True,  # Override config
            )

        # Should have expanded despite config being disabled
        assert len(hits) == 2
