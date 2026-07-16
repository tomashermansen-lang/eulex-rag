"""Tests for src/engine/streaming.py — streaming preparation stage."""

import pytest
from unittest.mock import MagicMock

from src.engine.streaming import prepare_for_streaming_stage


class TestPrepareForStreamingStage:
    """Tests for the module-level prepare_for_streaming_stage function."""

    def test_raises_on_empty_question(self):
        with pytest.raises(ValueError, match="empty"):
            prepare_for_streaming_stage(
                question="   ",
                corpus_id="ai-act",
                top_k=3,
                user_profile="LEGAL",
                query_fn=MagicMock(),
                collection_name="test_docs",
            )

    def test_returns_structured_result(self):
        """Basic path: retrieves hits and builds prompt."""
        hits = [
            ("Chunk about article 6", {"article": "6", "corpus_id": "ai-act"}),
            ("Chunk about article 7", {"article": "7", "corpus_id": "ai-act"}),
        ]
        distances = [0.3, 0.5]
        query_fn = MagicMock(return_value=(hits, distances))

        result = prepare_for_streaming_stage(
            question="Hvad kræver artikel 6?",
            corpus_id="ai-act",
            top_k=3,
            user_profile="LEGAL",
            query_fn=query_fn,
            collection_name="ai-act_documents",
        )

        assert "prompt" in result
        assert "references_structured" in result
        assert "reference_lines" in result
        assert "retrieval" in result
        assert len(result["references_structured"]) == 2
        assert result["retrieval"]["distances"] == [0.3, 0.5]

    def test_references_contain_article_labels(self):
        hits = [
            ("Chunk text", {"article": "6", "corpus_id": "ai-act"}),
        ]
        distances = [0.2]
        query_fn = MagicMock(return_value=(hits, distances))

        result = prepare_for_streaming_stage(
            question="Hvad kræver artikel 6?",
            corpus_id="ai-act",
            top_k=3,
            user_profile="LEGAL",
            query_fn=query_fn,
            collection_name="test_docs",
        )

        assert result["reference_lines"][0].startswith("[1]")
        assert "Artikel 6" in result["reference_lines"][0]

    def test_no_hits_returns_empty_references(self):
        query_fn = MagicMock(return_value=([], []))

        result = prepare_for_streaming_stage(
            question="Hvad kræver artikel 99?",
            corpus_id="ai-act",
            top_k=3,
            user_profile="LEGAL",
            query_fn=query_fn,
            collection_name="test_docs",
        )

        assert result["references_structured"] == []
        assert result["reference_lines"] == []
