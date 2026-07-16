"""Tests for src/engine/summary_generation.py — chapter summary stage."""

from unittest.mock import MagicMock, patch

from src.engine.summary_generation import answer_chapter_summary_stage


class TestAnswerChapterSummaryStage:
    """Tests for the module-level answer_chapter_summary_stage function."""

    def test_returns_none_for_non_chapter_question(self):
        result = answer_chapter_summary_stage(
            question="Hvad er artikel 6?",
            top_k=3,
            query_fn=MagicMock(),
            client=MagicMock(),
            chat_model="gpt-4",
            collection_name="test_docs",
            source_label_fn=lambda m: "test-source",
            hybrid_rerank_dict={"enabled": False},
            retrieved_ids_fn=lambda: [],
            retrieved_metadatas_fn=lambda: [],
        )
        assert result is None

    def test_returns_none_when_no_chapter_ref(self):
        """Question looks like chapter summary but has no chapter number."""
        with patch(
            "src.engine.query_helpers._looks_like_chapter_summary_question",
            return_value=True,
        ):
            with patch(
                "src.engine.query_helpers._extract_chapter_ref", return_value=None
            ):
                result = answer_chapter_summary_stage(
                    question="opsummér kapitlet",
                    top_k=3,
                    query_fn=MagicMock(),
                    client=MagicMock(),
                    chat_model="gpt-4",
                    collection_name="test_docs",
                    source_label_fn=lambda m: "test-source",
                    hybrid_rerank_dict={"enabled": False},
                    retrieved_ids_fn=lambda: [],
                    retrieved_metadatas_fn=lambda: [],
                )
        assert result is None

    def test_returns_none_when_no_hits(self):
        query_fn = MagicMock(return_value=([], []))
        with patch(
            "src.engine.query_helpers._looks_like_chapter_summary_question",
            return_value=True,
        ):
            with patch(
                "src.engine.query_helpers._extract_chapter_ref", return_value="III"
            ):
                result = answer_chapter_summary_stage(
                    question="Opsummér kapitel III",
                    top_k=3,
                    query_fn=query_fn,
                    client=MagicMock(),
                    chat_model="gpt-4",
                    collection_name="test_docs",
                    source_label_fn=lambda m: "test-source",
                    hybrid_rerank_dict={"enabled": False},
                    retrieved_ids_fn=lambda: [],
                    retrieved_metadatas_fn=lambda: [],
                )
        assert result is None

    def test_returns_structured_result_for_valid_chapter_question(self):
        """Full path: detects chapter question, retrieves hits, generates summary."""
        hits = [
            (
                "Chunk about chapter III requirements",
                {"article": "6", "corpus_id": "ai-act", "source": "test"},
            ),
            (
                "Another chunk about obligations",
                {"article": "7", "corpus_id": "ai-act", "source": "test"},
            ),
        ]
        distances = [0.3, 0.5]
        query_fn = MagicMock(return_value=(hits, distances))

        # Mock OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Summary of chapter III"))
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch(
            "src.engine.query_helpers._looks_like_chapter_summary_question",
            return_value=True,
        ):
            with patch(
                "src.engine.query_helpers._extract_chapter_ref", return_value="III"
            ):
                result = answer_chapter_summary_stage(
                    question="Opsummér kapitel III",
                    top_k=3,
                    query_fn=query_fn,
                    client=mock_client,
                    chat_model="gpt-4",
                    collection_name="test_docs",
                    source_label_fn=lambda m: "ai-act",
                    hybrid_rerank_dict={"enabled": False},
                    retrieved_ids_fn=lambda: ["id1", "id2"],
                    retrieved_metadatas_fn=lambda: [{"article": "6"}, {"article": "7"}],
                )

        assert result is not None
        assert "answer" in result
        assert result["answer"] == "Summary of chapter III"
        assert "references" in result
        assert "retrieval" in result
        assert result["retrieval"]["query_where"] == {"chapter": "III"}
        assert result["retrieval"]["distances"] == [0.3, 0.5]

    def test_state_updates_returned(self):
        """Verify the state update fields are present in retrieval dict."""
        hits = [
            ("Chunk text", {"article": "6", "corpus_id": "ai-act", "source": "s"}),
        ]
        distances = [0.2]
        query_fn = MagicMock(return_value=(hits, distances))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Summary"))]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch(
            "src.engine.query_helpers._looks_like_chapter_summary_question",
            return_value=True,
        ):
            with patch(
                "src.engine.query_helpers._extract_chapter_ref", return_value="5"
            ):
                result = answer_chapter_summary_stage(
                    question="Opsummér kapitel 5",
                    top_k=3,
                    query_fn=query_fn,
                    client=mock_client,
                    chat_model="gpt-4",
                    collection_name="my_collection",
                    source_label_fn=lambda m: "source",
                    hybrid_rerank_dict={"enabled": True},
                    retrieved_ids_fn=lambda: ["a"],
                    retrieved_metadatas_fn=lambda: [{}],
                )

        assert result is not None
        assert result["retrieval"]["query_collection"] == "my_collection"
        assert result["retrieval"]["query_where"] == {"chapter": "5"}
        assert result["retrieval"]["distances"] == [0.2]
        assert result["retrieval"]["hybrid_rerank"] == {"enabled": True}
