"""Tests for src/engine/indexing.py - Document enrichment and ingestion."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from src.engine.indexing import (
    _enrich_document_for_embedding,
)


# ---------------------------------------------------------------------------
# Test: _enrich_document_for_embedding
# ---------------------------------------------------------------------------


class TestEnrichDocumentForEmbedding:
    @pytest.mark.slow
    def test_adds_heading_path(self):
        doc = "Original content"
        meta = {"heading_path_display": "Chapter V > Article 53"}

        result = _enrich_document_for_embedding(doc, meta)

        assert "Chapter V > Article 53" in result
        assert "Original content" in result

    @pytest.mark.slow
    def test_adds_contextual_description(self):
        doc = "Original content"
        meta = {"contextual_description": "This describes AI system requirements"}

        result = _enrich_document_for_embedding(doc, meta)

        assert "[Kontekst: This describes AI system requirements]" in result
        assert "Original content" in result

    def test_adds_enrichment_terms(self):
        doc = "Original content"
        meta = {"enrichment_terms": ["musik AI", "generere billeder"]}

        result = _enrich_document_for_embedding(doc, meta)

        assert "[Søgetermer: musik AI | generere billeder]" in result
        assert "Original content" in result

    def test_combines_all_enrichments(self):
        doc = "Original content"
        meta = {
            "heading_path_display": "Chapter V",
            "contextual_description": "Context here",
            "enrichment_terms": ["term1", "term2"],
        }

        result = _enrich_document_for_embedding(doc, meta)

        assert "[Søgetermer: term1 | term2]" in result
        assert "[Kontekst: Context here]" in result
        assert "Chapter V" in result
        assert "Original content" in result

    def test_handles_empty_enrichments(self):
        doc = "Just content"
        meta = {}

        # Disable LLM enrichment to test pure metadata path
        with patch("src.ingestion.embedding_enrichment.is_enrichment_enabled", return_value=False):
            result = _enrich_document_for_embedding(doc, meta)

        # Should return original doc if no metadata enrichments and LLM disabled
        assert result == "Just content"

    @pytest.mark.slow
    def test_infers_title_for_annex_points(self):
        doc = "Systemer anvendt på følgende områder\nDetail content here"
        meta = {"annex_point": "1"}  # No annex_point_title

        result = _enrich_document_for_embedding(doc, meta)

        # First line should be used as title hint
        assert "Systemer anvendt på følgende områder" in result

    @pytest.mark.slow
    def test_skips_long_first_lines_for_title(self):
        long_first_line = "a) " + "x" * 150  # Starts with a) and is too long
        doc = f"{long_first_line}\nActual content"
        meta = {"annex_point": "1"}

        result = _enrich_document_for_embedding(doc, meta)

        # Should not use the long first line as title
        assert long_first_line not in result.split("\n\n")[0]


# ---------------------------------------------------------------------------
# Test: Metadata stringification (tested indirectly via index_jsonl)
# ---------------------------------------------------------------------------


class TestMetadataStringification:
    def test_enrichment_terms_converted_to_string(self):
        """List values like enrichment_terms should be joined for ChromaDB."""
        from src.engine.indexing import index_jsonl

        with TemporaryDirectory() as tmpdir:
            mock_engine = MagicMock()
            mock_engine.collection = MagicMock()
            mock_engine._embed = MagicMock(return_value=[[0.1] * 10])
            mock_engine.corpus_id = "test"

            jsonl_path = Path(tmpdir) / "test.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({
                    "text": "Content",
                    "metadata": {
                        "chunk_id": "chunk-1",
                        "enrichment_terms": ["term1", "term2", "term3"]
                    }
                }) + "\n")

            index_jsonl(mock_engine, str(jsonl_path))

            call_args = mock_engine.collection.upsert.call_args
            metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")
            # Should be converted to pipe-separated string
            assert metadatas[0].get("enrichment_terms") == "term1 | term2 | term3"


# ---------------------------------------------------------------------------
# Test: index_jsonl behavior
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIndexJsonl:
    def test_skips_toc_entries(self):
        """TOC entries (doc_type == 'toc') should be skipped."""
        from src.engine.indexing import index_jsonl

        with TemporaryDirectory() as tmpdir:
            # Create a mock engine
            mock_engine = MagicMock()
            mock_engine.collection = MagicMock()
            mock_engine._embed = MagicMock(return_value=[[0.1] * 10])
            mock_engine.corpus_id = "test"

            # Create JSONL with TOC and regular entries
            jsonl_path = Path(tmpdir) / "test.jsonl"
            with open(jsonl_path, "w") as f:
                # TOC entry - should be skipped
                f.write(json.dumps({
                    "text": "Table of Contents",
                    "metadata": {"doc_type": "toc", "chunk_id": "toc-1"}
                }) + "\n")
                # Regular entry - should be indexed
                f.write(json.dumps({
                    "text": "Regular content",
                    "metadata": {"chunk_id": "chunk-1"}
                }) + "\n")

            index_jsonl(mock_engine, str(jsonl_path))

            # Verify upsert was called
            mock_engine.collection.upsert.assert_called()

            # Check that only the regular entry was indexed
            call_args = mock_engine.collection.upsert.call_args
            ids = call_args.kwargs.get("ids") or call_args[1].get("ids")
            assert "chunk-1" in ids
            assert "toc-1" not in ids

    def test_stamps_corpus_id_if_missing(self):
        """Should add corpus_id to metadata if missing."""
        from src.engine.indexing import index_jsonl

        with TemporaryDirectory() as tmpdir:
            mock_engine = MagicMock()
            mock_engine.collection = MagicMock()
            mock_engine._embed = MagicMock(return_value=[[0.1] * 10])
            mock_engine.corpus_id = "ai-act"

            jsonl_path = Path(tmpdir) / "test.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({
                    "text": "Content without corpus_id",
                    "metadata": {"chunk_id": "chunk-1"}
                }) + "\n")

            index_jsonl(mock_engine, str(jsonl_path))

            # Check metadata includes corpus_id
            call_args = mock_engine.collection.upsert.call_args
            metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")
            assert metadatas[0].get("corpus_id") == "ai-act"

    def test_generates_chunk_id_if_missing(self):
        """Should generate chunk_id from source/page/index if missing."""
        from src.engine.indexing import index_jsonl

        with TemporaryDirectory() as tmpdir:
            mock_engine = MagicMock()
            mock_engine.collection = MagicMock()
            mock_engine._embed = MagicMock(return_value=[[0.1] * 10])
            mock_engine.corpus_id = "test"

            jsonl_path = Path(tmpdir) / "test.jsonl"
            with open(jsonl_path, "w") as f:
                f.write(json.dumps({
                    "text": "Content",
                    "metadata": {"source": "AI Act", "page": "p1", "chunk_index": 2}
                }) + "\n")

            index_jsonl(mock_engine, str(jsonl_path))

            call_args = mock_engine.collection.upsert.call_args
            ids = call_args.kwargs.get("ids") or call_args[1].get("ids")
            # Should generate id like "ai-act-p1-2"
            assert ids[0] == "ai-act-p1-2"


# ---------------------------------------------------------------------------
# Test: index_documents behavior
# ---------------------------------------------------------------------------


class TestIndexDocuments:
    def test_raises_if_directory_not_found(self):
        from src.engine.indexing import index_documents
        from src.engine.types import RAGEngineError

        mock_engine = MagicMock()
        mock_engine.docs_path = "/nonexistent/path"

        with pytest.raises(RAGEngineError, match="Document directory not found"):
            index_documents(mock_engine)

    def test_skips_already_indexed_documents(self):
        from src.engine.indexing import index_documents

        with TemporaryDirectory() as tmpdir:
            # Create a test document
            doc_path = Path(tmpdir) / "test.txt"
            doc_path.write_text("Test content")

            mock_engine = MagicMock()
            mock_engine.docs_path = tmpdir
            mock_engine.collection.get.return_value = {"documents": ["Already exists"]}

            index_documents(mock_engine)

            # Upsert should not be called since document exists
            mock_engine.collection.upsert.assert_not_called()

    @pytest.mark.slow
    def test_indexes_new_documents(self):
        from src.engine.indexing import index_documents

        with TemporaryDirectory() as tmpdir:
            # Create test documents
            (Path(tmpdir) / "doc1.txt").write_text("Content 1")
            (Path(tmpdir) / "doc2.txt").write_text("Content 2")

            mock_engine = MagicMock()
            mock_engine.docs_path = tmpdir
            mock_engine.collection.get.return_value = {"documents": []}  # Not indexed
            mock_engine.collection.count.return_value = 0
            mock_engine._embed = MagicMock(return_value=[[0.1] * 10, [0.2] * 10])

            index_documents(mock_engine)

            # Should have called upsert
            mock_engine.collection.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# Test: Embedding enrichment integration
# ---------------------------------------------------------------------------


class TestEmbeddingEnrichmentIntegration:
    def test_enrichment_disabled_skips_llm(self):
        """When enrichment is disabled, should not call LLM."""
        doc = "Test content"
        meta = {}

        with patch("src.ingestion.embedding_enrichment.is_enrichment_enabled", return_value=False):
            result = _enrich_document_for_embedding(doc, meta)

        assert result == "Test content"

    def test_uses_precomputed_terms_from_jsonl(self):
        """Pre-computed enrichment_terms should be used directly."""
        doc = "Test content"
        meta = {"enrichment_terms": ["precomputed", "terms"]}

        result = _enrich_document_for_embedding(doc, meta)

        assert "[Søgetermer: precomputed | terms]" in result


# ---------------------------------------------------------------------------
# Test: Dimension mismatch recovery
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDimensionMismatchRecovery:
    def test_resets_collection_on_dimension_error(self):
        """Should reset collection when embedding dimension changes."""
        from src.engine.indexing import _upsert_with_embeddings
        from chromadb.errors import InvalidArgumentError

        mock_engine = MagicMock()
        mock_engine.collection_name = "test"
        mock_engine._embed = MagicMock(return_value=[[0.1] * 10])

        # First call raises dimension error, second succeeds
        mock_engine.collection.upsert.side_effect = [
            InvalidArgumentError("dimension mismatch"),
            None,  # Success after reset
        ]
        mock_engine.chroma.create_collection.return_value = MagicMock()

        _upsert_with_embeddings(
            mock_engine,
            ids=["test-1"],
            documents=["Test"],
            metadatas=[{}],
        )

        # Should have called delete_collection and create_collection
        mock_engine.chroma.delete_collection.assert_called_once()
        mock_engine.chroma.create_collection.assert_called_once()
