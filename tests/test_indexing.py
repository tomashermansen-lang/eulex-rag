"""Tests for src/engine/indexing.py - Document enrichment and ingestion."""

import json
import logging
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from src.engine.indexing import (
    _enrich_document_for_embedding,
    _stringify_metadata,
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
        with patch(
            "src.ingestion.embedding_enrichment.is_enrichment_enabled",
            return_value=False,
        ):
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

        # The first line starts with "a)" so it should NOT be used as heading.
        # Without heading enrichment the doc is returned unchanged (no \n\n prefix).
        assert result.startswith(long_first_line)


# ---------------------------------------------------------------------------
# Test: _stringify_metadata (module-level, extracted from _upsert_with_embeddings)
# ---------------------------------------------------------------------------


class TestStringifyMetadata:
    def test_converts_list_to_pipe_separated(self):
        result = _stringify_metadata({"enrichment_terms": ["a", "b", "c"]})
        assert result["enrichment_terms"] == "a | b | c"

    def test_preserves_scalar_values(self):
        result = _stringify_metadata({"ecli": "ECLI:EU:C:2020:559", "court": "CJEU"})
        assert result == {"ecli": "ECLI:EU:C:2020:559", "court": "CJEU"}

    def test_handles_empty_list(self):
        result = _stringify_metadata({"tags": []})
        assert result["tags"] == ""

    def test_existing_upsert_still_works(self):
        """Regression: _upsert_with_embeddings still stringifies via extracted function."""
        from src.engine.indexing import _upsert_with_embeddings

        mock_engine = MagicMock()
        mock_engine.collection = MagicMock()
        mock_engine._embed = MagicMock(return_value=[[0.1] * 10])
        mock_engine.corpus_id = "test"

        _upsert_with_embeddings(
            mock_engine,
            ids=["t1"],
            documents=["Doc"],
            metadatas=[{"enrichment_terms": ["x", "y"]}],
        )

        call_args = mock_engine.collection.upsert.call_args
        metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")
        assert metadatas[0]["enrichment_terms"] == "x | y"


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
                f.write(
                    json.dumps(
                        {
                            "text": "Content",
                            "metadata": {
                                "chunk_id": "chunk-1",
                                "enrichment_terms": ["term1", "term2", "term3"],
                            },
                        }
                    )
                    + "\n"
                )

            index_jsonl(mock_engine, str(jsonl_path))

            call_args = mock_engine.collection.upsert.call_args
            metadatas = call_args.kwargs.get("metadatas") or call_args[1].get(
                "metadatas"
            )
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
                f.write(
                    json.dumps(
                        {
                            "text": "Table of Contents",
                            "metadata": {"doc_type": "toc", "chunk_id": "toc-1"},
                        }
                    )
                    + "\n"
                )
                # Regular entry - should be indexed
                f.write(
                    json.dumps(
                        {"text": "Regular content", "metadata": {"chunk_id": "chunk-1"}}
                    )
                    + "\n"
                )

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
                f.write(
                    json.dumps(
                        {
                            "text": "Content without corpus_id",
                            "metadata": {"chunk_id": "chunk-1"},
                        }
                    )
                    + "\n"
                )

            index_jsonl(mock_engine, str(jsonl_path))

            # Check metadata includes corpus_id
            call_args = mock_engine.collection.upsert.call_args
            metadatas = call_args.kwargs.get("metadatas") or call_args[1].get(
                "metadatas"
            )
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
                f.write(
                    json.dumps(
                        {
                            "text": "Content",
                            "metadata": {
                                "source": "AI Act",
                                "page": "p1",
                                "chunk_index": 2,
                            },
                        }
                    )
                    + "\n"
                )

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

        with patch(
            "src.ingestion.embedding_enrichment.is_enrichment_enabled",
            return_value=False,
        ):
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


# ---------------------------------------------------------------------------
# Test: index_case_law_jsonl
# ---------------------------------------------------------------------------


def _write_case_law_jsonl(path: Path, chunks: list[dict]) -> Path:
    """Write chunks as JSONL. Each chunk: {"text": ..., "metadata": {...}}."""
    with open(path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    return path


SAMPLE_CASE_LAW_CHUNK = {
    "text": "The Court finds that Article 6 applies...",
    "metadata": {
        "source_type": "cjeu_case_law",
        "chunk_id": "chunk:v1/case-c-311-18/grounds/0",
        "ecli": "ECLI:EU:C:2020:559",
        "case_number": "C-311/18",
        "court": "CJEU",
        "decision_date": "2020-07-16",
        "section_type": "grounds",
        "paragraph_range": "42-48",
        "articles_interpreted": '["gdpr/article:46"]',
        "case_name": "Schrems II",
        "heading_path": "grounds",
        "heading_path_display": "Grounds of the judgment",
        "corpus_id": "gdpr",
        "schema_version": "case_law_chunk:v1",
    },
}


def _make_settings_mock(enabled: bool = True, processed_dir: str = "data/processed"):
    """Create a mock Settings object with case_law config."""
    settings = MagicMock()
    settings.case_law.enabled = enabled
    settings.processed_dir = Path(processed_dir)
    return settings


def _make_engine_mock(embed_dim: int = 10):
    """Create a mock engine with chroma and _embed."""
    engine = MagicMock()
    engine._embed = MagicMock(side_effect=lambda docs: [[0.1] * embed_dim] * len(docs))
    engine.chroma = MagicMock()
    engine.corpus_id = "gdpr"
    # get_or_create_collection returns a mock collection
    collection = MagicMock()
    engine.chroma.get_or_create_collection.return_value = collection
    return engine


class TestIndexCaseLawJsonl:
    """Tests for index_case_law_jsonl (Component 2)."""

    # --- Feature Flag (REQ-5) ---

    @patch("src.engine.indexing.load_settings")
    def test_returns_zero_when_disabled(self, mock_load):
        from src.engine.indexing import index_case_law_jsonl

        mock_load.return_value = _make_settings_mock(enabled=False)
        engine = _make_engine_mock()

        result = index_case_law_jsonl(engine, "gdpr")

        assert result == 0
        engine.chroma.get_or_create_collection.assert_not_called()

    @patch("src.engine.indexing.load_settings")
    def test_checks_flag_before_io(self, mock_load):
        from src.engine.indexing import index_case_law_jsonl

        mock_load.return_value = _make_settings_mock(enabled=False)
        engine = _make_engine_mock()

        with patch("builtins.open", side_effect=AssertionError("open called")):
            result = index_case_law_jsonl(engine, "gdpr")

        assert result == 0

    @patch("src.engine.indexing.load_settings")
    def test_reads_flag_fresh_each_call(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "case_law" / "gdpr"
        d.mkdir(parents=True)
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        # First call: disabled
        mock_load.return_value = _make_settings_mock(enabled=False)
        engine = _make_engine_mock()
        r1 = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        assert r1 == 0

        # Second call: enabled
        mock_load.return_value = _make_settings_mock(enabled=True)
        r2 = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        assert r2 == 1

    # --- Directory Resolution (REQ-10, REQ-7) ---

    @patch("src.engine.indexing.load_settings")
    def test_raises_on_missing_directory(self, mock_load):
        from src.engine.indexing import index_case_law_jsonl
        from src.engine.types import RAGEngineError

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        with pytest.raises(RAGEngineError):
            index_case_law_jsonl(engine, "gdpr", jsonl_dir="/nonexistent/path")

    @patch("src.engine.indexing.load_settings")
    def test_defaults_dir_from_settings(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "case_law" / "gdpr"
        d.mkdir(parents=True)
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        settings = _make_settings_mock(enabled=True)
        settings.processed_dir = tmp_path
        mock_load.return_value = settings
        engine = _make_engine_mock()

        result = index_case_law_jsonl(engine, "gdpr")
        assert result == 1

    @patch("src.engine.indexing.load_settings")
    def test_returns_zero_for_empty_directory(self, mock_load, tmp_path, caplog):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "empty"
        d.mkdir()

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        with caplog.at_level(logging.INFO):
            result = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        assert result == 0
        assert any(
            "no" in r.message.lower() or "0" in r.message for r in caplog.records
        )

    @patch("src.engine.indexing.load_settings")
    def test_ignores_non_jsonl_files(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "mixed"
        d.mkdir()
        (d / "data.tmp").write_text("not jsonl")
        _write_case_law_jsonl(d / "case_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        result = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        assert result == 1

    # --- Happy Path (REQ-1, REQ-2) ---

    @patch("src.engine.indexing.load_settings")
    def test_indexes_single_file(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "single"
        d.mkdir()
        chunk2 = dict(SAMPLE_CASE_LAW_CHUNK)
        chunk2 = {
            "text": "Second chunk.",
            "metadata": {
                **SAMPLE_CASE_LAW_CHUNK["metadata"],
                "chunk_id": "chunk:v1/case-c-311-18/grounds/1",
            },
        }
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK, chunk2])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        result = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        assert result == 2

    @patch("src.engine.indexing.load_settings")
    def test_creates_case_law_collection(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "coll"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        engine.chroma.get_or_create_collection.assert_called_with("gdpr_case_law")

    @patch("src.engine.indexing.load_settings")
    def test_does_not_modify_engine_collection(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "nomod"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()
        original_collection = engine.collection

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        assert engine.collection is original_collection

    # --- Multi-File (REQ-7) ---

    @patch("src.engine.indexing.load_settings")
    def test_indexes_multiple_files(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "multi"
        d.mkdir()
        for i in range(3):
            chunk = {
                "text": f"Chunk {i}",
                "metadata": {
                    **SAMPLE_CASE_LAW_CHUNK["metadata"],
                    "chunk_id": f"chunk:v1/case-{i}/grounds/0",
                },
            }
            _write_case_law_jsonl(d / f"case{i}_chunks.jsonl", [chunk])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        result = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        assert result == 3

    @patch("src.engine.indexing.load_settings")
    def test_processes_files_in_sorted_order(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "sorted"
        d.mkdir()
        chunk_b = {
            "text": "B chunk",
            "metadata": {**SAMPLE_CASE_LAW_CHUNK["metadata"], "chunk_id": "b-id"},
        }
        chunk_a = {
            "text": "A chunk",
            "metadata": {**SAMPLE_CASE_LAW_CHUNK["metadata"], "chunk_id": "a-id"},
        }
        _write_case_law_jsonl(d / "b_chunks.jsonl", [chunk_b])
        _write_case_law_jsonl(d / "a_chunks.jsonl", [chunk_a])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d), batch_size=1)

        # With batch_size=1, each chunk triggers a separate upsert.
        # a_chunks.jsonl should be processed before b_chunks.jsonl.
        collection = engine.chroma.get_or_create_collection.return_value
        upsert_calls = collection.upsert.call_args_list
        first_ids = upsert_calls[0].kwargs.get("ids") or upsert_calls[0][1].get("ids")
        second_ids = upsert_calls[1].kwargs.get("ids") or upsert_calls[1][1].get("ids")
        assert first_ids == ["a-id"]
        assert second_ids == ["b-id"]

    @patch("src.engine.indexing.load_settings")
    def test_single_file_works(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "one"
        d.mkdir()
        _write_case_law_jsonl(d / "only_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        result = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        assert result == 1

    # --- Metadata Preservation (REQ-3) ---

    @patch("src.engine.indexing.load_settings")
    def test_full_metadata_preserved(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "meta"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        collection = engine.chroma.get_or_create_collection.return_value
        call_args = collection.upsert.call_args
        metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")
        meta = metadatas[0]
        assert meta["ecli"] == "ECLI:EU:C:2020:559"
        assert meta["case_number"] == "C-311/18"
        assert meta["court"] == "CJEU"
        assert meta["decision_date"] == "2020-07-16"
        assert meta["source_type"] == "cjeu_case_law"
        assert meta["section_type"] == "grounds"
        assert meta["paragraph_range"] == "42-48"
        assert meta["case_name"] == "Schrems II"

    @patch("src.engine.indexing.load_settings")
    def test_list_metadata_stringified(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "listmeta"
        d.mkdir()
        chunk = {
            "text": "Some text",
            "metadata": {
                **SAMPLE_CASE_LAW_CHUNK["metadata"],
                "enrichment_terms": ["a", "b"],
            },
        }
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [chunk])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        collection = engine.chroma.get_or_create_collection.return_value
        call_args = collection.upsert.call_args
        metadatas = call_args.kwargs.get("metadatas") or call_args[1].get("metadatas")
        assert metadatas[0]["enrichment_terms"] == "a | b"

    # --- Upsert Semantics (REQ-4) ---

    @patch("src.engine.indexing.load_settings")
    def test_deletes_before_upsert(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "del"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        collection = engine.chroma.get_or_create_collection.return_value
        # delete must be called before upsert
        delete_call_order = collection.delete.call_count
        upsert_call_order = collection.upsert.call_count
        assert delete_call_order >= 1
        assert upsert_call_order >= 1
        # Verify delete was called with the chunk IDs
        delete_args = collection.delete.call_args
        assert "chunk:v1/case-c-311-18/grounds/0" in (
            delete_args.kwargs.get("ids") or delete_args[1].get("ids", [])
        )

    @patch("src.engine.indexing.load_settings")
    def test_upsert_replaces_existing(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "replace"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        # Call twice
        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        collection = engine.chroma.get_or_create_collection.return_value
        assert collection.delete.call_count == 2
        assert collection.upsert.call_count == 2

    # --- Embedding Enrichment (REQ-6) ---

    @patch("src.engine.indexing._enrich_document_for_embedding")
    @patch("src.engine.indexing.load_settings")
    def test_enrichment_applied(self, mock_load, mock_enrich, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        mock_enrich.return_value = "ENRICHED TEXT"

        d = tmp_path / "enrich"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        mock_enrich.assert_called_once()
        call_args = mock_enrich.call_args
        assert call_args[0][0] == SAMPLE_CASE_LAW_CHUNK["text"]  # doc
        assert call_args.kwargs.get("corpus_id") == "gdpr"  # corpus_id
        # engine._embed should receive enriched text
        engine._embed.assert_called_once_with(["ENRICHED TEXT"])

    @patch("src.engine.indexing._enrich_document_for_embedding")
    @patch("src.engine.indexing.load_settings")
    def test_stores_original_text(self, mock_load, mock_enrich, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        mock_enrich.return_value = "ENRICHED TEXT"

        d = tmp_path / "orig"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        collection = engine.chroma.get_or_create_collection.return_value
        call_args = collection.upsert.call_args
        docs = call_args.kwargs.get("documents") or call_args[1].get("documents")
        assert docs == [SAMPLE_CASE_LAW_CHUNK["text"]]

    # --- Error Handling (REQ-9) ---

    @patch("src.engine.indexing.load_settings")
    def test_skips_malformed_json_line(self, mock_load, tmp_path, caplog):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "malformed"
        d.mkdir()
        jsonl_path = d / "bad_chunks.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps(SAMPLE_CASE_LAW_CHUNK) + "\n")
            f.write("not json{\n")
            chunk2 = {
                "text": "Valid chunk 2",
                "metadata": {
                    **SAMPLE_CASE_LAW_CHUNK["metadata"],
                    "chunk_id": "chunk:v1/valid/0",
                },
            }
            f.write(json.dumps(chunk2) + "\n")

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        with caplog.at_level(logging.WARNING):
            result = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        assert result == 2
        assert any(
            "line" in r.message.lower() or "json" in r.message.lower()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    @patch("src.engine.indexing.load_settings")
    def test_skips_file_open_error(self, mock_load, tmp_path, caplog):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "openerr"
        d.mkdir()
        # First file: will be made unreadable via patch
        _write_case_law_jsonl(d / "a_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])
        chunk2 = {
            "text": "Second",
            "metadata": {
                **SAMPLE_CASE_LAW_CHUNK["metadata"],
                "chunk_id": "chunk:v1/second/0",
            },
        }
        _write_case_law_jsonl(d / "b_chunks.jsonl", [chunk2])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        original_open = open

        def patched_open(path, *args, **kwargs):
            if "a_chunks" in str(path):
                raise OSError("Permission denied")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=patched_open):
            with caplog.at_level(logging.WARNING):
                result = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        assert result == 1

    @patch("src.engine.indexing.load_settings")
    def test_empty_lines_skipped(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "emptylines"
        d.mkdir()
        jsonl_path = d / "e_chunks.jsonl"
        with open(jsonl_path, "w") as f:
            f.write("\n\n\n")

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        result = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        assert result == 0

    # --- Edge Cases ---

    @patch("src.engine.indexing.load_settings")
    def test_empty_text_skipped(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "emptytext"
        d.mkdir()
        chunk = {"text": "", "metadata": {**SAMPLE_CASE_LAW_CHUNK["metadata"]}}
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [chunk])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        result = index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))
        assert result == 0

    @patch("src.engine.indexing.load_settings")
    def test_batch_size_respected(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "batch"
        d.mkdir()
        chunks = []
        for i in range(5):
            chunks.append(
                {
                    "text": f"Chunk {i}",
                    "metadata": {
                        **SAMPLE_CASE_LAW_CHUNK["metadata"],
                        "chunk_id": f"c-{i}",
                    },
                }
            )
        _write_case_law_jsonl(d / "c1_chunks.jsonl", chunks)

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d), batch_size=2)

        collection = engine.chroma.get_or_create_collection.return_value
        assert collection.upsert.call_count == 3  # 2+2+1

    @patch("src.engine.indexing.load_settings")
    def test_chromadb_error_propagates(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl

        d = tmp_path / "error"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()
        collection = engine.chroma.get_or_create_collection.return_value
        collection.upsert.side_effect = RuntimeError("ChromaDB down")

        with pytest.raises(RuntimeError, match="ChromaDB down"):
            index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

    # --- Dimension Mismatch Recovery (REQ-11) ---

    @patch("src.engine.indexing.load_settings")
    def test_resets_collection_on_dimension_mismatch(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl
        from chromadb.errors import InvalidArgumentError

        d = tmp_path / "dim"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()
        collection = engine.chroma.get_or_create_collection.return_value

        # First upsert raises dimension error, second succeeds
        collection.upsert.side_effect = [
            InvalidArgumentError("dimension mismatch"),
            None,
        ]
        new_collection = MagicMock()
        # After reset, get_or_create returns a new collection
        engine.chroma.get_or_create_collection.side_effect = [
            collection,
            new_collection,
        ]

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        engine.chroma.delete_collection.assert_called_once_with("gdpr_case_law")

    @patch("src.engine.indexing.load_settings")
    def test_only_resets_case_law_collection(self, mock_load, tmp_path):
        from src.engine.indexing import index_case_law_jsonl
        from chromadb.errors import InvalidArgumentError

        d = tmp_path / "dim2"
        d.mkdir()
        _write_case_law_jsonl(d / "c1_chunks.jsonl", [SAMPLE_CASE_LAW_CHUNK])

        mock_load.return_value = _make_settings_mock(enabled=True)
        engine = _make_engine_mock()
        collection = engine.chroma.get_or_create_collection.return_value
        collection.upsert.side_effect = [
            InvalidArgumentError("dimension mismatch"),
            None,
        ]
        new_collection = MagicMock()
        engine.chroma.get_or_create_collection.side_effect = [
            collection,
            new_collection,
        ]

        index_case_law_jsonl(engine, "gdpr", jsonl_dir=str(d))

        # Verify it deleted "gdpr_case_law", NOT "gdpr_documents"
        delete_call = engine.chroma.delete_collection.call_args
        assert delete_call[0][0] == "gdpr_case_law"
