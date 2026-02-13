"""Tests for prompt_builder module."""
import pytest
from typing import Dict, Any, Tuple

from src.engine.prompt_builder import (
    PromptContext,
    build_references_structured,
    build_kilder_block,
    build_context_string,
    build_prompt_context,
    _derive_structural_fields,
    _extract_raw_anchors_from_meta,
)
from src.engine.retrieval_pipeline import SelectedChunk, RetrievedChunk


def _make_mock_format_fn():
    """Create a simple format metadata function for testing."""
    def format_fn(meta: Dict[str, Any], doc: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        article = meta.get("article", "")
        annex = meta.get("annex", "")
        recital = meta.get("recital", "")

        parts = []
        if article:
            parts.append(f"Artikel {article}")
        if annex:
            parts.append(f"Bilag {annex}")
        if recital:
            parts.append(f"Betragtning {recital}")

        reference = ", ".join(parts) if parts else "Unknown"
        sanitized = {
            "article": article,
            "annex": annex,
            "recital": recital,
            "paragraph": meta.get("paragraph"),
        }
        validation = {"valid": True}
        return reference, sanitized, validation
    return format_fn


class TestDeriveStructuralFields:
    """Tests for _derive_structural_fields."""

    def test_empty_meta(self):
        assert _derive_structural_fields({}) == {}

    def test_no_location_id(self):
        assert _derive_structural_fields({"article": "5"}) == {}

    def test_parse_article_from_location_id(self):
        meta = {"location_id": "chapter:1/article:5/paragraph:2"}
        result = _derive_structural_fields(meta)
        assert result.get("chapter") == "1"
        assert result.get("article") == "5"
        assert result.get("paragraph") == "2"

    def test_parse_annex_from_location_id(self):
        meta = {"location_id": "annex:iii/section:2"}
        result = _derive_structural_fields(meta)
        assert result.get("annex") == "iii"
        assert result.get("section") == "2"


class TestExtractRawAnchors:
    """Tests for _extract_raw_anchors_from_meta."""

    def test_empty_meta(self):
        assert _extract_raw_anchors_from_meta({}) == []

    def test_article_only(self):
        result = _extract_raw_anchors_from_meta({"article": "6"})
        assert result == ["article:6"]

    def test_annex_only(self):
        result = _extract_raw_anchors_from_meta({"annex": "III"})
        assert result == ["annex:iii"]

    def test_multiple_anchors(self):
        result = _extract_raw_anchors_from_meta({
            "article": "5",
            "recital": "10",
            "annex": "II",
        })
        assert sorted(result) == ["annex:ii", "article:5", "recital:10"]


class TestBuildReferencesStructured:
    """Tests for build_references_structured."""

    def test_empty_included(self):
        refs, blocks, structured = build_references_structured([], _make_mock_format_fn())
        assert refs == []
        assert blocks == []
        assert structured == []

    def test_single_chunk(self):
        included = [
            ("This is the document text.", {"article": "6", "corpus_id": "test"}, "chunk-1", None)
        ]
        refs, blocks, structured = build_references_structured(included, _make_mock_format_fn())

        assert len(refs) == 1
        assert refs[0] == "[1] Artikel 6"

        assert len(blocks) == 1
        assert "[1] Artikel 6\nThis is the document text." in blocks[0]

        assert len(structured) == 1
        assert structured[0]["idx"] == 1
        assert structured[0]["chunk_id"] == "chunk-1"
        assert structured[0]["article"] == "6"

    def test_multiple_chunks(self):
        included = [
            ("Doc 1", {"article": "5"}, "chunk-1", None),
            ("Doc 2", {"annex": "III"}, "chunk-2", None),
            ("Doc 3", {"recital": "10"}, "chunk-3", "precise-ref"),
        ]
        refs, blocks, structured = build_references_structured(included, _make_mock_format_fn())

        assert len(refs) == 3
        assert "[1] Artikel 5" in refs[0]
        assert "[2] Bilag III" in refs[1]
        assert "[3] Betragtning 10" in refs[2]

        assert len(structured) == 3
        assert structured[2]["precise_ref"] == "precise-ref"


class TestBuildKilderBlock:
    """Tests for build_kilder_block."""

    def test_empty_references(self):
        assert build_kilder_block([], "test-corpus") == ""

    def test_single_reference(self):
        refs = [{"idx": 1, "corpus_id": "ai-act", "article": "6"}]
        result = build_kilder_block(refs, "default-corpus")
        assert "KILDER:" in result
        assert "[1]" in result
        assert "ai-act" in result

    def test_multiple_references_sorted(self):
        refs = [
            {"idx": 3, "corpus_id": "test"},
            {"idx": 1, "corpus_id": "test"},
            {"idx": 2, "corpus_id": "test"},
        ]
        result = build_kilder_block(refs, "test")
        lines = result.split("\n")
        # First line is KILDER:, then sorted by idx
        assert "[1]" in lines[1]
        assert "[2]" in lines[2]
        assert "[3]" in lines[3]


class TestBuildContextString:
    """Tests for build_context_string."""

    def test_empty_blocks(self):
        assert build_context_string([], "") == ""

    def test_no_kilder_block(self):
        blocks = ["[1] Ref\nDoc 1", "[2] Ref\nDoc 2"]
        result = build_context_string(blocks, "")
        assert result == "[1] Ref\nDoc 1\n\n[2] Ref\nDoc 2"

    def test_with_kilder_block(self):
        blocks = ["[1] Ref\nDoc"]
        kilder = "KILDER:\n- [1] test"
        result = build_context_string(blocks, kilder)
        assert result.startswith("KILDER:")
        assert "[1] Ref\nDoc" in result


class TestBuildPromptContext:
    """Tests for build_prompt_context (main entry point)."""

    def _make_selected_chunk(
        self,
        doc: str,
        meta: Dict[str, Any],
        chunk_id: str = "test-chunk",
        distance: float = 0.1,
        precise_ref: str | None = None,
    ) -> SelectedChunk:
        chunk = RetrievedChunk(
            chunk_id=chunk_id,
            document=doc,
            metadata=meta,
            distance=distance,
        )
        return SelectedChunk(chunk=chunk, is_citable=True, precise_ref=precise_ref)

    def test_empty_selected(self):
        result = build_prompt_context(
            selected=(),
            format_metadata_fn=_make_mock_format_fn(),
            corpus_id="test",
        )
        assert isinstance(result, PromptContext)
        assert result.citable_count == 0
        assert result.references == []
        assert result.context_string == ""

    def test_single_selected_chunk(self):
        selected = (
            self._make_selected_chunk(
                doc="Article 6 requires notification.",
                meta={"article": "6", "corpus_id": "ai-act"},
                chunk_id="art6-chunk-1",
            ),
        )
        result = build_prompt_context(
            selected=selected,
            format_metadata_fn=_make_mock_format_fn(),
            corpus_id="ai-act",
        )

        assert result.citable_count == 1
        assert len(result.references) == 1
        assert "[1] Artikel 6" in result.references[0]
        assert "KILDER:" in result.context_string
        assert "Article 6 requires notification." in result.context_string

    def test_multiple_selected_chunks(self):
        selected = (
            self._make_selected_chunk(
                doc="First chunk",
                meta={"article": "5"},
                chunk_id="chunk-1",
            ),
            self._make_selected_chunk(
                doc="Second chunk",
                meta={"annex": "III", "annex_point": "5"},
                chunk_id="chunk-2",
            ),
        )
        result = build_prompt_context(
            selected=selected,
            format_metadata_fn=_make_mock_format_fn(),
            corpus_id="test",
        )

        assert result.citable_count == 2
        assert len(result.references_structured) == 2
        assert result.references_structured[1]["annex"] == "III"

    def test_raw_anchor_logging(self):
        selected = (
            self._make_selected_chunk(
                doc="Content",
                meta={"article": "6", "recital": "10"},
                chunk_id="chunk-1",
            ),
        )
        result = build_prompt_context(
            selected=selected,
            format_metadata_fn=_make_mock_format_fn(),
            corpus_id="test",
            enable_raw_anchor_log=True,
        )

        assert "article:6" in result.raw_context_anchors
        assert "recital:10" in result.raw_context_anchors
        assert result.debug.get("raw_context_anchors_count") == 2

    def test_included_format_compatibility(self):
        """Test that included tuple format matches legacy expectations."""
        selected = (
            self._make_selected_chunk(
                doc="Test doc",
                meta={"article": "7"},
                chunk_id="test-id",
                precise_ref="Precise Reference",
            ),
        )
        result = build_prompt_context(
            selected=selected,
            format_metadata_fn=_make_mock_format_fn(),
            corpus_id="test",
        )

        # included should be (doc, meta, chunk_id, precise_ref)
        assert len(result.included) == 1
        doc, meta, chunk_id, precise = result.included[0]
        assert doc == "Test doc"
        assert meta.get("article") == "7"
        assert chunk_id == "test-id"
        assert precise == "Precise Reference"


# ===========================================================================
# Discovery Preamble
# ===========================================================================


class TestBuildDiscoveryPreamble:
    """Tests for build_discovery_preamble function."""

    def test_auto_gate(self) -> None:
        from src.engine.prompt_builder import build_discovery_preamble

        result = build_discovery_preamble(
            gate="AUTO",
            matches=[
                {"corpus_id": "ai_act", "confidence": 0.92, "display_name": "AI-Act"},
            ],
        )
        assert "AI-Act" in result
        assert result  # Non-empty

    def test_auto_gate_multiple(self) -> None:
        from src.engine.prompt_builder import build_discovery_preamble

        result = build_discovery_preamble(
            gate="AUTO",
            matches=[
                {"corpus_id": "ai_act", "confidence": 0.92, "display_name": "AI-Act"},
                {"corpus_id": "gdpr", "confidence": 0.78, "display_name": "GDPR"},
            ],
        )
        assert "AI-Act" in result
        assert "GDPR" in result

    def test_suggest_gate(self) -> None:
        from src.engine.prompt_builder import build_discovery_preamble

        result = build_discovery_preamble(
            gate="SUGGEST",
            matches=[
                {"corpus_id": "data_act", "confidence": 0.68, "display_name": "Data Act"},
            ],
        )
        assert "Data Act" in result
        assert result  # Non-empty

    def test_abstain_returns_empty(self) -> None:
        from src.engine.prompt_builder import build_discovery_preamble

        result = build_discovery_preamble(gate="ABSTAIN", matches=[])
        assert result == ""

    def test_uses_template(self) -> None:
        from src.engine.prompt_builder import build_discovery_preamble

        result = build_discovery_preamble(
            gate="AUTO",
            matches=[{"corpus_id": "ai_act", "confidence": 0.9, "display_name": "AI-Act"}],
            template_auto="Found law: {law_names}",
        )
        assert result == "Found law: AI-Act"
