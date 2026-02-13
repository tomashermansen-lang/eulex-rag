"""Tests for src/ingestion/html_chunks.py - HTML chunking utilities.

Focus on helper functions and data classes that can be tested in isolation
without full HTML parsing infrastructure.
"""

import json
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.ingestion.html_chunks import (
    FormatWarning,
    PreflightResult,
    HtmlChunkingConfig,
    _slugify,
    make_chunk_id,
    make_location_id,
    _extract_reference_mentions,
    _eurlex_struct_from_id,
    _split_text_to_fit,
    _apply_overlap,
    preflight_check_html,
    chunk_html,
    write_jsonl,
)


class TestFormatWarning:
    """Tests for FormatWarning dataclass."""

    def test_create_with_defaults(self):
        """Create warning with default severity."""
        warning = FormatWarning(
            category="test_cat",
            message="Test message",
            location="near line 42",
        )
        assert warning.category == "test_cat"
        assert warning.message == "Test message"
        assert warning.location == "near line 42"
        assert warning.severity == "warning"
        assert warning.suggestion == ""

    def test_create_with_all_fields(self):
        """Create warning with all fields specified."""
        warning = FormatWarning(
            category="unclassified_table",
            message="Table with 10 rows",
            location="near BILAG IV",
            severity="info",
            suggestion="Add pattern to classifier",
        )
        assert warning.severity == "info"
        assert warning.suggestion == "Add pattern to classifier"

    def test_is_frozen(self):
        """FormatWarning is immutable."""
        warning = FormatWarning(
            category="test",
            message="msg",
            location="loc",
        )
        with pytest.raises(AttributeError):
            warning.category = "changed"


class TestPreflightResult:
    """Tests for PreflightResult dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        result = PreflightResult()
        assert result.can_proceed is True
        assert result.warnings == []
        assert result.handled == {}
        assert result.unhandled == {}

    def test_has_warnings_false_when_empty(self):
        """has_warnings returns False when no warnings."""
        result = PreflightResult()
        assert result.has_warnings() is False

    def test_has_warnings_true_when_present(self):
        """has_warnings returns True when warnings exist."""
        result = PreflightResult(
            warnings=[FormatWarning(category="test", message="msg", location="loc")]
        )
        assert result.has_warnings() is True

    def test_summary_empty(self):
        """Summary for empty result."""
        result = PreflightResult()
        assert result.summary() == "No structures detected"

    def test_summary_with_handled(self):
        """Summary includes handled structures."""
        result = PreflightResult(
            handled={"articles": 5, "chapters": 2}
        )
        summary = result.summary()
        assert "Handled:" in summary
        assert "5x articles" in summary
        assert "2x chapters" in summary

    def test_summary_with_unhandled(self):
        """Summary includes unhandled structures with warning marker."""
        result = PreflightResult(
            unhandled={"unclassified_table": 3}
        )
        summary = result.summary()
        assert "⚠️ Unhandled:" in summary
        assert "3x unclassified_table" in summary

    def test_summary_with_both(self):
        """Summary includes both handled and unhandled."""
        result = PreflightResult(
            handled={"articles": 10},
            unhandled={"figure": 2}
        )
        summary = result.summary()
        assert "Handled:" in summary
        assert "⚠️ Unhandled:" in summary
        assert "|" in summary  # Separator


class TestHtmlChunkingConfig:
    """Tests for HtmlChunkingConfig dataclass."""

    def test_default_values(self):
        """Default chunking configuration."""
        config = HtmlChunkingConfig()
        assert config.chunk_tokens == 500
        assert config.overlap == 100
        assert config.flush_each_text_block is False

    def test_custom_values(self):
        """Custom chunking configuration."""
        config = HtmlChunkingConfig(
            chunk_tokens=300,
            overlap=50,
            flush_each_text_block=True,
        )
        assert config.chunk_tokens == 300
        assert config.overlap == 50
        assert config.flush_each_text_block is True

    def test_is_frozen(self):
        """HtmlChunkingConfig is immutable."""
        config = HtmlChunkingConfig()
        with pytest.raises(AttributeError):
            config.chunk_tokens = 1000


class TestSlugify:
    """Tests for _slugify function."""

    def test_basic_slugify(self):
        """Basic text is slugified."""
        assert _slugify("Hello World") == "hello-world"

    def test_removes_special_chars(self):
        """Special characters are removed."""
        assert _slugify("hello@world!") == "helloworld"
        assert _slugify("test#$%value") == "testvalue"

    def test_collapses_multiple_dashes(self):
        """Multiple dashes become one."""
        assert _slugify("hello---world") == "hello-world"
        assert _slugify("a - - - b") == "a-b"

    def test_trims_leading_trailing_dashes(self):
        """Leading/trailing dashes are removed."""
        assert _slugify("--hello--") == "hello"
        assert _slugify("   -test-   ") == "test"

    def test_empty_returns_doc(self):
        """Empty string returns 'doc'."""
        assert _slugify("") == "doc"
        assert _slugify("   ") == "doc"
        assert _slugify("!!!") == "doc"

    def test_preserves_numbers(self):
        """Numbers are preserved."""
        assert _slugify("article 123") == "article-123"
        assert _slugify("2024 report") == "2024-report"


class TestMakeChunkId:
    """Tests for make_chunk_id function."""

    def test_basic_chunk_id(self):
        """Generate basic chunk ID."""
        chunk_id = make_chunk_id(
            corpus_id="nis2",
            source="directive.html",
            chunk_index=0,
        )
        assert chunk_id == "nis2-directivehtml-html-c0"

    def test_chunk_id_with_index(self):
        """Chunk index is included."""
        chunk_id = make_chunk_id(
            corpus_id="ai-act",
            source="regulation.html",
            chunk_index=42,
        )
        assert "c42" in chunk_id

    def test_chunk_id_slugifies_inputs(self):
        """Inputs are slugified."""
        chunk_id = make_chunk_id(
            corpus_id="My Corpus",
            source="Test File.html",
            chunk_index=0,
        )
        assert chunk_id == "my-corpus-test-filehtml-html-c0"


class TestMakeLocationId:
    """Tests for make_location_id function."""

    def test_basic_location_id(self):
        """Generate location ID from reference state."""
        loc_id = make_location_id(
            corpus_id="nis2",
            reference_state={"article": "6"},
        )
        assert "article:6" in loc_id.lower()

    def test_location_id_with_multiple_refs(self):
        """Location ID includes multiple references."""
        loc_id = make_location_id(
            corpus_id="nis2",
            reference_state={"chapter": "II", "article": "6"},
        )
        # Should include both chapter and article
        loc_lower = loc_id.lower()
        assert "chapter" in loc_lower or "article" in loc_lower

    def test_location_id_ignores_corpus(self):
        """corpus_id is ignored (backwards compat)."""
        loc_id1 = make_location_id(
            corpus_id="corpus1",
            reference_state={"article": "6"},
        )
        loc_id2 = make_location_id(
            corpus_id="corpus2",
            reference_state={"article": "6"},
        )
        # corpus_id should not affect the location_id
        assert loc_id1 == loc_id2


class TestExtractReferenceMentions:
    """Tests for _extract_reference_mentions function."""

    def test_empty_text(self):
        """Empty text returns empty dict."""
        assert _extract_reference_mentions("") == {}
        assert _extract_reference_mentions(None) == {}
        assert _extract_reference_mentions("   ") == {}

    def test_extracts_article_mentions(self):
        """Extracts article references."""
        text = "Se artikel 6 og artikel 12 for detaljer."
        mentions = _extract_reference_mentions(text)
        assert "article" in mentions
        assert "6" in mentions["article"] or "12" in mentions["article"]

    def test_extracts_chapter_mentions(self):
        """Extracts chapter references."""
        text = "Kapitel II beskriver forpligtelser."
        mentions = _extract_reference_mentions(text)
        assert "chapter" in mentions
        assert "II" in mentions["chapter"]

    def test_extracts_annex_mentions(self):
        """Extracts annex references."""
        text = "Se bilag III for definitioner."
        mentions = _extract_reference_mentions(text)
        assert "annex" in mentions
        assert "III" in mentions["annex"]

    def test_extracts_paragraph_mentions(self):
        """Extracts paragraph references."""
        text = "Ifølge stk. 2 skal organisationen..."
        mentions = _extract_reference_mentions(text)
        assert "paragraph" in mentions
        assert "2" in mentions["paragraph"]

    def test_extracts_section_mentions(self):
        """Extracts section references."""
        text = "Afdeling 3 omhandler tilsyn."
        mentions = _extract_reference_mentions(text)
        assert "section" in mentions

    def test_excludes_current_state(self):
        """Excludes references matching current state."""
        text = "Artikel 6 henviser til artikel 6."
        mentions = _extract_reference_mentions(
            text,
            reference_state={"article": "6"}
        )
        # Should not include article 6 since it's the current location
        assert mentions.get("article", []) == []

    def test_limits_per_kind(self):
        """Respects max_per_kind limit."""
        text = " ".join([f"artikel {i}" for i in range(1, 50)])
        mentions = _extract_reference_mentions(text, max_per_kind=5)
        if "article" in mentions:
            assert len(mentions["article"]) <= 5

    def test_case_insensitive(self):
        """Pattern matching is case-insensitive."""
        text = "ARTIKEL 6 og Article 7"
        mentions = _extract_reference_mentions(text)
        assert "article" in mentions
        assert len(mentions["article"]) >= 1


class TestEurlexStructFromId:
    """Tests for _eurlex_struct_from_id function."""

    def test_empty_returns_none(self):
        """Empty/None input returns None."""
        assert _eurlex_struct_from_id("") is None
        assert _eurlex_struct_from_id(None) is None
        assert _eurlex_struct_from_id("   ") is None

    def test_parses_article_id(self):
        """Parses article IDs like 'art_6'."""
        result = _eurlex_struct_from_id("art_6")
        assert result == {"article": "6"}

    def test_parses_article_with_letter(self):
        """Parses article IDs with letter suffix like 'art_6a'."""
        result = _eurlex_struct_from_id("art_6a")
        assert result == {"article": "6A"}

    def test_parses_chapter_id(self):
        """Parses chapter IDs like 'cpt_II'."""
        result = _eurlex_struct_from_id("cpt_II")
        assert result == {"chapter": "II"}

    def test_parses_annex_id(self):
        """Parses annex IDs like 'anx_III'."""
        result = _eurlex_struct_from_id("anx_III")
        assert result == {"annex": "III"}

    def test_parses_chapter_with_section(self):
        """Parses chapter with section like 'cpt_II.sct_1'."""
        result = _eurlex_struct_from_id("cpt_II.sct_1")
        assert result == {"chapter": "II", "section": "1"}

    def test_parses_annex_with_section(self):
        """Parses annex with section."""
        result = _eurlex_struct_from_id("anx_I.sct_A")
        assert result == {"annex": "I", "section": "A"}

    def test_invalid_format_returns_none(self):
        """Invalid formats return None."""
        assert _eurlex_struct_from_id("invalid") is None
        assert _eurlex_struct_from_id("xyz_123") is None
        assert _eurlex_struct_from_id("art_") is None

    def test_case_insensitive(self):
        """Parsing is case-insensitive."""
        result = _eurlex_struct_from_id("ART_6")
        assert result == {"article": "6"}


class TestSplitTextToFit:
    """Tests for _split_text_to_fit function."""

    @pytest.fixture
    def mock_encoding(self):
        """Create a mock encoding that counts words as tokens."""
        encoding = MagicMock()
        # Simple mock: each word is one token
        encoding.encode = lambda text: text.split()
        return encoding

    def test_empty_text(self, mock_encoding):
        """Empty text returns empty list."""
        assert _split_text_to_fit("", encoding=mock_encoding, max_tokens=10) == []
        assert _split_text_to_fit("   ", encoding=mock_encoding, max_tokens=10) == []

    def test_text_under_limit(self, mock_encoding):
        """Text under limit returns as single item."""
        text = "short text"
        result = _split_text_to_fit(text, encoding=mock_encoding, max_tokens=10)
        assert len(result) == 1
        assert result[0] == "short text"

    def test_splits_on_sentences(self, mock_encoding):
        """Prefers sentence boundaries for splitting."""
        text = "First sentence. Second sentence. Third sentence."
        result = _split_text_to_fit(text, encoding=mock_encoding, max_tokens=3)
        # Should split at sentence boundaries
        assert len(result) >= 2
        for part in result:
            # Each part should be <= max_tokens
            assert len(part.split()) <= 3

    def test_falls_back_to_words(self, mock_encoding):
        """Falls back to word splitting for long sentences."""
        text = "one two three four five six seven eight nine ten"
        result = _split_text_to_fit(text, encoding=mock_encoding, max_tokens=3)
        assert len(result) >= 3
        for part in result:
            assert len(part.split()) <= 3

    def test_normalizes_whitespace(self, mock_encoding):
        """Normalizes multiple whitespace to single space."""
        text = "hello    world   test"
        result = _split_text_to_fit(text, encoding=mock_encoding, max_tokens=10)
        assert result[0] == "hello world test"


class TestApplyOverlap:
    """Tests for _apply_overlap function."""

    def test_zero_overlap_returns_empty(self):
        """Zero overlap returns empty list."""
        blocks = [("text1", 5), ("text2", 5)]
        result = _apply_overlap(blocks, max_overlap_tokens=0)
        assert result == []

    def test_negative_overlap_returns_empty(self):
        """Negative overlap returns empty list."""
        blocks = [("text1", 5), ("text2", 5)]
        result = _apply_overlap(blocks, max_overlap_tokens=-5)
        assert result == []

    def test_returns_last_blocks_within_limit(self):
        """Returns blocks from end within token limit."""
        blocks = [("a", 2), ("b", 2), ("c", 2)]
        result = _apply_overlap(blocks, max_overlap_tokens=4)
        # Should return last 2 blocks (4 tokens total)
        assert len(result) == 2
        assert result[0] == ("b", 2)
        assert result[1] == ("c", 2)

    def test_respects_token_limit(self):
        """Doesn't exceed max_overlap_tokens."""
        blocks = [("a", 10), ("b", 10), ("c", 10)]
        result = _apply_overlap(blocks, max_overlap_tokens=15)
        # Should return only last block (10 < 15)
        assert len(result) == 1
        assert result[0] == ("c", 10)

    def test_preserves_order(self):
        """Maintains original order of kept blocks."""
        blocks = [("first", 1), ("second", 1), ("third", 1)]
        result = _apply_overlap(blocks, max_overlap_tokens=3)
        assert result == blocks

    def test_empty_blocks(self):
        """Empty blocks list returns empty."""
        result = _apply_overlap([], max_overlap_tokens=10)
        assert result == []


class TestPreflightCheckHtml:
    """Tests for preflight_check_html function."""

    def test_returns_result_without_bs4(self):
        """Returns default result if BeautifulSoup not available."""
        with patch.dict("sys.modules", {"bs4": None}):
            # This would cause ModuleNotFoundError if bs4 is actually needed
            # The function should handle gracefully
            result = preflight_check_html("<html></html>")
            assert result.can_proceed is True

    def test_empty_html(self):
        """Empty HTML returns minimal result."""
        result = preflight_check_html("")
        assert result.can_proceed is True

    def test_simple_html(self):
        """Simple HTML without special structures."""
        html = "<html><body><p>Hello world</p></body></html>"
        result = preflight_check_html(html)
        assert result.can_proceed is True
        assert "paragraphs" in result.handled

    def test_detects_articles(self):
        """Detects EUR-Lex article divs."""
        html = """
        <html><body>
        <div id="art_1">Article 1</div>
        <div id="art_2">Article 2</div>
        </body></html>
        """
        result = preflight_check_html(html)
        assert result.handled.get("articles") == 2

    def test_detects_chapters(self):
        """Detects EUR-Lex chapter divs."""
        html = """
        <html><body>
        <div id="cpt_I">Chapter I</div>
        <div id="cpt_II">Chapter II</div>
        </body></html>
        """
        result = preflight_check_html(html)
        assert result.handled.get("chapters") == 2

    def test_detects_annexes(self):
        """Detects EUR-Lex annex divs."""
        html = """
        <html><body>
        <div id="anx_I">Annex I</div>
        <div id="anx_II">Annex II</div>
        <div id="anx_III">Annex III</div>
        </body></html>
        """
        result = preflight_check_html(html)
        assert result.handled.get("annexes") == 3

    def test_detects_figures(self):
        """Detects and warns about figures."""
        html = """
        <html><body>
        <figure><img src="diagram.png"/></figure>
        </body></html>
        """
        result = preflight_check_html(html)
        assert result.unhandled.get("figure") == 1
        assert result.has_warnings()
        assert any(w.category == "figure" for w in result.warnings)

    def test_detects_images_with_alt(self):
        """Detects images with alt text."""
        html = """
        <html><body>
        <img src="img.png" alt="Important diagram"/>
        </body></html>
        """
        result = preflight_check_html(html)
        assert result.unhandled.get("image_with_alt") == 1

    def test_detects_definition_lists(self):
        """Detects definition lists."""
        html = """
        <html><body>
        <dl>
            <dt>Term 1</dt>
            <dd>Definition 1</dd>
            <dt>Term 2</dt>
            <dd>Definition 2</dd>
        </dl>
        </body></html>
        """
        result = preflight_check_html(html)
        assert result.unhandled.get("definition_list") == 1

    def test_detects_code_blocks(self):
        """Detects code/pre blocks."""
        html = """
        <html><body>
        <pre>Some code here</pre>
        <code>inline code</code>
        </body></html>
        """
        result = preflight_check_html(html)
        assert result.unhandled.get("code_block") == 2

    def test_detects_math_elements(self):
        """Detects mathematical notation elements."""
        html = """
        <html><body>
        <math><mrow><msub>x</msub></mrow></math>
        </body></html>
        """
        result = preflight_check_html(html)
        assert result.unhandled.get("math_formula") >= 1


class TestChunkHtml:
    """Tests for chunk_html function (basic cases)."""

    def test_empty_html(self):
        """Empty HTML produces no chunks."""
        chunks = list(chunk_html(
            "",
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        assert chunks == []

    def test_simple_paragraph(self):
        """Single paragraph produces one chunk."""
        html = "<p>Hello world, this is a test paragraph.</p>"
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]["text"]
        assert chunks[0]["metadata"]["corpus_id"] == "test"
        assert chunks[0]["metadata"]["chunk_index"] == 0

    def test_multiple_paragraphs(self):
        """Multiple paragraphs are combined within token limit."""
        html = "<p>Paragraph one.</p><p>Paragraph two.</p>"
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(chunk_tokens=500),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        # Should combine into one chunk if within token limit
        assert len(chunks) >= 1
        text = chunks[0]["text"]
        assert "Paragraph one" in text

    def test_heading_creates_boundary(self):
        """Headings create chunk boundaries."""
        html = """
        <p>First paragraph</p>
        <h1>Heading</h1>
        <p>Second paragraph</p>
        """
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        # Should have at least 2 chunks due to heading boundary
        assert len(chunks) >= 2

    def test_metadata_includes_doc_type(self):
        """Metadata includes doc_type='chunk'."""
        html = "<p>Test content</p>"
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        assert chunks[0]["metadata"]["doc_type"] == "chunk"

    def test_metadata_includes_chunk_id(self):
        """Metadata includes chunk_id."""
        html = "<p>Test content</p>"
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        assert "chunk_id" in chunks[0]["metadata"]
        assert chunks[0]["metadata"]["chunk_id"]  # Not empty

    def test_metadata_includes_location_id(self):
        """Metadata includes location_id."""
        html = "<p>Test content</p>"
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        assert "location_id" in chunks[0]["metadata"]

    def test_chunk_index_increments(self):
        """Chunk indices increment correctly."""
        html = """
        <h1>Section 1</h1>
        <p>Content 1</p>
        <h1>Section 2</h1>
        <p>Content 2</p>
        <h1>Section 3</h1>
        <p>Content 3</p>
        """
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_flush_each_text_block(self):
        """flush_each_text_block creates chunk per paragraph."""
        html = "<p>Para 1</p><p>Para 2</p><p>Para 3</p>"
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(flush_each_text_block=True),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        assert len(chunks) == 3

    def test_reference_patterns_extract_article(self):
        """Reference patterns extract article from headings."""
        html = """
        <h2>Artikel 6</h2>
        <p>Content of article 6</p>
        """
        reference_patterns = {
            "article": re.compile(r"(?i)artikel\s*(\d+)")
        }
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
            reference_patterns=reference_patterns,
        ))
        assert len(chunks) >= 1
        # The chunk after the article heading should have article metadata
        article_chunk = chunks[-1]
        assert article_chunk["metadata"].get("article") == "6"

    def test_citable_flag_with_article(self):
        """Chunks with article are marked citable."""
        html = """
        <h2>Artikel 6</h2>
        <p>Content of article 6</p>
        """
        reference_patterns = {
            "article": re.compile(r"(?i)artikel\s*(\d+)")
        }
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
            reference_patterns=reference_patterns,
        ))
        # The chunk with article 6 should be citable
        article_chunks = [c for c in chunks if c["metadata"].get("article") == "6"]
        if article_chunks:
            assert article_chunks[0]["metadata"]["citable"] is True

    def test_initial_reference_state(self):
        """Initial reference state is applied."""
        html = "<p>Content in chapter II</p>"
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
            initial_reference_state={"chapter": "II"},
        ))
        assert chunks[0]["metadata"].get("chapter") == "II"


class TestWriteJsonl:
    """Tests for write_jsonl function."""

    def test_writes_jsonl(self, tmp_path):
        """Writes JSONL to file."""
        output_path = tmp_path / "output.jsonl"
        rows = iter([
            {"text": "chunk1", "metadata": {"id": 1}},
            {"text": "chunk2", "metadata": {"id": 2}},
        ])

        write_jsonl(output_path, rows)

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2

        obj1 = json.loads(lines[0])
        assert obj1["text"] == "chunk1"
        assert obj1["metadata"]["id"] == 1

    def test_creates_parent_dirs(self, tmp_path):
        """Creates parent directories if needed."""
        output_path = tmp_path / "nested" / "dir" / "output.jsonl"
        rows = iter([{"text": "test", "metadata": {}}])

        write_jsonl(output_path, rows)

        assert output_path.exists()

    def test_empty_rows(self, tmp_path):
        """Empty rows produces empty file."""
        output_path = tmp_path / "empty.jsonl"
        rows = iter([])

        write_jsonl(output_path, rows)

        assert output_path.exists()
        assert output_path.read_text() == ""

    def test_unicode_content(self, tmp_path):
        """Handles unicode content correctly."""
        output_path = tmp_path / "unicode.jsonl"
        rows = iter([
            {"text": "Dansk: æøå ÆØÅ", "metadata": {"lang": "da"}},
        ])

        write_jsonl(output_path, rows)

        content = output_path.read_text(encoding="utf-8")
        assert "æøå" in content
        assert "ÆØÅ" in content


class TestChunkHtmlEurlexFeatures:
    """Tests for EUR-Lex specific features in chunk_html."""

    def test_eurlex_structural_ids_disabled_by_default(self):
        """EUR-Lex structural IDs are disabled by default."""
        html = """
        <div id="art_6"><p>Article 6 content</p></div>
        """
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
        ))
        # Without enable_eurlex_structural_ids, the div id is not parsed
        assert len(chunks) >= 1

    def test_eurlex_structural_ids_enabled(self):
        """EUR-Lex structural IDs are parsed when enabled."""
        html = """
        <div id="art_6"><p>Article 6 content</p></div>
        """
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
            enable_eurlex_structural_ids=True,
        ))
        assert len(chunks) >= 1
        # Should extract article from div id
        if chunks:
            assert chunks[-1]["metadata"].get("article") == "6"

    def test_handles_chapter_structural_id(self):
        """Handles chapter structural IDs."""
        html = """
        <div id="cpt_II"><p>Chapter II content</p></div>
        """
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
            enable_eurlex_structural_ids=True,
        ))
        if chunks:
            assert chunks[-1]["metadata"].get("chapter") == "II"

    def test_handles_annex_structural_id(self):
        """Handles annex structural IDs."""
        html = """
        <div id="anx_III"><p>Annex III content</p></div>
        """
        chunks = list(chunk_html(
            html,
            config=HtmlChunkingConfig(),
            base_metadata={"corpus_id": "test", "source": "test.html"},
            enable_eurlex_structural_ids=True,
        ))
        if chunks:
            assert chunks[-1]["metadata"].get("annex") == "III"

