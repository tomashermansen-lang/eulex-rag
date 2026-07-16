"""Tests for src/ingestion/case_law_parser.py — CJEU judgment HTML parser.

Covers:
- Dataclasses (Component 1): SectionType, CaseDocumentType, Paragraph, Section, ParsedJudgment
- Section detection (Component 2): _detect_sections
- Paragraph extraction (Component 3): _extract_paragraphs
- Article reference extraction (Component 4): all three tiers + dedup + CELEX normalization
- Fallback logic (Component 5): unrecognized, partial, empty
- Public entry point (Component 6): parse_judgment_html integration tests
"""

import re
import time
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from src.ingestion.case_law_parser import (
    CaseDocumentType,
    ParsedJudgment,
    Paragraph,
    Section,
    SectionType,
    normalize_regulation_to_celex,
    parse_judgment_html,
    _detect_sections,
    _extract_paragraphs,
    _extract_article_references,
    _extract_rdfa_references,
    _extract_regex_references,
    _extract_legalhtml_references,
    _build_sections,
)
from src.common.config_loader import clear_config_cache

_FIXTURES = Path(__file__).parent / "fixtures" / "case_law"

# Shared corpus mapping for tests
_TEST_MAPPING = {"32016R0679": "gdpr", "32024R1689": "ai_act"}


def _read_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Component 1: Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


class TestParagraph:
    def test_create_with_number(self):
        p = Paragraph(number=42, text="The Court observes...")
        assert p.number == 42
        assert p.text == "The Court observes..."

    def test_create_without_number(self):
        p = Paragraph(number=None, text="Unnumbered text")
        assert p.number is None

    def test_is_frozen(self):
        p = Paragraph(number=1, text="text")
        with pytest.raises(AttributeError):
            p.number = 2


class TestSection:
    def test_paragraph_range_numbered(self):
        paras = (Paragraph(1, "a"), Paragraph(2, "b"), Paragraph(85, "c"))
        s = Section(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=paras,
            raw_html="<p>...</p>",
        )
        assert s.paragraph_range == (1, 85)

    def test_paragraph_range_unnumbered(self):
        paras = (Paragraph(None, "a"), Paragraph(None, "b"))
        s = Section(
            section_type=SectionType.SUMMARY,
            heading="Summary",
            paragraphs=paras,
            raw_html="",
        )
        assert s.paragraph_range is None

    def test_paragraph_range_mixed(self):
        paras = (Paragraph(None, "intro"), Paragraph(1, "a"), Paragraph(5, "b"))
        s = Section(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=paras,
            raw_html="",
        )
        assert s.paragraph_range == (1, 5)

    def test_paragraph_range_non_sequential(self):
        paras = (
            Paragraph(1, "a"),
            Paragraph(2, "b"),
            Paragraph(5, "c"),
            Paragraph(6, "d"),
        )
        s = Section(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=paras,
            raw_html="",
        )
        assert s.paragraph_range == (1, 6)

    def test_is_frozen(self):
        s = Section(
            section_type=SectionType.SUMMARY,
            heading="Summary",
            paragraphs=(),
            raw_html="",
        )
        with pytest.raises(AttributeError):
            s.heading = "Changed"

    def test_section_type_enum(self):
        assert SectionType.SUMMARY.value == "summary"
        assert SectionType.GROUNDS.value == "grounds"
        assert SectionType.OPERATIVE_PART.value == "operative_part"
        assert SectionType.FULL_TEXT.value == "full_text"
        assert SectionType.KEYWORDS.value == "keywords"
        assert SectionType.PARTIES.value == "parties"
        assert SectionType.SUBJECT.value == "subject"
        assert SectionType.COSTS.value == "costs"


class TestParsedJudgment:
    def test_create_with_all_fields(self):
        j = ParsedJudgment(
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            sections=(),
            articles_interpreted=("gdpr/article:6",),
            document_type=CaseDocumentType.JUDGMENT,
            parse_warnings=(),
        )
        assert j.ecli == "ECLI:EU:C:2020:790"
        assert j.document_type == CaseDocumentType.JUDGMENT

    def test_is_frozen(self):
        j = ParsedJudgment(
            ecli="test",
            case_number="C-1/00",
            sections=(),
            articles_interpreted=(),
            document_type=CaseDocumentType.JUDGMENT,
            parse_warnings=(),
        )
        with pytest.raises(AttributeError):
            j.ecli = "changed"

    def test_document_type_enum(self):
        assert CaseDocumentType.JUDGMENT.value == "judgment"
        assert CaseDocumentType.AG_OPINION.value == "ag_opinion"
        assert CaseDocumentType.ORDER.value == "order"


# ─────────────────────────────────────────────────────────────────────────────
# Component 2: Section Detection
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectSections:
    _PATTERNS = {
        "summary": ["Summary"],
        "grounds": ["Grounds"],
        "operative_part": ["Operative part"],
    }

    def test_single_heading_match(self):
        html = "<html><body><h2>Operative part</h2><p>The ruling.</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        sections = _detect_sections(soup, self._PATTERNS)
        assert len(sections) == 1
        assert sections[0].section_type == "operative_part"

    def test_multiple_headings(self):
        html = """<html><body>
        <h2>Summary</h2><p>Sum text.</p>
        <h2>Grounds</h2><p>Ground text.</p>
        <h2>Operative part</h2><p>Op text.</p>
        </body></html>"""
        soup = BeautifulSoup(html, "html.parser")
        sections = _detect_sections(soup, self._PATTERNS)
        assert len(sections) == 3
        types = [s.section_type for s in sections]
        assert types == ["summary", "grounds", "operative_part"]

    def test_case_insensitive_match(self):
        html = "<html><body><h2>GROUNDS</h2><p>Text.</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        sections = _detect_sections(soup, self._PATTERNS)
        assert len(sections) == 1
        assert sections[0].section_type == "grounds"

    def test_duplicate_heading_merge(self, caplog):
        import logging

        html = """<html><body>
        <h2>Grounds</h2><p>First part.</p>
        <h2>Grounds</h2><p>Second part.</p>
        </body></html>"""
        soup = BeautifulSoup(html, "html.parser")
        with caplog.at_level(logging.WARNING, logger="src.ingestion.case_law_parser"):
            result = _detect_sections(soup, self._PATTERNS)
        # Should merge into one section
        assert len(result) == 1
        assert result[0].section_type == "grounds"
        assert "First part" in result[0].raw_html
        assert "Second part" in result[0].raw_html
        # W3: Assert warning was logged about duplicate heading merge
        assert any(
            "Duplicate" in r.message or "merged" in r.message.lower()
            for r in caplog.records
        )

    def test_nested_h3_not_split(self):
        patterns = {
            "grounds": ["Grounds"],
            "summary": ["Summary", "Admissibility"],
        }
        html = """<html><body>
        <h2>Grounds</h2>
        <p>Main grounds text.</p>
        <h3>Admissibility</h3>
        <p>Sub-heading text that should stay in grounds.</p>
        </body></html>"""
        soup = BeautifulSoup(html, "html.parser")
        sections = _detect_sections(soup, patterns)
        # h3 "Admissibility" is inside h2 "Grounds" — should NOT become separate section
        assert len(sections) == 1
        assert sections[0].section_type == "grounds"
        assert "Sub-heading text" in sections[0].raw_html

    def test_french_heading_match(self):
        """REQ-7: French heading variants are recognized."""
        patterns = {
            "grounds": ["Grounds", "Motifs"],
            "operative_part": ["Operative part", "Dispositif"],
        }
        html = "<html><body><h2>Motifs</h2><p>French text.</p><h2>Dispositif</h2><p>Ruling text.</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        sections = _detect_sections(soup, patterns)
        assert len(sections) == 2
        types = [s.section_type for s in sections]
        assert "grounds" in types
        assert "operative_part" in types

    def test_no_headings_match(self):
        html = "<html><body><p>Just text.</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        sections = _detect_sections(soup, self._PATTERNS)
        assert sections == []

    def test_empty_section_omitted(self):
        """REQ-1 EC2: Sections with no <p> children are omitted."""
        html = """<html><body>
        <h2>Summary</h2>
        <h2>Grounds</h2><p>1. Real content.</p>
        </body></html>"""
        soup = BeautifulSoup(html, "html.parser")
        detected = _detect_sections(soup, self._PATTERNS)
        sections, _ = _build_sections(soup, detected, "test-ecli")
        # Summary has no <p> children, so only grounds should appear
        assert len(sections) == 1
        assert sections[0].section_type == SectionType.GROUNDS

    def test_pre_heading_content_discarded(self):
        html = """<html><body>
        <p>Institutional preamble that should be discarded.</p>
        <h2>Grounds</h2><p>Actual content.</p>
        </body></html>"""
        soup = BeautifulSoup(html, "html.parser")
        sections = _detect_sections(soup, self._PATTERNS)
        assert len(sections) == 1
        assert "preamble" not in sections[0].raw_html


# ─────────────────────────────────────────────────────────────────────────────
# Component 3: Paragraph Extraction
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractParagraphs:
    def test_numbered_paragraphs(self):
        html = "<p>42. The Court observes that...</p>"
        paragraphs = _extract_paragraphs(html)
        assert len(paragraphs) == 1
        assert paragraphs[0].number == 42
        assert paragraphs[0].text == "The Court observes that..."

    def test_unnumbered_paragraphs(self):
        html = "<p>A general observation.</p>"
        paragraphs = _extract_paragraphs(html)
        assert len(paragraphs) == 1
        assert paragraphs[0].number is None
        assert paragraphs[0].text == "A general observation."

    def test_mixed_paragraphs(self):
        html = "<p>Intro text.</p><p>1. First point.</p><p>2. Second point.</p>"
        paragraphs = _extract_paragraphs(html)
        assert len(paragraphs) == 3
        assert paragraphs[0].number is None
        assert paragraphs[1].number == 1
        assert paragraphs[2].number == 2

    def test_empty_paragraphs_skipped(self):
        html = "<p>  </p><p>Real text.</p><p>\n\t</p>"
        paragraphs = _extract_paragraphs(html)
        assert len(paragraphs) == 1
        assert paragraphs[0].text == "Real text."

    def test_non_sequential_numbers(self):
        html = "<p>1. First.</p><p>2. Second.</p><p>5. Fifth.</p><p>6. Sixth.</p>"
        paragraphs = _extract_paragraphs(html)
        assert [p.number for p in paragraphs] == [1, 2, 5, 6]


# ─────────────────────────────────────────────────────────────────────────────
# Component 4: Article Reference Extraction
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeCelex:
    def test_regulation_normalization(self):
        assert normalize_regulation_to_celex("2016/679", "R") == "32016R0679"

    def test_directive_normalization(self):
        assert normalize_regulation_to_celex("2016/680", "L") == "32016L0680"

    def test_already_celex_format(self):
        assert normalize_regulation_to_celex("32016R0679", "R") == "32016R0679"

    def test_zero_padding(self):
        assert normalize_regulation_to_celex("2016/5", "R") == "32016R0005"

    def test_four_digit_number(self):
        assert normalize_regulation_to_celex("2024/1689", "R") == "32024R1689"


class TestExtractArticleReferences:
    def test_rdfa_extraction(self):
        html = '<span property="eli:cites" resource="http://data.europa.eu/eli/reg/2016/679/art_6">Article 6</span>'
        soup = BeautifulSoup(html, "html.parser")
        refs = _extract_rdfa_references(soup, _TEST_MAPPING)
        assert "gdpr/article:6" in refs

    def test_legalhtml_extraction(self):
        html = '<div is="lh-citation" data-eli="http://data.europa.eu/eli/reg/2016/679/art_46">Article 46</div>'
        soup = BeautifulSoup(html, "html.parser")
        refs = _extract_legalhtml_references(soup, _TEST_MAPPING)
        assert "gdpr/article:46" in refs

    def test_regex_extraction(self):
        text = "Article 6 of Regulation 2016/679 requires a legal basis."
        patterns = (
            re.compile(
                r"Article\s+(\d+)(?:\(\d+\))*(?:\([a-z]\))*\s+of\s+Regulation\s+(?:\(EU\)\s+)?(?:No\s+)?(\d{4}/\d+|\d{4}/\d{4})"
            ),
        )
        refs = _extract_regex_references(text, _TEST_MAPPING, patterns)
        assert "gdpr/article:6" in refs

    def test_deduplication_across_tiers(self):
        # Article 6 appears as both RDFa and plain text
        html = """
        <span property="eli:cites" resource="http://data.europa.eu/eli/reg/2016/679/art_6">Article 6</span>
        <p>Article 6 of Regulation 2016/679 applies.</p>
        """
        soup = BeautifulSoup(html, "html.parser")
        patterns = (
            re.compile(
                r"Article\s+(\d+)(?:\(\d+\))*(?:\([a-z]\))*\s+of\s+Regulation\s+(?:\(EU\)\s+)?(?:No\s+)?(\d{4}/\d+|\d{4}/\d{4})"
            ),
        )
        refs = _extract_article_references(soup, _TEST_MAPPING, patterns)
        assert refs.count("gdpr/article:6") == 1

    def test_unknown_regulation_skipped(self):
        text = "Article 5 of Regulation 2018/1807 provides for free flow of data."
        patterns = (
            re.compile(
                r"Article\s+(\d+)(?:\(\d+\))*(?:\([a-z]\))*\s+of\s+Regulation\s+(?:\(EU\)\s+)?(?:No\s+)?(\d{4}/\d+|\d{4}/\d{4})"
            ),
        )
        refs = _extract_regex_references(text, _TEST_MAPPING, patterns)
        assert refs == []

    def test_malformed_reference_ignored(self):
        text = "Article of Regulation 2016/679 is mentioned."
        patterns = (
            re.compile(
                r"Article\s+(\d+)(?:\(\d+\))*(?:\([a-z]\))*\s+of\s+Regulation\s+(?:\(EU\)\s+)?(?:No\s+)?(\d{4}/\d+|\d{4}/\d{4})"
            ),
        )
        refs = _extract_regex_references(text, _TEST_MAPPING, patterns)
        assert refs == []

    def test_sub_paragraph_stripped(self):
        text = "Article 6(1)(a) of Regulation 2016/679 provides for consent."
        patterns = (
            re.compile(
                r"Article\s+(\d+)(?:\(\d+\))*(?:\([a-z]\))*\s+of\s+Regulation\s+(?:\(EU\)\s+)?(?:No\s+)?(\d{4}/\d+|\d{4}/\d{4})"
            ),
        )
        refs = _extract_regex_references(text, _TEST_MAPPING, patterns)
        assert refs == ["gdpr/article:6"]

    def test_multiple_regulations(self):
        text = (
            "Article 46 of Regulation 2016/679 and "
            "Article 3 of Regulation 2024/1689 both apply."
        )
        patterns = (
            re.compile(
                r"Article\s+(\d+)(?:\(\d+\))*(?:\([a-z]\))*\s+of\s+Regulation\s+(?:\(EU\)\s+)?(?:No\s+)?(\d{4}/\d+|\d{4}/\d{4})"
            ),
        )
        refs = _extract_regex_references(text, _TEST_MAPPING, patterns)
        assert "gdpr/article:46" in refs
        assert "ai_act/article:3" in refs


# ─────────────────────────────────────────────────────────────────────────────
# Components 5-6: Integration Tests (parse_judgment_html)
# ─────────────────────────────────────────────────────────────────────────────


class TestParseJudgmentHtml:
    """Integration tests using fixtures. Each maps to an acceptance scenario."""

    def setup_method(self):
        clear_config_cache()

    def teardown_method(self):
        clear_config_cache()

    def test_standard_judgment(self):
        """AS-1: Standard judgment with summary, grounds, operative part."""
        html = _read_fixture("judgment_standard.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        section_types = {s.section_type for s in result.sections}
        assert SectionType.SUMMARY in section_types
        assert SectionType.GROUNDS in section_types
        assert SectionType.OPERATIVE_PART in section_types
        assert len(result.sections) >= 3
        assert result.parse_warnings == ()

    def test_article_reference_extraction(self):
        """AS-2: Article references extracted from text."""
        html = _read_fixture("judgment_standard.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        assert "gdpr/article:6" in result.articles_interpreted

    def test_paragraph_number_preservation(self):
        """AS-3: Paragraph numbers preserved as integers with range."""
        html = _read_fixture("judgment_standard.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        grounds = [s for s in result.sections if s.section_type == SectionType.GROUNDS][
            0
        ]
        assert grounds.paragraph_range is not None
        first, last = grounds.paragraph_range
        assert first == 1
        assert last == 5
        # Check individual paragraphs have integer numbers
        numbered = [p for p in grounds.paragraphs if p.number is not None]
        assert all(isinstance(p.number, int) for p in numbered)

    def test_unrecognized_html_fallback(self):
        """AS-4: Unrecognized HTML falls back to single full_text section."""
        html = _read_fixture("unrecognized_structure.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:1964:66",
            case_number="6/64",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        assert len(result.sections) == 1
        assert result.sections[0].section_type == SectionType.FULL_TEXT
        assert len(result.parse_warnings) >= 1
        # Warning must mention the document identifier
        assert any("ECLI:EU:C:1964:66" in w for w in result.parse_warnings)

    def test_order_missing_sections(self):
        """AS-5: Order with only operative part — no empty sections."""
        html = _read_fixture("order_minimal.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2023:100",
            case_number="C-100/23",
            document_type="order",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        section_types = [s.section_type for s in result.sections]
        assert SectionType.OPERATIVE_PART in section_types
        # No empty sections for summary or grounds
        assert SectionType.SUMMARY not in section_types
        assert SectionType.GROUNDS not in section_types

    def test_multiple_regulation_references(self):
        """AS-6: References to both GDPR and AI Act extracted."""
        html = _read_fixture("multi_regulation.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2025:1",
            case_number="C-1/25",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        assert "gdpr/article:46" in result.articles_interpreted
        assert "ai_act/article:3" in result.articles_interpreted

    def test_rdfa_deduplication(self):
        """AS-7: Same article via RDFa and text appears only once."""
        html = _read_fixture("judgment_standard.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        count = result.articles_interpreted.count("gdpr/article:6")
        assert count == 1

    def test_no_article_references(self):
        """AS-8: Judgment with no article refs returns empty tuple."""
        html = _read_fixture("no_articles.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2025:2",
            case_number="C-2/25",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        assert result.articles_interpreted == ()

    def test_empty_html(self):
        """Empty HTML returns empty sections with warning."""
        result = parse_judgment_html(
            "",
            ecli="ECLI:EU:C:2025:3",
            case_number="C-3/25",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        assert result.sections == ()
        assert len(result.parse_warnings) >= 1

    def test_partial_recognition(self):
        """One heading matched, remaining content in full_text, no warning."""
        html = _read_fixture("partial_recognition.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2025:4",
            case_number="C-4/25",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        section_types = [s.section_type for s in result.sections]
        assert SectionType.OPERATIVE_PART in section_types
        assert SectionType.FULL_TEXT in section_types
        # No warning for partial recognition
        assert result.parse_warnings == ()

    def test_invalid_document_type(self):
        """ValueError on invalid document_type string (fail-closed)."""
        with pytest.raises(ValueError):
            parse_judgment_html(
                "<html><body><p>text</p></body></html>",
                ecli="test",
                case_number="C-1/00",
                document_type="invalid_type",
                corpus_celex_mapping=_TEST_MAPPING,
            )

    def test_document_type_string_coercion(self):
        """String 'judgment' coerced to CaseDocumentType.JUDGMENT."""
        result = parse_judgment_html(
            "<html><body><p>text</p></body></html>",
            ecli="test",
            case_number="C-1/00",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        assert result.document_type == CaseDocumentType.JUDGMENT

    def test_default_corpus_mapping(self):
        """None corpus_celex_mapping loads default from config."""
        clear_config_cache()
        html = _read_fixture("judgment_standard.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            document_type="judgment",
        )
        # Default config has GDPR mapping
        assert "gdpr/article:6" in result.articles_interpreted

    def test_custom_corpus_mapping(self):
        """Explicit mapping overrides config default."""
        html = _read_fixture("judgment_standard.html")
        # Use a mapping that doesn't include GDPR
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            document_type="judgment",
            corpus_celex_mapping={"32024R1689": "ai_act"},
        )
        # GDPR refs should not appear since mapping excludes it
        assert "gdpr/article:6" not in result.articles_interpreted

    def test_ag_opinion_full_structure(self):
        """REQ-5: AG opinion with full 7-section structure."""
        html = _read_fixture("ag_opinion.html")
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2025:AG1",
            case_number="C-AG/25",
            document_type="ag_opinion",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        assert result.document_type == CaseDocumentType.AG_OPINION
        section_types = {s.section_type for s in result.sections}
        assert SectionType.KEYWORDS in section_types
        assert SectionType.SUMMARY in section_types
        assert SectionType.GROUNDS in section_types
        assert SectionType.OPERATIVE_PART in section_types
        assert len(result.sections) == 7


# ─────────────────────────────────────────────────────────────────────────────
# Performance Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformance:
    @pytest.mark.slow
    def test_regex_no_catastrophic_backtracking(self):
        """Ensure article reference patterns complete in bounded time on adversarial input."""
        adversarial = "Article " + "9(1)" * 100 + " of Regulation 2016/679"
        patterns = [
            re.compile(
                r"Article\s+(\d+)(?:\(\d+\))*(?:\([a-z]\))*\s+of\s+Regulation\s+(?:\(EU\)\s+)?(?:No\s+)?(\d{4}/\d{1,5}|\d{4}/\d{4})"
            ),
            re.compile(
                r"Article\s+(\d+)(?:\(\d+\))*(?:\([a-z]\))*\s+of\s+Directive\s+(?:\(EU\)\s+)?(?:No\s+)?(\d{4}/\d{1,5}|\d{4}/\d{4})"
            ),
            re.compile(
                r"Article\s+(\d+)(?:\(\d+\))*(?:\([a-z]\))*\s+of\s+Decision\s+(?:\(EU\)\s+)?(?:No\s+)?(\d{4}/\d{1,5}|\d{4}/\d{4})"
            ),
        ]
        start = time.monotonic()
        for pattern in patterns:
            list(pattern.finditer(adversarial))
        elapsed = time.monotonic() - start
        assert elapsed < 2.0, (
            f"Regex took {elapsed:.2f}s — possible catastrophic backtracking"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Security Boundary Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSecurityBoundaries:
    def test_script_tag_not_in_paragraph_text(self):
        """Script tag content must not leak into paragraph text."""
        html = """<html><body>
        <h2>Grounds</h2>
        <p>1. Normal text.</p>
        <script>alert('xss')</script>
        <p>2. More text.</p>
        </body></html>"""
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2025:SEC1",
            case_number="C-SEC/25",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        for section in result.sections:
            for para in section.paragraphs:
                assert "alert" not in para.text

    def test_raw_html_preserves_script_tags(self):
        """Raw HTML is untrusted source — script tags are preserved (not sanitized)."""
        html = """<html><body>
        <h2>Grounds</h2>
        <p>1. Normal text.</p>
        <script>alert('xss')</script>
        <p>2. More text.</p>
        </body></html>"""
        result = parse_judgment_html(
            html,
            ecli="ECLI:EU:C:2025:SEC2",
            case_number="C-SEC/25",
            document_type="judgment",
            corpus_celex_mapping=_TEST_MAPPING,
        )
        grounds = [s for s in result.sections if s.section_type == SectionType.GROUNDS]
        assert len(grounds) == 1
        assert "<script>" in grounds[0].raw_html

    def test_oversized_input_rejected(self):
        """HTML exceeding 5MB must raise ValueError."""
        oversized_html = "<html><body><p>" + "x" * 5_000_001 + "</p></body></html>"
        with pytest.raises(ValueError, match="exceeds 5MB limit"):
            parse_judgment_html(
                oversized_html,
                ecli="ECLI:EU:C:2025:SEC3",
                case_number="C-SEC/25",
                document_type="judgment",
                corpus_celex_mapping=_TEST_MAPPING,
            )
