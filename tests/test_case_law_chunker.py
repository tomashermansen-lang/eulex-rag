"""Tests for src/ingestion/case_law_chunker.py — Case law chunking module.

Covers: court derivation, section group resolution, short section merging,
section splitting, paragraph range formatting, heading path/display,
chunk row assembly, chunk_judgment integration, JSONL round-trip.
"""

from __future__ import annotations

import json
import logging

import pytest

from src.common.config_loader import CaseLawChunkingSettings, CaseLawDomainSettings
from src.ingestion.case_law_types import (
    CaseDocumentType,
    Paragraph,
    ParsedJudgment,
    Section,
    SectionType,
)

# Import chunker (will fail until implemented — TDD RED phase)
from src.ingestion.case_law_chunker import (
    chunk_judgment,
    _derive_court,
    _resolve_section_group,
    _merge_short_sections,
    _split_section_into_chunks,
    _format_paragraph_range,
    _build_heading_path,
    _format_heading_display,
    _MergedSection,
    _RawChunk,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def chunking_settings() -> CaseLawChunkingSettings:
    """Default chunking settings (500/100/50 tokens)."""
    return CaseLawChunkingSettings()


@pytest.fixture
def domain_settings() -> CaseLawDomainSettings:
    """Default domain settings with court mapping and section groups."""
    return CaseLawDomainSettings(
        court_mapping={"C": "CJEU", "T": "General Court"},
        section_groups={
            "procedural": ["parties", "subject", "keywords"],
            "substantive": ["summary", "grounds", "costs"],
            "dispositive": ["operative_part"],
            "fallback": ["full_text"],
        },
    )


def _make_paragraph(number: int | None, text: str) -> Paragraph:
    return Paragraph(number=number, text=text)


def _make_section(
    section_type: SectionType,
    paragraphs: list[tuple[int | None, str]],
    heading: str = "",
) -> Section:
    return Section(
        section_type=section_type,
        heading=heading or section_type.value.replace("_", " ").title(),
        paragraphs=tuple(_make_paragraph(n, t) for n, t in paragraphs),
        raw_html="",
    )


def _long_text(tokens_approx: int) -> str:
    """Generate text approximately N tokens long (1 token ≈ 1 word)."""
    return " ".join(f"word{i}" for i in range(tokens_approx))


@pytest.fixture
def schrems_judgment() -> ParsedJudgment:
    """Schrems II judgment fixture with realistic section structure."""
    summary_section = _make_section(
        SectionType.SUMMARY,
        [(None, "This case concerns the validity of data transfers.")],
    )
    grounds_paragraphs = [
        (i, f"Paragraph {i} of the grounds. " * 10) for i in range(1, 86)
    ]
    grounds_section = _make_section(SectionType.GROUNDS, grounds_paragraphs)
    operative_section = _make_section(
        SectionType.OPERATIVE_PART,
        [(None, "The Court hereby rules that the Privacy Shield is invalid.")],
    )
    return ParsedJudgment(
        ecli="ECLI:EU:C:2020:790",
        case_number="C-311/18",
        sections=(summary_section, grounds_section, operative_section),
        articles_interpreted=("gdpr/article:46", "gdpr/article:49"),
        document_type=CaseDocumentType.JUDGMENT,
        parse_warnings=(),
    )


@pytest.fixture
def short_judgment() -> ParsedJudgment:
    """Judgment with all sections below min_chunk_tokens threshold."""
    return ParsedJudgment(
        ecli="ECLI:EU:C:2021:100",
        case_number="C-100/21",
        sections=(
            _make_section(SectionType.KEYWORDS, [(None, "Short keywords text.")]),
            _make_section(SectionType.SUBJECT, [(None, "Short subject text.")]),
            _make_section(SectionType.GROUNDS, [(1, "Short grounds.")]),
            _make_section(SectionType.OPERATIVE_PART, [(None, "The Court rules.")]),
        ),
        articles_interpreted=(),
        document_type=CaseDocumentType.JUDGMENT,
        parse_warnings=(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# T4.1: Court derivation from ECLI
# ─────────────────────────────────────────────────────────────────────────────


class TestDeriveCourt:
    def test_cjeu(self, domain_settings):
        """T4.1.1: ECLI:EU:C:2020:790 → CJEU."""
        assert (
            _derive_court("ECLI:EU:C:2020:790", domain_settings.court_mapping) == "CJEU"
        )

    def test_general_court(self, domain_settings):
        """T4.1.2: ECLI:EU:T:2021:906 → General Court."""
        assert (
            _derive_court("ECLI:EU:T:2021:906", domain_settings.court_mapping)
            == "General Court"
        )

    def test_unknown_code_logs_warning(self, domain_settings, caplog):
        """T4.1.3: Unknown court code F logs warning, returns Court (F)."""
        with caplog.at_level(logging.WARNING):
            result = _derive_court("ECLI:EU:F:2010:123", domain_settings.court_mapping)
        assert result == "Court (F)"
        assert "F" in caplog.text

    def test_adversarial_court_code_sanitized(self, domain_settings, caplog):
        """W4: Adversarial ECLI court code is not embedded in output."""
        adversarial = "ECLI:EU:<script>alert(1)</script>:2020:123"
        with caplog.at_level(logging.WARNING):
            result = _derive_court(adversarial, domain_settings.court_mapping)
        # Must NOT embed the raw adversarial code in the output
        assert "<script>" not in result
        assert result == "Court (UNKNOWN)"

    def test_valid_but_unknown_court_code_format(self, domain_settings, caplog):
        """W4: Valid-format unknown code (e.g. 'F') returns Court (F)."""
        with caplog.at_level(logging.WARNING):
            result = _derive_court("ECLI:EU:F:2010:123", domain_settings.court_mapping)
        assert result == "Court (F)"

    def test_long_court_code_sanitized(self, domain_settings, caplog):
        """W4: Court code exceeding 5 chars is rejected."""
        with caplog.at_level(logging.WARNING):
            result = _derive_court("ECLI:EU:TOOLONG:2020:123", domain_settings.court_mapping)
        assert result == "Court (UNKNOWN)"


# ─────────────────────────────────────────────────────────────────────────────
# T4.2: Section group resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveSectionGroup:
    def test_keywords_procedural(self, domain_settings):
        """T4.2.1."""
        assert (
            _resolve_section_group(SectionType.KEYWORDS, domain_settings.section_groups)
            == "procedural"
        )

    def test_grounds_substantive(self, domain_settings):
        """T4.2.2."""
        assert (
            _resolve_section_group(SectionType.GROUNDS, domain_settings.section_groups)
            == "substantive"
        )

    def test_operative_part_dispositive(self, domain_settings):
        """T4.2.3."""
        assert (
            _resolve_section_group(
                SectionType.OPERATIVE_PART, domain_settings.section_groups
            )
            == "dispositive"
        )

    def test_unknown_section_no_group(self):
        """T4.2.4: Unknown section type resolves to no group."""
        result = _resolve_section_group(
            SectionType.FULL_TEXT, {"procedural": ["parties"]}
        )
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# T4.3: Short section merging
# ─────────────────────────────────────────────────────────────────────────────


class TestMergeShortSections:
    def _get_encoding(self):
        from tiktoken import get_encoding

        return get_encoding("cl100k_base")

    def test_short_section_merged_with_next_in_same_group(self, domain_settings):
        """T4.3.1: Section below min_chunk_tokens merged with next."""
        encoding = self._get_encoding()
        sections = [
            _make_section(SectionType.KEYWORDS, [(None, "Short.")]),
            _make_section(SectionType.SUBJECT, [(None, "Also short.")]),
        ]
        result = _merge_short_sections(
            sections, 100, domain_settings.section_groups, encoding
        )
        assert len(result) == 1
        assert result[0].section_type == SectionType.KEYWORDS

    def test_merged_uses_first_section_type(self, domain_settings):
        """T4.3.2: Merged section uses first section's type."""
        encoding = self._get_encoding()
        sections = [
            _make_section(SectionType.PARTIES, [(None, "Short.")]),
            _make_section(SectionType.SUBJECT, [(None, "Also short.")]),
        ]
        result = _merge_short_sections(
            sections, 100, domain_settings.section_groups, encoding
        )
        assert result[0].section_type == SectionType.PARTIES

    def test_no_cross_group_merging(self, domain_settings):
        """T4.3.3: Merging does NOT cross group boundaries."""
        encoding = self._get_encoding()
        sections = [
            _make_section(SectionType.KEYWORDS, [(None, "Short.")]),  # procedural
            _make_section(SectionType.GROUNDS, [(1, "Short grounds.")]),  # substantive
        ]
        result = _merge_short_sections(
            sections, 100, domain_settings.section_groups, encoding
        )
        assert len(result) == 2  # Not merged

    def test_last_short_section_emitted_as_is(self, domain_settings):
        """T4.3.4: Last short section with no successor emitted as-is."""
        encoding = self._get_encoding()
        sections = [
            _make_section(SectionType.OPERATIVE_PART, [(None, "Short ruling.")]),
        ]
        result = _merge_short_sections(
            sections, 100, domain_settings.section_groups, encoding
        )
        assert len(result) == 1

    def test_all_sections_short_pairwise_merge(self, domain_settings):
        """T4.3.5: All sections short — adjacent pairs merged within groups."""
        encoding = self._get_encoding()
        sections = [
            _make_section(SectionType.PARTIES, [(None, "A.")]),
            _make_section(SectionType.SUBJECT, [(None, "B.")]),
            _make_section(SectionType.KEYWORDS, [(None, "C.")]),
        ]
        result = _merge_short_sections(
            sections, 100, domain_settings.section_groups, encoding
        )
        # First two merge (both procedural), third emitted as-is (no cascade)
        assert len(result) == 2

    def test_no_cascade_merging(self, domain_settings):
        """T4.3.6: No cascade — merged section still below min_chunk_tokens not merged with third."""
        encoding = self._get_encoding()
        sections = [
            _make_section(SectionType.SUMMARY, [(None, "A.")]),
            _make_section(SectionType.GROUNDS, [(1, "B.")]),
            _make_section(SectionType.COSTS, [(None, "C.")]),
        ]
        result = _merge_short_sections(
            sections, 100, domain_settings.section_groups, encoding
        )
        # A+B merged (both substantive), C emitted as-is (no cascade)
        assert len(result) == 2

    def test_at_min_not_merged(self, domain_settings):
        """T4.3.7: Section at exactly min_chunk_tokens NOT merged."""
        encoding = self._get_encoding()
        # Create text that is exactly 100 tokens
        text = " ".join(["word"] * 100)
        sections = [
            _make_section(SectionType.SUMMARY, [(None, text)]),
            _make_section(SectionType.GROUNDS, [(1, "Next.")]),
        ]
        token_count = len(encoding.encode(text))
        result = _merge_short_sections(
            sections, token_count, domain_settings.section_groups, encoding
        )
        assert len(result) == 2  # Not merged since at threshold


# ─────────────────────────────────────────────────────────────────────────────
# T4.4: Section splitting
# ─────────────────────────────────────────────────────────────────────────────


class TestSplitSectionIntoChunks:
    def _get_encoding(self):
        from tiktoken import get_encoding

        return get_encoding("cl100k_base")

    def test_split_at_paragraph_boundaries(self):
        """T4.4.1: Section exceeding chunk_tokens split at paragraph boundaries."""
        encoding = self._get_encoding()
        paragraphs = [_make_paragraph(i, _long_text(200)) for i in range(1, 6)]
        merged = _MergedSection(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=tuple(paragraphs),
            source_sections=(),
        )
        chunks = _split_section_into_chunks(
            merged, 500, 50, encoding, non_breaking_abbreviations=["Art."]
        )
        assert len(chunks) > 1

    def test_no_chunk_exceeds_limit(self):
        """T4.4.2: No chunk exceeds chunk_tokens (measured by tiktoken)."""
        encoding = self._get_encoding()
        paragraphs = [_make_paragraph(i, _long_text(200)) for i in range(1, 6)]
        merged = _MergedSection(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=tuple(paragraphs),
            source_sections=(),
        )
        chunks = _split_section_into_chunks(
            merged, 500, 50, encoding, non_breaking_abbreviations=["Art."]
        )
        for chunk in chunks:
            token_count = len(encoding.encode(chunk.text))
            # Sentence-boundary splitting can overshoot by up to one sentence.
            # With overlap_tokens=50, the worst case is chunk_tokens + overlap_tokens.
            assert token_count <= 550, f"Chunk has {token_count} tokens (limit 500+50 overlap)"

    def test_overlap_tokens(self):
        """T4.4.3: Overlap from chunk N prepended to chunk N+1."""
        encoding = self._get_encoding()
        # Use paragraphs of ~200 tokens each so 2-3 fit in 500 then split
        paragraphs = [_make_paragraph(i, _long_text(100)) for i in range(1, 8)]
        merged = _MergedSection(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=tuple(paragraphs),
            source_sections=(),
        )
        chunks = _split_section_into_chunks(
            merged, 500, 50, encoding, non_breaking_abbreviations=["Art."]
        )
        assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
        # The overlap means some text from end of chunk 0 appears in chunk 1
        # Check that chunk 1 starts with overlap content (decoded from last 50 tokens)
        chunk0_tokens = encoding.encode(chunks[0].text)
        overlap_decoded = encoding.decode(chunk0_tokens[-50:])
        assert overlap_decoded in chunks[1].text, "Expected overlap text in next chunk"

    def test_single_short_paragraph(self):
        """T4.4.4: Single-paragraph section shorter than chunk_tokens → one chunk."""
        encoding = self._get_encoding()
        merged = _MergedSection(
            section_type=SectionType.SUMMARY,
            heading="Summary",
            paragraphs=(_make_paragraph(None, "A brief summary."),),
            source_sections=(),
        )
        chunks = _split_section_into_chunks(
            merged, 500, 50, encoding, non_breaking_abbreviations=["Art."]
        )
        assert len(chunks) == 1

    def test_very_long_paragraph_sentence_split(self):
        """T4.4.5: Very long single paragraph split at sentence boundaries."""
        encoding = self._get_encoding()
        # Create a paragraph with many sentences, exceeding 500 tokens
        # Each sentence ~10 words so 150 sentences ≈ 1500 tokens
        sentences = [
            f"This is sentence number {i} about important legal matters in the case."
            for i in range(1, 150)
        ]
        long_text = " ".join(sentences)
        merged = _MergedSection(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=(_make_paragraph(42, long_text),),
            source_sections=(),
        )
        chunks = _split_section_into_chunks(
            merged, 500, 50, encoding, non_breaking_abbreviations=["Art."]
        )
        assert len(chunks) > 1
        # All sub-chunks should have source_paragraph_number = 42
        for c in chunks:
            assert c.source_paragraph_number == 42

    def test_abbreviation_not_split(self):
        """T4.4.6: Sentence split does not break on Art. abbreviation."""
        encoding = self._get_encoding()
        # Text with abbreviation followed by more text
        text = "Art. 6 of the GDPR provides that. " * 20 + "The court found that. " * 50
        merged = _MergedSection(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=(_make_paragraph(1, text),),
            source_sections=(),
        )
        chunks = _split_section_into_chunks(
            merged, 200, 0, encoding, non_breaking_abbreviations=["Art."]
        )
        # Verify no chunk starts with "6 of" (which would mean Art. was split)
        for c in chunks[1:]:
            assert not c.text.lstrip().startswith("6 of"), "Art. was incorrectly split"

    def test_empty_section_skipped(self):
        """T4.4.7: Empty section (no paragraphs) produces no chunks."""
        encoding = self._get_encoding()
        merged = _MergedSection(
            section_type=SectionType.COSTS,
            heading="Costs",
            paragraphs=(),
            source_sections=(),
        )
        chunks = _split_section_into_chunks(
            merged, 500, 50, encoding, non_breaking_abbreviations=["Art."]
        )
        assert len(chunks) == 0

    def test_intra_paragraph_no_overlap(self):
        """T4.4.8: Intra-paragraph sentence splits have no overlap."""
        encoding = self._get_encoding()
        sentences = [f"Sentence {i} is about something." for i in range(1, 80)]
        long_text = " ".join(sentences)
        merged = _MergedSection(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=(_make_paragraph(10, long_text),),
            source_sections=(),
        )
        chunks = _split_section_into_chunks(
            merged, 200, 50, encoding, non_breaking_abbreviations=["Art."]
        )
        assert len(chunks) >= 2, "Need multiple sentence-split chunks"
        for c in chunks:
            assert c.source_paragraph_number == 10
        # Verify no overlapping text between consecutive sentence-split chunks
        for i in range(len(chunks) - 1):
            tail = chunks[i].text[-30:]
            head = chunks[i + 1].text[:30]
            assert tail != head, f"Chunks {i} and {i+1} have overlapping text"

    def test_config_driven_300_tokens(self):
        """T4.4.9: chunk_tokens=300 produces no chunk exceeding 300 tokens."""
        encoding = self._get_encoding()
        # Use text with sentence boundaries so sentence splitting can work
        sentences = [
            f"This is sentence number {i} about legal matters." for i in range(50)
        ]
        long_para = " ".join(sentences)
        paragraphs = [_make_paragraph(i, long_para) for i in range(1, 4)]
        merged = _MergedSection(
            section_type=SectionType.GROUNDS,
            heading="Grounds",
            paragraphs=tuple(paragraphs),
            source_sections=(),
        )
        chunks = _split_section_into_chunks(
            merged, 300, 50, encoding, non_breaking_abbreviations=["Art."]
        )
        for chunk in chunks:
            token_count = len(encoding.encode(chunk.text))
            assert token_count <= 350, f"Chunk has {token_count} tokens (limit 300)"


# ─────────────────────────────────────────────────────────────────────────────
# T4.5: Paragraph range formatting
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatParagraphRange:
    def test_range_42_48(self):
        """T4.5.1: Chunk spanning paragraphs 42–48 returns '42-48'."""
        chunk = _RawChunk(
            text="text",
            paragraphs=tuple(_make_paragraph(i, "x") for i in range(42, 49)),
            section_type=SectionType.GROUNDS,
            chunk_index=0,
            source_paragraph_number=None,
        )
        assert _format_paragraph_range(chunk) == "42-48"

    def test_single_paragraph(self):
        """T4.5.2: Single paragraph 42 returns '42'."""
        chunk = _RawChunk(
            text="text",
            paragraphs=(_make_paragraph(42, "x"),),
            section_type=SectionType.GROUNDS,
            chunk_index=0,
            source_paragraph_number=None,
        )
        assert _format_paragraph_range(chunk) == "42"

    def test_unnumbered(self):
        """T4.5.3: Unnumbered paragraphs return ''."""
        chunk = _RawChunk(
            text="text",
            paragraphs=(_make_paragraph(None, "x"),),
            section_type=SectionType.SUMMARY,
            chunk_index=0,
            source_paragraph_number=None,
        )
        assert _format_paragraph_range(chunk) == ""

    def test_source_paragraph_number(self):
        """T4.5.4: Sub-chunk from sentence split uses source_paragraph_number."""
        chunk = _RawChunk(
            text="text",
            paragraphs=(_make_paragraph(42, "x"),),
            section_type=SectionType.GROUNDS,
            chunk_index=0,
            source_paragraph_number=42,
        )
        assert _format_paragraph_range(chunk) == "42"

    def test_mixed_numbering(self):
        """T4.5.5: Mixed numbering — only numbered paragraphs included."""
        chunk = _RawChunk(
            text="text",
            paragraphs=(
                _make_paragraph(None, "x"),
                _make_paragraph(5, "y"),
                _make_paragraph(6, "z"),
            ),
            section_type=SectionType.GROUNDS,
            chunk_index=0,
            source_paragraph_number=None,
        )
        assert _format_paragraph_range(chunk) == "5-6"


# ─────────────────────────────────────────────────────────────────────────────
# T4.6: Heading path and display
# ─────────────────────────────────────────────────────────────────────────────


class TestHeadingPath:
    def test_build_heading_path(self):
        """T4.6.1: Returns JSON list with section type."""
        result = _build_heading_path(SectionType.GROUNDS)
        assert json.loads(result) == ["section:grounds"]

    def test_display_with_range(self):
        """T4.6.2: Grounds with range 42-48."""
        assert (
            _format_heading_display(SectionType.GROUNDS, "42-48")
            == "Grounds > Paragraphs 42-48"
        )

    def test_display_singular(self):
        """T4.6.3: Grounds with single paragraph."""
        assert (
            _format_heading_display(SectionType.GROUNDS, "42")
            == "Grounds > Paragraph 42"
        )

    def test_display_no_paragraph(self):
        """T4.6.4: Operative Part with no paragraph suffix."""
        assert (
            _format_heading_display(SectionType.OPERATIVE_PART, "") == "Operative Part"
        )


# ─────────────────────────────────────────────────────────────────────────────
# T4.7 & T4.8: chunk_judgment integration
# ─────────────────────────────────────────────────────────────────────────────


class TestChunkJudgment:
    def test_produces_nonempty_results(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.8.1: Standard judgment produces non-empty list of chunk rows."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        assert len(rows) > 0

    def test_each_chunk_has_nonempty_text(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.8.2: Each chunk has non-empty text."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        for row in rows:
            assert row["text"].strip()

    def test_no_cross_group_chunks(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.8.3: Chunks do not span different section groups.

        Short sections within the same group (e.g. summary + grounds, both
        "substantive") may merge — that's expected.  But operative_part
        ("dispositive") must never mix with substantive content.
        """
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        # Verify multiple section types exist
        section_types = {row["metadata"]["section_type"] for row in rows}
        assert len(section_types) > 1, "Expected chunks from multiple sections"
        # Verify operative_part text does NOT appear in substantive chunks
        operative_rows = [
            r for r in rows if r["metadata"]["section_type"] == "operative_part"
        ]
        substantive_rows = [
            r
            for r in rows
            if r["metadata"]["section_type"] in ("summary", "grounds", "costs")
        ]
        assert operative_rows and substantive_rows, (
            "Need both operative and substantive chunks"
        )
        operative_text = operative_rows[0]["text"]
        for s in substantive_rows:
            assert operative_text not in s["text"], (
                "Operative text found in substantive chunk"
            )

    def test_metadata_matches_input(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.8.4: All chunks have matching ecli, case_number, court, decision_date."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        for row in rows:
            m = row["metadata"]
            assert m["ecli"] == "ECLI:EU:C:2020:790"
            assert m["case_number"] == "C-311/18"
            assert m["court"] == "CJEU"
            assert m["decision_date"] == "2020-07-16"

    def test_has_text_and_metadata_keys(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.7.1: Output dict has 'text' and 'metadata' keys."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        for row in rows:
            assert "text" in row
            assert "metadata" in row

    def test_all_required_fields_present(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.7.2: Metadata contains all CaseLawChunkMetadata required fields."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        required = {
            "ecli",
            "case_number",
            "court",
            "decision_date",
            "source_type",
            "articles_interpreted",
            "section_type",
            "paragraph_range",
            "case_name",
            "heading_path",
            "heading_path_display",
            "chunk_id",
            "text_hash",
            "doc_type",
            "chunk_index",
            "schema_version",
            "corpus_id",
        }
        for row in rows:
            assert required.issubset(set(row["metadata"].keys()))

    def test_source_type_cjeu(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.7.3: source_type == 'cjeu_case_law'."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        for row in rows:
            assert row["metadata"]["source_type"] == "cjeu_case_law"

    def test_schema_version(self, schrems_judgment, chunking_settings, domain_settings):
        """T4.7.4: schema_version == 'meta:v1'."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        for row in rows:
            assert row["metadata"]["schema_version"] == "meta:v1"

    def test_chunk_id_contains_case_number(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.7.5: chunk_id contains case-c-311-18."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        for row in rows:
            assert "case-c-311-18" in row["metadata"]["chunk_id"]

    def test_chunk_id_format(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.7.6: chunk_id format chunk:v1/{doc_id}/case-{norm}-{type}/{idx}/{hash}."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        for row in rows:
            cid = row["metadata"]["chunk_id"]
            assert cid.startswith("chunk:v1/")
            parts = cid.split("/")
            assert len(parts) == 5  # chunk:v1, doc_id, location_id, index, hash

    def test_invalid_articles_raises(self, chunking_settings, domain_settings):
        """T4.7.7: Invalid articles_interpreted raises ValueError."""
        judgment = ParsedJudgment(
            ecli="ECLI:EU:C:2020:790",
            case_number="C-311/18",
            sections=(_make_section(SectionType.GROUNDS, [(1, "Some text.")]),),
            articles_interpreted=("invalid_format",),
            document_type=CaseDocumentType.JUDGMENT,
            parse_warnings=(),
        )
        with pytest.raises(ValueError, match="invalid_format"):
            chunk_judgment(
                judgment,
                case_name="Test",
                decision_date="2020-01-01",
                corpus_id="test",
                doc_version="v1",
                source_path="test.html",
                chunking_settings=chunking_settings,
                domain_settings=domain_settings,
            )

    def test_case_name_in_metadata(
        self, schrems_judgment, chunking_settings, domain_settings
    ):
        """T4.7.10: case_name field present in metadata."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )
        for row in rows:
            assert row["metadata"]["case_name"] == "Schrems II"

    def test_settings_injected_directly(self, schrems_judgment):
        """T4.8.5: Settings injected directly (no YAML loading)."""
        custom_settings = CaseLawChunkingSettings(chunk_tokens=300)
        custom_domain = CaseLawDomainSettings(
            court_mapping={"C": "CJEU"},
            section_groups={"substantive": ["grounds"]},
        )
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=custom_settings,
            domain_settings=custom_domain,
        )
        assert len(rows) > 0


# ─────────────────────────────────────────────────────────────────────────────
# T4.9: JSONL round-trip
# ─────────────────────────────────────────────────────────────────────────────


class TestJsonlRoundTrip:
    def test_write_read_roundtrip(
        self, schrems_judgment, chunking_settings, domain_settings, tmp_path
    ):
        """T4.9.1: Write JSONL, read back, verify all metadata fields."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )

        jsonl_path = tmp_path / "test_output.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        read_rows = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                read_rows.append(json.loads(line))

        assert len(read_rows) == len(rows)
        for row in read_rows:
            assert "text" in row
            assert "metadata" in row
            m = row["metadata"]
            assert "ecli" in m
            assert "chunk_id" in m
            assert "schema_version" in m

    def test_articles_interpreted_valid_after_roundtrip(
        self, schrems_judgment, chunking_settings, domain_settings, tmp_path
    ):
        """T4.9.2: articles_interpreted entries valid after round-trip."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )

        jsonl_path = tmp_path / "test_output.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        import re

        pattern = re.compile(r"^[a-z_]+/article:\d+$")
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                entries = json.loads(row["metadata"]["articles_interpreted"])
                for entry in entries:
                    assert pattern.match(entry), f"Invalid entry: {entry}"

    def test_text_nonempty_after_roundtrip(
        self, schrems_judgment, chunking_settings, domain_settings, tmp_path
    ):
        """T4.9.3: Text content non-empty after round-trip."""
        rows = chunk_judgment(
            schrems_judgment,
            case_name="Schrems II",
            decision_date="2020-07-16",
            corpus_id="gdpr",
            doc_version="abc123",
            source_path="data/case_law/schrems.html",
            chunking_settings=chunking_settings,
            domain_settings=domain_settings,
        )

        jsonl_path = tmp_path / "test_output.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                assert row["text"].strip()
