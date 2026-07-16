"""Case law chunking module.

Converts ParsedJudgment output into chunked JSONL rows for vector store
ingestion. Section-aware chunking with short-section merging, paragraph-
boundary splitting, and sentence-level fallback for long paragraphs.

Public API:
    chunk_judgment() — the only public function
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, TypedDict

if TYPE_CHECKING:
    from tiktoken import Encoding

from src.common.config_loader import (
    CaseLawChunkingSettings,
    CaseLawDomainSettings,
    load_settings,
)
from src.common.metadata_schema import (
    CASE_LAW_REQUIRED_FIELDS,
    CaseLawChunkMetadata,
    build_chunk_id,
    compute_text_hash,
    normalize_case_number_for_id,
    stamp_common_metadata,
    validate_articles_interpreted,
    validate_case_name,
    validate_metadata_primitives,
    validate_required_fields,
)
from src.ingestion.case_law_types import (
    Paragraph,
    ParsedJudgment,
    Section,
    SectionType,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────


class ChunkRow(TypedDict):
    text: str
    metadata: CaseLawChunkMetadata


# ─────────────────────────────────────────────────────────────────────────────
# Internal dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _MergedSection:
    section_type: SectionType
    heading: str
    paragraphs: tuple[Paragraph, ...]
    source_sections: tuple[Section, ...]


@dataclass(frozen=True)
class _RawChunk:
    text: str
    paragraphs: tuple[Paragraph, ...]
    section_type: SectionType
    chunk_index: int
    source_paragraph_number: int | None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


_COURT_CODE_RE = re.compile(r"^[A-Z]{1,5}$")


def _derive_court(ecli: str, court_mapping: dict[str, str]) -> str:
    """Extract court code from ECLI and map to human-readable name."""
    parts = ecli.split(":")
    if len(parts) >= 3:
        code = parts[2]
    else:
        code = ecli
    court = court_mapping.get(code)
    if court is None:
        if not _COURT_CODE_RE.fullmatch(code):
            logger.warning("Invalid court code format %r in ECLI %r", code, ecli)
            return "Court (UNKNOWN)"
        logger.warning("Unknown court code %r in ECLI %r", code, ecli)
        return f"Court ({code})"
    return court


def _resolve_section_group(
    section_type: SectionType,
    section_groups: dict[str, list[str]],
) -> str | None:
    """Return the group name for a section type, or None if not found."""
    value = section_type.value
    for group_name, members in section_groups.items():
        if value in members:
            return group_name
    return None


def _count_tokens(text: str, encoding: Encoding) -> int:
    """Count tokens using tiktoken encoding."""
    return len(encoding.encode(text))


def _merge_short_sections(
    sections: Sequence[Section],
    min_tokens: int,
    section_groups: dict[str, list[str]],
    encoding: Encoding,
) -> list[_MergedSection]:
    """Single-pass merge of consecutive sections in same group when below min_tokens."""
    if not sections:
        return []

    result: list[_MergedSection] = []
    i = 0
    while i < len(sections):
        section = sections[i]
        section_text = " ".join(p.text for p in section.paragraphs)
        tokens = _count_tokens(section_text, encoding)

        # Check if below threshold and can merge with next
        if tokens < min_tokens and i + 1 < len(sections):
            next_section = sections[i + 1]
            my_group = _resolve_section_group(section.section_type, section_groups)
            next_group = _resolve_section_group(
                next_section.section_type, section_groups
            )

            if my_group is not None and my_group == next_group:
                # Merge with next section
                merged_paras = section.paragraphs + next_section.paragraphs
                result.append(
                    _MergedSection(
                        section_type=section.section_type,
                        heading=section.heading,
                        paragraphs=merged_paras,
                        source_sections=(section, next_section),
                    )
                )
                i += 2
                continue

        # Emit as-is
        result.append(
            _MergedSection(
                section_type=section.section_type,
                heading=section.heading,
                paragraphs=section.paragraphs,
                source_sections=(section,),
            )
        )
        i += 1

    return result


def _split_sentences(text: str, non_breaking_abbreviations: list[str]) -> list[str]:
    """Split text into sentences, respecting non-breaking abbreviations."""
    # Split on sentence-ending punctuation followed by whitespace
    raw_segments = re.split(r"(?<=[.!?])\s+", text)

    if len(raw_segments) <= 1:
        return raw_segments

    # Rejoin segments where the left ends with a known abbreviation
    merged: list[str] = [raw_segments[0]]
    for segment in raw_segments[1:]:
        prev = merged[-1]
        should_rejoin = False
        for abbr in non_breaking_abbreviations:
            if prev.endswith(abbr):
                should_rejoin = True
                break
        if should_rejoin:
            merged[-1] = prev + " " + segment
        else:
            merged.append(segment)

    return merged


def _split_long_paragraph_into_chunks(
    para: Paragraph,
    section_type: SectionType,
    chunk_tokens: int,
    encoding: Encoding,
    non_breaking_abbreviations: list[str],
    start_chunk_index: int,
) -> list[_RawChunk]:
    """Split a single long paragraph at sentence boundaries."""
    sentences = _split_sentences(para.text, non_breaking_abbreviations)
    chunks: list[_RawChunk] = []
    chunk_index = start_chunk_index
    sent_texts: list[str] = []
    sent_tokens = 0

    for sent in sentences:
        st = _count_tokens(sent, encoding)
        if sent_tokens + st > chunk_tokens and sent_texts:
            chunks.append(
                _RawChunk(
                    text=" ".join(sent_texts),
                    paragraphs=(para,),
                    section_type=section_type,
                    chunk_index=chunk_index,
                    source_paragraph_number=para.number,
                )
            )
            chunk_index += 1
            sent_texts = []
            sent_tokens = 0

        sent_texts.append(sent)
        sent_tokens += st

    if sent_texts:
        chunks.append(
            _RawChunk(
                text=" ".join(sent_texts),
                paragraphs=(para,),
                section_type=section_type,
                chunk_index=chunk_index,
                source_paragraph_number=para.number,
            )
        )

    return chunks


def _compute_overlap_text(
    texts: list[str],
    overlap_tokens: int,
    encoding: Encoding,
) -> str:
    """Compute the overlap text from the end of the emitted chunk."""
    if overlap_tokens <= 0:
        return ""
    emitted_text = "\n\n".join(texts)
    emitted_token_ids = encoding.encode(emitted_text)
    if len(emitted_token_ids) > overlap_tokens:
        return encoding.decode(emitted_token_ids[-overlap_tokens:])
    return emitted_text


def _make_paragraph_chunk(
    texts: list[str],
    paras: list[Paragraph],
    section_type: SectionType,
    chunk_index: int,
) -> _RawChunk | None:
    """Create a chunk from accumulated paragraph texts, or None if empty."""
    text = "\n\n".join(texts)
    if not text.strip():
        return None
    return _RawChunk(
        text=text,
        paragraphs=tuple(paras),
        section_type=section_type,
        chunk_index=chunk_index,
        source_paragraph_number=None,
    )


def _split_section_into_chunks(
    merged: _MergedSection,
    chunk_tokens: int,
    overlap_tokens: int,
    encoding: Encoding,
    *,
    non_breaking_abbreviations: list[str],
) -> list[_RawChunk]:
    """Split a merged section into chunks at paragraph boundaries."""
    if not merged.paragraphs:
        return []

    chunks: list[_RawChunk] = []
    chunk_index = 0
    current_texts: list[str] = []
    current_paras: list[Paragraph] = []
    current_tokens = 0

    for para in merged.paragraphs:
        para_tokens = _count_tokens(para.text, encoding)

        # Very long paragraph — split at sentence boundaries
        if para_tokens > chunk_tokens:
            chunk = _make_paragraph_chunk(
                current_texts, current_paras, merged.section_type, chunk_index
            )
            if chunk:
                chunks.append(chunk)
                chunk_index += 1
            current_texts = []
            current_paras = []
            current_tokens = 0

            sub_chunks = _split_long_paragraph_into_chunks(
                para,
                merged.section_type,
                chunk_tokens,
                encoding,
                non_breaking_abbreviations,
                chunk_index,
            )
            chunks.extend(sub_chunks)
            chunk_index += len(sub_chunks)
            continue

        # Normal paragraph — accumulate
        if current_tokens + para_tokens > chunk_tokens and current_texts:
            chunk = _make_paragraph_chunk(
                current_texts, current_paras, merged.section_type, chunk_index
            )
            if chunk:
                chunks.append(chunk)
                chunk_index += 1
            overlap_text = _compute_overlap_text(
                current_texts, overlap_tokens, encoding
            )
            current_texts = []
            current_paras = []
            current_tokens = 0

            if overlap_text:
                current_texts.append(overlap_text)
                current_tokens = _count_tokens(overlap_text, encoding)

        current_texts.append(para.text)
        current_paras.append(para)
        current_tokens += para_tokens

    # Flush remaining
    chunk = _make_paragraph_chunk(
        current_texts, current_paras, merged.section_type, chunk_index
    )
    if chunk:
        chunks.append(chunk)

    return chunks


def _format_paragraph_range(chunk: _RawChunk) -> str:
    """Format the paragraph range string for a chunk."""
    if chunk.source_paragraph_number is not None:
        return str(chunk.source_paragraph_number)

    numbered = [p.number for p in chunk.paragraphs if p.number is not None]
    if not numbered:
        return ""
    if len(numbered) == 1:
        return str(numbered[0])
    return f"{numbered[0]}-{numbered[-1]}"


def _build_heading_path(section_type: SectionType) -> str:
    """Return JSON-serialized heading path list."""
    return json.dumps([f"section:{section_type.value}"])


def _format_heading_display(section_type: SectionType, paragraph_range: str) -> str:
    """Format human-readable heading display string."""
    label = section_type.value.replace("_", " ").title()
    if not paragraph_range:
        return label
    if "-" in paragraph_range:
        return f"{label} > Paragraphs {paragraph_range}"
    return f"{label} > Paragraph {paragraph_range}"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def chunk_judgment(
    judgment: ParsedJudgment,
    *,
    case_name: str,
    decision_date: str,
    corpus_id: str,
    doc_version: str,
    source_path: str,
    language: str = "en",
    chunking_settings: CaseLawChunkingSettings | None = None,
    domain_settings: CaseLawDomainSettings | None = None,
) -> list[ChunkRow]:
    """Convert ParsedJudgment to JSONL-ready dicts.

    Each dict has {"text": ..., "metadata": {...}} matching
    the contract consumed by index_jsonl().

    Args:
        language: Default "en" matches EUR-Lex English fetch.
            Callers override for non-English documents.
            Passed to stamp_common_metadata.
    """
    # Lazy import tiktoken (same pattern as html_chunks.py)
    from tiktoken import get_encoding

    encoding = get_encoding("cl100k_base")

    # Resolve settings
    if chunking_settings is None or domain_settings is None:
        settings = load_settings()
        if chunking_settings is None:
            chunking_settings = settings.case_law.chunking
        if domain_settings is None:
            domain_settings = settings.case_law.domain

    if chunking_settings is None:
        raise ValueError("chunking_settings resolved to None")
    if domain_settings is None:
        raise ValueError("domain_settings resolved to None")

    # Validate case_name early (fail-fast, REQ-1)
    validate_case_name(case_name)

    # Validate articles_interpreted early (fail-fast)
    validate_articles_interpreted(judgment.articles_interpreted)

    # Derive court from ECLI
    court = _derive_court(judgment.ecli, domain_settings.court_mapping)

    # Normalize case number for chunk ID
    normalized_case = normalize_case_number_for_id(judgment.case_number)

    # Build doc_id from ECLI (use the full ECLI as doc identifier)
    doc_id = judgment.ecli

    # Serialize articles_interpreted as JSON string for metadata
    articles_json = json.dumps(list(judgment.articles_interpreted))

    # Merge short sections
    merged_sections = _merge_short_sections(
        list(judgment.sections),
        chunking_settings.min_chunk_tokens,
        domain_settings.section_groups,
        encoding,
    )

    # Split and build rows
    rows: list[ChunkRow] = []
    global_chunk_index = 0

    for merged in merged_sections:
        raw_chunks = _split_section_into_chunks(
            merged,
            chunking_settings.chunk_tokens,
            chunking_settings.overlap_tokens,
            encoding,
            non_breaking_abbreviations=chunking_settings.non_breaking_abbreviations,
        )

        for chunk in raw_chunks:
            paragraph_range = _format_paragraph_range(chunk)
            heading_path = _build_heading_path(chunk.section_type)
            heading_display = _format_heading_display(
                chunk.section_type, paragraph_range
            )
            text_hash = compute_text_hash(chunk.text)

            location_id = f"case-{normalized_case}-{chunk.section_type.value}"
            chunk_id = build_chunk_id(
                doc_id=doc_id,
                location_id=location_id,
                chunk_index=global_chunk_index,
                text_hash=text_hash,
            )

            metadata: dict[str, Any] = {
                "ecli": judgment.ecli,
                "case_number": judgment.case_number,
                "court": court,
                "decision_date": decision_date,
                "section_type": chunk.section_type.value,
                "paragraph_range": paragraph_range,
                "articles_interpreted": articles_json,
                "case_name": case_name,
                "heading_path": heading_path,
                "heading_path_display": heading_display,
                "source_type": "cjeu_case_law",
                "doc_type": "chunk",
                "chunk_index": global_chunk_index,
                "chunk_id": chunk_id,
                "text_hash": text_hash,
            }

            stamp_common_metadata(
                metadata,
                corpus_id=corpus_id,
                doc_id=doc_id,
                doc_version=doc_version,
                language=language,
                source_type="cjeu_case_law",
                source_path=source_path,
                source="case_law_chunker",
            )

            validate_required_fields(metadata, CASE_LAW_REQUIRED_FIELDS)
            validate_metadata_primitives(metadata)

            row: ChunkRow = {"text": chunk.text, "metadata": metadata}  # type: ignore[typeddict-item]
            rows.append(row)
            global_chunk_index += 1

    return rows
