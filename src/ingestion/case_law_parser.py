"""CJEU case law HTML parser.

Parses EUR-Lex judgment HTML into a frozen ParsedJudgment dataclass containing
typed sections, paragraph numbers, and article references. Pure function — no I/O
except config loading via the standard config loader.

Public API:
    parse_judgment_html(html, *, ecli, case_number, document_type, corpus_celex_mapping)
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from typing import Mapping, NamedTuple

from bs4 import BeautifulSoup
from bs4.element import Tag

from ..common.config_loader import get_case_law_settings

# Re-export types so consumers can import from the parser module directly
from .case_law_types import (  # noqa: F401
    CaseDocumentType,
    Paragraph,
    ParsedJudgment,
    Section,
    SectionType,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Component 2: Section Detection
# ─────────────────────────────────────────────────────────────────────────────


class DetectedSection(NamedTuple):
    section_type: str
    heading: str
    raw_html: str


_PARAGRAPH_NUMBER_RE = re.compile(r"^\s*(\d+)\.\s+(.*)", re.DOTALL)


def _build_pattern_lookup(heading_patterns: dict[str, list[str]]) -> dict[str, str]:
    """Build lowercase heading text → section_type lookup from config patterns."""
    lookup: dict[str, str] = {}
    for section_type, variants in heading_patterns.items():
        for variant in variants:
            lookup[variant.strip().lower()] = section_type
    return lookup


def _match_headings(
    headings: list[Tag],
    pattern_lookup: dict[str, str],
) -> list[tuple[str, str, Tag]]:
    """Match h2/h3 tags against pattern lookup. h3s only match before any h2."""
    has_matched_h2 = False
    matched: list[tuple[str, str, Tag]] = []

    for tag in headings:
        text = tag.get_text(strip=True)
        section_type = pattern_lookup.get(text.lower())
        if section_type is None:
            continue
        if tag.name == "h3" and has_matched_h2:
            continue
        if tag.name == "h2":
            has_matched_h2 = True
        matched.append((section_type, text, tag))

    return matched


def _collect_content_between(tag: Tag, next_tag: Tag | None) -> str:
    """Collect all sibling HTML between tag and next_tag."""
    parts: list[str] = []
    sibling = tag.next_sibling
    while sibling is not None:
        if sibling is next_tag:
            break
        parts.append(str(sibling))
        sibling = sibling.next_sibling
    return "".join(parts)


def _detect_sections(
    soup: BeautifulSoup,
    heading_patterns: dict[str, list[str]],
) -> list[DetectedSection]:
    """Split HTML into sections by matching h2/h3 heading text against patterns.

    Returns list of DetectedSection tuples. Content before the first matched
    heading is discarded (institutional preamble).
    """
    pattern_lookup = _build_pattern_lookup(heading_patterns)
    headings = soup.find_all(["h2", "h3"])
    matched = _match_headings(headings, pattern_lookup)

    if not matched:
        return []

    sections_by_type: OrderedDict[str, DetectedSection] = OrderedDict()

    for i, (section_type, heading_text, tag) in enumerate(matched):
        next_tag = matched[i + 1][2] if i + 1 < len(matched) else None
        raw_html = _collect_content_between(tag, next_tag)

        if section_type in sections_by_type:
            logger.warning(
                "Duplicate section heading '%s' merged in document", section_type
            )
            existing = sections_by_type[section_type]
            sections_by_type[section_type] = DetectedSection(
                section_type=section_type,
                heading=existing.heading,
                raw_html=existing.raw_html + raw_html,
            )
        else:
            sections_by_type[section_type] = DetectedSection(
                section_type=section_type,
                heading=heading_text,
                raw_html=raw_html,
            )

    return list(sections_by_type.values())


# ─────────────────────────────────────────────────────────────────────────────
# Component 3: Paragraph Extraction
# ─────────────────────────────────────────────────────────────────────────────


def _extract_paragraphs(section_html: str) -> list[Paragraph]:
    """Parse <p> elements, extract leading numbers, return Paragraph list."""
    section_soup = BeautifulSoup(section_html, "html.parser")
    paragraphs: list[Paragraph] = []

    for p_tag in section_soup.find_all("p"):
        text = p_tag.get_text(strip=True)
        if not text:
            continue

        match = _PARAGRAPH_NUMBER_RE.match(text)
        if match:
            paragraphs.append(
                Paragraph(number=int(match.group(1)), text=match.group(2).strip())
            )
        else:
            paragraphs.append(Paragraph(number=None, text=text))

    return paragraphs


# ─────────────────────────────────────────────────────────────────────────────
# Component 4: Article Reference Extraction
# ─────────────────────────────────────────────────────────────────────────────


def normalize_regulation_to_celex(reg_id: str, type_letter: str) -> str:
    """Convert regulation identifier to CELEX format.

    Args:
        reg_id: Regulation ID like "2016/679" or already CELEX like "32016R0679".
        type_letter: R for Regulation, L for Directive, D for Decision.

    Returns:
        CELEX code like "32016R0679".
    """
    # Already CELEX format
    if _CELEX_FORMAT_RE.match(reg_id):
        return reg_id

    parts = reg_id.split("/")
    if len(parts) != 2:
        return reg_id

    year, number = parts
    return f"3{year}{type_letter}{number.zfill(4)}"


_CELEX_FORMAT_RE = re.compile(r"^3\d{4}[RLD]\d{4}$")
_ELI_URI_RE = re.compile(r"/eli/(?:reg|dir|dec)/(\d{4})/(\d{1,5})/art_(\d+)")


def _resolve_eli_uri(
    uri: str,
    corpus_celex_mapping: Mapping[str, str],
) -> str | None:
    """Resolve an ELI URI to a corpus article reference.

    Parses the year, number, and article from the URI, determines the legislation
    type letter, normalizes to CELEX, and looks up the corpus.

    Returns:
        A string like ``"gdpr/article:6"`` or ``None`` if unmapped.
    """
    eli_match = _ELI_URI_RE.search(uri)
    if not eli_match:
        return None

    year, number, article = eli_match.groups()
    if "/reg/" in uri:
        type_letter = "R"
    elif "/dir/" in uri:
        type_letter = "L"
    else:
        type_letter = "D"

    celex = normalize_regulation_to_celex(f"{year}/{number}", type_letter)
    corpus = corpus_celex_mapping.get(celex)
    if corpus:
        return f"{corpus}/article:{article}"
    return None


def _extract_rdfa_references(
    soup: BeautifulSoup,
    corpus_celex_mapping: Mapping[str, str],
) -> list[str]:
    """Extract from <span property="eli:cites" resource="..."> elements."""
    refs: list[str] = []
    for span in soup.find_all("span", attrs={"property": "eli:cites"}):
        resource = span.get("resource", "")
        if not resource:
            continue
        ref = _resolve_eli_uri(resource, corpus_celex_mapping)
        if ref:
            refs.append(ref)
        else:
            logger.debug("RDFa reference to unmapped ELI URI — skipped: %s", resource)

    return refs


def _extract_legalhtml_references(
    soup: BeautifulSoup,
    corpus_celex_mapping: Mapping[str, str],
) -> list[str]:
    """Extract from <div is="lh-citation" data-eli="..."> elements."""
    refs: list[str] = []
    for elem in soup.find_all(attrs={"data-eli": True}):
        eli_uri = elem.get("data-eli", "")
        if not eli_uri:
            continue
        ref = _resolve_eli_uri(eli_uri, corpus_celex_mapping)
        if ref:
            refs.append(ref)
        else:
            logger.debug(
                "LegalHTML reference to unmapped ELI URI — skipped: %s", eli_uri
            )

    return refs


def _extract_regex_references(
    text: str,
    corpus_celex_mapping: Mapping[str, str],
    patterns: tuple[re.Pattern[str], ...],
) -> list[str]:
    """Extract article references from plain text using pre-compiled regex patterns."""
    refs: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            article_number = match.group(1)
            reg_id = match.group(2)

            # Determine type letter from the pattern's matched text
            matched_text = match.group(0)
            if "Regulation" in matched_text:
                type_letter = "R"
            elif "Directive" in matched_text:
                type_letter = "L"
            else:
                type_letter = "R"  # default
                logger.debug(
                    "Unrecognized legislation type in '%s', defaulting to Regulation",
                    matched_text[:80],
                )

            celex = normalize_regulation_to_celex(reg_id, type_letter)
            corpus = corpus_celex_mapping.get(celex)
            if corpus:
                refs.append(f"{corpus}/article:{article_number}")
            else:
                logger.debug("Regex reference to unmapped CELEX %s — skipped", celex)

    return refs


def _extract_article_references(
    soup: BeautifulSoup,
    corpus_celex_mapping: Mapping[str, str],
    regex_patterns: tuple[re.Pattern[str], ...],
) -> list[str]:
    """Orchestrate 3-tier extraction. Returns deduplicated reference list."""
    # Collect from all three tiers
    rdfa_refs = _extract_rdfa_references(soup, corpus_celex_mapping)
    legalhtml_refs = _extract_legalhtml_references(soup, corpus_celex_mapping)
    text = soup.get_text()
    regex_refs = _extract_regex_references(text, corpus_celex_mapping, regex_patterns)

    # Deduplicate preserving insertion order
    seen: set[str] = set()
    deduped: list[str] = []
    for ref in rdfa_refs + legalhtml_refs + regex_refs:
        if ref not in seen:
            seen.add(ref)
            deduped.append(ref)

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Component 5: Section Assembly (from detected headings or fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _build_fulltext_fallback(
    soup: BeautifulSoup,
    ecli: str,
) -> tuple[tuple[Section, ...], list[str]]:
    """Build a single FULL_TEXT section when no headings matched."""
    logger.warning(
        "No recognized section headings in %s — falling back to full_text", ecli
    )
    warnings = [f"No recognized section headings in {ecli} — full text fallback"]
    html_str = str(soup)
    paragraphs = _extract_paragraphs(html_str)
    if not paragraphs:
        return (), warnings
    return (
        Section(
            section_type=SectionType.FULL_TEXT,
            heading="",
            paragraphs=tuple(paragraphs),
            raw_html=html_str,
        ),
    ), warnings


def _collect_unmatched_content(
    soup: BeautifulSoup,
    detected: list[DetectedSection],
) -> str:
    """Collect HTML content under unmatched h2 headings for partial recognition."""
    matched_heading_texts = {d.heading.strip().lower() for d in detected}
    parts: list[str] = []

    for tag in soup.find_all(["h2", "h3"]):
        if tag.name != "h2":
            continue
        if tag.get_text(strip=True).strip().lower() in matched_heading_texts:
            continue
        content: list[str] = []
        sibling = tag.next_sibling
        while sibling is not None:
            if hasattr(sibling, "name") and sibling.name in ("h2", "h3"):
                break
            content.append(str(sibling))
            sibling = sibling.next_sibling
        if content:
            parts.append(str(tag) + "".join(content))

    return "".join(parts)


def _build_sections(
    soup: BeautifulSoup,
    detected: list[DetectedSection],
    ecli: str,
) -> tuple[tuple[Section, ...], list[str]]:
    """Build typed Section tuple from detected sections, with fallback."""
    if not detected:
        return _build_fulltext_fallback(soup, ecli)

    section_list: list[Section] = []
    for det in detected:
        paragraphs = _extract_paragraphs(det.raw_html)
        if not paragraphs:
            continue
        section_list.append(
            Section(
                section_type=SectionType(det.section_type),
                heading=det.heading,
                paragraphs=tuple(paragraphs),
                raw_html=det.raw_html,
            )
        )

    # Partial recognition: collect unmatched h2 content into FULL_TEXT
    unmatched_html = _collect_unmatched_content(soup, detected)
    if unmatched_html:
        unmatched_paragraphs = _extract_paragraphs(unmatched_html)
        if unmatched_paragraphs:
            section_list.append(
                Section(
                    section_type=SectionType.FULL_TEXT,
                    heading="",
                    paragraphs=tuple(unmatched_paragraphs),
                    raw_html=unmatched_html,
                )
            )

    return tuple(section_list), []


# ─────────────────────────────────────────────────────────────────────────────
# Component 6: Public Entry Point
# ─────────────────────────────────────────────────────────────────────────────


def parse_judgment_html(
    html: str,
    *,
    ecli: str,
    case_number: str,
    document_type: str | CaseDocumentType,
    corpus_celex_mapping: Mapping[str, str] | None = None,
) -> ParsedJudgment:
    """Parse CJEU judgment HTML into structured data.

    Args:
        html: Raw HTML content (not a file path).
        ecli: ECLI identifier (from caller metadata).
        case_number: Case number (from caller metadata).
        document_type: A CaseDocumentType enum value or its string name
            (e.g. "judgment"). Strings are coerced to CaseDocumentType.
        corpus_celex_mapping: CELEX code -> corpus ID mapping.
            Defaults to ``case_law.corpus_celex_mapping`` from ``settings.yaml``
            when None. Default mapping includes GDPR (32016R0679),
            AI Act (32024R1689), DSA (32022R2065), DMA (32022R1925),
            MiCA (32023R1114), DORA (32022R2554).

    Returns:
        Frozen ParsedJudgment dataclass.

    Raises:
        ValueError: If ``html`` exceeds 5 MB or ``document_type`` is invalid.
    """
    # Step 0: Input size guard
    _MAX_HTML_BYTES = 5_000_000
    if len(html) > _MAX_HTML_BYTES:
        raise ValueError(f"HTML input exceeds 5MB limit ({len(html)} bytes)")

    # Step 1: Load config
    settings = get_case_law_settings()

    # Step 2: Resolve corpus mapping
    if corpus_celex_mapping is None:
        corpus_celex_mapping = settings.corpus_celex_mapping

    # Step 3: Coerce document_type
    if isinstance(document_type, str):
        document_type = CaseDocumentType(document_type)

    # Step 4: Parse HTML
    soup = BeautifulSoup(html, "html.parser")

    # Step 5: Check for empty document
    if not soup.find(string=re.compile(r"\S")):
        logger.warning("Empty HTML document for %s", ecli)
        return ParsedJudgment(
            ecli=ecli,
            case_number=case_number,
            sections=(),
            articles_interpreted=(),
            document_type=document_type,
            parse_warnings=(f"Empty HTML document for {ecli}",),
        )

    # Step 6: Detect sections and build typed output
    detected = _detect_sections(soup, settings.section_heading_patterns)
    sections, section_warnings = _build_sections(soup, detected, ecli)

    # Step 7: Extract article references
    refs = _extract_article_references(
        soup, corpus_celex_mapping, settings.article_reference_patterns
    )

    return ParsedJudgment(
        ecli=ecli,
        case_number=case_number,
        sections=sections,
        articles_interpreted=tuple(refs),
        document_type=document_type,
        parse_warnings=tuple(section_warnings),
    )
