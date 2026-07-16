"""Dataclasses and enums for CJEU case law parsing.

These types form the parser's public output API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SectionType(str, Enum):
    """Valid section types for CJEU judgment sections."""

    SUMMARY = "summary"
    GROUNDS = "grounds"
    OPERATIVE_PART = "operative_part"
    KEYWORDS = "keywords"
    PARTIES = "parties"
    SUBJECT = "subject"
    COSTS = "costs"
    FULL_TEXT = "full_text"


class CaseDocumentType(str, Enum):
    """CJEU document types (named CaseDocumentType to avoid collision
    with eurlex_listing.DocumentType)."""

    JUDGMENT = "judgment"
    AG_OPINION = "ag_opinion"
    ORDER = "order"


@dataclass(frozen=True)
class Paragraph:
    number: int | None  # None if unnumbered
    text: str  # Text content, stripped of leading number


@dataclass(frozen=True)
class Section:
    section_type: SectionType
    heading: str  # Original heading text from HTML
    paragraphs: tuple[Paragraph, ...]
    raw_html: str  # Preserved for downstream processing
    # **Contract:** raw_html is untrusted source HTML preserved for text
    # extraction and downstream processing. It MUST NOT be rendered directly
    # in any browser context without sanitization.

    @property
    def paragraph_range(self) -> tuple[int, int] | None:
        """(first, last) from numbered paragraphs, or None."""
        numbered = [p.number for p in self.paragraphs if p.number is not None]
        if not numbered:
            return None
        return (numbered[0], numbered[-1])


@dataclass(frozen=True)
class ParsedJudgment:
    ecli: str
    case_number: str
    sections: tuple[Section, ...]
    articles_interpreted: tuple[str, ...]  # "{corpus}/article:{id}" format
    document_type: CaseDocumentType
    parse_warnings: tuple[str, ...]
