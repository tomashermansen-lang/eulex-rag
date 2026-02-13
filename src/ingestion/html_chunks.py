"""Generic utilities for extracting and chunking HTML into JSONL for ingestion.

Features:
- HTML -> structured blocks (headings + paragraphs/lists)
- stable-boundary chunking (no mid-word splitting from our side)
- optional reference extraction driven ONLY by headings to avoid in-text false positives

The produced JSONL lines have the shape:
  {"text": "...", "metadata": {...}}

Metadata always includes:
- source
- corpus_id
- doc_type="chunk"
- chunk_index
- chunk_id

HTML chunks typically do not include a page number.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Mapping, Pattern

from ..common.metadata_schema import (
    build_chunk_id as _build_chunk_id,
    compute_text_hash,
    make_heading_path,
    make_location_id as _make_location_id,
    validate_metadata_primitives,
)


ReferencePatterns = Mapping[str, Pattern[str]]


# --- Format Warning System ---
# Detects patterns that the current chunking mechanism can't handle optimally.
# Pipeline continues but warns user about potential issues.


@dataclass(frozen=True)
class FormatWarning:
    """Warning about an unhandled format pattern in the source document."""

    category: str  # e.g., "unclassified_table", "unknown_structure"
    message: str  # Human-readable description
    location: str  # Where in document (e.g., "after BILAG IV")
    severity: str = "warning"  # "warning" or "info"
    suggestion: str = ""  # How to fix or handle


@dataclass
class PreflightResult:
    """Result of pre-flight document analysis before chunking."""

    can_proceed: bool = True  # False if critical issues found
    warnings: list[FormatWarning] = field(default_factory=list)

    # What we found and can handle
    handled: dict[str, int] = field(default_factory=dict)
    # What we found but can't handle
    unhandled: dict[str, int] = field(default_factory=dict)

    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def summary(self) -> str:
        """One-line summary for logging."""
        parts = []
        if self.handled:
            handled_str = ", ".join(f"{v}x {k}" for k, v in self.handled.items())
            parts.append(f"Handled: {handled_str}")
        if self.unhandled:
            unhandled_str = ", ".join(f"{v}x {k}" for k, v in self.unhandled.items())
            parts.append(f"⚠️ Unhandled: {unhandled_str}")
        return " | ".join(parts) if parts else "No structures detected"


def preflight_check_html(html: str, *, enable_eurlex_structural_ids: bool = True) -> PreflightResult:
    """Pre-flight check: Analyze document for patterns we can/cannot handle.

    Run this BEFORE chunking to warn the user about unsupported content.

    Returns:
        PreflightResult with:
        - handled: dict of pattern->count for things we support
        - unhandled: dict of pattern->count for things we DON'T support
        - warnings: detailed warnings for each unhandled pattern
    """
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError:
        return PreflightResult(can_proceed=True)

    soup = BeautifulSoup(html, "html.parser")
    warnings: list[FormatWarning] = []
    handled: dict[str, int] = {}
    unhandled: dict[str, int] = {}

    # --- Check EUR-Lex Tables ---
    if enable_eurlex_structural_ids:
        for table in soup.find_all("table", class_="oj-table"):
            table_type = _classify_eurlex_table(table)
            row_count = len(table.find_all("tr"))

            # Get location context
            prev = table.find_previous(["h1", "h2", "h3", "h4", "p"])
            location = prev.get_text(" ", strip=True)[:50] if prev else "unknown"

            if table_type == "concordance":
                handled["concordance_table"] = handled.get("concordance_table", 0) + 1
            elif table_type == "definitional":
                handled["definitional_table"] = handled.get("definitional_table", 0) + 1
            elif row_count >= 3:  # Only warn for substantial tables
                # Unrecognized table pattern!
                unhandled["unclassified_table"] = unhandled.get("unclassified_table", 0) + 1

                # Get header preview
                first_row = table.find("tr")
                headers = []
                if first_row:
                    for cell in first_row.find_all("td")[:4]:
                        text = cell.get_text(" ", strip=True)[:30]
                        if text:
                            headers.append(text)

                warnings.append(FormatWarning(
                    category="unclassified_table",
                    message=f"Table ({row_count} rows) with unrecognized pattern. Headers: [{' | '.join(headers)}]",
                    location=f"near '{location}'",
                    severity="warning",
                    suggestion="Add table classification pattern to _classify_eurlex_table() in html_chunks.py",
                ))

    # --- Check for other EUR-Lex structures we handle ---
    # Count articles, chapters, annexes
    article_divs = soup.find_all("div", id=re.compile(r"^art_\d+", re.I))
    if article_divs:
        handled["articles"] = len(article_divs)

    chapter_divs = soup.find_all("div", id=re.compile(r"^cpt_", re.I))
    if chapter_divs:
        handled["chapters"] = len(chapter_divs)

    annex_divs = soup.find_all("div", id=re.compile(r"^anx_", re.I))
    if annex_divs:
        handled["annexes"] = len(annex_divs)

    # --- Check for patterns we DON'T handle well ---
    # Deeply nested lists (more than 3 levels)
    deep_lists = soup.find_all("ul")
    for ul in deep_lists:
        depth = len(ul.find_parents("ul"))
        if depth >= 3:
            unhandled["deep_nested_list"] = unhandled.get("deep_nested_list", 0) + 1
            if unhandled["deep_nested_list"] == 1:  # Only warn once
                warnings.append(FormatWarning(
                    category="deep_nested_list",
                    message=f"Found deeply nested list (depth {depth + 1}+). May lose structure in chunking.",
                    location="multiple locations",
                    severity="info",
                    suggestion="Consider flattening lists or adding nested list handler.",
                ))

    # Tables without oj-table class (might be layout tables we're ignoring)
    # EUR-Lex uses many layout tables for formatting - these are NOT data tables
    # We only warn about tables that:
    # 1. Are NOT inside an oj-table (nested layout tables are fine)
    # 2. Have multiple rows with structured data pattern (not just text wrappers)
    other_tables = soup.find_all("table")
    oj_tables_set = set(soup.find_all("table", class_="oj-table"))

    suspicious_tables = 0
    for table in other_tables:
        if table in oj_tables_set:
            continue
        # Skip if inside an oj-table (layout table)
        if table.find_parent("table", class_="oj-table"):
            continue

        rows = table.find_all("tr")
        if len(rows) < 3:
            continue  # Too small to be a data table

        # Check if it has a header-like structure (th elements or styled headers)
        has_headers = bool(table.find("th")) or bool(table.find("td", class_=re.compile(r"hdr|header", re.I)))
        if has_headers:
            suspicious_tables += 1

    if suspicious_tables > 0:
        unhandled["suspicious_table"] = suspicious_tables
        warnings.append(FormatWarning(
            category="suspicious_table",
            message=f"Found {suspicious_tables} table(s) with header structure that may need special handling.",
            location="various",
            severity="warning",
            suggestion="Verify these tables are layout wrappers, not data tables needing row-level extraction.",
        ))

    # --- Check for figures/images ---
    figures = soup.find_all("figure")
    if figures:
        unhandled["figure"] = len(figures)
        # Get some context
        fig_locations = []
        for fig in figures[:3]:
            prev = fig.find_previous(["h1", "h2", "h3", "h4", "p"])
            if prev:
                fig_locations.append(prev.get_text(" ", strip=True)[:30])
        warnings.append(FormatWarning(
            category="figure",
            message=f"Found {len(figures)} <figure> element(s). Visual content may not be extracted.",
            location=f"near: {', '.join(fig_locations)}" if fig_locations else "various",
            severity="warning",
            suggestion="Consider adding figure caption extraction or flagging visuals for manual review.",
        ))

    # Check for images with alt text we might want
    images = soup.find_all("img")
    images_with_alt = [img for img in images if img.get("alt", "").strip()]
    if images_with_alt:
        unhandled["image_with_alt"] = len(images_with_alt)
        warnings.append(FormatWarning(
            category="image_with_alt",
            message=f"Found {len(images_with_alt)} image(s) with alt text. Alt text content is not extracted.",
            location="various",
            severity="info",
            suggestion="Consider extracting alt text as supplementary content.",
        ))

    # --- Check for definition lists (dl/dt/dd) ---
    dl_elements = soup.find_all("dl")
    if dl_elements:
        total_terms = sum(len(dl.find_all("dt")) for dl in dl_elements)
        if total_terms > 0:
            unhandled["definition_list"] = len(dl_elements)
            warnings.append(FormatWarning(
                category="definition_list",
                message=f"Found {len(dl_elements)} definition list(s) with {total_terms} terms. May not preserve dt/dd structure.",
                location="various",
                severity="info",
                suggestion="Verify definition list content is properly chunked.",
            ))

    # --- Check for blockquotes ---
    blockquotes = soup.find_all("blockquote")
    if blockquotes:
        handled["blockquote"] = len(blockquotes)  # We handle these as text

    # --- Check for code/pre blocks (unusual in legal docs but possible) ---
    code_blocks = soup.find_all(["pre", "code"])
    if code_blocks:
        unhandled["code_block"] = len(code_blocks)
        warnings.append(FormatWarning(
            category="code_block",
            message=f"Found {len(code_blocks)} code/pre block(s). Formatting may be lost.",
            location="various",
            severity="info",
            suggestion="Verify code blocks are legal formulas/references that need special handling.",
        ))

    # --- Check for footnotes/endnotes ---
    footnotes = soup.find_all(class_=re.compile(r"footnote|endnote|note", re.I))
    if footnotes:
        handled["footnotes"] = len(footnotes)

    # --- Check for special EUR-Lex elements we might not handle ---
    # Mathematical formulas
    math_elements = soup.find_all(["math", "mrow", "msub", "msup"])
    if math_elements:
        unhandled["math_formula"] = len(math_elements)
        warnings.append(FormatWarning(
            category="math_formula",
            message=f"Found {len(math_elements)} mathematical notation element(s). Math rendering not preserved.",
            location="various",
            severity="warning",
            suggestion="Consider adding MathML text extraction.",
        ))

    # --- Summary of what we found ---
    # Count paragraphs and list items (our bread and butter)
    paragraphs = soup.find_all("p")
    list_items = soup.find_all("li")
    if paragraphs:
        handled["paragraphs"] = len(paragraphs)
    if list_items:
        handled["list_items"] = len(list_items)

    return PreflightResult(
        can_proceed=True,  # We don't block, just warn
        warnings=warnings,
        handled=handled,
        unhandled=unhandled,
    )


def _extract_reference_mentions(
    text: str,
    *,
    reference_state: Mapping[str, str | None] | None = None,
    max_per_kind: int = 25,
) -> dict[str, list[str]]:
    """Extract *mentioned* legal references from text.

    This is distinct from the chunk's structural location (chapter/article) which
    is derived from headings. Mentions are useful for cross-reference navigation
    and more professional citation handling.
    """

    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return {}

    # Keep these intentionally conservative: require explicit keywords.
    patterns: dict[str, Pattern[str]] = {
        "chapter": re.compile(r"(?i)\b(?:kapitel|chapter)\s+([ivxlcdm]+|\d+)\b"),
        "section": re.compile(r"(?i)\b(?:afsnit|afdeling|section)\s+([ivxlcdm]+|\d+)\b"),
        "article": re.compile(r"(?i)\b(?:artikel|article)\s*(\d{1,3}[a-z]?)\b"),
        "paragraph": re.compile(r"(?i)\b(?:stk\.|stykke|paragraph)\s*(\d{1,3})\b"),
        "annex": re.compile(r"(?i)\b(?:bilag|annex)\s+([ivxlcdm]+|\d+)\b"),
    }

    state = dict(reference_state or {})

    def normalize(kind: str, raw: str) -> str:
        v = (raw or "").strip()
        if kind in {"chapter", "annex", "section"}:
            return v.upper()
        if kind == "article":
            return v.upper()
        if kind == "paragraph":
            return v
        return v

    out: dict[str, list[str]] = {}
    for kind, pattern in patterns.items():
        seen: set[str] = set()
        for m in pattern.finditer(text):
            val = normalize(kind, m.group(1))
            if not val:
                continue

            # Drop the current structural location to keep mentions focused on cross-references.
            if kind in {"chapter", "section", "article", "paragraph", "annex"}:
                state_val = state.get(kind)
                if state_val and normalize(kind, str(state_val)) == val:
                    continue

            if val in seen:
                continue
            seen.add(val)
            out.setdefault(kind, []).append(val)
            if len(out[kind]) >= max_per_kind:
                break

    return {k: v for k, v in out.items() if v}


def _slugify(value: str) -> str:
    value = value.strip().lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9\-]+", "", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "doc"


def make_chunk_id(*, corpus_id: str, source: str, chunk_index: int) -> str:
    # Backwards-compatible wrapper (legacy shape). Prefer metadata_schema.build_chunk_id.
    return f"{_slugify(corpus_id)}-{_slugify(source)}-html-c{chunk_index}"


def make_location_id(*, corpus_id: str, reference_state: Mapping[str, str | None]) -> str:
    # Backwards-compatible wrapper. The new canonical id is in metadata_schema.make_location_id.
    _ = corpus_id
    return _make_location_id(reference_state=reference_state)


@dataclass(frozen=True)
class HtmlChunkingConfig:
    chunk_tokens: int = 500
    overlap: int = 100
    flush_each_text_block: bool = False


_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_TEXT_TAGS = ("p", "li")


# --- EUR-Lex Table Handling ---
# EUR-Lex annexes contain tables that need special treatment:
# - Concordance tables (BILAG III): map old→new article numbers, no substantive content
# - Definitional tables (BILAG I/II): sector→subsector→entity definitions


def _classify_eurlex_table(table_node) -> str | None:
    """Classify a EUR-Lex table by its structure.

    Returns:
        - "concordance": Article mapping table (skip from RAG)
        - "definitional": Sector/entity definition table (chunk row-by-row)
        - None: Not a recognized EUR-Lex table structure
    """
    # Find header row
    header_cells = []
    first_row = table_node.find("tr", class_="oj-table")
    if first_row:
        for cell in first_row.find_all("td", class_="oj-table"):
            header_p = cell.find("p", class_="oj-tbl-hdr")
            if header_p:
                header_cells.append(header_p.get_text(" ", strip=True).lower())

    if not header_cells:
        return None

    # Concordance table: "Direktiv (EU) XXXX/XXXX" | "Nærværende direktiv"
    # Pattern: 2 columns, one mentions "direktiv" and a year, other says "nærværende"
    if len(header_cells) == 2:
        header_text = " ".join(header_cells)
        if "direktiv" in header_text and "nærværende" in header_text:
            return "concordance"

    # Definitional table: "Sektor" | "Delsektor" | "Type enhed"
    if len(header_cells) >= 3:
        if "sektor" in header_cells[0] and ("delsektor" in header_cells[1] or "type" in header_cells[2]):
            return "definitional"

    return None


def _extract_definitional_table_rows(table_node) -> Iterator[dict[str, str]]:
    """Extract structured rows from a definitional table (BILAG I/II style).

    Handles rowspan for sectors spanning multiple subsectors.

    Yields dicts with:
        - sector: "1. Energi"
        - subsector: "a) Elektricitet" or ""
        - entity: "Elektricitetsvirksomheder som defineret i..."
    """
    rows = table_node.find_all("tr", class_="oj-table")
    if len(rows) < 2:
        return

    # Skip header row
    data_rows = rows[1:]

    # Track rowspan state for sector/subsector columns
    current_sector = ""
    current_subsector = ""
    sector_rowspan = 0
    subsector_rowspan = 0

    for row in data_rows:
        cells = row.find_all("td", class_="oj-table", recursive=False)
        if not cells:
            continue

        cell_idx = 0
        row_sector = current_sector
        row_subsector = current_subsector
        row_entity = ""

        # Decrement rowspans
        if sector_rowspan > 0:
            sector_rowspan -= 1
        if subsector_rowspan > 0:
            subsector_rowspan -= 1

        for cell in cells:
            # Get rowspan value
            rowspan = int(cell.get("rowspan", 1) or 1)

            # Extract cell text (from nested tables or direct content)
            cell_text = ""
            nested_table = cell.find("table")
            if nested_table:
                cell_text = nested_table.get_text(" ", strip=True)
            else:
                cell_text = cell.get_text(" ", strip=True)
            cell_text = re.sub(r"\s+", " ", cell_text).strip()

            # Determine which column this is
            if cell_idx == 0 and sector_rowspan == 0:
                # This is a sector cell
                row_sector = cell_text
                current_sector = cell_text
                sector_rowspan = rowspan - 1
                # Reset subsector when sector changes
                if rowspan > 1:
                    current_subsector = ""
                    subsector_rowspan = 0
                cell_idx += 1
            elif cell_idx <= 1 and subsector_rowspan == 0:
                # This is a subsector cell
                row_subsector = cell_text
                current_subsector = cell_text
                subsector_rowspan = rowspan - 1
                cell_idx += 1
            else:
                # This is an entity cell
                row_entity = cell_text
                cell_idx += 1

        if row_entity and row_entity.strip() not in ("", " "):
            yield {
                "sector": row_sector,
                "subsector": row_subsector,
                "entity": row_entity,
            }


_EURLEX_STRUCT_ID_RE = re.compile(
    r"^(?P<kind>cpt|anx|art)_(?P<value>[A-Za-z0-9]+(?:[a-z])?)(?:\.sct_(?P<section>[A-Za-z0-9]+))?$",
    flags=re.IGNORECASE,
)


def _eurlex_struct_from_id(raw_id: str) -> dict[str, str] | None:
    rid = str(raw_id or "").strip()
    if not rid:
        return None

    m = _EURLEX_STRUCT_ID_RE.match(rid)
    if not m:
        return None

    kind = (m.group("kind") or "").lower()
    value = (m.group("value") or "").strip()
    section = (m.group("section") or "").strip()
    if not value:
        return None

    out: dict[str, str] = {}
    if kind == "cpt":
        out["chapter"] = value.upper()
        if section:
            out["section"] = section.upper()
    elif kind == "anx":
        out["annex"] = value.upper()
        if section:
            out["section"] = section.upper()
    elif kind == "art":
        out["article"] = value.upper()
    else:
        return None

    return out or None


def _iter_html_blocks(html: str, *, enable_eurlex_structural_ids: bool) -> Iterator[tuple[str, str]]:
    """Yield (kind, text) blocks in document order.

    kind is one of: "heading", "text", "struct_ref", "struct_title", "table_row".

    struct_title yields structural title metadata for EUR-Lex documents:
      - article_title: from <p class="oj-sti-art">
      - chapter_title: from <p class="oj-ti-section-2"> after KAPITEL
      - section_title: from <p class="oj-ti-section-2"> after AFDELING
      - annex_title: from second <p class="oj-doc-ti"> after BILAG

    table_row yields structured table data for definitional tables (BILAG I/II):
      - sector, subsector, entity fields as key=value pairs
    """

    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency for HTML ingestion. Install `beautifulsoup4`."
        ) from exc

    soup = BeautifulSoup(html, "html.parser")

    # --- EUR-Lex Table Preprocessing ---
    # Identify and classify all tables. Skip concordance tables entirely.
    # Track which elements are inside processed tables to avoid double-processing.
    _processed_table_elements: set[int] = set()

    if enable_eurlex_structural_ids:
        for table in soup.find_all("table", class_="oj-table"):
            table_type = _classify_eurlex_table(table)
            if table_type == "concordance":
                # Mark all descendants as processed (skip them in main loop)
                for desc in table.descendants:
                    if hasattr(desc, "name"):
                        _processed_table_elements.add(id(desc))
            elif table_type == "definitional":
                # Mark descendants as processed
                for desc in table.descendants:
                    if hasattr(desc, "name"):
                        _processed_table_elements.add(id(desc))
                # Note: table_row blocks are emitted inline when we encounter the table

    # Track last seen structural context to associate titles correctly.
    _last_struct_kind: str | None = None  # "chapter", "section", "annex", "article"

    # EUR-Lex structural title CSS classes.
    _EURLEX_TITLE_CLASSES = frozenset({
        "oj-sti-art",       # Article title: "Anvendelsesområde"
        "oj-ti-section-2",  # Chapter/section title: "ALMINDELIGE BESTEMMELSER"
        "oj-doc-ti",        # Annex title (second occurrence after BILAG X)
        "oj-ti-grseq-1",    # Annex section/point title: "Afsnit A. Liste over...", "1. Indledning"
        "oj-ti-grseq-2",    # Annex sub-point title: "3.1 Udbyderens ansøgning..."
    })

    # Annex-internal structures vary widely across EUR-Lex corpora. We handle these
    # generically via CSS class markers and lightweight text parsing.
    _annex_section_re = re.compile(
        r"(?i)^\s*(?:afsnit|section)\s+([a-z0-9]+)\b(?:\s*[\.-]|\s*[\u2013\u2014-]|\s+)?\s*(.*)\s*$"
    )
    _annex_subpoint_title_re = re.compile(r"^\s*(\d{1,3}\.\d{1,3})(?:\.)?\s+(.+?)\s*$")
    _annex_point_title_re = re.compile(r"^\s*(\d{1,3})\.\s+(.+?)\s*$")

    def _select(node) -> bool:  # type: ignore[no-untyped-def]
        if node is None:
            return False
        # Skip nodes inside processed tables (concordance/definitional)
        if id(node) in _processed_table_elements:
            return False
        name = getattr(node, "name", None)
        if name in _HEADING_TAGS or name in _TEXT_TAGS:
            return True
        if enable_eurlex_structural_ids and name == "div":
            rid = node.get("id")
            return _eurlex_struct_from_id(str(rid or "")) is not None
        # Include definitional tables for row-level processing
        if enable_eurlex_structural_ids and name == "table":
            classes = node.get("class", [])
            if "oj-table" in (classes if isinstance(classes, list) else [classes]):
                return _classify_eurlex_table(node) == "definitional"
        return False

    def _get_css_classes(node) -> set[str]:
        """Extract CSS classes from a node."""
        classes = node.get("class", [])
        if isinstance(classes, str):
            return {classes}
        return set(classes) if classes else set()

    for node in soup.find_all(_select):
        if enable_eurlex_structural_ids and node.name == "div":
            rid = str(node.get("id") or "")
            struct = _eurlex_struct_from_id(rid)
            if struct:
                # Track what kind of structure we just entered.
                if "article" in struct:
                    _last_struct_kind = "article"
                elif "section" in struct:
                    _last_struct_kind = "section"
                elif "chapter" in struct:
                    _last_struct_kind = "chapter"
                elif "annex" in struct:
                    _last_struct_kind = "annex"
                # Encode as a deterministic marker without injecting visible heading text.
                payload = ";".join([f"{k}={v}" for k, v in sorted(struct.items())])
                yield "struct_ref", payload
            continue

        # Handle definitional tables (BILAG I/II style)
        if enable_eurlex_structural_ids and node.name == "table":
            for row_data in _extract_definitional_table_rows(node):
                # Emit as table_row with sector;subsector;entity payload
                parts = []
                if row_data.get("sector"):
                    parts.append(f"sector={row_data['sector']}")
                if row_data.get("subsector"):
                    parts.append(f"subsector={row_data['subsector']}")
                if row_data.get("entity"):
                    parts.append(f"entity={row_data['entity']}")
                if parts:
                    yield "table_row", ";".join(parts)
            continue

        text = node.get_text(" ", strip=True)
        if not text:
            continue

        css_classes = _get_css_classes(node)

        # Check for EUR-Lex structural titles.
        if enable_eurlex_structural_ids and css_classes & _EURLEX_TITLE_CLASSES:
            # Article title: <p class="oj-sti-art">Anvendelsesområde</p>
            if "oj-sti-art" in css_classes:
                yield "struct_title", f"article_title={text}"
                continue

            # Chapter/section title: <p class="oj-ti-section-2">ALMINDELIGE BESTEMMELSER</p>
            if "oj-ti-section-2" in css_classes:
                # Determine if this is a chapter or section title based on last struct.
                if _last_struct_kind == "section":
                    yield "struct_title", f"section_title={text}"
                else:
                    # Default to chapter title (covers chapter and fallback).
                    yield "struct_title", f"chapter_title={text}"
                continue

            # Annex title: second <p class="oj-doc-ti"> after BILAG X.
            if "oj-doc-ti" in css_classes:
                # First oj-doc-ti is "BILAG X", second is the title.
                # Check if this looks like "BILAG X" pattern.
                if re.match(r"^BILAG\s+[IVXLCDM\d]+$", text.upper().strip()):
                    _last_struct_kind = "annex"
                    # This is the annex number, not the title - emit as heading.
                    yield "heading", text
                else:
                    # This is the annex title.
                    yield "struct_title", f"annex_title={text}"
                continue

            # Annex internal structure headings.
            # These are emitted as both:
            #  - struct_ref: deterministic metadata updates (no visible text injected)
            #  - heading: stable boundary + heading prefix for the next chunk
            if "oj-ti-grseq-1" in css_classes or "oj-ti-grseq-2" in css_classes:
                payload_kv: dict[str, str] = {}
                t = re.sub(r"\s+", " ", text).strip()

                # Patterns:
                # - "Afsnit A. <title>" / "Afsnit 1" / "Section 2"
                m_sec = _annex_section_re.match(t)
                if m_sec:
                    payload_kv["annex_section"] = str(m_sec.group(1)).strip()
                    rest = str(m_sec.group(2) or "").strip()
                    if rest:
                        payload_kv["annex_section_title"] = rest

                # - "3.1 <title>" (subpoint)
                m_sub = _annex_subpoint_title_re.match(t)
                if m_sub:
                    payload_kv["annex_subpoint"] = str(m_sub.group(1)).strip()
                    payload_kv["annex_subpoint_title"] = str(m_sub.group(2)).strip()

                # - "1. <title>" (point with title)
                m_pt = _annex_point_title_re.match(t)
                if m_pt and not m_sub:
                    payload_kv["annex_point"] = str(m_pt.group(1)).strip()
                    payload_kv["annex_point_title"] = str(m_pt.group(2)).strip()

                # - Generic titled heading inside annex sections (e.g., after "Afsnit 1")
                #   If it doesn't match the above but is still a grseq title, keep it as the
                #   current annex section title.
                if not payload_kv and t:
                    payload_kv["annex_section_title"] = t

                if payload_kv:
                    payload = ";".join([f"{k}={v}" for k, v in sorted(payload_kv.items()) if v])
                    if payload:
                        yield "struct_ref", payload

                yield "heading", t
                continue

        # Detect chapter/section/annex from heading text to track _last_struct_kind.
        if node.name in _HEADING_TAGS or node.name in _TEXT_TAGS:
            text_upper = text.upper().strip()
            if re.match(r"^KAPITEL\s+[IVXLCDM\d]+$", text_upper):
                _last_struct_kind = "chapter"
            elif re.match(r"^AFDELING\s+[IVXLCDM\d]+$", text_upper):
                _last_struct_kind = "section"
            elif re.match(r"^BILAG\s+[IVXLCDM\d]+$", text_upper):
                _last_struct_kind = "annex"

        if node.name in _HEADING_TAGS:
            yield "heading", text
        else:
            yield "text", text


_sentence_split_re = re.compile(r"(?<=[\.!\?])\s+")


def _split_text_to_fit(text: str, *, encoding, max_tokens: int) -> list[str]:
    """Split a too-large block into smaller blocks without splitting mid-word.

    Prefers sentence boundaries; falls back to greedy word packing.
    """

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    if len(encoding.encode(text)) <= max_tokens:
        return [text]

    # Try sentence boundaries first.
    parts = [p.strip() for p in _sentence_split_re.split(text) if p.strip()]
    if len(parts) <= 1:
        parts = [text]

    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal buf, buf_tokens
        if buf:
            out.append(" ".join(buf).strip())
            buf = []
            buf_tokens = 0

    for part in parts:
        part_tokens = len(encoding.encode(part))
        if part_tokens > max_tokens:
            # Fall back to greedy word packing.
            words = part.split(" ")
            wbuf: list[str] = []
            wtokens = 0
            for w in words:
                if not w:
                    continue
                wt = len(encoding.encode(w))
                if wt > max_tokens:
                    # Single tokenization outlier; emit as-is.
                    if wbuf:
                        out.append(" ".join(wbuf).strip())
                        wbuf = []
                        wtokens = 0
                    out.append(w)
                    continue
                if wtokens + wt + (1 if wbuf else 0) <= max_tokens:
                    wbuf.append(w)
                    wtokens += wt
                else:
                    if wbuf:
                        out.append(" ".join(wbuf).strip())
                    wbuf = [w]
                    wtokens = wt
            if wbuf:
                out.append(" ".join(wbuf).strip())
            continue

        # Normal part
        if buf_tokens + part_tokens + (1 if buf else 0) <= max_tokens:
            buf.append(part)
            buf_tokens += part_tokens
        else:
            flush()
            buf.append(part)
            buf_tokens = part_tokens

    flush()
    return [x for x in out if x]


def _apply_overlap(blocks: list[tuple[str, int]], *, max_overlap_tokens: int) -> list[tuple[str, int]]:
    if max_overlap_tokens <= 0:
        return []

    kept: list[tuple[str, int]] = []
    total = 0
    for text, tok in reversed(blocks):
        if total + tok > max_overlap_tokens:
            break
        kept.append((text, tok))
        total += tok
    kept.reverse()
    return kept


def chunk_html(
    html: str,
    *,
    config: HtmlChunkingConfig,
    base_metadata: Mapping[str, object],
    reference_patterns: ReferencePatterns | None = None,
    initial_reference_state: Mapping[str, str | None] | None = None,
    reset_on: Mapping[str, tuple[str, ...]] | None = None,
    include_reference_mentions: bool = True,
    inline_location_patterns: ReferencePatterns | None = None,
    inline_location_requires: Mapping[str, tuple[str, ...]] | None = None,
    promote_text_headings: bool = True,
    prepend_headings_to_next_chunk: bool = True,
    enable_eurlex_structural_ids: bool = False,
) -> Iterator[dict]:
    from tiktoken import get_encoding

    encoding = get_encoding("cl100k_base")

    reference_patterns = reference_patterns or {}
    inline_location_patterns = inline_location_patterns or {}
    inline_location_requires = inline_location_requires or {}
    reset_on = reset_on or {}
    reference_state: Dict[str, str | None] = dict(initial_reference_state or {})

    source = str(base_metadata.get("source") or "doc")
    corpus_id = str(base_metadata.get("corpus_id") or "corpus")

    chunk_index = 0

    pending_blocks: list[tuple[str, int]] = []
    pending_tokens = 0
    pending_heading_prefix: str | None = None
    pending_text_prefix: str | None = None

    # EUR-Lex often splits list item markers into their own <p>/<li> block, e.g.
    # a block with only "g)" followed by the actual text in the next block.
    # If we emit the marker as its own chunk, we get useless micro-chunks.
    _litra_only_re = re.compile(r"(?i)^\s*([a-z])\)\s*$")
    _paragraph_only_re = re.compile(r"(?i)^\s*\(\s*(\d{1,3})\s*\)\s*$")
    _recital_only_re = re.compile(r"^\s*\(\s*(\d{1,4})\s*\)\s*$")
    _roman_enum_only_re = re.compile(r"(?i)^\s*([ivxlcdm]{2,12})\)\s*$")
    _numeric_enum_only_re = re.compile(r"^\s*(\d{1,3})\)\s*$")
    _dash_only_re = re.compile(r"^\s*[\-\u2013\u2014]+\s*$")
    # Annex point markers: "1.", "2.", etc. - EUR-Lex uses these for numbered categories in annexes.
    # Pattern: just a number followed by a period at the end (e.g., "<p>1.</p>").
    _annex_point_only_re = re.compile(r"^\s*(\d{1,2})\.\s*$")
    # Inline annex point/subpoint lines where the marker and content are in the same block.
    # Examples: "1. Something" or "3.1 Something".
    _annex_point_inline_re = re.compile(r"^\s*(\d{1,2})\.\s+(.+?)\s*$")
    _annex_subpoint_inline_re = re.compile(r"^\s*(\d{1,3}\.\d{1,3})(?:\.)?\s+(.+?)\s*$")

    def queue_heading_prefix(text: str) -> None:
        nonlocal pending_heading_prefix
        if not prepend_headings_to_next_chunk:
            return
        t = re.sub(r"\s+", " ", text or "").strip()
        if not t:
            return
        pending_heading_prefix = t

    def ensure_heading_prefix_block() -> None:
        nonlocal pending_heading_prefix, pending_blocks, pending_tokens
        if not pending_heading_prefix:
            return
        prefix = pending_heading_prefix
        pending_heading_prefix = None
        tok = len(encoding.encode(prefix))
        if tok <= 0:
            return
        pending_blocks.insert(0, (prefix, tok))
        pending_tokens += tok


    def emit_chunk() -> dict | None:
        nonlocal chunk_index, pending_blocks, pending_tokens
        if not pending_blocks:
            return None

        ensure_heading_prefix_block()

        text_value = "\n".join([b[0] for b in pending_blocks]).strip()
        if not text_value:
            pending_blocks = []
            pending_tokens = 0
            return None

        metadata: Dict[str, object] = dict(base_metadata)
        metadata.update({"doc_type": "chunk", "chunk_index": chunk_index})

        # Ensure baseline schema fields exist (best-effort for generic usage).
        metadata.setdefault("schema_version", "meta:v1")
        metadata.setdefault("doc_id", str(metadata.get("doc_id") or source))
        metadata.setdefault("doc_version", str(metadata.get("doc_version") or "unknown"))
        metadata.setdefault("language", str(metadata.get("language") or "und"))
        metadata.setdefault("source_type", str(metadata.get("source_type") or "html"))
        metadata.setdefault("source_path", str(metadata.get("source_path") or ""))

        extracted = {k: v for k, v in reference_state.items() if v}
        metadata.update(extracted)

        # Join key between chunks and TOC.
        # NOTE: keep paragraph/litra out of the join key; we flush on their changes
        # to preserve metadata precision without exploding chunk counts.
        scope_state: Dict[str, str | None] = {
            "preamble": reference_state.get("preamble"),
            "citation": reference_state.get("citation"),
            "recital": reference_state.get("recital"),
            "chapter": reference_state.get("chapter"),
            "annex": reference_state.get("annex"),
            "annex_section": reference_state.get("annex_section"),
            "annex_point": reference_state.get("annex_point"),
            "annex_subpoint": reference_state.get("annex_subpoint"),
            "annex_sector": reference_state.get("annex_sector"),
            "annex_subsector": reference_state.get("annex_subsector"),
            "section": reference_state.get("section"),
            "article": reference_state.get("article"),
            # Structural titles for heading_path_display.
            "chapter_title": reference_state.get("chapter_title"),
            "section_title": reference_state.get("section_title"),
            "article_title": reference_state.get("article_title"),
            "annex_title": reference_state.get("annex_title"),
            "annex_section_title": reference_state.get("annex_section_title"),
            "annex_point_title": reference_state.get("annex_point_title"),
            "annex_subpoint_title": reference_state.get("annex_subpoint_title"),
        }
        location_id = _make_location_id(reference_state=scope_state)
        metadata["location_id"] = location_id

        # Deterministic flag to support audit-safe retrieval filtering.
        # True when this chunk has a citable structural location.
        loc_lower = str(location_id or "").strip().lower()
        metadata["citable"] = bool(
            metadata.get("chapter")
            or metadata.get("article")
            or metadata.get("annex")
            or metadata.get("preamble")
            or metadata.get("citation")
            or metadata.get("recital")
            or ("chapter:" in loc_lower)
            or ("article:" in loc_lower)
            or ("annex:" in loc_lower)
            or ("preamble" in loc_lower)
            or ("citation:" in loc_lower)
            or ("recital:" in loc_lower)
        )

        heading_json, heading_display = make_heading_path(reference_state=scope_state)
        metadata["heading_path"] = heading_json
        metadata["heading_path_display"] = heading_display

        text_hash = compute_text_hash(text_value)
        metadata["text_hash"] = text_hash
        metadata["chunk_id"] = _build_chunk_id(
            doc_id=str(metadata.get("doc_id") or source),
            location_id=location_id,
            chunk_index=chunk_index,
            text_hash=text_hash,
        )

        if include_reference_mentions:
            mentions = _extract_reference_mentions(text_value, reference_state=reference_state)
            if mentions:
                # Chroma metadata values must be primitives; store as JSON.
                metadata["mentions"] = json.dumps(mentions, ensure_ascii=False)

        validate_metadata_primitives(metadata)

        chunk_index += 1

        return {"text": text_value, "metadata": metadata}

    def _apply_struct_update(payload: str) -> None:
        # payload: "chapter=III;section=1" etc.
        kv: dict[str, str] = {}
        for part in (payload or "").split(";"):
            if not part.strip() or "=" not in part:
                continue
            k, v = part.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if not k or not v:
                continue
            kv[k] = v

        if not kv:
            return

        # Deterministic reset rules to prevent structure bleed.
        if "chapter" in kv and kv.get("chapter") != reference_state.get("chapter"):
            reference_state["preamble"] = None
            reference_state["citation"] = None
            reference_state["recital"] = None
            reference_state["annex"] = None
            reference_state["annex_section"] = None
            reference_state["annex_point"] = None
            reference_state["annex_subpoint"] = None
            reference_state["section"] = None
            reference_state["article"] = None
            reference_state["paragraph"] = None
            reference_state["litra"] = None
            # Reset structural titles when chapter changes.
            reference_state["chapter_title"] = None
            reference_state["section_title"] = None
            reference_state["article_title"] = None
            reference_state["annex_title"] = None
            reference_state["annex_section_title"] = None
            reference_state["annex_point_title"] = None
            reference_state["annex_subpoint_title"] = None
        if "annex" in kv and kv.get("annex") != reference_state.get("annex"):
            reference_state["preamble"] = None
            reference_state["citation"] = None
            reference_state["recital"] = None
            reference_state["chapter"] = None
            reference_state["section"] = None
            reference_state["article"] = None
            reference_state["paragraph"] = None
            reference_state["litra"] = None
            reference_state["annex_section"] = None
            reference_state["annex_point"] = None
            reference_state["annex_subpoint"] = None
            # Reset structural titles when annex changes.
            reference_state["chapter_title"] = None
            reference_state["section_title"] = None
            reference_state["article_title"] = None
            reference_state["annex_title"] = None
            reference_state["annex_section_title"] = None
            reference_state["annex_point_title"] = None
            reference_state["annex_subpoint_title"] = None

        if "annex_section" in kv and kv.get("annex_section") != reference_state.get("annex_section"):
            reference_state["annex_point"] = None
            reference_state["annex_subpoint"] = None
            reference_state["annex_point_title"] = None
            reference_state["annex_subpoint_title"] = None

        if "annex_point" in kv and kv.get("annex_point") != reference_state.get("annex_point"):
            reference_state["annex_subpoint"] = None
            reference_state["annex_subpoint_title"] = None
        if "section" in kv and kv.get("section") != reference_state.get("section"):
            reference_state["article"] = None
            reference_state["paragraph"] = None
            reference_state["litra"] = None
            # Reset article title when section changes.
            reference_state["section_title"] = None
            reference_state["article_title"] = None
        if "article" in kv and kv.get("article") != reference_state.get("article"):
            reference_state["preamble"] = None
            reference_state["citation"] = None
            reference_state["recital"] = None
            reference_state["paragraph"] = None
            reference_state["litra"] = None
            # Reset article title when article changes.
            reference_state["article_title"] = None

        for k, v in kv.items():
            if k in {"chapter", "annex", "section", "article"}:
                reference_state[k] = str(v).strip().upper()
            elif k == "litra":
                reference_state[k] = str(v).strip().lower()
            else:
                reference_state[k] = str(v).strip()

        # Apply caller-provided reset_on on top, for compatibility.
        for key in kv.keys():
            for reset_field in reset_on.get(key, ()):  # pragma: no branch
                reference_state[reset_field] = None

    for kind, raw_text in _iter_html_blocks(html, enable_eurlex_structural_ids=enable_eurlex_structural_ids):
        if kind == "struct_ref":
            pending_text_prefix = None
            # Treat as a stable boundary like headings, but do NOT inject heading text.
            row = emit_chunk()
            if row:
                yield row
            # Do NOT carry overlap across structural state changes.
            pending_blocks = []
            pending_tokens = 0
            _apply_struct_update(raw_text)
            continue

        if kind == "struct_title":
            # Apply structural title to reference_state (e.g., "article_title=Anvendelsesområde").
            # These are NOT chunk boundaries - they enrich the current structural context.
            if "=" in raw_text:
                title_key, title_val = raw_text.split("=", 1)
                title_key = title_key.strip().lower()
                title_val = title_val.strip()
                if title_key and title_val:
                    reference_state[title_key] = title_val
            continue

        if kind == "table_row":
            # Definitional table row (BILAG I/II style): sector;subsector;entity
            # Each row becomes its own chunk with rich metadata.
            pending_text_prefix = None
            row = emit_chunk()
            if row:
                yield row
            pending_blocks = []
            pending_tokens = 0

            # Parse the row payload and update reference_state
            row_data: dict[str, str] = {}
            for part in (raw_text or "").split(";"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    row_data[k.strip().lower()] = v.strip()

            # Update sector/subsector in reference_state for metadata.
            # Always set both - use None if not present to prevent carry-over.
            reference_state["annex_sector"] = row_data.get("sector") or None
            reference_state["annex_subsector"] = row_data.get("subsector") or None

            # Build the chunk text with sector context
            text_parts = []
            if row_data.get("sector"):
                text_parts.append(f"Sektor: {row_data['sector']}")
            if row_data.get("subsector"):
                text_parts.append(f"Delsektor: {row_data['subsector']}")
            if row_data.get("entity"):
                text_parts.append(f"Type enhed: {row_data['entity']}")

            if text_parts:
                block_text = "\n".join(text_parts)
                tok = len(encoding.encode(block_text))
                pending_blocks.append((block_text, tok))
                pending_tokens = tok
            continue

        if kind == "heading":
            pending_text_prefix = None
            # Headings are stable boundaries. Flush any accumulated text so metadata
            # does not get overwritten by a later heading within the same chunk.
            row = emit_chunk()
            if row:
                yield row
            # Do NOT carry overlap across heading boundaries (metadata changes).
            pending_blocks = []
            pending_tokens = 0

        if kind == "heading" and reference_patterns:
            # Update state ONLY from headings to avoid in-text false positives.
            heading = raw_text.strip()
            queue_heading_prefix(heading)
            for key, pattern in reference_patterns.items():
                m = pattern.search(heading)
                if not m:
                    continue
                val = m.group(1)
                if key in {"chapter", "annex"}:
                    val = val.upper()
                elif key in {"article"}:
                    val = val.upper()
                elif key in {"litra"}:
                    val = val.lower()
                # Recitals belong to the preamble; once we enter numbered structure, clear them.
                if key in {"chapter", "annex", "article"}:
                    reference_state["preamble"] = None
                    reference_state["citation"] = None
                    reference_state["recital"] = None
                reference_state[key] = val
                for reset_field in reset_on.get(key, ()):  # pragma: no branch
                    reference_state[reset_field] = None
            continue

        if kind != "text":
            continue

        # If the previous block was a standalone marker (e.g. "g)") we prefix it to
        # the current text block to keep the label with its content.
        if pending_text_prefix:
            raw_text = f"{pending_text_prefix} {raw_text}".strip()
            pending_text_prefix = None

        # Capture standalone list markers that EUR-Lex emits as separate blocks.
        # These should not become chunks on their own.
        candidate_marker = (raw_text or "").strip()

        # Inline annex point/subpoint markers like "1. ..." and "3.1 ...".
        # Some annexes use display:inline formatting where the marker and text are combined.
        if reference_state.get("annex") and not reference_state.get("article"):
            m_sub_inline = _annex_subpoint_inline_re.match(candidate_marker)
            if m_sub_inline:
                # Boundary before switching annex_subpoint state.
                row = emit_chunk()
                if row:
                    yield row
                pending_blocks = []
                pending_tokens = 0

                reference_state["annex_subpoint"] = str(m_sub_inline.group(1)).strip()
                reference_state["annex_subpoint_title"] = None
                pending_text_prefix = f"{reference_state['annex_subpoint']}"
                raw_text = str(m_sub_inline.group(2) or "").strip()
                if not raw_text:
                    continue
                candidate_marker = raw_text.strip()

            m_pt_inline = _annex_point_inline_re.match(candidate_marker)
            if m_pt_inline:
                # Boundary before switching annex_point state.
                row = emit_chunk()
                if row:
                    yield row
                pending_blocks = []
                pending_tokens = 0

                reference_state["annex_point"] = str(m_pt_inline.group(1)).strip()
                reference_state["annex_point_title"] = None
                reference_state["annex_subpoint"] = None
                reference_state["annex_subpoint_title"] = None
                reference_state["litra"] = None
                pending_text_prefix = f"{reference_state['annex_point']}."
                raw_text = str(m_pt_inline.group(2) or "").strip()
                if not raw_text:
                    continue
                candidate_marker = raw_text.strip()

        # Ignore standalone bullet separators like "—" that come from list formatting.
        if _dash_only_re.match(candidate_marker):
            continue

        # Standalone recital markers in the preamble: "(42)" etc.
        # Only treat as a recital when we are NOT inside an article; otherwise it's paragraph numbering.
        m_rec = _recital_only_re.match(candidate_marker)
        if m_rec and not reference_state.get("article"):
            # Boundary before switching recital state.
            row = emit_chunk()
            if row:
                yield row
            pending_blocks = []
            pending_tokens = 0

            # Recitals are part of the preamble. Clear numbered structure to prevent bleed.
            reference_state["chapter"] = None
            reference_state["annex"] = None
            reference_state["section"] = None
            reference_state["article"] = None
            reference_state["paragraph"] = None
            reference_state["litra"] = None
            reference_state["citation"] = None
            reference_state["preamble"] = "1"
            reference_state["recital"] = str(m_rec.group(1)).strip()
            pending_text_prefix = f"({reference_state['recital']})"
            continue

        # Roman numeral list markers like "ii)" / "iii)".
        # Merge into next block and clear litra to avoid bleed from prior (a)/(b)/... items.
        m_roman = _roman_enum_only_re.match(candidate_marker)
        if m_roman:
            pending_text_prefix = f"{m_roman.group(1)})".strip()
            reference_state["litra"] = None
            continue

        # Numeric list markers like "2)".
        m_num = _numeric_enum_only_re.match(candidate_marker)
        if m_num:
            pending_text_prefix = f"{m_num.group(1)})".strip()
            continue
        m_litra = _litra_only_re.match(candidate_marker)
        if m_litra:
            reference_state["litra"] = str(m_litra.group(1)).strip().lower()
            pending_text_prefix = f"{reference_state['litra']})"
            continue

        m_par = _paragraph_only_re.match(candidate_marker)
        if m_par and reference_state.get("article"):
            # Only accept bare "(2)" as a paragraph marker if we're inside an article.
            reference_state["paragraph"] = str(m_par.group(1)).strip()
            # New paragraph -> litra must not bleed from previous (a)/(b)/... items.
            reference_state["litra"] = None
            pending_text_prefix = f"({reference_state['paragraph']})"
            continue

        # Annex point markers: "1.", "2.", etc.
        # Only treat as annex point when we are inside an annex (not an article).
        # This creates chunk boundaries for each numbered category in annexes like Annex III.
        m_annex_point = _annex_point_only_re.match(candidate_marker)
        if m_annex_point and reference_state.get("annex") and not reference_state.get("article"):
            # Boundary before switching annex_point state.
            row = emit_chunk()
            if row:
                yield row
            pending_blocks = []
            pending_tokens = 0

            # Update annex_point and reset sub-structure.
            reference_state["annex_point"] = str(m_annex_point.group(1)).strip()
            reference_state["annex_point_title"] = None
            reference_state["annex_subpoint"] = None
            reference_state["annex_subpoint_title"] = None
            reference_state["litra"] = None
            pending_text_prefix = f"{reference_state['annex_point']}."
            continue

        # Some HTML sources (e.g. EUR-Lex) use <p> for headings. Optionally promote
        # heading-like text lines to headings (but still update state only when the
        # heading appears at the start of the line).
        if promote_text_headings and reference_patterns:
            candidate = raw_text.strip()
            if candidate and len(candidate) <= 220:
                if not candidate.endswith("."):
                    matched_any = False
                    for key, pattern in reference_patterns.items():
                        m = pattern.match(candidate)
                        if not m:
                            continue
                        # Boundary before switching state
                        row = emit_chunk()
                        if row:
                            yield row
                        overlap_blocks = _apply_overlap(pending_blocks, max_overlap_tokens=config.overlap)
                        pending_blocks = overlap_blocks.copy()
                        pending_tokens = sum(t for _, t in pending_blocks)

                        matched_any = True
                        queue_heading_prefix(candidate)

                        val = m.group(1)
                        if key in {"chapter", "annex"}:
                            val = val.upper()
                        elif key in {"article"}:
                            val = val.upper()
                        elif key in {"litra"}:
                            val = val.lower()
                        if key in {"chapter", "annex", "article"}:
                            reference_state["preamble"] = None
                            reference_state["citation"] = None
                            reference_state["recital"] = None
                        reference_state[key] = val
                        for reset_field in reset_on.get(key, ()):  # pragma: no branch
                            reference_state[reset_field] = None

                    if matched_any:
                        continue

        # Update location state from text blocks (conservative; optional).
        if inline_location_patterns:
            candidate = raw_text.strip()
            for key, pattern in inline_location_patterns.items():
                required = inline_location_requires.get(key, ())
                if required and any(not reference_state.get(req) for req in required):
                    continue
                m = pattern.match(candidate)
                if not m:
                    continue
                groups = [g for g in (m.groups() or ()) if g]
                val = (groups[0] if groups else (m.group(1) if m.lastindex else "") or "").strip()
                if not val:
                    continue

                # If this text block changes the structural location (e.g., new paragraph/litra),
                # flush pending content so chunk metadata remains precise without splitting every <p>.
                prev_val = reference_state.get(key)
                if prev_val is not None and str(prev_val).strip() != "" and str(prev_val).strip() != str(val).strip():
                    row = emit_chunk()
                    if row:
                        yield row
                    # Do NOT carry overlap across paragraph/litra changes; it breaks metadata correctness.
                    pending_blocks = []
                    pending_tokens = 0

                # Paragraph changes must reset litra to avoid invalid addresses like (2)(j).
                if key == "paragraph":
                    reference_state["litra"] = None

                if key in {"chapter", "annex", "section"}:
                    val = val.upper()
                elif key == "article":
                    val = val.upper()
                elif key == "litra":
                    val = val.lower()
                reference_state[key] = val
                for reset_field in reset_on.get(key, ()):  # pragma: no branch
                    reference_state[reset_field] = None

        for block in _split_text_to_fit(raw_text, encoding=encoding, max_tokens=config.chunk_tokens):
            btok = len(encoding.encode(block))
            if btok <= 0:
                continue

            ensure_heading_prefix_block()

            if pending_tokens + btok <= config.chunk_tokens:
                pending_blocks.append((block, btok))
                pending_tokens += btok
                continue

            # flush current chunk, then start a new one with overlap
            row = emit_chunk()
            if row:
                yield row

            overlap_blocks = _apply_overlap(pending_blocks, max_overlap_tokens=config.overlap)
            pending_blocks = overlap_blocks.copy()
            pending_tokens = sum(t for _, t in pending_blocks)

            # If overlap itself fills the window, drop it.
            if pending_tokens + btok > config.chunk_tokens:
                pending_blocks = []
                pending_tokens = 0

            pending_blocks.append((block, btok))
            pending_tokens += btok

        if config.flush_each_text_block:
            row = emit_chunk()
            if row:
                yield row
            pending_blocks = []
            pending_tokens = 0

    last = emit_chunk()
    if last:
        yield last


def chunk_html_file(
    html_path: Path,
    *,
    config: HtmlChunkingConfig,
    base_metadata: Mapping[str, object],
    reference_patterns: ReferencePatterns | None = None,
    initial_reference_state: Mapping[str, str | None] | None = None,
    reset_on: Mapping[str, tuple[str, ...]] | None = None,
    include_reference_mentions: bool = True,
    inline_location_patterns: ReferencePatterns | None = None,
    inline_location_requires: Mapping[str, tuple[str, ...]] | None = None,
    enable_eurlex_structural_ids: bool = False,
) -> Iterator[dict]:
    html = html_path.read_text(encoding="utf-8", errors="replace")
    return chunk_html(
        html,
        config=config,
        base_metadata=base_metadata,
        reference_patterns=reference_patterns,
        initial_reference_state=initial_reference_state,
        reset_on=reset_on,
        include_reference_mentions=include_reference_mentions,
        inline_location_patterns=inline_location_patterns,
        inline_location_requires=inline_location_requires,
        enable_eurlex_structural_ids=enable_eurlex_structural_ids,
    )


def write_jsonl(path: Path, rows: Iterator[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
