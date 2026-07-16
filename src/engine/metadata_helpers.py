"""Metadata normalization and anchor extraction utilities.

Pure functions for normalizing chunk metadata fields (case, format) and extracting
structural anchors (article, recital, annex) from metadata and answer text.

Extracted from helpers.py (Phase 8b) — single responsibility.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def normalize_anchor(anchor: str) -> str:
    """Normalize an anchor string for comparison (lowercase, no whitespace)."""
    raw = str(anchor or "").strip().lower()
    return re.sub(r"\s+", "", raw)


def normalize_annex_for_chroma(annex_value: str | Any) -> str:
    """Normalize an annex value for Chroma queries.

    Chroma stores annex as uppercase Roman numerals (III, IV, etc.).
    This function ensures case-insensitive matching by converting to uppercase.

    Args:
        annex_value: The annex value (e.g., "iii", "III", "iv")

    Returns:
        Uppercase version of the annex value (e.g., "III", "IV")
    """
    return str(annex_value or "").strip().upper()


# Fields that should be normalized to uppercase (structural identifiers)
_UPPERCASE_META_FIELDS = frozenset(
    {
        "article",
        "annex",
        "recital",
        "chapter",
        "section",
        "paragraph",
        "annex_point",
        "annex_section",
    }
)

# Fields that should be normalized to lowercase (corpus/source identifiers)
_LOWERCASE_META_FIELDS = frozenset(
    {
        "corpus_id",
        "law_id",
    }
)


def get_meta_value(meta: dict[str, Any] | None, key: str, default: str = "") -> str:
    """Get a metadata value with automatic case normalization.

    This eliminates case-sensitivity as an error source by normalizing
    values at read-time. Structural fields (article, annex, chapter, etc.)
    are uppercased. Identifier fields (corpus_id, law_id) are lowercased.
    Other fields are returned stripped but case-preserved.

    Args:
        meta: The metadata dict (from Chroma results or chunk)
        key: The field name to retrieve
        default: Default value if key is missing or empty

    Returns:
        Normalized string value
    """
    if not meta:
        return default
    raw = meta.get(key)
    if raw is None:
        return default
    val = str(raw).strip()
    if not val:
        return default

    if key in _UPPERCASE_META_FIELDS:
        return val.upper()
    if key in _LOWERCASE_META_FIELDS:
        return val.lower()
    return val


def normalize_metadata(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of metadata with all known fields normalized.

    This is useful when you need to work with multiple fields and want
    consistent casing throughout. Unknown fields are passed through unchanged.

    Args:
        meta: The metadata dict to normalize

    Returns:
        New dict with normalized values (original is not mutated)
    """
    if not meta:
        return {}
    out = dict(meta)
    for key in _UPPERCASE_META_FIELDS:
        if key in out and out[key] is not None:
            out[key] = str(out[key]).strip().upper()
    for key in _LOWERCASE_META_FIELDS:
        if key in out and out[key] is not None:
            out[key] = str(out[key]).strip().lower()
    return out


def normalize_anchor_list(xs: Any, *, require_colon: bool = False) -> list[str]:
    """Normalize a list of anchors.

    Args:
        xs: List of anchor strings (or a single string).
        require_colon: If True, only include anchors containing ':' (e.g., 'article:1').

    Returns:
        Deduplicated list of normalized anchor strings.
    """
    if not xs:
        return []
    out: list[str] = []
    items = [xs] if isinstance(xs, str) else (xs if isinstance(xs, list) else [])
    for x in items:
        if isinstance(x, str) and x.strip():
            na = normalize_anchor(x)
            if require_colon and ":" not in na:
                continue
            out.append(na)
    # Deduplicate while preserving order.
    return list(dict.fromkeys(out))


def _derive_structural_fields_from_location_id(
    metadata: dict[str, Any] | None,
) -> dict[str, str]:
    """Derive structural fields from canonical location_id without mutating input."""

    m = dict(metadata or {})
    loc = str(m.get("location_id") or "").strip()
    if not loc:
        return {}
    segs = [p.strip() for p in loc.split("/") if p.strip()]
    derived: dict[str, str] = {}
    for s in segs:
        low = s.lower()
        if low.startswith("chapter:"):
            derived["chapter"] = s.split(":", 1)[1].strip().upper()
        elif low.startswith("section:"):
            derived["section"] = s.split(":", 1)[1].strip().upper()
        elif low.startswith("article:"):
            derived["article"] = s.split(":", 1)[1].strip().upper()
        elif low.startswith("annex:"):
            derived["annex"] = s.split(":", 1)[1].strip().upper()
        elif low.startswith("recital:"):
            derived["recital"] = s.split(":", 1)[1].strip().upper()
    return derived


def _extract_raw_anchors_from_chunk(meta: dict[str, Any]) -> list[str]:
    """Extract canonical anchor strings (article:X, recital:Y) from chunk metadata."""
    out: list[str] = []
    mm = dict(meta or {})
    derived = _derive_structural_fields_from_location_id(mm)

    art = mm.get("article") or derived.get("article")
    rec = mm.get("recital") or derived.get("recital")
    ann = mm.get("annex") or derived.get("annex")

    if art:
        out.append(re.sub(r"\s+", "", f"article:{str(art).strip()}").lower())
    if rec:
        out.append(re.sub(r"\s+", "", f"recital:{str(rec).strip()}").lower())
    if ann:
        out.append(re.sub(r"\s+", "", f"annex:{str(ann).strip()}").lower())
    return out


def anchors_from_metadata(meta: Dict[str, Any] | None) -> set:
    """Extract normalized anchors from chunk metadata as a set."""
    return set(_extract_raw_anchors_from_chunk(dict(meta or {})))


def anchors_present_from_hits(hits: List[Tuple[str, Dict[str, Any]]]) -> set:
    """Extract all normalized anchors present in a list of (doc, metadata) hits."""
    out: set = set()
    for _doc, meta in list(hits or []):
        out.update(anchors_from_metadata(meta))
    return out


def _extract_anchor_mentions_from_answer(
    text: str,
) -> dict[str, list[tuple[str, str | None]] | list[str]]:
    """Extract (article, paragraph?) plus recitals/annexes mentioned in answer text."""

    out_articles: list[tuple[str, str | None]] = []
    out_recitals: list[str] = []
    out_annexes: list[str] = []

    t = str(text or "")
    # Articles, optionally with paragraph (stk./(n)/paragraph).
    art_re = re.compile(
        r"(?i)\b(?:Artikel|Article|Art\.?)(?:\s+)(\d{1,3}[a-z]?)\b(?:\s*(?:\(|,)?\s*(?:stk\.?|stykke|paragraph)?\s*(\d{1,3})\s*\)?)?"
    )
    for m in art_re.finditer(t):
        art = str(m.group(1) or "").strip().upper()
        par = str(m.group(2) or "").strip()
        out_articles.append((art, par or None))

    # Recitals / betragtninger
    rec_re = re.compile(r"(?i)\b(?:betragtning(?:er)?|recital)\s+(\d{1,4})\b")
    for m in rec_re.finditer(t):
        out_recitals.append(str(m.group(1) or "").strip())

    # Annex / Bilag
    annex_re = re.compile(r"(?i)\b(?:bilag|annex)\s+([ivxlcdm]+|\d{1,3})\b")
    for m in annex_re.finditer(t):
        out_annexes.append(str(m.group(1) or "").strip().upper())

    return {"articles": out_articles, "recitals": out_recitals, "annexes": out_annexes}
