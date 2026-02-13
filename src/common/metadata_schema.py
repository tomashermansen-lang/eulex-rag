from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, NotRequired, TypedDict


Primitive = str | int | float | bool | None


class CommonMetadata(TypedDict, total=False):
    # Versioning
    schema_version: str

    # Corpus/document identity
    corpus_id: str
    doc_id: str
    doc_version: str
    language: str

    # Provenance
    source_type: str
    source_path: str
    source: str
    canonical_url: str | None
    source_url: str | None


class ChunkMetadata(CommonMetadata, total=False):
    doc_type: str
    chunk_index: int
    chunk_id: str
    location_id: str
    text_hash: str
    heading_path: str
    heading_path_display: str
    mentions: str

    # Optional structural fields (kept for filtering and citations)
    chapter: str
    section: str
    article: str
    paragraph: str
    litra: str
    annex: str
    annex_point: str
    annex_section: str
    annex_subpoint: str
    annex_title: str
    annex_section_title: str
    annex_point_title: str
    annex_subpoint_title: str


class TocMetadata(CommonMetadata, total=False):
    doc_type: str
    toc_kind: str

    node_id: str
    parent_id: str | None
    order: int

    chunk_index: int
    html_heading_index: int
    chunk_id: str

    location_id: str
    heading_path: str
    heading_path_display: str

    display_title: str
    title: str

    chapter: str
    section: str
    article: str
    annex: str

    preamble_kind: str
    recital: str
    section_kind: str


_SCHEMA_VERSION = "meta:v1"
_LOCATION_SCHEMA_VERSION = "loc:v1"


_slug_re = re.compile(r"[^a-z0-9\-]+")


def slugify(value: str) -> str:
    v = (value or "").strip().lower().replace(" ", "-")
    v = _slug_re.sub("", v)
    v = re.sub(r"-+", "-", v).strip("-")
    return v or "doc"


def normalize_ref_values(reference_state: Mapping[str, str | None]) -> dict[str, str]:
    """Normalize structural reference parts for stable ids.

    Output values are safe for identifiers (lowercase/normalized).
    """

    out: dict[str, str] = {}

    def _norm(kind: str, raw: str) -> str:
        r = (raw or "").strip()
        if not r:
            return ""

        if kind in {"chapter", "annex"}:
            # Keep roman numerals stable; normalize to lowercase.
            return r.upper().lower()
        if kind == "annex_section":
            # Annex sections are typically alphanumeric (e.g., "A" / "1").
            return slugify(r)
        if kind == "annex_point":
            # Keep numeric annex points as-is.
            return re.sub(r"\s+", "", r)
        if kind == "annex_subpoint":
            # Keep dotted numeric subpoints stable (e.g., "3.1").
            return re.sub(r"\s+", "", r)
        if kind == "section":
            return r.upper().lower()
        if kind == "article":
            return r.upper().lower()
        if kind == "paragraph":
            return re.sub(r"\s+", "", r)
        if kind == "litra":
            return r.lower()

        return slugify(r)

    for k in (
        "preamble",
        "citation",
        "recital",
        "chapter",
        "annex",
        "annex_section",
        "annex_point",
        "annex_subpoint",
        "section",
        "article",
        "paragraph",
        "litra",
    ):
        v = reference_state.get(k)
        if v is None or str(v).strip() == "":
            continue
        nv = _norm(k, str(v))
        if nv:
            out[k] = nv

    return out


def make_location_id(*, reference_state: Mapping[str, str | None]) -> str:
    """Build a stable structural location id.

    This is intentionally corpus-agnostic. Join by (doc_id, location_id).

    Example: loc:v1/chapter:iv/section:2/article:10
    """

    refs = normalize_ref_values(reference_state)

    # Special-case preamble/citations/recitals.
    if refs.get("preamble"):
        return f"{_LOCATION_SCHEMA_VERSION}/preamble"
    if refs.get("citation"):
        return f"{_LOCATION_SCHEMA_VERSION}/citation:{refs['citation']}"
    if refs.get("recital"):
        return f"{_LOCATION_SCHEMA_VERSION}/recital:{refs['recital']}"

    parts: list[str] = [_LOCATION_SCHEMA_VERSION]
    for kind in (
        "chapter",
        "annex",
        "annex_section",
        "annex_point",
        "annex_subpoint",
        "section",
        "article",
        "paragraph",
        "litra",
    ):
        if kind in refs:
            parts.append(f"{kind}:{refs[kind]}")

    return "/".join(parts)


def make_heading_path(*, reference_state: Mapping[str, str | None]) -> tuple[str, str]:
    """Return (heading_path_json, display_string).

    The display string includes structural titles when available, e.g.:
      "Chapter I (Almindelige bestemmelser) > Article 2 (Anvendelsesområde)"

    This enables evidence type classification to detect scope/definition keywords
    from metadata without requiring text analysis.
    """

    refs = normalize_ref_values(reference_state)
    segments: list[str] = []

    # Special-case preamble/citations/recitals.
    if refs.get("preamble"):
        segments.append("preamble")
    if "citation" in refs:
        segments.append(f"citation:{refs['citation']}")
    if "recital" in refs:
        segments.append(f"recital:{refs['recital']}")

    for kind in (
        "chapter",
        "annex",
        "annex_section",
        "annex_point",
        "annex_subpoint",
        "section",
        "article",
        "paragraph",
        "litra",
    ):
        if kind in refs:
            segments.append(f"{kind}:{refs[kind]}")

    # Helper to format display part with optional title.
    def _fmt(label: str, value: str, title_key: str) -> str:
        title = reference_state.get(title_key)
        if title:
            return f"{label} {value} ({title})"
        return f"{label} {value}"

    display_parts: list[str] = []
    if refs.get("preamble"):
        display_parts.append("Preamble")
    if "citation" in refs:
        display_parts.append("Citation")
    if "recital" in refs:
        display_parts.append(f"Recital {refs['recital']}")
    if "chapter" in refs:
        display_parts.append(_fmt("Chapter", refs["chapter"].upper(), "chapter_title"))
    if "annex" in refs:
        display_parts.append(_fmt("Annex", refs["annex"].upper(), "annex_title"))
    if "annex_section" in refs:
        display_parts.append(_fmt("Annex section", refs["annex_section"].upper(), "annex_section_title"))
    if "annex_point" in refs:
        display_parts.append(_fmt("Point", refs["annex_point"], "annex_point_title"))
    if "annex_subpoint" in refs:
        display_parts.append(_fmt("Subpoint", refs["annex_subpoint"], "annex_subpoint_title"))
    if "section" in refs:
        display_parts.append(_fmt("Section", refs["section"].upper(), "section_title"))
    if "article" in refs:
        display_parts.append(_fmt("Article", refs["article"].upper(), "article_title"))
    if "paragraph" in refs:
        display_parts.append(f"para {refs['paragraph']}")
    if "litra" in refs:
        display_parts.append(f"litra {refs['litra']}")

    return json.dumps(segments, ensure_ascii=False), " > ".join(display_parts)


def build_doc_id(*, corpus_id: str, source_stem: str) -> str:
    """Human-meaningful doc id.

    Uses the original file stem when available (e.g. 'GDPR', 'AI ACT'), otherwise
    falls back to the corpus id.
    """

    stem = (source_stem or "").strip()
    return stem or (corpus_id or "doc").strip() or "doc"


def build_source_path(*, html_path: Path, project_root: Path | None = None) -> str:
    try:
        if project_root is not None:
            return str(html_path.resolve().relative_to(project_root.resolve()))
    except Exception:  # noqa: BLE001
        pass
    return str(html_path)


def compute_doc_version_from_file(path: Path) -> str:
    """Content hash of the raw input file (sha256 hex)."""

    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


_whitespace_re = re.compile(r"\s+")


def normalize_text_for_hash(text: str) -> str:
    return _whitespace_re.sub(" ", (text or "").strip())


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()


def build_chunk_id(*, doc_id: str, location_id: str, chunk_index: int, text_hash: str) -> str:
    # Keep it readable, but safe.
    doc_part = slugify(doc_id)
    loc_part = slugify(location_id)
    return f"chunk:v1/{doc_part}/{loc_part}/{int(chunk_index)}/{text_hash[:12]}"


def build_toc_node_id(*, doc_id: str, kind: str, location_id: str | None, extra: str | None = None) -> str:
    doc_part = slugify(doc_id)
    kind_part = slugify(kind)
    if location_id:
        loc_part = slugify(location_id)
        base = f"toc:v1/{doc_part}/{kind_part}/{loc_part}"
    else:
        base = f"toc:v1/{doc_part}/{kind_part}"
    if extra:
        base = f"{base}/{slugify(extra)}"
    return base


def validate_required_fields(metadata: Mapping[str, Any], required_fields: Iterable[str]) -> None:
    missing: list[str] = []
    for field in required_fields:
        if field not in metadata:
            missing.append(field)
            continue
        v = metadata.get(field)
        if v is None:
            missing.append(field)
        elif isinstance(v, str) and not v.strip():
            missing.append(field)
    if missing:
        raise ValueError(f"Missing required metadata fields: {', '.join(missing)}")


def validate_metadata_primitives(metadata: Mapping[str, Any]) -> None:
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            continue
        raise ValueError(f"Metadata value for {k!r} must be primitive (got {type(v).__name__})")


def stamp_common_metadata(
    base: MutableMapping[str, Any],
    *,
    corpus_id: str,
    doc_id: str,
    doc_version: str,
    language: str,
    source_type: str,
    source_path: str,
    source: str,
    canonical_url: str | None = None,
    source_url: str | None = None,
) -> None:
    base.setdefault("schema_version", _SCHEMA_VERSION)
    base.setdefault("corpus_id", corpus_id)
    base.setdefault("doc_id", doc_id)
    base.setdefault("doc_version", doc_version)
    base.setdefault("language", language)
    base.setdefault("source_type", source_type)
    base.setdefault("source_path", source_path)
    base.setdefault("source", source)
    if canonical_url is not None:
        base.setdefault("canonical_url", canonical_url)
    if source_url is not None:
        base.setdefault("source_url", source_url)
