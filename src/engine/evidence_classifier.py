"""Evidence type classification from chunk metadata.

Classifies retrieved chunks into evidence types (FORBIDDEN, ENFORCEMENT, DEFINITION,
SCOPE, CLASSIFICATION, UNKNOWN) based on metadata text labels. Pure function with no
state or side effects.

Extracted from policy.py (Phase 8a) — single responsibility.
"""

from typing import Any

from .types import EvidenceType


def classify_evidence_type_from_metadata(
    ref_or_chunk_metadata: dict[str, Any] | None,
) -> EvidenceType:
    """Infer evidence type from existing metadata strings.

    Must NOT hardcode specific article numbers; relies only on metadata text labels.

    Searches these metadata fields for evidence type keywords:
    - heading_path, heading_path_display, toc_path, title, location_id, source, display
    - article_title, chapter_title, section_title, annex_title (EUR-Lex structural titles)
    """

    meta = dict(ref_or_chunk_metadata or {})
    # Collect likely label fields (TOC/heading/title/location) into a single searchable text.
    parts: list[str] = []
    for key in [
        "heading_path",
        "heading_path_display",
        "toc_path",
        "title",
        "location_id",
        "source",
        "display",
        # EUR-Lex structural title fields (enriched at ingestion time).
        "article_title",
        "chapter_title",
        "section_title",
        "annex_title",
    ]:
        v = meta.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, list):
            # Some ingests may store heading_path as a list.
            parts.extend([str(x).strip() for x in v if str(x).strip()])

    hay = " ".join(parts).lower()
    if not hay:
        return EvidenceType.UNKNOWN

    forbidden_kw = ["forbud", "forbudte", "prohibited", "forbidden"]
    if any(k in hay for k in forbidden_kw):
        return EvidenceType.FORBIDDEN

    enforcement_kw = [
        "håndhævelse",
        "sanktion",
        "sanktioner",
        "bøde",
        "bøder",
        "klage",
        "tilsyn",
        "markedsovervåg",
        "complaint",
        "enforcement",
        "market surveillance",
        "surveillance authority",
        "supervision",
        "penalt",
        "fine",
        "sanction",
        "remedy",
        "redress",
    ]
    if any(k in hay for k in enforcement_kw):
        return EvidenceType.ENFORCEMENT

    definition_kw = [
        "definition",
        "definitions",
        "definitioner",
        "begreb",
        "begreber",
        "forstås ved",
        "means",
        "shall mean",
    ]
    if any(k in hay for k in definition_kw):
        return EvidenceType.DEFINITION

    scope_kw = [
        "anvendelsesområde",
        "scope",
        "applicability",
        "definition",
        "definitions",
        "omfang",
    ]
    if any(k in hay for k in scope_kw):
        return EvidenceType.SCOPE

    classification_kw = [
        "højrisiko",
        "high-risk",
        "high risk",
        "klassific",
        "classification",
        "annex",
        "bilag",
        "kategori",
    ]
    if any(k in hay for k in classification_kw):
        return EvidenceType.CLASSIFICATION

    return EvidenceType.UNKNOWN
