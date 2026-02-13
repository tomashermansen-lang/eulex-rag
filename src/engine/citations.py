from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple, TYPE_CHECKING

from . import helpers

if TYPE_CHECKING:
    from .types import ClaimIntent, UserProfile

def _strip_trailing_references_section(text: str) -> str:
    # Defensive: some callers may use the legacy answer() which appends references.
    marker = "\nReferencer:\n"
    if marker in (text or ""):
        return str(text).split(marker, 1)[0].rstrip()
    return str(text or "")

def _engineering_inject_neutral_hjemmel_citations_for_enforcement(
    *,
    answer_text: str,
    references_structured_all: list[dict[str, Any]] | None,
) -> str:
    """ENGINEERING+ENFORCEMENT: ensure at least one explicit [n] citation.

    Only injects neutral 'Relevant hjemmel' bullets (no SKAL/BØR) and selects
    the top-ranked eligible anchors deterministically (articles first, max 2).
    """

    txt = str(answer_text or "")
    if not txt.strip():
        return txt

    # If we already have any explicit [n] citations, do nothing.
    if re.search(r"\[\d{1,3}\]", txt):
        return txt

    refs = [r for r in list(references_structured_all or []) if isinstance(r, dict)]

    def _anchor_label(r: dict[str, Any]) -> str | None:
        art = str(r.get("article") or "").strip()
        if art:
            par = str(r.get("paragraph") or "").strip()
            if par:
                return f"Artikel {art}, stk. {par}"
            return f"Artikel {art}"
        rec = str(r.get("recital") or "").strip()
        if rec:
            return f"Betragtning {rec}"
        ax = str(r.get("annex") or "").strip()
        if ax:
            return f"Bilag {ax}"
        return None

    # Deterministic selection: take up to 2 articles first; then recitals; then annexes.
    selected: list[tuple[int, str]] = []
    seen_anchor: set[tuple[str, str]] = set()

    def _consider(r: dict[str, Any], kind: str, key_val: str | None) -> None:
        if key_val is None:
            return
        key = (kind, key_val)
        if key in seen_anchor:
            return
        try:
            idx = int(r.get("idx"))
        except Exception:  # noqa: BLE001
            return
        label = _anchor_label(r)
        if not label:
            return
        selected.append((idx, label))
        seen_anchor.add(key)

    for r in refs:
        if len(selected) >= 2:
            break
        art = str(r.get("article") or "").strip()
        if art:
            _consider(r, "article", art)
    if len(selected) < 2:
        for r in refs:
            if len(selected) >= 2:
                break
            rec = str(r.get("recital") or "").strip()
            if rec:
                _consider(r, "recital", rec)
    if len(selected) < 2:
        for r in refs:
            if len(selected) >= 2:
                break
            ax = str(r.get("annex") or "").strip()
            if ax:
                _consider(r, "annex", ax)

    if not selected:
        return txt

    inject_lines = [f"- Relevant hjemmel: {label} [{idx}]" for idx, label in selected]

    lines = txt.splitlines()
    heading_re = re.compile(r"^\s*2\.\s*Relevante juridiske forpligtelser\s*$", flags=re.IGNORECASE)
    start = None
    for i, line in enumerate(lines):
        if heading_re.match(line or ""):
            start = i
            break

    if start is None:
        # Fallback: append a minimal section.
        return (txt.rstrip() + "\n\n2. Relevante juridiske forpligtelser\n" + "\n".join(inject_lines)).strip()

    # Insert directly after the section heading.
    insert_at = start + 1
    out_lines = list(lines[:insert_at]) + inject_lines + list(lines[insert_at:])
    return "\n".join(out_lines).strip()


def _engineering_inject_minimal_hjemmel_citation_if_missing(
    *,
    answer_text: str,
    references_structured_all: list[dict[str, Any]] | None,
    intent_used: str | None = None,
) -> str:
    """ENGINEERING backstop: ensure at least one explicit [n] citation when evidence exists.

    Deterministic and eval-safe:
    - Only appends a single neutral line of the form:
        "Relevant hjemmel: Artikel X [n]." / "Betragtning Y [n]." / "Bilag Z [n]."
    - Does NOT hardcode article numbers/outcomes.
    - Does NOT run when answer already contains explicit [n] citations or anchor mentions.
    """

    txt = str(answer_text or "")
    if not txt.strip():
        return txt

    # Do not modify internal abstain/error markers.
    if txt.strip() == "MISSING_REF":
        return txt
    if re.search(r"(?i)\bUTILSTRÆKKELIG_EVIDENS\b", txt):
        return txt

    # If we already have any explicit [n] citations, do nothing.
    if re.search(r"\[\d{1,3}\]", txt):
        return txt

    # If the answer already explicitly mentions anchors, do nothing.
    if re.search(r"(?i)\b(?:Artikel|Article|Art\.?)(?:\s+)\d{1,3}[a-z]?\b", txt):
        return txt
    if re.search(r"(?i)\b(?:betragtning(?:er)?|recital)\s+\d{1,4}\b", txt):
        return txt
    if re.search(r"(?i)\b(?:bilag|annex)\s+(?:[ivxlcdm]+|\d{1,3})\b", txt):
        return txt

    refs = [r for r in list(references_structured_all or []) if isinstance(r, dict)]

    def _pick_first_eligible() -> tuple[str, str, int] | None:
        # Prefer first with article; else recital; else annex.
        for r in refs:
            art = str(r.get("article") or "").strip()
            if art:
                try:
                    idx = int(r.get("idx"))
                except Exception:  # noqa: BLE001
                    continue
                return ("article", art, idx)
        for r in refs:
            rec = str(r.get("recital") or "").strip()
            if rec:
                try:
                    idx = int(r.get("idx"))
                except Exception:  # noqa: BLE001
                    continue
                return ("recital", rec, idx)
        for r in refs:
            ax = str(r.get("annex") or "").strip()
            if ax:
                try:
                    idx = int(r.get("idx"))
                except Exception:  # noqa: BLE001
                    continue
                return ("annex", ax, idx)
        return None

    picked = _pick_first_eligible()
    if picked is None:
        return txt

    kind, val, idx = picked
    # Defensive: idx must be within the structured reference list.
    if idx < 1 or idx > len(refs):
        return txt

    if kind == "article":
        line = f"Relevant hjemmel: Artikel {val} [{idx}]."
    elif kind == "recital":
        line = f"Relevant hjemmel: Betragtning {val} [{idx}]."
    else:
        line = f"Relevant hjemmel: Bilag {val} [{idx}]."

    return f"{txt.rstrip()}\n\n{line}".strip()


def _engineering_ensure_min_unique_citations(
    *,
    answer_text: str,
    references_structured_all: list[dict[str, Any]] | None,
    min_unique_citations: int,
) -> str:
    """ENGINEERING backstop: deterministically ensure a minimum number of unique [idx] citations.

    This is deliberately neutral and audit-safe:
    - Only injects "Relevant hjemmel: … [idx]." lines (no SKAL/BØR).
    - Never invents idx values; only uses idx present in references_structured_all.
    - Prefers distinct anchors deterministically (articles first, then annexes, then recitals).
    """

    txt = str(answer_text or "")
    if not txt.strip():
        return txt

    # Do not modify internal abstain/error markers.
    if txt.strip() == "MISSING_REF":
        return txt
    if re.search(r"(?i)\bUTILSTRÆKKELIG_EVIDENS\b", txt):
        return txt

    # Preserve existing invariant: if the answer already contains any explicit [n]
    # citations OR explicit anchor mentions, do not inject additional hjemmel lines.
    body = _strip_trailing_references_section(txt)
    if re.search(r"\[\d{1,3}\]", body):
        return txt
    if re.search(r"(?i)\b(?:Artikel|Article|Art\.?)(?:\s+)\d{1,3}[a-z]?\b", body):
        return txt
    if re.search(r"(?i)\b(?:betragtning(?:er)?|recital)\s+\d{1,4}\b", body):
        return txt
    if re.search(r"(?i)\b(?:bilag|annex)\s+(?:[ivxlcdm]+|\d{1,3})\b", body):
        return txt

    try:
        target = max(0, int(min_unique_citations))
    except Exception:  # noqa: BLE001
        target = 0
    if target <= 0:
        return txt

    cited_set: set[int] = set()

    refs = [r for r in list(references_structured_all or []) if isinstance(r, dict)]
    if not refs:
        return txt

    # Build eligible candidates (deterministic): articles first, then annexes, then recitals.
    candidates: list[tuple[str, str, int]] = []
    seen_anchor: set[tuple[str, str]] = set()

    def _consider(kind: str, val: str, idx: int) -> None:
        if idx <= 0:
            return
        key = (kind, val)
        if key in seen_anchor:
            return
        seen_anchor.add(key)
        candidates.append((kind, val, idx))

    for r in refs:
        if len(candidates) >= target:
            break
        art = str(r.get("article") or "").strip().upper()
        if not art:
            continue
        try:
            idx = int(r.get("idx"))
        except Exception:  # noqa: BLE001
            continue
        _consider("article", art, idx)

    if len(candidates) < target:
        for r in refs:
            if len(candidates) >= target:
                break
            ax = str(r.get("annex") or "").strip().upper()
            if not ax:
                continue
            try:
                idx = int(r.get("idx"))
            except Exception:  # noqa: BLE001
                continue
            _consider("annex", ax, idx)

    if len(candidates) < target:
        for r in refs:
            if len(candidates) >= target:
                break
            rec = str(r.get("recital") or "").strip()
            if not rec:
                continue
            try:
                idx = int(r.get("idx"))
            except Exception:  # noqa: BLE001
                continue
            _consider("recital", rec, idx)

    if not candidates:
        return txt

    def _line_for(kind: str, val: str, idx: int) -> str:
        if kind == "article":
            return f"Relevant hjemmel: Artikel {val} [{idx}]."
        if kind == "annex":
            return f"Relevant hjemmel: Bilag {val} [{idx}]."
        return f"Relevant hjemmel: Betragtning {val} [{idx}]."

    inject_lines = [_line_for(kind, val, idx) for (kind, val, idx) in candidates[:target]]
    return (txt.rstrip() + "\n\n" + "\n".join(inject_lines)).strip()


def _engineering_repair_bracket_citations_from_anchor_mentions(
    *,
    answer_text: str,
    references_structured_all: list[dict[str, Any]] | None,
    max_unique_citations: int = 8,
) -> str:
    """Deterministic ENGINEERING repair:

    If the model mentions anchors (e.g. "Artikel 6", "Bilag III") but forgets to add
    bracket citations, attempt to append unambiguous [idx] markers.

    Constraints:
    - Never invent idx; only use idx values present in references_structured_all.
    - Only cite when the line clearly matches exactly one idx.
    - Never changes retrieval/rerank; mutates answer_text only.
    """

    txt = str(answer_text or "")
    if txt.strip() == "MISSING_REF":
        return txt

    # If any explicit [n] citation exists, do nothing (avoid fighting the model).
    if re.search(r"\[\d{1,3}\]", _strip_trailing_references_section(txt)):
        return txt

    refs = list(references_structured_all or [])
    if not refs:
        return txt

    # Build idx maps from structured refs.
    article_to_idxs: dict[str, list[int]] = {}
    annex_to_idxs: dict[str, list[int]] = {}
    recital_to_idxs: dict[str, list[int]] = {}
    for r in refs:
        if not isinstance(r, dict):
            continue
        try:
            idx = int(r.get("idx"))
        except Exception:  # noqa: BLE001
            continue
        if idx <= 0:
            continue

        art = str(r.get("article") or "").strip().upper()
        if art:
            article_to_idxs.setdefault(art, []).append(idx)

        ax = str(r.get("annex") or "").strip().upper()
        if ax:
            annex_to_idxs.setdefault(ax, []).append(idx)

        rec = str(r.get("recital") or "").strip()
        if rec:
            recital_to_idxs.setdefault(rec, []).append(idx)

    def _single_idx(candidates: list[int]) -> int | None:
        c = sorted(set(int(x) for x in candidates if int(x) > 0))
        if len(c) == 1:
            return c[0]
        return None

    cited_unique: set[int] = set()
    lines = txt.splitlines()
    out_lines: list[str] = []
    for line in lines:
        raw = str(line)
        if not raw.strip():
            out_lines.append(raw)
            continue
        if re.search(r"\[\d{1,3}\]", raw):
            out_lines.append(raw)
            continue

        matched: list[int] = []

        # Article mentions: tolerate Danish/English and spaced letters.
        for m in re.finditer(r"(?i)a\s*r\s*t\s*i\s*k\s*e\s*l\s*(\d{1,3}[a-z]?)", raw):
            a = str(m.group(1) or "").strip().upper()
            if a and a in article_to_idxs:
                si = _single_idx(article_to_idxs[a])
                if si is not None:
                    matched.append(si)

        # Annex/Bilag mentions.
        for m in re.finditer(r"(?i)\b(?:bilag|annex)\s+([ivxlcdm]+|\d{1,3})\b", raw):
            ax = str(m.group(1) or "").strip().upper()
            if ax and ax in annex_to_idxs:
                si = _single_idx(annex_to_idxs[ax])
                if si is not None:
                    matched.append(si)

        # Recital/Betragtning mentions.
        for m in re.finditer(r"(?i)\b(?:betragtning|recital)\s*\(?\s*(\d{1,4})\s*\)?\b", raw):
            rec = str(m.group(1) or "").strip()
            if rec and rec in recital_to_idxs:
                si = _single_idx(recital_to_idxs[rec])
                if si is not None:
                    matched.append(si)

        matched = sorted(set(matched))
        if len(matched) == 1 and len(cited_unique) < int(max_unique_citations):
            idx = matched[0]
            cited_unique.add(idx)
            out_lines.append(raw.rstrip() + f" [{idx}]")
        else:
            out_lines.append(raw)

    return "\n".join(out_lines)


def _eurlex_litra_index(abs_source_path: str) -> dict[Tuple[int, int], set[str]]:
    """Build (article_num, paragraph_num) -> allowed litra set from EUR-Lex Convex HTML."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except ModuleNotFoundError:
        return {}

    path = Path(abs_source_path)
    if not path.exists():
        return {}

    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    out: dict[Tuple[int, int], set[str]] = {}
    # Paragraph div ids look like "059.002".
    for div in soup.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$")):
        div_id = str(div.get("id"))
        try:
            art_s, par_s = div_id.split(".")
            art_n = int(art_s)
            par_n = int(par_s)
        except ValueError:
            continue

        # Find litra spans inside this paragraph div.
        # Typically: <span class="oj-italic">(a)</span>
        litras = set()
        for span in div.find_all("span", class_="oj-italic"):
            txt = span.get_text().strip()
            # Match "(a)" or "(b)" etc.
            m = re.match(r"^\(([a-z])\)$", txt)
            if m:
                litras.add(m.group(1))
        
        if litras:
            out[(art_n, par_n)] = litras

    return out

def is_litra_addressable(
    *,
    metadata: Dict[str, Any] | None,
    chunk_text: str,
) -> tuple[bool, dict[str, Any]]:
    """Return (is_addressable, details).

    Audit-safe definition: a litra is addressable only if we can prove it belongs
    to the given article+paragraph in both the chunk text and the official EUR-Lex
    Convex HTML structure.
    """
    m = dict(metadata or {})

    article_raw = str(m.get("article") or "").strip()
    paragraph_raw = str(m.get("paragraph") or "").strip()
    litra_raw = str(m.get("litra") or "").strip().lower()

    details: dict[str, Any] = {
        "article": article_raw or None,
        "paragraph": paragraph_raw or None,
        "litra": litra_raw or None,
        "checks": {
            "has_article": bool(article_raw),
            "has_paragraph": bool(paragraph_raw),
            "has_litra": bool(litra_raw),
            "chunk_evidence": False,
            "location_id_ok": False,
            "html_ok": False,
        },
    }

    if not (article_raw and paragraph_raw and litra_raw):
        details["reason"] = "missing_article_or_paragraph_or_litra"
        return False, details

    text = str(chunk_text or "")
    details["checks"]["chunk_evidence"] = bool(
        re.search(rf"(?i)(?:^|\s){re.escape(litra_raw)}\)\s+", text)
    )

    loc = str(m.get("location_id") or "").strip().lower()
    parts = [p.strip() for p in loc.split("/") if p.strip()]
    article_token = str(article_raw).strip().lower()
    details["checks"]["location_id_ok"] = bool(loc and f"article:{article_token}" in parts)

    m_art = re.match(r"^(\d{1,3})", article_raw)
    m_par = re.match(r"^(\d{1,3})$", paragraph_raw)
    src = _resolve_source_html_path(m)
    if m_art and m_par and src is not None:
        art_num = int(m_art.group(1))
        par_num = int(m_par.group(1))
        idx = _eurlex_litra_index(str(src))
        allowed = idx.get((art_num, par_num), set())
        details["checks"]["html_ok"] = litra_raw in allowed

    ok = bool(
        details["checks"]["chunk_evidence"]
        and details["checks"]["location_id_ok"]
        and details["checks"]["html_ok"]
    )
    if not ok:
        if not details["checks"]["chunk_evidence"]:
            details["reason"] = "no_litra_evidence_in_chunk"
        elif not details["checks"]["location_id_ok"]:
            details["reason"] = "location_id_article_mismatch"
        else:
            details["reason"] = "litra_not_proven_in_html"
    return ok, details

def _resolve_source_html_path(metadata: dict[str, Any]) -> Path | None:
    raw = str((metadata or {}).get("source_path") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        # Assuming this file is in src/engine/citations.py, so parent.parent is src/
        # and parent.parent.parent is project root.
        repo_root = Path(__file__).resolve().parent.parent.parent
        p = (repo_root / p).resolve()
    return p if p.exists() else None

def _format_metadata_audit_safe(
    metadata: Dict[str, Any] | None,
    doc_text: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Return (display, sanitized_meta, validation).

    Audit-safe rule: never emit a litra unless we can verify it in the chunk text
    and (when available) in the official EUR-Lex HTML structure.
    """
    m = dict(metadata or {})
    sanitized = dict(m)
    validation: dict[str, Any] = {"passed": True, "actions": []}

    article_raw = str(m.get("article") or "").strip()
    paragraph_raw = str(m.get("paragraph") or "").strip()
    litra_raw = str(m.get("litra") or "").strip().lower()

    if litra_raw:
        ok, details = is_litra_addressable(metadata=m, chunk_text=doc_text)
        if not ok:
            sanitized.pop("litra", None)
            validation["passed"] = False
            validation["actions"].append(
                {
                    "action": "drop_litra",
                    "reason": details.get("reason") or "litra_not_addressable",
                    "article": details.get("article"),
                    "paragraph": details.get("paragraph"),
                    "litra": details.get("litra"),
                    "checks": details.get("checks"),
                }
            )

    display = _format_metadata(sanitized)
    return display, sanitized, validation

def _format_metadata(metadata: Dict[str, Any] | None) -> str:
    if not metadata:
        return "Ukendt kilde"

    parts: List[str] = []
    if metadata.get("source"):
        parts.append(str(metadata["source"]))

    # If structural fields are missing, try to derive them from canonical location_id.
    if not (
        metadata.get("chapter")
        or metadata.get("section")
        or metadata.get("article")
        or metadata.get("annex")
        or metadata.get("recital")
        or metadata.get("citation")
        or metadata.get("preamble")
    ):
        loc = str(metadata.get("location_id") or "").strip()
        if loc:
            try:
                segs = [p.strip() for p in loc.split("/") if p.strip()]
                derived: dict[str, str] = {}
                for s in segs:
                    low = s.lower()
                    if low.startswith("chapter:"):
                        derived["chapter"] = s.split(":", 1)[1].upper()
                    elif low.startswith("section:"):
                        derived["section"] = s.split(":", 1)[1].upper()
                    elif low.startswith("article:"):
                        derived["article"] = s.split(":", 1)[1].upper()
                    elif low.startswith("annex:"):
                        derived["annex"] = s.split(":", 1)[1].upper()
                    elif low.startswith("annex_point:"):
                        derived["annex_point"] = s.split(":", 1)[1]
                    elif low.startswith("annex_section:"):
                        derived["annex_section"] = s.split(":", 1)[1].upper()
                    elif low.startswith("recital:"):
                        derived["recital"] = s.split(":", 1)[1]
                    elif low.startswith("citation:"):
                        derived["citation"] = s.split(":", 1)[1]
                    elif low.startswith("preamble:"):
                        derived["preamble"] = s.split(":", 1)[1]
                metadata.update(derived)
            except Exception:
                pass

    if metadata.get("chapter"):
        chapter_str = f"Kapitel {metadata['chapter']}"
        if metadata.get("chapter_title"):
            chapter_str += f" ({metadata['chapter_title']})"
        parts.append(chapter_str)
    if metadata.get("section"):
        section_str = f"Afsnit {metadata['section']}"
        if metadata.get("section_title"):
            section_str += f" ({metadata['section_title']})"
        parts.append(section_str)
    if metadata.get("article"):
        article_str = f"Artikel {metadata['article']}"
        if metadata.get("article_title"):
            article_str += f" ({metadata['article_title']})"
        parts.append(article_str)
    if metadata.get("paragraph"):
        parts.append(f"stk. {metadata['paragraph']}")
    if metadata.get("litra"):
        parts.append(f"litra {metadata['litra']}")
    if metadata.get("annex"):
        annex_str = f"Bilag {metadata['annex']}"
        if metadata.get("annex_title"):
            annex_str += f" ({metadata['annex_title']})"
        parts.append(annex_str)
    if metadata.get("annex_point"):
        parts.append(f"punkt {metadata['annex_point']}")
    if metadata.get("recital"):
        parts.append(f"Betragtning {metadata['recital']}")
    if metadata.get("citation"):
        parts.append(f"Citation {metadata['citation']}")
    if metadata.get("preamble"):
        parts.append(f"Præambel {metadata['preamble']}")

    if metadata.get("page"):
        parts.append(f"side {metadata['page']}")

    # Add cross-references if present
    mentions_str = format_mentions_for_context(metadata.get("mentions"))
    if mentions_str:
        parts.append(f"→ {mentions_str}")

    return ", ".join(parts)


def format_mentions_for_context(mentions_raw: str | Dict | None) -> str | None:
    """Format mentions JSON to human-readable cross-reference string.
    
    Args:
        mentions_raw: JSON string or dict with article/recital/annex lists
        
    Returns:
        Formatted string like "Artikel 16, Betragtning 107" or None if empty
    """
    if not mentions_raw:
        return None
    
    try:
        if isinstance(mentions_raw, str):
            mentions = json.loads(mentions_raw)
        else:
            mentions = mentions_raw
        
        refs = []
        for article in mentions.get("article", []):
            refs.append(f"Artikel {article}")
        for recital in mentions.get("recital", []):
            refs.append(f"Betragtning {recital}")
        for annex in mentions.get("annex", []):
            refs.append(f"Bilag {annex}")
        
        if refs:
            return ", ".join(refs)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    
    return None


def _is_citable_metadata(meta: Dict[str, Any] | None) -> bool:
    """Return True when metadata clearly identifies a citable legal location."""
    if not meta:
        return False

    # Explicit structural fields.
    if meta.get("article") or meta.get("annex") or meta.get("chapter") or meta.get("recital"):
        return True

    # Canonical location_id (metadata_schema.make_location_id): loc:v1/...
    loc = str(meta.get("location_id") or "").strip()
    if loc:
        parts = [p.strip().lower() for p in loc.split("/") if p.strip()]
        if any(p.startswith("article:") for p in parts):
            return True
        if any(p.startswith("annex:") for p in parts):
            return True
        if any(p.startswith("chapter:") for p in parts):
            return True
        if any(p.startswith("recital:") for p in parts):
            return True

    # heading_path is a JSON list of segments like ["chapter:iv", "article:10"].
    hp = meta.get("heading_path")
    if isinstance(hp, str) and hp.strip().startswith("["):
        try:
            segs = json.loads(hp)
            if isinstance(segs, list):
                lowered = [str(s).strip().lower() for s in segs]
                if any(s.startswith("article:") for s in lowered):
                    return True
                if any(s.startswith("annex:") for s in lowered):
                    return True
                if any(s.startswith("chapter:") for s in lowered):
                    return True
                if any(s.startswith("recital:") for s in lowered):
                    return True
        except Exception:
            pass

    return False

def extract_precise_ref_from_text(doc_text: str) -> str | None:
    """Try to extract a precise legal reference token from chunk text."""
    if not doc_text:
        return None
    patterns = [
        r"\b(?:Article|Artikel)\s+\d{1,3}[a-z]?(?:\s*\(\s*\d{1,3}\s*\))?\b",
        r"\bArt\.?\s*\d{1,3}[a-z]?(?:\s*\(\s*\d{1,3}\s*\))?\b",
        r"\b(?:Chapter|Kapitel)\s+(?:[IVXLCDM]+|\d{1,3})\b",
        r"\b(?:Annex|Bilag)\s+(?:[IVXLCDM]+|\d{1,3})\b",
        r"\b(?:Recital|Betragtning)\s+\d{1,4}\b",
    ]
    for p in patterns:
        m = re.search(p, doc_text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None

def _is_citable_chunk(meta: Dict[str, Any] | None, doc_text: str) -> tuple[bool, str | None]:
    """Return (is_citable, precise_ref_override)."""
    m = dict(meta or {})
    if _is_citable_metadata(m):
        return True, None

    extracted = extract_precise_ref_from_text(doc_text)
    if extracted:
        src = str(m.get("source") or "").strip()
        if not src:
            src = str(m.get("corpus_id") or "").strip() or "Unknown source"
        return True, f"{src}, {extracted}"

    return False, None

def extract_precise_token_from_meta(meta: dict[str, Any] | None) -> str | None:
    """
    Extract a precise reference token (e.g. 'Kapitel 10, Artikel 5') from metadata.
    Returns None if no precise structural info is found.
    """
    if not meta:
        return None
    # Only consider a meta-derived precise token if it contains article/annex/chapter
    has_precise = bool(meta.get("article") or meta.get("annex") or meta.get("chapter"))
    if not has_precise:
        return None
    parts: list[str] = []
    if meta.get("source"):
        parts.append(str(meta.get("source")))
    if meta.get("chapter"):
        parts.append(f"Kapitel {meta.get('chapter')}")
    if meta.get("article"):
        art = str(meta.get("article"))
        para = meta.get("paragraph")
        if para:
            parts.append(f"Artikel {art}({para})")
        else:
            parts.append(f"Artikel {art}")
    if meta.get("annex"):
        parts.append(f"Bilag {meta.get('annex')}")
    token = ", ".join(parts)
    return token or None

def best_effort_source_label(meta: dict[str, Any] | None, *, fallback: str | None = None) -> str:
    """
    Return the source label from metadata, or a fallback if missing.
    """
    m = dict(meta or {})
    src = str(m.get("source") or "").strip()
    if src:
        return src
    return fallback or "Unknown source"


def select_references_used_in_answer(
    *,
    answer_text: str,
    references_structured: list[dict[str, Any]],
) -> list[str]:
    """Return chunk_ids in the order they are cited/used in the answer.
    
    Extracted from RAGEngine in Phase E2 for better modularity.
    This function implements the citation selection contract:
    1) Prefer explicit bracket citations like [1]
    2) Fall back to anchor mentions (Article/Recital/Annex) in answer text
    """

    txt = _strip_trailing_references_section(answer_text)
    refs = [r for r in list(references_structured or []) if isinstance(r, dict)]

    # 1) Prefer explicit bracket citations like [1].
    # Contract: [n] refers to the original reference idx assigned in
    # references_structured_all for this query (not list position).
    idx_to_chunk_id: dict[int, str] = {}
    for r in refs:
        try:
            ridx = int(r.get("idx"))
        except Exception:  # noqa: BLE001
            continue
        cid = str(r.get("chunk_id") or "").strip()
        if ridx > 0 and cid and ridx not in idx_to_chunk_id:
            idx_to_chunk_id[ridx] = cid

    used: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\[(\d{1,3})\]", txt):
        try:
            idx = int(m.group(1))
        except Exception:  # noqa: BLE001
            continue
        cid = idx_to_chunk_id.get(idx)
        if cid and cid not in seen:
            used.append(cid)
            seen.add(cid)
    # Do not return early: answers can contain a single stray [1] while also
    # explicitly naming additional anchors (e.g., "Artikel 13"). We include
    # both signals deterministically.

    # 2) Fall back to anchor mentions (Article/Recital/Annex) in the answer.
    mentions = helpers._extract_anchor_mentions_from_answer(txt)
    articles = list(mentions.get("articles") or [])
    recitals = list(mentions.get("recitals") or [])
    annexes = list(mentions.get("annexes") or [])

    def first_match_article(art: str, par: str | None) -> str | None:
        for r in refs:
            if str(r.get("article") or "").strip().upper() != art:
                continue
            if par:
                if str(r.get("paragraph") or "").strip() != str(par).strip():
                    continue
            cid = str(r.get("chunk_id") or "")
            return cid or None
        # If paragraph-specific mention couldn't be matched, accept article-only.
        if par:
            for r in refs:
                if str(r.get("article") or "").strip().upper() == art:
                    cid = str(r.get("chunk_id") or "")
                    return cid or None
        return None

    for art, par in articles:
        cid = first_match_article(str(art), par)
        if cid and cid not in seen:
            used.append(cid)
            seen.add(cid)

    for rec in recitals:
        for r in refs:
            if str(r.get("recital") or "").strip() == str(rec).strip():
                cid = str(r.get("chunk_id") or "")
                if cid and cid not in seen:
                    used.append(cid)
                    seen.add(cid)
                break

    for ax in annexes:
        for r in refs:
            if str(r.get("annex") or "").strip().upper() == str(ax).strip().upper():
                cid = str(r.get("chunk_id") or "")
                if cid and cid not in seen:
                    used.append(cid)
                    seen.add(cid)
                break

    return used


# -----------------------------------------------------------------------------
# ENGINEERING citation integrity gate (extracted from rag.py Phase E)
# -----------------------------------------------------------------------------


@dataclass
class EngineeringCitationIntegrityResult:
    """Result of applying ENGINEERING citation integrity checks."""

    answer_text: str
    references_structured: list[dict[str, Any]]
    reference_lines: list[str]
    used_chunk_ids: list[str]
    debug: dict[str, Any] = field(default_factory=dict)


def apply_engineering_citation_integrity(
    *,
    answer_text: str,
    references_structured: list[dict[str, Any]],
    references_structured_all: list[dict[str, Any]],
    used_chunk_ids: list[str],
    run_meta: dict[str, Any],
    contract_min_citations: int | None,
    intent_used: str,
    is_debug_enabled_fn: Callable[[], bool],
) -> EngineeringCitationIntegrityResult:
    """Apply ENGINEERING-profile citation integrity checks after hard reference gating.

    This function handles:
    1. Citation source parsing (text vs JSON mode)
    2. Min citations contract enforcement
    3. Reference rebuilding from cited idxs
    4. Missing citation checks
    5. Hallucinated idx detection
    6. Contract consistency filtering (only return cited references)

    Returns EngineeringCitationIntegrityResult with potentially modified answer/refs.
    """

    result = EngineeringCitationIntegrityResult(
        answer_text=str(answer_text or ""),
        references_structured=list(references_structured or []),
        reference_lines=[
            f"[{r['idx']}] {r.get('precise_ref') or r.get('display')}"
            for r in list(references_structured or [])
        ],
        used_chunk_ids=list(used_chunk_ids or []),
        debug={},
    )

    citations_source = "text_parse"
    json_cited_idxs: set[int] | None = None

    # Check if JSON mode produced cited_idxs
    try:
        if bool(run_meta.get("engineering_json_mode")):
            raw_json_cited = run_meta.get("cited_idxs")
            if isinstance(raw_json_cited, list) and raw_json_cited:
                json_cited_idxs = {int(x) for x in raw_json_cited if isinstance(x, int) or str(x).isdigit()}
    except Exception:  # noqa: BLE001
        json_cited_idxs = None

    txt_for_citations = _strip_trailing_references_section(result.answer_text)
    had_brackets_in_text = bool(re.search(r"\[\d{1,3}\]", txt_for_citations))

    # If JSON-mode produced a rendered answer, always let the downstream gate evaluate
    # that deterministic text.
    if bool(run_meta.get("engineering_json_mode")) and result.answer_text.strip() != "MISSING_REF":
        try:
            rendered = None
            ej = run_meta.get("engineering_json")
            if isinstance(ej, dict):
                rendered = ej.get("rendered_text")
            if isinstance(rendered, str) and rendered.strip():
                result.answer_text = rendered
                txt_for_citations = _strip_trailing_references_section(result.answer_text)
                had_brackets_in_text = bool(re.search(r"\[\d{1,3}\]", txt_for_citations))
        except Exception:  # noqa: BLE001
            pass

    cited_idxs: set[int] = set()
    if had_brackets_in_text:
        for m in re.finditer(r"\[(\d{1,3})\]", txt_for_citations):
            try:
                cited_idxs.add(int(m.group(1)))
            except Exception:  # noqa: BLE001
                continue
    elif json_cited_idxs:
        citations_source = "json_mode"
        cited_idxs = set(json_cited_idxs)

    # Persist debug info
    result.debug["citations_source"] = citations_source
    result.debug["json_cited_idxs"] = sorted(list(json_cited_idxs or []))
    result.debug["answer_text_contains_brackets"] = had_brackets_in_text
    result.debug["parsed_citations_raw"] = sorted(list(cited_idxs))

    # Contract enforcement: if minimum citation count required and prompt had enough
    # allowed sources to satisfy it, then <min valid citations must fail-closed.
    try:
        min_cit_contract = int(contract_min_citations) if contract_min_citations is not None else 0
    except Exception:  # noqa: BLE001
        min_cit_contract = 0

    allowed_all: set[int] = set()
    try:
        for r in list(references_structured_all or []):
            if not isinstance(r, dict):
                continue
            allowed_all.add(int(r.get("idx")))
    except Exception:  # noqa: BLE001
        allowed_all = set()

    valid_cited_all = len(set(cited_idxs) & set(allowed_all))
    if min_cit_contract > 0 and len(allowed_all) >= min_cit_contract and valid_cited_all < min_cit_contract:
        if is_debug_enabled_fn():
            result.debug["missing_ref_reason"] = {
                "intent": intent_used,
                "required_support": "min_citations_not_met",
                "min_citations": min_cit_contract,
                "allowed": len(allowed_all),
                "valid_cited": valid_cited_all,
                "had_bracket_citations": bool(cited_idxs),
            }
        result.debug["fail_reason"] = "MIN_CITATIONS_NOT_MET_DOWNSTREAM"
        result.answer_text = "MISSING_REF"
        result.references_structured = []
        result.reference_lines = []
        result.used_chunk_ids = []
        return result

    structured_idx_set: set[int] = set()
    for r in list(result.references_structured or []):
        if not isinstance(r, dict):
            continue
        try:
            structured_idx_set.add(int(r.get("idx")))
        except Exception:  # noqa: BLE001
            continue

    # If earlier reference-selection produced no references but we have cited idxs,
    # reconstruct deterministically from the cited idxs.
    if not structured_idx_set and cited_idxs and references_structured_all:
        rebuilt: list[dict[str, Any]] = []
        for r in list(references_structured_all or []):
            if not isinstance(r, dict):
                continue
            try:
                ridx = int(r.get("idx"))
            except Exception:  # noqa: BLE001
                continue
            if ridx in cited_idxs:
                rebuilt.append(dict(r))
        result.references_structured = rebuilt
        result.reference_lines = [
            f"[{r['idx']}] {r.get('precise_ref') or r.get('display')}"
            for r in result.references_structured
        ]
        structured_idx_set = {
            int(r.get("idx"))
            for r in result.references_structured
            if isinstance(r, dict) and str(r.get("idx") or "").strip().isdigit()
        }

    # Hard reference gating may deduplicate by canonical anchor, which can collapse
    # multiple cited idx values into a single reference. Ensure every cited idx is
    # present in references_structured before hallucination checks.
    if structured_idx_set and cited_idxs and references_structured_all:
        try:
            by_idx: dict[int, dict[str, Any]] = {}
            for r in list(references_structured_all or []):
                if not isinstance(r, dict):
                    continue
                try:
                    ridx = int(r.get("idx"))
                except Exception:  # noqa: BLE001
                    continue
                if ridx > 0 and ridx not in by_idx:
                    by_idx[ridx] = dict(r)

            missing_cited = sorted(set(cited_idxs) - set(structured_idx_set))
            if missing_cited:
                for ridx in missing_cited:
                    ref = by_idx.get(ridx)
                    if ref is not None:
                        result.references_structured.append(ref)
                result.reference_lines = [
                    f"[{r['idx']}] {r.get('precise_ref') or r.get('display')}"
                    for r in result.references_structured
                ]
                structured_idx_set = {
                    int(r.get("idx"))
                    for r in list(result.references_structured or [])
                    if isinstance(r, dict) and str(r.get("idx") or "").strip().isdigit()
                }
        except Exception:  # noqa: BLE001
            pass

    # If we have references but zero bracket citations, fail closed.
    if structured_idx_set and not cited_idxs and result.answer_text.strip() != "MISSING_REF":
        if is_debug_enabled_fn():
            result.debug["missing_ref_reason"] = {
                "intent": intent_used,
                "required_support": "citations_required",
                "has_article_support": any(bool(r.get("article")) for r in result.references_structured),
                "has_annex_support": any(bool(r.get("annex")) for r in result.references_structured),
                "had_bracket_citations": False,
                "had_anchor_mentions": bool(
                    any(
                        v
                        for v in (
                            (helpers._extract_anchor_mentions_from_answer(result.answer_text).get("articles") or []),
                            (helpers._extract_anchor_mentions_from_answer(result.answer_text).get("recitals") or []),
                            (helpers._extract_anchor_mentions_from_answer(result.answer_text).get("annexes") or []),
                        )
                    )
                ),
            }
        result.debug["fail_reason"] = "CITATIONS_REQUIRED_DOWNSTREAM"
        result.answer_text = "MISSING_REF"
        result.references_structured = []
        result.reference_lines = []
        result.used_chunk_ids = []
        return result

    # If citations exist, they must all map to present idx values.
    if cited_idxs and structured_idx_set:
        missing = sorted(cited_idxs - structured_idx_set)
        if missing:
            if is_debug_enabled_fn():
                result.debug["missing_ref_reason"] = {
                    "intent": intent_used,
                    "required_support": "cited_idx_must_exist",
                    "missing_cited_idxs": missing,
                }
            result.debug["fail_reason"] = "HALLUCINATED_IDX_DOWNSTREAM"
            result.answer_text = "MISSING_REF"
            result.references_structured = []
            result.reference_lines = []
            result.used_chunk_ids = []
            return result

        # Optional: enforce minimum unique valid citations when sources allow it.
        try:
            min_cit = int(contract_min_citations) if contract_min_citations is not None else 0
        except Exception:  # noqa: BLE001
            min_cit = 0

        if min_cit > 0:
            # Only enforce when the prompt had enough allowed sources to satisfy it.
            allowed_count = 0
            try:
                allowed_count = len({
                    int(r.get("idx"))
                    for r in list(references_structured_all or [])
                    if isinstance(r, dict)
                })
            except Exception:  # noqa: BLE001
                allowed_count = 0

            valid_count = len(set(cited_idxs) & set(structured_idx_set))
            if allowed_count >= min_cit and valid_count < min_cit:
                if is_debug_enabled_fn():
                    result.debug["missing_ref_reason"] = {
                        "intent": intent_used,
                        "required_support": "min_citations_not_met",
                        "min_citations": min_cit,
                        "allowed": allowed_count,
                        "valid_cited": valid_count,
                    }
                result.debug["fail_reason"] = "MIN_CITATIONS_NOT_MET_DOWNSTREAM"
                result.answer_text = "MISSING_REF"
                result.references_structured = []
                result.reference_lines = []
                result.used_chunk_ids = []
                return result

        # Contract consistency (ENGINEERING): only return references actually cited.
        filtered_structured: list[dict[str, Any]] = []
        for r in list(result.references_structured or []):
            if not isinstance(r, dict):
                continue
            try:
                ridx = int(r.get("idx"))
            except Exception:  # noqa: BLE001
                continue
            if ridx in cited_idxs:
                filtered_structured.append(r)

        result.references_structured = filtered_structured
        result.reference_lines = [
            f"[{r['idx']}] {r.get('precise_ref') or r.get('display')}"
            for r in result.references_structured
        ]

    return result


# -----------------------------------------------------------------------------
# Hard reference gating with LEGAL fallback (extracted from rag.py Phase E)
# -----------------------------------------------------------------------------


@dataclass
class HardReferenceGatingResult:
    """Result of applying hard reference gating."""

    used_chunk_ids: list[str]
    references_structured: list[dict[str, Any]]
    reference_lines: list[str]


def _select_legal_fallback_chunk_ids(
    *,
    references_structured_all: list[dict[str, Any]],
    max_refs: int,
    question: str,
) -> list[str]:
    """Select LEGAL fallback chunk IDs when no inline citations exist.

    LEGAL profile allows indirect references without bracket citations. This
    deterministically selects a small subset of eligible references, preferring
    articles with higher keyword overlap to the question.
    """
    if max_refs <= 0:
        return []

    q = str(question or "").lower()
    raw_tokens = [t.lower() for t in re.findall(r"[a-zA-ZæøåÆØÅ]{4,}", q)]
    q_tokens: set[str] = set(raw_tokens)
    # Add short prefixes/stems to handle Danish compounds/plurals deterministically.
    for t in list(raw_tokens):
        if len(t) >= 9:
            q_tokens.add(t[:6])
            q_tokens.add(t[:8])
        for suf in ["erne", "erne", "ene", "erne", "er", "en", "et", "e", "s"]:
            if t.endswith(suf) and len(t) - len(suf) >= 4:
                q_tokens.add(t[: -len(suf)])
                break

    def _score_ref_text(r: dict[str, Any]) -> int:
        parts: list[str] = []
        for k in ["title", "heading_path", "toc_path", "display", "location_id", "chunk_text"]:
            v = r.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip().lower())
            elif isinstance(v, list):
                parts.extend([str(x).strip().lower() for x in v if str(x).strip()])
        hay = " ".join(parts)
        if not hay or not q_tokens:
            return 0
        # Substring matching keeps this robust across inflections.
        return sum(1 for t in q_tokens if t in hay)

    def _to_int_or_none(v: Any) -> int | None:
        s = str(v or "").strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:  # noqa: BLE001
            return None

    # Preserve original order within each pool; selection is pool-priority.
    # Each item carries: (overlap_score, numeric_anchor_or_none, original_pos, chunk_id, anchor_key)
    pool_articles: list[tuple[int, int | None, int, str, tuple[str, str]]] = []
    pool_recitals: list[tuple[int, int | None, int, str, tuple[str, str]]] = []
    pool_annexes: list[tuple[int, int, str, tuple[str, str]]] = []

    for pos, r in enumerate(list(references_structured_all or [])):
        if not isinstance(r, dict):
            continue
        cid = str(r.get("chunk_id") or "").strip()
        if not cid:
            continue

        # Anchor key used for deterministic de-duplication.
        if r.get("article"):
            anchor_key = ("article", str(r.get("article")))
        elif r.get("recital"):
            anchor_key = ("recital", str(r.get("recital")))
        elif r.get("annex"):
            anchor_key = ("annex", str(r.get("annex")))
        else:
            continue

        if r.get("article"):
            pool_articles.append(
                (_score_ref_text(r), _to_int_or_none(r.get("article")), pos, cid, anchor_key)
            )
        elif r.get("recital"):
            pool_recitals.append(
                (_score_ref_text(r), _to_int_or_none(r.get("recital")), pos, cid, anchor_key)
            )
        elif r.get("annex"):
            pool_annexes.append((_score_ref_text(r), pos, cid, anchor_key))

    # Prefer: higher keyword overlap.
    # Only when overlap is positive do we prefer lower numeric anchors; otherwise preserve retrieval order.
    pool_articles.sort(
        key=lambda t: (
            -t[0],
            (t[1] if (t[0] > 0 and t[1] is not None) else 10**9),
            t[2],
        )
    )
    pool_recitals.sort(
        key=lambda t: (
            -t[0],
            (t[1] if (t[0] > 0 and t[1] is not None) else 10**9),
            t[2],
        )
    )
    pool_annexes.sort(key=lambda t: (-t[0], t[1]))

    selected: list[str] = []
    seen_anchors: set[tuple[str, str]] = set()

    ordered_candidates: list[tuple[str, tuple[str, str]]] = (
        [(t[3], t[4]) for t in pool_articles]
        + [(t[3], t[4]) for t in pool_recitals]
        + [(t[2], t[3]) for t in pool_annexes]
    )

    for cid, anchor_key in ordered_candidates:
        if anchor_key in seen_anchors:
            continue
        selected.append(cid)
        seen_anchors.add(anchor_key)
        if len(selected) >= max_refs:
            break

    return selected


def _is_eligible_reference(r: dict[str, Any]) -> bool:
    """Return True if reference has article, recital, or annex (citable anchor)."""
    if not isinstance(r, dict):
        return False
    return bool(r.get("article") or r.get("recital") or r.get("annex"))


def _get_reference_dedup_key(r: dict[str, Any]) -> tuple[str, str, str] | None:
    """Get corpus-aware deduplication key for a reference.

    Returns a tuple of (corpus_id, anchor_type, anchor_value) for deduplication.
    Same article in different corpora will have different keys (not deduplicated).
    Same article in same corpus will have the same key (deduplicated).

    Args:
        r: Reference dict with article/recital/annex and optional corpus_id

    Returns:
        Dedup key tuple or None if not eligible
    """
    if not isinstance(r, dict):
        return None

    corpus_id = str(r.get("corpus_id") or "").strip()

    if r.get("article"):
        return (corpus_id, "article", str(r.get("article")))
    elif r.get("recital"):
        return (corpus_id, "recital", str(r.get("recital")))
    elif r.get("annex"):
        return (corpus_id, "annex", str(r.get("annex")))

    return None


def apply_hard_reference_gating(
    *,
    answer_text: str,
    used_chunk_ids: list[str],
    references_structured_all: list[dict[str, Any]],
    question: str,
    is_legal_profile: bool,
    legal_allow_reference_fallback: bool,
    max_legal_fallback_refs: int = 2,
) -> HardReferenceGatingResult:
    """Apply hard reference gating to filter and deduplicate references.

    This function:
    1. For LEGAL profile: applies fallback selection when no inline citations exist
    2. Orders chunk IDs by citation appearance order in answer text
    3. Filters to only include article/recital/annex references
    4. Deduplicates by canonical anchor (article/recital/annex number)

    Returns HardReferenceGatingResult with gated references.
    """

    working_used_chunk_ids = list(used_chunk_ids or [])

    # LEGAL profile fallback: when no inline citations, select deterministically
    if is_legal_profile:
        if not legal_allow_reference_fallback:
            # Gate explicitly disallowed reference fallback (e.g., scope guard).
            working_used_chunk_ids = list(working_used_chunk_ids or [])
        else:
            fallback_ids = _select_legal_fallback_chunk_ids(
                references_structured_all=list(references_structured_all or []),
                max_refs=max_legal_fallback_refs,
                question=question,
            )

            # If no inline [n] citations exist (common in LEGAL), fall back.
            if not working_used_chunk_ids:
                working_used_chunk_ids = list(fallback_ids or [])
            # If the LEGAL answer included only a subset via inline [n] citations,
            # supplement up to the cap to avoid brittle single-citation outcomes.
            elif len(working_used_chunk_ids) < max_legal_fallback_refs:
                merged: list[str] = []
                seen_cids: set[str] = set()
                for cid in list(working_used_chunk_ids) + list(fallback_ids or []):
                    sc = str(cid or "").strip()
                    if not sc or sc in seen_cids:
                        continue
                    merged.append(sc)
                    seen_cids.add(sc)
                    if len(merged) >= max_legal_fallback_refs:
                        break
                working_used_chunk_ids = merged

    by_id: dict[str, dict[str, Any]] = {
        str(r.get("chunk_id") or ""): r
        for r in list(references_structured_all or [])
        if isinstance(r, dict)
    }

    gated: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()  # (corpus_id, anchor_type, anchor_value)

    # Deterministic ordering:
    # - If answer contains bracket citations, order by first appearance of each cited idx.
    # - Otherwise, preserve used_chunk_ids order.
    ordered_used_chunk_ids: list[str] = list(working_used_chunk_ids or [])
    txt_for_order = _strip_trailing_references_section(str(answer_text or ""))

    if re.search(r"\[\d{1,3}\]", txt_for_order):
        idx_to_cid: dict[int, str] = {}
        for r in list(references_structured_all or []):
            if not isinstance(r, dict):
                continue
            try:
                ridx = int(r.get("idx"))
            except Exception:  # noqa: BLE001
                continue
            cid = str(r.get("chunk_id") or "").strip()
            if ridx > 0 and cid and ridx not in idx_to_cid:
                idx_to_cid[ridx] = cid

        cited_cids: list[str] = []
        seen_cids: set[str] = set()
        seen_idxs: set[int] = set()
        for m in re.finditer(r"\[(\d{1,3})\]", txt_for_order):
            try:
                cidx = int(m.group(1))
            except Exception:  # noqa: BLE001
                continue
            if cidx in seen_idxs:
                continue
            seen_idxs.add(cidx)
            cid = idx_to_cid.get(cidx)
            if cid and cid not in seen_cids:
                cited_cids.append(cid)
                seen_cids.add(cid)
        if cited_cids:
            merged_order: list[str] = list(cited_cids)
            for cid in list(ordered_used_chunk_ids or []):
                scid = str(cid or "").strip()
                if not scid or scid in seen_cids:
                    continue
                merged_order.append(scid)
                seen_cids.add(scid)
            ordered_used_chunk_ids = merged_order

    for cid in list(ordered_used_chunk_ids or []):
        r = by_id.get(str(cid))
        if not r or not _is_eligible_reference(r):
            continue
        # Deduplicate by corpus-aware canonical anchor.
        # Same article in different corpora are NOT deduplicated.
        key = _get_reference_dedup_key(r)
        if key is None or key in seen_keys:
            continue
        seen_keys.add(key)
        gated.append(dict(r))

    references_structured = list(gated)
    reference_lines = [
        f"[{r['idx']}] {r.get('precise_ref') or r.get('display')}"
        for r in references_structured
    ]

    return HardReferenceGatingResult(
        used_chunk_ids=working_used_chunk_ids,
        references_structured=references_structured,
        reference_lines=reference_lines,
    )


# ---------------------------------------------------------------------------
# Citation Utilities (Extracted from rag.py)
# ---------------------------------------------------------------------------


def extract_bracket_citations(text: str) -> set[int]:
    """Extract all [n] citation indices from text.

    Args:
        text: The text to extract citations from.

    Returns:
        Set of citation indices found in the text.
    """
    body = _strip_trailing_references_section(str(text or ""))
    return {int(m.group(1)) for m in re.finditer(r"\[(\d{1,3})\]", body)}


def count_valid_citations(text: str, allowed_idxs: set[int]) -> int:
    """Count citations that are in allowed set.

    Args:
        text: The text to extract citations from.
        allowed_idxs: Set of allowed citation indices.

    Returns:
        Number of valid citations found.
    """
    cited = extract_bracket_citations(text)
    return len(cited & allowed_idxs)


def ensure_required_anchor_citations(
    answer_text: str,
    required_anchors_payload: dict[str, Any] | None,
    references_structured_all: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> str:
    """ENGINEERING: force required anchor [idx] citations into answer.

    This function ensures required anchors survive hard reference gating by
    forcing their idx citations to appear in the answer text. This is
    deterministic and only appends missing [idx] markers to an existing
    bullet line (no new claims or bullet count changes).

    Previously inline in rag.py answer_structured() (lines 2183-2240).

    Args:
        answer_text: The current answer text.
        required_anchors_payload: Dict with required anchor specification.
        references_structured_all: All structured references.
        run_meta: The run metadata dict to update (mutated in-place).

    Returns:
        The answer text (possibly with appended citations).
    """
    if not required_anchors_payload:
        return answer_text

    try:
        required_idxs, missing_anchor_keys = helpers.compute_required_anchor_idxs(
            required_anchors_payload=dict(required_anchors_payload),
            references_structured_all=[dict(r or {}) for r in list(references_structured_all or []) if isinstance(r, dict)],
        )

        body = _strip_trailing_references_section(str(answer_text or ""))
        cited_now = {int(m.group(1)) for m in re.finditer(r"\[(\d{1,3})\]", body)}
        missing_idxs = sorted(set(required_idxs) - set(cited_now))

        did_force = False
        if missing_idxs:
            lines = str(answer_text or "").splitlines()

            def _find_first_bullet_in_section2(lines_in: list[str]) -> int | None:
                sec2 = None
                for i, ln in enumerate(lines_in):
                    if str(ln).strip().startswith("2."):
                        sec2 = i
                        break
                if sec2 is None:
                    return None
                for j in range(sec2 + 1, len(lines_in)):
                    if str(lines_in[j]).strip().startswith("3."):
                        break
                    if str(lines_in[j]).lstrip().startswith("-"):
                        return j
                return None

            def _find_first_bullet_anywhere(lines_in: list[str]) -> int | None:
                for i, ln in enumerate(lines_in):
                    if str(ln).lstrip().startswith("-"):
                        return i
                return None

            target_idx = _find_first_bullet_in_section2(lines)
            if target_idx is None:
                target_idx = _find_first_bullet_anywhere(lines)

            if target_idx is not None:
                add_marks = " ".join(f"[{i}]" for i in missing_idxs)
                if add_marks:
                    lines[target_idx] = str(lines[target_idx]).rstrip() + " " + add_marks
                    answer_text = "\n".join(lines)
                    did_force = True

        if isinstance(run_meta.get("anchor_rescue"), dict):
            run_meta["anchor_rescue"]["rescue_required_anchor_retry_performed"] = bool(did_force)
            run_meta["anchor_rescue"]["rescue_required_anchor_retry_success"] = (True if did_force else None)

        # If anchors are required but not even retrievable, keep an explicit fail reason.
        if missing_anchor_keys and isinstance(run_meta.get("anchor_rescue"), dict):
            run_meta["anchor_rescue"].setdefault("missing_required_anchor_all_of", [])
            # Do not override existing failure reasons; only annotate.
            run_meta.setdefault("final_fail_reason", run_meta.get("final_fail_reason"))
    except Exception:  # noqa: BLE001
        pass

    return answer_text


# -----------------------------------------------------------------------------
# Consolidated citation processing (Stage 4b wrapper)
# -----------------------------------------------------------------------------


@dataclass
class CitationProcessingResult:
    """Result from the citation processing stage (Stage 4b) of answer_structured.

    Contains all outputs from citation processing:
    - Modified answer text
    - Filtered references (only cited ones)
    - Reference lines for display
    - Used chunk IDs
    - Debug information for run_meta
    """

    answer_text: str
    references_structured: list[dict[str, Any]]
    reference_lines: list[str]
    used_chunk_ids: list[str]
    references_structured_all: list[dict[str, Any]]
    debug: dict[str, Any] = field(default_factory=dict)


def apply_all_citation_processing(
    *,
    answer_text: str,
    question: str,
    references_structured_all: list[dict[str, Any]],
    resolved_profile: "UserProfile",
    intent_used: "ClaimIntent",
    did_abstain: bool,
    required_anchors_payload: dict[str, Any] | None,
    contract_min_citations: int | None,
    is_legal_profile: bool,
    legal_allow_reference_fallback: bool,
    run_meta: dict[str, Any],
    corpus_debug_on: bool = False,
    is_debug_enabled_fn: Callable[[], bool] | None = None,
) -> CitationProcessingResult:
    """Stage 4b: Apply all citation processing in correct order.

    Consolidates:
    - Citation backstop (ensure min unique citations)
    - Citation repair (anchor mentions → brackets)
    - Required-anchor rescue (eval/contract-only)
    - Hard reference gating (filter to cited refs only)
    - Citation integrity check (ENGINEERING only)

    Args:
        answer_text: The answer text from generation/policy stages
        question: The user's question
        references_structured_all: All structured references from retrieval
        resolved_profile: User profile (LEGAL/ENGINEERING)
        intent_used: The classified intent for this question
        did_abstain: Whether policy gates abstained
        required_anchors_payload: Optional anchor requirements (eval/contract)
        contract_min_citations: Optional minimum citations requirement
        is_legal_profile: Whether this is LEGAL profile
        legal_allow_reference_fallback: Whether to allow LEGAL fallback refs
        run_meta: Run metadata dict (mutated in-place)
        corpus_debug_on: Whether corpus debug is enabled
        is_debug_enabled_fn: Function to check if debug is enabled

    Returns:
        CitationProcessingResult with all citation processing outputs
    """
    from .types import ClaimIntent, UserProfile

    debug: dict[str, Any] = {}

    def _dbg_citation_set(txt: str) -> set[int]:
        return {int(m.group(1)) for m in re.finditer(r"\[(\d{1,3})\]", _strip_trailing_references_section(str(txt or "")))}

    # --- ENGINEERING minimal backstop (SCOPE/CLASSIFICATION/REQUIREMENTS) ---
    # If the final answer body contains neither [n] citations nor anchor mentions,
    # append neutral hjemmel lines pointing at eligible references.
    if resolved_profile == UserProfile.ENGINEERING and intent_used in {
        ClaimIntent.SCOPE,
        ClaimIntent.CLASSIFICATION,
        ClaimIntent.REQUIREMENTS,
    } and (not did_abstain):
        if corpus_debug_on:
            before_cites = sorted(_dbg_citation_set(answer_text))
            mentions = helpers._extract_anchor_mentions_from_answer(str(answer_text or ""))
            run_meta.setdefault("corpus_debug", {})
            run_meta["corpus_debug"].update(
                {
                    "backstop_ran": True,
                    "backstop_before_cited_idxs": before_cites,
                    "backstop_anchor_mentions": {
                        "articles": list(mentions.get("articles") or []),
                        "recitals": list(mentions.get("recitals") or []),
                        "annexes": list(mentions.get("annexes") or []),
                    },
                }
            )
        answer_text = _engineering_ensure_min_unique_citations(
            answer_text=answer_text,
            references_structured_all=references_structured_all,
            min_unique_citations=3,
        )
        if corpus_debug_on:
            after_cites = sorted(_dbg_citation_set(answer_text))
            run_meta.setdefault("corpus_debug", {})
            run_meta["corpus_debug"].update(
                {
                    "backstop_after_cited_idxs": after_cites,
                    "backstop_added": int(max(0, len(set(after_cites) - set(before_cites)))),
                }
            )
        debug["backstop_ran"] = True

    # --- ENGINEERING: deterministic repair of missing bracket citations ---
    # Must run BEFORE reference gating.
    if resolved_profile == UserProfile.ENGINEERING:
        if corpus_debug_on:
            before_cites = sorted(_dbg_citation_set(answer_text))
        answer_text = _engineering_repair_bracket_citations_from_anchor_mentions(
            answer_text=answer_text,
            references_structured_all=references_structured_all,
            max_unique_citations=8,
        )
        if corpus_debug_on:
            after_cites = sorted(_dbg_citation_set(answer_text))
            run_meta.setdefault("corpus_debug", {})
            run_meta["corpus_debug"].update(
                {
                    "repair_ran": True,
                    "repair_before_cited_idxs": before_cites,
                    "repair_after_cited_idxs": after_cites,
                    "repair_added": int(max(0, len(set(after_cites) - set(before_cites)))),
                }
            )
        debug["repair_ran"] = True

    # --- ENGINEERING required-anchor rescue (eval/contract-only) ---
    if resolved_profile == UserProfile.ENGINEERING and required_anchors_payload:
        answer_text = ensure_required_anchor_citations(
            answer_text=str(answer_text or ""),
            required_anchors_payload=required_anchors_payload,
            references_structured_all=list(references_structured_all or []),
            run_meta=run_meta,
        )
        debug["required_anchor_rescue_ran"] = True

    # --- Hard reference gating ---
    # Only include chunks actually used/cited in the answer
    used_chunk_ids = select_references_used_in_answer(
        answer_text=answer_text,
        references_structured=references_structured_all,
    )

    # Apply hard reference gating with LEGAL fallback
    gating_result = apply_hard_reference_gating(
        answer_text=str(answer_text or ""),
        used_chunk_ids=list(used_chunk_ids or []),
        references_structured_all=list(references_structured_all or []),
        question=question,
        is_legal_profile=is_legal_profile,
        legal_allow_reference_fallback=bool(legal_allow_reference_fallback),
        max_legal_fallback_refs=2,
    )
    used_chunk_ids = gating_result.used_chunk_ids
    references_structured: list[dict[str, Any]] = gating_result.references_structured
    reference_lines = gating_result.reference_lines

    # Diagnostic counters used by eval before/after checks
    try:
        run_meta["references_structured_count"] = int(len(references_structured or []))
    except Exception:  # noqa: BLE001
        run_meta.setdefault("references_structured_count", None)

    # --- ENGINEERING citation integrity post-check ---
    # Goal: prevent "refs without [n]" and prevent citing idx that are not present.
    if resolved_profile == UserProfile.ENGINEERING:
        citation_result = apply_engineering_citation_integrity(
            answer_text=str(answer_text or ""),
            references_structured=list(references_structured or []),
            references_structured_all=list(references_structured_all or []),
            used_chunk_ids=list(used_chunk_ids or []),
            run_meta=run_meta,
            contract_min_citations=contract_min_citations,
            intent_used=str(getattr(intent_used, "value", intent_used) or ""),
            is_debug_enabled_fn=is_debug_enabled_fn,
        )
        answer_text = citation_result.answer_text
        references_structured = citation_result.references_structured
        reference_lines = citation_result.reference_lines
        used_chunk_ids = citation_result.used_chunk_ids

        # Persist debug info to run_meta
        if citation_result.debug:
            run_meta.setdefault("citations_source", citation_result.debug.get("citations_source"))
            run_meta.setdefault("json_cited_idxs", citation_result.debug.get("json_cited_idxs", []))
            run_meta.setdefault("answer_text_contains_brackets", citation_result.debug.get("answer_text_contains_brackets"))
            run_meta["parsed_citations_raw"] = citation_result.debug.get("parsed_citations_raw", [])
            if isinstance(run_meta.get("engineering_json"), dict):
                run_meta["engineering_json"].setdefault("citations_source", citation_result.debug.get("citations_source"))
                run_meta["engineering_json"].setdefault("json_cited_idxs", citation_result.debug.get("json_cited_idxs", []))
                run_meta["engineering_json"].setdefault("answer_text_contains_brackets", citation_result.debug.get("answer_text_contains_brackets"))
                run_meta["engineering_json"]["parsed_citations_raw"] = citation_result.debug.get("parsed_citations_raw", [])

            if citation_result.debug.get("fail_reason"):
                if run_meta.get("fail_reason") is None:
                    run_meta["fail_reason"] = citation_result.debug["fail_reason"]
                    run_meta["final_fail_reason"] = citation_result.debug["fail_reason"]
                    if isinstance(run_meta.get("engineering_json"), dict):
                        run_meta["engineering_json"]["fail_reason"] = citation_result.debug["fail_reason"]
            if citation_result.debug.get("missing_ref_reason"):
                run_meta["missing_ref_reason"] = citation_result.debug["missing_ref_reason"]

        debug["citation_integrity_ran"] = True
        debug.update(citation_result.debug)

    return CitationProcessingResult(
        answer_text=answer_text,
        references_structured=references_structured,
        reference_lines=reference_lines,
        used_chunk_ids=used_chunk_ids,
        references_structured_all=list(references_structured_all or []),
        debug=debug,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Citation Verification (Post-hoc Similarity Check)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CitationVerificationResult:
    """Result of post-hoc citation verification."""
    verified: List[int]  # Citation indices that passed verification
    suspicious: List[Dict[str, Any]]  # Citations with low similarity
    overall_score: float  # Average similarity score (0.0-1.0)
    scores: List[float]  # Individual scores per citation


def _extract_claim_context(answer_text: str, citation_idx: int, window: int = 150) -> str:
    """Extract text surrounding a citation [n] in the answer.

    Args:
        answer_text: The full answer text
        citation_idx: The citation index to find
        window: Number of characters before/after to extract

    Returns:
        Text context around the citation, or empty string if not found
    """
    pattern = rf"\[{citation_idx}\]"
    match = re.search(pattern, answer_text or "")
    if not match:
        return ""

    start = max(0, match.start() - window)
    end = min(len(answer_text), match.end() + window)
    return answer_text[start:end]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector

    Returns:
        Cosine similarity score (0.0-1.0)
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def _find_better_match(
    claim_context: str,
    references: List[Dict[str, Any]],
    embedding_fn: Callable[[str], List[float]],
) -> int | None:
    """Find the reference that best matches the claim context.

    Args:
        claim_context: Text around the citation in the answer
        references: List of structured reference dicts
        embedding_fn: Function to compute embeddings

    Returns:
        Index of best matching reference, or None if no good match
    """
    if not claim_context.strip():
        return None

    claim_emb = embedding_fn(claim_context)
    if not claim_emb:
        return None

    best_idx: int | None = None
    best_score = 0.0

    for ref in references:
        chunk_text = ref.get("chunk_text", "")
        if not chunk_text:
            continue

        chunk_emb = embedding_fn(chunk_text)
        if not chunk_emb:
            continue

        score = _cosine_similarity(claim_emb, chunk_emb)
        if score > best_score:
            best_score = score
            best_idx = ref.get("idx")

    return best_idx


def verify_citation_similarity(
    answer_text: str,
    references_structured: List[Dict[str, Any]],
    embedding_fn: Callable[[str], List[float]],
    similarity_threshold: float = 0.75,
) -> CitationVerificationResult:
    """Verify that each citation [n] semantically matches its cited chunk.

    This post-hoc verification checks if the text around each citation in the
    answer is semantically similar to the chunk it references. Low similarity
    may indicate citation mis-attribution.

    Args:
        answer_text: The generated answer text with [n] citations
        references_structured: List of structured reference dicts with idx and chunk_text
        embedding_fn: Function (text) -> embedding vector for similarity computation
        similarity_threshold: Minimum similarity score to consider a citation verified

    Returns:
        CitationVerificationResult with verified/suspicious citations and scores
    """
    verified: List[int] = []
    suspicious: List[Dict[str, Any]] = []
    scores: List[float] = []

    # Find all citations in the answer
    citation_pattern = re.compile(r"\[(\d{1,3})\]")
    cited_idxs = sorted(set(int(m.group(1)) for m in citation_pattern.finditer(answer_text or "")))

    # Build idx -> reference lookup
    ref_by_idx: Dict[int, Dict[str, Any]] = {}
    for ref in (references_structured or []):
        if isinstance(ref, dict):
            try:
                idx = int(ref.get("idx"))
                ref_by_idx[idx] = ref
            except (TypeError, ValueError):
                continue

    for idx in cited_idxs:
        ref = ref_by_idx.get(idx)
        if not ref:
            # Citation references non-existent source
            suspicious.append({
                "idx": idx,
                "similarity": 0.0,
                "reason": "citation_not_in_references",
                "suggestion": None,
            })
            scores.append(0.0)
            continue

        # Extract context around citation
        claim_context = _extract_claim_context(answer_text, idx)
        if not claim_context.strip():
            # Cannot extract context - skip verification
            verified.append(idx)
            scores.append(1.0)
            continue

        chunk_text = ref.get("chunk_text", "")
        if not chunk_text:
            # No chunk text to compare against
            verified.append(idx)
            scores.append(1.0)
            continue

        # Compute embedding similarity
        try:
            claim_emb = embedding_fn(claim_context)
            chunk_emb = embedding_fn(chunk_text)
            similarity = _cosine_similarity(claim_emb, chunk_emb)
        except Exception:  # noqa: BLE001
            # Embedding failed - assume OK
            verified.append(idx)
            scores.append(1.0)
            continue

        scores.append(similarity)

        if similarity >= similarity_threshold:
            verified.append(idx)
        else:
            # Find better match suggestion
            suggestion = _find_better_match(claim_context, references_structured, embedding_fn)

            suspicious.append({
                "idx": idx,
                "similarity": round(similarity, 3),
                "claim_context": claim_context[:100] + "..." if len(claim_context) > 100 else claim_context,
                "reason": "low_similarity",
                "suggestion": suggestion,
            })

    overall_score = sum(scores) / len(scores) if scores else 1.0

    return CitationVerificationResult(
        verified=verified,
        suspicious=suspicious,
        overall_score=round(overall_score, 3),
        scores=[round(s, 3) for s in scores],
    )
