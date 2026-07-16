"""Query heuristics and reference extraction utilities.

Pure functions for classifying query intent, extracting structural references
(article, annex, recital, chapter, section) from question text, and detecting
question patterns (multi-part, structure, substantive, chapter overview/summary).

Extracted from helpers.py (Phase 8b) — single responsibility.
"""

from __future__ import annotations

import re


def classify_query_intent(question: str) -> str:
    """Classify the user's intent for ranking/retrieval.

    Deterministic, heuristic-only.
    """
    q = str(question or "").strip().lower()
    if not q:
        return "CONTEXT"

    # Order matters: these are intentionally simple and deterministic.
    if re.search(
        r"\b(penalt(y|ies)|fine(s)?|sanction(s)?|authority|myndighed|b\u00f8de(r)?)\b",
        q,
    ):
        return "ENFORCEMENT"
    if re.search(
        r"\b(inform|tell\s+user|ui|disclosure|transparency|gennemsigtighed|oplysn|information)\b",
        q,
    ):
        return "TRANSPARENCY"
    if re.search(
        r"\b(what\s+must|must\b|shall\b|required|requirements|krav|skal\b)\b", q
    ):
        return "OBLIGATIONS"
    if re.search(r"\b(why|meaning|what\s+is|hvad\s+er|hvad\s+betyder|forklar)\b", q):
        return "CONTEXT"

    # Default intent: context-seeking.
    return "CONTEXT"


def _extract_article_ref(question: str) -> str | None:
    # Tolerate PDFs/user input with spaces between letters ("a r t i k e l").
    article_word = r"a\s*r\s*t\s*i\s*k\s*e\s*l"
    match = re.search(rf"(?i){article_word}\s*(\d{{1,3}}[a-z]?)", question)
    if not match:
        return None
    return match.group(1).upper()


def _extract_article_refs(question: str) -> list[str]:
    """Return all explicit article refs mentioned in the question (unique, stable order)."""
    q = str(question or "")
    article_word = r"a\s*r\s*t\s*i\s*k\s*e\s*l"
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(rf"(?i){article_word}\s*(\d{{1,3}}[a-z]?)", q):
        a = str(m.group(1) or "").strip().upper()
        if not a or a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out


def _looks_like_multi_part_question(question: str) -> bool:
    """Heuristic: multi-part prompts with 2+ numbered items (1), 2), 3) ... or 1., 2., 3.)."""
    q = str(question or "")
    hits = 0
    for line in q.splitlines():
        if re.match(r"^\s*\d+\s*[\)\.]\s+", line):
            hits += 1
            if hits >= 2:
                return True
    return False


def _extract_annex_refs(question: str) -> list[str]:
    """Return all explicit annex/bilag refs mentioned in the question (unique, stable order)."""
    q = str(question or "")
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"(?i)\b(?:bilag|annex)\s+([ivxlcdm]+|\d{1,3})\b", q):
        ax = str(m.group(1) or "").strip().upper()
        if not ax or ax in seen:
            continue
        seen.add(ax)
        out.append(ax)
    return out


def _extract_recital_ref(question: str) -> str | None:
    # Danish: "betragtning (180)" or "betragtning 180". English: "recital 180".
    q = str(question or "")
    m = re.search(r"(?i)\b(?:betragtning|recital)\s*\(?\s*(\d{1,4})\s*\)?\b", q)
    if not m:
        return None
    return str(m.group(1)).strip()


def _looks_like_recital_quote_question(question: str) -> bool:
    q = str(question or "").lower().strip()
    if "betragtning" not in q and "recital" not in q:
        return False
    return any(
        token in q for token in ("hvad siger", "hvad st\u00e5r der", "ordlyd", "citer")
    )


def _extract_chapter_ref(question: str) -> str | None:
    chapter_word = r"k\s*a\s*p\s*i\s*t\s*e\s*l"
    match = re.search(rf"(?i){chapter_word}\s*([0-9]+|[ivxlcdm]+)", question)
    if not match:
        return None
    return match.group(1).upper()


def _extract_section_ref(question: str) -> str | None:
    section_word = r"a\s*f\s*s\s*n\s*i\s*t"
    afdeling_word = r"a\s*f\s*d\s*e\s*l\s*i\s*n\s*g"
    match = re.search(
        rf"(?i)(?:{section_word}|{afdeling_word})\s*([0-9]+|[ivxlcdm]+)", question
    )
    if not match:
        return None
    return match.group(1).upper()


def _roman_to_int(value: str) -> int | None:
    roman_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    v = (value or "").strip().upper()
    if not v or any(ch not in roman_map for ch in v):
        return None
    total = 0
    prev = 0
    for ch in reversed(v):
        cur = roman_map[ch]
        if cur < prev:
            total -= cur
        else:
            total += cur
            prev = cur
    return total


def _ref_to_int(value: str | None) -> int | None:
    s = str(value or "").strip().upper()
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:  # noqa: BLE001
            return None
    return _roman_to_int(s)


def _looks_like_structure_question(question: str) -> bool:
    q = question.lower().strip()

    # Only treat as a TOC/navigation question when the user explicitly asks
    # about structure (where something is, list/overview, TOC), not when they
    # ask substantive content questions that merely mention an article.
    explicit_markers = (
        "indholdsfortegnelse",
        "toc",
        "table of contents",
        "oversigt",
        "struktur",
    )
    if any(marker in q for marker in explicit_markers):
        return True

    if re.search(
        r"\b(hvor\s+ligger|hvilke\s+artikler|hvilket\s+kapitel|hvilke\s+kapitler|hvilket\s+afsnit|hvilke\s+afsnit|hvilken\s+afdeling|hvilke\s+afdelinger|liste\s+over|oversigt\s+over|hvad\s+(?:handler|omhandler)\s+kapitel|hvad\s+st\s*å\s*r\s+der\s+i\s+kapitel)\b",
        q,
    ):
        return True

    return False


def _looks_like_substantive_question(question: str) -> bool:
    q = question.lower()
    return any(
        token in q
        for token in (
            "hvad",
            "forklar",
            "beskriv",
            "handler",
            "betyder",
            "krav",
            "forbud",
            "formål",
            "definition",
            "hvordan",
            "hvem",
        )
    )


def _looks_like_chapter_overview_question(question: str) -> bool:
    """Check if the question is asking for a chapter overview."""
    q = question.lower().strip()
    if "kapitel" not in q:
        return False
    # Examples: "hvad handler kapitel 10 om?", "hvad omhandler kapitel X?"
    return bool(
        re.search(
            r"\b(hvad\s+(?:handler|omhandler)|hvad\s+drejer\s+kapitel\s+sig\s+om)\b", q
        )
    )


def _looks_like_chapter_summary_question(question: str) -> bool:
    """Check if the question is asking for a chapter summary."""
    q = question.lower().strip()
    if "kapitel" not in q:
        return False
    return any(
        token in q
        for token in (
            "sammenfat",
            "opsummer",
            "resumé",
            "resume",
            "kort fortalt",
            "hvad står der i kapitel",
        )
    )
