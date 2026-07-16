"""Answer text normalization and transformation utilities.

Pure functions for normalizing answer text: counting normative sentences,
translating English modals to Danish, stripping trailing references,
and canonicalizing abstain text.

Extracted from helpers.py (Phase 8b) — single responsibility.
"""

from __future__ import annotations

import re

from .constants import _NORMATIVE_SENTENCE_TOKEN_RE


def _count_normative_sentences(text: str) -> int:
    """Count sentences that contain normative tokens (deterministic heuristic)."""
    txt = str(text or "")
    if not txt.strip():
        return 0
    # Keep it deterministic; treat newline and sentence-ending punctuation as boundaries.
    parts = re.split(r"(?<=[.!?])\s+|\n+", txt)
    return sum(1 for p in parts if p.strip() and _NORMATIVE_SENTENCE_TOKEN_RE.search(p))


def _normalize_modals_to_danish(answer_text: str) -> str:
    """Deterministisk dansk-normalisering af output.

    Kører som sidste trin før output returneres til UI.
    - Udskifter engelske modalverber deterministisk (MUST/SHOULD/MAY/SHALL + små bogstaver).
    - Omskriver hyppige engelske imperative linjer i starten (fx "Implement ...")
        til en dansk, normativ passivform ("Der SKAL ...").

    Må ikke ændre citationsmarkører som [1] eller referenceindeksering.
    """

    txt = str(answer_text or "")
    if not txt:
        return txt

    # Replace modal verbs (word-boundary safe).
    replacements: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bMUST\b", flags=re.IGNORECASE), "SKAL"),
        (re.compile(r"\bSHALL\b", flags=re.IGNORECASE), "SKAL"),
        (re.compile(r"\bSHOULD\b", flags=re.IGNORECASE), "BØR"),
        (re.compile(r"\bMAY\b", flags=re.IGNORECASE), "KAN"),
        # Additional English modals seen in practice.
        (re.compile(r"\bCANNOT\b", flags=re.IGNORECASE), "KAN IKKE"),
        (re.compile(r"\bCAN'T\b", flags=re.IGNORECASE), "KAN IKKE"),
        (re.compile(r"\bCAN\b", flags=re.IGNORECASE), "KAN"),
    ]
    out = txt
    for pat, repl in replacements:
        out = pat.sub(repl, out)

    # Optional, minimal Danish-first normalization for standalone YES/NO tokens (allow bullet prefixes).
    out = re.sub(r"(?m)^(\s*(?:[-*]\s*)?)YES\b", r"\1JA", out)
    out = re.sub(r"(?m)^(\s*(?:[-*]\s*)?)NO\b", r"\1NEJ", out)
    out = re.sub(r"(?m)^(\s*(?:[-*]\s*)?)Yes\b", r"\1Ja", out)
    out = re.sub(r"(?m)^(\s*(?:[-*]\s*)?)No\b", r"\1Nej", out)

    # Rewrite common English imperative sentences/lines at the start.
    # This is intentionally a small whitelist to avoid unintended translations.
    imperative_re = re.compile(
        r"(?m)^(?P<prefix>\s*(?:[-*]\s+|\d+\.\s+)?)\s*"
        r"(?P<verb>Implement|Ensure|Provide|Allow|Maintain|Use|Include|Establish|Document|Record|Report|Verify)\b"
        r"(?P<rest>.*)$",
        flags=re.IGNORECASE,
    )

    def _rewrite_imperative_line(m: re.Match[str]) -> str:
        prefix = m.group("prefix") or ""
        verb = (m.group("verb") or "").strip().lower()
        rest = m.group("rest") or ""
        rest_stripped = rest.lstrip()

        # Keep punctuation spacing stable.
        if verb == "implement":
            return f"{prefix}Der SKAL implementeres {rest_stripped}".rstrip()
        if verb == "ensure":
            # Prefer ", at" when the English line begins with "Ensure that ...".
            rest2 = re.sub(r"(?i)^that\b\s*", "at ", rest_stripped)
            # If it doesn't start with "at", keep it as-is (already Danish-ish in many templates).
            if not re.match(r"(?i)^at\b", rest2):
                return f"{prefix}Der SKAL sikres {rest2}".rstrip()
            return f"{prefix}Der SKAL sikres, {rest2}".rstrip()
        if verb == "provide":
            return f"{prefix}Der SKAL stilles til rådighed {rest_stripped}".rstrip()
        if verb == "allow":
            return f"{prefix}Der SKAL gives mulighed for {rest_stripped}".rstrip()
        if verb == "maintain":
            return f"{prefix}Der SKAL opretholdes {rest_stripped}".rstrip()
        if verb == "use":
            return f"{prefix}Der SKAL anvendes {rest_stripped}".rstrip()
        if verb == "include":
            return f"{prefix}Der SKAL inkluderes {rest_stripped}".rstrip()
        if verb == "establish":
            return f"{prefix}Der SKAL etableres {rest_stripped}".rstrip()
        if verb == "document":
            return f"{prefix}Der SKAL dokumenteres {rest_stripped}".rstrip()
        if verb == "record":
            return f"{prefix}Der SKAL registreres {rest_stripped}".rstrip()
        if verb == "report":
            return f"{prefix}Der SKAL rapporteres {rest_stripped}".rstrip()
        if verb == "verify":
            return f"{prefix}Der SKAL verificeres {rest_stripped}".rstrip()
        return m.group(0)

    out = imperative_re.sub(_rewrite_imperative_line, out)

    return out


def _strip_trailing_references_section(text: str) -> str:
    # Defensive: some callers may use the legacy answer() which appends references.
    marker = "\nReferencer:\n"
    if marker in (text or ""):
        return str(text).split(marker, 1)[0].rstrip()
    return str(text or "")


def _normalize_abstain_text(answer_text: str) -> str:
    text = str(answer_text or "")
    stripped = text.strip()
    if not stripped:
        return text

    # Eval heuristics expect the exact substring "Jeg kan ikke".
    if "Jeg kan ikke" in stripped:
        return text

    # Canonicalize common Danish abstain openings (case-insensitive).
    # Examples we want to normalize:
    # - "jeg kan ikke ..."
    # - "jeg kan desværre ikke ..."
    # - "det kan jeg ikke ..."
    # - "det kan jeg desværre ikke ..."
    m = re.match(r"(?is)^(jeg\s+kan(?:\s+\w+){0,3}\s+ikke)(.*)$", stripped)
    if m:
        rest = m.group(2) or ""
        return f"Jeg kan ikke{rest}".strip()

    m = re.match(r"(?is)^(det\s+kan\s+jeg(?:\s+\w+){0,3}\s+ikke)(.*)$", stripped)
    if m:
        rest = m.group(2) or ""
        return f"Jeg kan ikke{rest}".strip()

    return text
