"""SCOPE-specific display transforms for answer text and reference lines.

Pure text transformation functions that apply scope-related display consistency:
- Litra consistency between answer and references
- Normative bullet removal from ENGINEERING systemkrav sections

Extracted from policy.py (Phase 8a) — single responsibility.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .types import ClaimIntent, UserProfile


def _scope_extract_article_stk_litra_mentions(
    text: str,
) -> dict[tuple[str, str], set[str]]:
    """Extract {(article, stk): {litra letters}} from a display string.

    Only uses deterministic regexes; intended for SCOPE display consistency.
    """

    txt = str(text or "")
    if not txt.strip():
        return {}

    out: dict[tuple[str, str], set[str]] = {}
    # Match: Artikel 2, stk. 1[, litra c]
    pat = re.compile(
        r"(?i)\b(?:artikel|article)\s+(\d{1,3}[a-z]?)\s*,\s*stk\.?\s*(\d{1,3})\b(?:\s*,\s*litra\s+([a-z])\b)?"
    )
    for m in pat.finditer(txt):
        art = str(m.group(1) or "").strip().upper()
        stk = str(m.group(2) or "").strip()
        lit = str(m.group(3) or "").strip().lower() or None
        if not (art and stk):
            continue
        key = (art, stk)
        if key not in out:
            out[key] = set()
        if lit:
            out[key].add(lit)
    return out


def _scope_apply_litra_consistency_to_display(
    *,
    answer_text: str,
    reference_lines: list[str],
) -> tuple[str, list[str]]:
    """SCOPE-only display consistency for litra.

    Conditional: only applies when the same (Artikel, stk.) appears in both
    `answer_text` and at least one `reference_line`.

    Rule per matching (Artikel, stk.):
    - If both sides have litra and letters mismatch => downgrade both to Artikel+stk.
    - If only one side has litra => downgrade both to Artikel+stk.
    - If both have litra and letters match => keep litra.

    Never touches citation markers like [1].
    """

    ans = str(answer_text or "")
    ref_lines = [str(x or "") for x in list(reference_lines or [])]
    if not ans.strip() or not ref_lines:
        return ans, ref_lines

    ans_map = _scope_extract_article_stk_litra_mentions(ans)
    if not ans_map:
        return ans, ref_lines

    # Aggregate litra letters per (Artikel, stk.) across all reference lines.
    ref_map: dict[tuple[str, str], set[str]] = {}
    for line in ref_lines:
        m = _scope_extract_article_stk_litra_mentions(line)
        for key, lits in m.items():
            ref_map.setdefault(key, set()).update(set(lits or set()))

    matching_pairs = set(ans_map.keys()) & set(ref_map.keys())
    if not matching_pairs:
        return ans, ref_lines

    def _needs_downgrade(pair: tuple[str, str]) -> bool:
        a = set(ans_map.get(pair) or set())
        r = set(ref_map.get(pair) or set())
        if bool(a) != bool(r):
            return True
        if a and r and a.isdisjoint(r):
            return True
        return False

    downgrade_pairs = [p for p in sorted(matching_pairs) if _needs_downgrade(p)]
    if not downgrade_pairs:
        return ans, ref_lines

    def _remove_litra_for_pair(s: str, art: str, stk: str) -> str:
        # Remove only the litra for the specific (Artikel, stk.) mention.
        art_esc = re.escape(str(art))
        stk_esc = re.escape(str(stk))
        base = rf"(?i)\b((?:artikel|article)\s+{art_esc}\s*,\s*stk\.?\s*{stk_esc})\b"
        out = re.sub(base + r"\s*,\s*litra\s+[a-z]\b", r"\1", s)
        out = re.sub(base + r"\s+litra\s+[a-z]\b", r"\1", out)
        # Keep spacing tidy without touching [n] markers.
        out = re.sub(r"[ \t]{2,}", " ", out)
        out = re.sub(r"\s+([,.;:])", r"\1", out)
        return out

    out_ans = ans
    out_refs = list(ref_lines)
    for art, stk in downgrade_pairs:
        out_ans = _remove_litra_for_pair(out_ans, art, stk)
        out_refs = [_remove_litra_for_pair(x, art, stk) for x in out_refs]

    return out_ans.strip(), [x.strip() for x in out_refs]


def _engineering_remove_normative_bullets_from_systemkrav_section_for_scope(
    answer_text: str,
) -> str:
    """ENGINEERING+SCOPE: remove '- SKAL'/'- BØR' bullets from 'Konkrete systemkrav'.

    If the section becomes empty, replace it with a single neutral line.
    """

    txt = str(answer_text or "")
    if not txt.strip():
        return txt

    lines = txt.splitlines()

    def _find_section_range(
        section_heading_re: re.Pattern[str],
    ) -> tuple[int, int] | None:
        start = None
        for i, line in enumerate(lines):
            if section_heading_re.match(line or ""):
                start = i
                break
        if start is None:
            return None
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if re.match(r"^\s*\d+\.\s+\S+", lines[j] or ""):
                end = j
                break
        return int(start), int(end)

    rng = _find_section_range(
        re.compile(r"^\s*3\.\s*Konkrete systemkrav\s*$", flags=re.IGNORECASE)
    )
    if rng is None:
        return txt
    start, end = rng

    body = lines[start + 1 : end]
    filtered: list[str] = []
    for line in body:
        line_str = str(line or "")
        if re.match(r"^\s*-\s*(SKAL|BØR)\b", line_str, flags=re.IGNORECASE):
            # Never remove explicit citations; rewrite cited requirement bullets to neutral hjemmel.
            cites = re.findall(r"\[\d{1,3}\]", line_str)
            if cites:
                filtered.append(f"- Relevant hjemmel: {' '.join(cites)}")
            continue
        # Defensive: in case normalization hasn't run yet.
        if re.match(r"^\s*-\s*(MUST|SHALL|SHOULD)\b", line_str, flags=re.IGNORECASE):
            cites = re.findall(r"\[\d{1,3}\]", line_str)
            if cites:
                filtered.append(f"- Relevant hjemmel: {' '.join(cites)}")
            continue
        filtered.append(line_str)

    has_content = any(str(l).strip() for l in filtered)
    if not has_content:
        filtered = ["Ingen konkrete systemkrav for et anvendelsesområde-spørgsmål."]

    lines = list(lines[: start + 1]) + filtered + list(lines[end:])
    return "\n".join(lines).strip()


def apply_scope_post_processing(
    *,
    answer_text: str,
    reference_lines: List[str],
    intent_used: ClaimIntent,
    resolved_profile: UserProfile,
) -> Tuple[str, List[str]]:
    """Apply scope-specific display post-processing.

    Only applies when intent_used == SCOPE.
    Mutates answer_text and reference_lines for UI display only.

    Args:
        answer_text: The answer text
        reference_lines: Reference lines for display
        intent_used: The classified intent
        resolved_profile: User profile

    Returns:
        Tuple of (answer_text, reference_lines)
    """
    if intent_used != ClaimIntent.SCOPE:
        return answer_text, reference_lines

    if resolved_profile == UserProfile.ENGINEERING:
        answer_text = (
            _engineering_remove_normative_bullets_from_systemkrav_section_for_scope(
                answer_text
            )
        )

    answer_text, reference_lines = _scope_apply_litra_consistency_to_display(
        answer_text=answer_text,
        reference_lines=list(reference_lines or []),
    )

    return answer_text, reference_lines
