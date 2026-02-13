from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from ..engine.constants import _NORMATIVE_SENTENCE_TOKEN_RE


_CIT_RE = re.compile(r"\[(\d{1,3})\]")
_LINE_IDX_RE = re.compile(r"^\[(\d{1,3})\]")

# Use consolidated normative pattern from constants.py
_NORMATIVE_RE = _NORMATIVE_SENTENCE_TOKEN_RE


@dataclass(frozen=True)
class ContractViolation:
    code: str
    message: str
    context: dict[str, Any]


def strip_trailing_references_section(answer_text: str) -> str:
    """Best-effort: avoid counting citations inside an appended References section."""

    txt = str(answer_text or "")
    marker = "\n\nReferencer:\n"
    if marker in txt:
        return txt.split(marker, 1)[0].rstrip()
    return txt


def _split_paragraphs(text: str) -> list[str]:
    raw = str(text or "")
    # Normalisér line-endings og split på blank-linjer.
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Vi splitter kun på *tydelige* afsnit (mindst to blanke linjer) for at undgå
    # falske positives når LLM'en bruger enkelte blanklinjer som layout.
    parts = re.split(r"\n\s*\n\s*\n+", raw)
    return [p.strip() for p in parts if p and p.strip()]


def _split_sentences(text: str) -> list[str]:
    raw = str(text or "")
    parts = re.split(r"(?<=[.!?])\s+|\n+", raw)
    return [p.strip() for p in parts if p and p.strip()]


def parse_bracket_citations(answer_text: str) -> list[int]:
    """Return unique cited idx values in sorted order."""

    txt = strip_trailing_references_section(str(answer_text or ""))
    found: set[int] = set()
    for m in _CIT_RE.finditer(txt):
        try:
            found.add(int(m.group(1)))
        except Exception:  # noqa: BLE001
            continue
    return sorted(found)


def contract_audit_detail(
    *,
    answer_text: str,
    references_structured: Iterable[Any] | None,
    min_citations: int | None,
    max_citations: int | None,
) -> dict[str, Any]:
    """Deterministisk kontrakt-detaljeobjekt.

    NOTE: Dette er ren eval/audit; må ikke påvirke retrieval eller prompt.
    """

    cited_idxs = parse_bracket_citations(answer_text)
    structured_idxs = extract_idxs_from_structured(references_structured)

    # Normativ regel (enkel og robust):
    # Hvis et afsnit indeholder normativt udsagn, skal samme afsnit indeholde mindst én [n].
    # Dette reducerer falske positives ift. "citations i slutningen af afsnittet".
    missing_citations: list[str] = []
    txt = strip_trailing_references_section(str(answer_text or ""))
    for para in _split_paragraphs(txt):
        if not _NORMATIVE_RE.search(para):
            continue
        if _CIT_RE.search(para):
            continue
        # Medtag konkrete sætninger for deterministisk debug.
        for s in _split_sentences(para):
            if _NORMATIVE_RE.search(s):
                excerpt = s.strip()
                if len(excerpt) > 240:
                    excerpt = excerpt[:240] + "…"
                missing_citations.append(excerpt)

    found_citations = [f"[{i}]" for i in cited_idxs]
    refs_cited = list(cited_idxs)
    refs_returned = sorted(structured_idxs)
    extras_uncited = sorted(set(structured_idxs) - set(cited_idxs))

    return {
        "missing_citations": missing_citations,
        "found_citations": found_citations,
        "refs_cited": refs_cited,
        "refs_returned": refs_returned,
        "extras_uncited": extras_uncited,
        "min_citations": int(min_citations) if min_citations is not None else None,
        "max_citations": int(max_citations) if max_citations is not None else None,
    }


def extract_idxs_from_structured(references_structured: Iterable[Any] | None) -> set[int]:
    out: set[int] = set()
    for ref in list(references_structured or []):
        if not isinstance(ref, dict):
            continue
        try:
            idx = int(ref.get("idx"))
        except Exception:  # noqa: BLE001
            continue
        if idx > 0:
            out.add(idx)
    return out


def extract_idxs_from_lines(reference_lines_or_references: Iterable[Any] | None) -> set[int]:
    out: set[int] = set()
    for line in list(reference_lines_or_references or []):
        s = str(line or "")
        m = _LINE_IDX_RE.match(s)
        if not m:
            continue
        try:
            idx = int(m.group(1))
        except Exception:  # noqa: BLE001
            continue
        if idx > 0:
            out.add(idx)
    return out


def validate_engineering_contract(
    *,
    case_id: str,
    law: str,
    profile: str,
    rerank_state: str,
    answer_text: str,
    references_structured: Iterable[Any] | None,
    reference_lines_or_references: Iterable[Any] | None,
    allow_additional_uncited_refs: bool,
    min_citations: int | None = None,
    max_citations: int | None = None,
) -> list[ContractViolation]:
    violations: list[ContractViolation] = []

    cited_idxs = parse_bracket_citations(answer_text)
    structured_idxs = extract_idxs_from_structured(references_structured)
    line_idxs = extract_idxs_from_lines(reference_lines_or_references)

    detail = contract_audit_detail(
        answer_text=answer_text,
        references_structured=references_structured,
        min_citations=min_citations,
        max_citations=max_citations,
    )

    base_ctx = {
        "case_id": case_id,
        "law": law,
        "profile": profile,
        "rerank_state": rerank_state,
        "cited_idxs": cited_idxs,
        "structured_idxs": sorted(structured_idxs),
        "line_idxs": sorted(line_idxs),
        "contract_detail": detail,
    }

    # 1) Normative udsagn uden bracket-citations [n] (konservativt / fail-closed).
    if detail.get("missing_citations"):
        violations.append(
            ContractViolation(
                code="NORMATIVE_WITHOUT_BRACKET_CITATION",
                message="Normative paragraph(s) without any [n] bracket citations.",
                context={**base_ctx, "missing_citations": list(detail.get("missing_citations") or [])},
            )
        )

    if min_citations is not None and min_citations >= 0:
        if len(cited_idxs) < int(min_citations):
            violations.append(
                ContractViolation(
                    code="MIN_CITATIONS_NOT_MET",
                    message=f"Too few unique citations: {len(cited_idxs)} < {int(min_citations)}.",
                    context={**base_ctx, "min_citations": int(min_citations)},
                )
            )

    if cited_idxs and not structured_idxs:
        violations.append(
            ContractViolation(
                code="CITATIONS_WITHOUT_STRUCTURED_REFS",
                message="Answer contains [n] citations but references_structured is empty.",
                context={**base_ctx},
            )
        )
        return violations

    missing_in_structured = sorted(set(cited_idxs) - set(structured_idxs))
    if missing_in_structured:
        violations.append(
            ContractViolation(
                code="CITED_IDX_MISSING_IN_STRUCTURED",
                message="One or more cited [n] values are missing in references_structured.",
                context={
                    **base_ctx,
                    "missing_in_structured": missing_in_structured,
                },
            )
        )

    if cited_idxs and not (set(cited_idxs) & set(structured_idxs)):
        violations.append(
            ContractViolation(
                code="CITATIONS_NO_MATCHING_REFS",
                message="Answer contains [n] citations but none match any references_structured idx.",
                context={**base_ctx},
            )
        )

    if not allow_additional_uncited_refs:
        uncited_structured = sorted(set(structured_idxs) - set(cited_idxs))
        if uncited_structured:
            violations.append(
                ContractViolation(
                    code="UNCITED_STRUCTURED_REFS",
                    message="references_structured contains idx values not cited in answer_text.",
                    context={
                        **base_ctx,
                        "uncited_structured": uncited_structured,
                    },
                )
            )

    if max_citations is not None and max_citations >= 0:
        if len(cited_idxs) > int(max_citations):
            violations.append(
                ContractViolation(
                    code="MAX_CITATIONS_EXCEEDED",
                    message=f"Too many unique citations: {len(cited_idxs)} > {int(max_citations)}.",
                    context={**base_ctx, "max_citations": int(max_citations)},
                )
            )

    if cited_idxs and line_idxs:
        missing_in_lines = sorted(set(cited_idxs) - set(line_idxs))
        if missing_in_lines:
            suspected = False
            if line_idxs:
                max_line = max(line_idxs)
                if line_idxs == set(range(1, max_line + 1)) and any(i > max_line for i in cited_idxs):
                    suspected = True
            violations.append(
                ContractViolation(
                    code="REN_NUMBERING_MISMATCH",
                    message="Cited idx not found in reference_lines/references (renumbering suspected).",
                    context={
                        **base_ctx,
                        "missing_in_lines": missing_in_lines,
                        "renumbering_suspected": bool(suspected),
                    },
                )
            )

    return violations
