from __future__ import annotations

from src.engine.contract_validation import (
    extract_idxs_from_lines,
    extract_idxs_from_structured,
    parse_bracket_citations,
    validate_engineering_contract,
)


def test_parse_bracket_citations_basic():
    assert parse_bracket_citations("Se [2] og [1] og [2].") == [1, 2]


def test_contract_pass_all_match():
    v = validate_engineering_contract(
        case_id="c1",
        law="ai-act",
        profile="ENGINEERING",
        rerank_state="OFF",
        answer_text="Krav følger af [3].",
        references_structured=[{"idx": 3, "chunk_id": "cid-3"}],
        reference_lines_or_references=["[3] AI Act — Artikel 6"],
        allow_additional_uncited_refs=False,
        max_citations=None,
    )
    assert v == []


def test_contract_fail_missing_structured_for_citation():
    v = validate_engineering_contract(
        case_id="c1",
        law="ai-act",
        profile="ENGINEERING",
        rerank_state="OFF",
        answer_text="Se [1].",
        references_structured=[],
        reference_lines_or_references=["[1] x"],
        allow_additional_uncited_refs=False,
        max_citations=None,
    )
    assert any(x.code == "CITATIONS_WITHOUT_STRUCTURED_REFS" for x in v)


def test_contract_fail_uncited_structured_refs():
    v = validate_engineering_contract(
        case_id="c1",
        law="ai-act",
        profile="ENGINEERING",
        rerank_state="OFF",
        answer_text="Se [1].",
        references_structured=[{"idx": 1}, {"idx": 2}],
        reference_lines_or_references=["[1] a", "[2] b"],
        allow_additional_uncited_refs=False,
        max_citations=None,
    )
    assert any(x.code == "UNCITED_STRUCTURED_REFS" for x in v)


def test_contract_fail_renumbering_lines_missing_cited():
    v = validate_engineering_contract(
        case_id="c1",
        law="ai-act",
        profile="ENGINEERING",
        rerank_state="OFF",
        answer_text="Se [3].",
        references_structured=[{"idx": 3}],
        reference_lines_or_references=["[1] something"],
        allow_additional_uncited_refs=False,
        max_citations=None,
    )
    assert any(x.code == "REN_NUMBERING_MISMATCH" for x in v)


def test_contract_fail_max_citations():
    v = validate_engineering_contract(
        case_id="c1",
        law="ai-act",
        profile="ENGINEERING",
        rerank_state="OFF",
        answer_text="Se [1] [2] [3].",
        references_structured=[{"idx": 1}, {"idx": 2}, {"idx": 3}],
        reference_lines_or_references=["[1] a", "[2] b", "[3] c"],
        allow_additional_uncited_refs=False,
        max_citations=2,
    )
    assert any(x.code == "MAX_CITATIONS_EXCEEDED" for x in v)


def test_contract_matches_by_reference_idx_not_list_position() -> None:
    # references_structured er out-of-order; [5] skal matche idx=5 (ikke element #5).
    v = validate_engineering_contract(
        case_id="c2",
        law="gdpr",
        profile="ENGINEERING",
        rerank_state="BASE",
        answer_text="Se [5].",
        references_structured=[{"idx": 99}, {"idx": 5}],
        reference_lines_or_references=["[99] x", "[5] y"],
        allow_additional_uncited_refs=True,
        max_citations=None,
    )
    assert not any(x.code in {"CITED_IDX_MISSING_IN_STRUCTURED", "CITATIONS_NO_MATCHING_REFS"} for x in v)

def test_contract_fail_min_citations_not_met() -> None:
    violations = validate_engineering_contract(
        case_id="case-1",
        law="ai-act",
        profile="ENGINEERING",
        rerank_state="BASE",
        answer_text="Kun én citation [7].",
        references_structured=[{"idx": 7}],
        reference_lines_or_references=["[7] Ref"],
        allow_additional_uncited_refs=False,
        min_citations=3,
        max_citations=None,
    )
    assert any(v.code == "MIN_CITATIONS_NOT_MET" for v in violations)


def test_extract_idxs_helpers_are_robust():
    assert extract_idxs_from_structured([{"idx": "2"}, {"idx": None}, "x"]) == {2}
    assert extract_idxs_from_lines(["[10] a", "nope", None]) == {10}
