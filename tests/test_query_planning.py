from __future__ import annotations

from src.engine.planning import (
    FocusSelection,
    FocusType,
    Intent,
    QueryContext,
    UserProfile,
    build_retrieval_plan,
    detect_intent,
    focus_to_where,
    ref_to_int,
    roman_to_int,
)
from src.engine.prompt_builder import build_prompt


def test_roman_to_int_and_ref_to_int():
    assert roman_to_int("X") == 10
    assert roman_to_int("III") == 3
    assert roman_to_int("invalid") is None
    assert ref_to_int("10") == 10
    assert ref_to_int("x") == 10


def test_focus_to_where_chapter_article_annex():
    corpus_id = "ai-act"

    chapter = FocusSelection(type=FocusType.CHAPTER, node_id="c", chapter="III")
    assert focus_to_where(corpus_id=corpus_id, focus=chapter) == {"corpus_id": corpus_id, "chapter": "III"}

    article = FocusSelection(type=FocusType.ARTICLE, node_id="a", article="10")
    assert focus_to_where(corpus_id=corpus_id, focus=article) == {"corpus_id": corpus_id, "article": "10"}

    annex = FocusSelection(type=FocusType.ANNEX, node_id="ax", annex="I")
    assert focus_to_where(corpus_id=corpus_id, focus=annex) == {"corpus_id": corpus_id, "annex": "I"}


def test_detect_intent_chapter_summary_with_focus():
    focus = FocusSelection(type=FocusType.CHAPTER, node_id="c", chapter="III")
    intent = detect_intent(question="hvad handler kapitlet om?", focus=focus)
    assert intent == Intent.CHAPTER_SUMMARY


def test_build_retrieval_plan_profile_biases_top_k_and_intent():
    focus = FocusSelection(type=FocusType.CHAPTER, node_id="c", chapter="III")

    legal = QueryContext(corpus_id="ai-act", user_profile=UserProfile.LEGAL, focus=focus, top_k=10, question="opsummer kapitlet")
    plan = build_retrieval_plan(legal)
    assert plan.intent == Intent.CHAPTER_SUMMARY
    assert plan.top_k >= 8
    assert plan.where == {"corpus_id": "ai-act", "chapter": "III"}

    eng = QueryContext(corpus_id="ai-act", user_profile=UserProfile.ENGINEERING, focus=focus, top_k=3, question="opsummer kapitlet")
    plan2 = build_retrieval_plan(eng)
    assert plan2.top_k >= 12
    assert plan2.allow_low_evidence_answer is True


def test_refine_retrieval_plan_explicit_article():
    from src.engine.planning import refine_retrieval_plan, RetrievalPlan
    from src.engine.types import ClaimIntent

    plan = RetrievalPlan(
        intent=Intent.FREEFORM,
        top_k=3,
        where={"corpus_id": "ai-act"},
        
        allow_low_evidence_answer=False
    )

    # Scenario: Question mentions "artikel 5", so we expect effective_where to include "article": "5"

    effective_where, effective_top_k, dbg = refine_retrieval_plan(
        plan=plan,
        question="hvad siger artikel 5?",
        corpus_id="ai-act",
        user_profile=UserProfile.LEGAL,
        claim_intent=ClaimIntent.GENERAL,
        requirements_cues_detected=False,
    )

    assert effective_where["article"] == "5"
    assert effective_where["corpus_id"] == "ai-act"


def test_normalize_chroma_where_multi_field_in_or():
    """Regression test: multi-field dicts inside $or must be normalized to $and wrappers.
    
    Bug repro: TOC routing can produce scopes like {"chapter": "III", "section": "1"}.
    When put into $or, Chroma rejects the invalid where clause. The fix makes
    _normalize_chroma_where recurse into $or/$and items.
    """
    from src.engine.retrieval import Retriever
    
    # Case 1: Multi-field dict inside $or
    where = {
        "$or": [
            {"chapter": "III", "section": "1"},  # Invalid for Chroma - needs $and wrapper
            {"article": "6"},
        ]
    }
    result = Retriever._normalize_chroma_where(where)
    
    # The multi-field dict should be wrapped in $and
    assert result is not None
    assert "$or" in result
    or_items = result["$or"]
    assert len(or_items) == 2
    
    # First item should now be an $and with the two fields
    first_item = or_items[0]
    assert "$and" in first_item
    and_clauses = first_item["$and"]
    # Should contain {"chapter": "III"} and {"section": "1"} (sorted order)
    assert {"chapter": "III"} in and_clauses
    assert {"section": "1"} in and_clauses
    
    # Second item stays unchanged (single field)
    assert or_items[1] == {"article": "6"}
    
    # Case 2: Nested $or inside $and with multi-field
    where2 = {
        "corpus_id": "ai-act",
        "$or": [
            {"chapter": "II", "section": "2"},
            {"article": "27"},
        ]
    }
    result2 = Retriever._normalize_chroma_where(where2)
    assert result2 is not None
    # Top level should be $and (corpus_id + $or)
    assert "$and" in result2
    # The $or inside should have its first item normalized
    or_clause = None
    for clause in result2["$and"]:
        if "$or" in clause:
            or_clause = clause["$or"]
            break
    assert or_clause is not None
    # First item in $or should now be $and-wrapped
    assert "$and" in or_clause[0]


def test_build_prompt_engineering_includes_term_rules_block():
    ctx = QueryContext(corpus_id="ai-act", user_profile=UserProfile.ENGINEERING, focus=None, top_k=3, question="Hvad skal vi logge?")
    plan = build_retrieval_plan(ctx)
    prompt = build_prompt(ctx=ctx, plan=plan, context="[1] AI Act, Artikel 10\n...", focus_block="")

    assert "Svar på dansk" in prompt
    # Check that English tech terms are preserved (either in SPROGTERMER or inline)
    assert "audit log" in prompt
    assert "append-only" in prompt
    assert "role-based access control" in prompt
