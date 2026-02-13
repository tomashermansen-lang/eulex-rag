#!/usr/bin/env python3
"""Analyze the 3 failed eval cases to understand LLM citation behavior."""

from src.services.ask import ask, build_engine
from src.engine.types import UserProfile
import re

def analyze_case(engine, case_id, prompt, expected, forbidden=None):
    print("=" * 70)
    print(f"CASE: {case_id}")
    print(f"EXPECTED: {expected}")
    if forbidden:
        print(f"FORBIDDEN: {forbidden}")
    print("=" * 70)
    
    r = ask(question=prompt, law="ai-act", user_profile=UserProfile.LEGAL, engine=engine)
    
    # Extract cited articles
    cited_idxs = set(re.findall(r"\[(\d+)\]", r.answer))
    
    cited_articles = []
    for ref in r.references_structured:
        idx = str(ref.get("idx", ""))
        if idx in cited_idxs:
            art = ref.get("article", "")
            anx = ref.get("annex", "")
            if art:
                cited_articles.append(f"article:{art}")
            if anx:
                cited_articles.append(f"annex:{anx}")
    
    print(f"\nCITED BY LLM: {cited_articles}")
    
    # Check if expected is met
    expected_list = [e.strip() for e in expected.replace(" OR ", ",").replace(" AND ", ",").split(",")]
    found_expected = [e for e in expected_list if e.lower() in [c.lower() for c in cited_articles]]
    missing_expected = [e for e in expected_list if e.lower() not in [c.lower() for c in cited_articles]]
    
    print(f"FOUND EXPECTED: {found_expected}")
    print(f"MISSING EXPECTED: {missing_expected}")
    
    # Check forbidden
    if forbidden:
        forbidden_list = [f.strip() for f in forbidden.split(",")]
        found_forbidden = [f for f in forbidden_list if f.lower() in [c.lower() for c in cited_articles]]
        if found_forbidden:
            print(f"⚠️  CITED FORBIDDEN: {found_forbidden}")
    
    # Show first part of answer
    print(f"\nANSWER (first 800 chars):")
    print(r.answer[:800])
    print("\n")


def main():
    engine = build_engine(law="ai-act")
    
    # Case 02
    analyze_case(
        engine,
        "ai-act-02-access-to-logs",
        "Hvem skal have adgang til logs for et højrisiko-AI-system efter AI-forordningen, og med hvilket formål?",
        "article:12 OR article:21"
    )
    
    # Case 16
    analyze_case(
        engine,
        "ai-act-16-multi-concept-logging-transparency",
        "AI Act: Hvad skal logges for et højrisiko-AI-system, og hvilke gennemsigtighedsforpligtelser har operatøren overfor slutbrugeren?",
        "article:12 OR article:21, article:13"
    )
    
    # Case 31
    analyze_case(
        engine,
        "ai-act-31-financial-product-recommendations",
        "Vi er i gang med at bygge et system, der automatisk foreslår pensionsprodukter og investeringsprofiler til kunder. Kan det falde ind under højrisiko-AI ifølge EUs AI-lov, og hvad afhænger det af?",
        "article:6 OR annex:iii",
        "article:51, annex:xiii"
    )


if __name__ == "__main__":
    main()
