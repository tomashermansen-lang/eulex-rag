#!/usr/bin/env python3
"""Validate failing eval cases against actual retrieval results."""

import yaml
from src.engine.rag import RAGEngine


def validate_corpus(corpus_id: str, yaml_file: str, failing_ids: list[str]) -> None:
    """Validate failing cases for a corpus."""
    with open(yaml_file) as f:
        cases = yaml.safe_load(f)

    engine = RAGEngine('./data/processed', corpus_id=corpus_id)

    for case in cases:
        if case['id'] not in failing_ids:
            continue

        print(f"\n{'='*70}")
        print(f"CASE: {case['id']}")
        print(f"Prompt: {case['prompt'][:100]}...")

        exp = case['expected']
        must_any = exp.get('must_include_any_of', [])
        must_all = exp.get('must_include_all_of', [])
        print(f"Expected ANY of: {must_any}")
        print(f"Expected ALL of: {must_all}")

        # Retrieval
        results = engine.query(case['prompt'], k=20)
        found = set()
        for i, (doc, meta) in enumerate(results[:20]):
            art = meta.get('article', '')
            annex = meta.get('annex', '')
            if art:
                found.add(f"article:{art}")
            if annex:
                found.add(f"annex:{annex}")

        print(f"Retrieved (top20): {sorted(found)}")

        # Check missing
        all_expected = set(must_any + must_all)
        missing = all_expected - found
        if missing:
            print(f"❌ MISSING: {sorted(missing)}")
        else:
            print(f"✅ All expected anchors found in retrieval")

        # Verdict
        any_ok = not must_any or any(a in found for a in must_any)
        all_ok = all(a in found for a in must_all)
        
        if any_ok and all_ok:
            print("VERDICT: Retrieval OK - issue is downstream (LLM citation)")
        else:
            print("VERDICT: Retrieval issue - expected articles not in top-k")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("VALIDATING AI-ACT FAILING CASES")
    print("="*70)
    validate_corpus(
        'ai-act',
        'data/evals/golden_cases_ai_act.yaml',
        ['ai-act-13-recruitment-screening-high-risk', 'ai-act-23-edge-cross-reference']
    )

    print("\n" + "="*70)
    print("VALIDATING GDPR FAILING CASES")
    print("="*70)
    validate_corpus(
        'gdpr',
        'data/evals/golden_cases_gdpr.yaml',
        ['gdpr-24-adversarial-ambiguous']
    )

    print("\n" + "="*70)
    print("VALIDATING DORA FAILING CASES")
    print("="*70)
    validate_corpus(
        'dora',
        'data/evals/golden_cases_dora.yaml',
        [
            'dora-11-ict-third-party-contract-citations',
            'dora-15-multi-concept-testing-tlpt',
            'dora-16-multi-concept-thirdparty-contracts',
            'dora-27-competent-authorities',
            'dora-29-incident-reporting-focused'
        ]
    )
