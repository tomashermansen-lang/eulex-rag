#!/usr/bin/env python3
"""
Migration script: Add test_types and origin fields to existing eval YAML files.

Heuristics for inferring test_types:
- All cases get "retrieval" (base category)
- Cases with must_include_all_of -> add "multi_hop" (requires multiple sources)
- Cases with behavior: abstain -> add "abstention"
- Cases with contract_check: true -> add "faithfulness"
- Cases with adversarial/edge/negation in id -> add "robustness"
- Cases with "multi" in id -> add "multi_hop"

Origin:
- All existing cases are LLM-generated, so origin = "auto"

Usage:
    python scripts/migrate_eval_cases.py                    # Dry run
    python scripts/migrate_eval_cases.py --apply            # Apply changes
    python scripts/migrate_eval_cases.py --file golden_cases_ai_act.yaml  # Single file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def infer_test_types(case: dict) -> list[str]:
    """Infer test_types based on case characteristics."""
    types = ["retrieval"]  # All cases test retrieval

    case_id = case.get("id", "").lower()
    expected = case.get("expected", {})

    # Multi-hop: requires multiple sources
    if expected.get("must_include_all_of"):
        if "multi_hop" not in types:
            types.append("multi_hop")

    # Also check for "multi" in case id
    if "multi" in case_id and "multi_hop" not in types:
        types.append("multi_hop")

    # Abstention: cases that should refuse/clarify
    if expected.get("behavior") == "abstain":
        types.append("abstention")

    # Faithfulness: contract-first cases with strict citation requirements
    if expected.get("contract_check"):
        types.append("faithfulness")

    # Robustness: adversarial, edge, negation cases
    robustness_keywords = ["adversarial", "edge", "negation", "hypothetical", "ambiguous"]
    if any(kw in case_id for kw in robustness_keywords):
        types.append("robustness")

    # Relevancy: not easily inferred from structure, leave as retrieval only
    # unless it's a complex case that already has other types

    return types


def migrate_case(case: dict) -> dict:
    """Add test_types and origin to a case if not present."""
    # Don't modify if already has test_types
    if "test_types" in case:
        return case

    migrated = dict(case)

    # Add test_types (inferred)
    migrated["test_types"] = infer_test_types(case)

    # Add origin (all existing cases are LLM-generated)
    if "origin" not in migrated:
        migrated["origin"] = "auto"

    return migrated


def migrate_file(path: Path, apply: bool = False) -> tuple[int, int]:
    """Migrate a single YAML file.

    Returns: (total_cases, migrated_cases)
    """
    with open(path, encoding="utf-8") as f:
        cases = yaml.safe_load(f)

    if not isinstance(cases, list):
        print(f"  WARNING: {path.name} is not a list, skipping")
        return 0, 0

    total = len(cases)
    migrated_count = 0
    migrated_cases = []

    for case in cases:
        if not isinstance(case, dict):
            migrated_cases.append(case)
            continue

        original_keys = set(case.keys())
        migrated = migrate_case(case)
        migrated_cases.append(migrated)

        if "test_types" not in original_keys:
            migrated_count += 1

    if apply and migrated_count > 0:
        # Preserve order: id, profile, prompt, test_types, origin, expected
        ordered_cases = []
        for case in migrated_cases:
            ordered = {}
            for key in ["id", "profile", "prompt", "test_types", "origin", "expected"]:
                if key in case:
                    ordered[key] = case[key]
            # Add any remaining keys
            for key, value in case.items():
                if key not in ordered:
                    ordered[key] = value
            ordered_cases.append(ordered)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                ordered_cases,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )

    return total, migrated_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate eval YAML files to add test_types and origin fields"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (default: dry run)",
    )
    parser.add_argument(
        "--file",
        help="Migrate single file (default: all golden_cases_*.yaml)",
    )
    args = parser.parse_args()

    evals_dir = Path(__file__).parent.parent / "data" / "evals"

    if args.file:
        files = [evals_dir / args.file]
    else:
        files = sorted(evals_dir.glob("golden_cases_*.yaml"))

    if not files:
        print("No files found to migrate")
        return 1

    mode = "APPLYING" if args.apply else "DRY RUN"
    print(f"\n{'=' * 60}")
    print(f"  Eval Case Migration ({mode})")
    print(f"{'=' * 60}\n")

    total_all = 0
    migrated_all = 0

    for path in files:
        if not path.exists():
            print(f"  File not found: {path}")
            continue

        total, migrated = migrate_file(path, apply=args.apply)
        total_all += total
        migrated_all += migrated

        status = "MIGRATED" if args.apply and migrated > 0 else ""
        print(f"  {path.name}: {migrated}/{total} cases need migration {status}")

        # Show sample of inferred test_types
        if migrated > 0 and not args.apply:
            with open(path, encoding="utf-8") as f:
                cases = yaml.safe_load(f)

            for case in cases[:3]:  # Show first 3
                if isinstance(case, dict) and "test_types" not in case:
                    inferred = infer_test_types(case)
                    print(f"    - {case.get('id', '?')}: {inferred}")
            if total > 3:
                print(f"    ... and {total - 3} more")

    print(f"\n{'=' * 60}")
    print(f"  TOTAL: {migrated_all}/{total_all} cases {'migrated' if args.apply else 'to migrate'}")
    print(f"{'=' * 60}\n")

    if not args.apply and migrated_all > 0:
        print("  Run with --apply to write changes\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
