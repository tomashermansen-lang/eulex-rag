#!/usr/bin/env python3
"""Migrate existing corpora to use eurovoc_labels instead of category.

This script fetches EuroVoc labels from EUR-Lex for all existing laws
and updates corpora.json with the new eurovoc_labels field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.eurlex_listing import list_available_legislation, DocumentType


def load_corpora(path: Path) -> dict[str, Any]:
    """Load corpora.json."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_corpora(path: Path, data: dict[str, Any]) -> None:
    """Save corpora.json with proper formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def migrate_corpora_to_eurovoc(corpora_path: Path) -> None:
    """Migrate all corpora to use eurovoc_labels."""
    print(f"Loading corpora from {corpora_path}")
    data = load_corpora(corpora_path)
    corpora = data.get("corpora", {})

    if not corpora:
        print("No corpora found.")
        return

    # Collect all CELEX numbers
    celex_numbers = []
    for corpus_id, corpus_data in corpora.items():
        celex = corpus_data.get("celex_number")
        if celex:
            celex_numbers.append(celex)

    print(f"Found {len(celex_numbers)} laws with CELEX numbers")
    print("Fetching EuroVoc labels from EUR-Lex (this may take a minute)...")

    # Fetch legislation info for all laws
    # We'll fetch a broad date range to ensure we get all laws
    legislation_list = list_available_legislation(
        year_from=2016,
        year_to=2026,
        document_type=DocumentType.ALL,
        in_force_only=False,
    )

    # Build lookup by CELEX
    eurovoc_by_celex: dict[str, list[str]] = {}
    for leg in legislation_list:
        if leg.eurovoc_labels:
            eurovoc_by_celex[leg.celex_number] = leg.eurovoc_labels

    print(f"Retrieved EuroVoc labels for {len(eurovoc_by_celex)} laws")

    # Update corpora with eurovoc_labels
    updated_count = 0
    missing_count = 0

    for corpus_id, corpus_data in corpora.items():
        celex = corpus_data.get("celex_number")
        if not celex:
            continue

        if celex in eurovoc_by_celex:
            corpus_data["eurovoc_labels"] = eurovoc_by_celex[celex]
            updated_count += 1
            print(f"  {corpus_id}: {eurovoc_by_celex[celex]}")
        else:
            print(f"  {corpus_id}: No EuroVoc labels found for {celex}")
            missing_count += 1
            # Set empty list if not found
            corpus_data["eurovoc_labels"] = []

    # Save updated corpora
    save_corpora(corpora_path, data)

    print(f"\nMigration complete:")
    print(f"  Updated: {updated_count}")
    print(f"  Missing: {missing_count}")
    print(f"  Saved to: {corpora_path}")


if __name__ == "__main__":
    corpora_path = PROJECT_ROOT / "data" / "processed" / "corpora.json"

    if not corpora_path.exists():
        print(f"Error: {corpora_path} not found")
        sys.exit(1)

    migrate_corpora_to_eurovoc(corpora_path)
