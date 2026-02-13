"""Example question generator for new corpora.

Single Responsibility: Manage example questions shown in the UI chat dropdown.
Uses centralized LLM generation from ingestion_generation.py.

Output is stored in data/processed/example_questions.json.
"""

from __future__ import annotations

import json
import logging

from src.ingestion.ingestion_generation import (
    generate_example_questions,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)

EXAMPLES_FILE = PROJECT_ROOT / "data" / "processed" / "example_questions.json"


def load_examples() -> dict[str, dict[str, list[str]]]:
    """Load existing example questions from JSON file."""
    if EXAMPLES_FILE.exists():
        try:
            with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load examples file: %s", e)
    return {}


def save_examples(examples: dict[str, dict[str, list[str]]]) -> None:
    """Save example questions to JSON file."""
    EXAMPLES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


def add_corpus_examples(
    corpus_id: str,
    display_name: str,
    celex_number: str,
) -> bool:
    """Generate and save example questions for a new corpus.

    Args:
        corpus_id: Short corpus ID (e.g., 'nis2')
        display_name: Full display name
        celex_number: CELEX number

    Returns:
        True if successful, False otherwise
    """
    questions = generate_example_questions(corpus_id, display_name, celex_number)
    if not questions:
        logger.warning("Failed to generate example questions for %s", corpus_id)
        return False

    examples = load_examples()
    examples[corpus_id] = questions
    save_examples(examples)

    logger.info("Generated example questions for %s", corpus_id)
    return True


def remove_corpus_examples(corpus_id: str) -> bool:
    """Remove example questions for a corpus.

    Args:
        corpus_id: Corpus ID to remove

    Returns:
        True if questions were removed, False if not found
    """
    examples = load_examples()
    if corpus_id in examples:
        del examples[corpus_id]
        save_examples(examples)
        logger.info("Removed example questions for %s", corpus_id)
        return True
    return False


def get_corpus_examples(corpus_id: str) -> dict[str, list[str]] | None:
    """Get example questions for a specific corpus.

    Args:
        corpus_id: Corpus ID

    Returns:
        Dict with LEGAL and ENGINEERING question lists, or None if not found
    """
    examples = load_examples()
    return examples.get(corpus_id)
