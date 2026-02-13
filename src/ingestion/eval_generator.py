"""Eval case generator for new corpora.

Single Responsibility: Generate and manage golden test cases (eval cases)
for the RAG evaluation system. Uses LLM to generate starter cases that
can be manually reviewed and refined.

Output format matches existing golden_cases_*.yaml files in data/evals/.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.ingestion.ingestion_generation import (
    generate_eval_cases,
    save_eval_cases,
    EvalCase,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)

EVALS_DIR = PROJECT_ROOT / "data" / "evals"


def add_corpus_eval_cases(
    corpus_id: str,
    display_name: str,
    celex_number: str,
    *,
    num_cases: int = 15,
) -> bool:
    """Generate and save eval cases for a new corpus.

    Creates a golden_cases_{corpus_id}.yaml file with LLM-generated
    test cases. These should be manually reviewed for quality.

    Args:
        corpus_id: Short corpus ID (e.g., 'nis2')
        display_name: Full display name
        celex_number: CELEX number
        num_cases: Number of cases to generate (default 15)

    Returns:
        True if successful, False otherwise
    """
    cases = generate_eval_cases(
        corpus_id,
        display_name,
        celex_number,
        num_cases=num_cases,
    )

    if not cases:
        logger.warning("Failed to generate eval cases for %s", corpus_id)
        return False

    try:
        save_eval_cases(cases, corpus_id)
        logger.info("Generated %d eval cases for %s", len(cases), corpus_id)
        return True
    except Exception as e:
        logger.error("Failed to save eval cases for %s: %s", corpus_id, e)
        return False


def remove_corpus_eval_cases(corpus_id: str) -> bool:
    """Remove eval cases for a corpus.

    Args:
        corpus_id: Corpus ID to remove

    Returns:
        True if file was removed, False if not found
    """
    # Check both underscore and hyphen variants
    for variant in [corpus_id, corpus_id.replace("-", "_"), corpus_id.replace("_", "-")]:
        eval_file = EVALS_DIR / f"golden_cases_{variant}.yaml"
        if eval_file.exists():
            eval_file.unlink()
            logger.info("Removed eval cases for %s", corpus_id)
            return True

    return False


def get_eval_cases_path(corpus_id: str) -> Path | None:
    """Get the path to eval cases file if it exists.

    Args:
        corpus_id: Corpus ID

    Returns:
        Path to eval file or None if not found
    """
    for variant in [corpus_id, corpus_id.replace("-", "_"), corpus_id.replace("_", "-")]:
        eval_file = EVALS_DIR / f"golden_cases_{variant}.yaml"
        if eval_file.exists():
            return eval_file

    return None


def corpus_has_eval_cases(corpus_id: str) -> bool:
    """Check if a corpus has eval cases.

    Args:
        corpus_id: Corpus ID

    Returns:
        True if eval file exists
    """
    return get_eval_cases_path(corpus_id) is not None


def load_eval_cases(corpus_id: str) -> list[EvalCase] | None:
    """Load eval cases from file.

    Args:
        corpus_id: Corpus ID

    Returns:
        List of EvalCase objects or None if not found
    """
    import yaml

    path = get_eval_cases_path(corpus_id)
    if not path:
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            cases_data = yaml.safe_load(f)

        if not isinstance(cases_data, list):
            return None

        cases = []
        for case_dict in cases_data:
            expected = case_dict.get("expected", {})
            case = EvalCase(
                id=case_dict.get("id", ""),
                profile=case_dict.get("profile", "LEGAL"),
                prompt=case_dict.get("prompt", ""),
                must_include_any_of=expected.get("must_include_any_of", []),
                notes=expected.get("notes", ""),
                allow_empty_references=expected.get("allow_empty_references", False),
                must_have_article_support_for_normative=expected.get(
                    "must_have_article_support_for_normative", True
                ),
                must_not_include_any_of=expected.get("must_not_include_any_of", []),
            )
            cases.append(case)

        return cases

    except Exception as e:
        logger.error("Failed to load eval cases for %s: %s", corpus_id, e)
        return None
