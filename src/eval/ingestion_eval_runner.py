"""Ingestion evaluation runner for role classification.

Compares keyword-based vs LLM-based role detection against golden cases.
This is the eval-first approach to validating LLM role classification
before replacing keyword lists.

Usage:
    # Baseline eval (keywords)
    python -m src.eval.ingestion_eval_runner --corpus ai-act --method keywords

    # LLM-based eval
    python -m src.eval.ingestion_eval_runner --corpus ai-act --method llm

    # Compare both methods
    python -m src.eval.ingestion_eval_runner --corpus ai-act --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .ingestion_scorers import RoleClassificationScorer, ScorerResult

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_CASES_PATH = Path(__file__).parent.parent.parent / "data" / "evals" / "ingestion_role_cases.yaml"
DEFAULT_CHUNKS_DIR = Path(__file__).parent.parent.parent / "data" / "processed"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "runs" / "ingestion_eval"


@dataclass
class IngestionCase:
    """A golden case for role classification evaluation."""

    id: str
    corpus_id: str
    location: str
    expected_roles: list[str]
    chunk_id: str | None = None
    text_snippet: str = ""
    must_not_have_roles: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class CaseResult:
    """Result of evaluating a single case."""

    case_id: str
    method: str
    passed: bool
    score: float
    expected_roles: list[str]
    actual_roles: list[str]
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of a full evaluation run."""

    corpus_id: str
    method: str
    total_cases: int
    passed: int
    failed: int
    pass_rate: float
    avg_recall: float
    avg_precision: float
    case_results: list[CaseResult]
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


def load_cases(cases_path: Path, corpus_id: str | None = None) -> list[IngestionCase]:
    """Load golden cases from YAML file.

    Args:
        cases_path: Path to YAML file
        corpus_id: Optional filter by corpus

    Returns:
        List of IngestionCase objects
    """
    with open(cases_path, encoding="utf-8") as f:
        raw_cases = yaml.safe_load(f) or []

    cases = []
    for raw in raw_cases:
        case = IngestionCase(
            id=raw["id"],
            corpus_id=raw["corpus_id"],
            location=raw["location"],
            expected_roles=raw.get("expected_roles", []),
            chunk_id=raw.get("chunk_id"),
            text_snippet=raw.get("text_snippet", ""),
            must_not_have_roles=raw.get("must_not_have_roles", []),
            notes=raw.get("notes", ""),
        )

        # Filter by corpus if specified
        if corpus_id is None or case.corpus_id == corpus_id:
            cases.append(case)

    return cases


def load_chunks(corpus_id: str, chunks_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load chunks from JSONL file.

    Args:
        corpus_id: Corpus identifier
        chunks_dir: Directory containing chunks files

    Returns:
        Dict mapping chunk_id to chunk data
    """
    if chunks_dir is None:
        chunks_dir = DEFAULT_CHUNKS_DIR

    chunks_path = chunks_dir / f"{corpus_id}_chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks: dict[str, dict[str, Any]] = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            chunk_id = chunk.get("metadata", {}).get("chunk_id", "")
            if chunk_id:
                chunks[chunk_id] = chunk

    return chunks


def find_chunk_for_case(
    case: IngestionCase,
    chunks: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Find the chunk matching a case.

    If chunk_id is specified, use that. Otherwise, find by location.

    Args:
        case: The eval case
        chunks: Dict of all chunks

    Returns:
        Matching chunk or None
    """
    # Direct chunk_id lookup
    if case.chunk_id and case.chunk_id in chunks:
        return chunks[case.chunk_id]

    # Search by location
    for chunk_id, chunk in chunks.items():
        meta = chunk.get("metadata", {})
        location_id = meta.get("location_id", "")

        # Match location patterns
        if case.location.startswith("article:"):
            art_num = case.location.split(":")[1]
            if f"/article:{art_num}" in location_id:
                return chunk

        elif case.location.startswith("annex:"):
            parts = case.location.split("/")
            annex_part = parts[0].split(":")[1]  # e.g., "iii"
            if f"/annex:{annex_part}" in location_id.lower():
                # Check for specific point if specified
                if len(parts) > 1 and "annex_point:" in parts[1]:
                    point = parts[1].split(":")[1]
                    if f"/annex_point:{point}" in location_id:
                        return chunk
                else:
                    return chunk

        elif case.location.startswith("recital:"):
            rec_num = case.location.split(":")[1]
            if f"recital:{rec_num}" in location_id:
                return chunk

    return None


def detect_roles_keywords(text: str, title: str = "") -> list[str]:
    """Detect roles using keyword matching (baseline method).

    This is the current production method from citation_graph.py.
    """
    # Import the production keyword lists
    try:
        from ..ingestion.citation_graph import _detect_roles_from_text
        return _detect_roles_from_text(text, title)
    except ImportError:
        # Fallback if import fails
        from ..engine.constants import (
            _INTENT_CLASSIFICATION_KEYWORDS_SUBSTR,
            _INTENT_DEFINITIONS_KEYWORDS_SUBSTR,
            _INTENT_ENFORCEMENT_KEYWORDS_SUBSTR,
            _INTENT_REQUIREMENTS_KEYWORDS_STRONG_SUBSTR,
            _INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR,
        )

        role_keyword_map = {
            "scope": _INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR,
            "definitions": _INTENT_DEFINITIONS_KEYWORDS_SUBSTR,
            "classification": _INTENT_CLASSIFICATION_KEYWORDS_SUBSTR,
            "obligations": _INTENT_REQUIREMENTS_KEYWORDS_STRONG_SUBSTR,
            "enforcement": _INTENT_ENFORCEMENT_KEYWORDS_SUBSTR,
        }

        combined = f"{title} {text}".lower()
        detected: list[str] = []

        for role, keywords in role_keyword_map.items():
            for kw in keywords:
                if kw.lower() in combined:
                    if role not in detected:
                        detected.append(role)
                    break

        return detected


def detect_roles_llm(text: str, title: str = "", corpus_id: str = "") -> list[str]:
    """Detect roles using LLM classification.

    This uses the updated embedding_enrichment module which now includes
    role classification in its prompt.

    Args:
        text: Chunk text content
        title: Article/annex title
        corpus_id: Corpus identifier

    Returns:
        List of detected roles from LLM
    """
    try:
        from ..ingestion.embedding_enrichment import generate_enrichment, is_enrichment_enabled

        # Temporarily ensure enrichment is enabled for this call
        # (we want to call LLM even if enrichment is disabled in config)
        import os

        original_enabled = os.environ.get("_FORCE_ENRICHMENT_ENABLED")
        os.environ["_FORCE_ENRICHMENT_ENABLED"] = "1"

        try:
            result = generate_enrichment(
                text,
                article_title=title,
                corpus_id=corpus_id,
            )

            if result and result.roles:
                return result.roles
            return []
        finally:
            if original_enabled is None:
                os.environ.pop("_FORCE_ENRICHMENT_ENABLED", None)
            else:
                os.environ["_FORCE_ENRICHMENT_ENABLED"] = original_enabled

    except Exception as e:
        logger.warning("LLM role detection failed: %s, falling back to keywords", e)
        return detect_roles_keywords(text, title)


def evaluate_case(
    case: IngestionCase,
    chunk: dict[str, Any],
    method: str = "keywords",
) -> CaseResult:
    """Evaluate a single case.

    Args:
        case: The golden case
        chunk: The chunk data
        method: Detection method ("keywords" or "llm")

    Returns:
        CaseResult with pass/fail and details
    """
    text = chunk.get("text", "")
    meta = chunk.get("metadata", {})
    title = meta.get("article_title") or meta.get("annex_title") or ""
    corpus_id = meta.get("corpus_id", "")

    # Detect roles using specified method
    if method == "keywords":
        actual_roles = detect_roles_keywords(text, title)
    elif method == "llm":
        actual_roles = detect_roles_llm(text, title, corpus_id)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Score the result
    scorer = RoleClassificationScorer()
    result = scorer.score(
        expected_roles=case.expected_roles,
        actual_roles=actual_roles,
        must_not_have_roles=case.must_not_have_roles,
    )

    return CaseResult(
        case_id=case.id,
        method=method,
        passed=result.passed,
        score=result.score,
        expected_roles=case.expected_roles,
        actual_roles=actual_roles,
        message=result.message,
        details=result.details,
    )


def run_eval(
    corpus_id: str,
    method: str = "keywords",
    cases_path: Path | None = None,
    chunks_dir: Path | None = None,
    verbose: bool = False,
) -> EvalResult:
    """Run evaluation for a corpus.

    Args:
        corpus_id: Corpus to evaluate
        method: Detection method ("keywords" or "llm")
        cases_path: Path to golden cases YAML
        chunks_dir: Directory containing chunks files
        verbose: Print detailed output

    Returns:
        EvalResult with aggregated metrics
    """
    if cases_path is None:
        cases_path = DEFAULT_CASES_PATH
    if chunks_dir is None:
        chunks_dir = DEFAULT_CHUNKS_DIR

    # Load cases and chunks
    cases = load_cases(cases_path, corpus_id)
    if not cases:
        logger.warning("No cases found for corpus: %s", corpus_id)
        return EvalResult(
            corpus_id=corpus_id,
            method=method,
            total_cases=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            avg_recall=0.0,
            avg_precision=0.0,
            case_results=[],
        )

    chunks = load_chunks(corpus_id, chunks_dir)

    # Evaluate each case
    case_results: list[CaseResult] = []
    total_recall = 0.0
    total_precision = 0.0

    for case in cases:
        chunk = find_chunk_for_case(case, chunks)
        if chunk is None:
            logger.warning("Could not find chunk for case: %s", case.id)
            # Create failed result
            result = CaseResult(
                case_id=case.id,
                method=method,
                passed=False,
                score=0.0,
                expected_roles=case.expected_roles,
                actual_roles=[],
                message="Chunk not found",
                details={},
            )
        else:
            result = evaluate_case(case, chunk, method)

        case_results.append(result)
        total_recall += result.details.get("recall", 0.0)
        total_precision += result.details.get("precision", 0.0)

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {case.id}: {result.message or 'OK'}")

    # Calculate aggregates
    passed = sum(1 for r in case_results if r.passed)
    failed = len(case_results) - passed
    pass_rate = passed / len(case_results) if case_results else 0.0
    avg_recall = total_recall / len(case_results) if case_results else 0.0
    avg_precision = total_precision / len(case_results) if case_results else 0.0

    return EvalResult(
        corpus_id=corpus_id,
        method=method,
        total_cases=len(case_results),
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        avg_recall=avg_recall,
        avg_precision=avg_precision,
        case_results=case_results,
    )


def print_results(result: EvalResult) -> None:
    """Print evaluation results to console."""
    print(f"\n{'='*60}")
    print(f"Ingestion Eval: {result.corpus_id} ({result.method})")
    print(f"{'='*60}")
    print(f"Total cases:  {result.total_cases}")
    print(f"Passed:       {result.passed}")
    print(f"Failed:       {result.failed}")
    print(f"Pass rate:    {result.pass_rate:.1%}")
    print(f"Avg recall:   {result.avg_recall:.1%}")
    print(f"Avg precision:{result.avg_precision:.1%}")

    # Show failed cases
    failed_cases = [r for r in result.case_results if not r.passed]
    if failed_cases:
        print(f"\nFailed cases ({len(failed_cases)}):")
        print("-" * 60)
        for r in failed_cases:
            print(f"  {r.case_id}")
            print(f"    Expected: {r.expected_roles}")
            print(f"    Actual:   {r.actual_roles}")
            if r.message:
                print(f"    Message:  {r.message}")


def save_results(result: EvalResult, output_dir: Path | None = None) -> Path:
    """Save evaluation results to JSON file."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ingestion_eval_{result.corpus_id}_{result.method}_{timestamp}.json"
    output_path = output_dir / filename

    # Convert to serializable dict
    data = {
        "corpus_id": result.corpus_id,
        "method": result.method,
        "timestamp": result.timestamp,
        "total_cases": result.total_cases,
        "passed": result.passed,
        "failed": result.failed,
        "pass_rate": result.pass_rate,
        "avg_recall": result.avg_recall,
        "avg_precision": result.avg_precision,
        "case_results": [
            {
                "case_id": r.case_id,
                "method": r.method,
                "passed": r.passed,
                "score": r.score,
                "expected_roles": r.expected_roles,
                "actual_roles": r.actual_roles,
                "message": r.message,
                "details": r.details,
            }
            for r in result.case_results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def compare_methods(
    corpus_id: str,
    cases_path: Path | None = None,
    chunks_dir: Path | None = None,
    verbose: bool = False,
) -> None:
    """Compare keyword vs LLM role detection.

    Args:
        corpus_id: Corpus to evaluate
        cases_path: Path to golden cases
        chunks_dir: Directory containing chunks
        verbose: Print detailed output
    """
    print(f"\nComparing role detection methods for: {corpus_id}")
    print("=" * 60)

    # Run both methods
    keyword_result = run_eval(corpus_id, "keywords", cases_path, chunks_dir, verbose)
    llm_result = run_eval(corpus_id, "llm", cases_path, chunks_dir, verbose)

    # Print comparison
    print(f"\n{'Method':<15} {'Pass Rate':<12} {'Recall':<12} {'Precision':<12}")
    print("-" * 60)
    print(f"{'Keywords':<15} {keyword_result.pass_rate:<12.1%} {keyword_result.avg_recall:<12.1%} {keyword_result.avg_precision:<12.1%}")
    print(f"{'LLM':<15} {llm_result.pass_rate:<12.1%} {llm_result.avg_recall:<12.1%} {llm_result.avg_precision:<12.1%}")

    # Show cases where methods differ
    print("\nCases with different results:")
    print("-" * 60)

    keyword_by_id = {r.case_id: r for r in keyword_result.case_results}
    llm_by_id = {r.case_id: r for r in llm_result.case_results}

    diff_count = 0
    for case_id in keyword_by_id:
        kw = keyword_by_id[case_id]
        llm = llm_by_id.get(case_id)
        if llm and kw.passed != llm.passed:
            diff_count += 1
            kw_status = "PASS" if kw.passed else "FAIL"
            llm_status = "PASS" if llm.passed else "FAIL"
            print(f"  {case_id}: Keywords={kw_status}, LLM={llm_status}")
            if not kw.passed:
                print(f"    Keywords missing: {kw.details.get('missing', [])}")
            if llm and not llm.passed:
                print(f"    LLM missing: {llm.details.get('missing', [])}")

    if diff_count == 0:
        print("  (No differences)")

    print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate ingestion role classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run keyword baseline
  python -m src.eval.ingestion_eval_runner --corpus ai-act --method keywords

  # Run LLM-based (when implemented)
  python -m src.eval.ingestion_eval_runner --corpus ai-act --method llm

  # Compare both methods
  python -m src.eval.ingestion_eval_runner --corpus ai-act --compare
        """,
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Corpus to evaluate (e.g., ai-act, gdpr, dora)",
    )
    parser.add_argument(
        "--method",
        choices=["keywords", "llm"],
        default="keywords",
        help="Role detection method (default: keywords)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare keywords vs LLM methods",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Path to golden cases YAML",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        if args.compare:
            compare_methods(args.corpus, args.cases, verbose=args.verbose)
        else:
            result = run_eval(
                args.corpus,
                args.method,
                args.cases,
                verbose=args.verbose,
            )
            print_results(result)
            output_path = save_results(result, args.out)
            print(f"\nResults saved to: {output_path}")

            # Exit code based on pass rate
            return 0 if result.pass_rate >= 0.8 else 1

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except Exception as e:
        logger.exception("Error during evaluation: %s", e)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
