"""Case law ingestion CLI — orchestrates the full case law pipeline.

Composes CELLAR client, HTML parser, chunker, and enrichment to ingest
CJEU case law for a given legislation corpus. Follows the same structural
pattern as eurlex_engine.py.

Usage:
    python -m src.ingestion.case_law_engine --corpus gdpr [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from src.common.config_loader import (
    get_case_law_settings,
    load_settings,
)
from src.ingestion.case_law_chunker import ChunkRow, chunk_judgment
from src.ingestion.case_law_parser import parse_judgment_html
from src.ingestion.case_law_types import CaseDocumentType
from src.ingestion.cellar_client import (
    fetch_case_document_url,
    fetch_cases_for_legislation,
    wait_for_rate_limit,
)
from src.ingestion.cellar_models import CaseMetadata, CellarQueryError
from src.ingestion.embedding_enrichment import (
    generate_enrichment,
    get_max_concurrent,
    is_enrichment_enabled,
)
from src.ingestion.eurlex_listing import validate_eurlex_url
from src.ingestion.html_chunks import write_jsonl

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CaseResult:
    """Result of processing a single case."""

    celex_id: str
    ecli: str
    chunk_count: int
    enriched_count: int = 0


@dataclass
class IngestionSummary:
    """Accumulates pipeline statistics."""

    cases_discovered: int = 0
    cases_processed: int = 0
    cases_skipped: int = 0
    total_chunks: int = 0
    enriched_chunks: int = 0
    enrichment_enabled: bool = False
    output_dir: str = ""
    elapsed_secs: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI arguments
# ─────────────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for case law ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest CJEU case law for a legislation corpus."
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Legislation corpus ID (e.g., gdpr)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview mode — query CELLAR without downloading",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: from settings.yaml case_law.default_output_dir)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (default: from settings.yaml case_law.default_language)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max cases to process (0 = no limit)",
    )

    args = parser.parse_args(argv)

    # Validate non-empty corpus
    if not args.corpus.strip():
        parser.error(
            "corpus cannot be empty. Valid corpus IDs can be found in settings.yaml"
        )

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Pure functions
# ─────────────────────────────────────────────────────────────────────────────


def resolve_corpus_to_celex(corpus_id: str, mapping: dict[str, str]) -> str:
    """Invert corpus_celex_mapping to find CELEX for a corpus ID.

    Case-insensitive lookup. Raises SystemExit on miss.
    """
    corpus_lower = corpus_id.lower()
    # mapping is CELEX -> corpus, invert to find CELEX
    for celex, corpus in mapping.items():
        if corpus.lower() == corpus_lower:
            return celex

    valid = sorted(set(mapping.values()))
    sys.exit(f"Unknown corpus '{corpus_id}'. Valid corpus IDs: {valid}")


def detect_document_type(
    celex_id: str, type_mapping: dict[str, str]
) -> CaseDocumentType:
    """Match CELEX against configurable patterns. First match wins."""
    for pattern, doc_type in type_mapping.items():
        if pattern in celex_id:
            return CaseDocumentType(doc_type)

    logger.warning(
        "Unrecognized CELEX pattern in '%s' — defaulting to judgment", celex_id
    )
    return CaseDocumentType.JUDGMENT


def compute_doc_version(html_content: str) -> str:
    """Compute SHA256 hex digest of UTF-8 encoded content."""
    return hashlib.sha256(html_content.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# HTML download
# ─────────────────────────────────────────────────────────────────────────────


def download_case_html(
    url: str,
    *,
    timeout_secs: int,
    max_response_bytes: int,
    user_agent: str,
) -> str:
    """Download HTML from a EUR-Lex URL with rate limiting and size guard.

    Validates URL against EUR-Lex domain, calls wait_for_rate_limit()
    before requesting, and uses streaming to enforce max_response_bytes.
    """
    validate_eurlex_url(url)
    wait_for_rate_limit()

    with requests.get(
        url,
        timeout=timeout_secs,
        stream=True,
        headers={"User-Agent": user_agent},
    ) as resp:
        resp.raise_for_status()

        chunks: list[bytes] = []
        total_bytes = 0
        for chunk in resp.iter_content(chunk_size=65_536):
            total_bytes += len(chunk)
            if total_bytes > max_response_bytes:
                raise ValueError(
                    f"Response size ({total_bytes} bytes) exceeds "
                    f"limit ({max_response_bytes} bytes)"
                )
            chunks.append(chunk)

    return b"".join(chunks).decode("utf-8")


def enrich_case_law_rows(
    rows: list[ChunkRow],
    *,
    corpus_id: str,
) -> tuple[list[ChunkRow], int]:
    """Concurrent enrichment of a single case's chunk rows.

    Returns (enriched_rows, enriched_count) where enriched_count is the
    number of rows that were successfully enriched.
    Reuses the same pattern as eurlex_engine._enrich_rows_for_embedding().
    """
    if not is_enrichment_enabled() or not rows:
        return rows, 0

    def _enrich_one(idx: int) -> int | None:
        row = rows[idx]
        meta: dict[str, Any] = dict(row.get("metadata", {}))
        case_name = meta.get("case_name", "")
        case_number = meta.get("case_number", "")
        article_title = f"{case_name} ({case_number})" if case_name else case_number

        result = generate_enrichment(
            row["text"],
            article_title=article_title,
            corpus_id=corpus_id,
            metadata=meta,
        )
        if result is not None:
            meta["enrichment_terms"] = result.search_terms
            meta["contextual_description"] = result.contextual_description
            meta["roles"] = result.roles
            row["metadata"] = meta  # type: ignore[typeddict-item]
            return idx
        return None

    enriched_count = 0
    with ThreadPoolExecutor(max_workers=get_max_concurrent()) as executor:
        futures = [executor.submit(_enrich_one, i) for i in range(len(rows))]
        for f in futures:
            if f.result() is not None:
                enriched_count += 1

    return rows, enriched_count


# ─────────────────────────────────────────────────────────────────────────────
# Per-case processing
# ─────────────────────────────────────────────────────────────────────────────


def process_single_case(
    case: CaseMetadata,
    *,
    corpus_id: str,
    out_dir: Path,
    language: str,
    corpus_celex_mapping: dict[str, str],
    celex_document_type_mapping: dict[str, str],
    timeout_secs: int,
    max_response_bytes: int,
    user_agent: str,
) -> CaseResult | None:
    """Orchestrate one case: resolve URL -> download -> parse -> chunk -> enrich -> write.

    Returns CaseResult on success, None on failure (logged as WARNING).
    """
    case_name = case.title if case.title.strip() else case.case_number

    # Step 1: Resolve HTML URL
    url = fetch_case_document_url(case.celex_id)
    if url is None:
        logger.warning(
            "[no_url] No HTML URL found for case %s (%s)", case.ecli, case.case_number
        )
        return None

    # Step 2: Download HTML
    try:
        html = download_case_html(
            url,
            timeout_secs=timeout_secs,
            max_response_bytes=max_response_bytes,
            user_agent=user_agent,
        )
    except (requests.RequestException, ValueError) as exc:
        logger.warning("[download_error] Failed to download %s: %s", case.ecli, exc)
        return None

    # Step 3: Detect document type
    doc_type = detect_document_type(case.celex_id, celex_document_type_mapping)

    # Step 4: Parse
    try:
        judgment = parse_judgment_html(
            html,
            ecli=case.ecli,
            case_number=case.case_number,
            document_type=doc_type,
            corpus_celex_mapping=corpus_celex_mapping,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.warning("[parse_error] Failed to parse %s: %s", case.ecli, exc)
        return None

    # Step 5: Chunk
    doc_version = compute_doc_version(html)
    source_path = url

    rows = chunk_judgment(
        judgment,
        case_name=case_name,
        decision_date=case.date,
        corpus_id=corpus_id,
        doc_version=doc_version,
        source_path=source_path,
        language=language,
    )

    if not rows:
        logger.warning(
            "[no_chunks] No chunks produced for case %s (%s)",
            case.ecli,
            case.case_number,
        )
        return None

    # Step 6: Enrich
    rows, enriched_count = enrich_case_law_rows(rows, corpus_id=corpus_id)

    # Step 7: Write JSONL (with path traversal guard)
    output_path = out_dir / corpus_id / f"{case.celex_id}_chunks.jsonl"
    if not output_path.resolve().is_relative_to(out_dir.resolve()):
        logger.warning(
            "[path_traversal] Output path %s escapes output directory %s for case %s",
            output_path,
            out_dir,
            case.ecli,
        )
        return None
    write_jsonl(output_path, iter(rows))  # type: ignore[arg-type]

    logger.info("Processed %s — %d chunks → %s", case.ecli, len(rows), output_path)

    return CaseResult(
        celex_id=case.celex_id,
        ecli=case.ecli,
        chunk_count=len(rows),
        enriched_count=enriched_count,
    )


def print_dry_run_table(cases: list[CaseMetadata]) -> None:
    """Print preview table of discovered cases."""
    print(f"\n{'Case Number':<20} {'ECLI':<30} {'Date':<12} {'Title'}")
    print("-" * 80)
    for case in cases:
        print(f"{case.case_number:<20} {case.ecli:<30} {case.date:<12} {case.title}")
    print(f"\nTotal: {len(cases)} cases")


def print_summary(summary: IngestionSummary) -> None:
    """Print completion statistics."""
    if summary.enrichment_enabled and summary.total_chunks > 0:
        rate = f"{summary.enriched_chunks / summary.total_chunks * 100:.1f}%"
    elif summary.enrichment_enabled:
        rate = "0.0% (0 chunks)"
    else:
        rate = "N/A"

    print(
        f"\n{'=' * 60}\n"
        f"Ingestion Summary\n"
        f"{'=' * 60}\n"
        f"  Cases discovered:    {summary.cases_discovered}\n"
        f"  Cases processed:     {summary.cases_processed}\n"
        f"  Cases skipped:       {summary.cases_skipped}\n"
        f"  Total chunks:        {summary.total_chunks}\n"
        f"  Enrichment rate:     {rate}\n"
        f"  Output directory:    {summary.output_dir}\n"
        f"  Elapsed time:        {summary.elapsed_secs:.1f}s\n"
        f"{'=' * 60}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def run_ingestion() -> None:
    """Top-level entry point for case law ingestion CLI."""
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = load_settings()

    # Resolve defaults from config
    if args.out_dir is None:
        args.out_dir = settings.case_law.default_output_dir
    if args.language is None:
        args.language = settings.case_law.default_language

    # Feature flag check
    if not settings.case_law.enabled:
        print("Case law ingestion is disabled (case_law.enabled=false)")
        return

    # Corpus resolution
    parser_settings = get_case_law_settings()
    celex_id = resolve_corpus_to_celex(
        args.corpus, parser_settings.corpus_celex_mapping
    )

    logger.info(
        "Starting case law ingestion for corpus '%s' (CELEX: %s)",
        args.corpus,
        celex_id,
    )

    # Discovery
    cases = fetch_cases_for_legislation(celex_id)

    if not cases:
        print(f"No cases found for corpus '{args.corpus}' (CELEX: {celex_id})")
        return

    logger.info("Discovered %d cases", len(cases))

    # Dry run
    if args.dry_run:
        print_dry_run_table(cases)
        return

    # Capture total discovered count before applying limit
    discovered_count = len(cases)

    # Apply limit
    if args.limit > 0:
        cases = cases[: args.limit]

    # Process cases
    out_dir = Path(args.out_dir)
    start_time = time.time()
    corpus_id = args.corpus.lower()
    enrichment_on = is_enrichment_enabled()

    summary = IngestionSummary(
        cases_discovered=discovered_count,
        enrichment_enabled=enrichment_on,
        output_dir=str(out_dir / corpus_id),
    )

    for i, case in enumerate(cases, 1):
        logger.info(
            "Processing case %d/%d: %s (%s)",
            i,
            len(cases),
            case.ecli,
            case.case_number,
        )
        try:
            result = process_single_case(
                case,
                corpus_id=corpus_id,
                out_dir=out_dir,
                language=args.language,
                corpus_celex_mapping=parser_settings.corpus_celex_mapping,
                celex_document_type_mapping=parser_settings.celex_document_type_mapping,
                timeout_secs=settings.case_law.request_timeout_secs,
                max_response_bytes=settings.case_law.max_response_bytes,
                user_agent=settings.case_law.user_agent,
            )
            if result is not None:
                summary.cases_processed += 1
                summary.total_chunks += result.chunk_count
                summary.enriched_chunks += result.enriched_count
            else:
                summary.cases_skipped += 1
        except KeyboardInterrupt:
            raise
        except (
            requests.RequestException,
            ValueError,
            OSError,
            CellarQueryError,
        ) as exc:
            logger.warning("[error] Case %s failed: %s", case.ecli, exc)
            summary.cases_skipped += 1

    summary.elapsed_secs = time.time() - start_time

    # enriched_chunks accumulated per-case from CaseResult.enriched_count

    print_summary(summary)


if __name__ == "__main__":
    run_ingestion()
