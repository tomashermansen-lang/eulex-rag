"""Ingest EUR-Lex Convex HTML into chunked JSONL.

Hard-break HTML-only ingestion:
- Place one or more EUR-Lex Convex HTML files in data/raw/*.html
- The corpus id is derived from the filename stem (slugified)
- This script produces data/processed/<corpus_id>_chunks.jsonl

Run:
  python -m src.ingest_eurlex_engine --corpus gdpr
  python -m src.ingest_eurlex_engine --corpus ai-act

If --corpus is omitted, it ingests all discovered corpora.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from ..common.corpora_inventory import default_corpora_path, load_corpora_inventory, save_corpora_inventory, upsert_corpus_inventory
from ..common.corpus_registry import default_registry_path, derive_aliases, load_registry, save_registry, upsert_corpus
from .html_chunks import HtmlChunkingConfig, PreflightResult, chunk_html_file, preflight_check_html, write_jsonl
from ..common.metadata_schema import build_doc_id, build_source_path, compute_doc_version_from_file, stamp_common_metadata
from ..common.config_loader import load_settings


logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of ingesting a document with preflight analysis."""

    output_path: Path
    chunk_count: int
    preflight: PreflightResult

    # Stats
    chunks_with_structure: int = 0
    structure_coverage_pct: float = 0.0

    def summary(self) -> str:
        """Human-readable summary of ingestion result."""
        lines = [
            f"✓ {self.chunk_count} chunks written to {self.output_path.name}",
            f"  Structure coverage: {self.structure_coverage_pct:.1f}%",
        ]
        if self.preflight.handled:
            handled = ", ".join(f"{v} {k}" for k, v in sorted(self.preflight.handled.items()))
            lines.append(f"  Handled: {handled}")
        if self.preflight.unhandled:
            unhandled = ", ".join(f"{v} {k}" for k, v in sorted(self.preflight.unhandled.items()))
            lines.append(f"  ⚠️ Unhandled: {unhandled}")
        return "\n".join(lines)


def _slugify(value: str) -> str:
    value = (value or "").strip().lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9\-]+", "", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "doc"


def _spaced_word(word: str) -> str:
    return r"\s*".join(map(re.escape, word))


EURLEX_HTML_REFERENCE_PATTERNS = {
    "chapter": re.compile(rf"(?i)^\s*(?:{_spaced_word('chapter')}|{_spaced_word('kapitel')})\s+([ivxlcdm]+)\b"),
    "section": re.compile(
        rf"(?i)^\s*(?:{_spaced_word('section')}|{_spaced_word('afsnit')}|{_spaced_word('afdeling')})\s+([0-9]+|[ivxlcdm]+)\b"
    ),
    "article": re.compile(rf"(?i)^\s*(?:{_spaced_word('article')}|{_spaced_word('artikel')})\s*(\d{{1,3}}[a-z]?)\b"),
    "annex": re.compile(rf"(?i)^\s*(?:{_spaced_word('annex')}|{_spaced_word('bilag')})\s+([ivxlcdm]+)\b"),
}


def _enrich_rows_for_embedding(
    rows: Iterator[dict],
    *,
    corpus_id: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterator[dict]:
    """Apply LLM enrichment to generate contextual description, search terms, and roles.

    Uses concurrent processing for ~10x speedup over sequential calls.
    Results are stored in metadata for visibility in JSONL:
    - enrichment_terms: List of colloquial search terms
    - contextual_description: Semantic description of the chunk content
    - roles: LLM-classified roles (scope, definitions, classification, obligations, enforcement)

    Args:
        rows: Iterator of chunk dicts with 'text' and 'metadata'
        corpus_id: Corpus identifier

    Yields:
        Chunk dicts with enrichment added to metadata
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .embedding_enrichment import (
        generate_enrichment,
        is_enrichment_enabled,
        get_max_concurrent,
        get_batch_size,
    )

    if not is_enrichment_enabled():
        logger.info("Embedding enrichment disabled - skipping term generation")
        yield from rows
        return

    # Collect all rows first (needed for concurrent processing)
    all_rows = list(rows)
    total_count = len(all_rows)

    if total_count == 0:
        return

    max_concurrent = get_max_concurrent()
    batch_size = get_batch_size()

    logger.info(
        "Starting concurrent enrichment: %d chunks, max_concurrent=%d",
        total_count,
        max_concurrent,
    )

    def _build_article_title(meta: dict) -> str:
        """Build article title from metadata."""
        art = meta.get("article")
        title = meta.get("article_title", "")
        annex = meta.get("annex")
        annex_title = meta.get("annex_title", "")

        if art:
            if title:
                return f"Artikel {art} - {title}"
            return f"Artikel {art}"
        elif annex:
            if annex_title:
                return f"Bilag {annex} - {annex_title}"
            return f"Bilag {annex}"
        return ""

    def _enrich_single(idx: int, row: dict) -> tuple[int, object]:
        """Enrich a single row. Returns (index, result or None)."""
        text = row.get("text", "")
        meta = row.get("metadata", {})
        article_title = _build_article_title(meta)

        result = generate_enrichment(
            text,
            article_title=article_title,
            corpus_id=corpus_id,
            metadata=meta,
        )
        return (idx, result)

    # Process concurrently
    enriched_count = 0
    results: dict[int, object] = {}

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(_enrich_single, i, row): i
            for i, row in enumerate(all_rows)
        }

        completed = 0
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            completed += 1

            # Progress logging
            if completed <= 3 or completed % batch_size == 0 or completed == total_count:
                logger.info(
                    "Enrichment progress: %d/%d (%.1f%%)",
                    completed,
                    total_count,
                    100.0 * completed / total_count,
                )

            # Invoke progress callback for UI updates
            if progress_callback is not None:
                progress_callback(completed, total_count)

    # Apply results to rows in original order and yield
    for i, row in enumerate(all_rows):
        result = results.get(i)
        meta = row.get("metadata", {})

        if result:
            meta["enrichment_terms"] = result.search_terms
            if result.contextual_description:
                meta["contextual_description"] = result.contextual_description
            if result.roles:
                meta["roles"] = result.roles
            enriched_count += 1

        yield row

    logger.info(
        "Embedding enrichment complete: %d/%d chunks enriched (%.1f%%)",
        enriched_count,
        total_count,
        (100.0 * enriched_count / total_count) if total_count else 0,
    )


EURLEX_HTML_INLINE_LOCATION_PATTERNS = {
    "paragraph": re.compile(r"(?i)^\s*(?:(?:stk\.|stykke)\s*(\d{1,3})\b|\((\d{1,3})\)\s+|(\d{1,3})\.\s+)"),
    # EUR-Lex sometimes separates 'g)' into its own block; allow zero-or-more whitespace.
    "litra": re.compile(r"(?i)^\s*(?:(?:lit\.|litra)\s*([a-z])\b|([a-z])\)\s*)"),
}


def _discover_html_files(raw_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for p in sorted(raw_dir.glob("*.html")):
        corpus_id = _slugify(p.stem)
        if corpus_id in mapping and mapping[corpus_id] != p:
            raise SystemExit(
                f"Duplicate corpus id {corpus_id!r} from files: {mapping[corpus_id].name!r} and {p.name!r}. "
                "Rename one of the HTML files."
            )
        mapping[corpus_id] = p
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EUR-Lex HTML -> chunk JSONL")
    parser.add_argument("--corpus", default="", help="Corpus id derived from filename (slug of <file>.html)")
    parser.add_argument("--raw-dir", default="", help="Override raw HTML directory (default from settings)")
    parser.add_argument("--out-dir", default="", help="Override processed output directory (default from settings)")
    parser.add_argument("--language", default="da", help="Language code for metadata (default: da)")
    parser.add_argument("--registry-path", default="", help="Path to corpus registry (default: data/processed/corpus_registry.json)")
    parser.add_argument("--corpora-path", default="", help="Path to corpora inventory (default: data/processed/corpora.json)")
    parser.add_argument(
        "--no-update-corpora-inventory",
        action="store_true",
        help="Disable automatic update of corpora.json",
    )
    parser.add_argument("--display-name", default="", help="Display name for registry entry (default: CORPUS_ID.upper())")
    parser.add_argument("--alias", action="append", default=[], help="Alias for registry entry (can be repeated)")
    parser.add_argument(
        "--doc-version",
        default="",
        help="Override doc_version. If omitted, uses sha256 of the raw HTML file.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING)")
    return parser.parse_args()


def run_ingestion_for_file(
    *,
    corpus_id: str,
    html_path: Path,
    out_dir: Path,
    chunk_tokens: int,
    overlap: int,
    language: str = "da",
    doc_version: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> IngestionResult:
    """Ingest a EUR-Lex HTML file into chunked JSONL.

    Returns:
        IngestionResult with output path, chunk stats, and preflight analysis.
    """
    out_path = out_dir / f"{corpus_id}_chunks.jsonl"

    # --- PRE-FLIGHT CHECK ---
    # Analyze document for patterns we can/cannot handle before chunking
    html_content = html_path.read_text(encoding="utf-8", errors="replace")
    preflight = preflight_check_html(html_content, enable_eurlex_structural_ids=True)

    # Log preflight results with visual separator
    print(f"\n{'─' * 60}")
    print(f"[preflight] corpus={corpus_id}")
    print(f"{'─' * 60}")

    if preflight.handled:
        print("  Handled patterns:")
        for pattern, count in sorted(preflight.handled.items()):
            print(f"    ✓ {pattern}: {count}")

    if preflight.unhandled:
        print("  ⚠️  Unhandled patterns:")
        for pattern, count in sorted(preflight.unhandled.items()):
            print(f"    ⚠️ {pattern}: {count}")

    # Detailed warnings
    if preflight.has_warnings():
        print(f"\n  Format warnings ({len(preflight.warnings)}):")
        for w in preflight.warnings:
            print(f"    [{w.severity.upper()}] {w.message}")
            if w.suggestion:
                print(f"      → {w.suggestion}")

    if not preflight.unhandled and not preflight.has_warnings():
        print("  ✓ All patterns recognized - ready for chunking")

    print(f"{'─' * 60}\n")

    project_root = Path(__file__).resolve().parent.parent
    resolved_doc_id = build_doc_id(corpus_id=corpus_id, source_stem=html_path.stem)
    resolved_doc_version = doc_version or compute_doc_version_from_file(html_path)
    source_path = build_source_path(html_path=html_path, project_root=project_root)

    base_metadata: dict[str, object] = {}
    stamp_common_metadata(
        base_metadata,
        corpus_id=corpus_id,
        doc_id=resolved_doc_id,
        doc_version=resolved_doc_version,
        language=language,
        source_type="eurlex_html",
        source_path=source_path,
        source=html_path.stem,
    )

    rows = chunk_html_file(
        html_path,
        config=HtmlChunkingConfig(
            chunk_tokens=chunk_tokens,
            overlap=overlap,
            flush_each_text_block=False,
        ),
        base_metadata=base_metadata,
        reference_patterns=EURLEX_HTML_REFERENCE_PATTERNS,
        inline_location_patterns={
            "paragraph": EURLEX_HTML_INLINE_LOCATION_PATTERNS["paragraph"],
            "litra": EURLEX_HTML_INLINE_LOCATION_PATTERNS["litra"],
        },
        inline_location_requires={
            "paragraph": ("article",),
            "litra": ("article",),
        },
        reset_on={"article": ("paragraph", "litra")},
        initial_reference_state={
            "preamble": None,
            "citation": None,
            "recital": None,
            "chapter": None,
            "section": None,
            "article": None,
            "paragraph": None,
            "litra": None,
            "annex": None,
        },
        enable_eurlex_structural_ids=True,
    )

    # Post-ingest coverage report (stdout): how many chunks have citable structure.
    total = 0
    with_struct = 0
    examples: list[tuple[str, str, str]] = []
    max_examples = 8

    def _rows_with_stats():
        nonlocal total, with_struct, examples
        for row in rows:
            total += 1
            meta = (row or {}).get("metadata") or {}
            has = bool(
                meta.get("chapter")
                or meta.get("article")
                or meta.get("annex")
                or meta.get("preamble")
                or meta.get("citation")
                or meta.get("recital")
            )
            if has:
                with_struct += 1
            else:
                if len(examples) < max_examples:
                    examples.append(
                        (
                            str(meta.get("chunk_id") or ""),
                            str(meta.get("location_id") or ""),
                            str(meta.get("heading_path") or ""),
                        )
                    )
            yield row

    # Apply LLM enrichment to generate colloquial search terms
    enriched_rows = _enrich_rows_for_embedding(
        _rows_with_stats(),
        corpus_id=corpus_id,
        progress_callback=progress_callback,
    )

    write_jsonl(out_path, enriched_rows)

    pct = (100.0 * float(with_struct) / float(total)) if total else 0.0
    print(f"[eurlex_ingest_report] corpus={corpus_id} chunks_total={total} chunks_with_(chapter|article|annex)={with_struct} ({pct:.1f}%)")
    if examples:
        print("[eurlex_ingest_report] examples_missing_citable_metadata:")
        for chunk_id, location_id, heading_path in examples:
            print(f"- chunk_id={chunk_id} location_id={location_id} heading_path={heading_path}")

    return IngestionResult(
        output_path=out_path,
        chunk_count=total,
        preflight=preflight,
        chunks_with_structure=with_struct,
        structure_coverage_pct=pct,
    )


def run_ingestion() -> None:
    settings = load_settings()

    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    project_root = Path(__file__).resolve().parent.parent.parent
    registry_path = (
        Path(args.registry_path).expanduser().resolve()
        if (args.registry_path or "").strip()
        else default_registry_path(project_root)
    )
    corpora_path = (
        Path(args.corpora_path).expanduser().resolve()
        if (args.corpora_path or "").strip()
        else default_corpora_path(project_root)
    )

    raw_dir = (settings.raw_html_dir if getattr(settings, "raw_html_dir", None) is not None else Path("data/raw"))
    if not raw_dir.is_absolute():
        raw_dir = Path(__file__).resolve().parent.parent.parent / raw_dir

    if args.raw_dir:
        raw_dir = Path(args.raw_dir).expanduser().resolve()

    out_dir = (settings.processed_dir if getattr(settings, "processed_dir", None) is not None else Path("data/processed"))
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent.parent.parent / out_dir

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()

    files = _discover_html_files(raw_dir)

    wanted = _slugify(args.corpus) if args.corpus else ""

    if wanted:
        html_path = files.get(wanted)
        if html_path is None:
            raise SystemExit(f"Ukendt --corpus {wanted!r}. Gyldige værdier: {', '.join(sorted(files.keys()))}")
        out = run_ingestion_for_file(
            corpus_id=wanted,
            html_path=html_path,
            out_dir=out_dir,
            chunk_tokens=int(getattr(settings, "eurlex_chunk_tokens", 500)),
            overlap=int(getattr(settings, "eurlex_overlap", 100)),
            language=str(args.language or "da"),
            doc_version=(str(args.doc_version).strip() or None),
        )
        logger.info("Wrote %s", out.output_path)
        logger.info(out.summary())

        if not out.output_path.exists() or out.output_path.stat().st_size <= 0:
            raise SystemExit(f"Ingestion fejlede: output-fil blev ikke skrevet korrekt: {out.output_path}")

        display_name = (str(args.display_name).strip() or wanted.upper())
        aliases = list(args.alias) if args.alias else derive_aliases(wanted, display_name)
        registry = load_registry(registry_path)
        upsert_corpus(registry, wanted, display_name, aliases)
        save_registry(registry_path, registry)
        logger.info("Opdaterede corpus registry: %s (%s)", registry_path, wanted)

        if not bool(args.no_update_corpora_inventory):
            inv = load_corpora_inventory(corpora_path)
            upsert_corpus_inventory(
                inv,
                wanted,
                display_name=(str(args.display_name).strip() or wanted.upper()),
                enabled=True,
                extra={
                    "chunks_collection": f"{wanted}_documents",
                    "max_distance": None,
                },
            )
            save_corpora_inventory(corpora_path, inv)
            logger.info("Opdaterede corpora inventory: %s (1 corpus)", corpora_path)
        return

    if not files:
        raise SystemExit(f"Ingen HTML-filer fundet i {raw_dir}")

    registry = load_registry(registry_path)

    updated_corpora: list[str] = []
    inv = None
    if not bool(args.no_update_corpora_inventory):
        inv = load_corpora_inventory(corpora_path)

    for corpus_id, html_path in files.items():
        out = run_ingestion_for_file(
            corpus_id=corpus_id,
            html_path=html_path,
            out_dir=out_dir,
            chunk_tokens=int(getattr(settings, "eurlex_chunk_tokens", 500)),
            overlap=int(getattr(settings, "eurlex_overlap", 100)),
            language=str(args.language or "da"),
            doc_version=(str(args.doc_version).strip() or None),
        )
        logger.info("Wrote %s", out.output_path)
        logger.info(out.summary())

        if not out.output_path.exists() or out.output_path.stat().st_size <= 0:
            raise SystemExit(f"Ingestion fejlede: output-fil blev ikke skrevet korrekt: {out.output_path}")

        display_name = corpus_id.upper()
        aliases = derive_aliases(corpus_id, display_name)
        upsert_corpus(registry, corpus_id, display_name, aliases)

        if inv is not None:
            upsert_corpus_inventory(
                inv,
                corpus_id,
                display_name=corpus_id.upper(),
                enabled=True,
                extra={
                    "chunks_collection": f"{corpus_id}_documents",
                    "max_distance": None,
                },
            )
            updated_corpora.append(corpus_id)

    save_registry(registry_path, registry)
    logger.info("Opdaterede corpus registry: %s (%s)", registry_path, ", ".join(sorted(files.keys())))

    if inv is not None:
        save_corpora_inventory(corpora_path, inv)
        logger.info("Opdaterede corpora inventory: %s (%d corpora)", corpora_path, len(updated_corpora))


if __name__ == "__main__":
    run_ingestion()
