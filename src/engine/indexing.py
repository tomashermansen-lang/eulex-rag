"""Indexing module: vectorstore ingestion logic extracted from rag.py.

This module handles:
- Loading .txt documents into vectorstore
- Ingesting JSONL chunks
- Embedding generation and Chroma upsert operations
- Collection reset/recovery logic

All functions accept an engine instance to access collections, embedding model, etc.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from chromadb.errors import InvalidArgumentError

from ..common.config_loader import load_settings
from .types import RAGEngineError

logger = logging.getLogger(__name__)


def index_documents(engine) -> None:
    """Load .txt files from engine.docs_path into the main collection.

    Args:
        engine: RAGEngine instance with docs_path, collection, etc.
    """
    if not os.path.isdir(engine.docs_path):
        raise RAGEngineError(f"Document directory not found: {engine.docs_path}")

    documents: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for filename in os.listdir(engine.docs_path):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(engine.docs_path, filename)
        if not os.path.isfile(file_path):
            continue

        doc_id = filename
        existing = engine.collection.get(ids=[doc_id])
        if existing.get("documents"):
            continue

        with open(file_path, "r", encoding="utf-8") as file_handle:
            documents.append(file_handle.read())
            ids.append(doc_id)
            metadatas.append({"source": filename})

    if not documents:
        if engine.collection.count() == 0:
            raise RAGEngineError("No .txt documents found to ingest.")
        return

    _upsert_with_embeddings(engine, ids=ids, documents=documents, metadatas=metadatas)


def index_jsonl(engine, jsonl_path: str, batch_size: int = 32) -> None:
    """Ingest JSONL chunks into vectorstore (main collection only).

    Args:
        engine: RAGEngine instance
        jsonl_path: Path to JSONL file with chunks
        batch_size: Number of chunks to batch before upserting
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise RAGEngineError(f"Chunk file not found: {jsonl_path}")

    documents: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    def flush_docs_batch():
        if not documents:
            return
        # Chroma metadata upserts may keep stale keys when the new metadata omits them.
        # Delete first to ensure the stored metadata exactly matches the regenerated JSONL.
        try:
            engine.collection.delete(ids=ids.copy())
        except Exception:  # noqa: BLE001
            pass
        _upsert_with_embeddings(
            engine,
            ids=ids.copy(),
            documents=documents.copy(),
            metadatas=metadatas.copy(),
        )
        documents.clear()
        ids.clear()
        metadatas.clear()

    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            payload = line.strip()
            if not payload:
                continue
            data = json.loads(payload)
            text = data.get("text", "").strip()
            metadata = data.get("metadata", {}) or {}
            if not text:
                continue

            # Skip TOC metadata entries (doc_type == "toc") - we no longer index TOC
            if metadata.get("doc_type") == "toc":
                continue

            # Stamp corpus id if missing so mixed pipelines can still be traced.
            if (
                isinstance(metadata, dict)
                and "corpus_id" not in metadata
                and "law_id" not in metadata
            ):
                cid = str(getattr(engine, "corpus_id", "") or "").strip()
                if cid:
                    metadata["corpus_id"] = cid

            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                source_slug = (
                    str(metadata.get("source", "doc")).lower().replace(" ", "-")
                )
                page = metadata.get("page", "p0")
                chunk_index = metadata.get("chunk_index", 0)
                chunk_id = f"{source_slug}-{page}-{chunk_index}"

            documents.append(text)
            ids.append(chunk_id)
            metadatas.append(metadata)
            if len(documents) >= batch_size:
                flush_docs_batch()

    flush_docs_batch()


def _resolve_jsonl_dir(corpus_id: str, jsonl_dir: str | None) -> Path:
    """Resolve and validate the JSONL directory path."""
    if jsonl_dir is None:
        settings = load_settings()
        jsonl_dir = str(Path(settings.processed_dir) / "case_law" / corpus_id)

    dir_path = Path(jsonl_dir)
    if not dir_path.is_dir():
        raise RAGEngineError(f"Case law JSONL directory not found: {jsonl_dir}")
    return dir_path


def _upsert_with_dimension_recovery(
    engine,
    collection,
    collection_name: str,
    batch_ids: List[str],
    batch_docs: List[str],
    batch_metas: List[Dict[str, Any]],
    embeddings: List,
) -> Any:
    """Upsert a batch, recovering from dimension mismatch by recreating the collection."""
    try:
        collection.upsert(
            ids=batch_ids, documents=batch_docs,
            metadatas=batch_metas, embeddings=embeddings,
        )
    except InvalidArgumentError as exc:
        if "dimension" not in str(exc).lower():
            raise
        try:
            engine.chroma.delete_collection(collection_name)
        except Exception:  # noqa: BLE001
            pass
        collection = engine.chroma.get_or_create_collection(collection_name)
        collection.upsert(
            ids=batch_ids, documents=batch_docs,
            metadatas=batch_metas, embeddings=embeddings,
        )
    return collection


class _SkipLine:
    """Sentinel: line was blank or had no usable text (not an error)."""


class _MalformedLine:
    """Sentinel: line contained invalid JSON."""


def _parse_jsonl_line(
    line: str, corpus_id: str, line_num: int
) -> tuple[str, str, Dict[str, Any]] | _SkipLine | _MalformedLine:
    """Parse a single JSONL line.

    Returns (chunk_id, text, metadata) on success, _SkipLine for blank/empty-text
    lines, or _MalformedLine for unparseable JSON.
    """
    stripped = line.strip()
    if not stripped:
        return _SkipLine()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return _MalformedLine()
    text = data.get("text", "").strip()
    if not text:
        return _SkipLine()
    metadata = data.get("metadata", {}) or {}
    chunk_id = metadata.get("chunk_id", f"{corpus_id}-{line_num}")
    return chunk_id, text, metadata


@dataclass
class _BatchState:
    """Mutable batch accumulator for case law indexing."""

    engine: Any
    corpus_id: str
    collection_name: str
    collection: Any
    total_indexed: int = 0
    ids: List[str] = field(default_factory=list)
    docs: List[str] = field(default_factory=list)
    metas: List[Dict[str, Any]] = field(default_factory=list)

    def flush(self) -> None:
        if not self.ids:
            return
        try:
            self.collection.delete(ids=self.ids.copy())
        except Exception:  # noqa: BLE001
            pass

        enriched = [
            _enrich_document_for_embedding(doc, meta, corpus_id=self.corpus_id)
            for doc, meta in zip(self.docs, self.metas)
        ]
        embeddings = self.engine._embed(enriched)
        cleaned = [_stringify_metadata(meta) for meta in self.metas]

        self.collection = _upsert_with_dimension_recovery(
            self.engine, self.collection, self.collection_name,
            self.ids.copy(), self.docs.copy(), cleaned, embeddings,
        )

        self.total_indexed += len(self.ids)
        self.ids.clear()
        self.docs.clear()
        self.metas.clear()


def _read_jsonl_file(
    jsonl_file: Path, corpus_id: str, batch: _BatchState, batch_size: int,
) -> None:
    """Read a single JSONL file and add parsed chunks to the batch."""
    try:
        fh = open(jsonl_file, "r", encoding="utf-8")  # noqa: SIM115
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Skipping file %s: %s", jsonl_file, exc)
        return

    with fh:
        for line_num, raw_line in enumerate(fh, start=1):
            parsed = _parse_jsonl_line(raw_line, corpus_id, line_num)
            if isinstance(parsed, _SkipLine):
                continue
            if isinstance(parsed, _MalformedLine):
                logger.warning(
                    "Malformed JSON at %s line %d, skipping", jsonl_file, line_num,
                )
                continue
            chunk_id, text, metadata = parsed
            batch.ids.append(chunk_id)
            batch.docs.append(text)
            batch.metas.append(metadata)

            if len(batch.ids) >= batch_size:
                batch.flush()


def index_case_law_jsonl(
    engine,
    corpus_id: str,
    *,
    jsonl_dir: str | None = None,
    batch_size: int = 32,
) -> int:
    """Index case law JSONL files into a dedicated ChromaDB collection.

    Reads all *_chunks.jsonl files from a per-corpus directory and indexes them
    into a ``{corpus_id}_case_law`` collection. Gated by ``case_law.enabled``.

    Args:
        engine: RAGEngine instance with chroma client and _embed method.
        corpus_id: Corpus identifier (e.g. "gdpr").
        jsonl_dir: Directory containing JSONL files. Defaults to
            ``{paths.processed_dir}/case_law/{corpus_id}/``.
        batch_size: Chunks per upsert batch.

    Returns:
        Total number of chunks indexed.

    Raises:
        RAGEngineError: If the resolved directory does not exist.
    """
    settings = load_settings()
    if not settings.case_law.enabled:
        return 0

    dir_path = _resolve_jsonl_dir(corpus_id, jsonl_dir)

    jsonl_files = sorted(dir_path.glob("*_chunks.jsonl"))
    if not jsonl_files:
        logger.info("No *_chunks.jsonl files found in %s", dir_path)
        return 0

    collection_name = f"{corpus_id}_case_law"
    collection = engine.chroma.get_or_create_collection(collection_name)

    batch = _BatchState(
        engine=engine, corpus_id=corpus_id,
        collection_name=collection_name, collection=collection,
    )

    for jsonl_file in jsonl_files:
        _read_jsonl_file(jsonl_file, corpus_id, batch, batch_size)

    batch.flush()
    return batch.total_indexed


def _resolve_heading(doc: str, meta: Dict[str, Any]) -> str:
    """Resolve the heading for a chunk, inferring from first line for annex points."""
    heading = str(meta.get("heading_path_display", "")).strip()

    if meta.get("annex_point") and not meta.get("annex_point_title"):
        first_line = doc.split("\n")[0].strip() if doc else ""
        if first_line and len(first_line) < 100 and not first_line.startswith("a)"):
            heading = f"{heading} ({first_line})" if heading else first_line

    return heading


def _enrich_terms_fallback(result: str, meta: Dict[str, Any], corpus_id: str) -> str:
    """Attempt LLM-based term enrichment as a fallback when pre-computed terms are absent."""
    try:
        from ..ingestion.embedding_enrichment import (
            enrich_text_for_embedding,
            is_enrichment_enabled,
        )

        if is_enrichment_enabled():
            return enrich_text_for_embedding(result, meta, corpus_id=corpus_id)
    except ImportError:
        pass
    except Exception as e:
        logger.debug("Embedding enrichment failed: %s", e)
    return result


def _enrich_document_for_embedding(
    doc: str, meta: Dict[str, Any], corpus_id: str = ""
) -> str:
    """Prepend contextual description, heading_path_display and enrichment terms for embedding.

    This enables retrieval to match on:
    1. Semantic context (what the chunk is about - from contextual_description)
    2. Structural context (chapter titles, article names - from heading_path_display)
    3. Colloquial search terms (from metadata or LLM-generated)

    The enrichment bridges the semantic gap between legal terminology
    and everyday user queries (e.g., "musik AI" → "syntetisk indhold").
    """
    result = doc

    heading = _resolve_heading(doc, meta)
    if heading:
        result = f"{heading}\n\n{doc}"

    contextual_description = meta.get("contextual_description", "")
    if contextual_description and isinstance(contextual_description, str):
        result = f"[Kontekst: {contextual_description}]\n\n{result}"

    enrichment_terms = meta.get("enrichment_terms")
    if enrichment_terms and isinstance(enrichment_terms, list):
        terms_block = " | ".join(str(t) for t in enrichment_terms)
        result = f"[Søgetermer: {terms_block}]\n\n{result}"
    else:
        result = _enrich_terms_fallback(result, meta, corpus_id)

    return result


def _stringify_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Convert list-valued metadata entries to ChromaDB-compatible strings.

    ChromaDB doesn't support list values. Enrichment terms are joined with
    " | " (pipe-separated); other lists use ", " (comma-separated).
    """
    result = {}
    for k, v in meta.items():
        if k in ("enrichment_terms", "_enrichment_terms") and isinstance(v, list):
            result[k] = " | ".join(str(t) for t in v) if v else ""
        elif isinstance(v, list):
            result[k] = ", ".join(str(t) for t in v) if v else ""
        else:
            result[k] = v
    return result


def _upsert_with_embeddings(
    engine,
    *,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    """Upsert documents into main collection with embeddings.

    Documents are enriched with heading_path_display and LLM-generated terms
    before embedding to improve semantic retrieval.

    Args:
        engine: RAGEngine instance
        ids: Document IDs
        documents: Document texts
        metadatas: Document metadata dicts
    """
    # Get corpus_id from engine or first metadata entry
    corpus_id = str(getattr(engine, "corpus_id", "") or "").strip()
    if not corpus_id and metadatas:
        corpus_id = str(
            metadatas[0].get("corpus_id", "") or metadatas[0].get("law_id", "")
        ).strip()

    # Enrich documents with heading and LLM-generated terms for better semantic matching
    enriched_documents = [
        _enrich_document_for_embedding(doc, meta, corpus_id=corpus_id)
        for doc, meta in zip(documents, metadatas)
    ]

    embeddings = engine._embed(enriched_documents)

    cleaned_metadatas = [_stringify_metadata(meta) for meta in metadatas]

    # Store ORIGINAL documents (not enriched) to avoid duplication in LLM context
    try:
        engine.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=cleaned_metadatas,
            embeddings=embeddings,
        )
    except InvalidArgumentError as exc:
        message = str(exc).lower()
        if "dimension" not in message:
            raise
        _reset_collection(engine, engine.collection_name, "collection")
        engine.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=cleaned_metadatas,
            embeddings=embeddings,
        )


def _upsert_with_embeddings_to(
    engine,
    *,
    collection_name: str,
    collection_attr: str,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    """Upsert documents into a specific collection with embeddings.

    Args:
        engine: RAGEngine instance
        collection_name: Name of collection (for reset)
        collection_attr: Attribute name on engine (e.g., 'collection')
        ids: Document IDs
        documents: Document texts
        metadatas: Document metadata dicts
    """
    embeddings = engine._embed(documents)
    collection = getattr(engine, collection_attr)
    try:
        collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )
    except InvalidArgumentError as exc:
        message = str(exc).lower()
        if "dimension" not in message:
            raise
        _reset_collection(engine, collection_name, collection_attr)
        collection = getattr(engine, collection_attr)
        collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )


def _reset_collection(engine, collection_name: str, collection_attr: str) -> None:
    """Reset (delete and recreate) a Chroma collection.

    Args:
        engine: RAGEngine instance
        collection_name: Name of collection to reset
        collection_attr: Attribute name on engine to update
    """
    try:
        engine.chroma.delete_collection(collection_name)
    except Exception:  # noqa: BLE001
        pass
    setattr(engine, collection_attr, engine.chroma.create_collection(collection_name))
