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
import os
from pathlib import Path
from typing import Any, Dict, List

from chromadb.errors import InvalidArgumentError

from .types import RAGEngineError


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
        _upsert_with_embeddings(engine, ids=ids.copy(), documents=documents.copy(), metadatas=metadatas.copy())
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
            if isinstance(metadata, dict) and "corpus_id" not in metadata and "law_id" not in metadata:
                cid = str(getattr(engine, "corpus_id", "") or "").strip()
                if cid:
                    metadata["corpus_id"] = cid

            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                source_slug = str(metadata.get("source", "doc")).lower().replace(" ", "-")
                page = metadata.get("page", "p0")
                chunk_index = metadata.get("chunk_index", 0)
                chunk_id = f"{source_slug}-{page}-{chunk_index}"

            documents.append(text)
            ids.append(chunk_id)
            metadatas.append(metadata)
            if len(documents) >= batch_size:
                flush_docs_batch()

    flush_docs_batch()


def _enrich_document_for_embedding(doc: str, meta: Dict[str, Any], corpus_id: str = "") -> str:
    """Prepend contextual description, heading_path_display and enrichment terms for embedding.
    
    This enables retrieval to match on:
    1. Semantic context (what the chunk is about - from contextual_description)
    2. Structural context (chapter titles, article names - from heading_path_display)
    3. Colloquial search terms (from metadata or LLM-generated)
    
    The enrichment bridges the semantic gap between legal terminology
    and everyday user queries (e.g., "musik AI" → "syntetisk indhold").
    
    For annex points without extracted titles, we use the first line of the 
    chunk text as a title hint (common in EUR-Lex table formatting).
    
    Example output:
        "[Kontekst: Denne bestemmelse fastsætter krav om at udbydere af AI...]
        [Søgetermer: musik AI | generere billeder | AI-genereret indhold]
        
        Chapter V (AI-MODELLER TIL ALMEN BRUG) > Article 53 (Forpligtelser...)
        
        [Original chunk text here]"
    """
    result = doc
    
    # Step 1: Heading enrichment (structural context)
    heading = str(meta.get("heading_path_display", "")).strip()
    
    # For annex points without titles, extract first line as title hint
    # This handles EUR-Lex table formatting where title is in separate cell
    if meta.get("annex_point") and not meta.get("annex_point_title"):
        first_line = doc.split("\n")[0].strip() if doc else ""
        # Only use if it looks like a title (not too long, no colon mid-text)
        if first_line and len(first_line) < 100 and not first_line.startswith("a)"):
            # Append the inferred title to heading
            if heading:
                heading = f"{heading} ({first_line})"
            else:
                heading = first_line
    
    if heading:
        result = f"{heading}\n\n{doc}"
    
    # Step 2: Contextual description (semantic context - Anthropic best practice)
    contextual_description = meta.get("contextual_description", "")
    if contextual_description and isinstance(contextual_description, str):
        result = f"[Kontekst: {contextual_description}]\n\n{result}"
    
    # Step 3: Term enrichment - prefer pre-computed from JSONL, fallback to LLM
    enrichment_terms = meta.get("enrichment_terms")
    if enrichment_terms and isinstance(enrichment_terms, list):
        # Use pre-computed terms from JSONL (generated during eurlex_engine ingestion)
        terms_block = " | ".join(str(t) for t in enrichment_terms)
        result = f"[Søgetermer: {terms_block}]\n\n{result}"
    else:
        # Fallback: generate via LLM if enabled (backwards compatibility)
        try:
            from ..ingestion.embedding_enrichment import enrich_text_for_embedding, is_enrichment_enabled
            if is_enrichment_enabled():
                result = enrich_text_for_embedding(result, meta, corpus_id=corpus_id)
        except ImportError:
            pass  # Module not available, skip enrichment
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Embedding enrichment failed: %s", e)
    
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
        corpus_id = str(metadatas[0].get("corpus_id", "") or metadatas[0].get("law_id", "")).strip()
    
    # Enrich documents with heading and LLM-generated terms for better semantic matching
    enriched_documents = [
        _enrich_document_for_embedding(doc, meta, corpus_id=corpus_id)
        for doc, meta in zip(documents, metadatas)
    ]
    
    embeddings = engine._embed(enriched_documents)

    # Convert list-valued metadata to strings for ChromaDB compatibility
    # ChromaDB doesn't support list values, so we join enrichment_terms with " | "
    # This preserves the terms in metadata for visibility AND allows BM25/text search
    def _stringify_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for k, v in meta.items():
            if k in ("enrichment_terms", "_enrichment_terms") and isinstance(v, list):
                # Convert list to pipe-separated string
                result[k] = " | ".join(str(t) for t in v) if v else ""
            elif isinstance(v, list):
                # Other lists: convert to comma-separated (fallback)
                result[k] = ", ".join(str(t) for t in v) if v else ""
            else:
                result[k] = v
        return result

    cleaned_metadatas = [_stringify_metadata(meta) for meta in metadatas]
    
    # Store ORIGINAL documents (not enriched) to avoid duplication in LLM context
    try:
        engine.collection.upsert(ids=ids, documents=documents, metadatas=cleaned_metadatas, embeddings=embeddings)
    except InvalidArgumentError as exc:
        message = str(exc).lower()
        if "dimension" not in message:
            raise
        _reset_collection(engine, engine.collection_name, "collection")
        engine.collection.upsert(ids=ids, documents=documents, metadatas=cleaned_metadatas, embeddings=embeddings)


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
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    except InvalidArgumentError as exc:
        message = str(exc).lower()
        if "dimension" not in message:
            raise
        _reset_collection(engine, collection_name, collection_attr)
        collection = getattr(engine, collection_attr)
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


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
