"""Embedding enrichment via LLM-generated term variants.

This module generates colloquial search terms for legal chunks during ingestion.
The goal is to bridge the semantic gap between user queries (everyday language)
and legal text (formal terminology).

Example:
    Legal text: "syntetisk indhold" (synthetic content)
    Generated terms: ["musik AI", "generere billeder", "AI-genereret indhold"]

    This enables the query "musik vha ai" to retrieve Article 50 chunks.

The enrichment is prepended to the embedding text (not stored document) to:
1. Improve retrieval recall without polluting LLM context
2. Be automatic/programmatic (no manual keyword lists per law)
3. Scale to new laws without intervention

Configuration in config/settings.yaml:
    embedding_enrichment:
      enabled: true
      model: "gpt-4o-mini"
      max_terms: 5
      batch_size: 10
      cache_enabled: true

Note: Uses centralized prompts and types from ingestion_generation.py.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

# Import shared types and prompts from centralized module
from src.ingestion.ingestion_generation import (
    EnrichmentResult,
    ENRICHMENT_PROMPT,
    VALID_ROLES,
)

logger = logging.getLogger(__name__)

# Cache directory for enrichment results
_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "enrichment_cache"
_config_cache: dict[str, Any] | None = None


def _load_config() -> dict[str, Any]:
    """Load embedding enrichment config from settings.yaml."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        with open(config_path) as f:
            settings = yaml.safe_load(f) or {}
        _config_cache = settings.get("embedding_enrichment", {})
    except Exception as e:
        logger.warning("Could not load embedding_enrichment config: %s", e)
        _config_cache = {}
    
    return _config_cache


def is_enrichment_enabled() -> bool:
    """Check if embedding enrichment is enabled in config.

    Can be forced on via _FORCE_ENRICHMENT_ENABLED env var (used by eval runner).
    """
    # Allow forcing on for eval/testing
    if os.environ.get("_FORCE_ENRICHMENT_ENABLED") == "1":
        return True
    config = _load_config()
    return bool(config.get("enabled", False))  # Off by default until tested


def get_enrichment_model() -> str:
    """Get model for enrichment generation."""
    config = _load_config()
    return str(config.get("model", "gpt-4o-mini"))


def get_max_terms() -> int:
    """Get max terms to generate per chunk."""
    config = _load_config()
    return int(config.get("max_terms", 5))


def is_cache_enabled() -> bool:
    """Check if enrichment caching is enabled."""
    config = _load_config()
    return bool(config.get("cache_enabled", True))


def get_max_concurrent() -> int:
    """Get max concurrent LLM calls for parallel processing."""
    config = _load_config()
    return int(config.get("max_concurrent", 10))


def get_batch_size() -> int:
    """Get batch size for progress logging during concurrent processing."""
    config = _load_config()
    return int(config.get("batch_size", 50))


def _get_cache_key(chunk_text: str, article_title: str, corpus_id: str) -> str:
    """Generate cache key for a chunk."""
    # Version prefix ensures cache invalidation when prompt changes
    # v3: Added role classification to prompt
    content = f"v3:{corpus_id}:{article_title}:{chunk_text[:500]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _get_cached_enrichment(cache_key: str) -> EnrichmentResult | None:
    """Get cached enrichment result if available."""
    if not is_cache_enabled():
        return None

    cache_file = _CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                data = json.load(f)
                # Support v3 format (with roles), v2 format (description + terms), skip older
                if "roles" in data:
                    # v3 format - full result with roles
                    return EnrichmentResult(
                        contextual_description=data.get("contextual_description", ""),
                        search_terms=data.get("terms", []),
                        roles=data.get("roles", []),
                    )
                # v2 or older format - skip, will regenerate with new prompt
                return None
        except Exception:
            return None
    return None


def _cache_enrichment(cache_key: str, result: EnrichmentResult) -> None:
    """Cache enrichment result for a chunk."""
    if not is_cache_enabled():
        return

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _CACHE_DIR / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump({
                "contextual_description": result.contextual_description,
                "terms": result.search_terms,
                "roles": result.roles,
            }, f)
    except Exception as e:
        logger.debug("Could not cache enrichment result: %s", e)


# Re-export for backward compatibility
__all__ = ["EnrichmentResult", "VALID_ROLES", "generate_enrichment", "generate_enrichment_terms"]


def generate_enrichment_terms(
    chunk_text: str,
    *,
    article_title: str = "",
    corpus_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> list[str]:
    """Generate colloquial search terms for a legal chunk using LLM.
    
    This is a backward-compatible wrapper that returns only search terms.
    Use generate_enrichment() for the full EnrichmentResult with contextual description.
    
    Args:
        chunk_text: The legal text to generate terms for
        article_title: Title of the article (e.g., "Artikel 50 - Gennemsigtighed")
        corpus_id: Corpus identifier (e.g., "ai-act")
        metadata: Optional chunk metadata for additional context
        
    Returns:
        List of colloquial search terms (empty if disabled or error)
    """
    result = generate_enrichment(
        chunk_text,
        article_title=article_title,
        corpus_id=corpus_id,
        metadata=metadata,
    )
    return result.search_terms if result else []


def generate_enrichment(
    chunk_text: str,
    *,
    article_title: str = "",
    corpus_id: str = "",
    metadata: dict[str, Any] | None = None,
) -> EnrichmentResult | None:
    """Generate contextual description and search terms for a legal chunk using LLM.
    
    This is the core function that bridges the semantic gap between
    formal legal text and everyday user queries. It generates:
    1. Contextual description (50-80 words) explaining WHAT the text is about
    2. Search terms (3-5 terms) in everyday language
    
    Args:
        chunk_text: The legal text to generate enrichment for
        article_title: Title of the article (e.g., "Artikel 50 - Gennemsigtighed")
        corpus_id: Corpus identifier (e.g., "ai-act")
        metadata: Optional chunk metadata for additional context
        
    Returns:
        EnrichmentResult with contextual_description and search_terms, or None if disabled/error
    """
    if not is_enrichment_enabled():
        return None
    
    # Skip empty chunks only
    # Per Anthropic best practice: ALL chunks should have contextual enrichment
    if not chunk_text.strip():
        return None
    
    # Check cache first
    cache_key = _get_cache_key(chunk_text, article_title, corpus_id)
    cached = _get_cached_enrichment(cache_key)
    if cached is not None:
        logger.debug("Enrichment cache hit for %s", cache_key)
        return cached
    
    # Generate via LLM
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OPENAI_API_KEY - skipping enrichment")
            return None
        
        client = OpenAI(api_key=api_key)
        model = get_enrichment_model()
        max_terms = get_max_terms()
        
        # Build article title from metadata if not provided
        if not article_title and metadata:
            art = metadata.get("article")
            title = metadata.get("article_title", "")
            if art:
                article_title = f"Artikel {art}"
                if title:
                    article_title = f"{article_title} - {title}"
        
        prompt = ENRICHMENT_PROMPT.format(
            article_title=article_title or "Ukendt artikel",
            chunk_text=chunk_text[:2000],  # Increased for better context
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Some creativity, but consistent
            max_tokens=300,  # Increased for description + terms
        )
        
        content = response.choices[0].message.content or ""

        # Parse structured output - handle both multi-line and single-line formats
        # LLM may return:
        #   Format A: "KONTEKST: ...\nSØGETERMER: ...\nROLLER: ..."
        #   Format B: "KONTEKST: ... SØGETERMER: ... ROLLER: ..."  (all on one line)
        contextual_description = ""
        search_terms: list[str] = []
        roles: list[str] = []

        # First, try to find SØGETERMER anywhere in content
        terms_match = re.search(r'SØGETERMER:\s*(.+?)(?:\n|ROLLER:|$)', content, re.IGNORECASE)
        if terms_match:
            terms_str = terms_match.group(1).strip()
            search_terms = [
                t.strip() for t in terms_str.split("|")
                if t.strip() and len(t.strip()) > 2
            ][:max_terms]

        # Extract KONTEKST (everything between "KONTEKST:" and "SØGETERMER:" or end)
        kontekst_match = re.search(r'KONTEKST:\s*(.+?)(?:\s*SØGETERMER:|$)', content, re.IGNORECASE | re.DOTALL)
        if kontekst_match:
            contextual_description = kontekst_match.group(1).strip()
            # Clean up any trailing newlines
            contextual_description = ' '.join(contextual_description.split())

        # Extract ROLLER (new in v3)
        roles_match = re.search(r'ROLLER:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        if roles_match:
            roles_str = roles_match.group(1).strip().lower()
            # Handle "ingen" (none) case
            if roles_str not in ("ingen", "none", ""):
                raw_roles = [r.strip() for r in roles_str.split("|") if r.strip()]
                # Validate roles against allowed list
                roles = [r for r in raw_roles if r in VALID_ROLES]

        # Create result
        result = EnrichmentResult(
            contextual_description=contextual_description,
            search_terms=search_terms,
            roles=roles,
        )

        # Cache the results
        _cache_enrichment(cache_key, result)

        logger.debug(
            "Generated enrichment for %s: desc=%d chars, terms=%s, roles=%s",
            article_title,
            len(contextual_description),
            search_terms[:3],
            roles,
        )

        return result
        
    except Exception as e:
        logger.warning("Enrichment generation failed: %s", e)
        return None


def enrich_text_for_embedding(
    doc: str,
    metadata: dict[str, Any],
    *,
    corpus_id: str = "",
) -> str:
    """Enrich document text with structural context and search terms for embedding.
    
    This is the main entry point called during indexing. It prepends:
    1. Chapter context (if available)
    2. Article/Annex title (if available)
    3. LLM-generated colloquial search terms
    
    This follows Anthropic's Contextual Retrieval best practice (Sep 2024)
    which showed 49% reduction in retrieval failure when prepending context.
    
    Args:
        doc: Original document text
        metadata: Chunk metadata
        corpus_id: Corpus identifier
        
    Returns:
        Enriched text for embedding (original doc if enrichment disabled/failed)
    """
    if not is_enrichment_enabled():
        return doc
    
    # Build structural context for embedding (Anthropic best practice)
    context_parts: list[str] = []
    
    # Chapter context
    chapter = metadata.get("chapter")
    chapter_title = metadata.get("chapter_title", "")
    if chapter:
        if chapter_title:
            context_parts.append(f"[Kapitel {chapter}: {chapter_title}]")
        else:
            context_parts.append(f"[Kapitel {chapter}]")
    
    # Article/Annex title context
    art = metadata.get("article")
    art_title = metadata.get("article_title", "")
    annex = metadata.get("annex")
    annex_title = metadata.get("annex_title", "")
    recital = metadata.get("recital")
    
    if art:
        if art_title:
            context_parts.append(f"[Artikel {art}: {art_title}]")
        else:
            context_parts.append(f"[Artikel {art}]")
    elif annex:
        if annex_title:
            context_parts.append(f"[Bilag {annex}: {annex_title}]")
        else:
            context_parts.append(f"[Bilag {annex}]")
    elif recital:
        context_parts.append(f"[Betragtning {recital}]")
    
    # Build article title for LLM term generation
    article_title = ""
    if art:
        article_title = f"Artikel {art}"
        if art_title:
            article_title = f"{article_title} - {art_title}"
    
    # Generate enrichment terms
    terms = generate_enrichment_terms(
        doc,
        article_title=article_title,
        corpus_id=corpus_id or metadata.get("corpus_id", ""),
        metadata=metadata,
    )
    
    # Add search terms to context
    if terms:
        terms_block = " | ".join(terms)
        context_parts.append(f"[Søgetermer: {terms_block}]")
        # Store terms in metadata for debugging
        metadata["_enrichment_terms"] = terms
    
    # If no context at all, return original doc
    if not context_parts:
        return doc
    
    # Prepend all context to document for embedding
    context_header = "\n".join(context_parts)
    enriched = f"{context_header}\n\n{doc}"
    
    return enriched


def batch_generate_enrichment_terms(
    chunks: list[tuple[str, dict[str, Any]]],
    *,
    corpus_id: str = "",
) -> dict[str, list[str]]:
    """Batch generate enrichment terms for multiple chunks.
    
    More efficient than calling generate_enrichment_terms individually
    when processing many chunks during ingestion.
    
    Args:
        chunks: List of (chunk_text, metadata) tuples
        corpus_id: Corpus identifier
        
    Returns:
        Dict mapping chunk_id to generated terms
    """
    if not is_enrichment_enabled():
        return {}
    
    results: dict[str, list[str]] = {}
    
    for chunk_text, metadata in chunks:
        chunk_id = metadata.get("chunk_id", "")
        if not chunk_id:
            continue
            
        # Build article title
        article_title = ""
        art = metadata.get("article")
        title = metadata.get("article_title", "")
        if art:
            article_title = f"Artikel {art}"
            if title:
                article_title = f"{article_title} - {title}"
        
        terms = generate_enrichment_terms(
            chunk_text,
            article_title=article_title,
            corpus_id=corpus_id,
            metadata=metadata,
        )
        
        if terms:
            results[chunk_id] = terms
    
    return results


def clear_enrichment_cache() -> int:
    """Clear all cached enrichment terms.
    
    Returns:
        Number of cache files deleted
    """
    if not _CACHE_DIR.exists():
        return 0
    
    count = 0
    for cache_file in _CACHE_DIR.glob("*.json"):
        try:
            cache_file.unlink()
            count += 1
        except Exception:
            pass
    
    return count
