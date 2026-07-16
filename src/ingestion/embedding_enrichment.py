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
    CASE_LAW_ENRICHMENT_PROMPT,
    VALID_ROLES,
)

logger = logging.getLogger(__name__)


def _build_case_law_context_headers(metadata: dict[str, Any]) -> list[str]:
    """Build context header lines for case law chunks.

    Headers follow the same pattern as legislation headers but use
    case law identity fields (case name, ECLI, court, section type).

    Args:
        metadata: Case law chunk metadata dict.

    Returns:
        List of header strings like ["[Schrems II (C-311/18)]", "[CJEU, 2020-07-16]", ...].
    """
    headers: list[str] = []

    # Case identity
    case_name = metadata.get("case_name", "")
    case_number = metadata.get("case_number", "")
    if case_name:
        headers.append(f"[{case_name} ({case_number})]")
    else:
        headers.append(f"[({case_number})]")

    # Court and date
    court = metadata.get("court", "")
    decision_date = metadata.get("decision_date", "")
    headers.append(f"[{court}, {decision_date}]")

    # Section context — underscore to space, title case
    section_type = metadata.get("section_type", "")
    if section_type:
        label = section_type.replace("_", " ").title()
        headers.append(f"[{label}]")

    # Paragraph context — only when non-empty
    paragraph_range = metadata.get("paragraph_range", "")
    if paragraph_range:
        headers.append(f"[Paragraphs {paragraph_range}]")

    return headers


def _build_legislation_context_headers(metadata: dict[str, Any]) -> list[str]:
    """Build context header lines for legislation chunks.

    Extracted from enrich_text_for_embedding() for symmetry with
    _build_case_law_context_headers and testability.

    Args:
        metadata: Legislation chunk metadata dict.

    Returns:
        List of header strings like ["[Kapitel 2: Forbud]", "[Artikel 5: ...]"].
    """
    headers: list[str] = []

    # Chapter context
    chapter = metadata.get("chapter")
    chapter_title = metadata.get("chapter_title", "")
    if chapter:
        if chapter_title:
            headers.append(f"[Kapitel {chapter}: {chapter_title}]")
        else:
            headers.append(f"[Kapitel {chapter}]")

    # Article/Annex title context
    art = metadata.get("article")
    art_title = metadata.get("article_title", "")
    annex = metadata.get("annex")
    annex_title = metadata.get("annex_title", "")
    recital = metadata.get("recital")

    if art:
        if art_title:
            headers.append(f"[Artikel {art}: {art_title}]")
        else:
            headers.append(f"[Artikel {art}]")
    elif annex:
        if annex_title:
            headers.append(f"[Bilag {annex}: {annex_title}]")
        else:
            headers.append(f"[Bilag {annex}]")
    elif recital:
        headers.append(f"[Betragtning {recital}]")

    return headers


# Cache directory for enrichment results
_CACHE_DIR = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "enrichment_cache"
)
_config_cache: dict[str, Any] | None = None
_full_config_cache: dict[str, Any] | None = None

# Default case law enrichment roles (REQ-4)
_DEFAULT_CASE_LAW_ENRICHMENT_ROLES = frozenset(
    ["interpretation", "application", "principle", "procedural", "remedy"]
)


def _load_full_config() -> dict[str, Any]:
    """Load the full settings.yaml dict, cached after first read."""
    global _full_config_cache
    if _full_config_cache is not None:
        return _full_config_cache

    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        with open(config_path) as f:
            _full_config_cache = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Could not load settings.yaml: %s", e)
        _full_config_cache = {}

    return _full_config_cache


def _load_config() -> dict[str, Any]:
    """Load embedding enrichment config from settings.yaml."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    full = _load_full_config()
    _config_cache = full.get("embedding_enrichment", {})

    return _config_cache


def _load_case_law_enrichment_roles() -> frozenset[str]:
    """Load case law enrichment roles from config.

    Reads ``case_law.enrichment.enrichment_roles`` from settings.yaml.
    Falls back to default roles if config key is missing (EC-12).

    Returns:
        Frozenset of allowed role strings for O(1) validation.
    """
    full = _load_full_config()
    case_law = full.get("case_law", {})
    enrichment = case_law.get("enrichment", {})
    roles_list = enrichment.get("enrichment_roles")
    if roles_list is None:
        return _DEFAULT_CASE_LAW_ENRICHMENT_ROLES
    return frozenset(roles_list)


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


def _get_case_law_cache_key(
    chunk_text: str,
    *,
    ecli: str,
    section_type: str,
    corpus_id: str,
) -> str:
    """Generate cache key for a case law chunk.

    Uses v4 prefix to isolate from legislation cache entries (v3).
    """
    content = f"v4:{corpus_id}:{ecli}:{section_type}:{chunk_text[:500]}"
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
            json.dump(
                {
                    "contextual_description": result.contextual_description,
                    "terms": result.search_terms,
                    "roles": result.roles,
                },
                f,
            )
    except Exception as e:
        logger.debug("Could not cache enrichment result: %s", e)


# Re-export for backward compatibility
__all__ = [
    "EnrichmentResult",
    "VALID_ROLES",
    "generate_enrichment",
    "generate_enrichment_terms",
]


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
    prompt: str | None = None,
    valid_roles: frozenset[str] | None = None,
    cache_key: str | None = None,
) -> EnrichmentResult | None:
    """Generate contextual description and search terms for a chunk using LLM.

    Source-agnostic executor: when ``prompt``, ``valid_roles``, and ``cache_key``
    are provided the function uses them directly. When omitted it falls back to
    legislation defaults (``ENRICHMENT_PROMPT``, ``VALID_ROLES``, v3 cache key).

    Args:
        chunk_text: The text to generate enrichment for.
        article_title: Title string for prompt formatting.
        corpus_id: Corpus identifier.
        metadata: Optional chunk metadata.
        prompt: Override prompt template (must accept ``{article_title}`` and ``{chunk_text}``).
        valid_roles: Override role validation set.
        cache_key: Override cache key (skips default v3 key generation).

    Returns:
        EnrichmentResult or None if disabled/error.
    """
    if not is_enrichment_enabled():
        return None

    if not chunk_text.strip():
        return None

    # Resolve defaults for legislation path
    effective_roles = valid_roles if valid_roles is not None else VALID_ROLES

    # Cache key
    if cache_key is None:
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

        # Build article title from metadata if not provided (legislation default)
        if not article_title and metadata and prompt is None:
            art = metadata.get("article")
            title = metadata.get("article_title", "")
            if art:
                article_title = f"Artikel {art}"
                if title:
                    article_title = f"{article_title} - {title}"

        # Resolve prompt template
        effective_prompt = prompt if prompt is not None else ENRICHMENT_PROMPT

        # Format prompt — case law prompt uses {court} and {section_type} too
        fmt_kwargs: dict[str, str] = {
            "article_title": article_title or "Ukendt artikel",
            "chunk_text": chunk_text[:2000],
        }
        if metadata and prompt is not None:
            fmt_kwargs["court"] = metadata.get("court", "")
            fmt_kwargs["section_type"] = metadata.get("section_type", "")

        formatted_prompt = effective_prompt.format(**fmt_kwargs)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.3,
            max_tokens=300,
        )

        content = response.choices[0].message.content or ""

        # Empty response is a failure (EC-7)
        if not content.strip():
            return None

        # Parse structured output
        contextual_description = ""
        search_terms: list[str] = []
        roles: list[str] = []

        terms_match = re.search(
            r"SØGETERMER:\s*(.+)(?:\n|ROLLER:|$)", content, re.IGNORECASE
        )
        if terms_match:
            terms_str = terms_match.group(1).strip()
            search_terms = [
                t.strip()
                for t in terms_str.split("|")
                if t.strip() and len(t.strip()) > 2
            ][:max_terms]

        kontekst_match = re.search(
            r"KONTEKST:\s*(.+?)SØGETERMER:", content, re.IGNORECASE | re.DOTALL
        )
        if not kontekst_match:
            kontekst_match = re.search(
                r"KONTEKST:\s*(.+)", content, re.IGNORECASE | re.DOTALL
            )
        if kontekst_match:
            contextual_description = kontekst_match.group(1).strip()
            contextual_description = " ".join(contextual_description.split())

        roles_match = re.search(r"ROLLER:\s*(.+)(?:\n|$)", content, re.IGNORECASE)
        if roles_match:
            roles_str = roles_match.group(1).strip().lower()
            if roles_str not in ("ingen", "none", ""):
                raw_roles = [r.strip() for r in roles_str.split("|") if r.strip()]
                roles = [r for r in raw_roles if r in effective_roles]

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

    effective_corpus_id = corpus_id or metadata.get("corpus_id", "")

    # Dispatch based on source_type (REQ-1)
    if metadata.get("source_type") == "cjeu_case_law":
        return _enrich_case_law_chunk(doc, metadata, corpus_id=effective_corpus_id)

    # Legislation path (default — unchanged behavior)
    return _enrich_legislation_chunk(doc, metadata, corpus_id=effective_corpus_id)


def _enrich_case_law_chunk(
    doc: str,
    metadata: dict[str, Any],
    *,
    corpus_id: str,
) -> str:
    """Case law enrichment path: context headers, LLM call, search terms.

    If enrichment fails, returns original doc (REQ-6).
    """
    context_parts = _build_case_law_context_headers(metadata)

    # Build article_title from case name + case number (REQ-10, AS-9)
    case_name = metadata.get("case_name", "")
    case_number = metadata.get("case_number", "")
    if case_name:
        article_title = f"{case_name} ({case_number})"
    else:
        article_title = f"({case_number})"

    # Build cache key with v4 prefix (REQ-7)
    cache_key = _get_case_law_cache_key(
        doc,
        ecli=metadata.get("ecli", ""),
        section_type=metadata.get("section_type", ""),
        corpus_id=corpus_id,
    )

    # Load case law roles from config (REQ-4)
    valid_roles = _load_case_law_enrichment_roles()

    # Call generate_enrichment with case-law-specific parameters
    result = generate_enrichment(
        doc,
        article_title=article_title,
        corpus_id=corpus_id,
        metadata=metadata,
        prompt=CASE_LAW_ENRICHMENT_PROMPT,
        valid_roles=valid_roles,
        cache_key=cache_key,
    )

    if result is None:
        # Graceful failure — log and return original doc (REQ-6)
        logger.warning(
            "Case law enrichment failed for ecli=%s section=%s",
            metadata.get("ecli", "?"),
            metadata.get("section_type", "?"),
        )
        return doc

    # Append search terms
    if result.search_terms:
        terms_block = " | ".join(result.search_terms)
        context_parts.append(f"[Søgetermer: {terms_block}]")
        metadata["_enrichment_terms"] = result.search_terms

    if not context_parts:
        return doc

    context_header = "\n".join(context_parts)
    return f"{context_header}\n\n{doc}"


def _enrich_legislation_chunk(
    doc: str,
    metadata: dict[str, Any],
    *,
    corpus_id: str,
) -> str:
    """Legislation enrichment path (original behavior, extracted for clarity)."""
    context_parts = _build_legislation_context_headers(metadata)

    # Build article title for LLM term generation
    article_title = ""
    art = metadata.get("article")
    art_title = metadata.get("article_title", "")
    if art:
        article_title = f"Artikel {art}"
        if art_title:
            article_title = f"{article_title} - {art_title}"

    # Generate enrichment terms
    terms = generate_enrichment_terms(
        doc,
        article_title=article_title,
        corpus_id=corpus_id,
        metadata=metadata,
    )

    # Add search terms to context
    if terms:
        terms_block = " | ".join(terms)
        context_parts.append(f"[Søgetermer: {terms_block}]")
        metadata["_enrichment_terms"] = terms

    if not context_parts:
        return doc

    context_header = "\n".join(context_parts)
    return f"{context_header}\n\n{doc}"


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
