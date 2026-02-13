"""Citation-graph-based retrieval expansion.

This module provides automatic discovery of related articles during retrieval,
replacing the manual bump_hints approach. It uses the pre-built citation graph
to expand retrieval based on:

1. Articles mentioned in the question ("jf. artikel 6")
2. Articles found in initial retrieval results
3. Co-citation patterns from the corpus

This is the professional approach to multi-concept retrieval in legal RAG,
as opposed to hardcoded bump_hints which are a workaround.

Configuration is in config/settings.yaml under citation_expansion:
  enabled: true
  max_expansion: 10        # Total articles to inject (Anthropic: top-20 recommended)
  seed_limit: 20            # Seed articles to consider for graph expansion  
  retrieved_boost_limit: 10 # Retrieved articles to boost for anchor retention
  min_weight: 0.15
  bump_bonus: 0.15

Usage:
    from src.engine.citation_expansion import get_citation_expansion_for_query
    
    # After initial retrieval, expand with related articles
    expanded_articles = get_citation_expansion_for_query(
        corpus_id="ai-act",
        question="...",
        retrieved_metadatas=[...],
    )
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Cache loaded graphs and config in memory
_graph_cache: dict[str, Any] = {}
_config_cache: dict[str, Any] | None = None


def clear_caches() -> None:
    """Clear all module-level caches. Call this to force reload of citation graph."""
    global _graph_cache, _config_cache
    _graph_cache.clear()
    _config_cache = None
    logger.info("Citation expansion caches cleared")


def _load_config() -> dict[str, Any]:
    """Load citation expansion config from settings.yaml."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "settings.yaml"
        with open(config_path) as f:
            settings = yaml.safe_load(f) or {}
        _config_cache = settings.get("citation_expansion", {})
    except Exception as e:
        logger.warning("Could not load citation_expansion config: %s", e)
        _config_cache = {}
    
    return _config_cache


def is_citation_expansion_enabled() -> bool:
    """Check if citation expansion is enabled in config."""
    config = _load_config()
    return bool(config.get("enabled", True))


def get_max_expansion() -> int:
    """Get max_expansion from config."""
    config = _load_config()
    return int(config.get("max_expansion", 5))


def get_min_weight() -> float:
    """Get min_weight from config."""
    config = _load_config()
    return float(config.get("min_weight", 0.15))


def get_bump_bonus() -> float:
    """Get bump_bonus from config."""
    config = _load_config()
    return float(config.get("bump_bonus", 0.15))


def get_seed_limit() -> int:
    """Get seed_limit from config - how many seed articles to consider for graph expansion."""
    config = _load_config()
    return int(config.get("seed_limit", 20))


def get_retrieved_boost_limit() -> int:
    """Get retrieved_boost_limit from config - how many retrieved articles to boost."""
    config = _load_config()
    return int(config.get("retrieved_boost_limit", 10))


def _get_citation_graph(corpus_id: str) -> Any | None:
    """Load citation graph with caching."""
    if corpus_id in _graph_cache:
        return _graph_cache[corpus_id]
    
    try:
        from ..ingestion.citation_graph import load_citation_graph
        graph = load_citation_graph(corpus_id)
        if graph is not None:
            _graph_cache[corpus_id] = graph
        return graph
    except ImportError:
        logger.warning("citation_graph module not available")
        return None


def expand_retrieval_with_citations(
    corpus_id: str,
    *,
    retrieved_articles: list[str] | None = None,
    mentioned_articles: list[str] | None = None,
    max_expansion: int = 5,
    min_weight: float = 0.15,
) -> list[str]:
    """Find additional articles to retrieve based on citation relationships.
    
    This replaces hardcoded bump_hints with automatic discovery.
    
    Args:
        corpus_id: The corpus to look up
        retrieved_articles: Articles already retrieved (from initial query)
        mentioned_articles: Articles explicitly mentioned in the question
        max_expansion: Maximum number of related articles to add
        min_weight: Minimum citation weight to consider
        
    Returns:
        List of additional article IDs to retrieve
    """
    graph = _get_citation_graph(corpus_id)
    if graph is None:
        return []
    
    # Combine seed articles (mentioned takes priority)
    seed_articles: list[str] = []
    if mentioned_articles:
        seed_articles.extend(str(a).upper() for a in mentioned_articles)
    if retrieved_articles:
        seed_articles.extend(str(a).upper() for a in retrieved_articles if str(a).upper() not in seed_articles)

    if not seed_articles:
        return []

    # Find related articles from graph
    # Use config-based seed limit (default: 20) per Anthropic best practice
    seed_limit = get_seed_limit()

    # IMPORTANT: Only exclude articles in the TOP seed_limit seeds from expansion.
    # Articles that ARE in retrieved but BEYOND seed_limit should still be eligible
    # for graph expansion (e.g., ANNEX:III at position 27 when seed_limit=20).
    # This ensures high-score graph links to "retrieved but distant" articles are included.
    top_seeds_set = set(seed_articles[:seed_limit])

    related_scores: dict[str, float] = {}
    for article_id in seed_articles[:seed_limit]:
        related = graph.get_related_articles(
            article_id,
            max_depth=2,  # Depth 2 to reach punkt-level nodes (e.g., Article 6 → ANNEX:III → ANNEX:III:5)
            min_weight=min_weight,
        )
        for rel_id, score in related:
            # Only exclude from top seeds, not ALL retrieved articles
            if rel_id not in top_seeds_set:
                # Accumulate scores across multiple seeds
                related_scores[rel_id] = max(related_scores.get(rel_id, 0), score)

    # BOOST punkt-level nodes whose parent is in seeds OR top results
    # This ensures children of high-score parents (e.g., ANNEX:III:5 when ANNEX:III is retrieved)
    # get enough score to make it into the final expansion list.
    # Without this, punkt-level nodes get filtered out due to low absolute scores.
    # NOTE: 0.2 boost ensures punkt-nodes score ~0.22 which beats most article scores (~0.14)
    parent_boost = 0.2  # Boost children of top-scoring parents or seeds
    top_parents = {art_id for art_id, score in related_scores.items() if score >= 0.1}
    # Also consider seeds as valid parents (ANNEX:III in seeds means ANNEX:III:5 should be boosted)
    seed_annexes = {s for s in top_seeds_set if s.startswith("ANNEX:")}
    for rel_id in list(related_scores.keys()):
        # Check if this is a punkt-level node (format: ANNEX:III:5 or ANNEX:VIII:A:5)
        if rel_id.startswith("ANNEX:") and rel_id.count(":") >= 2:
            # Extract parent (e.g., ANNEX:III:5 -> ANNEX:III or ANNEX:VIII:A:5 -> ANNEX:VIII:A)
            parts = rel_id.split(":")
            if len(parts) >= 3:
                # Try parent as ANNEX:X (for ANNEX:III:5 -> ANNEX:III)
                parent_annex = f"{parts[0]}:{parts[1]}"
                # Boost if parent is in top_parents, related_scores, OR seeds
                if parent_annex in top_parents or parent_annex in related_scores or parent_annex in seed_annexes:
                    # Boost this punkt-level node
                    related_scores[rel_id] = related_scores[rel_id] + parent_boost

    # Sort by score and take top N
    sorted_related = sorted(related_scores.items(), key=lambda x: (-x[1], x[0]))
    expansion = [art_id for art_id, score in sorted_related[:max_expansion]]
    
    if expansion:
        logger.debug(
            "Citation expansion for %s: seeds=%s → expansion=%s",
            corpus_id,
            seed_articles[:3],
            expansion,
        )
    
    return expansion


def get_scope_articles_from_graph(
    corpus_id: str,
    *,
    scope_indicators: list[str] | None = None,
) -> list[str]:
    """Get scope/definition articles for a corpus based on citation patterns.
    
    Scope articles (like AI Act art. 2, 3) are typically:
    - Heavily cited by other articles
    - In the first chapter (low article numbers)
    - Referenced when discussing applicability
    
    This auto-discovers scope articles without configuration.
    """
    graph = _get_citation_graph(corpus_id)
    if graph is None:
        return []
    
    # Find scope candidates using heuristic: low article numbers + citation counts
    # We look at ALL articles, not just most-cited, because scope articles
    # (e.g., art. 2, 3) may have moderate citation counts but low numbers
    scope_candidates: list[tuple[str, float]] = []
    for article_id, node in graph.nodes.items():
        if node.node_type != "article":
            continue
        try:
            # Parse article number
            art_num = int("".join(c for c in article_id if c.isdigit()) or "999")
            # Only consider articles 1-20 (scope/general provisions typically)
            if art_num > 20:
                continue
            # Lower numbers + higher counts = more likely scope article
            mention_bonus = node.mention_count * 0.5
            # Strong preference for articles 1-5 (scope, definitions, etc.)
            scope_score = (21 - art_num) + mention_bonus
            scope_candidates.append((article_id, scope_score))
        except ValueError:
            continue
    
    scope_candidates.sort(key=lambda x: (-x[1], x[0]))
    return [art_id for art_id, score in scope_candidates[:5]]


def get_classification_articles_from_graph(corpus_id: str) -> list[str]:
    """Get classification articles for a corpus from citation graph roles.
    
    Returns articles tagged with 'classification' role (e.g., AI Act art. 6, Annex III).
    These define how to classify entities/systems under the regulation.
    """
    graph = _get_citation_graph(corpus_id)
    if graph is None:
        return []
    
    return graph.get_articles_by_role("classification")


def extract_mentioned_articles(question: str) -> list[str]:
    """Extract article and annex references from question text.
    
    Parses patterns like:
    - "artikel 5"
    - "article 10"
    - "art. 6"
    - "jf. artikel 3"
    - "bilag III"
    - "annex III"
    """
    import re
    
    result: list[str] = []
    
    # Article patterns
    article_pattern = re.compile(
        r"(?:artikel|article|art\.?)\s*(\d{1,3}[a-z]?)",
        re.IGNORECASE
    )
    
    article_matches = article_pattern.findall(question)
    result.extend(m.upper() for m in article_matches)
    
    # Annex patterns (bilag I, II, III, IV, etc. or annex 1, 2, 3)
    annex_pattern = re.compile(
        r"(?:bilag|annex)\s*([IVX]+|\d+)",
        re.IGNORECASE
    )
    
    annex_matches = annex_pattern.findall(question)
    for m in annex_matches:
        # Normalize: keep roman numerals uppercase
        annex_id = f"ANNEX:{m.upper()}"
        result.append(annex_id)
    
    return result


def should_include_scope_articles(question: str) -> bool:
    """Determine if question is asking about scope/applicability.

    Keywords are defined in src/engine/constants.py (_INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR).
    Uses the same list as intent detection - single source of truth.
    """
    from src.engine.constants import _INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR

    question_lower = question.lower()
    return any(kw in question_lower for kw in _INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR)


def get_citation_expansion_for_query(
    corpus_id: str,
    question: str,
    retrieved_metadatas: list[dict[str, Any]],
    *,
    max_expansion: int | None = None,
) -> list[str]:
    """Main entry point: get articles to inject based on citations.
    
    This is the replacement for bump_hints. Call this after initial
    retrieval to get additional articles that should be included.
    
    IMPORTANT: Returns articles for BOTH:
    - Anchor boosting (ensures high-value chunks stay in context)
    - Chunk injection (fetches new chunks for missing anchors)
    
    The return value is used by PROD to:
    1. Boost distance scores for chunks matching these anchors
    2. Inject additional chunks for anchors not in initial retrieval
    
    Therefore we include:
    - Graph expansion (articles cited by retrieved) - gets new relevant chunks
    - Top retrieved articles - ensures they get anchor boost to stay in context
    - Scope/classification (for applicability questions) - foundational context
    
    Args:
        corpus_id: The corpus
        question: The user's question
        retrieved_metadatas: Metadata from initial retrieval
        max_expansion: Max articles to add
        
    Returns:
        List of article IDs for anchor boosting and injection
    """
    # 0. Use config-based max_expansion if not explicitly provided
    if max_expansion is None:
        max_expansion = get_max_expansion()
    
    # 1. Extract articles mentioned in question
    mentioned = extract_mentioned_articles(question)
    
    # 2. Extract articles from initial retrieval
    retrieved: list[str] = []
    for meta in retrieved_metadatas:
        art = meta.get("article")
        if art:
            retrieved.append(str(art).upper())
        annex = meta.get("annex")
        if annex:
            retrieved.append(f"ANNEX:{str(annex).upper()}")
    
    retrieved_set = set(retrieved)
    
    # 3. Collect ALL candidate sources
    already_added: set[str] = set()
    result: list[str] = []
    
    def add_unique(articles: list[str], limit: int | None = None) -> int:
        """Add articles to result, avoiding duplicates. Returns count added."""
        added = 0
        for art in articles:
            art_upper = str(art).upper()
            if art_upper not in already_added:
                result.append(art_upper)
                already_added.add(art_upper)
                added += 1
                if limit and added >= limit:
                    break
        return added
    
    # 4. PRIORITY 1: Mentioned articles (explicit in question) - always include
    add_unique(mentioned)
    
    # 5. PRIORITY 2: Scope articles (for applicability questions)
    # When question is about scope/applicability, we MUST include scope articles
    # (typically art. 2, 3) because they define WHAT the law covers.
    # These are injected BEFORE graph expansion to guarantee they get slots.
    if should_include_scope_articles(question):
        scope_articles = get_scope_articles_from_graph(corpus_id)
        # Only add scope articles not already in retrieved (those come via step 6)
        scope_filtered = [a for a in scope_articles if a not in retrieved_set]
        add_unique(scope_filtered[:2])  # Guarantee 2 scope articles

    
    # 6. PRIORITY 3: Top retrieved articles (CRITICAL for anchor boost)
    # These are what semantic search found most relevant - we MUST include them
    # so they get anchor boost and don't get displaced by injected chunks.
    # Without this, high-value chunks like Annex III can be pushed out.
    # Use config-based limit (default: 15) per Anthropic "top-20" best practice
    # NOTE: We iterate through retrieved_boost_limit chunks but extract ALL unique
    # articles from them - this ensures we don't miss relevant articles just because
    # some articles have multiple chunks in top results.
    retrieved_boost_limit = get_retrieved_boost_limit()
    top_retrieved = [str(a).upper() for a in retrieved[:retrieved_boost_limit]]
    add_unique(top_retrieved)  # No artificial limit - add all unique from top chunks
    
    # 7. PRIORITY 4: Graph expansion (articles cited by retrieved)
    # This finds related articles via citation graph. Since we already added
    # retrieved articles, this will only add NEW ones (related but not retrieved).
    # IMPORTANT: Only pass top_retrieved (not all retrieved) to graph expansion.
    # This ensures articles that ARE in full retrieval but outside top-N chunks
    # (e.g., ANNEX:III at position 27) can still be discovered via graph links.
    # Without this, ANNEX:III would be in seed_articles and thus excluded from
    # graph expansion results, even though it has high citation score from Art. 6.
    graph_expansion = expand_retrieval_with_citations(
        corpus_id,
        retrieved_articles=top_retrieved,  # Only top chunks, not all retrieved
        mentioned_articles=mentioned,
        max_expansion=max_expansion,
        min_weight=get_min_weight(),  # Use config value (lowered to include punkt-level nodes)
    )
    add_unique(graph_expansion, limit=max_expansion - len(result))
    
    return result[:max_expansion]
