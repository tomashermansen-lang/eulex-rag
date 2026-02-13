"""Build citation graph from corpus chunks.

Extracts cross-reference relationships from the 'mentions' field in chunk metadata
to build a graph of article relationships. This enables automatic discovery of
related articles without hardcoded bump_hints.

Works with any legal corpus (AI Act, GDPR, DORA, etc.) that has been ingested
with the standard chunking pipeline, which extracts 'mentions' metadata from
cross-references like "jf. artikel 5", "se artikel 10, stk. 2", etc.

Usage:
    from src.ingestion.citation_graph import CitationGraph
    
    # Build graph for any corpus
    graph = CitationGraph.from_corpus("ai-act")  # or "gdpr", "dora", etc.
    
    # Find related articles
    related = graph.get_related_articles("6", max_depth=1)
    
    # Build graphs for all corpora
    for corpus_id in ["ai-act", "gdpr", "dora"]:
        graph = CitationGraph.from_corpus(corpus_id)
        graph.save()
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "citation_graph:v1"
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed"


@dataclass
class CitationEdge:
    """An edge in the citation graph."""

    source: str  # Source article
    target: str  # Target article
    edge_type: str  # "cites", "co-occurrence", "same_chapter"
    count: int = 1  # Number of times this relationship occurs
    weight: float = 1.0  # Normalized weight (computed from count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type,
            "count": self.count,
            "weight": round(self.weight, 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CitationEdge:
        return cls(
            source=d["source"],
            target=d["target"],
            edge_type=d["type"],
            count=d.get("count", 1),
            weight=d.get("weight", 1.0),
        )


# Article role detection uses existing keyword lists from constants.py.
# This is the single source of truth - no duplication.
try:
    from ..engine.constants import (
        _INTENT_DEFINITIONS_KEYWORDS_SUBSTR,
        _INTENT_CLASSIFICATION_KEYWORDS_SUBSTR,
        _INTENT_ENFORCEMENT_KEYWORDS_SUBSTR,
        _INTENT_REQUIREMENTS_KEYWORDS_STRONG_SUBSTR,
        _INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR,
    )
    _ROLE_KEYWORD_MAP: dict[str, list[str]] = {
        "scope": _INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR,
        "definitions": _INTENT_DEFINITIONS_KEYWORDS_SUBSTR,
        "classification": _INTENT_CLASSIFICATION_KEYWORDS_SUBSTR,
        "obligations": _INTENT_REQUIREMENTS_KEYWORDS_STRONG_SUBSTR,
        "enforcement": _INTENT_ENFORCEMENT_KEYWORDS_SUBSTR,
    }
except ImportError:
    # Fallback for standalone usage (e.g., scripts)
    _ROLE_KEYWORD_MAP: dict[str, list[str]] = {
        "scope": ["anvendelsesområde", "scope", "this regulation applies"],
        "definitions": ["definitioner", "definitions"],
        "classification": ["klassificering", "classification", "high-risk"],
        "obligations": ["forpligtelser", "obligations", "requirements"],
        "enforcement": ["sanktioner", "penalties", "bøder"],
    }


@dataclass
class CitationNode:
    """A node (article) in the citation graph."""

    article_id: str
    node_type: str = "article"  # article, recital, annex, chapter
    title: str = ""
    chapter: str | None = None
    mention_count: int = 0  # How often this article is mentioned elsewhere
    roles: list[str] = field(default_factory=list)  # Detected roles: scope, definitions, etc.

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": self.node_type,
        }
        if self.title:
            d["title"] = self.title
        if self.chapter:
            d["chapter"] = self.chapter
        if self.roles:
            d["roles"] = self.roles
        if self.mention_count > 0:
            d["mention_count"] = self.mention_count
        return d

    @classmethod
    def from_dict(cls, article_id: str, d: dict[str, Any]) -> CitationNode:
        return cls(
            article_id=article_id,
            node_type=d.get("type", "article"),
            title=d.get("title", ""),
            chapter=d.get("chapter"),
            mention_count=d.get("mention_count", 0),
            roles=list(d.get("roles", [])),
        )


@dataclass
class CitationGraph:
    """A graph of citation relationships between legal provisions.

    Nodes are articles (or other citable units like recitals, annexes).
    Edges represent:
    - "cites": Article A explicitly references article B
    - "co-occurrence": Articles A and B are mentioned in the same chunk
    - "same_chapter": Articles A and B belong to the same structural chapter
    """

    corpus_id: str
    nodes: dict[str, CitationNode] = field(default_factory=dict)
    edges: list[CitationEdge] = field(default_factory=list)
    generated_at: str = ""

    # Internal index for fast lookup
    _adjacency: dict[str, dict[str, CitationEdge]] = field(
        default_factory=lambda: defaultdict(dict), repr=False
    )

    def __post_init__(self) -> None:
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        # Build adjacency index
        for edge in self.edges:
            self._adjacency[edge.source][edge.target] = edge
            # Bidirectional for structural relationships
            # - co-occurrence: articles mentioned together
            # - same_chapter: articles in same chapter
            # - contains: parent-child hierarchy (enables boost propagation to children)
            if edge.edge_type in ("co-occurrence", "same_chapter", "contains"):
                self._adjacency[edge.target][edge.source] = edge

    def add_node(self, node: CitationNode) -> None:
        """Add or update a node."""
        if node.article_id in self.nodes:
            existing = self.nodes[node.article_id]
            existing.mention_count += node.mention_count
            if node.title and not existing.title:
                existing.title = node.title
            if node.chapter and not existing.chapter:
                existing.chapter = node.chapter
            # Accumulate roles from all chunks
            for role in node.roles:
                if role not in existing.roles:
                    existing.roles.append(role)
        else:
            self.nodes[node.article_id] = node

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str = "cites",
        count: int = 1,
    ) -> None:
        """Add or increment an edge between two nodes."""
        if source == target:
            return  # Skip self-references

        # Check if edge exists
        existing = self._adjacency.get(source, {}).get(target)
        if existing and existing.edge_type == edge_type:
            existing.count += count
        else:
            edge = CitationEdge(
                source=source, target=target, edge_type=edge_type, count=count
            )
            self.edges.append(edge)
            self._adjacency[source][target] = edge
            if edge_type in ("co-occurrence", "same_chapter", "contains"):
                self._adjacency[target][source] = edge

    def normalize_weights(self) -> None:
        """Normalize edge weights based on counts."""
        if not self.edges:
            return

        max_count = max(e.count for e in self.edges)
        if max_count == 0:
            return

        for edge in self.edges:
            edge.weight = edge.count / max_count

    def get_related_articles(
        self,
        article: str,
        *,
        max_depth: int = 1,
        min_weight: float = 0.0,
        edge_types: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Get articles related to the given article.

        Args:
            article: The article ID to find relations for
            max_depth: How many hops to follow (1 = direct only)
            min_weight: Minimum edge weight to include
            edge_types: Filter by edge types (None = all)

        Returns:
            List of (article_id, relevance_score) tuples, sorted by score desc
        """
        if article not in self._adjacency:
            return []

        visited: set[str] = {article}
        scores: dict[str, float] = {}

        def explore(current: str, depth: int, accumulated_weight: float) -> None:
            if depth > max_depth:
                return

            for target, edge in self._adjacency.get(current, {}).items():
                if target in visited:
                    continue

                if edge_types and edge.edge_type not in edge_types:
                    continue

                if edge.weight < min_weight:
                    continue

                # Decay weight with depth
                score = accumulated_weight * edge.weight * (0.5 ** (depth - 1))
                scores[target] = max(scores.get(target, 0), score)

                if depth < max_depth:
                    visited.add(target)
                    explore(target, depth + 1, score)

        explore(article, 1, 1.0)

        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))

    def get_articles_by_role(self, role: str) -> list[str]:
        """Get all articles with a specific role.

        Args:
            role: Role name like "scope", "definitions", "classification", etc.

        Returns:
            List of article IDs with that role, sorted by article number (low first)
            then mention_count (high first). Foundational articles (scope, definitions)
            typically appear early in the law (art. 1-10).
        """
        matching: list[tuple[str, int, int]] = []
        for node in self.nodes.values():
            if role not in node.roles:
                continue
            # Extract numeric part of article ID for sorting
            art_num = 999
            try:
                art_num = int("".join(c for c in node.article_id if c.isdigit()) or "999")
            except ValueError:
                pass
            matching.append((node.article_id, art_num, node.mention_count))

        # Sort by article number (low first), then mention_count (high first)
        matching.sort(key=lambda x: (x[1], -x[2]))
        return [a for a, _, _ in matching]

    def get_foundational_articles(self) -> dict[str, list[str]]:
        """Get foundational articles organized by role.
        
        These are the articles that should typically be included when 
        answering questions about a topic. Returns dict like:
        {
            "scope": ["2"],
            "definitions": ["3"],
            "classification": ["6"],
            ...
        }
        """
        result: dict[str, list[str]] = {}
        for role in _ROLE_KEYWORD_MAP.keys():
            articles = self.get_articles_by_role(role)
            if articles:
                result[role] = articles[:5]  # Top 5 per role (aligned with config)
        return result

    def get_context_expansion_for_articles(
        self,
        retrieved_articles: list[str],
        *,
        include_foundational_roles: list[str] | None = None,
    ) -> list[str]:
        """Get additional articles to include based on what was retrieved.
        
        This is the key function for prod - given retrieved articles,
        figure out what else should be included automatically.
        
        Args:
            retrieved_articles: Articles found by initial retrieval
            include_foundational_roles: Roles to always include (e.g., ["scope", "definitions"])
                                       If None, uses smart defaults based on what was retrieved.
        
        Returns:
            List of additional article IDs to inject into context.
        """
        expansion: list[str] = []
        retrieved_set = set(a.upper() for a in retrieved_articles)
        
        # 1. If a classification article is retrieved, include scope + definitions
        if include_foundational_roles is None:
            include_foundational_roles = []
            for art in retrieved_articles:
                art_upper = art.upper()
                node = self.nodes.get(art_upper)
                if node and "classification" in node.roles:
                    # Classification questions need scope and definitions
                    include_foundational_roles.extend(["scope", "definitions"])
                    break
        
        # 2. Add foundational articles for requested roles
        for role in include_foundational_roles:
            for art in self.get_articles_by_role(role)[:2]:  # Max 2 per role
                if art not in retrieved_set and art not in expansion:
                    expansion.append(art)
        
        # 3. Add directly cited articles from retrieved articles
        # Import config-based limits (defaults align with Anthropic best practice)
        try:
            from ..engine.citation_expansion import get_seed_limit, get_max_expansion
            seed_limit = get_seed_limit()
            max_exp = get_max_expansion()
        except ImportError:
            seed_limit = 20
            max_exp = 10
        
        for art in retrieved_articles[:seed_limit]:
            art_upper = art.upper()
            for related, weight in self.get_related_articles(art_upper, min_weight=0.3):
                if related not in retrieved_set and related not in expansion:
                    expansion.append(related)
                    if len(expansion) >= max_exp:
                        break
            if len(expansion) >= max_exp:
                break
        
        return expansion[:max_exp]

    def get_co_cited_articles(
        self,
        articles: list[str],
        *,
        min_weight: float = 0.1,
    ) -> list[str]:
        """Get articles that frequently appear together with ALL given articles.

        Useful for finding articles that "bridge" multiple concepts.
        """
        if not articles:
            return []

        # Get candidates from first article
        first = articles[0]
        candidates = {
            a for a, score in self.get_related_articles(first, min_weight=min_weight)
        }

        # Intersect with candidates from other articles
        for article in articles[1:]:
            other = {
                a
                for a, score in self.get_related_articles(article, min_weight=min_weight)
            }
            candidates &= other

        return sorted(candidates)

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "schema_version": SCHEMA_VERSION,
            "corpus_id": self.corpus_id,
            "generated_at": self.generated_at,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CitationGraph:
        """Deserialize graph from dictionary."""
        nodes = {k: CitationNode.from_dict(k, v) for k, v in d.get("nodes", {}).items()}
        edges = [CitationEdge.from_dict(e) for e in d.get("edges", [])]

        graph = cls(
            corpus_id=d["corpus_id"],
            nodes=nodes,
            edges=edges,
            generated_at=d.get("generated_at", ""),
        )
        return graph

    def save(self, path: Path | str | None = None) -> Path:
        """Save graph to JSON file."""
        if path is None:
            path = DEFAULT_DATA_DIR / f"citation_graph_{self.corpus_id}.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(
            "Saved citation graph for %s: %d nodes, %d edges → %s",
            self.corpus_id,
            len(self.nodes),
            len(self.edges),
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> CitationGraph:
        """Load graph from JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_corpus(
        cls,
        corpus_id: str,
        *,
        data_dir: Path | str | None = None,
        include_recitals: bool = False,
    ) -> CitationGraph:
        """Build citation graph from corpus chunks.

        Reads the chunks.jsonl file and extracts:
        1. Direct citations from the 'mentions' metadata field
        2. Co-occurrence relationships (articles mentioned in same chunk)
        3. Structural relationships from chapter metadata
        """
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        data_dir = Path(data_dir)

        chunks_path = data_dir / f"{corpus_id}_chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

        graph = cls(corpus_id=corpus_id)
        _build_graph_from_chunks(graph, chunks_path, include_recitals=include_recitals)

        # Normalize weights after building
        graph.normalize_weights()

        logger.info(
            "Built citation graph for %s: %d nodes, %d edges",
            corpus_id,
            len(graph.nodes),
            len(graph.edges),
        )
        return graph


def _detect_roles_from_text(text: str, title: str = "") -> list[str]:
    """Detect article roles from chunk text and title.
    
    Uses keyword lists from constants.py (single source of truth).
    Returns list of role names like ["scope", "definitions"].
    """
    combined = f"{title} {text}".lower()
    detected: list[str] = []
    
    for role, keywords in _ROLE_KEYWORD_MAP.items():
        for kw in keywords:
            if kw.lower() in combined:
                if role not in detected:
                    detected.append(role)
                break  # One keyword match is enough for this role
    
    return detected


def _build_graph_from_chunks(
    graph: CitationGraph,
    chunks_path: Path,
    *,
    include_recitals: bool = False,
) -> None:
    """Parse chunks and build graph edges."""

    # Track article locations for structural relationships
    article_chapters: dict[str, str] = {}

    # Track co-occurrences per chunk
    chunk_mentions: list[set[str]] = []
    
    # Track detected roles per article (accumulated across chunks)
    article_roles: dict[str, set[str]] = defaultdict(set)

    for chunk in _iter_chunks(chunks_path):
        meta = chunk.get("metadata", {})
        chunk_text = chunk.get("text", "")

        # Identify what structural unit this chunk belongs to.
        # Historically we only created outgoing edges from article chunks;
        # annex chunks must also contribute edges to make annex navigation work.
        chunk_article = meta.get("article")
        chunk_annex = meta.get("annex")
        chunk_chapter = meta.get("chapter")
        chunk_title = meta.get("article_title") or meta.get("annex_title") or ""

        chunk_node: str | None = None
        chunk_node_type: str | None = None
        if chunk_article:
            chunk_node = str(chunk_article).upper()
            chunk_node_type = "article"
        elif chunk_annex:
            chunk_node = f"ANNEX:{str(chunk_annex).upper()}"
            chunk_node_type = "annex"

        # Also create sub-structure nodes for annexes:
        # - section-level: ANNEX:VIII:A (for annexes with sections like A, B, C)
        # - point-level: ANNEX:III:5 or ANNEX:VIII:A:5 (for points within annexes/sections)
        chunk_annex_section = meta.get("annex_section")
        chunk_annex_point = meta.get("annex_point")

        section_node: str | None = None
        punkt_node: str | None = None

        if chunk_annex and chunk_annex_section:
            # Section-level node (e.g., ANNEX:VIII:A)
            section_node = f"ANNEX:{str(chunk_annex).upper()}:{chunk_annex_section}"

        if chunk_annex and chunk_annex_point:
            if chunk_annex_section:
                # Point within section (e.g., ANNEX:VIII:A:5)
                punkt_node = f"ANNEX:{str(chunk_annex).upper()}:{chunk_annex_section}:{chunk_annex_point}"
            else:
                # Point directly under annex (e.g., ANNEX:III:5)
                punkt_node = f"ANNEX:{str(chunk_annex).upper()}:{chunk_annex_point}"

        if chunk_node and chunk_node_type:
            # Detect roles: prefer LLM-generated roles from metadata if available
            # This enables replacing keyword detection with LLM-based classification
            llm_roles = meta.get("roles", [])
            if llm_roles:
                # Use LLM-generated roles from ingestion enrichment
                roles = llm_roles if isinstance(llm_roles, list) else []
            else:
                # Fallback to keyword detection (backwards compatibility)
                roles = _detect_roles_from_text(chunk_text, chunk_title)
            article_roles[chunk_node].update(roles)
            
            graph.add_node(
                CitationNode(
                    article_id=chunk_node,
                    node_type=chunk_node_type,
                    chapter=str(chunk_chapter) if chunk_chapter else None,
                    title=str(chunk_title) if chunk_title else "",
                    roles=list(article_roles[chunk_node]),
                )
            )
            # Track chapters only for article structural relationships.
            if chunk_node_type == "article" and chunk_chapter:
                article_chapters[chunk_node] = str(chunk_chapter)

            # Create section-level node for annexes with sections
            if section_node and chunk_node_type == "annex":
                section_title = meta.get("annex_section_title", "")
                article_roles[section_node].update(roles)
                graph.add_node(
                    CitationNode(
                        article_id=section_node,
                        node_type="annex_section",
                        title=str(section_title) if section_title else "",
                        roles=list(article_roles[section_node]),
                    )
                )
                # Link section to parent annex
                graph.add_edge(chunk_node, section_node, edge_type="contains")

            # Create punkt-level node for annex points
            if punkt_node and chunk_node_type == "annex":
                punkt_title = meta.get("annex_point_title", "") or str(chunk_title)
                article_roles[punkt_node].update(roles)
                graph.add_node(
                    CitationNode(
                        article_id=punkt_node,
                        node_type="annex_point",
                        title=str(punkt_title) if punkt_title else "",
                        roles=list(article_roles[punkt_node]),
                    )
                )
                # Link punkt to parent (section if exists, otherwise annex)
                parent_node = section_node if section_node else chunk_node
                graph.add_edge(parent_node, punkt_node, edge_type="contains")

        # Parse mentions field
        mentions_raw = meta.get("mentions")
        if not mentions_raw:
            continue

        # Parse JSON string if needed
        if isinstance(mentions_raw, str):
            try:
                mentions = json.loads(mentions_raw)
            except json.JSONDecodeError:
                continue
        else:
            mentions = mentions_raw

        mentioned_articles: set[str] = set()

        # Extract article mentions
        for article_ref in mentions.get("article", []):
            article_id = str(article_ref).upper()
            mentioned_articles.add(article_id)
            graph.add_node(
                CitationNode(article_id=article_id, node_type="article", mention_count=1)
            )

        # Include annex mentions (important for high-risk classification)
        for annex_ref in mentions.get("annex", []):
            annex_id = f"ANNEX:{annex_ref}".upper()
            mentioned_articles.add(annex_id)
            graph.add_node(
                CitationNode(article_id=annex_id, node_type="annex", mention_count=1)
            )

        # Optionally include recital mentions
        if include_recitals:
            for recital_ref in mentions.get("recital", []):
                recital_id = f"recital:{recital_ref}"
                graph.add_node(
                    CitationNode(
                        article_id=recital_id, node_type="recital", mention_count=1
                    )
                )

        # Add direct citation edges (from chunk's structural unit to mentioned units)
        if chunk_node and mentioned_articles:
            for target in mentioned_articles:
                if target != chunk_node:
                    graph.add_edge(
                        source=chunk_node,
                        target=target,
                        edge_type="cites",
                    )

        # Track for co-occurrence analysis
        if len(mentioned_articles) > 1:
            chunk_mentions.append(mentioned_articles)

    # Build co-occurrence edges
    co_occurrence_counts: dict[tuple[str, str], int] = defaultdict(int)
    for mentions in chunk_mentions:
        sorted_articles = sorted(mentions)
        for i, a in enumerate(sorted_articles):
            for b in sorted_articles[i + 1 :]:
                co_occurrence_counts[(a, b)] += 1

    for (a, b), count in co_occurrence_counts.items():
        if count >= 2:  # Only include if they co-occur at least twice
            graph.add_edge(a, b, edge_type="co-occurrence", count=count)

    # Build same-chapter edges
    chapter_articles: dict[str, list[str]] = defaultdict(list)
    for article, chapter in article_chapters.items():
        chapter_articles[chapter].append(article)

    for chapter, articles in chapter_articles.items():
        if len(articles) > 1:
            sorted_articles = sorted(articles)
            for i, a in enumerate(sorted_articles):
                for b in sorted_articles[i + 1 :]:
                    graph.add_edge(a, b, edge_type="same_chapter", count=1)


def _iter_chunks(path: Path) -> Iterator[dict[str, Any]]:
    """Iterate over chunks in a JSONL file."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Failed to parse chunk line: %s...", line[:50])


def load_citation_graph(corpus_id: str, data_dir: Path | str | None = None) -> CitationGraph | None:
    """Load a pre-built citation graph for a corpus.

    Returns None if the graph file doesn't exist.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    graph_path = data_dir / f"citation_graph_{corpus_id}.json"
    if not graph_path.exists():
        return None

    return CitationGraph.load(graph_path)


def get_or_build_citation_graph(
    corpus_id: str,
    *,
    data_dir: Path | str | None = None,
    rebuild: bool = False,
) -> CitationGraph:
    """Get citation graph, building it if necessary.

    Args:
        corpus_id: The corpus to get graph for
        data_dir: Data directory (defaults to data/processed)
        rebuild: Force rebuild even if cached graph exists
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    graph_path = data_dir / f"citation_graph_{corpus_id}.json"

    if not rebuild and graph_path.exists():
        return CitationGraph.load(graph_path)

    graph = CitationGraph.from_corpus(corpus_id, data_dir=data_dir)
    graph.save(graph_path)
    return graph
