"""Cross-law evaluation case generator.

This module provides LLM-assisted generation of cross-law test cases
for evaluation suites.

Single Responsibility: Generate test cases using LLM.
Does NOT persist cases - that's cross_law_suite_manager.py's job.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.llm_helpers import (
    call_generation_llm,
    parse_json_response,
    load_article_content,
)
from src.ingestion.citation_graph import CitationGraph, load_citation_graph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CASES_LIMIT = 20  # R11.6: Max cases per generation request
MAX_LLM_RETRIES = 3  # Max retries for LLM call + parse

# R9.3: Default test type distribution (sums to 15)
DEFAULT_DISTRIBUTION: dict[str, int] = {
    "corpus_coverage": 4,
    "synthesis_balance": 3,
    "comparison_completeness": 3,
    "routing_precision": 2,
    "cross_reference_accuracy": 2,
    "abstention": 1,
}

# R9.1: Base types auto-applied to non-abstention cases
BASE_TEST_TYPES = ("retrieval", "faithfulness", "relevancy")


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------


class CaseGenerationError(Exception):
    """Raised when case generation fails."""


# ---------------------------------------------------------------------------
# Inverted Generation Data Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArticleSpec:
    """A specific article selected as an anchor seed for inverted generation."""

    corpus_id: str
    article_id: str        # "6", "ANNEX:III"
    anchor_key: str        # "article:6", "annex:III"
    title: str
    content_snippet: str   # First ~400 chars of article text


@dataclass(frozen=True)
class ArticleGroup:
    """A group of articles from different corpora sharing a thematic role."""

    articles: tuple[ArticleSpec, ...]
    shared_role: str       # "scope", "obligations", etc.


# ---------------------------------------------------------------------------
# Article Chunk Loading (for inverted generation)
# ---------------------------------------------------------------------------

# Roles to search for shared thematic groups across corpora.
# Includes procedural/rights/exemptions roles that map to deeper articles (10+).
_SHARED_ROLES = (
    "obligations",
    "enforcement",
    "scope",
    "definitions",
    "classification",
    "procedures",
    "rights",
    "exemptions",
)

# Articles at or below this number are considered "early/foundational" —
# scope, definitions, etc. that are trivially retrieved. We prefer deeper ones.
_EARLY_ARTICLE_THRESHOLD = 9


def _article_num(article_id: str) -> int:
    """Extract numeric part of an article ID. Returns 999 for non-numeric."""
    try:
        return int("".join(c for c in article_id if c.isdigit()) or "999")
    except ValueError:
        return 999


def _build_specs_from_pool(
    pool: dict[str, list[str]],
    available: dict[str, Any],
    data_dir: str,
) -> list[ArticleSpec]:
    """Build ArticleSpec list by picking the top article per corpus from pool."""
    specs: list[ArticleSpec] = []
    for cid, aids in pool.items():
        top_aid = aids[0]
        graph = available[cid]
        node = graph.nodes.get(top_aid)
        title = node.title if node else ""

        chunk_text = _load_article_chunks(data_dir, cid, [top_aid], max_chars=400)
        snippet = chunk_text.get(top_aid, title)

        if top_aid.startswith("ANNEX:"):
            anchor_key = f"{cid}:annex:{top_aid.replace('ANNEX:', '')}"
        else:
            anchor_key = f"{cid}:article:{top_aid}"

        specs.append(ArticleSpec(
            corpus_id=cid,
            article_id=top_aid,
            anchor_key=anchor_key,
            title=title,
            content_snippet=snippet,
        ))
    return specs


def _load_article_chunks(
    data_dir: str,
    corpus_id: str,
    article_ids: list[str],
    max_chars: int = 400,
) -> dict[str, str]:
    """Load article text from chunks JSONL for specific articles.

    Args:
        data_dir: Directory containing processed chunks files
        corpus_id: Corpus ID (e.g. "ai-act")
        article_ids: Article IDs to load (e.g. ["6", "21"])
        max_chars: Max characters of text per article

    Returns:
        Dict mapping article_id → text snippet. Missing articles omitted.
    """
    chunks_path = Path(data_dir) / f"{corpus_id}_chunks.jsonl"
    if not chunks_path.exists():
        return {}

    wanted = set(article_ids)
    articles: dict[str, list[str]] = {aid: [] for aid in wanted}

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                meta = chunk.get("metadata", {})
                art = str(meta.get("article", ""))
                if art in wanted:
                    text = chunk.get("text", "")
                    if text:
                        articles[art].append(text)
    except Exception:
        logger.warning("Could not load chunks for %s", corpus_id)
        return {}

    result: dict[str, str] = {}
    for aid, chunks in articles.items():
        if not chunks:
            continue
        combined = " ".join(chunks)
        if len(combined) > max_chars:
            combined = combined[:max_chars] + "..."
        result[aid] = combined

    return result


def _select_anchor_articles(
    target_corpora: tuple[str, ...],
    max_groups: int,
    graph_cache: dict[str, CitationGraph | None],
    data_dir: str | None = None,
) -> list[ArticleGroup]:
    """Select article groups for inverted generation.

    Finds articles that share thematic roles across corpora, forming
    ArticleGroups suitable for generating cross-law questions.

    Args:
        target_corpora: Corpus IDs to select from
        max_groups: Maximum number of groups to return
        graph_cache: Pre-loaded citation graphs keyed by corpus_id
        data_dir: Directory for chunks JSONL (defaults to data/processed)

    Returns:
        List of ArticleGroup, each containing one ArticleSpec per corpus
    """
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parents[2] / "data" / "processed")

    # Filter to corpora with available graphs
    available = {cid: g for cid, g in graph_cache.items() if g is not None and cid in target_corpora}
    if len(available) < 2:
        return []

    groups: list[ArticleGroup] = []

    # Try shared roles — for each role, produce groups preferring deep articles
    for role in _SHARED_ROLES:
        if len(groups) >= max_groups:
            break

        # Collect all candidate articles per corpus for this role
        candidates_per_corpus: dict[str, list[str]] = {}
        for cid, graph in available.items():
            role_articles = graph.get_articles_by_role(role)
            if role_articles:
                candidates_per_corpus[cid] = role_articles

        # Need at least 2 corpora to form cross-law groups
        if len(candidates_per_corpus) < 2:
            continue

        # Partition articles into deep (>threshold) and early (<=threshold)
        deep_per_corpus: dict[str, list[str]] = {}
        for cid, aids in candidates_per_corpus.items():
            deep = [
                aid for aid in aids
                if aid.startswith("ANNEX:") or _article_num(aid) > _EARLY_ARTICLE_THRESHOLD
            ]
            if deep:
                deep_per_corpus[cid] = deep

        # Partition early articles too
        early_per_corpus: dict[str, list[str]] = {}
        for cid, aids in candidates_per_corpus.items():
            early = [
                aid for aid in aids
                if not aid.startswith("ANNEX:") and _article_num(aid) <= _EARLY_ARTICLE_THRESHOLD
            ]
            if early:
                early_per_corpus[cid] = early

        # Group 1: deep articles (preferred — substantive, harder to retrieve)
        has_deep_group = False
        if len(deep_per_corpus) >= 2:
            specs = _build_specs_from_pool(deep_per_corpus, available, data_dir)
            if len(specs) >= 2:
                groups.append(ArticleGroup(articles=tuple(specs), shared_role=role))
                has_deep_group = True

        # Group 2: complementary group for diversity
        if len(groups) < max_groups:
            if has_deep_group and len(early_per_corpus) >= 2:
                # Deep group exists → add early group for variety
                alt_pool = early_per_corpus
            elif not has_deep_group:
                # No deep group possible → use best available per corpus
                alt_pool = {cid: aids[:1] for cid, aids in candidates_per_corpus.items()}
            else:
                alt_pool = {}

            if len(alt_pool) >= 2:
                specs = _build_specs_from_pool(alt_pool, available, data_dir)
                if len(specs) >= 2:
                    groups.append(ArticleGroup(articles=tuple(specs), shared_role=role))

    # Fallback: use foundational articles if no shared roles found
    if not groups:
        all_specs: list[ArticleSpec] = []
        for cid, graph in available.items():
            foundational = graph.get_foundational_articles()
            # Pick first article from first available role
            for role, aids in foundational.items():
                if aids:
                    top_aid = aids[0]
                    node = graph.nodes.get(top_aid)
                    title = node.title if node else ""
                    chunk_text = _load_article_chunks(data_dir, cid, [top_aid], max_chars=400)
                    snippet = chunk_text.get(top_aid, title)
                    anchor_key = f"{cid}:article:{top_aid}"
                    all_specs.append(ArticleSpec(
                        corpus_id=cid,
                        article_id=top_aid,
                        anchor_key=anchor_key,
                        title=title,
                        content_snippet=snippet,
                    ))
                    break

        if len(all_specs) >= 2:
            groups.append(ArticleGroup(articles=tuple(all_specs), shared_role="foundational"))

    return groups[:max_groups]


# ---------------------------------------------------------------------------
# Article Index & Anchor Validation (R5, R6)
# ---------------------------------------------------------------------------


def _build_article_index(corpus_id: str) -> str:
    """Build compact article index from citation graph for generation prompt.

    Returns one line per article/annex with number and title.
    Falls back to load_article_content() if citation graph is unavailable.

    Args:
        corpus_id: Corpus ID to build index for

    Returns:
        Formatted string with all articles and top-level annexes
    """
    graph = load_citation_graph(corpus_id)
    if graph is None:
        return load_article_content(corpus_id)

    lines: list[str] = []

    # Collect articles (sorted numerically)
    articles = []
    annexes = []
    for key, node in graph.nodes.items():
        if node.node_type == "article":
            articles.append((key, node))
        elif node.node_type == "annex":
            annexes.append((key, node))
        # Skip annex_point, annex_section

    # Sort articles numerically
    articles.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 9999)
    for key, node in articles:
        lines.append(f"Artikel {key}: {node.title}")

    # Sort annexes by Roman numeral key (ANNEX:I, ANNEX:II, etc.)
    annexes.sort(key=lambda x: x[0])
    for key, node in annexes:
        # ANNEX:III → Bilag III
        roman = key.replace("ANNEX:", "")
        title_part = f": {node.title}" if node.title else ""
        lines.append(f"Bilag {roman}{title_part}")

    return "\n".join(lines)


def _validate_anchors(
    anchors: tuple[str, ...],
    expected_corpora: tuple[str, ...],
    graph_cache: dict[str, Any],
) -> tuple[str, ...]:
    """Validate anchors against citation graphs, dropping invalid ones.

    Args:
        anchors: Tuple of anchors (e.g., "article:13", "annex:III")
        expected_corpora: Corpus IDs to check against
        graph_cache: Pre-loaded citation graphs keyed by corpus_id

    Returns:
        Tuple of valid anchors (invalid ones dropped)
    """
    # If no graphs available at all, skip validation
    available_graphs = [
        graph_cache.get(cid) for cid in expected_corpora
        if graph_cache.get(cid) is not None
    ]
    if not available_graphs:
        return anchors

    validated: list[str] = []
    for anchor in anchors:
        parts = anchor.split(":", 1)
        if len(parts) != 2:
            # Unknown format — keep it
            validated.append(anchor)
            continue

        anchor_type, anchor_key = parts[0].lower(), parts[1]

        # Map anchor format to graph node key
        if anchor_type == "article":
            graph_key = anchor_key
        elif anchor_type == "annex":
            graph_key = f"ANNEX:{anchor_key}"
        else:
            # Unknown type (e.g., recital) — keep it
            validated.append(anchor)
            continue

        # Check if key exists in ANY expected corpus graph
        found = False
        for cid in expected_corpora:
            graph = graph_cache.get(cid)
            if graph is not None and graph_key in graph.nodes:
                found = True
                break

        if found:
            validated.append(anchor)
        else:
            logger.warning("Dropped invalid anchor %s (not found in %s)", anchor, expected_corpora)

    if not validated and anchors:
        logger.error(
            "All anchors dropped for case with corpora %s: %s",
            expected_corpora, anchors,
        )

    return tuple(validated)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class GenerationRequest:
    """Request for generating cross-law test cases.

    Validates constraints at construction time.
    """

    target_corpora: tuple[str, ...]
    synthesis_mode: str  # "aggregation" | "comparison" | "unified" | "routing"
    max_cases: int = 10
    topic_hints: tuple[str, ...] = ()
    generation_strategy: str = "standard"  # "standard" | "inverted"

    def __post_init__(self):
        """Validate request constraints."""
        # E31: Require at least 2 corpora for cross-law tests
        if len(self.target_corpora) < 2:
            raise ValueError(
                "Auto-generation requires at least 2 corpora for cross-law tests"
            )

        # R11.6: Cap max_cases at 20
        if self.max_cases > MAX_CASES_LIMIT:
            object.__setattr__(self, "max_cases", MAX_CASES_LIMIT)


@dataclass(frozen=True)
class GeneratedCase:
    """A generated cross-law test case.

    Immutable and always has origin="auto-generated".
    """

    id: str
    prompt: str
    synthesis_mode: str
    expected_corpora: tuple[str, ...]
    expected_anchors: tuple[str, ...]
    test_types: tuple[str, ...] = ()
    retrieval_confirmed: bool | None = None
    origin: str = field(default="auto-generated", init=False)


# ---------------------------------------------------------------------------
# Generation Function
# ---------------------------------------------------------------------------


async def generate_cross_law_cases(
    request: GenerationRequest,
    corpus_metadata: dict[str, dict[str, Any]],
) -> list[GeneratedCase]:
    """Generate cross-law test cases using LLM.

    Args:
        request: Generation parameters (corpora, mode, max_cases)
        corpus_metadata: Metadata for target corpora (name, example_questions)

    Returns:
        List of GeneratedCase objects

    Raises:
        CaseGenerationError: If LLM call fails

    Process:
    1. Build prompt with corpus metadata and synthesis mode
    2. Ask LLM to generate realistic cross-law questions
    3. Parse and validate generated cases
    4. Return with unique, auto-incremented IDs
    """
    # Pre-load citation graphs for anchor validation (R6)
    graph_cache: dict[str, Any] = {
        cid: load_citation_graph(cid) for cid in request.target_corpora
    }

    # Inverted generation: document-first approach
    if request.generation_strategy == "inverted":
        groups = _select_anchor_articles(
            target_corpora=request.target_corpora,
            max_groups=request.max_cases,
            graph_cache=graph_cache,
        )
        if groups:
            return await _generate_inverted_cases(
                groups=groups,
                corpus_metadata=corpus_metadata,
                synthesis_mode=request.synthesis_mode,
                max_cases=request.max_cases,
            )
        # Fall back to standard if no article groups found
        logger.warning("Inverted generation: no article groups found, falling back to standard")

    # Standard generation path
    try:
        if request.synthesis_mode == "discovery":
            prompt = _build_discovery_prompt(request, corpus_metadata)
        else:
            prompt = _build_generation_prompt(request, corpus_metadata)

        raw_cases = await _call_llm_for_cases(
            prompt=prompt,
            max_cases=request.max_cases,
            synthesis_mode=request.synthesis_mode,
        )
    except Exception as e:
        raise CaseGenerationError(f"Case generation failed: {e}")

    # Convert raw cases to GeneratedCase objects with unique IDs
    cases = _process_raw_cases(raw_cases, request.synthesis_mode, graph_cache)

    # Enforce limit
    return cases[: min(len(cases), MAX_CASES_LIMIT, request.max_cases)]


async def _generate_inverted_cases(
    groups: list[ArticleGroup],
    corpus_metadata: dict[str, dict[str, Any]],
    synthesis_mode: str,
    max_cases: int,
) -> list[GeneratedCase]:
    """Generate cases using inverted (document-first) approach.

    For each article group, builds an inverted prompt and calls the LLM.
    Anchors come from the seed articles (reliable by construction),
    NOT from LLM output.

    Args:
        groups: Article groups to generate from
        corpus_metadata: Metadata for corpora
        synthesis_mode: Synthesis mode for all cases
        max_cases: Maximum cases to generate

    Returns:
        List of GeneratedCase with reliable expected_anchors
    """
    cases: list[GeneratedCase] = []
    seen_ids: set[str] = set()

    # Generate multiple cases per group when max_cases > number of groups
    cases_per_group = max(1, -(-max_cases // len(groups)))  # ceiling division

    for group in groups:
        prompt = _build_inverted_prompt(
            group, corpus_metadata, synthesis_mode, num_cases=cases_per_group,
        )

        try:
            raw_cases = await _call_llm_for_cases(
                prompt=prompt,
                max_cases=cases_per_group,
                synthesis_mode=synthesis_mode,
            )
        except CaseGenerationError:
            logger.warning("Inverted generation failed for group %s", group.shared_role)
            continue

        if not raw_cases:
            continue

        # Seed anchors = validation set (articles we know exist)
        seed_anchor_set = frozenset(spec.anchor_key for spec in group.articles)
        seed_corpora = tuple(spec.corpus_id for spec in group.articles)

        for raw in raw_cases:
            # Generate unique ID
            base_id = raw.get("id", f"auto_inverted_{group.shared_role}")
            unique_id = _make_unique_id(base_id, seen_ids)
            seen_ids.add(unique_id)

            # Routing mode: let LLM decide which corpora the question targets
            # Other modes (comparison, aggregation): all seed corpora expected
            if synthesis_mode == "routing":
                llm_corpora = tuple(raw.get("expected_corpora", []))
                validated_corpora = tuple(c for c in llm_corpora if c in seed_corpora)
                case_corpora = validated_corpora if validated_corpora else seed_corpora
            else:
                case_corpora = seed_corpora

            # Use LLM's anchor selection, validated against seed set.
            # LLM was shown these articles and chose which are relevant to its question.
            # Fall back to seed anchors (filtered by corpora) if LLM returns nothing valid.
            llm_anchors = tuple(raw.get("expected_anchors", []))
            validated_anchors = tuple(a for a in llm_anchors if a in seed_anchor_set)
            if validated_anchors:
                case_anchors = validated_anchors
            else:
                # Fallback: all seed anchors filtered to matching corpora
                case_anchors = tuple(
                    a for a in seed_anchor_set
                    if any(a.startswith(f"{cid}:") for cid in case_corpora)
                )

            case = GeneratedCase(
                id=unique_id,
                prompt=raw.get("prompt", ""),
                synthesis_mode=synthesis_mode,
                expected_corpora=case_corpora,
                expected_anchors=case_anchors,
            )
            cases.append(case)

    if not cases:
        raise CaseGenerationError("Inverted generation produced no cases")

    return cases[:min(len(cases), MAX_CASES_LIMIT, max_cases)]


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _build_generation_prompt(
    request: GenerationRequest,
    corpus_metadata: dict[str, dict[str, Any]],
) -> str:
    """Build the LLM prompt for case generation."""
    corpora_info = []
    for corpus_id in request.target_corpora:
        meta = corpus_metadata.get(corpus_id, {})
        name = meta.get("name", corpus_id)
        fullname = meta.get("fullname", name)
        examples = meta.get("example_questions", [])
        examples_str = "\n".join(f"  - {q}" for q in examples[:3])

        # Load article index from citation graph (R5) — falls back to content
        article_content = _build_article_index(corpus_id)

        corpora_info.append(f"""
**{name}** ({fullname}):
{examples_str if examples_str else "  (no example questions available)"}

{article_content}
""")

    corpora_section = "\n".join(corpora_info)

    topic_section = ""
    if request.topic_hints:
        topics = ", ".join(request.topic_hints)
        topic_section = f"\n\nFocus on these topics: {topics}"

    mode_instructions = {
        "comparison": "Generate questions that compare requirements, definitions, or approaches between two or more of these laws.",
        "aggregation": "Generate questions that ask about a topic across all laws (e.g., 'What do all laws say about X?').",
        "unified": "Generate questions that could be answered by any of these laws without specifying which.",
        "routing": "Generate questions that ask which law(s) apply to a specific scenario or topic.",
    }

    return f"""Generate {request.max_cases} cross-law evaluation test cases for EU regulations.

Skriv alle spørgsmål på dansk (write all prompts in Danish).

## Target Laws
{corpora_section}

## Synthesis Mode: {request.synthesis_mode}
{mode_instructions.get(request.synthesis_mode, "")}
{topic_section}

## Output Format
Return a JSON object with this exact schema:
```json
{{
  "cases": [
    {{
      "id": "auto_compare_ai_gdpr_transparency",
      "prompt": "Sammenlign kravene til gennemsigtighed i AI-forordningen og GDPR",
      "synthesis_mode": "{request.synthesis_mode}",
      "expected_corpora": ["corpus-id-1", "corpus-id-2"],
      "expected_anchors": []
    }}
  ]
}}
```

IMPORTANT: Leave expected_anchors as an empty array []. Anchor validation is handled separately.

Generate realistic, useful test cases that would help evaluate cross-law synthesis quality.
"""


async def _call_llm_for_cases(
    prompt: str,
    max_cases: int,
    synthesis_mode: str,
) -> list[dict[str, Any]]:
    """Call LLM to generate cases with retry on parse failure.

    Uses call_generation_llm + parse_json_response from llm_helpers.
    Retries up to MAX_LLM_RETRIES times if parsing fails.

    Raises:
        CaseGenerationError: If all retries fail.
    """
    for attempt in range(MAX_LLM_RETRIES):
        raw = call_generation_llm(
            prompt,
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=3000,
        )
        if raw is None:
            logger.warning("LLM returned None (attempt %d/%d)", attempt + 1, MAX_LLM_RETRIES)
            continue

        parsed = parse_json_response(raw)
        if parsed is None:
            logger.warning("Failed to parse LLM response (attempt %d/%d)", attempt + 1, MAX_LLM_RETRIES)
            continue

        cases = parsed.get("cases", [])
        if isinstance(cases, list) and len(cases) > 0:
            return cases

        logger.warning("No cases in parsed response (attempt %d/%d)", attempt + 1, MAX_LLM_RETRIES)

    raise CaseGenerationError(
        f"Failed to generate cases after {MAX_LLM_RETRIES} attempts"
    )


def _process_raw_cases(
    raw_cases: list[dict[str, Any]],
    synthesis_mode: str,
    graph_cache: dict[str, Any] | None = None,
) -> list[GeneratedCase]:
    """Process raw LLM output into GeneratedCase objects.

    Handles:
    - ID uniqueness (auto-increment duplicates)
    - Type conversion (lists to tuples)
    - Mode override (ensure matches request)
    - Anchor validation against citation graphs (R6)
    """
    if graph_cache is None:
        graph_cache = {}

    seen_ids: set[str] = set()
    cases: list[GeneratedCase] = []

    for raw in raw_cases:
        # Extract and sanitize ID
        base_id = raw.get("id", "auto_case")
        unique_id = _make_unique_id(base_id, seen_ids)
        seen_ids.add(unique_id)

        # Parse raw anchors and corpora before validation
        raw_anchors = tuple(
            a.strip() for a in raw.get("expected_anchors", [])
        )
        expected_corpora = tuple(raw.get("expected_corpora", []))

        # Validate anchors against citation graphs (R6)
        validated_anchors = _validate_anchors(
            raw_anchors, expected_corpora, graph_cache
        )

        # Build GeneratedCase with validated anchors
        case = GeneratedCase(
            id=unique_id,
            prompt=raw.get("prompt", ""),
            synthesis_mode=synthesis_mode,
            expected_corpora=expected_corpora,
            expected_anchors=validated_anchors,
        )
        cases.append(case)

    return cases


def _make_unique_id(base_id: str, seen_ids: set[str]) -> str:
    """Generate a unique ID by appending suffix if needed.

    E32: Auto-increment suffix for duplicates.
    """
    if base_id not in seen_ids:
        return base_id

    counter = 1
    while f"{base_id}_{counter}" in seen_ids:
        counter += 1

    return f"{base_id}_{counter}"


# ---------------------------------------------------------------------------
# Inverted Prompt (document-first generation)
# ---------------------------------------------------------------------------


def _build_inverted_prompt(
    article_group: ArticleGroup,
    corpus_metadata: dict[str, dict[str, Any]],
    synthesis_mode: str,
    num_cases: int = 1,
) -> str:
    """Build LLM prompt for inverted (document-first) case generation.

    Shows actual article text and asks the LLM to generate a question
    answerable by those specific articles. The expected_anchors are
    pre-filled from the seed articles (reliable by construction).

    Args:
        article_group: Group of articles from different corpora
        corpus_metadata: Metadata for corpora (name, fullname)
        synthesis_mode: The synthesis mode for the generated case
        num_cases: Number of cases to generate from this group

    Returns:
        Prompt string for the LLM
    """
    # Build article sections
    article_sections: list[str] = []
    anchor_keys: list[str] = []
    corpus_ids: list[str] = []

    for spec in article_group.articles:
        meta = corpus_metadata.get(spec.corpus_id, {})
        law_name = meta.get("name", spec.corpus_id)
        article_sections.append(
            f"### {law_name} — {spec.title} ({spec.anchor_key})\n{spec.content_snippet}"
        )
        anchor_keys.append(f"{spec.anchor_key} — {spec.title}")
        if spec.corpus_id not in corpus_ids:
            corpus_ids.append(spec.corpus_id)

    articles_text = "\n\n".join(article_sections)
    anchors_str = "\n".join(f"- {ak}" for ak in anchor_keys)
    corpora_str = ", ".join(f'"{cid}"' for cid in corpus_ids)

    discovery_instruction = ""
    if synthesis_mode == "discovery":
        discovery_instruction = (
            "\nIMPORTANT: Do NOT mention specific law names (e.g., GDPR, AI Act, NIS2, DORA) "
            "in the generated question. The question must be topic-based so the system "
            "must discover which laws are relevant.\n"
        )

    mode_instructions = {
        "comparison": "Generate a question that compares requirements or approaches between these laws.",
        "aggregation": "Generate a question that asks about a topic across all these laws.",
        "discovery": "Generate a topic-based question that these articles can answer.",
        "routing": (
            "Generate a question that asks which of these laws applies to a scenario. "
            "In expected_corpora, list ONLY the 1-3 most relevant corpus IDs for your "
            "question — do NOT include all corpora. A routing question typically targets "
            "a subset of the available laws."
        ),
        "unified": "Generate a question answerable by any of these laws.",
    }

    # For routing mode, tell LLM to choose relevant corpora from the list
    if synthesis_mode == "routing":
        corpora_note = f"Choose the 1-3 most relevant corpus IDs from: [{corpora_str}]"
    else:
        corpora_note = ""

    return f"""Generate {num_cases} cross-law evaluation test case based on these specific articles.

Skriv alle spørgsmål på dansk (write all prompts in Danish).

## Source Articles

{articles_text}

## Available anchor keys
{anchors_str}

## Instructions

{mode_instructions.get(synthesis_mode, "Generate a question answerable by these articles.")}
{discovery_instruction}
The generated question MUST be answerable using the articles above.

CRITICAL for expected_anchors: Each anchor you select must be the SPECIFIC article whose content directly answers the question. Do NOT pick an article just because it is from the right corpus — verify the article title and content match the question topic. For example, if your question is about complaint procedures, pick the article about complaints, NOT a definitions article.
{corpora_note}

## Output Format
Return a JSON object with this exact schema:
```json
{{
  "cases": [
    {{
      "id": "auto_inverted_{article_group.shared_role}",
      "prompt": "Dit spørgsmål her...",
      "synthesis_mode": "{synthesis_mode}",
      "expected_corpora": [{corpora_str}],
      "expected_anchors": ["pick only relevant anchors from the list above"]
    }}
  ]
}}
```

Generate realistic, useful questions in Danish that test cross-law synthesis quality.
"""


def _build_discovery_prompt(
    request: GenerationRequest,
    corpus_metadata: dict[str, dict[str, Any]],
) -> str:
    """Build LLM prompt for discovery-mode case generation.

    Discovery cases are topic-based: the generated questions must NOT mention
    specific law names so the system is forced to identify relevant corpora
    from the topic alone.

    Args:
        request: Generation parameters (corpora, mode, max_cases)
        corpus_metadata: Metadata for target corpora

    Returns:
        Prompt string for the LLM
    """
    # Collect topic areas from corpus metadata (without revealing law names)
    topic_areas: list[str] = []
    for corpus_id in request.target_corpora:
        meta = corpus_metadata.get(corpus_id, {})
        # Use fullname to extract topic keywords, but don't put it in prompt
        article_content = _build_article_index(corpus_id)
        if article_content:
            topic_areas.append(article_content)

    topic_section = ""
    if request.topic_hints:
        topics = ", ".join(request.topic_hints)
        topic_section = f"\n\nFocus on these topics: {topics}"

    # Build corpus ID list for expected_corpora (LLM must use real IDs)
    corpus_ids = list(request.target_corpora)
    corpus_ids_str = ", ".join(f'"{cid}"' for cid in corpus_ids)
    # Build example using first two corpus IDs (or all if fewer)
    example_corpora = corpus_ids[:2] if len(corpus_ids) >= 2 else corpus_ids
    example_corpora_str = ", ".join(f'"{cid}"' for cid in example_corpora)

    return f"""Generate {request.max_cases} topic-based evaluation test cases for EU regulation discovery.

Skriv alle spørgsmål på dansk (write all prompts in Danish).

## Instructions
Generate questions about regulatory topics that span multiple EU laws.
The questions must be topic-based — do NOT mention specific law names (e.g., GDPR, AI Act, NIS2, DORA, CRA).
Instead, describe the topic or scenario so the system must discover which laws are relevant.

IMPORTANT: You must not include any law names or regulation names in the generated questions.
The purpose is to test whether the system can identify the correct laws from the topic alone.

## Available Corpus IDs
Use ONLY these corpus IDs in the expected_corpora field: [{corpus_ids_str}]
Each case should list which of these corpora are relevant to the question.

## Example Good Questions (topic-based, uden lovnavne)
- "Hvilke regler gælder for håndtering af persondata i digitale systemer?"
- "Hvad er kravene til risikostyring for kritiske IT-systemer?"
- "Hvilke forpligtelser har virksomheder vedrørende cybersikkerhed?"

## Example Bad Questions (mentions law names — do NOT generate these)
- "Hvad siger GDPR om databehandling?"
- "Sammenlign AI-forordningen og NIS2"
{topic_section}

## Output Format
Return a JSON object with this exact schema:
```json
{{
  "cases": [
    {{
      "id": "auto_discover_topic_example",
      "prompt": "Hvilke regler gælder for ...",
      "synthesis_mode": "discovery",
      "expected_corpora": [{example_corpora_str}],
      "expected_anchors": []
    }}
  ]
}}
```

Generate realistic questions about topics that span {len(request.target_corpora)} EU regulations.
The expected_corpora must use ONLY corpus IDs from the list above: [{corpus_ids_str}]
"""


# ---------------------------------------------------------------------------
# Synthesis Mode Distribution (R1.7)
# ---------------------------------------------------------------------------


def _distribute_by_synthesis_mode(
    distribution: dict[str, float],
    max_cases: int,
) -> dict[str, int]:
    """Distribute case count across synthesis modes by weight.

    Args:
        distribution: Mode → weight mapping (e.g. {"comparison": 0.5, "discovery": 0.3})
        max_cases: Total number of cases to distribute

    Returns:
        Dict mapping mode → integer count, summing to max_cases

    Raises:
        ValueError: If weights sum to more than 1.0
    """
    total_weight = sum(distribution.values())

    if total_weight > 1.0 + 1e-9:
        raise ValueError(
            f"Synthesis distribution weights sum to {total_weight:.2f}, must be <= 1.0"
        )

    # Normalize if sum < 1.0
    if total_weight < 1e-9:
        raise ValueError("Synthesis distribution weights sum to 0")

    # Calculate proportional counts
    result: dict[str, int] = {}
    allocated = 0
    items = list(distribution.items())

    for mode, weight in items[:-1]:
        count = round((weight / total_weight) * max_cases)
        result[mode] = count
        allocated += count

    # Last mode gets remainder to ensure exact sum
    last_mode = items[-1][0]
    result[last_mode] = max_cases - allocated

    return result



# ---------------------------------------------------------------------------
# Difficulty Assignment
# ---------------------------------------------------------------------------


def assign_difficulty(case: GeneratedCase) -> str:
    """Assign difficulty based on structural complexity.

    Rules:
    - discovery mode → "hard" (topic-based corpus identification)
    - 3+ expected_corpora with comparison → "hard"
    - aggregation mode → "medium"
    - 2 corpora + multi-article anchors → "medium"
    - 2 corpora + single-article anchors → "easy"
    - Default → "medium"

    Args:
        case: Generated case to classify

    Returns:
        One of "easy", "medium", "hard"
    """
    if case.synthesis_mode == "discovery":
        return "hard"

    num_corpora = len(case.expected_corpora)
    num_anchors = len(case.expected_anchors)

    if num_corpora >= 3 and case.synthesis_mode != "aggregation":
        return "hard"

    if case.synthesis_mode == "aggregation":
        return "medium"

    if num_corpora <= 2 and num_anchors <= 1:
        return "easy"

    return "medium"


# ---------------------------------------------------------------------------
# Test Type Distribution
# ---------------------------------------------------------------------------

# R1.5: Mode-specific test type mapping
MODE_TEST_TYPE: dict[str, str] = {
    "comparison": "comparison_completeness",
    "discovery": "corpus_coverage",
    "aggregation": "synthesis_balance",
    "routing": "routing_precision",
}


def assign_test_types(
    cases: list[GeneratedCase],
    distribution: dict[str, int] | None = None,
) -> list[GeneratedCase]:
    """Assign test types to generated cases.

    When distribution is None (default), uses mode-based routing: each case's
    primary test type is determined by its synthesis_mode via MODE_TEST_TYPE.
    When an explicit distribution is provided, uses queue-based assignment.

    Each non-abstention case also gets base types (retrieval, faithfulness,
    relevancy).

    Args:
        cases: Generated cases (frozen, so we return new instances).
        distribution: Type → count mapping. If None, uses mode-based routing.

    Returns:
        New list of GeneratedCase with test_types assigned.
    """
    if distribution is not None:
        return _assign_from_distribution(cases, distribution)

    # Mode-based routing (default)
    result: list[GeneratedCase] = []
    for case in cases:
        primary_type = MODE_TEST_TYPE.get(case.synthesis_mode)
        if primary_type is None:
            primary_type = "corpus_coverage"

        types = (primary_type,) + BASE_TEST_TYPES

        result.append(
            GeneratedCase(
                id=case.id,
                prompt=case.prompt,
                synthesis_mode=case.synthesis_mode,
                expected_corpora=case.expected_corpora,
                expected_anchors=case.expected_anchors,
                test_types=types,
            )
        )

    return result


def _assign_from_distribution(
    cases: list[GeneratedCase],
    distribution: dict[str, int],
) -> list[GeneratedCase]:
    """Assign test types from explicit distribution queue (legacy path)."""
    type_queue: list[str] = []
    for test_type, count in distribution.items():
        type_queue.extend([test_type] * count)

    result: list[GeneratedCase] = []
    for i, case in enumerate(cases):
        if i < len(type_queue):
            primary_type = type_queue[i]
        else:
            primary_type = type_queue[i % len(type_queue)]

        if primary_type == "abstention":
            types = (primary_type,)
        else:
            types = (primary_type,) + BASE_TEST_TYPES

        result.append(
            GeneratedCase(
                id=case.id,
                prompt=case.prompt,
                synthesis_mode=case.synthesis_mode,
                expected_corpora=case.expected_corpora,
                expected_anchors=case.expected_anchors,
                test_types=types,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Suite Name/Description Suggestion
# ---------------------------------------------------------------------------

_MODE_DESCRIPTIONS = {
    "comparison": "sammenligner krav og definitioner mellem lovene",
    "routing": "identificerer hvilke love der gælder for et scenarie",
    "aggregation": "samler information om et emne på tværs af lovene",
}


def suggest_suite_text(
    suggest_type: str,
    corpora_ids: list[str],
    corpora_names: list[str] | None = None,
    synthesis_mode: str = "comparison",
) -> str:
    """Generate an AI-powered name or description for a cross-law eval suite.

    Args:
        suggest_type: "name" or "description"
        corpora_ids: Corpus IDs (e.g. ["eucs-cir-2025-2540"])
        corpora_names: Human-readable law names (preferred over IDs in prompts)
        synthesis_mode: comparison/routing/aggregation

    Returns:
        Suggested name or description string.
    """
    if corpora_names and len(corpora_names) == len(corpora_ids):
        laws_str = ", ".join(corpora_names)
    else:
        laws_str = ", ".join(corpora_ids)

    mode_desc = _MODE_DESCRIPTIONS.get(synthesis_mode, synthesis_mode)

    if suggest_type == "name":
        prompt = (
            f"Du er navngiver for en juridisk test-suite. "
            f"Suiten tester et RAG-system der {mode_desc}.\n"
            f"Love: {laws_str}\n\n"
            f"Foreslå et kort, præcist dansk navn (2-5 ord) der beskriver "
            f"det juridiske emneområde lovene dækker sammen. "
            f"Brug fagtermer, ikke generiske ord. "
            f"Eksempler på gode navne: 'Juridisk Cybervurdering', "
            f"'Digital Markedsregulering', 'AI-Compliance Krydscheck'.\n"
            f"Svar KUN med navnet."
        )
        max_tokens = 50
    else:
        prompt = (
            f"Du skriver en kort beskrivelse af en juridisk test-suite.\n"
            f"Love: {laws_str}\n"
            f"Modus: {mode_desc}\n\n"
            f"Skriv 1-2 sætninger på dansk der forklarer hvad suiten tester "
            f"og hvilke juridiske aspekter der er i spil. Vær konkret om "
            f"lovenes indhold — nævn specifikke krav, rettigheder eller "
            f"forpligtelser der sammenlignes.\n"
            f"Svar KUN med beskrivelsen."
        )
        max_tokens = 150

    result = call_generation_llm(
        prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=max_tokens
    )

    if result is None:
        names = corpora_names or corpora_ids
        short_names = [n.split("(")[0].strip() for n in names]
        if suggest_type == "name":
            return " & ".join(short_names[:3])
        return f"Tester {mode_desc} for {', '.join(short_names)}."

    return result.strip().strip('"').strip("'")
