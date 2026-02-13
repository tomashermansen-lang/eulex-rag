"""Prompt building for RAG pipeline.

This module handles Stage 5 of the RAG pipeline: building the LLM prompt
from selected chunks. It transforms retrieved/selected chunks into:
- Structured references for payload
- KILDER: block for source attribution
- Context string for the LLM prompt
- LLM prompt construction with profile-specific formatting

Single Responsibility: Build LLM prompts and context from selected chunks.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Dict, Callable, TYPE_CHECKING

from .types import UserProfile
from .planning import Intent, QueryContext, RetrievalPlan, FocusSelection
from . import prompt_templates as PT

if TYPE_CHECKING:
    from .retrieval_pipeline import SelectedChunk


@dataclass
class PromptContext:
    """Result of the prompt building stage.

    Contains all data needed to construct the LLM prompt and populate
    the response payload with reference information.
    """
    included: List[Tuple[str, Dict[str, Any], str, str | None]]
    references: List[str]
    context_blocks: List[str]
    references_structured: List[Dict[str, Any]]
    context_string: str
    kilder_block: str
    citable_count: int
    raw_context_anchors: List[str]
    debug: Dict[str, Any] = field(default_factory=dict)


def _derive_structural_fields(meta: Dict[str, Any]) -> Dict[str, str]:
    """Extract structural fields from location_id if not in metadata.

    Parses location_id format: chapter:X/section:Y/article:Z/paragraph:N/...
    """
    out: Dict[str, str] = {}
    location_id = str(meta.get("location_id") or "").strip()
    if not location_id:
        return out

    parts = location_id.split("/")
    for part in parts:
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        key = key.strip().lower()
        val = val.strip()
        if key in ("chapter", "section", "article", "paragraph", "annex", "recital", "litra", "annex_point", "annex_section", "annex_subpoint"):
            if val and key not in out:
                out[key] = val
    return out


def _extract_raw_anchors_from_meta(meta: Dict[str, Any]) -> List[str]:
    """Extract raw anchor strings from metadata."""
    out: List[str] = []
    art = meta.get("article")
    rec = meta.get("recital")
    ann = meta.get("annex")
    if art:
        out.append(f"article:{str(art).strip().lower()}")
    if rec:
        out.append(f"recital:{str(rec).strip().lower()}")
    if ann:
        out.append(f"annex:{str(ann).strip().lower()}")
    return out


def _sandwich_order(items: List[Any]) -> List[Any]:
    """Reorder items to place most relevant at start and end (attention optimization).

    Addresses "Lost in the Middle" problem where LLMs focus on start/end of context.
    Items are assumed to be pre-sorted by relevance (highest first).

    Input:  [1, 2, 3, 4, 5, 6, 7, 8] (sorted by relevance, highest first)
    Output: [1, 3, 5, 7, 8, 6, 4, 2]

    Effect: Items 1,2 (most relevant) end up at start/end.
            Items 7,8 (least relevant) end up in the middle.

    Args:
        items: List of items sorted by relevance (highest first)

    Returns:
        Reordered list with most relevant items at start and end
    """
    if len(items) <= 3:
        return list(items)

    result: List[Any] = []
    # Even indices first (0, 2, 4, ...)
    for i in range(0, len(items), 2):
        result.append(items[i])
    # Odd indices in reverse order (..., 3, 1)
    for i in range(len(items) - 1 if len(items) % 2 == 0 else len(items) - 2, 0, -2):
        result.append(items[i])

    return result


def build_references_structured(
    included: List[Tuple[str, Dict[str, Any], str, str | None]],
    format_metadata_fn: Callable[[Dict[str, Any], str], Tuple[str, Dict[str, Any], Dict[str, Any]]],
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Build structured references from included chunks.

    Args:
        included: List of (doc, meta, chunk_id, precise_override) tuples
        format_metadata_fn: Function to format metadata -> (reference_str, sanitized_meta, validation)

    Returns:
        Tuple of (references, context_blocks, references_structured_all)
        - references: List of "[idx] reference" strings
        - context_blocks: List of "[idx] reference\\ndoc" strings
        - references_structured_all: List of structured reference dicts
    """
    references: List[str] = []
    context_blocks: List[str] = []
    references_structured_all: List[Dict[str, Any]] = []

    for idx, (doc, meta, chunk_id, precise_override) in enumerate(included, start=1):
        reference, sanitized_meta, validation = format_metadata_fn(meta, doc)
        derived = _derive_structural_fields(dict(meta or {}))

        article_val = sanitized_meta.get("article") or derived.get("article")
        paragraph_val = sanitized_meta.get("paragraph") or derived.get("paragraph")
        litra_val = sanitized_meta.get("litra") or derived.get("litra")
        annex_val = (meta or {}).get("annex") or derived.get("annex")
        annex_point_val = (meta or {}).get("annex_point") or derived.get("annex_point")
        annex_section_val = (meta or {}).get("annex_section") or derived.get("annex_section")
        recital_val = (meta or {}).get("recital") or derived.get("recital")
        chapter_val = (meta or {}).get("chapter") or derived.get("chapter")
        section_val = (meta or {}).get("section") or derived.get("section")

        references.append(f"[{idx}] {reference}")
        context_blocks.append(f"[{idx}] {reference}\n{doc}")
        references_structured_all.append(
            {
                "idx": idx,
                "chunk_id": chunk_id,
                "display": reference,
                "chunk_text": doc,
                "precise_ref": precise_override,
                "missing_ref": False,
                "validation": validation,
                "source": meta.get("source"),
                "corpus_id": meta.get("corpus_id"),
                "doc_id": meta.get("doc_id"),
                "location_id": meta.get("location_id"),
                "heading_path": meta.get("heading_path"),
                "heading_path_display": meta.get("heading_path_display"),
                "toc_path": meta.get("toc_path"),
                "chapter": chapter_val,
                "section": section_val,
                "article": article_val,
                "paragraph": paragraph_val,
                "litra": litra_val,
                "annex": annex_val,
                "annex_point": annex_point_val,
                "annex_section": annex_section_val,
                "recital": recital_val,
                "page": meta.get("page"),
                "title": meta.get("title"),
                "article_title": meta.get("article_title"),
                "chapter_title": meta.get("chapter_title"),
                "section_title": meta.get("section_title"),
                "annex_title": meta.get("annex_title"),
                # Semantic enrichment metadata
                "contextual_description": meta.get("contextual_description"),
                "enrichment_terms": meta.get("enrichment_terms"),
                "roles": meta.get("roles"),
            }
        )

    return references, context_blocks, references_structured_all


def anchor_label_for_prompt(r: Dict[str, Any]) -> str:
    """Build anchor label for KILDER block in prompt.

    Includes paragraph/stk. number to help LLM distinguish between
    different paragraphs of the same article (critical for citation accuracy).
    """
    if not isinstance(r, dict):
        return "(ukendt)"

    # Build base label
    base = ""
    if r.get("article"):
        base = f"Artikel {str(r.get('article')).strip()}"
    elif r.get("annex"):
        base = f"Bilag {str(r.get('annex')).strip()}"
    elif r.get("recital"):
        base = f"Betragtning {str(r.get('recital')).strip()}"

    if base:
        # Add structural details for granular citation matching
        # For annexes: add point number (annex_point)
        annex_point = r.get("annex_point")
        if annex_point:
            base += f", punkt {str(annex_point).strip()}"
            # Add subpoint if available
            annex_subpoint = r.get("annex_subpoint")
            if annex_subpoint:
                base += f", {str(annex_subpoint).strip()}"

        # For articles: add paragraph/stk. number
        para = r.get("paragraph")
        if para:
            base += f", stk. {str(para).strip()}"
            # Add litra if available
            litra = r.get("litra")
            if litra:
                base += f", litra {str(litra).strip()}"
        return base

    # Fallbacks (deterministic, no free text).
    for k in ["precise_ref", "location_id", "heading_path", "display"]:
        v = r.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "(ukendt)"


def excerpt_for_prompt(r: Dict[str, Any]) -> str:
    """Build rich excerpt from chunk metadata for KILDER block.

    Combines available metadata to give LLM maximum semantic context:
    - heading_path_display: Structural context with titles
    - contextual_description: Semantic summary of content
    - enrichment_terms: Search terms in everyday language
    - roles: Juridical function classification
    """
    if not isinstance(r, dict):
        return "<ukendt>"

    parts: List[str] = []

    # 1. Heading path display (structural context with titles)
    heading_display = r.get("heading_path_display")
    if heading_display and isinstance(heading_display, str) and heading_display.strip():
        parts.append(heading_display.strip())

    # 2. Contextual description (semantic summary)
    ctx_desc = r.get("contextual_description")
    if ctx_desc and isinstance(ctx_desc, str) and ctx_desc.strip():
        parts.append(ctx_desc.strip())

    # 3. Enrichment terms (everyday search terms)
    terms = r.get("enrichment_terms")
    if terms and isinstance(terms, list) and len(terms) > 0:
        terms_str = ", ".join(str(t) for t in terms[:5] if t)
        if terms_str:
            parts.append(f"Søgetermer: {terms_str}")

    # 4. Roles (juridical function)
    roles = r.get("roles")
    if roles and isinstance(roles, list) and len(roles) > 0:
        roles_str = ", ".join(str(role) for role in roles if role)
        if roles_str:
            parts.append(f"Roller: {roles_str}")

    # If we have metadata, join with " | "
    if parts:
        result = " | ".join(parts)
        return result[:400]  # Cap total length

    # Fallback to raw text excerpt
    raw = r.get("chunk_text")
    if isinstance(raw, str) and raw.strip():
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        excerpt = " / ".join(lines[:2]).strip()
        if excerpt:
            return excerpt[:260]

    return f"<{anchor_label_for_prompt(r)}>"


def build_kilder_block(
    references_structured_all: List[Dict[str, Any]],
    corpus_id: str,
) -> str:
    """Build the KILDER: block for the prompt.

    Args:
        references_structured_all: List of structured reference dicts
        corpus_id: Default corpus ID to use if not in reference

    Returns:
        kilder_block string (empty string if no references)
    """
    kilder_lines: List[str] = []

    if not references_structured_all:
        return ""

    kilder_lines.append("KILDER:")
    for r in sorted(
        [x for x in list(references_structured_all or []) if isinstance(x, dict)],
        key=lambda d: int(d.get("idx") or 10**9),
    ):
        if not isinstance(r, dict):
            continue
        try:
            ridx = int(r.get("idx"))
        except Exception:  # noqa: BLE001
            continue

        corpus = str(r.get("corpus_id") or r.get("source") or corpus_id or "").strip()
        if not corpus:
            corpus = "(ukendt)"

        anchor_label = anchor_label_for_prompt(r)
        excerpt = excerpt_for_prompt(r)
        kilder_lines.append(f"- [{ridx}] {corpus} / {anchor_label} — {excerpt}")

    return "\n".join(kilder_lines).strip()


def build_context_string(
    context_blocks: List[str],
    kilder_block: str,
) -> str:
    """Combine kilder_block and context_blocks into final context string.

    Args:
        context_blocks: List of "[idx] reference\\ndoc" strings
        kilder_block: KILDER: block string (may be empty)

    Returns:
        Combined context string for the prompt
    """
    if kilder_block:
        return f"{kilder_block}\n\n" + "\n\n".join(context_blocks)
    else:
        return "\n\n".join(context_blocks)


def build_prompt_context(
    selected: "Tuple[SelectedChunk, ...]",
    format_metadata_fn: Callable[[Dict[str, Any], str], Tuple[str, Dict[str, Any], Dict[str, Any]]],
    corpus_id: str,
    *,
    enable_raw_anchor_log: bool = False,
    context_positioning: str = "sandwich",
) -> PromptContext:
    """Build LLM prompt context from selected chunks.

    This is Stage 5 of the RAG pipeline, transforming selected chunks into
    the format needed for LLM prompt construction.

    Args:
        selected: Tuple of SelectedChunk from context selection stage
        format_metadata_fn: Function (meta, doc) -> (reference_str, sanitized_meta, validation)
        corpus_id: Default corpus ID for kilder_block
        enable_raw_anchor_log: Whether to extract and log raw anchors
        context_positioning: Strategy for ordering chunks in context.
            - "sandwich": Place most relevant chunks at start and end (default)
            - "relevance": Keep original relevance order (highest first)
            - "none": No reordering

    Returns:
        PromptContext with all prompt building outputs
    """
    # Convert SelectedChunk tuples to the included format
    included: List[Tuple[str, Dict[str, Any], str, str | None]] = []
    for sel in selected:
        chunk = sel.chunk
        included.append((
            chunk.document,
            dict(chunk.metadata),
            chunk.chunk_id,
            sel.precise_ref,
        ))

    # Apply context positioning strategy
    if context_positioning == "sandwich" and len(included) > 3:
        included = _sandwich_order(included)

    # Extract raw anchors if requested
    raw_context_anchors: List[str] = []
    if enable_raw_anchor_log:
        for _, meta, _, _ in included:
            raw_context_anchors.extend(_extract_raw_anchors_from_meta(meta))
        raw_context_anchors = sorted(set(raw_context_anchors))

    # Build references and context blocks
    references, context_blocks, references_structured = build_references_structured(
        included=included,
        format_metadata_fn=format_metadata_fn,
    )

    # Build kilder_block and context_string
    kilder_block = build_kilder_block(
        references_structured_all=references_structured,
        corpus_id=corpus_id,
    )
    context_string = build_context_string(context_blocks, kilder_block)

    # Build debug info
    debug: Dict[str, Any] = {
        "citable_count": len(included),
        "context_positioning": context_positioning,
    }
    if enable_raw_anchor_log:
        debug["raw_context_anchors_count"] = len(raw_context_anchors)
        debug["raw_context_anchors_top"] = raw_context_anchors[:20]

    return PromptContext(
        included=included,
        references=references,
        context_blocks=context_blocks,
        references_structured=references_structured,
        context_string=context_string,
        kilder_block=kilder_block,
        citable_count=len(included),
        raw_context_anchors=raw_context_anchors,
        debug=debug,
    )


# ---------------------------------------------------------------------------
# LLM Prompt Construction Functions
# (Moved from generation.py as part of SOLID refactoring)
# ---------------------------------------------------------------------------


def build_prompt(
    *,
    ctx: QueryContext,
    plan: RetrievalPlan,
    context: str,
    focus_block: str,
    contract_min_citations: int | None = None,
    legal_json_mode: bool = False,
    history_context: str = "",
) -> str:
    """Return the LLM prompt for a given profile + intent.

    The engine is responsible for providing context blocks and citations separately.

    Args:
        ctx: Query context with user profile and question.
        plan: Retrieval plan with intent.
        context: Formatted context string with sources.
        focus_block: Focus block for prompt.
        contract_min_citations: Minimum required citations from contract.
        legal_json_mode: If True, use JSON output format for LEGAL profile.
        history_context: Formatted conversation history string (optional).
    """
    # Default minimum citations if not specified
    min_cit = contract_min_citations if contract_min_citations is not None else 2

    # Use centralized grounding rules from prompt_templates
    common = PT.COMMON_GROUNDING_RULES

    if ctx.user_profile == UserProfile.LEGAL:
        if legal_json_mode:
            # LEGAL JSON mode: structured output for machine-testability
            format_rules = PT.LEGAL_JSON_FORMAT.format(min_citations=min_cit)
        else:
            # LEGAL prose mode: traditional structured text output with markdown
            format_rules = PT.LEGAL_PROSE_FORMAT.format(min_citations=min_cit)
        if plan.intent in {Intent.CHAPTER_SUMMARY, Intent.ARTICLE_SUMMARY}:
            format_rules += PT.LEGAL_SUMMARY_SUFFIX

    else:
        engineering_json_mode = str(os.getenv("ENGINEERING_JSON_MODE", "") or "").strip().lower() in {"1", "true", "yes", "on"}

        if engineering_json_mode:
            # For ENGINEERING JSON-mode: enforce a strict machine-validated schema.
            format_rules = PT.ENGINEERING_JSON_FORMAT
        else:
            # For ENGINEERING: request a structured technical answer with markdown
            format_rules = PT.ENGINEERING_PROSE_FORMAT.format(min_citations=min_cit)

    task = ""
    if plan.intent == Intent.CHAPTER_SUMMARY:
        ch = (ctx.focus.chapter if ctx.focus else None) or str((plan.where or {}).get("chapter") or "")
        task = PT.CHAPTER_SUMMARY_TASK.format(chapter=ch)
    elif plan.intent == Intent.ARTICLE_SUMMARY:
        art = (ctx.focus.article if ctx.focus else None) or str((plan.where or {}).get("article") or "")
        task = PT.ARTICLE_SUMMARY_TASK.format(article=art)
    elif plan.intent == Intent.STRUCTURE:
        task = PT.STRUCTURE_TASK

    # Build history section if provided (simplified - query rewriting handles context)
    history_section = ""
    if history_context and history_context.strip():
        history_section = PT.HISTORY_CONTEXT_TEMPLATE.format(history_context=history_context)

    # Use centralized prompt assembly template
    return PT.PROMPT_TEMPLATE.format(
        common=common,
        user_profile=ctx.user_profile.value,
        intent=plan.intent.value,
        task=task,
        format_rules=format_rules,
        history_section=history_section,
        focus_block=focus_block,
        context=context,
        question=ctx.question,
    )


def build_disclaimer(*, ctx: QueryContext, low_evidence: bool) -> str:
    """Build disclaimer text for low evidence situations."""
    if ctx.user_profile == UserProfile.LEGAL:
        return ""
    if low_evidence:
        return "Lav evidens: svaret er baseret på svagt matchende uddrag og kan kræve manuel verifikation."
    return ""


def focus_block_for_prompt(focus: FocusSelection | None) -> str:
    """Build focus block for prompt from FocusSelection."""
    if not focus:
        return ""
    parts: list[str] = ["FOCUS (hard constraint):"]
    parts.append(f"- type: {focus.type.value}")
    if focus.title:
        parts.append(f"- title: {focus.title}")
    if focus.chapter:
        parts.append(f"- chapter: {focus.chapter}")
    if focus.section:
        parts.append(f"- section: {focus.section}")
    if focus.article:
        parts.append(f"- article: {focus.article}")
    if focus.annex:
        parts.append(f"- annex: {focus.annex}")
    if focus.recital:
        parts.append(f"- recital: {focus.recital}")
    return "\n".join(parts)


def build_answer_policy_suffix(
    answer_policy: Any | None,
    user_profile: UserProfile,
) -> str:
    """Build SVAR-POLICY suffix for ENGINEERING prompts.

    Args:
        answer_policy: The answer policy object (may have intent_category,
                       requirements_first, include_audit_evidence, min_section3_bullets).
        user_profile: The user profile.

    Returns:
        Prompt suffix string (empty if not applicable).
    """
    if user_profile != UserProfile.ENGINEERING or answer_policy is None:
        return ""

    try:
        intent_category = str(getattr(answer_policy, "intent_category", "") or "").strip().upper()
        requirements_first = bool(getattr(answer_policy, "requirements_first", False))
        include_audit = bool(getattr(answer_policy, "include_audit_evidence", False))
        min_bullets = getattr(answer_policy, "min_section3_bullets", None)

        lines = ["\n\nSVAR-POLICY (ENGINEERING):"]
        lines.append(f"- intent_category: {intent_category}")

        if requirements_first and intent_category == "REQUIREMENTS":
            lines.append("- requirements_first: Når der er tvetydighed, SKAL svaret være krav-/implementeringsorienteret (Sektion 3) fremfor håndhævelse/sanktioner.")

        if min_bullets is not None:
            try:
                mb = int(min_bullets)
                if mb > 0:
                    lines.append(f"- Sektion 3 SKAL have mindst {mb} bullets.")
            except Exception:  # noqa: BLE001
                pass

        if include_audit:
            lines.append("- Under Sektion 3: tilføj undersektion 'Minimum ved tilsyn (evidens/artefakter)'. Hvis utilstrækkelig evidens, brug UTILSTRÆKKELIG_EVIDENS.")

        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return ""


def build_citation_requirement_suffix(
    user_profile: UserProfile,
    contract_min_citations: int | None,
    references_structured_all: list[dict[str, Any]],
    json_mode: bool,
) -> str:
    """Build CITATION-KRAV suffix for prompts.

    Args:
        user_profile: The user profile.
        contract_min_citations: Minimum citations required.
        references_structured_all: All structured references.
        json_mode: Whether JSON mode is enabled.

    Returns:
        Prompt suffix string (empty if not applicable).
    """
    if user_profile != UserProfile.ENGINEERING or contract_min_citations is None:
        return ""

    try:
        min_cit = int(contract_min_citations)
    except Exception:  # noqa: BLE001
        return ""

    if min_cit <= 0:
        return ""

    allowed_idxs: set[int] = set()
    for r in list(references_structured_all or []):
        if not isinstance(r, dict):
            continue
        try:
            allowed_idxs.add(int(r.get("idx")))
        except Exception:  # noqa: BLE001
            continue

    if len(allowed_idxs) < min_cit:
        return ""

    allowed_marks = " ".join(f"[{i}]" for i in sorted(allowed_idxs))

    if json_mode:
        return (
            "\n\n"
            "CITATION-KRAV (maskin-check, JSON):\n"
            f"- Minimum unikke citations (union): {min_cit}\n"
            f"- Tilladte idx: {allowed_marks}\n"
            "- citations-arrays må KUN indeholde disse idx.\n"
        )
    else:
        return (
            "\n\n"
            "CITATION-KRAV (maskin-check):\n"
            f"- Minimum unikke citations: {min_cit}\n"
            f"- Tilladte citations: {allowed_marks}\n"
            "- Format SKAL være præcis [n] (f.eks. [1]).\n"
            "- Citationsmarkører SKAL stå i selve svaret (inline). Tilføj IKKE en 'KILDER'/'REFERENCER' sektion i svaret.\n"
            "- Hvis et punkt er UTILSTRÆKKELIG_EVIDENS, tilføj stadig [n] der viser hvilke kilder der blev vurderet.\n"
        )


# ---------------------------------------------------------------------------
# Multi-Corpus / Cross-Law Prompt Functions
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from .corpus_resolver import CorpusResolver
    from .synthesis_router import SynthesisMode


def get_corpus_display_name(
    corpus_id: str,
    resolver: "CorpusResolver | None" = None,
) -> str:
    """Get display name for corpus, using resolver if available.

    Args:
        corpus_id: The corpus identifier (e.g., "ai_act", "gdpr")
        resolver: Optional CorpusResolver for looking up display names

    Returns:
        Human-readable display name (e.g., "AI-Act", "GDPR")
    """
    if resolver is not None:
        name = resolver.display_name_for(corpus_id)
        if name:
            return name
    # Fallback: uppercase with underscore-to-dash normalization
    if not corpus_id:
        return "(ukendt)"
    return corpus_id.upper().replace("_", "-")


def _corpus_display_name(corpus_id: str, resolver: "CorpusResolver | None" = None) -> str:
    """Get display name for corpus ID (internal helper)."""
    return get_corpus_display_name(corpus_id, resolver)


def format_reference_display(
    ref: Dict[str, Any],
    include_corpus: bool = False,
) -> str:
    """Format a reference for display with optional corpus prefix.

    Args:
        ref: Reference dict with article/recital/annex and optional corpus_id
        include_corpus: Whether to include corpus prefix

    Returns:
        Formatted reference string like "AI-Act, Artikel 6, stk. 1"
    """
    parts: List[str] = []

    if include_corpus:
        corpus_id = ref.get("corpus_id") or ref.get("source")
        if corpus_id:
            parts.append(_corpus_display_name(str(corpus_id)))

    # Build anchor part
    if ref.get("article"):
        anchor = f"Artikel {ref['article']}"
        if ref.get("paragraph"):
            anchor += f", stk. {ref['paragraph']}"
        if ref.get("litra"):
            anchor += f", litra {ref['litra']}"
        parts.append(anchor)
    elif ref.get("annex"):
        anchor = f"Bilag {ref['annex']}"
        if ref.get("annex_point"):
            anchor += f", punkt {ref['annex_point']}"
        parts.append(anchor)
    elif ref.get("recital"):
        parts.append(f"Betragtning {ref['recital']}")

    return ", ".join(parts) if parts else "(ukendt)"


def anchor_label_for_prompt_multi_corpus(
    ref: Dict[str, Any],
    include_corpus: bool = False,
) -> str:
    """Build anchor label with optional corpus prefix.

    Args:
        ref: Reference dict
        include_corpus: Whether to include corpus prefix

    Returns:
        Label string like "AI-Act, Artikel 6, stk. 1"
    """
    return format_reference_display(ref, include_corpus=include_corpus)


def build_aggregation_prompt(
    question: str,
    chunks_by_corpus: Dict[str, List[Dict[str, Any]]],
    user_profile: str,
) -> str:
    """Build prompt for aggregation synthesis mode.

    Groups evidence by corpus/law and instructs the LLM to synthesize
    across all sources with per-law citations.

    Args:
        question: User's question
        chunks_by_corpus: Dict mapping corpus_id to list of chunk refs
        user_profile: LEGAL or ENGINEERING

    Returns:
        Prompt string
    """
    lines: List[str] = []
    lines.append("Du skal besvare et spørgsmål baseret på flere EU-retsakter.")
    lines.append("Identificer hvad HVER lov siger om emnet og giv et samlet svar.")
    lines.append("Citer ALTID den specifikke lov sammen med artiklen (f.eks. [1] AI-Act, Artikel 5).")
    lines.append("")
    lines.append("KILDER PER LOV:")

    for corpus_id, chunks in sorted(chunks_by_corpus.items()):
        display_name = _corpus_display_name(corpus_id)
        lines.append(f"\n## {display_name}")

        for chunk in chunks:
            idx = chunk.get("idx", "?")
            anchor = format_reference_display(chunk, include_corpus=False)
            excerpt = str(chunk.get("chunk_text", ""))[:200]
            lines.append(f"[{idx}] {anchor}: {excerpt}...")

    lines.append("")
    lines.append(f"SPØRGSMÅL: {question}")
    lines.append("")
    lines.append("Besvar spørgsmålet og sammenfat hvad HVER lov siger. Citer kilder fra hver relevant lov.")

    return "\n".join(lines)


def build_comparison_prompt(
    question: str,
    corpus_a_name: str,
    corpus_a_chunks: List[Dict[str, Any]],
    corpus_b_name: str,
    corpus_b_chunks: List[Dict[str, Any]],
    user_profile: str,
) -> str:
    """Build prompt for comparison synthesis mode.

    Creates distinct sections for each law being compared and instructs
    the LLM to identify similarities and differences.

    Args:
        question: User's question
        corpus_a_name: Display name of first corpus
        corpus_a_chunks: Chunks from first corpus
        corpus_b_name: Display name of second corpus
        corpus_b_chunks: Chunks from second corpus
        user_profile: LEGAL or ENGINEERING

    Returns:
        Prompt string
    """
    lines: List[str] = []
    lines.append("Du skal sammenligne to EU-retsakter og identificere ligheder og forskelle.")
    lines.append("Strukturer dit svar med: 1) Ligheder, 2) Forskelle, 3) Konklusion.")
    lines.append("Citer ALTID den specifikke lov sammen med artiklen.")
    lines.append("")

    # First law section
    lines.append(f"## {corpus_a_name}")
    for chunk in corpus_a_chunks:
        idx = chunk.get("idx", "?")
        anchor = format_reference_display(chunk, include_corpus=False)
        excerpt = str(chunk.get("chunk_text", ""))[:200]
        lines.append(f"[{idx}] {anchor}: {excerpt}...")

    lines.append("")

    # Second law section
    lines.append(f"## {corpus_b_name}")
    for chunk in corpus_b_chunks:
        idx = chunk.get("idx", "?")
        anchor = format_reference_display(chunk, include_corpus=False)
        excerpt = str(chunk.get("chunk_text", ""))[:200]
        lines.append(f"[{idx}] {anchor}: {excerpt}...")

    lines.append("")
    lines.append(f"SPØRGSMÅL: {question}")
    lines.append("")
    lines.append(f"Sammenlign {corpus_a_name} og {corpus_b_name}. Identificer ligheder og forskelle.")

    return "\n".join(lines)


def build_routing_prompt(
    question: str,
    corpus_matches: Dict[str, int],
    user_profile: str,
) -> str:
    """Build prompt for routing synthesis mode.

    Creates a lightweight prompt for identifying which law(s) cover a topic.

    Args:
        question: User's question
        corpus_matches: Dict mapping corpus_id to match count
        user_profile: LEGAL or ENGINEERING

    Returns:
        Prompt string (kept concise for routing)
    """
    lines: List[str] = []
    lines.append("Identificer hvilke(n) EU-retsakt(er) der dækker det stillede emne.")
    lines.append("Giv et kort svar med de relevante love og en kort begrundelse.")
    lines.append("")
    lines.append("RELEVANS PER LOV:")

    for corpus_id, count in sorted(corpus_matches.items(), key=lambda x: -x[1]):
        display_name = _corpus_display_name(corpus_id)
        relevance = "Høj" if count >= 5 else "Medium" if count >= 2 else "Lav" if count >= 1 else "Ingen"
        lines.append(f"- {display_name}: {relevance} ({count} relevante uddrag)")

    lines.append("")
    lines.append(f"SPØRGSMÅL: {question}")
    lines.append("")
    lines.append("Besvar kort: Hvilke(n) lov(e) er mest relevante og hvorfor?")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-Corpus Prompt Dispatcher
# ---------------------------------------------------------------------------


def _group_refs_by_corpus(references: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group references by corpus_id."""
    by_corpus: Dict[str, List[Dict[str, Any]]] = {}
    for ref in references:
        corpus = str(ref.get("corpus_id") or ref.get("source") or "unknown")
        if corpus not in by_corpus:
            by_corpus[corpus] = []
        by_corpus[corpus].append(ref)
    return by_corpus


def _count_refs_by_corpus(references: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count references per corpus."""
    counts: Dict[str, int] = {}
    for ref in references:
        corpus = str(ref.get("corpus_id") or ref.get("source") or "unknown")
        counts[corpus] = counts.get(corpus, 0) + 1
    return counts


def _get_format_rules_for_profile(user_profile: str) -> str:
    """Get format rules based on user profile."""
    profile_upper = str(user_profile or "").upper()
    if profile_upper == "LEGAL":
        return PT.LEGAL_PROSE_FORMAT
    elif profile_upper == "ENGINEERING":
        return PT.ENGINEERING_PROSE_FORMAT
    return PT.LEGAL_PROSE_FORMAT  # Default


def build_multi_corpus_prompt(
    *,
    mode: "SynthesisMode",
    question: str,
    context: str,
    kilder_block: str,
    references_structured: List[Dict[str, Any]],
    user_profile: str,
    resolver: "CorpusResolver | None" = None,
    comparison_corpora: List[str] | None = None,
    **kwargs: Any,
) -> str:
    """Build prompt for multi-corpus synthesis with grounding rules.

    Dispatches to appropriate mode-specific instructions while wrapping
    with COMMON_GROUNDING_RULES for fail-closed behavior.

    Args:
        mode: SynthesisMode enum (UNIFIED, AGGREGATION, COMPARISON, ROUTING)
        question: User's question
        context: Combined context string (kilder_block + context_blocks)
        kilder_block: KILDER: block (included in context)
        references_structured: List of reference dicts with corpus_id
        user_profile: "LEGAL" or "ENGINEERING"
        resolver: Optional CorpusResolver for display names
        comparison_corpora: For COMPARISON mode, the two corpora being compared
        **kwargs: Additional arguments (ignored)

    Returns:
        Complete prompt string with grounding rules
    """
    # Import here to avoid circular imports
    from .synthesis_router import SynthesisMode

    # 1. Get mode-specific instructions
    if mode == SynthesisMode.AGGREGATION:
        mode_instructions = PT.AGGREGATION_MODE_INSTRUCTIONS
    elif mode == SynthesisMode.COMPARISON:
        mode_instructions = PT.COMPARISON_MODE_INSTRUCTIONS
    elif mode == SynthesisMode.ROUTING:
        mode_instructions = PT.ROUTING_MODE_INSTRUCTIONS
    else:  # UNIFIED (default)
        mode_instructions = PT.UNIFIED_MODE_INSTRUCTIONS

    # 2. Get format rules based on profile
    format_rules = _get_format_rules_for_profile(user_profile)

    # 3. Assemble with grounding rules
    return PT.MULTI_CORPUS_PROMPT_TEMPLATE.format(
        common=PT.COMMON_GROUNDING_RULES,
        mode_instructions=mode_instructions,
        format_rules=format_rules,
        context=context,
        question=question,
    )


# ---------------------------------------------------------------------------
# Discovery Preamble
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATE_AUTO = "Baseret på dit spørgsmål har jeg fundet relevante bestemmelser i {law_names}."
_DEFAULT_TEMPLATE_SUGGEST = "Dit spørgsmål kan relatere sig til {law_names}. Svaret kan være ufuldstændigt."


def build_discovery_preamble(
    gate: str,
    matches: list[dict[str, Any]],
    template_auto: str | None = None,
    template_suggest: str | None = None,
) -> str:
    """Build discovery transparency text for the answer preamble.

    Args:
        gate: Discovery gate ("AUTO", "SUGGEST", "ABSTAIN").
        matches: List of match dicts with corpus_id, confidence, display_name.
        template_auto: Optional custom template for AUTO gate.
        template_suggest: Optional custom template for SUGGEST gate.

    Returns:
        Preamble string, or empty string for ABSTAIN gate.
    """
    if gate == "ABSTAIN" or not matches:
        return ""

    law_names = ", ".join(m.get("display_name", m["corpus_id"]) for m in matches)

    if gate == "AUTO":
        template = template_auto or _DEFAULT_TEMPLATE_AUTO
    else:
        template = template_suggest or _DEFAULT_TEMPLATE_SUGGEST

    return template.format(law_names=law_names)
