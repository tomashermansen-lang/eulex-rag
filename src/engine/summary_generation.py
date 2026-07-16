"""Chapter and article summary generation.

Single Responsibility: Generate summaries for document sections.
Separate from structured answer generation - simpler prompts, no JSON validation.

Best practices applied:
- Explicit grounding constraints
- Clear output format specification
- Consistent indentation (no whitespace in f-strings)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List

from . import citations
from . import query_helpers
from .rag_config import _get_default_temperature, _get_rag_settings
from .types import RAGEngineError

logger = logging.getLogger(__name__)


# Summary prompt template - extracted for consistency
_SUMMARY_PROMPT_TEMPLATE = """\
Du er en omhyggelig assistent.
Svar på dansk.

GROUNDING (KRITISK):
- Brug UDELUKKENDE den givne kontekst.
- Hvis konteksten ikke dækker et punkt, sig eksplicit at informationen mangler.
- Gæt ALDRIG - det er bedre at sige "utilstrækkelig information" end at opdigte.

OPGAVE:
Opsummér hvad Kapitel {chapter_ref} indeholder.

OUTPUTFORMAT:
### Formål
1-2 linjer: Kapitellets overordnede formål/tema.

### Hovedpunkter
5-10 bullets: de vigtigste punkter/krav fra konteksten.

### Praktiske konsekvenser
2-5 bullets: hvem påvirkes, hvad skal de gøre, hvornår.

### Manglende information (hvis relevant)
List emner der ikke er dækket af konteksten.

KONTEKST:
{context}

SPØRGSMÅL:
{question}
"""


def generate_chapter_summary_from_chunks(
    *,
    client: Any,
    model: str,
    context: str,
    question: str,
    chapter_ref: str,
    temperature: float = 0.0,
) -> str:
    """Generate a summary for a specific chapter based on retrieved chunks.

    Args:
        client: OpenAI client instance.
        model: Model name to use.
        context: Formatted context string with retrieved chunks.
        question: User's question about the chapter.
        chapter_ref: Chapter reference (e.g., "III", "5").
        temperature: LLM temperature setting.

    Returns:
        Generated summary text.

    Raises:
        RAGEngineError: If OpenAI request fails.
    """
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(
        chapter_ref=chapter_ref,
        context=context,
        question=question,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return str(response.choices[0].message.content or "")
    except Exception as exc:
        raise RAGEngineError("OpenAI request failed.") from exc


# TOC-aware summary prompt template
_TOC_SUMMARY_PROMPT_TEMPLATE = """\
Du er en omhyggelig assistent.
Svar på dansk.

GROUNDING (KRITISK):
- Brug UDELUKKENDE den givne kontekst og TOC-information.
- Hvis konteksten ikke dækker et punkt, sig eksplicit at informationen mangler.
- Gæt ALDRIG - det er bedre at sige "utilstrækkelig information" end at opdigte.

BRUGERVALGT TOC-NODE:
{toc_node_meta_display}
{toc_node_text}

OPGAVE:
Opsummér hvad Kapitel {chapter} indeholder baseret på konteksten.

OUTPUTFORMAT:
### Formål
1-2 linjer: Kapitellets overordnede formål/tema.

### Hovedpunkter
5-10 bullets: de vigtigste punkter/krav fra konteksten.

### Praktiske konsekvenser
2-5 bullets: hvem påvirkes, hvad skal de gøre, hvornår.

### Manglende information (hvis relevant)
List emner der ikke er dækket af konteksten.

KONTEKST:
{context}

SPØRGSMÅL:
{question}
"""


def generate_selected_chapter_summary(
    *,
    client: Any,
    model: str,
    context: str,
    question: str,
    chapter: str,
    toc_node_meta_display: str,
    toc_node_text: str,
    temperature: float = 0.0,
) -> str:
    """Generate a summary for a selected chapter node (TOC-aware).

    Args:
        client: OpenAI client instance.
        model: Model name to use.
        context: Formatted context string with retrieved chunks.
        question: User's question about the chapter.
        chapter: Chapter identifier.
        toc_node_meta_display: Display metadata for the TOC node.
        toc_node_text: Text content of the TOC node.
        temperature: LLM temperature setting.

    Returns:
        Generated summary text.

    Raises:
        RAGEngineError: If OpenAI request fails.
    """
    prompt = _TOC_SUMMARY_PROMPT_TEMPLATE.format(
        chapter=chapter,
        toc_node_meta_display=toc_node_meta_display,
        toc_node_text=toc_node_text,
        context=context,
        question=question,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return str(response.choices[0].message.content or "")
    except Exception as exc:
        raise RAGEngineError("OpenAI request failed.") from exc


# ---------------------------------------------------------------------------
# Step 6.7: Stage function extracted from RAGEngine._answer_chapter_summary_from_chunks
# ---------------------------------------------------------------------------


def answer_chapter_summary_stage(
    *,
    question: str,
    top_k: int,
    query_fn: Callable[..., tuple[list[tuple[str, dict[str, Any]]], list[float]]],
    client: Any,
    chat_model: str,
    collection_name: str | None,
    source_label_fn: Callable[[dict[str, Any]], str],
    hybrid_rerank_dict: dict[str, Any],
    retrieved_ids_fn: Callable[[], list[str]],
    retrieved_metadatas_fn: Callable[[], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Execute chapter summary pipeline stage.

    Detects chapter summary questions, retrieves scoped chunks,
    generates a summary, and returns a structured result dict.

    Returns None if the question is not a chapter summary question
    or if no relevant chunks are found.

    The result dict contains:
        - answer: str
        - references: list[str]
        - retrieval: dict (includes distances, query_where, etc.)
    """
    if not query_helpers._looks_like_chapter_summary_question(question):
        return None

    chapter_ref = query_helpers._extract_chapter_ref(question)
    if not chapter_ref:
        return None

    canonical = chapter_ref.upper()

    rag_cfg = _get_rag_settings()
    min_k = int(rag_cfg.get("chapter_summary_min_k", 8))
    fallback_k = int(rag_cfg.get("chapter_summary_fallback_k", 12))
    max_k = int(rag_cfg.get("chapter_summary_max_k", 20))

    try:
        k = max(min_k, int(top_k) * 4)
    except Exception:  # noqa: BLE001
        k = fallback_k

    try:
        hits, distances = query_fn(
            question=f"Sammenfat Kapitel {canonical}.",
            k=min(max_k, k),
            where={"chapter": canonical},
        )
    except Exception:  # noqa: BLE001
        return None

    if not hits:
        return None

    references: List[str] = []
    context_blocks: List[str] = []
    references_structured: list[dict[str, Any]] = []

    filtered_hits: list[tuple[str, dict[str, Any], str | None]] = []
    for doc, metadata in hits:
        meta_dict = dict(metadata or {})
        doc_str = str(doc or "")

        if citations._is_citable_metadata(meta_dict):
            filtered_hits.append((doc_str, meta_dict, None))
            continue

        extracted = citations.extract_precise_ref_from_text(doc_str)
        if extracted:
            src = source_label_fn(meta_dict)
            filtered_hits.append((doc_str, meta_dict, f"{src}, {extracted}"))

    if not filtered_hits:
        return None

    for idx, (doc, metadata, precise_override) in enumerate(filtered_hits, start=1):
        display = citations._format_metadata(metadata)
        precise = precise_override or citations.extract_precise_token_from_meta(
            dict(metadata or {})
        )
        missing = False
        if not precise:
            extracted = citations.extract_precise_ref_from_text(doc or "")
            if extracted:
                src = citations.best_effort_source_label(
                    dict(metadata or {}),
                    fallback=source_label_fn(dict(metadata or {})),
                )
                precise = f"{src}, {extracted}"
            else:
                src = citations.best_effort_source_label(
                    dict(metadata or {}),
                    fallback=source_label_fn(dict(metadata or {})),
                )
                precise = f"MISSING_REF — {src} (kilden mangler artikel/bilag i metadata og tekst)"
                missing = True

        references.append(f"[{idx}] {precise}")
        context_blocks.append(f"[{idx}] {display}\n{doc}")

        references_structured.append(
            {
                "idx": idx,
                "chunk_id": (metadata or {}).get("chunk_id") or f"hit-{idx}",
                "display": display,
                "precise_ref": precise,
                "missing_ref": missing,
                "source": (metadata or {}).get("source"),
                **dict(metadata or {}),
            }
        )

    context = "\n\n".join(context_blocks)

    answer_text = generate_chapter_summary_from_chunks(
        client=client,
        model=chat_model,
        context=context,
        question=question,
        chapter_ref=canonical,
        temperature=_get_default_temperature(),
    )

    return {
        "answer": answer_text,
        "references": references,
        "retrieval": {
            "distances": list(distances or []),
            "query_collection": collection_name,
            "query_where": {"chapter": canonical},
            "retrieved_ids": list(retrieved_ids_fn() or []),
            "retrieved_metadatas": list(retrieved_metadatas_fn() or []),
            "hybrid_rerank": hybrid_rerank_dict,
        },
    }
