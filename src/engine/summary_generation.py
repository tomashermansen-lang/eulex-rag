"""Chapter and article summary generation.

Single Responsibility: Generate summaries for document sections.
Separate from structured answer generation - simpler prompts, no JSON validation.

Best practices applied:
- Explicit grounding constraints
- Clear output format specification
- Consistent indentation (no whitespace in f-strings)
"""

from __future__ import annotations

from typing import Any

from .types import RAGEngineError


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
