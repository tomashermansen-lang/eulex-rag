"""Conversation context management for multi-turn RAG.

Responsibilities:
- Format conversation history for LLM prompt injection
- Rewrite ambiguous follow-up queries using conversation context (industry-standard approach)
- Extract last exchange for intent context augmentation
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Threshold below which a rewritten query is considered too short
# for standalone intent classification (requires context augmentation).
SHORT_QUERY_THRESHOLD: int = 40


@dataclass(frozen=True)
class HistoryMessage:
    """A single message in conversation history.

    Attributes:
        role: Either "user" or "assistant"
        content: The message text content
    """
    role: str
    content: str


def truncate_history(
    history: list[HistoryMessage] | None,
    max_messages: int = 10,
) -> list[HistoryMessage]:
    """Keep only the most recent messages.

    Args:
        history: List of history messages, or None
        max_messages: Maximum number of messages to keep (default 10 = 5 exchanges)

    Returns:
        List of the most recent messages, or empty list if input is None/empty
    """
    if not history:
        return []

    if len(history) <= max_messages:
        return list(history)

    # Keep the most recent messages
    return list(history[-max_messages:])


def format_history_for_prompt(
    history: list[HistoryMessage] | None,
    max_chars_per_message: int = 4000,
) -> str:
    """Format conversation history as a prompt section.

    Creates a formatted string suitable for injection into the LLM prompt,
    providing context from previous conversation turns.

    Args:
        history: List of history messages, or None
        max_chars_per_message: Maximum characters per message before truncation
            (default 4000 to preserve full answers with sources)

    Returns:
        Formatted history string, or empty string if no history
    """
    if not history:
        return ""

    lines = ["TIDLIGERE SAMTALE:"]

    for msg in history:
        role_label = "Bruger:" if msg.role == "user" else "Assistent:"
        content = msg.content

        # Truncate very long messages (but preserve most content)
        if len(content) > max_chars_per_message:
            content = content[:max_chars_per_message - 3] + "..."

        lines.append(f"{role_label} {content}")

    lines.append("---")
    lines.append("")  # Empty line before next section

    return "\n".join(lines)


def rewrite_query_for_retrieval(
    question: str,
    history: list[HistoryMessage] | None,
) -> str:
    """Rewrite an ambiguous follow-up query using conversation history.

    Industry-standard approach (used by Perplexity, ChatGPT, Copilot):
    Use a fast LLM call to expand the query BEFORE retrieval so that
    the vector search finds relevant documents.

    Examples:
        - "hvad med stk 4?" → "Hvad siger artikel 9, stk. 4 i GDPR?"
        - "uddyb dette" → "Uddyb artikel 9 om behandling af særlige kategorier"
        - "og bilag 1?" → "Hvad indeholder bilag 1 i AI-forordningen?"

    Args:
        question: The current user question (potentially ambiguous)
        history: Conversation history for context

    Returns:
        Rewritten query suitable for retrieval, or original question if
        no rewriting is needed.
    """
    # Skip rewriting if no history - nothing to contextualize
    if not history or len(history) == 0:
        return question

    # Skip rewriting for questions that are already self-contained
    # (contains specific legal references)
    question_lower = question.lower()
    has_specific_ref = any(
        pattern in question_lower
        for pattern in ["artikel ", "art. ", "bilag ", "annex ", "betragtning "]
    )
    if has_specific_ref and len(question) > 50:
        # Question already has specific references and is substantial
        return question

    # Build the rewriting prompt
    history_summary = _build_history_summary_for_rewrite(history)

    rewrite_prompt = f"""Du er en query-rewriting assistent for et juridisk RAG-system.

DIN OPGAVE: Omskriv brugerens spørgsmål så det er selvstændigt og kan bruges til dokumentsøgning.

REGLER:
1. Bevar brugerens PRÆCISE intention - tilføj IKKE information de ikke har spurgt om
2. Indsæt relevante referencer fra samtalehistorikken (artikel, stk., bilag, etc.)
3. Hvis spørgsmålet allerede er selvstændigt, returnér det uændret
4. Output KUN det omskrevne spørgsmål - ingen forklaringer
5. Hvis samtalehistorikken omhandler en specifik lov eller forordning, og det nye spørgsmål ikke nævner en anden lov, SKAL du inkludere lovens navn i det omskrevne spørgsmål

SAMTALEHISTORIK:
{history_summary}

NUVÆRENDE SPØRGSMÅL: {question}

OMSKREVET SPØRGSMÅL:"""

    try:
        from .llm_client import call_llm
        # Use a fast model for rewriting (gpt-4o-mini or similar)
        rewrite_model = os.getenv("QUERY_REWRITE_MODEL", "gpt-4o-mini")
        rewritten = call_llm(rewrite_prompt, model=rewrite_model, temperature=0.0)
        rewritten = rewritten.strip().strip('"').strip("'")

        # Sanity check: if rewritten is empty or much longer, use original
        if not rewritten or len(rewritten) > len(question) * 3:
            return question

        return rewritten
    except Exception:
        # On any error, fall back to original question
        return question


def last_exchange(
    history: list[HistoryMessage] | None,
) -> list[HistoryMessage]:
    """Extract the last complete user+assistant exchange from history.

    Returns at most 2 messages: the last user message and its corresponding
    assistant reply. Used to provide minimal context for intent classification.

    Args:
        history: Conversation history, or None.

    Returns:
        List of 0-2 HistoryMessage objects (last user + last assistant).
        Empty list if no complete exchange is found.
    """
    if not history:
        return []

    # Walk backwards to find the last user message with an assistant reply
    last_user_idx: int | None = None
    last_asst_idx: int | None = None

    for i in range(len(history) - 1, -1, -1):
        if history[i].role == "assistant" and last_asst_idx is None:
            last_asst_idx = i
        elif history[i].role == "user" and last_asst_idx is not None:
            last_user_idx = i
            break

    if last_user_idx is None or last_asst_idx is None:
        return []

    return [history[last_user_idx], history[last_asst_idx]]


def needs_context_augmentation(
    original_query: str | None,
    rewritten_query: str,
) -> bool:
    """Determine if the intent classifier needs conversation context.

    Context augmentation is needed when:
    - The rewriter returned the query unchanged (could not resolve ambiguity), OR
    - The rewritten query is too short for standalone classification.

    First-turn queries (original_query=None) never need augmentation.

    Args:
        original_query: The query before rewriting, or None for first turn.
        rewritten_query: The query after rewriting.

    Returns:
        True if context augmentation should be applied.
    """
    if original_query is None:
        return False

    if rewritten_query == original_query:
        return True

    if len(rewritten_query) < SHORT_QUERY_THRESHOLD:
        return True

    return False


def _build_history_summary_for_rewrite(history: list[HistoryMessage]) -> str:
    """Build a concise history summary for query rewriting.

    Only includes essential context - not full answers.
    """
    lines = []
    for msg in history[-6:]:  # Last 3 exchanges max
        if msg.role == "user":
            lines.append(f"Bruger: {msg.content[:200]}")
        else:
            # For assistant messages, only include first part (the answer, not sources)
            content = msg.content[:500]
            # Try to extract just the main answer (before sources/references)
            if "Kilder" in content:
                content = content.split("Kilder")[0]
            if "Referencer" in content:
                content = content.split("Referencer")[0]
            lines.append(f"Assistent: {content.strip()}")

    return "\n".join(lines)
