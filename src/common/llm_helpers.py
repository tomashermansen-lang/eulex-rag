"""Shared LLM helper functions for generation tasks.

Single Responsibility: Provide common LLM call, JSON parsing, and article
content loading utilities used by both ingestion and eval generation.

Extracted from src/ingestion/ingestion_generation.py to avoid cross-module
import of private functions.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_openai_client():
    """Get OpenAI client, raising if not configured."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


def call_generation_llm(
    prompt: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> str | None:
    """Call LLM with standard error handling.

    Args:
        prompt: The prompt to send
        model: Model to use
        temperature: Sampling temperature
        max_tokens: Max response tokens

    Returns:
        Response content or None on error
    """
    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        return None


def parse_json_response(content: str) -> dict[str, Any] | None:
    """Parse JSON from LLM response, handling markdown wrappers.

    Args:
        content: Raw LLM response

    Returns:
        Parsed dict or None on error
    """
    if not content:
        return None

    # Try to extract JSON if wrapped in markdown
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    try:
        return json.loads(content.strip())
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON response: %s", e)
        return None


def load_article_content(
    corpus_id: str, max_chars_per_article: int = 400
) -> str:
    """Load article titles AND content summaries from chunks file.

    For generation tasks, includes actual text snippets to help the LLM
    understand what each article covers.

    Args:
        corpus_id: Corpus ID to load
        max_chars_per_article: Max characters of content per article

    Returns:
        Formatted string with articles and their content summaries
    """
    chunks_path = PROJECT_ROOT / "data" / "processed" / f"{corpus_id}_chunks.jsonl"
    if not chunks_path.exists():
        return "(Artikelstruktur ikke tilgængelig)"

    articles: dict[int, dict] = {}

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                metadata = chunk.get("metadata", {})
                art = metadata.get("article")
                title = metadata.get("article_title", "")
                text = chunk.get("text", "")

                if art and str(art).isdigit():
                    art_num = int(art)
                    if art_num not in articles:
                        articles[art_num] = {"title": title, "chunks": []}
                    if len(text) > 50:
                        articles[art_num]["chunks"].append(text)

    except Exception as e:
        logger.warning("Could not load article content: %s", e)
        return "(Artikelstruktur ikke tilgængelig)"

    if not articles:
        return "(Ingen artikler fundet)"

    lines = []
    for art_num in sorted(articles.keys())[:30]:
        data = articles[art_num]
        title = data["title"]
        chunks = data["chunks"]

        content_summary = ""
        if chunks:
            first_chunk = chunks[0][:max_chars_per_article]
            if len(chunks[0]) > max_chars_per_article:
                first_chunk += "..."
            content_summary = first_chunk.replace("\n", " ").strip()

        lines.append(f"### Artikel {art_num}: {title}")
        if content_summary:
            lines.append(f"Indhold: {content_summary}")
        lines.append("")

    if len(articles) > 30:
        lines.append(f"... og {len(articles) - 30} yderligere artikler")

    return "\n".join(lines)
