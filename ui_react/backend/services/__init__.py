"""Service layer - thin wrapper around src.services.ask.

Single Responsibility: Bridge between FastAPI routes and RAG engine.
No business logic duplication - delegates to existing engine.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.ask import ask, ask_stream, build_engine, AskResult
from src.common.config_loader import load_settings, Settings
from src.engine.conversation import HistoryMessage as CoreHistoryMessage




def get_settings() -> Settings:
    """Load application settings."""
    return load_settings()


# Official display names for EU regulations (Danish with short name in parentheses)
def get_corpora() -> list[dict[str, Any]]:
    """Get list of available corpora with metadata.

    Reads display_name from corpora.json inventory file.
    """
    from src.common.corpora_inventory import load_corpora_inventory, default_corpora_path

    settings = get_settings()
    corpora = settings.corpora or {}

    # Load inventory to get display names
    inventory = load_corpora_inventory(default_corpora_path(PROJECT_ROOT))
    inv_corpora = inventory.get("corpora", {})

    result = []
    for corpus_id, corpus_config in corpora.items():
        # Get display name and celex_number from inventory, fallback to formatted ID
        inv_entry = inv_corpora.get(corpus_id, {})
        display_name = inv_entry.get(
            "display_name",
            corpus_id.upper().replace("-", " ").replace("_", " ")
        )
        celex_number = inv_entry.get("celex_number")
        result.append({
            "id": corpus_id,
            "name": display_name,
            "fullname": inv_entry.get("fullname"),
            "source_url": getattr(corpus_config, "source_url", None),
            "celex_number": celex_number,
            "eurovoc_labels": inv_entry.get("eurovoc_labels", []),
        })

    return result


def get_examples() -> dict[str, dict[str, list[str]]]:
    """Get example questions for all corpora and profiles.

    Loads from data/processed/example_questions.json (single source of truth).
    Returns empty dict if file doesn't exist.
    """
    import json

    examples_path = PROJECT_ROOT / "data" / "processed" / "example_questions.json"
    if examples_path.exists():
        try:
            with open(examples_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {}


def get_source_url(corpus_id: str) -> str | None:
    """Get source URL for a corpus."""
    settings = get_settings()
    corpora = settings.corpora or {}

    if corpus_id in corpora:
        return getattr(corpora[corpus_id], "source_url", None)

    # Handle underscore/hyphen drift
    alt = corpus_id.replace("_", "-")
    if alt in corpora:
        return getattr(corpora[alt], "source_url", None)

    alt2 = corpus_id.replace("-", "_")
    if alt2 in corpora:
        return getattr(corpora[alt2], "source_url", None)

    return None


def _convert_history(history: list[dict] | None) -> list[CoreHistoryMessage]:
    """Convert API history messages to core engine format."""
    if not history:
        return []
    return [
        CoreHistoryMessage(role=msg["role"], content=msg["content"])
        for msg in history
    ]


def get_answer(
    question: str,
    law: str,
    user_profile: str = "LEGAL",
    history: list[dict] | None = None,
    corpus_scope: str = "single",
    target_corpora: list[str] | None = None,
) -> AskResult:
    """Get a complete answer (non-streaming)."""
    return ask(
        question=question,
        law=law,
        user_profile=user_profile,
        history=_convert_history(history),
        corpus_scope=corpus_scope,
        target_corpora=target_corpora,
    )


def stream_answer(
    question: str,
    law: str,
    user_profile: str = "LEGAL",
    history: list[dict] | None = None,
    corpus_scope: str = "single",
    target_corpora: list[str] | None = None,
) -> Iterator[str | AskResult]:
    """Stream answer chunks, then yield final AskResult."""
    yield from ask_stream(
        question=question,
        law=law,
        user_profile=user_profile,
        history=_convert_history(history),
        corpus_scope=corpus_scope,
        target_corpora=target_corpora,
    )
