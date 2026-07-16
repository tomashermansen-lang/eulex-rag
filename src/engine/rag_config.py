"""RAG configuration helpers and data types.

Extracted from rag.py (Step 7.1). All defaults come from config/settings.yaml.
Engine modules import config from here instead of duplicating helpers.
"""

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Config helpers — all defaults come from config/settings.yaml
# ---------------------------------------------------------------------------


def _get_openai_settings() -> dict:
    """Get OpenAI settings from config. Environment variables can override."""
    from ..common.config_loader import get_settings_yaml

    settings = get_settings_yaml()
    return settings.get("openai", {})


def _get_model_capabilities() -> dict:
    """Get model capabilities from config."""
    from ..common.config_loader import get_settings_yaml

    settings = get_settings_yaml()
    return settings.get("model_capabilities", {})


def _get_default_chat_model() -> str:
    """Get default chat model from config. Env OPENAI_CHAT_MODEL can override."""
    openai_settings = _get_openai_settings()
    return os.getenv("OPENAI_CHAT_MODEL") or openai_settings.get("chat_model")


def _get_default_embedding_model() -> str:
    """Get default embedding model from config. Env OPENAI_EMBEDDING_MODEL can override."""
    openai_settings = _get_openai_settings()
    return os.getenv("OPENAI_EMBEDDING_MODEL") or openai_settings.get("embedding_model")


def _get_default_temperature() -> float:
    """Get default temperature from config. Env RAG_OPENAI_TEMPERATURE can override."""
    openai_settings = _get_openai_settings()
    temp = os.getenv("RAG_OPENAI_TEMPERATURE")
    if temp is not None:
        return float(temp)
    return float(openai_settings.get("temperature", 0.0))


def _get_rag_settings() -> dict:
    """Get RAG pipeline settings from config."""
    from ..common.config_loader import get_settings_yaml

    settings = get_settings_yaml()
    return settings.get("rag", {})


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class _RetrievalResult:
    """Result from retrieval stage (modular pipeline)."""

    hits: List[Tuple[str, Dict[str, Any]]]
    distances: List[float]
    retrieved_ids: List[str]
    retrieved_metas: List[Dict[str, Any]]
    run_meta_updates: Dict[str, Any]  # Updates to merge into run_meta
    # New fields for prompt building
    selected_chunks: Tuple[Any, ...] = ()  # Tuple[SelectedChunk, ...]
    total_retrieved: int = 0  # Total chunks before citable filtering
    citable_count: int = 0  # Citable chunks before cap


# ---------------------------------------------------------------------------
# System utilities (extracted from helpers.py, Phase 8b)
# ---------------------------------------------------------------------------


def iso_utc_now() -> str:
    """Return current UTC time in ISO 8601 format with Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def best_effort_git_commit_short() -> str | None:
    """Return short git commit hash from env or git CLI."""
    env = str(os.getenv("GIT_COMMIT", "") or "").strip()
    if env:
        return env[:12]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="ignore").strip() or None
    except Exception:  # noqa: BLE001
        return None


def collection_name_best_effort(
    *,
    collection: Any,
    engine_collection: Any,
    engine_collection_name: str | None,
) -> str | None:
    """Best-effort collection name from a collection object."""
    try:
        if collection is engine_collection:
            return str(engine_collection_name or "") or None
    except Exception:  # noqa: BLE001
        pass
    name = getattr(collection, "name", None)
    try:
        return str(name) if name else None
    except Exception:  # noqa: BLE001
        return None
