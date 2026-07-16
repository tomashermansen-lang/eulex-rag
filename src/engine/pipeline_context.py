"""Pipeline context for the RAG orchestrator.

PipelineContext is a frozen snapshot of all configuration and services
needed by a single answer_structured() invocation. It is constructed ONCE
from self.* at pipeline start — no stage function touches self after that.

NOTE: Not yet wired into stage functions (they receive individual parameters).
This is scaffolding for a future architectural pass that will adopt PipelineContext
as the single parameter bag for all stage functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class PipelineContext:
    """Immutable snapshot — constructed ONCE from self.* at pipeline start.

    No stage function touches self after this is built.
    """

    # Config (frozen at __init__)
    corpus_id: str
    top_k: int
    max_distance: float | None
    hard_max_distance: float | None
    chat_model: str
    embedding_model: str
    enable_hybrid_rerank: bool
    hybrid_vec_k: int
    ranking_weights: Any  # RankingWeights from config_loader

    # Services (injected)
    retriever: Any  # Retriever
    ranker: Any  # Ranker
    collection: Any  # ChromaDB Collection
    chroma: Any  # ChromaDB Client
    project_root: Path | None
    resolver_fn: Callable  # lazy corpus resolver

    # Computed at pipeline start
    collection_name: str
