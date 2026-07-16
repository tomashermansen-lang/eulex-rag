"""Pytest configuration and shared fixtures for tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

# Import the _RetrievalResult type for creating mock results
from src.engine.retrieval_pipeline import RetrievedChunk, SelectedChunk


# ─────────────────────────────────────────────────────────────────────────────
# Offline tiktoken stub (avoids network download of BPE data)
# ─────────────────────────────────────────────────────────────────────────────


class _FakeEncoding:
    """Deterministic token encoding that needs no network access.

    Splits on whitespace so 1 word ≈ 1 token.  Sufficient for
    chunk-size / overlap logic tests that care about relative counts,
    not exact GPT tokenization.

    The encode/decode round-trip works by assigning each unique word a
    stable integer ID (via a shared vocabulary dict).
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._reverse: dict[int, str] = {}
        self._next_id = 0

    @property
    def name(self) -> str:
        return "cl100k_base"

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for word in text.split():
            if word not in self._vocab:
                self._vocab[word] = self._next_id
                self._reverse[self._next_id] = word
                self._next_id += 1
            ids.append(self._vocab[word])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._reverse.get(tid, "") for tid in token_ids)


def _fake_get_encoding(name: str) -> _FakeEncoding:
    return _FakeEncoding()


@pytest.fixture(autouse=True)
def _mock_tiktoken():
    """Patch tiktoken.get_encoding globally so tests never hit the network."""
    with patch("tiktoken.get_encoding", _fake_get_encoding):
        yield


def _hits_to_selected_chunks(hits, distances, retrieved_ids):
    """Convert hits to SelectedChunk format for prompt building."""
    selected = []
    for i, (doc, meta) in enumerate(hits):
        chunk = RetrievedChunk(
            chunk_id=retrieved_ids[i] if i < len(retrieved_ids) else f"chunk-{i}",
            document=doc,
            metadata=dict(meta),
            distance=distances[i] if i < len(distances) else 0.1,
        )
        selected.append(
            SelectedChunk(
                chunk=chunk,
                is_citable=True,
                precise_ref=None,
                rank=i,
            )
        )
    return tuple(selected)


@pytest.fixture(autouse=True)
def auto_mock_modular_retrieval(monkeypatch):
    """Automatically provide mock ro.modular_retrieval for the retrieval stage.

    This fixture patches retrieval_orchestration.modular_retrieval to use a mock
    that bridges to legacy retriever._query_collection_raw mocks when present.
    """

    def mock_modular_retrieval(
        *,
        question,
        resolved_profile,
        where_for_retrieval,
        corpus_id,
        retriever,
        collection,
        is_citable_fn,
    ):
        # Priority 1: pre-set mock result dict on retriever (set by _setup_mock_retrieval)
        mock_result = getattr(retriever, "_mock_modular_result", None)
        if mock_result is not None:
            return dict(mock_result)

        hits = []
        distances = []
        retrieved_ids = []
        retrieved_metas = []

        # Priority 2: legacy query_with_where / query mock function on retriever
        legacy_fn = getattr(retriever, "_test_query_fn", None)
        if legacy_fn and callable(legacy_fn):
            try:
                legacy_hits = legacy_fn(question, k=50, where=where_for_retrieval)
                if legacy_hits:
                    hits = legacy_hits
                    distances = getattr(retriever, "_last_distances", None) or [
                        0.1
                    ] * len(hits)
                    retrieved_ids = getattr(retriever, "_last_retrieved_ids", None) or [
                        f"chunk-{i}" for i in range(len(hits))
                    ]
                    retrieved_metas = getattr(
                        retriever, "_last_retrieved_metadatas", None
                    ) or [m for _, m in hits]
            except TypeError:
                # query-style functions don't accept k/where kwargs
                try:
                    legacy_hits = legacy_fn(question)
                    if legacy_hits:
                        hits = legacy_hits
                        distances = [0.1] * len(hits)
                        retrieved_ids = [f"chunk-{i}" for i in range(len(hits))]
                        retrieved_metas = [m for _, m in hits]
                except Exception:
                    pass

        # Priority 3: _query_collection_raw mock on the retriever
        if not hits:
            qcr = getattr(retriever, "_query_collection_raw", None)
            if qcr and callable(qcr):
                try:
                    result = qcr(
                        collection=collection,
                        question=question,
                        k=50,
                        where=where_for_retrieval,
                    )
                    if result and len(result) >= 4:
                        chunk_ids, docs, metas, dists = result
                        hits = list(zip(docs, metas))
                        distances = list(dists)
                        retrieved_ids = list(chunk_ids)
                        retrieved_metas = list(metas)
                except Exception:
                    pass

        # Handle hits that might be (doc, meta) tuples or just documents
        if hits and isinstance(hits[0], str):
            hits = [(h, {}) for h in hits]

        # Fallback defaults
        if not distances:
            distances = getattr(retriever, "_last_distances", None) or [0.1] * len(hits)
        if not retrieved_ids:
            retrieved_ids = getattr(retriever, "_last_retrieved_ids", None) or [
                f"chunk-{i}" for i in range(len(hits))
            ]
        if not retrieved_metas:
            retrieved_metas = getattr(retriever, "_last_retrieved_metadatas", None) or [
                m for _, m in hits
            ]

        # Convert hits to SelectedChunk format
        selected_chunks = _hits_to_selected_chunks(hits, distances, retrieved_ids)

        # Return dict (same format as ro.modular_retrieval)
        return {
            "hits": hits,
            "distances": distances,
            "retrieved_ids": retrieved_ids,
            "retrieved_metas": retrieved_metas,
            "run_meta_updates": {},
            "selected_chunks": selected_chunks,
            "total_retrieved": len(hits),
            "citable_count": len(hits),
        }

    monkeypatch.setattr(
        "src.engine.retrieval_orchestration.modular_retrieval",
        mock_modular_retrieval,
    )
