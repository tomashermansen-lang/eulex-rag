"""Pytest configuration and shared fixtures for tests."""

import pytest

# Import the _RetrievalResult type for creating mock results
from src.engine.rag import _RetrievalResult
from src.engine.retrieval_pipeline import RetrievedChunk, SelectedChunk


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
        selected.append(SelectedChunk(
            chunk=chunk,
            is_citable=True,
            precise_ref=None,
            rank=i,
        ))
    return tuple(selected)


@pytest.fixture(autouse=True)
def auto_mock_modular_retrieval(monkeypatch):
    """Automatically provide mock _modular_retrieval for RAGEngine instances.

    This fixture patches RAGEngine._modular_retrieval to use a mock that
    delegates to the legacy query/query_with_where mocks if they exist.
    This enables backward compatibility with tests that mock the legacy methods.
    """
    from src.engine.rag import RAGEngine

    original_modular_retrieval = RAGEngine._modular_retrieval

    def mock_modular_retrieval(self, **kwargs):
        # Check if query or query_with_where is mocked (instance-level)
        question = kwargs.get('question', '')
        where = kwargs.get('where_for_retrieval')
        hits = []
        distances = []
        retrieved_ids = []
        retrieved_metas = []

        # Check for _retriever mocks first (lower-level)
        retriever = getattr(self, '_retriever', None)
        if retriever:
            qcr = getattr(retriever, '_query_collection_raw', None)
            if qcr and callable(qcr):
                try:
                    # Call with collection=None since we're mocking
                    result = qcr(collection=None, question=question, k=50, where=where)
                    if result and len(result) >= 4:
                        chunk_ids, docs, metas, dists = result
                        hits = list(zip(docs, metas))
                        distances = list(dists)
                        retrieved_ids = list(chunk_ids)
                        retrieved_metas = list(metas)
                except Exception:
                    pass

        # If no retriever mock hits, try query_with_where
        if not hits:
            has_mock_query_with_where = (
                hasattr(self, 'query_with_where') and
                callable(self.query_with_where) and
                not hasattr(self.query_with_where, '__self__')  # Not a bound method
            )

            has_mock_query = (
                hasattr(self, 'query') and
                callable(self.query) and
                not hasattr(self.query, '__self__')  # Not a bound method
            )

            if has_mock_query_with_where:
                try:
                    hits = self.query_with_where(question, k=50, where=where)
                except TypeError:
                    try:
                        hits = self.query_with_where(question)
                    except Exception:
                        hits = []
            elif has_mock_query:
                try:
                    hits = self.query(question)
                except Exception:
                    hits = []

        # Handle hits that might be (doc, meta) tuples or just documents
        if hits and isinstance(hits[0], str):
            hits = [(h, {}) for h in hits]

        # Get distances and IDs from instance attributes or retriever if not already set
        if not distances:
            distances = getattr(self, '_last_distances', None) or getattr(retriever, '_last_distances', None) or [0.1] * len(hits)
        if not retrieved_ids:
            retrieved_ids = getattr(self, '_last_retrieved_ids', None) or getattr(retriever, '_last_retrieved_ids', None) or [f"chunk-{i}" for i in range(len(hits))]
        if not retrieved_metas:
            retrieved_metas = getattr(self, '_last_retrieved_metadatas', None) or getattr(retriever, '_last_retrieved_metadatas', None) or [m for _, m in hits]

        # Convert hits to SelectedChunk format
        selected_chunks = _hits_to_selected_chunks(hits, distances, retrieved_ids)

        return _RetrievalResult(
            hits=hits,
            distances=distances,
            retrieved_ids=retrieved_ids,
            retrieved_metas=retrieved_metas,
            run_meta_updates={},
            selected_chunks=selected_chunks,
            total_retrieved=len(hits),
            citable_count=len(hits),
        )

    # Patch at the class level
    monkeypatch.setattr(RAGEngine, '_modular_retrieval', mock_modular_retrieval)
