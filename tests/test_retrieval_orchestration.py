"""Tests for retrieval_orchestration module — Step 6.3.

Tests standalone functions extracted from RAGEngine.
"""

from types import SimpleNamespace

from src.engine.retrieval_orchestration import (
    get_collection_for_corpus,
    get_case_law_collection_for_corpus,
    make_query_fn,
    make_inject_fn,
    _anchor_to_where_hint,
    QueryWithWhereResult,
    execute_query_with_where,
)


class TestGetCollectionForCorpus:
    def test_returns_collection_with_corpus_name(self):
        mock_chroma = SimpleNamespace(
            get_or_create_collection=lambda name: SimpleNamespace(name=name)
        )
        coll = get_collection_for_corpus(mock_chroma, "ai-act")
        assert coll.name == "ai-act_documents"

    def test_uses_corpus_id_in_collection_name(self):
        calls = []
        mock_chroma = SimpleNamespace(
            get_or_create_collection=lambda name: (
                calls.append(name),
                SimpleNamespace(name=name),
            )[1]
        )
        get_collection_for_corpus(mock_chroma, "gdpr")
        assert calls == ["gdpr_documents"]


class TestGetCaseLawCollectionForCorpus:
    def test_returns_case_law_collection(self):
        mock_chroma = SimpleNamespace(
            get_or_create_collection=lambda name: SimpleNamespace(name=name)
        )
        coll = get_case_law_collection_for_corpus(mock_chroma, "gdpr")
        assert coll.name == "gdpr_case_law"

    def test_different_corpus_id(self):
        calls = []
        mock_chroma = SimpleNamespace(
            get_or_create_collection=lambda name: (
                calls.append(name),
                SimpleNamespace(name=name),
            )[1]
        )
        get_case_law_collection_for_corpus(mock_chroma, "ai-act")
        assert calls == ["ai-act_case_law"]


class TestMakeQueryFn:
    def test_returns_callable(self):
        mock_retriever = SimpleNamespace(
            _query_collection_raw=lambda **kw: (["id1"], ["doc1"], [{"a": 1}], [0.1])
        )
        coll = SimpleNamespace()
        fn = make_query_fn(mock_retriever, coll)
        assert callable(fn)

    def test_query_fn_returns_zipped_results(self):
        mock_retriever = SimpleNamespace(
            _query_collection_raw=lambda **kw: (
                ["id1", "id2"],
                ["doc1", "doc2"],
                [{"a": 1}, {"a": 2}],
                [0.1, 0.2],
            )
        )
        coll = SimpleNamespace()
        fn = make_query_fn(mock_retriever, coll)
        results, dists = fn("question", 2, None)
        assert len(results) == 2
        assert results[0] == ("id1", "doc1", {"a": 1})
        assert dists == [0.1, 0.2]


class TestMakeInjectFn:
    def test_returns_callable(self):
        mock_retriever = SimpleNamespace(
            _query_collection_raw=lambda **kw: ([], [], [], [])
        )
        coll = SimpleNamespace()
        fn = make_inject_fn(mock_retriever, coll, "ai-act")
        assert callable(fn)

    def test_inject_fn_returns_empty_for_invalid_anchor(self):
        mock_retriever = SimpleNamespace(
            _query_collection_raw=lambda **kw: ([], [], [], [])
        )
        coll = SimpleNamespace()
        fn = make_inject_fn(mock_retriever, coll, "ai-act")
        result = fn("question", "invalid", 2)
        assert result == []

    def test_inject_fn_queries_with_article_where(self):
        captured_where = {}

        def mock_raw(**kw):
            captured_where.update(kw.get("where", {}))
            return (["id1"], ["doc1"], [{"article": "5"}], [0.1])

        mock_retriever = SimpleNamespace(_query_collection_raw=mock_raw)
        coll = SimpleNamespace()
        fn = make_inject_fn(mock_retriever, coll, "ai-act")
        result = fn("question", "article:5", 2)
        assert len(result) == 1
        assert captured_where.get("article") == "5"
        assert captured_where.get("corpus_id") == "ai-act"


class TestAnchorToWhereHint:
    """Verify _anchor_to_where_hint extracts correct where-hint dicts."""

    def test_article_anchor(self):
        result = _anchor_to_where_hint("article:5", "ai-act")
        assert result == {"corpus_id": "ai-act", "article": "5"}

    def test_recital_anchor(self):
        result = _anchor_to_where_hint("recital:42", "gdpr")
        assert result == {"corpus_id": "gdpr", "recital": "42"}

    def test_annex_anchor(self):
        result = _anchor_to_where_hint("annex:III", "ai-act")
        assert result is not None
        assert result["corpus_id"] == "ai-act"
        assert "annex" in result

    def test_annex_with_point(self):
        result = _anchor_to_where_hint("annex:III:5", "ai-act")
        assert result is not None
        assert result.get("annex_point") == "5"

    def test_invalid_no_colon(self):
        assert _anchor_to_where_hint("invalid", "ai-act") is None

    def test_invalid_kind(self):
        assert _anchor_to_where_hint("chapter:2", "ai-act") is None


class TestQueryWithWhereResult:
    """Verify QueryWithWhereResult dataclass structure."""

    def test_is_dataclass_with_required_fields(self):
        from dataclasses import fields

        field_names = [f.name for f in fields(QueryWithWhereResult)]
        assert "hits" in field_names
        assert "distances" in field_names
        assert "ids" in field_names
        assert "metadatas" in field_names
        assert "query_where" in field_names
        assert "sibling_expansion" in field_names


class TestExecuteQueryWithWhere:
    """Verify execute_query_with_where function."""

    def test_is_callable(self):
        assert callable(execute_query_with_where)

    def test_article_filter_shortcut(self):
        """When question mentions exactly one article with results, use filtered path."""

        def mock_query_with_distances(*, collection, question, k, where=None):
            return [("doc1", {"article": "12"})], [0.1]

        retriever = SimpleNamespace(
            _query_collection_with_distances=mock_query_with_distances,
            _query_collection_raw=lambda **kw: ([], [], [], []),
            _last_sibling_expansion={},
        )

        result = execute_query_with_where(
            question="Hvad siger artikel 12?",
            k=3,
            where=None,
            collection=object(),
            collection_name="test_docs",
            retriever=retriever,
            ranker=SimpleNamespace(),
            enable_hybrid_rerank=True,
            hybrid_vec_k=30,
            ranking_weights=None,
        )

        assert len(result.hits) == 1
        assert result.distances == [0.1]
        assert result.query_where == {"article": "12"}

    def test_hybrid_rerank_path(self):
        """When hybrid rerank is enabled and no article shortcut, use hybrid path."""
        calls = []

        def mock_query_raw(*, collection, question, k, where=None, track_state=True):
            calls.append("raw")
            return (
                ["id1", "id2"],
                ["doc1 text", "doc2 text"],
                [{"article": "12"}, {"article": "14"}],
                [0.1, 0.2],
            )

        def mock_hybrid_rerank(
            *, question, ids, documents, metadatas, distances, k, weights
        ):
            calls.append("rerank")
            return (
                [("doc1 text", {"article": "12"}), ("doc2 text", {"article": "14"})],
                [0.1, 0.2],
                ["id1", "id2"],
            )

        retriever = SimpleNamespace(
            _query_collection_with_distances=lambda **kw: ([], []),
            _query_collection_raw=mock_query_raw,
            _expand_to_siblings=None,
            _last_sibling_expansion={},
        )
        ranker = SimpleNamespace(_hybrid_rerank_hits=mock_hybrid_rerank)

        result = execute_query_with_where(
            question="Hvilke krav gælder for record-keeping?",
            k=2,
            where=None,
            collection=object(),
            collection_name="test_docs",
            retriever=retriever,
            ranker=ranker,
            enable_hybrid_rerank=True,
            hybrid_vec_k=30,
            ranking_weights=None,
        )

        assert "raw" in calls
        assert "rerank" in calls
        assert len(result.hits) == 2
        assert result.distances == [0.1, 0.2]
