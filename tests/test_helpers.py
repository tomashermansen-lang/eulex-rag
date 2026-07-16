"""Tests for src/engine/helpers.py - Utility functions."""

from src.engine.helpers import (
    normalize_anchor,
    normalize_annex_for_chroma,
    get_meta_value,
    normalize_metadata,
    normalize_anchor_list,
    classify_query_intent,
    _extract_article_ref,
    _extract_article_refs,
    _extract_annex_refs,
    _extract_recital_ref,
    _extract_chapter_ref,
    _roman_to_int,
    _ref_to_int,
    _normalize_modals_to_danish,
    _normalize_abstain_text,
    _looks_like_structure_question,
    _looks_like_substantive_question,
    anchors_from_metadata,
    anchors_present_from_hits,
)


class TestNormalizeAnchor:
    def test_lowercases_and_strips(self):
        assert normalize_anchor("  ARTICLE:6  ") == "article:6"

    def test_removes_whitespace(self):
        assert normalize_anchor("article : 6") == "article:6"

    def test_handles_none(self):
        assert normalize_anchor(None) == ""


class TestNormalizeAnnexForChroma:
    def test_uppercases(self):
        assert normalize_annex_for_chroma("iii") == "III"

    def test_handles_empty(self):
        assert normalize_annex_for_chroma("") == ""


class TestGetMetaValue:
    def test_uppercases_structural_fields(self):
        assert get_meta_value({"article": "6a"}, "article") == "6A"

    def test_lowercases_identifier_fields(self):
        assert get_meta_value({"corpus_id": "AI-ACT"}, "corpus_id") == "ai-act"

    def test_returns_default(self):
        assert get_meta_value({}, "article", "X") == "X"


class TestNormalizeMetadata:
    def test_normalizes_fields(self):
        result = normalize_metadata({"article": "6a", "annex": "iii"})
        assert result["article"] == "6A"
        assert result["annex"] == "III"


class TestNormalizeAnchorList:
    def test_normalizes_list(self):
        assert normalize_anchor_list(["ARTICLE:6"]) == ["article:6"]

    def test_deduplicates(self):
        assert normalize_anchor_list(["article:6", "ARTICLE:6"]) == ["article:6"]

    def test_require_colon(self):
        assert normalize_anchor_list(["article:6", "bad"], require_colon=True) == [
            "article:6"
        ]


class TestClassifyQueryIntent:
    def test_enforcement(self):
        assert classify_query_intent("What are the penalties?") == "ENFORCEMENT"

    def test_obligations(self):
        assert classify_query_intent("What must we do?") == "OBLIGATIONS"

    def test_context_default(self):
        assert classify_query_intent("random question") == "CONTEXT"


class TestExtractArticleRef:
    def test_extracts_number(self):
        assert _extract_article_ref("artikel 6") == "6"

    def test_returns_none(self):
        assert _extract_article_ref("no article") is None


class TestExtractArticleRefs:
    def test_extracts_multiple(self):
        result = _extract_article_refs("artikel 6 og artikel 7")
        assert "6" in result and "7" in result


class TestExtractAnnexRefs:
    def test_extracts_annex(self):
        assert "III" in _extract_annex_refs("bilag III")


class TestExtractRecitalRef:
    def test_extracts_recital(self):
        assert _extract_recital_ref("betragtning 42") == "42"


class TestExtractChapterRef:
    def test_extracts_chapter(self):
        assert _extract_chapter_ref("kapitel 5") == "5"


class TestRomanToInt:
    def test_converts(self):
        assert _roman_to_int("III") == 3
        assert _roman_to_int("IV") == 4

    def test_invalid(self):
        assert _roman_to_int("ABC") is None


class TestRefToInt:
    def test_numeric(self):
        assert _ref_to_int("42") == 42

    def test_roman(self):
        assert _ref_to_int("III") == 3


class TestNormalizeModalsToDanish:
    def test_replaces_must(self):
        assert "SKAL" in _normalize_modals_to_danish("You MUST comply")

    def test_replaces_should(self):
        assert "BØR" in _normalize_modals_to_danish("You SHOULD")


class TestNormalizeAbstainText:
    def test_normalizes_lowercase(self):
        result = _normalize_abstain_text("jeg kan ikke svare")
        assert result.startswith("Jeg kan ikke")


class TestLooksLikeStructureQuestion:
    def test_detects_toc(self):
        assert _looks_like_structure_question("indholdsfortegnelse")

    def test_detects_navigation(self):
        assert _looks_like_structure_question("hvor ligger artikel 6?")


class TestLooksLikeSubstantiveQuestion:
    def test_detects_hvad(self):
        assert _looks_like_substantive_question("hvad er kravene?")


class TestAnchorsFromMetadata:
    def test_extracts_anchors(self):
        result = anchors_from_metadata({"article": "6"})
        assert "article:6" in result


class TestAnchorsPresentFromHits:
    def test_extracts_from_hits(self):
        hits = [("doc", {"article": "6"})]
        result = anchors_present_from_hits(hits)
        assert "article:6" in result


# ---------------------------------------------------------------------------
# Step 6.2a: Module-level utility functions extracted from RAGEngine
# ---------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import patch
from src.engine.helpers import (
    build_retrieval_state_dict,
    build_hybrid_rerank_dict,
    iso_utc_now,
    best_effort_git_commit_short,
    collection_name_best_effort,
)


class TestBuildRetrievalStateDict:
    def test_builds_dict_from_retriever_state(self):
        retriever = SimpleNamespace(
            _last_effective_where={"article": "5"},
            _last_effective_collection_name="ai-act_documents",
            _last_effective_collection_type="chunks",
            _last_retrieved_ids=["id1", "id2"],
            _last_retrieved_metadatas=[{"a": 1}, {"a": 2}],
        )
        result = build_retrieval_state_dict(
            retriever=retriever,
            query_collection_name="ai-act_documents",
            query_where={"article": "5"},
            planned_where={"article": "5"},
            planned_collection_type="chunks",
        )
        assert result["query_collection"] == "ai-act_documents"
        assert result["query_where"] == {"article": "5"}
        assert result["planned_collection_type"] == "chunks"
        assert result["retrieved_ids"] == ["id1", "id2"]

    def test_handles_missing_retriever_attrs(self):
        retriever = SimpleNamespace()
        result = build_retrieval_state_dict(
            retriever=retriever,
            query_collection_name=None,
            query_where=None,
            planned_where=None,
            planned_collection_type="chunks",
        )
        assert result["retrieved_ids"] == []
        assert result["effective_where"] is None


class TestBuildHybridRerankDict:
    def test_builds_dict_from_weights(self):
        weights = SimpleNamespace(
            alpha_vec=0.4, beta_bm25=0.3, gamma_cite=0.2, delta_role=0.1
        )
        result = build_hybrid_rerank_dict(
            enable_hybrid_rerank=True,
            ranking_weights=weights,
            hybrid_vec_k=30,
        )
        assert result["enabled"] is True
        assert result["weights"]["alpha_vec"] == 0.4
        assert result["vec_k"] == 30


class TestIsoUtcNow:
    def test_returns_iso_string_ending_with_z(self):
        result = iso_utc_now()
        assert result.endswith("Z")
        assert "T" in result


class TestBestEffortGitCommitShort:
    def test_returns_env_var_if_set(self):
        with patch.dict("os.environ", {"GIT_COMMIT": "abc123def456789"}):
            result = best_effort_git_commit_short()
            assert result == "abc123def456"

    def test_truncates_to_12_chars(self):
        with patch.dict("os.environ", {"GIT_COMMIT": "a" * 40}):
            result = best_effort_git_commit_short()
            assert len(result) == 12


class TestCollectionNameBestEffort:
    def test_returns_collection_name_when_same_object(self):
        coll = SimpleNamespace(name="test_documents")
        result = collection_name_best_effort(
            collection=coll,
            engine_collection=coll,
            engine_collection_name="my_collection",
        )
        assert result == "my_collection"

    def test_returns_name_attr_for_different_collection(self):
        coll = SimpleNamespace(name="other_documents")
        engine_coll = SimpleNamespace(name="engine_documents")
        result = collection_name_best_effort(
            collection=coll,
            engine_collection=engine_coll,
            engine_collection_name="engine_collection",
        )
        assert result == "other_documents"

    def test_returns_none_for_no_name(self):
        coll = SimpleNamespace()
        result = collection_name_best_effort(
            collection=coll,
            engine_collection=None,
            engine_collection_name=None,
        )
        assert result is None
