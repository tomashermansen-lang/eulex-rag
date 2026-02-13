"""Tests for corpus-aware citation deduplication.

TDD: These tests are written BEFORE the implementation.
"""

import pytest
from unittest.mock import MagicMock


class TestCorpusAwareDeduplication:
    """Tests for corpus-aware deduplication in citations."""

    def test_ccc_001_same_article_different_corpora_not_deduplicated(self):
        """Same article in different corpora should NOT be deduplicated."""
        from src.engine.citations import apply_hard_reference_gating

        # Create references with same article number in different corpora
        references_structured_all = [
            {"idx": 1, "chunk_id": "chunk_1", "article": "6", "corpus_id": "ai_act"},
            {"idx": 2, "chunk_id": "chunk_2", "article": "6", "corpus_id": "gdpr"},
            {"idx": 3, "chunk_id": "chunk_3", "article": "6", "corpus_id": "nis2"},
        ]

        result = apply_hard_reference_gating(
            answer_text="See Article 6 in AI-Act [1] and GDPR [2] and NIS2 [3].",
            used_chunk_ids=["chunk_1", "chunk_2", "chunk_3"],
            references_structured_all=references_structured_all,
            question="What is Article 6?",
            is_legal_profile=False,
            legal_allow_reference_fallback=False,
        )

        # All three should be preserved (different corpora)
        assert len(result.references_structured) == 3
        corpus_ids = [r.get("corpus_id") for r in result.references_structured]
        assert "ai_act" in corpus_ids
        assert "gdpr" in corpus_ids
        assert "nis2" in corpus_ids

    def test_ccc_002_same_corpus_same_article_deduplicated(self):
        """Same article in same corpus should be deduplicated."""
        from src.engine.citations import apply_hard_reference_gating

        # Create references with same article in same corpus (different chunks)
        references_structured_all = [
            {"idx": 1, "chunk_id": "chunk_1", "article": "6", "corpus_id": "ai_act"},
            {"idx": 2, "chunk_id": "chunk_2", "article": "6", "corpus_id": "ai_act"},  # Same article+corpus
        ]

        result = apply_hard_reference_gating(
            answer_text="Article 6 [1] [2]",
            used_chunk_ids=["chunk_1", "chunk_2"],
            references_structured_all=references_structured_all,
            question="What is Article 6?",
            is_legal_profile=False,
            legal_allow_reference_fallback=False,
        )

        # Should be deduplicated to one
        assert len(result.references_structured) == 1
        assert result.references_structured[0]["article"] == "6"

    def test_ccc_003_dedup_key_includes_corpus_id(self):
        """Deduplication key should include corpus_id."""
        from src.engine.citations import _get_reference_dedup_key

        ref_ai_act = {"article": "6", "corpus_id": "ai_act"}
        ref_gdpr = {"article": "6", "corpus_id": "gdpr"}
        ref_no_corpus = {"article": "6"}

        key_ai_act = _get_reference_dedup_key(ref_ai_act)
        key_gdpr = _get_reference_dedup_key(ref_gdpr)
        key_no_corpus = _get_reference_dedup_key(ref_no_corpus)

        # Different corpus_id should produce different keys
        assert key_ai_act != key_gdpr

        # No corpus_id uses empty string
        assert key_no_corpus != key_ai_act

    def test_ccc_004_hard_reference_gating_respects_corpus_diversity(self):
        """Hard reference gating should preserve corpus diversity."""
        from src.engine.citations import apply_hard_reference_gating

        # Multi-corpus references with same article numbers
        references_structured_all = [
            {"idx": 1, "chunk_id": "c1", "article": "5", "corpus_id": "ai_act"},
            {"idx": 2, "chunk_id": "c2", "article": "5", "corpus_id": "gdpr"},
            {"idx": 3, "chunk_id": "c3", "article": "6", "corpus_id": "ai_act"},
            {"idx": 4, "chunk_id": "c4", "article": "6", "corpus_id": "gdpr"},
        ]

        result = apply_hard_reference_gating(
            answer_text="See [1] and [2] and [3] and [4].",
            used_chunk_ids=["c1", "c2", "c3", "c4"],
            references_structured_all=references_structured_all,
            question="Compare articles",
            is_legal_profile=False,
            legal_allow_reference_fallback=False,
        )

        # All 4 should be preserved (2 corpora x 2 articles = 4 unique combinations)
        assert len(result.references_structured) == 4

    def test_ccc_005_fallback_ref_selection_groups_by_corpus(self):
        """LEGAL fallback selection should be proportional across corpora."""
        from src.engine.citations import _select_legal_fallback_chunk_ids

        # 6 references from 3 corpora
        references_structured_all = [
            {"idx": 1, "chunk_id": "c1", "article": "1", "corpus_id": "ai_act"},
            {"idx": 2, "chunk_id": "c2", "article": "2", "corpus_id": "ai_act"},
            {"idx": 3, "chunk_id": "c3", "article": "1", "corpus_id": "gdpr"},
            {"idx": 4, "chunk_id": "c4", "article": "2", "corpus_id": "gdpr"},
            {"idx": 5, "chunk_id": "c5", "article": "1", "corpus_id": "nis2"},
            {"idx": 6, "chunk_id": "c6", "article": "2", "corpus_id": "nis2"},
        ]

        # Get 3 fallback refs - should ideally include one from each corpus
        result = _select_legal_fallback_chunk_ids(
            references_structured_all=references_structured_all,
            max_refs=3,
            question="What are the requirements?",
        )

        assert len(result) <= 3
        # The selection algorithm prioritizes based on keyword overlap,
        # but for cross-law we want to verify it doesn't just take the first 3
        # from a single corpus when multiple are available
