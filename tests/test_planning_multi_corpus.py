"""Tests for multi-corpus context assembly in planning.

TDD: These tests are written BEFORE the implementation.
"""

import pytest
from unittest.mock import MagicMock


class TestPrepareMultiCorpusContext:
    """Tests for prepare_multi_corpus_context function."""

    def test_pmc_001_sets_synthesis_mode(self):
        """prepare_multi_corpus_context sets synthesis_mode in context."""
        from src.engine.planning import prepare_multi_corpus_context
        from src.engine.synthesis_router import SynthesisMode, SynthesisContext

        synthesis_ctx = SynthesisContext(
            mode=SynthesisMode.AGGREGATION,
            target_corpora=("ai_act", "gdpr"),
        )

        result = prepare_multi_corpus_context(
            question="What do all laws say about data protection?",
            synthesis_context=synthesis_ctx,
            resolved_profile_str="LEGAL",
            top_k=20,
        )

        assert result.synthesis_mode == SynthesisMode.AGGREGATION

    def test_pmc_002_tracks_per_corpus_evidence(self):
        """per_corpus_evidence tracks which corpora contributed."""
        from src.engine.planning import prepare_multi_corpus_context, MultiCorpusContext
        from src.engine.synthesis_router import SynthesisMode, SynthesisContext

        synthesis_ctx = SynthesisContext(
            mode=SynthesisMode.AGGREGATION,
            target_corpora=("ai_act", "gdpr", "nis2"),
        )

        result = prepare_multi_corpus_context(
            question="What are the requirements?",
            synthesis_context=synthesis_ctx,
            resolved_profile_str="LEGAL",
            top_k=20,
        )

        # per_corpus_evidence should be initialized with all target corpora
        assert "ai_act" in result.per_corpus_evidence
        assert "gdpr" in result.per_corpus_evidence
        assert "nis2" in result.per_corpus_evidence

    def test_pmc_003_empty_corpora_handled(self):
        """Empty target_corpora returns appropriate context."""
        from src.engine.planning import prepare_multi_corpus_context
        from src.engine.synthesis_router import SynthesisMode, SynthesisContext

        synthesis_ctx = SynthesisContext(
            mode=SynthesisMode.UNIFIED,
            target_corpora=(),  # Empty
        )

        result = prepare_multi_corpus_context(
            question="What are the requirements?",
            synthesis_context=synthesis_ctx,
            resolved_profile_str="LEGAL",
            top_k=20,
        )

        # Should still return valid context
        assert result is not None
        assert result.synthesis_mode == SynthesisMode.UNIFIED

    def test_comparison_mode_includes_comparison_pairs(self):
        """COMPARISON mode includes comparison_pairs in context."""
        from src.engine.planning import prepare_multi_corpus_context
        from src.engine.synthesis_router import SynthesisMode, SynthesisContext

        synthesis_ctx = SynthesisContext(
            mode=SynthesisMode.COMPARISON,
            target_corpora=("ai_act", "gdpr"),
            comparison_pairs=(("ai_act", "gdpr"),),
        )

        result = prepare_multi_corpus_context(
            question="Compare AI-Act and GDPR",
            synthesis_context=synthesis_ctx,
            resolved_profile_str="LEGAL",
            top_k=20,
        )

        assert result.synthesis_mode == SynthesisMode.COMPARISON
        assert result.comparison_pairs is not None
        assert ("ai_act", "gdpr") in result.comparison_pairs

    def test_routing_mode_sets_routing_only(self):
        """ROUTING mode sets routing_only flag."""
        from src.engine.planning import prepare_multi_corpus_context
        from src.engine.synthesis_router import SynthesisMode, SynthesisContext

        synthesis_ctx = SynthesisContext(
            mode=SynthesisMode.ROUTING,
            target_corpora=("ai_act", "gdpr"),
            routing_only=True,
        )

        result = prepare_multi_corpus_context(
            question="Which law covers AI?",
            synthesis_context=synthesis_ctx,
            resolved_profile_str="LEGAL",
            top_k=20,
        )

        assert result.synthesis_mode == SynthesisMode.ROUTING
        assert result.routing_only is True


class TestMultiCorpusContextDataclass:
    """Tests for MultiCorpusContext dataclass."""

    def test_multi_corpus_context_has_required_fields(self):
        """MultiCorpusContext has all required fields."""
        from src.engine.planning import MultiCorpusContext
        from src.engine.synthesis_router import SynthesisMode

        ctx = MultiCorpusContext(
            question="Test question",
            synthesis_mode=SynthesisMode.AGGREGATION,
            target_corpora=("ai_act", "gdpr"),
            per_corpus_evidence={"ai_act": [], "gdpr": []},
            resolved_profile="LEGAL",
            top_k=20,
            comparison_pairs=None,
            routing_only=False,
        )

        assert ctx.question == "Test question"
        assert ctx.synthesis_mode == SynthesisMode.AGGREGATION
        assert ctx.target_corpora == ("ai_act", "gdpr")
        assert ctx.resolved_profile == "LEGAL"
