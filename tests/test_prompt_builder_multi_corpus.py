"""Tests for multi-corpus prompt building.

TDD: These tests are written BEFORE the implementation.
"""

import pytest
from unittest.mock import MagicMock

from src.engine.synthesis_router import SynthesisMode


# ---------------------------------------------------------------------------
# PT-001 to PT-006: Template existence tests
# ---------------------------------------------------------------------------


class TestMultiCorpusTemplates:
    """Tests for multi-corpus prompt templates in prompt_templates.py."""

    def test_pt_001_multi_corpus_prompt_template_exists(self):
        """MULTI_CORPUS_PROMPT_TEMPLATE must be defined."""
        from src.engine import prompt_templates as PT

        assert hasattr(PT, "MULTI_CORPUS_PROMPT_TEMPLATE")
        assert isinstance(PT.MULTI_CORPUS_PROMPT_TEMPLATE, str)
        assert len(PT.MULTI_CORPUS_PROMPT_TEMPLATE) > 50

    def test_pt_002_multi_corpus_prompt_template_has_placeholders(self):
        """Template must have required placeholders."""
        from src.engine import prompt_templates as PT

        template = PT.MULTI_CORPUS_PROMPT_TEMPLATE
        assert "{common}" in template
        assert "{mode_instructions}" in template
        assert "{format_rules}" in template
        assert "{context}" in template
        assert "{question}" in template

    def test_pt_003_unified_mode_instructions_exists(self):
        """UNIFIED_MODE_INSTRUCTIONS must be defined."""
        from src.engine import prompt_templates as PT

        assert hasattr(PT, "UNIFIED_MODE_INSTRUCTIONS")
        assert isinstance(PT.UNIFIED_MODE_INSTRUCTIONS, str)
        assert len(PT.UNIFIED_MODE_INSTRUCTIONS) > 20

    def test_pt_004_aggregation_mode_instructions_exists(self):
        """AGGREGATION_MODE_INSTRUCTIONS must be defined."""
        from src.engine import prompt_templates as PT

        assert hasattr(PT, "AGGREGATION_MODE_INSTRUCTIONS")
        assert isinstance(PT.AGGREGATION_MODE_INSTRUCTIONS, str)

    def test_pt_005_comparison_mode_instructions_exists(self):
        """COMPARISON_MODE_INSTRUCTIONS must be defined."""
        from src.engine import prompt_templates as PT

        assert hasattr(PT, "COMPARISON_MODE_INSTRUCTIONS")
        assert isinstance(PT.COMPARISON_MODE_INSTRUCTIONS, str)

    def test_pt_006_routing_mode_instructions_exists(self):
        """ROUTING_MODE_INSTRUCTIONS must be defined."""
        from src.engine import prompt_templates as PT

        assert hasattr(PT, "ROUTING_MODE_INSTRUCTIONS")
        assert isinstance(PT.ROUTING_MODE_INSTRUCTIONS, str)


# ---------------------------------------------------------------------------
# PB-001 to PB-004: Display name function tests
# ---------------------------------------------------------------------------


class TestGetCorpusDisplayName:
    """Tests for get_corpus_display_name function."""

    def test_pb_001_uses_resolver_when_available(self):
        """Returns resolver.display_name_for() when resolver is provided."""
        from src.engine.prompt_builder import get_corpus_display_name

        mock_resolver = MagicMock()
        mock_resolver.display_name_for.return_value = "AI-Forordningen"

        result = get_corpus_display_name("ai_act", resolver=mock_resolver)

        assert result == "AI-Forordningen"
        mock_resolver.display_name_for.assert_called_once_with("ai_act")

    def test_pb_002_fallback_no_resolver(self):
        """Falls back to uppercase when resolver is None."""
        from src.engine.prompt_builder import get_corpus_display_name

        result = get_corpus_display_name("ai_act", resolver=None)

        assert result == "AI-ACT"

    def test_pb_003_fallback_unknown_corpus(self):
        """Falls back when resolver returns None for unknown corpus."""
        from src.engine.prompt_builder import get_corpus_display_name

        mock_resolver = MagicMock()
        mock_resolver.display_name_for.return_value = None

        result = get_corpus_display_name("unknown_law", resolver=mock_resolver)

        assert result == "UNKNOWN-LAW"

    def test_pb_004_normalizes_underscores(self):
        """Converts underscores to dashes in fallback."""
        from src.engine.prompt_builder import get_corpus_display_name

        result = get_corpus_display_name("my_custom_regulation", resolver=None)

        assert result == "MY-CUSTOM-REGULATION"


# ---------------------------------------------------------------------------
# PB-020 to PB-027: Multi-corpus prompt dispatcher tests
# ---------------------------------------------------------------------------


class TestBuildMultiCorpusPrompt:
    """Tests for build_multi_corpus_prompt dispatcher."""

    def test_pb_020_includes_grounding_rules(self):
        """Output contains COMMON_GROUNDING_RULES content."""
        from src.engine.prompt_builder import build_multi_corpus_prompt
        from src.engine import prompt_templates as PT

        result = build_multi_corpus_prompt(
            mode=SynthesisMode.UNIFIED,
            question="Test question",
            context="Test context",
            kilder_block="KILDER: [1] Test",
            references_structured=[{"corpus_id": "ai_act", "idx": 1}],
            user_profile="LEGAL",
        )

        # Should contain key phrases from COMMON_GROUNDING_RULES
        assert "KILDER" in result or "kilder" in result.lower()
        assert "UDELUKKENDE" in result or "udelukkende" in result.lower()

    def test_pb_021_includes_format_rules(self):
        """Output contains profile-specific format rules."""
        from src.engine.prompt_builder import build_multi_corpus_prompt

        result = build_multi_corpus_prompt(
            mode=SynthesisMode.UNIFIED,
            question="Test question",
            context="Test context",
            kilder_block="KILDER: [1] Test",
            references_structured=[{"corpus_id": "ai_act", "idx": 1}],
            user_profile="LEGAL",
        )

        # LEGAL profile should have legal format rules
        assert "MÅLGRUPPE" in result or "Juridisk" in result.lower() or "jurist" in result.lower()

    def test_pb_022_unified_mode_uses_unified_instructions(self):
        """UNIFIED mode includes unified-specific instructions."""
        from src.engine.prompt_builder import build_multi_corpus_prompt

        result = build_multi_corpus_prompt(
            mode=SynthesisMode.UNIFIED,
            question="Test question",
            context="Test context",
            kilder_block="KILDER: [1] Test",
            references_structured=[{"corpus_id": "ai_act", "idx": 1}],
            user_profile="LEGAL",
        )

        # Should have unified mode language
        assert "UNIFIED" in result or "unified" in result.lower() or "samlet" in result.lower()

    def test_pb_023_aggregation_mode_uses_aggregation_instructions(self):
        """AGGREGATION mode includes aggregation-specific instructions."""
        from src.engine.prompt_builder import build_multi_corpus_prompt

        result = build_multi_corpus_prompt(
            mode=SynthesisMode.AGGREGATION,
            question="Test question",
            context="Test context",
            kilder_block="KILDER: [1] Test",
            references_structured=[
                {"corpus_id": "ai_act", "idx": 1, "chunk_text": "AI text", "article": "5"},
                {"corpus_id": "gdpr", "idx": 2, "chunk_text": "GDPR text", "article": "6"},
            ],
            user_profile="LEGAL",
        )

        # Should have aggregation mode language
        assert "AGGREGATION" in result or "aggreg" in result.lower() or "gruppér" in result.lower()

    def test_pb_024_comparison_mode_uses_comparison_instructions(self):
        """COMPARISON mode includes comparison-specific instructions."""
        from src.engine.prompt_builder import build_multi_corpus_prompt

        result = build_multi_corpus_prompt(
            mode=SynthesisMode.COMPARISON,
            question="Compare AI-Act and GDPR",
            context="Test context",
            kilder_block="KILDER: [1] Test",
            references_structured=[
                {"corpus_id": "ai_act", "idx": 1, "chunk_text": "AI text", "article": "5"},
                {"corpus_id": "gdpr", "idx": 2, "chunk_text": "GDPR text", "article": "6"},
            ],
            user_profile="LEGAL",
            comparison_corpora=["ai_act", "gdpr"],
        )

        # Should have comparison mode language
        assert "COMPARISON" in result or "comparison" in result.lower() or "sammenlign" in result.lower()

    def test_pb_025_routing_mode_uses_routing_instructions(self):
        """ROUTING mode includes routing-specific instructions."""
        from src.engine.prompt_builder import build_multi_corpus_prompt

        result = build_multi_corpus_prompt(
            mode=SynthesisMode.ROUTING,
            question="Which law covers AI?",
            context="Test context",
            kilder_block="KILDER: [1] Test",
            references_structured=[{"corpus_id": "ai_act", "idx": 1}],
            user_profile="LEGAL",
        )

        # Should have routing mode language
        assert "ROUTING" in result or "routing" in result.lower() or "identificer" in result.lower()

    def test_pb_026_includes_context(self):
        """Output includes provided context."""
        from src.engine.prompt_builder import build_multi_corpus_prompt

        result = build_multi_corpus_prompt(
            mode=SynthesisMode.UNIFIED,
            question="Test question",
            context="UNIQUE_CONTEXT_STRING_12345",
            kilder_block="KILDER: [1] Test",
            references_structured=[{"corpus_id": "ai_act", "idx": 1}],
            user_profile="LEGAL",
        )

        assert "UNIQUE_CONTEXT_STRING_12345" in result

    def test_pb_027_includes_question(self):
        """Output includes provided question."""
        from src.engine.prompt_builder import build_multi_corpus_prompt

        result = build_multi_corpus_prompt(
            mode=SynthesisMode.UNIFIED,
            question="UNIQUE_QUESTION_STRING_67890",
            context="Test context",
            kilder_block="KILDER: [1] Test",
            references_structured=[{"corpus_id": "ai_act", "idx": 1}],
            user_profile="LEGAL",
        )

        assert "UNIQUE_QUESTION_STRING_67890" in result


class TestFormatReferenceDisplay:
    """Tests for format_reference_display function."""

    def test_pbm_001_includes_corpus_prefix(self):
        """format_reference_display includes corpus prefix when enabled."""
        from src.engine.prompt_builder import format_reference_display

        ref = {
            "article": "6",
            "corpus_id": "ai_act",
            "idx": 1,
        }

        result = format_reference_display(ref, include_corpus=True)

        # Accept any valid display format: "AI-Act", "AI-ACT", or lowercase
        assert "ai" in result.lower() and "act" in result.lower()
        assert "Artikel 6" in result or "Article 6" in result

    def test_pbm_002_omits_corpus_when_disabled(self):
        """format_reference_display omits corpus when disabled."""
        from src.engine.prompt_builder import format_reference_display

        ref = {
            "article": "6",
            "corpus_id": "ai_act",
            "idx": 1,
        }

        result = format_reference_display(ref, include_corpus=False)

        assert "ai_act" not in result.lower()
        assert "ai-act" not in result.lower()


class TestBuildSynthesisPrompts:
    """Tests for synthesis-mode specific prompt builders."""

    def test_pbm_003_aggregation_prompt_groups_by_corpus(self):
        """build_aggregation_prompt groups chunks by corpus with headers."""
        from src.engine.prompt_builder import build_aggregation_prompt

        chunks_by_corpus = {
            "ai_act": [
                {"idx": 1, "chunk_text": "AI Act provision", "article": "5"},
            ],
            "gdpr": [
                {"idx": 2, "chunk_text": "GDPR provision", "article": "6"},
            ],
        }

        result = build_aggregation_prompt(
            question="What do all laws say about automated decisions?",
            chunks_by_corpus=chunks_by_corpus,
            user_profile="LEGAL",
        )

        # Should have separate sections for each law
        assert "AI-Act" in result or "ai_act" in result
        assert "GDPR" in result or "gdpr" in result

    def test_pbm_004_comparison_prompt_has_both_sections(self):
        """build_comparison_prompt has distinct sections for each law."""
        from src.engine.prompt_builder import build_comparison_prompt

        result = build_comparison_prompt(
            question="Compare AI-Act and GDPR on automated decisions",
            corpus_a_name="AI-Act",
            corpus_a_chunks=[{"idx": 1, "chunk_text": "AI Act text", "article": "5"}],
            corpus_b_name="GDPR",
            corpus_b_chunks=[{"idx": 2, "chunk_text": "GDPR text", "article": "22"}],
            user_profile="LEGAL",
        )

        # Should have distinct comparison sections
        assert "AI-Act" in result
        assert "GDPR" in result
        assert "sammenlign" in result.lower() or "compare" in result.lower() or "forskell" in result.lower()

    def test_pbm_005_routing_prompt_is_lightweight(self):
        """build_routing_prompt returns brief format for law identification."""
        from src.engine.prompt_builder import build_routing_prompt

        corpus_matches = {
            "ai_act": 5,  # 5 relevant chunks
            "gdpr": 2,  # 2 relevant chunks
            "nis2": 0,  # no relevant chunks
        }

        result = build_routing_prompt(
            question="Which law covers AI systems in healthcare?",
            corpus_matches=corpus_matches,
            user_profile="LEGAL",
        )

        # Should be concise and focus on identifying relevant laws
        assert len(result) < 2000  # Lightweight
        assert "ai_act" in result.lower() or "ai-act" in result.lower()

    def test_pbm_006_aggregation_instructs_citation_per_law(self):
        """Aggregation prompt instructs to cite from each law."""
        from src.engine.prompt_builder import build_aggregation_prompt

        chunks_by_corpus = {
            "ai_act": [{"idx": 1, "chunk_text": "AI", "article": "5"}],
            "gdpr": [{"idx": 2, "chunk_text": "GDPR", "article": "6"}],
        }

        result = build_aggregation_prompt(
            question="What do laws say?",
            chunks_by_corpus=chunks_by_corpus,
            user_profile="LEGAL",
        )

        # Should instruct to cite from each law
        assert "cit" in result.lower() or "kilde" in result.lower() or "[" in result

    def test_pbm_007_comparison_instructs_similarities_differences(self):
        """Comparison prompt instructs to identify similarities and differences."""
        from src.engine.prompt_builder import build_comparison_prompt

        result = build_comparison_prompt(
            question="Compare AI-Act and GDPR",
            corpus_a_name="AI-Act",
            corpus_a_chunks=[{"idx": 1, "chunk_text": "AI", "article": "5"}],
            corpus_b_name="GDPR",
            corpus_b_chunks=[{"idx": 2, "chunk_text": "GDPR", "article": "22"}],
            user_profile="LEGAL",
        )

        # Should mention comparing/contrasting
        result_lower = result.lower()
        has_comparison_language = (
            "forskel" in result_lower or
            "lighed" in result_lower or
            "similar" in result_lower or
            "differ" in result_lower or
            "sammenlign" in result_lower or
            "compare" in result_lower
        )
        assert has_comparison_language


class TestAnchorLabelCorpusPrefix:
    """Tests for corpus prefix in anchor labels."""

    def test_anchor_label_with_corpus(self):
        """anchor_label_for_prompt can include corpus prefix."""
        from src.engine.prompt_builder import anchor_label_for_prompt_multi_corpus

        ref = {
            "article": "6",
            "paragraph": "1",
            "corpus_id": "ai_act",
        }

        result = anchor_label_for_prompt_multi_corpus(ref, include_corpus=True)

        # Accept any valid display format: "AI-Act", "AI-ACT", or lowercase
        assert "ai" in result.lower() and "act" in result.lower()
        assert "Artikel 6" in result
        assert "stk. 1" in result
