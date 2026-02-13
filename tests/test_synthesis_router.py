"""Tests for synthesis mode detection.

TDD: These tests are written BEFORE the implementation.
"""

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def import_module():
    """Import the module under test."""
    from src.engine.synthesis_router import (
        SynthesisMode,
        SynthesisContext,
        detect_synthesis_mode,
    )
    return {
        "SynthesisMode": SynthesisMode,
        "SynthesisContext": SynthesisContext,
        "detect_synthesis_mode": detect_synthesis_mode,
    }


@pytest.fixture
def mock_resolver():
    """Create a mock CorpusResolver."""
    resolver = MagicMock()
    resolver.by_key = {
        "ai_act": MagicMock(corpus_key="ai_act", display_name="AI-Act"),
        "gdpr": MagicMock(corpus_key="gdpr", display_name="GDPR"),
        "nis2": MagicMock(corpus_key="nis2", display_name="NIS2"),
        "dora": MagicMock(corpus_key="dora", display_name="DORA"),
    }
    resolver.get_all_corpus_ids = MagicMock(return_value=["ai_act", "gdpr", "nis2", "dora"])
    resolver.mentioned_corpus_keys = MagicMock(return_value=[])
    return resolver


# ---------------------------------------------------------------------------
# Discovery Mode Enum Tests (SM-001, SM-002)
# ---------------------------------------------------------------------------


class TestDiscoveryModeEnum:
    """Tests for DISCOVERY synthesis mode enum value."""

    def test_sm_001_discovery_enum_exists(self, import_module):
        """SM-001: SynthesisMode.DISCOVERY exists with value 'discovery'."""
        SynthesisMode = import_module["SynthesisMode"]
        assert SynthesisMode.DISCOVERY.value == "discovery"

    def test_sm_002_existing_modes_unchanged(self, import_module):
        """SM-002: All existing SynthesisMode values are unchanged."""
        SynthesisMode = import_module["SynthesisMode"]
        assert SynthesisMode.SINGLE.value == "single"
        assert SynthesisMode.AGGREGATION.value == "aggregation"
        assert SynthesisMode.COMPARISON.value == "comparison"
        assert SynthesisMode.UNIFIED.value == "unified"
        assert SynthesisMode.ROUTING.value == "routing"


# ---------------------------------------------------------------------------
# Single Mode Tests (SR-001)
# ---------------------------------------------------------------------------


class TestSingleMode:
    """Tests for SINGLE synthesis mode."""

    def test_sr_001_single_scope_returns_single_mode(self, import_module, mock_resolver):
        """corpus_scope='single' always returns SINGLE mode."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="What is Article 6?",
            corpus_scope="single",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.SINGLE


# ---------------------------------------------------------------------------
# Comparison Mode Tests (SR-002 to SR-004, SR-011)
# ---------------------------------------------------------------------------


class TestComparisonMode:
    """Tests for COMPARISON synthesis mode detection."""

    def test_sr_002_compare_x_and_y_detects_comparison(self, import_module, mock_resolver):
        """'Compare X and Y' pattern triggers COMPARISON mode."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        # Mock resolver to find mentioned corpora
        mock_resolver.mentioned_corpus_keys = MagicMock(return_value=["ai_act", "gdpr"])

        result = detect_synthesis_mode(
            question="Compare AI-Act and GDPR on automated decisions",
            corpus_scope="explicit",
            selected_corpora=["ai_act", "gdpr"],
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.COMPARISON

    def test_sr_003_vs_keyword_detects_comparison(self, import_module, mock_resolver):
        """'vs' keyword triggers COMPARISON mode."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        mock_resolver.mentioned_corpus_keys = MagicMock(return_value=["ai_act", "dora"])

        result = detect_synthesis_mode(
            question="AI-Act vs DORA penalties",
            corpus_scope="explicit",
            selected_corpora=["ai_act", "dora"],
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.COMPARISON

    def test_sr_004_danish_comparison_keywords(self, import_module, mock_resolver):
        """Danish 'forskellen mellem' triggers COMPARISON."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        mock_resolver.mentioned_corpus_keys = MagicMock(return_value=["ai_act", "gdpr"])

        result = detect_synthesis_mode(
            question="Hvad er forskellen mellem AI-Act og GDPR?",
            corpus_scope="explicit",
            selected_corpora=["ai_act", "gdpr"],
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.COMPARISON

    def test_sr_011_comparison_extracts_target_corpora(self, import_module, mock_resolver):
        """COMPARISON mode extracts target corpora from question."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        mock_resolver.mentioned_corpus_keys = MagicMock(return_value=["ai_act", "gdpr"])

        result = detect_synthesis_mode(
            question="Compare AI-Act and GDPR",
            corpus_scope="explicit",
            selected_corpora=["ai_act", "gdpr"],
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.COMPARISON
        assert "ai_act" in result.target_corpora
        assert "gdpr" in result.target_corpora


# ---------------------------------------------------------------------------
# Routing Mode Tests (SR-005 to SR-007, SR-014)
# ---------------------------------------------------------------------------


class TestRoutingMode:
    """Tests for ROUTING synthesis mode detection."""

    def test_sr_005_which_law_detects_routing(self, import_module, mock_resolver):
        """'Which law' pattern triggers ROUTING mode."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="Which law covers cloud providers?",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.ROUTING

    def test_sr_006_what_law_covers_detects_routing(self, import_module, mock_resolver):
        """'What law covers' pattern triggers ROUTING mode."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="What law covers AI systems in healthcare?",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.ROUTING

    def test_sr_007_danish_routing_keywords(self, import_module, mock_resolver):
        """Danish 'Hvilken lov' triggers ROUTING."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="Hvilken lov dækker cloud-udbydere?",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.ROUTING

    def test_sr_014_routing_sets_routing_only_flag(self, import_module, mock_resolver):
        """ROUTING mode sets routing_only=True."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="Which law governs AI?",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.ROUTING
        assert result.routing_only is True


# ---------------------------------------------------------------------------
# Aggregation Mode Tests (SR-008, SR-009)
# ---------------------------------------------------------------------------


class TestAggregationMode:
    """Tests for AGGREGATION synthesis mode detection."""

    def test_sr_008_all_laws_say_detects_aggregation(self, import_module, mock_resolver):
        """'all laws say' pattern triggers AGGREGATION mode."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="What do all laws say about incident notification?",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.AGGREGATION

    def test_sr_009_across_laws_detects_aggregation(self, import_module, mock_resolver):
        """'across laws' pattern triggers AGGREGATION mode."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="Notification deadlines across laws",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.AGGREGATION

    def test_every_law_detects_aggregation(self, import_module, mock_resolver):
        """'every law' pattern triggers AGGREGATION mode."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="What does every law require for data protection?",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.AGGREGATION

    def test_danish_aggregation_keywords(self, import_module, mock_resolver):
        """Danish 'alle love' triggers AGGREGATION."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="Hvad siger alle love om databeskyttelse?",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.AGGREGATION


# ---------------------------------------------------------------------------
# Unified Mode Tests (SR-010)
# ---------------------------------------------------------------------------


class TestUnifiedMode:
    """Tests for UNIFIED synthesis mode detection."""

    def test_sr_010_multi_corpus_without_keywords_is_unified(self, import_module, mock_resolver):
        """Multi-corpus without comparison/routing/aggregation keywords → UNIFIED."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]
        SynthesisMode = import_module["SynthesisMode"]

        result = detect_synthesis_mode(
            question="What are the penalties for non-compliance?",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert result.mode == SynthesisMode.UNIFIED


# ---------------------------------------------------------------------------
# Target Corpora Tests (SR-012, SR-013)
# ---------------------------------------------------------------------------


class TestTargetCorpora:
    """Tests for target corpora resolution."""

    def test_sr_012_explicit_scope_uses_selected_corpora(self, import_module, mock_resolver):
        """explicit scope uses the provided selected_corpora list."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]

        result = detect_synthesis_mode(
            question="What are the requirements?",
            corpus_scope="explicit",
            selected_corpora=["ai_act", "gdpr"],
            resolver=mock_resolver,
        )

        assert set(result.target_corpora) == {"ai_act", "gdpr"}

    def test_sr_013_all_scope_returns_all_corpus_ids(self, import_module, mock_resolver):
        """all scope returns all registered corpora."""
        detect_synthesis_mode = import_module["detect_synthesis_mode"]

        result = detect_synthesis_mode(
            question="What are the requirements?",
            corpus_scope="all",
            selected_corpora=None,
            resolver=mock_resolver,
        )

        assert set(result.target_corpora) == {"ai_act", "gdpr", "nis2", "dora"}
