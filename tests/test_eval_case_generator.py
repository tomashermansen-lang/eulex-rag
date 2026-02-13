"""Tests for eval case generator.

TDD: These tests verify eval_case_generator.py provides correct LLM-assisted
generation of cross-law test cases.

Requirement mapping:
- ECG-001: Generates requested number of cases (R11.2)
- ECG-002: Cases have origin: auto-generated (R11.3)
- ECG-003: Auto-increments duplicate IDs (E32)
- ECG-004: Respects max 20 cases limit (R11.6)
- ECG-005: Cases have appropriate synthesis_mode (R11.5)
- ECG-006: Rejects single-corpus request (E31)
- ECG-007: Handles LLM failure gracefully (E33)
"""

from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.eval_case_generator import (
    GenerationRequest,
    GeneratedCase,
    generate_cross_law_cases,
    CaseGenerationError,
    _build_discovery_prompt,
    _build_generation_prompt,
)


def _run(coro):
    """Run async coroutine in sync test."""
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture
def mock_corpus_metadata():
    """Sample corpus metadata for generation."""
    return {
        "ai-act": {
            "name": "AI Act",
            "fullname": "Artificial Intelligence Act",
            "example_questions": [
                "What are high-risk AI systems?",
                "What are the requirements for AI system providers?",
            ],
        },
        "gdpr": {
            "name": "GDPR",
            "fullname": "General Data Protection Regulation",
            "example_questions": [
                "What are the rights of data subjects?",
                "What are the requirements for data controllers?",
            ],
        },
        "nis2": {
            "name": "NIS2",
            "fullname": "Network and Information Security Directive",
            "example_questions": [
                "What are the incident notification requirements?",
                "Who must comply with NIS2?",
            ],
        },
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response with generated cases."""
    return [
        {
            "id": "auto_compare_ai_gdpr_transparency",
            "prompt": "Compare AI-Act and GDPR transparency requirements",
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act", "gdpr"],
            "expected_anchors": [],
        },
        {
            "id": "auto_compare_ai_gdpr_rights",
            "prompt": "How do AI-Act and GDPR approach individual rights?",
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act", "gdpr"],
            "expected_anchors": [],
        },
    ]


class TestGenerationRequest:
    """Tests for GenerationRequest validation."""

    def test_request_requires_at_least_2_corpora(self):
        """Request with single corpus should raise error."""
        with pytest.raises(ValueError) as exc_info:
            GenerationRequest(
                target_corpora=("ai-act",),
                synthesis_mode="comparison",
                max_cases=5,
            )
        assert "2" in str(exc_info.value)

    def test_request_valid_with_2_corpora(self):
        """Request with 2+ corpora should be valid."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
        )
        assert request.target_corpora == ("ai-act", "gdpr")

    def test_request_caps_max_cases_at_20(self):
        """Request max_cases should be capped at 20."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=50,  # Over limit
        )
        # Should be capped at 20
        assert request.max_cases == 20


class TestGeneratedCase:
    """Tests for GeneratedCase dataclass."""

    def test_generated_case_has_origin_auto_generated(self):
        """Generated case should have origin: auto-generated."""
        case = GeneratedCase(
            id="test_case",
            prompt="Test prompt",
            synthesis_mode="comparison",
            expected_corpora=("ai-act", "gdpr"),
            expected_anchors=(),
        )
        assert case.origin == "auto-generated"

    def test_generated_case_is_frozen(self):
        """GeneratedCase should be immutable."""
        case = GeneratedCase(
            id="test",
            prompt="Test",
            synthesis_mode="comparison",
            expected_corpora=("ai-act",),
            expected_anchors=(),
        )
        with pytest.raises(AttributeError):
            case.prompt = "Modified"


class TestGenerateCrossLawCases:
    """Tests for the generate_cross_law_cases function."""

    def test_ecg_001_generates_requested_number_of_cases(
        self, mock_corpus_metadata, mock_llm_response
    ):
        """Generator should produce the requested number of cases."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=2,
        )

        with patch(
            "src.eval.eval_case_generator._call_llm_for_cases",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        assert len(cases) == 2

    def test_ecg_002_cases_have_origin_auto_generated(
        self, mock_corpus_metadata, mock_llm_response
    ):
        """All generated cases should have origin: auto-generated."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=2,
        )

        with patch(
            "src.eval.eval_case_generator._call_llm_for_cases",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        assert all(c.origin == "auto-generated" for c in cases)

    def test_ecg_003_auto_increments_duplicate_ids(self, mock_corpus_metadata):
        """Duplicate IDs should be auto-incremented."""
        # LLM returns cases with duplicate IDs
        duplicate_response = [
            {
                "id": "auto_compare_ai_gdpr",
                "prompt": "Question 1",
                "synthesis_mode": "comparison",
                "expected_corpora": ["ai-act", "gdpr"],
                "expected_anchors": [],
            },
            {
                "id": "auto_compare_ai_gdpr",  # Duplicate!
                "prompt": "Question 2",
                "synthesis_mode": "comparison",
                "expected_corpora": ["ai-act", "gdpr"],
                "expected_anchors": [],
            },
            {
                "id": "auto_compare_ai_gdpr",  # Another duplicate!
                "prompt": "Question 3",
                "synthesis_mode": "comparison",
                "expected_corpora": ["ai-act", "gdpr"],
                "expected_anchors": [],
            },
        ]

        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=3,
        )

        with patch(
            "src.eval.eval_case_generator._call_llm_for_cases",
            new_callable=AsyncMock,
            return_value=duplicate_response,
        ):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        # All IDs should be unique
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids))
        assert "auto_compare_ai_gdpr" in ids
        assert "auto_compare_ai_gdpr_1" in ids
        assert "auto_compare_ai_gdpr_2" in ids

    def test_ecg_004_respects_max_20_limit(self, mock_corpus_metadata):
        """Generator should cap at 20 cases even if LLM returns more."""
        # LLM returns 25 cases
        many_cases = [
            {
                "id": f"case_{i}",
                "prompt": f"Question {i}",
                "synthesis_mode": "comparison",
                "expected_corpora": ["ai-act", "gdpr"],
                "expected_anchors": [],
            }
            for i in range(25)
        ]

        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=30,  # Will be capped to 20
        )

        with patch(
            "src.eval.eval_case_generator._call_llm_for_cases",
            new_callable=AsyncMock,
            return_value=many_cases,
        ):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        assert len(cases) <= 20

    def test_ecg_005_cases_have_appropriate_synthesis_mode(
        self, mock_corpus_metadata, mock_llm_response
    ):
        """Cases should have the requested synthesis_mode."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=2,
        )

        with patch(
            "src.eval.eval_case_generator._call_llm_for_cases",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        assert all(c.synthesis_mode == "comparison" for c in cases)

    def test_ecg_006_rejects_single_corpus_request(self):
        """Request with single corpus should be rejected at construction."""
        with pytest.raises(ValueError) as exc_info:
            GenerationRequest(
                target_corpora=("ai-act",),
                synthesis_mode="unified",
                max_cases=5,
            )
        assert "2" in str(exc_info.value) or "at least" in str(exc_info.value).lower()

    def test_ecg_007_handles_llm_failure_gracefully(self, mock_corpus_metadata):
        """LLM failure should raise CaseGenerationError."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
        )

        with patch(
            "src.eval.eval_case_generator._call_llm_for_cases",
            new_callable=AsyncMock,
            side_effect=Exception("LLM API error"),
        ):
            with pytest.raises(CaseGenerationError) as exc_info:
                _run(generate_cross_law_cases(request, mock_corpus_metadata))

            assert "failed" in str(exc_info.value).lower()


class TestCaseGenerationModes:
    """Tests for different synthesis modes."""

    def test_aggregation_mode_generation(self, mock_corpus_metadata):
        """Aggregation mode should work correctly."""
        aggregation_response = [
            {
                "id": "auto_agg_incident_reporting",
                "prompt": "What are the incident reporting requirements across all laws?",
                "synthesis_mode": "aggregation",
                "expected_corpora": ["ai-act", "gdpr", "nis2"],
                "expected_anchors": [],
            }
        ]

        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr", "nis2"),
            synthesis_mode="aggregation",
            max_cases=1,
        )

        with patch(
            "src.eval.eval_case_generator._call_llm_for_cases",
            new_callable=AsyncMock,
            return_value=aggregation_response,
        ):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        assert len(cases) == 1
        assert cases[0].synthesis_mode == "aggregation"

    def test_routing_mode_generation(self, mock_corpus_metadata):
        """Routing mode should work correctly."""
        routing_response = [
            {
                "id": "auto_routing_cloud_providers",
                "prompt": "Which law covers cloud service providers?",
                "synthesis_mode": "routing",
                "expected_corpora": ["nis2"],
                "expected_anchors": [],
            }
        ]

        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr", "nis2"),
            synthesis_mode="routing",
            max_cases=1,
        )

        with patch(
            "src.eval.eval_case_generator._call_llm_for_cases",
            new_callable=AsyncMock,
            return_value=routing_response,
        ):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        assert len(cases) == 1
        assert cases[0].synthesis_mode == "routing"


class TestTopicHints:
    """Tests for topic hint functionality."""

    def test_topic_hints_passed_to_llm(self, mock_corpus_metadata, mock_llm_response):
        """Topic hints should be included in LLM prompt."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=2,
            topic_hints=("transparency", "automated decisions"),
        )

        with patch(
            "src.eval.eval_case_generator._call_llm_for_cases",
            new_callable=AsyncMock,
            return_value=mock_llm_response,
        ) as mock_call:
            _run(generate_cross_law_cases(request, mock_corpus_metadata))

        # Verify the function was called
        mock_call.assert_called_once()


# =========================================================================
# C1: Tests for _call_llm_for_cases implementation (replacing stub)
# =========================================================================

import json

from src.eval.eval_case_generator import (
    _call_llm_for_cases,
    _build_generation_prompt,
    assign_test_types,
    DEFAULT_DISTRIBUTION,
)


class TestCallLlmForCases:
    """Tests for the actual _call_llm_for_cases implementation (CG-001 to CG-006)."""

    def test_cg_001_calls_llm_and_parses_json_response(self):
        """Should call call_generation_llm and parse JSON response."""
        json_cases = [
            {"id": "test_1", "prompt": "Q1", "expected_corpora": ["ai-act"]},
            {"id": "test_2", "prompt": "Q2", "expected_corpora": ["gdpr"]},
        ]
        llm_response = json.dumps({"cases": json_cases})

        with patch(
            "src.eval.eval_case_generator.call_generation_llm",
            return_value=llm_response,
        ):
            result = _run(_call_llm_for_cases("test prompt", max_cases=2, synthesis_mode="comparison"))

        assert len(result) == 2
        assert result[0]["id"] == "test_1"

    def test_cg_002_retries_on_parse_failure(self):
        """Should retry when JSON parsing fails."""
        bad_response = "not json"
        json_cases = [{"id": "test_1", "prompt": "Q1", "expected_corpora": []}]
        good_response = json.dumps({"cases": json_cases})

        call_count = 0

        def mock_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_response
            return good_response

        with patch(
            "src.eval.eval_case_generator.call_generation_llm",
            side_effect=mock_llm,
        ):
            result = _run(_call_llm_for_cases("test prompt", max_cases=1, synthesis_mode="comparison"))

        assert len(result) == 1
        assert call_count == 2

    def test_cg_003_raises_after_all_retries_fail(self):
        """Should raise CaseGenerationError after exhausting retries."""
        with patch(
            "src.eval.eval_case_generator.call_generation_llm",
            return_value=None,
        ):
            with pytest.raises(CaseGenerationError):
                _run(_call_llm_for_cases("test prompt", max_cases=5, synthesis_mode="comparison"))

    def test_cg_004_prompt_includes_article_index(self, mock_corpus_metadata):
        """Prompt should include article index from citation graph for each corpus."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
        )

        mock_graph = MagicMock()
        node = MagicMock()
        node.article_id = "1"
        node.node_type = "article"
        node.title = "Test Article"
        mock_graph.nodes = {"1": node}

        with patch(
            "src.eval.eval_case_generator.load_citation_graph",
            return_value=mock_graph,
        ) as mock_load:
            prompt = _build_generation_prompt(request, mock_corpus_metadata)

        # Should call load_citation_graph for each corpus
        assert mock_load.call_count == 2
        assert "Artikel 1" in prompt

    def test_cg_005_prompt_requests_danish_output(self, mock_corpus_metadata):
        """Prompt should request Danish prompts."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
        )

        with patch(
            "src.eval.eval_case_generator.load_citation_graph",
            return_value=None,
        ):
            prompt = _build_generation_prompt(request, mock_corpus_metadata)

        assert "dansk" in prompt.lower() or "danish" in prompt.lower()

    def test_cg_006_prompt_includes_json_schema(self, mock_corpus_metadata):
        """Prompt should include explicit JSON output schema."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
        )

        with patch(
            "src.eval.eval_case_generator.load_citation_graph",
            return_value=None,
        ):
            prompt = _build_generation_prompt(request, mock_corpus_metadata)

        assert "json" in prompt.lower()
        assert "cases" in prompt.lower()

    def test_cg_007_generated_case_has_test_types(self):
        """GeneratedCase should have test_types field."""
        case = GeneratedCase(
            id="test",
            prompt="Test",
            synthesis_mode="comparison",
            expected_corpora=("ai-act",),
            expected_anchors=(),
            test_types=("retrieval", "corpus_coverage"),
        )
        assert case.test_types == ("retrieval", "corpus_coverage")

    def test_cg_007b_generated_case_default_test_types_empty(self):
        """GeneratedCase test_types should default to empty tuple."""
        case = GeneratedCase(
            id="test",
            prompt="Test",
            synthesis_mode="comparison",
            expected_corpora=("ai-act",),
            expected_anchors=(),
        )
        assert case.test_types == ()


class TestAssignTestTypes:
    """Tests for test type distribution logic (TD-001 to TD-005)."""

    def test_td_001_default_distribution_sums_to_15(self):
        """Default distribution should sum to 15."""
        assert sum(DEFAULT_DISTRIBUTION.values()) == 15

    def test_td_002_custom_distribution_applied(self):
        """Custom distribution should assign correct test types to cases."""
        cases = [
            GeneratedCase(id=f"c_{i}", prompt=f"Q{i}", synthesis_mode="comparison",
                          expected_corpora=("ai-act", "gdpr"), expected_anchors=())
            for i in range(5)
        ]
        distribution = {"corpus_coverage": 3, "synthesis_balance": 2}

        result = assign_test_types(cases, distribution)

        coverage_count = sum(1 for c in result if "corpus_coverage" in c.test_types)
        balance_count = sum(1 for c in result if "synthesis_balance" in c.test_types)
        assert coverage_count == 3
        assert balance_count == 2

    def test_td_003_base_types_added_to_non_abstention(self):
        """Non-abstention cases should get base types (retrieval, faithfulness, relevancy)."""
        cases = [
            GeneratedCase(id="c_0", prompt="Q", synthesis_mode="comparison",
                          expected_corpora=("ai-act",), expected_anchors=())
        ]
        distribution = {"corpus_coverage": 1}

        result = assign_test_types(cases, distribution)

        assert "retrieval" in result[0].test_types
        assert "faithfulness" in result[0].test_types
        assert "relevancy" in result[0].test_types

    def test_td_004_abstention_cases_no_base_types(self):
        """Abstention cases should NOT get base types."""
        cases = [
            GeneratedCase(id="c_0", prompt="Q", synthesis_mode="comparison",
                          expected_corpora=("ai-act",), expected_anchors=())
        ]
        distribution = {"abstention": 1}

        result = assign_test_types(cases, distribution)

        assert "abstention" in result[0].test_types
        assert "retrieval" not in result[0].test_types
        assert "faithfulness" not in result[0].test_types
        assert "relevancy" not in result[0].test_types

    def test_td_005_total_respects_distribution_count(self):
        """assign_test_types should handle fewer cases than distribution total."""
        cases = [
            GeneratedCase(id=f"c_{i}", prompt=f"Q{i}", synthesis_mode="comparison",
                          expected_corpora=("ai-act",), expected_anchors=())
            for i in range(3)
        ]
        # Distribution sums to 5 but only 3 cases
        distribution = {"corpus_coverage": 3, "synthesis_balance": 2}

        result = assign_test_types(cases, distribution)

        # All 3 cases should get types, none should be empty
        assert all(len(c.test_types) > 0 for c in result)


# =========================================================================
# R5: Tests for _build_article_index (citation graph-based article index)
# =========================================================================

from unittest.mock import MagicMock
from src.eval.eval_case_generator import _build_article_index


class TestBuildArticleIndex:
    """Tests for _build_article_index — R5."""

    def _make_mock_graph(self, nodes_data: dict) -> MagicMock:
        """Helper: create a mock CitationGraph with given nodes."""
        mock_graph = MagicMock()
        nodes = {}
        for key, data in nodes_data.items():
            node = MagicMock()
            node.article_id = key
            node.node_type = data.get("type", "article")
            node.title = data.get("title", "")
            nodes[key] = node
        mock_graph.nodes = nodes
        return mock_graph

    def test_build_article_index_all_articles(self):
        """All articles from citation graph appear in output."""
        graph = self._make_mock_graph({
            "1": {"type": "article", "title": "Genstand og formål"},
            "2": {"type": "article", "title": "Anvendelsesområde"},
            "3": {"type": "article", "title": "Definitioner"},
            "4": {"type": "article", "title": "Risikokategorier"},
            "5": {"type": "article", "title": "Forbudte praksisser"},
        })

        with patch("src.eval.eval_case_generator.load_citation_graph", return_value=graph):
            result = _build_article_index("test-corpus")

        assert "Artikel 1:" in result
        assert "Artikel 2:" in result
        assert "Artikel 3:" in result
        assert "Artikel 4:" in result
        assert "Artikel 5:" in result
        assert "Genstand og formål" in result

    def test_build_article_index_excludes_annex_subnodes(self):
        """annex_point and annex_section nodes are excluded."""
        graph = self._make_mock_graph({
            "1": {"type": "article", "title": "Scope"},
            "ANNEX:I": {"type": "annex", "title": "Top-level annex"},
            "ANNEX:I:A:1": {"type": "annex_point", "title": "Sub-point"},
            "ANNEX:I:Section1": {"type": "annex_section", "title": "Section"},
        })

        with patch("src.eval.eval_case_generator.load_citation_graph", return_value=graph):
            result = _build_article_index("test-corpus")

        assert "Artikel 1:" in result
        assert "Bilag I" in result
        assert "Sub-point" not in result
        assert "Section" not in result

    def test_build_article_index_includes_top_level_annexes(self):
        """Top-level annex nodes (ANNEX:III) appear as Bilag III."""
        graph = self._make_mock_graph({
            "ANNEX:III": {"type": "annex", "title": "Højrisiko-AI-systemer"},
        })

        with patch("src.eval.eval_case_generator.load_citation_graph", return_value=graph):
            result = _build_article_index("test-corpus")

        assert "Bilag III" in result
        assert "Højrisiko-AI-systemer" in result

    def test_build_article_index_fallback_no_graph(self):
        """Falls back to load_article_content when citation graph is None."""
        with patch("src.eval.eval_case_generator.load_citation_graph", return_value=None), \
             patch("src.eval.eval_case_generator.load_article_content", return_value="fallback content") as mock_fallback:
            result = _build_article_index("missing-corpus")

        mock_fallback.assert_called_once_with("missing-corpus")
        assert result == "fallback content"

    def test_prompt_instructs_empty_anchors(self, mock_corpus_metadata):
        """Generation prompt instructs LLM to leave expected_anchors empty."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
        )

        mock_graph = self._make_mock_graph({
            "1": {"type": "article", "title": "Scope"},
        })

        with patch("src.eval.eval_case_generator.load_citation_graph", return_value=mock_graph):
            prompt = _build_generation_prompt(request, mock_corpus_metadata)

        assert "expected_anchors" in prompt
        assert "empty" in prompt.lower() or "[]" in prompt


# =========================================================================
# R6: Tests for _validate_anchors (post-generation anchor validation)
# =========================================================================

from src.eval.eval_case_generator import _validate_anchors


class TestValidateAnchors:
    """Tests for _validate_anchors — R6."""

    def _make_mock_graph(self, node_keys: set[str]) -> MagicMock:
        """Helper: create a mock CitationGraph with given node keys."""
        mock_graph = MagicMock()
        mock_graph.nodes = {k: MagicMock() for k in node_keys}
        return mock_graph

    def test_validate_anchors_drops_invalid(self):
        """Invalid anchor (article:999) is dropped."""
        graph = self._make_mock_graph({"13"})
        cache = {"gdpr": graph}

        result = _validate_anchors(
            anchors=("article:13", "article:999"),
            expected_corpora=("gdpr",),
            graph_cache=cache,
        )
        assert result == ("article:13",)

    def test_validate_anchors_keeps_valid(self):
        """Both valid anchors are kept."""
        graph = self._make_mock_graph({"6", "5"})
        cache = {"ai-act": graph}

        result = _validate_anchors(
            anchors=("article:6", "article:5"),
            expected_corpora=("ai-act",),
            graph_cache=cache,
        )
        assert result == ("article:6", "article:5")

    def test_validate_anchors_no_graphs_skips(self):
        """When all graphs are None, all anchors are kept."""
        cache = {"gdpr": None, "ai-act": None}

        result = _validate_anchors(
            anchors=("article:13", "article:999"),
            expected_corpora=("gdpr", "ai-act"),
            graph_cache=cache,
        )
        assert result == ("article:13", "article:999")

    def test_validate_anchors_annex_format(self):
        """annex:III maps to ANNEX:III for graph lookup."""
        graph = self._make_mock_graph({"ANNEX:III"})
        cache = {"ai-act": graph}

        result = _validate_anchors(
            anchors=("annex:III",),
            expected_corpora=("ai-act",),
            graph_cache=cache,
        )
        assert result == ("annex:III",)

    def test_validate_anchors_cross_corpus(self):
        """Anchor valid in one corpus but not the other is kept."""
        gdpr_graph = self._make_mock_graph({"13"})
        ai_graph = self._make_mock_graph({"6"})  # No article 13
        cache = {"gdpr": gdpr_graph, "ai-act": ai_graph}

        result = _validate_anchors(
            anchors=("article:13",),
            expected_corpora=("gdpr", "ai-act"),
            graph_cache=cache,
        )
        assert result == ("article:13",)

    def test_validate_anchors_all_invalid_returns_empty(self):
        """All invalid anchors → empty tuple."""
        graph = self._make_mock_graph({"1", "2"})
        cache = {"gdpr": graph}

        result = _validate_anchors(
            anchors=("article:998", "article:999"),
            expected_corpora=("gdpr",),
            graph_cache=cache,
        )
        assert result == ()


# =========================================================================
# Phase 7: Tests for assign_difficulty (R3.2)
# =========================================================================

from src.eval.eval_case_generator import (
    assign_difficulty,
    _build_discovery_prompt,
    _distribute_by_synthesis_mode,
)


class TestAssignDifficulty:
    """Tests for difficulty assignment based on structural complexity."""

    def test_da_001_discovery_mode_is_hard(self):
        """DA-001: discovery mode → 'hard'."""
        case = GeneratedCase(
            id="disc", prompt="Topic question",
            synthesis_mode="discovery",
            expected_corpora=("dora", "nis2"),
            expected_anchors=(),
        )
        assert assign_difficulty(case) == "hard"

    def test_da_002_comparison_2_corpora_single_anchor_easy(self):
        """DA-002: comparison + 2 corpora + single-article anchors → 'easy'."""
        case = GeneratedCase(
            id="easy", prompt="Compare A and B",
            synthesis_mode="comparison",
            expected_corpora=("ai-act", "gdpr"),
            expected_anchors=("article:6",),
        )
        assert assign_difficulty(case) == "easy"

    def test_da_003_comparison_multi_article_medium(self):
        """DA-003: comparison + 2+ corpora + multi-article anchors → 'medium'."""
        case = GeneratedCase(
            id="med", prompt="Compare A and B on multiple points",
            synthesis_mode="comparison",
            expected_corpora=("ai-act", "gdpr"),
            expected_anchors=("article:6", "article:13", "article:21"),
        )
        assert assign_difficulty(case) == "medium"

    def test_da_004_aggregation_3_plus_corpora_medium(self):
        """DA-004: aggregation + 3+ corpora → 'medium'."""
        case = GeneratedCase(
            id="agg", prompt="What do all say?",
            synthesis_mode="aggregation",
            expected_corpora=("ai-act", "gdpr", "nis2"),
            expected_anchors=(),
        )
        assert assign_difficulty(case) == "medium"

    def test_da_005_3_plus_corpora_named_hard(self):
        """DA-005: 3+ corpora (non-discovery, non-aggregation) → 'hard'."""
        case = GeneratedCase(
            id="hard", prompt="Compare all three",
            synthesis_mode="comparison",
            expected_corpora=("ai-act", "gdpr", "nis2"),
            expected_anchors=("article:6", "article:13"),
        )
        assert assign_difficulty(case) == "hard"


# =========================================================================
# Phase 8: Tests for discovery prompt generation (R1.3, R1.4, R1.7)
# =========================================================================


class TestBuildDiscoveryPrompt:
    """Tests for _build_discovery_prompt (R1.3)."""

    def test_dp_001_no_corpus_names_in_prompt(self, mock_corpus_metadata):
        """DP-001: Discovery prompt instructs LLM to avoid law names."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="discovery",
            max_cases=5,
        )

        with patch("src.eval.eval_case_generator.load_citation_graph", return_value=None), \
             patch("src.eval.eval_case_generator.load_article_content", return_value=""):
            prompt = _build_discovery_prompt(request, mock_corpus_metadata)

        prompt_lower = prompt.lower()
        assert "without" in prompt_lower or "must not" in prompt_lower or "do not" in prompt_lower or "uden" in prompt_lower

    def test_dp_003_prompt_mentions_topic_based(self, mock_corpus_metadata):
        """DP-003: Discovery prompt is explicitly about topic-based questions."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="discovery",
            max_cases=5,
        )

        with patch("src.eval.eval_case_generator.load_citation_graph", return_value=None), \
             patch("src.eval.eval_case_generator.load_article_content", return_value=""):
            prompt = _build_discovery_prompt(request, mock_corpus_metadata)

        assert "topic" in prompt.lower() or "emne" in prompt.lower()


class TestDistributeBySynthesisMode:
    """Tests for _distribute_by_synthesis_mode (R1.7)."""

    def test_dp_004_distribution_correct_counts(self):
        """DP-004: {comparison: 0.6, discovery: 0.4} and 10 → 6 comparison, 4 discovery."""
        result = _distribute_by_synthesis_mode(
            {"comparison": 0.6, "discovery": 0.4}, max_cases=10
        )
        assert result["comparison"] == 6
        assert result["discovery"] == 4

    def test_dp_005_normalizes_when_sum_less_than_one(self):
        """DP-005: Sum < 1.0 → normalizes proportionally."""
        result = _distribute_by_synthesis_mode(
            {"comparison": 0.3, "discovery": 0.2}, max_cases=10
        )
        # 0.3/(0.3+0.2) = 0.6, 0.2/(0.3+0.2) = 0.4
        assert result["comparison"] == 6
        assert result["discovery"] == 4

    def test_dp_006_raises_when_sum_exceeds_one(self):
        """DP-006: Sum > 1.0 → raises ValueError."""
        with pytest.raises(ValueError):
            _distribute_by_synthesis_mode(
                {"comparison": 0.7, "discovery": 0.5}, max_cases=10
            )



# =========================================================================
# Phase 10: Tests for mode-based test type assignment (R1.5)
# =========================================================================

from src.eval.eval_case_generator import MODE_TEST_TYPE


class TestModeBasedTestTypes:
    """Tests for mode-based test type assignment in assign_test_types."""

    def test_tt_001_discovery_gets_corpus_coverage(self):
        """TT-001: Discovery case gets corpus_coverage + base types."""
        cases = [
            GeneratedCase(
                id="disc", prompt="Topic question",
                synthesis_mode="discovery",
                expected_corpora=("ai-act", "gdpr"),
                expected_anchors=(),
            )
        ]

        result = assign_test_types(cases)

        assert "corpus_coverage" in result[0].test_types
        assert "retrieval" in result[0].test_types
        assert "faithfulness" in result[0].test_types
        assert "relevancy" in result[0].test_types

    def test_tt_002_comparison_gets_comparison_completeness(self):
        """TT-002: Comparison case gets comparison_completeness + base types."""
        cases = [
            GeneratedCase(
                id="comp", prompt="Compare X and Y",
                synthesis_mode="comparison",
                expected_corpora=("ai-act", "gdpr"),
                expected_anchors=("article:6",),
            )
        ]

        result = assign_test_types(cases)

        assert "comparison_completeness" in result[0].test_types
        assert "retrieval" in result[0].test_types

    def test_tt_003_mixed_modes_get_respective_types(self):
        """TT-003: Mixed discovery + comparison cases get mode-specific types."""
        cases = [
            GeneratedCase(
                id="disc", prompt="Topic question",
                synthesis_mode="discovery",
                expected_corpora=("ai-act", "gdpr"),
                expected_anchors=(),
            ),
            GeneratedCase(
                id="comp", prompt="Compare X and Y",
                synthesis_mode="comparison",
                expected_corpora=("ai-act", "gdpr"),
                expected_anchors=("article:6",),
            ),
            GeneratedCase(
                id="agg", prompt="What do all say?",
                synthesis_mode="aggregation",
                expected_corpora=("ai-act", "gdpr", "nis2"),
                expected_anchors=(),
            ),
        ]

        result = assign_test_types(cases)

        assert "corpus_coverage" in result[0].test_types
        assert "comparison_completeness" in result[1].test_types
        assert "synthesis_balance" in result[2].test_types

    def test_tt_004_mode_test_type_constant_complete(self):
        """TT-004: MODE_TEST_TYPE covers all expected synthesis modes."""
        assert "discovery" in MODE_TEST_TYPE
        assert "comparison" in MODE_TEST_TYPE
        assert "aggregation" in MODE_TEST_TYPE
        assert "routing" in MODE_TEST_TYPE
        assert MODE_TEST_TYPE["discovery"] == "corpus_coverage"


# =========================================================================
# Manual test bug fixes
# =========================================================================


class TestDiscoveryPromptDispatch:
    """Bug fix: generate_cross_law_cases must use _build_discovery_prompt for discovery mode."""

    def test_discovery_mode_uses_discovery_prompt(self, mock_corpus_metadata):
        """Discovery mode must dispatch to _build_discovery_prompt, not _build_generation_prompt."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="discovery",
            max_cases=3,
        )

        with patch("src.eval.eval_case_generator._build_discovery_prompt") as mock_disc, \
             patch("src.eval.eval_case_generator._build_generation_prompt") as mock_gen, \
             patch("src.eval.eval_case_generator._call_llm_for_cases", return_value=[]), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            mock_disc.return_value = "discovery prompt"
            mock_gen.return_value = "general prompt"

            from src.eval.eval_case_generator import generate_cross_law_cases
            _run(generate_cross_law_cases(request, mock_corpus_metadata))

            mock_disc.assert_called_once()
            mock_gen.assert_not_called()

    def test_comparison_mode_uses_general_prompt(self, mock_corpus_metadata):
        """Non-discovery modes still use _build_generation_prompt."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=3,
        )

        with patch("src.eval.eval_case_generator._build_discovery_prompt") as mock_disc, \
             patch("src.eval.eval_case_generator._build_generation_prompt") as mock_gen, \
             patch("src.eval.eval_case_generator._call_llm_for_cases", return_value=[]), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            mock_disc.return_value = "discovery prompt"
            mock_gen.return_value = "general prompt"

            from src.eval.eval_case_generator import generate_cross_law_cases
            _run(generate_cross_law_cases(request, mock_corpus_metadata))

            mock_gen.assert_called_once()
            mock_disc.assert_not_called()


class TestDiscoveryPromptFormat:
    """Bug fix: _build_discovery_prompt must produce valid JSON in its example block."""

    def test_discovery_prompt_json_example_has_single_braces(self):
        """The JSON example in the discovery prompt must use single braces, not double."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="discovery",
            max_cases=5,
        )
        metadata = {
            "ai-act": {"name": "AI Act", "fullname": "AI Act"},
            "gdpr": {"name": "GDPR", "fullname": "GDPR"},
        }

        with patch("src.eval.eval_case_generator._build_article_index", return_value=""):
            prompt = _build_discovery_prompt(request, metadata)

        # The prompt must NOT contain double braces (invalid JSON)
        assert "{{" not in prompt, "Prompt contains '{{' — invalid JSON example"
        assert "}}" not in prompt, "Prompt contains '}}' — invalid JSON example"

        # The prompt MUST contain single braces (valid JSON)
        assert '{ ' in prompt or '{"' in prompt or '{\n' in prompt, "Prompt missing JSON opening brace"

    def test_generation_prompt_json_example_has_single_braces(self):
        """Sanity check: the standard generation prompt also uses single braces."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
        )
        metadata = {
            "ai-act": {"name": "AI Act", "fullname": "AI Act"},
            "gdpr": {"name": "GDPR", "fullname": "GDPR"},
        }

        with patch("src.eval.eval_case_generator._build_article_index", return_value=""):
            prompt = _build_generation_prompt(request, metadata)

        assert "{{" not in prompt, "Prompt contains '{{' — invalid JSON example"
        assert "}}" not in prompt, "Prompt contains '}}' — invalid JSON example"

    def test_discovery_prompt_includes_real_corpus_ids(self):
        """Discovery prompt must list real corpus IDs so LLM uses them in expected_corpora."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="discovery",
            max_cases=5,
        )
        metadata = {
            "ai-act": {"name": "AI Act", "fullname": "AI Act"},
            "gdpr": {"name": "GDPR", "fullname": "GDPR"},
        }

        with patch("src.eval.eval_case_generator._build_article_index", return_value=""):
            prompt = _build_discovery_prompt(request, metadata)

        # Prompt must mention the actual corpus IDs for expected_corpora usage
        assert "ai-act" in prompt, "Prompt must include corpus ID 'ai-act'"
        assert "gdpr" in prompt, "Prompt must include corpus ID 'gdpr'"
        # Must NOT use generic placeholders as the only reference
        assert "corpus-id-1" not in prompt, "Prompt must not use placeholder corpus IDs"


# =========================================================================
# Phase 1: Tests for inverted generation data layer
# =========================================================================


class TestArticleSpecAndGroup:
    """Tests for ArticleSpec and ArticleGroup dataclasses."""

    def test_article_spec_is_frozen(self):
        """ArticleSpec should be immutable."""
        from src.eval.eval_case_generator import ArticleSpec
        spec = ArticleSpec(
            corpus_id="ai-act",
            article_id="6",
            anchor_key="article:6",
            title="Klassificeringssystem",
            content_snippet="Artikel 6 handler om klassificering...",
        )
        with pytest.raises(AttributeError):
            spec.title = "Modified"

    def test_article_spec_correct_fields(self):
        """ArticleSpec has all required fields."""
        from src.eval.eval_case_generator import ArticleSpec
        spec = ArticleSpec(
            corpus_id="gdpr",
            article_id="13",
            anchor_key="article:13",
            title="Oplysningspligt",
            content_snippet="Når personoplysninger indsamles...",
        )
        assert spec.corpus_id == "gdpr"
        assert spec.article_id == "13"
        assert spec.anchor_key == "article:13"
        assert spec.title == "Oplysningspligt"
        assert spec.content_snippet == "Når personoplysninger indsamles..."

    def test_article_group_is_frozen(self):
        """ArticleGroup should be immutable."""
        from src.eval.eval_case_generator import ArticleSpec, ArticleGroup
        group = ArticleGroup(
            articles=(
                ArticleSpec("ai-act", "6", "article:6", "Title", "Content"),
            ),
            shared_role="obligations",
        )
        with pytest.raises(AttributeError):
            group.shared_role = "scope"

    def test_article_group_tuple_of_specs(self):
        """ArticleGroup contains a tuple of ArticleSpec."""
        from src.eval.eval_case_generator import ArticleSpec, ArticleGroup
        specs = (
            ArticleSpec("ai-act", "6", "article:6", "Klassificering", "AI..."),
            ArticleSpec("gdpr", "35", "article:35", "DPIA", "Beskyttelse..."),
        )
        group = ArticleGroup(articles=specs, shared_role="obligations")
        assert len(group.articles) == 2
        assert group.articles[0].corpus_id == "ai-act"
        assert group.shared_role == "obligations"


class TestGenerationRequestStrategy:
    """Tests for generation_strategy field on GenerationRequest."""

    def test_generation_strategy_defaults_to_standard(self):
        """GenerationRequest.generation_strategy should default to 'standard'."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
        )
        assert request.generation_strategy == "standard"

    def test_generation_strategy_accepts_inverted(self):
        """GenerationRequest should accept generation_strategy='inverted'."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
            generation_strategy="inverted",
        )
        assert request.generation_strategy == "inverted"


class TestGeneratedCaseRetrievalConfirmed:
    """Tests for retrieval_confirmed field on GeneratedCase."""

    def test_generated_case_retrieval_confirmed_default_none(self):
        """GeneratedCase.retrieval_confirmed defaults to None."""
        case = GeneratedCase(
            id="test",
            prompt="Test",
            synthesis_mode="comparison",
            expected_corpora=("ai-act",),
            expected_anchors=(),
        )
        assert case.retrieval_confirmed is None

    def test_generated_case_retrieval_confirmed_true(self):
        """GeneratedCase.retrieval_confirmed can be set to True."""
        case = GeneratedCase(
            id="test",
            prompt="Test",
            synthesis_mode="comparison",
            expected_corpora=("ai-act",),
            expected_anchors=("article:6",),
            retrieval_confirmed=True,
        )
        assert case.retrieval_confirmed is True


# =========================================================================
# Phase 2: Tests for article selection and chunk loading
# =========================================================================

from src.eval.eval_case_generator import ArticleSpec, ArticleGroup


class TestLoadArticleChunks:
    """Tests for _load_article_chunks()."""

    def test_load_returns_truncated_text(self, tmp_path):
        """_load_article_chunks returns text truncated to max_chars."""
        import json
        chunks_path = tmp_path / "test-corpus_chunks.jsonl"
        chunk = {
            "text": "A" * 500,
            "metadata": {"article": "6", "article_title": "Test Article"},
        }
        chunks_path.write_text(json.dumps(chunk) + "\n")

        from src.eval.eval_case_generator import _load_article_chunks
        result = _load_article_chunks(str(tmp_path), "test-corpus", ["6"], max_chars=100)
        assert "6" in result
        assert len(result["6"]) <= 103  # 100 + "..."

    def test_load_handles_missing_file(self, tmp_path):
        """_load_article_chunks returns empty dict for missing file."""
        from src.eval.eval_case_generator import _load_article_chunks
        result = _load_article_chunks(str(tmp_path), "nonexistent", ["6"])
        assert result == {}

    def test_load_filters_by_article_id(self, tmp_path):
        """_load_article_chunks only returns requested articles."""
        import json
        chunks_path = tmp_path / "test-corpus_chunks.jsonl"
        lines = [
            json.dumps({"text": "Article 6 text", "metadata": {"article": "6", "article_title": "Art 6"}}),
            json.dumps({"text": "Article 7 text", "metadata": {"article": "7", "article_title": "Art 7"}}),
            json.dumps({"text": "Article 8 text", "metadata": {"article": "8", "article_title": "Art 8"}}),
        ]
        chunks_path.write_text("\n".join(lines) + "\n")

        from src.eval.eval_case_generator import _load_article_chunks
        result = _load_article_chunks(str(tmp_path), "test-corpus", ["6", "8"])
        assert "6" in result
        assert "8" in result
        assert "7" not in result

    def test_load_concatenates_multiple_chunks(self, tmp_path):
        """_load_article_chunks concatenates text from multiple chunks for same article."""
        import json
        chunks_path = tmp_path / "test-corpus_chunks.jsonl"
        lines = [
            json.dumps({"text": "First chunk.", "metadata": {"article": "6", "article_title": "Art 6"}}),
            json.dumps({"text": "Second chunk.", "metadata": {"article": "6", "article_title": "Art 6"}}),
        ]
        chunks_path.write_text("\n".join(lines) + "\n")

        from src.eval.eval_case_generator import _load_article_chunks
        result = _load_article_chunks(str(tmp_path), "test-corpus", ["6"], max_chars=500)
        assert "First chunk." in result["6"]
        assert "Second chunk." in result["6"]


class TestSelectAnchorArticles:
    """Tests for _select_anchor_articles()."""

    def _make_mock_graph(self, roles_map: dict[str, list[str]], titles: dict[str, str] | None = None):
        """Create mock CitationGraph with role → article_id mapping."""
        mock_graph = MagicMock()
        nodes = {}
        for role, article_ids in roles_map.items():
            for aid in article_ids:
                if aid not in nodes:
                    node = MagicMock()
                    node.article_id = aid
                    node.title = (titles or {}).get(aid, f"Article {aid}")
                    node.mention_count = 5
                    node.roles = []
                    nodes[aid] = node
                nodes[aid].roles.append(role)

        mock_graph.nodes = nodes

        def get_articles_by_role(role):
            return roles_map.get(role, [])

        mock_graph.get_articles_by_role = get_articles_by_role

        def get_foundational_articles():
            result = {}
            for role, aids in roles_map.items():
                result[role] = aids[:5]
            return result

        mock_graph.get_foundational_articles = get_foundational_articles

        return mock_graph

    def test_two_corpora_shared_role_forms_group(self):
        """Two corpora with 'obligations' articles form an ArticleGroup."""
        from src.eval.eval_case_generator import _select_anchor_articles

        graph_a = self._make_mock_graph({"obligations": ["6"]}, {"6": "Risikostyring"})
        graph_b = self._make_mock_graph({"obligations": ["21"]}, {"21": "Risikostyring"})
        cache = {"ai-act": graph_a, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={"6": "Text A", "21": "Text B"}):
            groups = _select_anchor_articles(
                target_corpora=("ai-act", "gdpr"),
                max_groups=5,
                graph_cache=cache,
            )

        assert len(groups) >= 1
        obligations_group = next((g for g in groups if g.shared_role == "obligations"), None)
        assert obligations_group is not None
        assert len(obligations_group.articles) == 2

    def test_missing_citation_graph_returns_empty(self):
        """Missing citation graphs → empty list."""
        from src.eval.eval_case_generator import _select_anchor_articles

        cache = {"ai-act": None, "gdpr": None}
        groups = _select_anchor_articles(
            target_corpora=("ai-act", "gdpr"),
            max_groups=5,
            graph_cache=cache,
        )
        assert groups == []

    def test_max_groups_respected(self):
        """Number of groups is capped at max_groups."""
        from src.eval.eval_case_generator import _select_anchor_articles

        # Both corpora have all 5 roles
        roles = {"obligations": ["6"], "enforcement": ["10"], "scope": ["1"], "definitions": ["3"], "classification": ["5"]}
        graph_a = self._make_mock_graph(roles)
        graph_b = self._make_mock_graph(roles)
        cache = {"ai-act": graph_a, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={"6": "T", "10": "T", "1": "T", "3": "T", "5": "T"}):
            groups = _select_anchor_articles(
                target_corpora=("ai-act", "gdpr"),
                max_groups=2,
                graph_cache=cache,
            )

        assert len(groups) <= 2

    def test_no_role_overlap_falls_back_to_foundational(self):
        """No shared roles → falls back to foundational articles."""
        from src.eval.eval_case_generator import _select_anchor_articles

        graph_a = self._make_mock_graph({"obligations": ["6"]})
        graph_b = self._make_mock_graph({"enforcement": ["10"]})  # Different role
        cache = {"ai-act": graph_a, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={"6": "T", "10": "T"}):
            groups = _select_anchor_articles(
                target_corpora=("ai-act", "gdpr"),
                max_groups=5,
                graph_cache=cache,
            )

        # Should still produce at least one group via fallback
        assert len(groups) >= 1


# =========================================================================
# Phase 3: Tests for inverted prompt builder
# =========================================================================


class TestBuildInvertedPrompt:
    """Tests for _build_inverted_prompt()."""

    def _make_group(self) -> ArticleGroup:
        """Helper: create a sample ArticleGroup."""
        return ArticleGroup(
            articles=(
                ArticleSpec("ai-act", "6", "article:6", "Klassificering", "AI-systemer klassificeres..."),
                ArticleSpec("gdpr", "35", "article:35", "DPIA", "Konsekvensanalyse af databeskyttelse..."),
            ),
            shared_role="obligations",
        )

    def test_prompt_contains_article_content(self):
        """Inverted prompt includes article text from both corpora."""
        from src.eval.eval_case_generator import _build_inverted_prompt
        group = self._make_group()
        metadata = {"ai-act": {"name": "AI Act"}, "gdpr": {"name": "GDPR"}}
        prompt = _build_inverted_prompt(group, metadata, "comparison")
        assert "AI-systemer klassificeres" in prompt
        assert "Konsekvensanalyse af databeskyttelse" in prompt

    def test_prompt_lists_anchor_keys(self):
        """Inverted prompt lists anchor keys with titles for LLM accuracy."""
        from src.eval.eval_case_generator import _build_inverted_prompt
        group = self._make_group()
        metadata = {"ai-act": {"name": "AI Act"}, "gdpr": {"name": "GDPR"}}
        prompt = _build_inverted_prompt(group, metadata, "comparison")
        assert "article:6" in prompt
        assert "article:35" in prompt
        # Titles shown alongside anchors so LLM can judge relevance
        assert "Klassificering" in prompt
        assert "DPIA" in prompt

    def test_prompt_specifies_synthesis_mode(self):
        """Inverted prompt mentions the synthesis mode."""
        from src.eval.eval_case_generator import _build_inverted_prompt
        group = self._make_group()
        metadata = {"ai-act": {"name": "AI Act"}, "gdpr": {"name": "GDPR"}}
        prompt = _build_inverted_prompt(group, metadata, "comparison")
        assert "comparison" in prompt.lower()

    def test_prompt_json_schema_parseable(self):
        """Inverted prompt contains valid JSON schema example."""
        from src.eval.eval_case_generator import _build_inverted_prompt
        group = self._make_group()
        metadata = {"ai-act": {"name": "AI Act"}, "gdpr": {"name": "GDPR"}}
        prompt = _build_inverted_prompt(group, metadata, "comparison")
        assert '"cases"' in prompt
        assert '"expected_anchors"' in prompt
        # No double braces (broken f-string)
        assert "{{" not in prompt
        assert "}}" not in prompt

    def test_prompt_requests_danish(self):
        """Inverted prompt requests Danish output."""
        from src.eval.eval_case_generator import _build_inverted_prompt
        group = self._make_group()
        metadata = {"ai-act": {"name": "AI Act"}, "gdpr": {"name": "GDPR"}}
        prompt = _build_inverted_prompt(group, metadata, "comparison")
        assert "dansk" in prompt.lower() or "danish" in prompt.lower()

    def test_discovery_mode_avoids_law_names(self):
        """Discovery variant instructs to not mention law names."""
        from src.eval.eval_case_generator import _build_inverted_prompt
        group = self._make_group()
        metadata = {"ai-act": {"name": "AI Act"}, "gdpr": {"name": "GDPR"}}
        prompt = _build_inverted_prompt(group, metadata, "discovery")
        prompt_lower = prompt.lower()
        assert "do not" in prompt_lower or "must not" in prompt_lower or "uden" in prompt_lower


# =========================================================================
# Phase 4: Tests for wiring inverted generation into generate_cross_law_cases
# =========================================================================


class TestInvertedGeneration:
    """Tests for inverted generation path in generate_cross_law_cases."""

    def test_inverted_cases_have_nonempty_anchors(self, mock_corpus_metadata):
        """Inverted generation → cases have non-empty expected_anchors."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=1,
            generation_strategy="inverted",
        )

        mock_group = ArticleGroup(
            articles=(
                ArticleSpec("ai-act", "6", "ai-act:article:6", "Klassificering", "AI text"),
                ArticleSpec("gdpr", "35", "gdpr:article:35", "DPIA", "GDPR text"),
            ),
            shared_role="obligations",
        )

        # LLM picks both seed anchors as relevant
        llm_response = [{
            "id": "auto_inverted_obligations",
            "prompt": "Sammenlign krav",
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act", "gdpr"],
            "expected_anchors": ["ai-act:article:6", "gdpr:article:35"],
        }]

        with patch("src.eval.eval_case_generator._select_anchor_articles", return_value=[mock_group]), \
             patch("src.eval.eval_case_generator._call_llm_for_cases", new_callable=AsyncMock, return_value=llm_response), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        assert len(cases) >= 1
        assert len(cases[0].expected_anchors) > 0
        assert "ai-act:article:6" in cases[0].expected_anchors
        assert "gdpr:article:35" in cases[0].expected_anchors

    def test_inverted_anchors_validated_against_seeds(self, mock_corpus_metadata):
        """LLM anchors not in seed set are dropped; falls back to all seeds."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=1,
            generation_strategy="inverted",
        )

        mock_group = ArticleGroup(
            articles=(
                ArticleSpec("ai-act", "6", "ai-act:article:6", "Klassificering", "AI text"),
                ArticleSpec("gdpr", "35", "gdpr:article:35", "DPIA", "GDPR text"),
            ),
            shared_role="obligations",
        )

        # LLM returns WRONG anchors — none match seeds → fall back to all seeds
        llm_response = [{
            "id": "auto_inverted",
            "prompt": "Sammenlign krav",
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act", "gdpr"],
            "expected_anchors": ["article:99", "article:100"],
        }]

        with patch("src.eval.eval_case_generator._select_anchor_articles", return_value=[mock_group]), \
             patch("src.eval.eval_case_generator._call_llm_for_cases", new_callable=AsyncMock, return_value=llm_response), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        # Invalid anchors dropped, falls back to all corpus-qualified seeds
        assert "ai-act:article:6" in cases[0].expected_anchors
        assert "gdpr:article:35" in cases[0].expected_anchors
        assert "article:99" not in cases[0].expected_anchors

    def test_inverted_uses_llm_anchor_subset(self, mock_corpus_metadata):
        """LLM can select a subset of seed anchors matching its question."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=1,
            generation_strategy="inverted",
        )

        mock_group = ArticleGroup(
            articles=(
                ArticleSpec("ai-act", "6", "ai-act:article:6", "Klassificering", "AI text"),
                ArticleSpec("gdpr", "35", "gdpr:article:35", "DPIA", "GDPR text"),
            ),
            shared_role="obligations",
        )

        # LLM picks only one of the two seed anchors
        llm_response = [{
            "id": "auto_inverted",
            "prompt": "Hvad kræver AI-forordningen om klassificering?",
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act", "gdpr"],
            "expected_anchors": ["ai-act:article:6"],
        }]

        with patch("src.eval.eval_case_generator._select_anchor_articles", return_value=[mock_group]), \
             patch("src.eval.eval_case_generator._call_llm_for_cases", new_callable=AsyncMock, return_value=llm_response), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        # LLM's subset used (only the one it chose)
        assert cases[0].expected_anchors == ("ai-act:article:6",)
        assert "gdpr:article:35" not in cases[0].expected_anchors

    def test_standard_strategy_unchanged(self, mock_corpus_metadata, mock_llm_response):
        """Standard strategy still works as before."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=2,
            generation_strategy="standard",
        )

        with patch("src.eval.eval_case_generator._call_llm_for_cases", new_callable=AsyncMock, return_value=mock_llm_response), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        assert len(cases) == 2
        # Standard path: anchors come from LLM (empty in this case)
        assert cases[0].expected_anchors == ()

    def test_inverted_falls_back_when_no_graphs(self, mock_corpus_metadata, mock_llm_response):
        """Inverted with no citation graphs → falls back to standard."""
        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=2,
            generation_strategy="inverted",
        )

        with patch("src.eval.eval_case_generator._select_anchor_articles", return_value=[]), \
             patch("src.eval.eval_case_generator._call_llm_for_cases", new_callable=AsyncMock, return_value=mock_llm_response), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        # Falls back to standard path
        assert len(cases) == 2

    def test_inverted_routing_uses_llm_expected_corpora(self, mock_corpus_metadata):
        """Routing mode: expected_corpora comes from LLM output (not all seed corpora)."""
        from src.eval.eval_case_generator import ArticleSpec, ArticleGroup

        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr", "nis2"),
            synthesis_mode="routing",
            max_cases=1,
            generation_strategy="inverted",
        )

        mock_group = ArticleGroup(
            articles=(
                ArticleSpec("ai-act", "6", "ai-act:article:6", "Klassificering", "AI text"),
                ArticleSpec("gdpr", "35", "gdpr:article:35", "DPIA", "GDPR text"),
                ArticleSpec("nis2", "1", "nis2:article:1", "Scope", "NIS2 text"),
            ),
            shared_role="obligations",
        )

        # LLM decides the routing question is about ai-act and gdpr (not nis2)
        # LLM also picks only the relevant anchors
        llm_response = [{
            "id": "routing_q",
            "prompt": "Hvilken lov gælder for AI-systemer der behandler persondata?",
            "synthesis_mode": "routing",
            "expected_corpora": ["ai-act", "gdpr"],
            "expected_anchors": ["ai-act:article:6", "gdpr:article:35"],
        }]

        with patch("src.eval.eval_case_generator._select_anchor_articles", return_value=[mock_group]), \
             patch("src.eval.eval_case_generator._call_llm_for_cases", new_callable=AsyncMock, return_value=llm_response), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        # LLM's expected_corpora used (not all 3 seed corpora)
        assert set(cases[0].expected_corpora) == {"ai-act", "gdpr"}
        # LLM's anchors used (only the relevant ones it chose)
        assert "ai-act:article:6" in cases[0].expected_anchors
        assert "gdpr:article:35" in cases[0].expected_anchors
        assert "nis2:article:1" not in cases[0].expected_anchors

    def test_inverted_comparison_uses_seed_corpora(self, mock_corpus_metadata):
        """Non-routing modes: expected_corpora uses all seed corpora."""
        from src.eval.eval_case_generator import ArticleSpec, ArticleGroup

        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=1,
            generation_strategy="inverted",
        )

        mock_group = ArticleGroup(
            articles=(
                ArticleSpec("ai-act", "6", "ai-act:article:6", "Klassificering", "AI text"),
                ArticleSpec("gdpr", "35", "gdpr:article:35", "DPIA", "GDPR text"),
            ),
            shared_role="obligations",
        )

        llm_response = [{
            "id": "comp_q",
            "prompt": "Sammenlign krav",
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act"],  # LLM only says 1, but comparison needs all
            "expected_anchors": [],
        }]

        with patch("src.eval.eval_case_generator._select_anchor_articles", return_value=[mock_group]), \
             patch("src.eval.eval_case_generator._call_llm_for_cases", new_callable=AsyncMock, return_value=llm_response), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        # Comparison mode: all seed corpora expected
        assert set(cases[0].expected_corpora) == {"ai-act", "gdpr"}

    def test_inverted_generates_multiple_cases_per_group(self, mock_corpus_metadata):
        """When max_cases > groups, generate multiple cases per group."""
        from src.eval.eval_case_generator import ArticleSpec, ArticleGroup

        request = GenerationRequest(
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=4,  # 4 cases, only 2 groups → 2 per group
            generation_strategy="inverted",
        )

        mock_groups = [
            ArticleGroup(
                articles=(
                    ArticleSpec("ai-act", "6", "ai-act:article:6", "Klassificering", "AI text"),
                    ArticleSpec("gdpr", "35", "gdpr:article:35", "DPIA", "GDPR text"),
                ),
                shared_role="obligations",
            ),
            ArticleGroup(
                articles=(
                    ArticleSpec("ai-act", "1", "ai-act:article:1", "Scope", "Scope text"),
                    ArticleSpec("gdpr", "1", "gdpr:article:1", "Scope", "Scope text"),
                ),
                shared_role="scope",
            ),
        ]

        # Each LLM call returns 2 cases with relevant anchors from seeds
        llm_batch = [
            {"id": "q1", "prompt": "Spørgsmål 1", "synthesis_mode": "comparison",
             "expected_corpora": ["ai-act", "gdpr"],
             "expected_anchors": ["ai-act:article:6", "gdpr:article:35"]},
            {"id": "q2", "prompt": "Spørgsmål 2", "synthesis_mode": "comparison",
             "expected_corpora": ["ai-act", "gdpr"],
             "expected_anchors": ["ai-act:article:6"]},
        ]

        with patch("src.eval.eval_case_generator._select_anchor_articles", return_value=mock_groups), \
             patch("src.eval.eval_case_generator._call_llm_for_cases", new_callable=AsyncMock, return_value=llm_batch), \
             patch("src.eval.eval_case_generator.load_citation_graph", return_value=None):
            cases = _run(generate_cross_law_cases(request, mock_corpus_metadata))

        # Should produce 4 cases (2 groups × 2 per group)
        assert len(cases) == 4
        # All cases should have seed anchors (not from LLM)
        for case in cases:
            assert len(case.expected_anchors) > 0


# =========================================================================
# Phase 5: Tests for corpus-qualified anchor keys
# =========================================================================


class TestCorpusQualifiedAnchorKeys:
    """Verify _select_anchor_articles() produces corpus-qualified anchor keys.

    Bug fix: anchor keys like article:1 are not corpus-specific, so article:1
    from any corpus trivially matches in the AnchorScorer. Corpus-qualified
    keys like ai-act:article:6 prevent false positives.
    """

    def _make_mock_graph(self, roles_map: dict[str, list[str]], titles: dict[str, str] | None = None):
        """Create mock CitationGraph."""
        mock_graph = MagicMock()
        nodes = {}
        for role, article_ids in roles_map.items():
            for aid in article_ids:
                if aid not in nodes:
                    node = MagicMock()
                    node.title = (titles or {}).get(aid, f"Article {aid}")
                    nodes[aid] = node

        mock_graph.nodes = nodes

        def get_articles_by_role(role):
            return roles_map.get(role, [])

        mock_graph.get_articles_by_role = get_articles_by_role

        def get_foundational_articles():
            result = {}
            for role, aids in roles_map.items():
                result[role] = aids[:5]
            return result

        mock_graph.get_foundational_articles = get_foundational_articles
        return mock_graph

    def test_anchor_keys_prefixed_with_corpus_id(self):
        """_select_anchor_articles() must produce corpus-qualified anchor keys."""
        from src.eval.eval_case_generator import _select_anchor_articles

        graph_a = self._make_mock_graph({"obligations": ["6"]}, {"6": "Risikostyring"})
        graph_b = self._make_mock_graph({"obligations": ["21"]}, {"21": "Risikostyring"})
        cache = {"ai-act": graph_a, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={"6": "Text A", "21": "Text B"}):
            groups = _select_anchor_articles(
                target_corpora=("ai-act", "gdpr"),
                max_groups=5,
                graph_cache=cache,
            )

        assert len(groups) >= 1
        for group in groups:
            for spec in group.articles:
                assert spec.anchor_key.startswith(f"{spec.corpus_id}:"), \
                    f"anchor_key '{spec.anchor_key}' must start with '{spec.corpus_id}:'"

    def test_article_anchor_key_format(self):
        """Article anchor keys follow {corpus_id}:article:{N} format."""
        from src.eval.eval_case_generator import _select_anchor_articles

        graph_a = self._make_mock_graph({"scope": ["1"]})
        graph_b = self._make_mock_graph({"scope": ["1"]})
        cache = {"data-act": graph_a, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={"1": "Text"}):
            groups = _select_anchor_articles(
                target_corpora=("data-act", "gdpr"),
                max_groups=5,
                graph_cache=cache,
            )

        assert len(groups) >= 1
        keys = [spec.anchor_key for g in groups for spec in g.articles]
        assert "data-act:article:1" in keys
        assert "gdpr:article:1" in keys
        # Plain format must NOT appear
        assert "article:1" not in keys

    def test_annex_anchor_key_corpus_qualified(self):
        """Annex anchor keys follow {corpus_id}:annex:{N} format."""
        from src.eval.eval_case_generator import _select_anchor_articles

        graph = MagicMock()
        node = MagicMock()
        node.title = "Annex III"
        graph.nodes = {"ANNEX:III": node}
        graph.get_articles_by_role = lambda role: ["ANNEX:III"] if role == "obligations" else []
        graph.get_foundational_articles = lambda: {"obligations": ["ANNEX:III"]}

        graph_b = self._make_mock_graph({"obligations": ["6"]})
        cache = {"ai-act": graph, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={"ANNEX:III": "Text", "6": "Text"}):
            groups = _select_anchor_articles(
                target_corpora=("ai-act", "gdpr"),
                max_groups=5,
                graph_cache=cache,
            )

        assert len(groups) >= 1
        ai_act_specs = [s for g in groups for s in g.articles if s.corpus_id == "ai-act"]
        assert any(s.anchor_key == "ai-act:annex:III" for s in ai_act_specs)

    def test_fallback_foundational_also_corpus_qualified(self):
        """Fallback to foundational articles also uses corpus-qualified keys."""
        from src.eval.eval_case_generator import _select_anchor_articles

        graph_a = self._make_mock_graph({"obligations": ["6"]})
        graph_b = self._make_mock_graph({"enforcement": ["10"]})  # Different role — no overlap
        cache = {"ai-act": graph_a, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={"6": "T", "10": "T"}):
            groups = _select_anchor_articles(
                target_corpora=("ai-act", "gdpr"),
                max_groups=5,
                graph_cache=cache,
            )

        assert len(groups) >= 1
        for group in groups:
            for spec in group.articles:
                assert spec.anchor_key.startswith(f"{spec.corpus_id}:"), \
                    f"Fallback anchor_key '{spec.anchor_key}' must be corpus-qualified"


# =========================================================================
# Phase 6: Tests for deep article selection diversity
# =========================================================================


class TestDeepArticleSelection:
    """Verify article selection prefers deeper, substantive articles over
    early scope/definitions articles (article 1, 2, 3).

    The current problem: _select_anchor_articles always picks article 1-3
    because get_articles_by_role sorts by article number ascending.
    These foundational articles are trivially retrieved and don't test
    the retrieval system's ability to find specific substantive content.
    """

    def _make_mock_graph(self, roles_map: dict[str, list[str]], titles: dict[str, str] | None = None):
        """Create mock CitationGraph with multiple articles per role."""
        mock_graph = MagicMock()
        nodes = {}
        for role, article_ids in roles_map.items():
            for aid in article_ids:
                if aid not in nodes:
                    node = MagicMock()
                    node.title = (titles or {}).get(aid, f"Article {aid}")
                    node.article_id = aid
                    node.mention_count = 5
                    node.roles = [role]
                    nodes[aid] = node

        mock_graph.nodes = nodes

        def get_articles_by_role(role):
            return roles_map.get(role, [])

        mock_graph.get_articles_by_role = get_articles_by_role

        def get_foundational_articles():
            result = {}
            for role, aids in roles_map.items():
                result[role] = aids[:5]
            return result

        mock_graph.get_foundational_articles = get_foundational_articles
        return mock_graph

    def test_expanded_roles_used(self):
        """_SHARED_ROLES includes procedural/rights/exemption roles for deeper articles."""
        from src.eval.eval_case_generator import _SHARED_ROLES

        # These roles map to deeper articles (10+) in most EU regulations
        assert "procedures" in _SHARED_ROLES
        assert "rights" in _SHARED_ROLES
        assert "exemptions" in _SHARED_ROLES

    def test_prefers_deep_articles_over_early(self):
        """When a role has both early and deep articles, prefer deep ones."""
        from src.eval.eval_case_generator import _select_anchor_articles

        # "obligations" has articles 1, 2, 9, 25, 47 — should prefer 25 or 47 over 1
        graph_a = self._make_mock_graph(
            {"obligations": ["1", "2", "9", "25", "47"]},
            {"1": "Scope", "2": "Definitions", "9": "Risk mgmt", "25": "Obligations", "47": "Conformity"},
        )
        graph_b = self._make_mock_graph(
            {"obligations": ["1", "2", "24", "42"]},
            {"1": "Scope", "2": "Definitions", "24": "DPO", "42": "Certification"},
        )
        cache = {"ai-act": graph_a, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={}):
            groups = _select_anchor_articles(
                target_corpora=("ai-act", "gdpr"),
                max_groups=10,
                graph_cache=cache,
            )

        # Find the obligations group
        oblig_groups = [g for g in groups if g.shared_role == "obligations"]
        assert len(oblig_groups) >= 1

        # At least one obligations group should have articles > 9
        has_deep = False
        for g in oblig_groups:
            for spec in g.articles:
                try:
                    if int(spec.article_id) > 9:
                        has_deep = True
                except ValueError:
                    pass
        assert has_deep, "obligations group should include at least one article > 9"

    def test_procedures_role_selects_deep_articles(self):
        """Procedures role naturally maps to deeper articles."""
        from src.eval.eval_case_generator import _select_anchor_articles

        graph_a = self._make_mock_graph(
            {"procedures": ["29", "43", "56"]},
            {"29": "Notification", "43": "Conformity", "56": "Sandbox"},
        )
        graph_b = self._make_mock_graph(
            {"procedures": ["35", "58"]},
            {"35": "DPIA", "58": "Cooperation"},
        )
        cache = {"ai-act": graph_a, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={}):
            groups = _select_anchor_articles(
                target_corpora=("ai-act", "gdpr"),
                max_groups=10,
                graph_cache=cache,
            )

        proc_groups = [g for g in groups if g.shared_role == "procedures"]
        assert len(proc_groups) >= 1
        # All articles should be deep (>9)
        for g in proc_groups:
            for spec in g.articles:
                assert int(spec.article_id) > 9

    def test_multiple_groups_per_role_for_diversity(self):
        """When a role has many articles, produce multiple groups for variety."""
        from src.eval.eval_case_generator import _select_anchor_articles

        # Both corpora have many obligation articles at different depths
        graph_a = self._make_mock_graph(
            {"obligations": ["1", "9", "25", "47"]},
        )
        graph_b = self._make_mock_graph(
            {"obligations": ["1", "24", "42", "55"]},
        )
        cache = {"ai-act": graph_a, "gdpr": graph_b}

        with patch("src.eval.eval_case_generator._load_article_chunks", return_value={}):
            groups = _select_anchor_articles(
                target_corpora=("ai-act", "gdpr"),
                max_groups=10,
                graph_cache=cache,
            )

        # Should get multiple groups — not just one per role
        oblig_groups = [g for g in groups if g.shared_role == "obligations"]
        assert len(oblig_groups) >= 2, \
            f"Expected multiple obligation groups for diversity, got {len(oblig_groups)}"
