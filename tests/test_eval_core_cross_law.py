"""Tests for cross-law scorer integration in eval_core.

TDD: These tests are written BEFORE the implementation.
"""

import json

import pytest
import yaml
from unittest.mock import MagicMock, patch


class TestCrossLawScorerIntegration:
    """Tests for cross-law scorer integration in evaluate_cases_iter."""

    def test_ecc_001_cross_law_case_applies_corpus_coverage_scorer(self):
        """Cross-law case (corpus_scope != single) applies corpus coverage scorer."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        # Create a cross-law case
        case = GoldenCase(
            id="cross_law_test",
            profile="LEGAL",
            prompt="What do AI-Act and GDPR say about data protection?",
            expected=ExpectedBehavior(
                min_corpora_cited=2,
                required_corpora=("ai_act", "gdpr"),
            ),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="aggregation",
            test_types=("corpus_coverage",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        # Mock engine and result
        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = [
            {"corpus_id": "ai_act", "article": "5"},
            {"corpus_id": "gdpr", "article": "6"},
        ]
        mock_result.answer = "Both laws address this."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        # Cross-law scorer should have been applied
        assert "corpus_coverage" in result.scores

    def test_ecc_002_single_law_case_skips_cross_law_scorers(self):
        """Single-law case (corpus_scope == single) does not apply cross-law scorers."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        # Create a single-law case
        case = GoldenCase(
            id="single_law_test",
            profile="LEGAL",
            prompt="What is Article 6?",
            expected=ExpectedBehavior(),
            corpus_scope="single",  # Default single-law
            test_types=("retrieval",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = []
        mock_result.answer = "Article 6 says..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        # Cross-law scorers should NOT be applied
        assert "corpus_coverage" not in result.scores
        assert "synthesis_balance" not in result.scores

    def test_ecc_003_cross_law_scores_appear_in_result(self):
        """Cross-law case result contains synthesis_balance when test_type includes it."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="balance_test",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR",
            expected=ExpectedBehavior(),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="aggregation",
            test_types=("synthesis_balance",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = [
            {"corpus_id": "ai_act", "article": "5"},
            {"corpus_id": "ai_act", "article": "6"},
            {"corpus_id": "ai_act", "article": "7"},
            {"corpus_id": "gdpr", "article": "12"},
            {"corpus_id": "gdpr", "article": "13"},
        ]
        mock_result.answer = "Both laws..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        assert "synthesis_balance" in result.scores

    def test_ecc_004_comparison_completeness_applied_for_comparison_mode(self):
        """Comparison completeness scorer applied for comparison synthesis mode."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="comparison_test",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR definitions",
            expected=ExpectedBehavior(),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            test_types=("comparison_completeness",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = [
            {"corpus_id": "ai_act", "article": "3"},
            {"corpus_id": "gdpr", "article": "4"},
        ]
        mock_result.answer = "AI-Act defines... GDPR defines..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        assert "comparison_completeness" in result.scores

    def test_ecc_005_routing_precision_applied_for_routing_mode(self):
        """Routing precision scorer applied when test_types includes routing_precision."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="routing_test",
            profile="LEGAL",
            prompt="Which law covers digital operational resilience?",
            expected=ExpectedBehavior(
                required_corpora=("dora",),
            ),
            corpus_scope="discover",
            target_corpora=(),
            synthesis_mode="routing",
            test_types=("routing_precision",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = [
            {"corpus_id": "dora", "article": "5"},
        ]
        mock_result.answer = "DORA covers digital operational resilience..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        assert "routing_precision" in result.scores


class TestComparisonCompletenessBugFix:
    """R4: ComparisonCompletenessScorer must use case-level expected corpora."""

    def test_bf_001_uses_expected_corpora_not_target_corpora(self):
        """BF-001: Scorer receives case.expected.required_corpora, not case.target_corpora.

        Bug: eval_core.py:298 passes set(case.target_corpora) to the scorer.
        Fix: Should pass set(case.expected.required_corpora) — the case-level ground truth.

        When target_corpora (suite pool) is larger than expected.required_corpora (case-level),
        the scorer should only check against case-level corpora.
        """
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        # Case expects only ai_act + gdpr, but suite pool includes dsa + dma too
        case = GoldenCase(
            id="bugfix_test",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR definitions",
            expected=ExpectedBehavior(
                required_corpora=("ai_act", "gdpr"),
            ),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr", "dsa", "dma"),  # Suite-level pool (larger)
            synthesis_mode="comparison",
            test_types=("comparison_completeness",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = [
            {"corpus_id": "ai_act", "article": "3"},
            {"corpus_id": "gdpr", "article": "4"},
        ]
        mock_result.answer = "AI-Act defines... GDPR defines..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        # With bug: scorer gets {ai_act, gdpr, dsa, dma} → fails (missing dsa, dma)
        # With fix: scorer gets {ai_act, gdpr} → passes (both cited)
        assert result.scores["comparison_completeness"].passed is True


class TestDiscoveryCoverageThreshold:
    """Discovery mode uses corpus_coverage with configurable threshold."""

    def test_discovery_case_uses_corpus_coverage_with_threshold(self):
        """Discovery case dispatches corpus_coverage with 0.8 threshold."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="discovery_dispatch",
            profile="LEGAL",
            prompt="Hvad kræver EU-lovgivning om ICT-risikostyring?",
            expected=ExpectedBehavior(
                required_corpora=("dora", "nis2"),
            ),
            corpus_scope="all",
            target_corpora=("dora", "nis2", "cra"),
            synthesis_mode="discovery",
            test_types=("corpus_coverage",),
        )

        config = EvalConfig(law="dora", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = [
            {"corpus_id": "dora", "article": "6"},
            {"corpus_id": "nis2", "article": "21"},
        ]
        mock_result.answer = "Both DORA and NIS2 require..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        assert "corpus_coverage" in result.scores
        assert result.scores["corpus_coverage"].passed is True
        assert result.scores["corpus_coverage"].score == 1.0

    def test_discovery_partial_coverage_fails_below_threshold(self):
        """Discovery case with partial coverage below 0.8 threshold fails."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="partial_discovery",
            profile="LEGAL",
            prompt="Topic question",
            expected=ExpectedBehavior(
                required_corpora=("dora", "nis2", "cra"),
            ),
            corpus_scope="all",
            target_corpora=("dora", "nis2", "cra"),
            synthesis_mode="discovery",
            test_types=("corpus_coverage",),
        )

        config = EvalConfig(law="dora", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = [
            {"corpus_id": "dora", "article": "6"},
            # Missing nis2 and cra
        ]
        mock_result.answer = "Only DORA..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        assert "corpus_coverage" in result.scores
        assert result.scores["corpus_coverage"].passed is False
        assert result.scores["corpus_coverage"].score == pytest.approx(1 / 3, rel=0.01)

    def test_eval_config_has_corpus_coverage_threshold(self):
        """EvalConfig has corpus_coverage_threshold with default 0.8."""
        from src.eval.eval_core import EvalConfig

        config = EvalConfig(law="test")
        assert config.corpus_coverage_threshold == 0.8


class TestCrossLawScorerExtractCorpora:
    """Tests for extracting cited corpora from references."""

    def test_extract_cited_corpora_from_references(self):
        """Extract unique corpus IDs from references_structured."""
        from src.eval.eval_core import _extract_cited_corpora

        refs = [
            {"corpus_id": "ai_act", "article": "5"},
            {"corpus_id": "gdpr", "article": "6"},
            {"corpus_id": "ai_act", "article": "6"},  # Duplicate corpus
        ]

        result = _extract_cited_corpora(refs)

        assert result == {"ai_act", "gdpr"}

    def test_extract_cited_corpora_handles_missing_corpus_id(self):
        """Handle references without corpus_id gracefully."""
        from src.eval.eval_core import _extract_cited_corpora

        refs = [
            {"corpus_id": "ai_act", "article": "5"},
            {"article": "6"},  # No corpus_id
            {"corpus_id": "", "article": "7"},  # Empty corpus_id
        ]

        result = _extract_cited_corpora(refs)

        assert result == {"ai_act"}


class TestDeriveCitationCounts:
    """Tests for _derive_citation_counts — R1."""

    def test_derive_citation_counts_from_refs(self):
        """Count references per corpus_id."""
        from src.eval.eval_core import _derive_citation_counts

        refs = [
            {"corpus_id": "ai-act", "article": "5"},
            {"corpus_id": "gdpr", "article": "6"},
            {"corpus_id": "ai-act", "article": "13"},
            {"corpus_id": "ai-act", "article": "14"},
            {"corpus_id": "gdpr", "article": "7"},
        ]
        result = _derive_citation_counts(refs)
        assert result == {"ai-act": 3, "gdpr": 2}

    def test_derive_citation_counts_empty(self):
        """Empty references → empty dict."""
        from src.eval.eval_core import _derive_citation_counts

        assert _derive_citation_counts([]) == {}

    def test_derive_citation_counts_missing_corpus_id(self):
        """References without corpus_id are excluded."""
        from src.eval.eval_core import _derive_citation_counts

        refs = [
            {"corpus_id": "ai-act", "article": "5"},
            {"article": "6"},  # No corpus_id
            {"corpus_id": "", "article": "7"},  # Empty corpus_id
        ]
        result = _derive_citation_counts(refs)
        assert result == {"ai-act": 1}

    def test_derive_citation_counts_non_dict_refs(self):
        """Non-dict entries in list are skipped."""
        from src.eval.eval_core import _derive_citation_counts

        refs = [
            {"corpus_id": "gdpr", "article": "5"},
            "not a dict",
            42,
            None,
        ]
        result = _derive_citation_counts(refs)
        assert result == {"gdpr": 1}


class TestCrossLawParamsPassedToAsk:
    """Verify _evaluate_single_case forwards cross-law params to ask.ask()."""

    def test_corpus_scope_and_target_corpora_passed_to_ask(self):
        """Cross-law case must pass corpus_scope and target_corpora to ask.ask()."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="params_test",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR",
            expected=ExpectedBehavior(
                min_corpora_cited=2,
                required_corpora=("ai_act", "gdpr"),
            ),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            test_types=("corpus_coverage",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = [
            {"corpus_id": "ai_act", "article": "5"},
            {"corpus_id": "gdpr", "article": "6"},
        ]
        mock_result.answer = "Both laws address this."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result) as mock_ask:
            _evaluate_single_case(case, config, mock_engine, scorers=[])

        # Assert corpus_scope and target_corpora were forwarded
        call_kwargs = mock_ask.call_args.kwargs
        assert call_kwargs["corpus_scope"] == "explicit", (
            "corpus_scope not passed to ask.ask()"
        )
        assert call_kwargs["target_corpora"] == ["ai_act", "gdpr"], (
            "target_corpora not passed to ask.ask()"
        )

    def test_single_law_case_passes_default_scope(self):
        """Single-law case passes corpus_scope='single' to ask.ask()."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="single_params_test",
            profile="LEGAL",
            prompt="What is Article 6?",
            expected=ExpectedBehavior(),
            corpus_scope="single",
            test_types=("retrieval",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.retrieval_metrics = {}
        mock_result.references_structured = []
        mock_result.answer = "Article 6 says..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result) as mock_ask:
            _evaluate_single_case(case, config, mock_engine, scorers=[])

        call_kwargs = mock_ask.call_args.kwargs
        assert call_kwargs["corpus_scope"] == "single"
        assert call_kwargs["target_corpora"] is None or call_kwargs["target_corpora"] == []


class TestEvalCrossLawRouteLogger:
    """Verify eval_cross_law route module has a logger defined."""

    def test_eval_cross_law_module_has_logger(self):
        """eval_cross_law.py must define a logger to avoid NameError at runtime."""
        import importlib
        mod = importlib.import_module("ui_react.backend.routes.eval_cross_law")
        assert hasattr(mod, "logger"), (
            "eval_cross_law.py is missing 'logger' - will crash on error path"
        )


class TestOverviewScorerPassRates:
    """Verify overview endpoint returns per-scorer pass rates from latest run."""

    def test_overview_includes_scorer_pass_rates(self, tmp_path):
        """CrossLawSuiteStats must include scorer_pass_rates from latest run."""
        from ui_react.backend.routes.eval_cross_law import CrossLawEvalService

        # Setup: create a suite as YAML in evals_dir (cross_law_ prefix)
        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()

        suite_data = {
            "id": "test_suite",
            "name": "Test Suite",
            "description": "test",
            "target_corpora": ["ai_act", "gdpr"],
            "cases": [
                {
                    "id": "case1",
                    "prompt": "Q1",
                    "synthesis_mode": "comparison",
                    "target_corpora": ["ai_act", "gdpr"],
                    "expected_anchors": [],
                    "test_types": ["corpus_coverage"],
                    "origin": "manual",
                },
            ],
        }
        with open(evals_dir / "cross_law_test_suite.yaml", "w") as f:
            yaml.dump(suite_data, f)

        # Setup: create a run with scorer results
        runs_dir = evals_dir / "runs"
        runs_dir.mkdir()
        run_data = {
            "run_id": "run_abc",
            "suite_id": "test_suite",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "total": 2,
            "passed": 1,
            "failed": 1,
            "pass_rate": 0.5,
            "duration_seconds": 10.0,
            "run_mode": "full",
            "results": [
                {
                    "case_id": "case1",
                    "prompt": "Q1",
                    "synthesis_mode": "comparison",
                    "target_corpora": ["ai_act", "gdpr"],
                    "passed": True,
                    "duration_ms": 5000,
                    "scores": {
                        "corpus_coverage": {"passed": True, "score": 1.0, "message": "ok"},
                        "faithfulness": {"passed": True, "score": 0.9, "message": "ok"},
                        "answer_relevancy": {"passed": False, "score": 0.3, "message": "low"},
                    },
                },
                {
                    "case_id": "case2",
                    "prompt": "Q2",
                    "synthesis_mode": "comparison",
                    "target_corpora": ["ai_act", "gdpr"],
                    "passed": False,
                    "duration_ms": 5000,
                    "scores": {
                        "corpus_coverage": {"passed": False, "score": 0.0, "message": "miss"},
                        "faithfulness": {"passed": True, "score": 0.8, "message": "ok"},
                    },
                },
            ],
        }
        (runs_dir / "run_abc.json").write_text(json.dumps(run_data))

        service = CrossLawEvalService(
            evals_dir=evals_dir,
            valid_corpus_ids={"ai_act", "gdpr"},
        )
        overview = service.get_overview()

        suite_stats = overview.suites[0]
        assert hasattr(suite_stats, "scorer_pass_rates"), (
            "CrossLawSuiteStats must have scorer_pass_rates field"
        )
        rates = suite_stats.scorer_pass_rates

        # corpus_coverage: 1 passed out of 2 = 0.5
        assert rates["corpus_coverage"] == pytest.approx(0.5)
        # faithfulness: 2 passed out of 2 = 1.0
        assert rates["faithfulness"] == pytest.approx(1.0)
        # answer_relevancy: 0 passed out of 1 = 0.0
        assert rates["answer_relevancy"] == pytest.approx(0.0)

    def test_overview_no_runs_returns_empty_scorer_rates(self, tmp_path):
        """Suite with no runs should have empty scorer_pass_rates."""
        from ui_react.backend.routes.eval_cross_law import CrossLawEvalService

        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()

        suite_data = {
            "id": "empty_suite",
            "name": "Empty",
            "description": "",
            "target_corpora": ["ai_act"],
            "cases": [],
        }
        with open(evals_dir / "cross_law_empty_suite.yaml", "w") as f:
            yaml.dump(suite_data, f)

        service = CrossLawEvalService(
            evals_dir=evals_dir,
            valid_corpus_ids={"ai_act"},
        )
        overview = service.get_overview()

        suite_stats = overview.suites[0]
        assert suite_stats.scorer_pass_rates == {}


class TestConvertCaseToGoldenTestTypes:
    """Verify _convert_case_to_golden assigns cross-law test_types for empty cases."""

    def test_empty_test_types_comparison_gets_cross_law_defaults(self, tmp_path):
        """Cross-law case with empty test_types gets corpus_coverage + comparison_completeness."""
        from ui_react.backend.routes.eval_cross_law import CrossLawEvalService
        from src.eval.cross_law_suite_manager import CrossLawGoldenCase

        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()

        service = CrossLawEvalService(
            evals_dir=evals_dir,
            valid_corpus_ids={"ai_act", "gdpr"},
        )

        case = CrossLawGoldenCase(
            id="test_case",
            prompt="Compare AI-Act and GDPR",
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("ai_act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
            test_types=(),  # Empty!
        )

        golden = service._convert_case_to_golden(case)
        assert "corpus_coverage" in golden.test_types
        assert "comparison_completeness" in golden.test_types
        assert "retrieval" in golden.test_types
        assert "faithfulness" in golden.test_types
        assert "relevancy" in golden.test_types

    def test_empty_test_types_aggregation_gets_synthesis_balance(self, tmp_path):
        """Cross-law aggregation case gets corpus_coverage + synthesis_balance."""
        from ui_react.backend.routes.eval_cross_law import CrossLawEvalService
        from src.eval.cross_law_suite_manager import CrossLawGoldenCase

        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()

        service = CrossLawEvalService(
            evals_dir=evals_dir,
            valid_corpus_ids={"ai_act"},
        )

        case = CrossLawGoldenCase(
            id="agg_case",
            prompt="What do all laws say?",
            corpus_scope="all",
            target_corpora=("ai_act",),
            synthesis_mode="aggregation",
            expected_anchors=(),
            expected_corpora=("ai_act",),
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
            test_types=(),
        )

        golden = service._convert_case_to_golden(case)
        assert "corpus_coverage" in golden.test_types
        assert "synthesis_balance" in golden.test_types

    def test_explicit_test_types_preserved(self, tmp_path):
        """When test_types are explicitly set, they are preserved as-is."""
        from ui_react.backend.routes.eval_cross_law import CrossLawEvalService
        from src.eval.cross_law_suite_manager import CrossLawGoldenCase

        evals_dir = tmp_path / "evals"
        evals_dir.mkdir()

        service = CrossLawEvalService(
            evals_dir=evals_dir,
            valid_corpus_ids={"ai_act"},
        )

        case = CrossLawGoldenCase(
            id="explicit_case",
            prompt="Q",
            corpus_scope="explicit",
            target_corpora=("ai_act",),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=(),
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
            test_types=("routing_precision",),
        )

        golden = service._convert_case_to_golden(case)
        assert golden.test_types == ("routing_precision",)


class TestRetrievalLevelCorpusScoring:
    """Bug fix: corpus_coverage, comparison_completeness, routing_precision must use
    retrieval-level refs (references_structured_all), not LLM-cited refs.

    The LLM may choose not to cite a corpus that was in the retrieval context.
    Coverage scorers measure retrieval quality, not generation quality.
    """

    def test_corpus_coverage_uses_retrieval_refs_not_cited_refs(self):
        """corpus_coverage passes when retrieval found the corpus,
        even if LLM didn't cite it."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="retrieval_coverage_test",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR data protection",
            expected=ExpectedBehavior(
                required_corpora=("ai_act", "gdpr"),
                min_corpora_cited=2,
            ),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="aggregation",
            test_types=("corpus_coverage",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        # LLM only cited ai_act (generation-level)
        mock_result.references_structured = [
            {"corpus_id": "ai_act", "article": "5"},
        ]
        # But retrieval found BOTH corpora (retrieval-level)
        mock_result.retrieval_metrics = {
            "references_structured_all": [
                {"corpus_id": "ai_act", "article": "5"},
                {"corpus_id": "ai_act", "article": "6"},
                {"corpus_id": "gdpr", "article": "32"},
                {"corpus_id": "gdpr", "article": "35"},
            ],
        }
        mock_result.answer = "AI-Act requires..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        # Should PASS because retrieval found both corpora
        assert result.scores["corpus_coverage"].passed is True
        assert result.scores["corpus_coverage"].score == 1.0

    def test_comparison_completeness_uses_retrieval_refs(self):
        """comparison_completeness passes when retrieval found both targets,
        even if LLM only cited one."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="retrieval_comparison_test",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR definitions",
            expected=ExpectedBehavior(
                required_corpora=("ai_act", "gdpr"),
            ),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            test_types=("comparison_completeness",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        # LLM only cited ai_act
        mock_result.references_structured = [
            {"corpus_id": "ai_act", "article": "3"},
        ]
        # Retrieval found both
        mock_result.retrieval_metrics = {
            "references_structured_all": [
                {"corpus_id": "ai_act", "article": "3"},
                {"corpus_id": "gdpr", "article": "4"},
            ],
        }
        mock_result.answer = "AI-Act defines..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        assert result.scores["comparison_completeness"].passed is True

    def test_discovery_coverage_uses_retrieval_refs_with_threshold(self):
        """Discovery corpus_coverage passes with 0.8 threshold when retrieval
        found expected corpora, even if LLM didn't cite all of them."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="retrieval_discovery_test",
            profile="LEGAL",
            prompt="What does EU regulation require for ICT risk?",
            expected=ExpectedBehavior(
                required_corpora=("dora", "nis2"),
            ),
            corpus_scope="all",
            target_corpora=("dora", "nis2", "cra"),
            synthesis_mode="discovery",
            test_types=("corpus_coverage",),
        )

        config = EvalConfig(law="dora", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        # LLM only cited dora
        mock_result.references_structured = [
            {"corpus_id": "dora", "article": "6"},
        ]
        # Retrieval found both dora and nis2
        mock_result.retrieval_metrics = {
            "references_structured_all": [
                {"corpus_id": "dora", "article": "6"},
                {"corpus_id": "nis2", "article": "21"},
            ],
        }
        mock_result.answer = "DORA requires..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        assert result.scores["corpus_coverage"].passed is True
        assert result.scores["corpus_coverage"].score == 1.0

    def test_synthesis_balance_still_uses_cited_refs(self):
        """synthesis_balance should still use LLM-cited refs (generation-level),
        because it measures whether the LLM balanced its citations."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="balance_cited_test",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR",
            expected=ExpectedBehavior(),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="aggregation",
            test_types=("synthesis_balance",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        # LLM cited heavily from ai_act only (imbalanced)
        mock_result.references_structured = [
            {"corpus_id": "ai_act", "article": "5"},
            {"corpus_id": "ai_act", "article": "6"},
            {"corpus_id": "ai_act", "article": "7"},
            {"corpus_id": "ai_act", "article": "8"},
        ]
        # Retrieval was balanced
        mock_result.retrieval_metrics = {
            "references_structured_all": [
                {"corpus_id": "ai_act", "article": "5"},
                {"corpus_id": "ai_act", "article": "6"},
                {"corpus_id": "gdpr", "article": "32"},
                {"corpus_id": "gdpr", "article": "35"},
            ],
        }
        mock_result.answer = "AI-Act..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        # synthesis_balance should FAIL because LLM citations are imbalanced
        # (100% ai_act > 70% threshold)
        assert result.scores["synthesis_balance"].passed is False

    def test_fallback_to_cited_refs_when_all_refs_missing(self):
        """When references_structured_all is not available, fall back to
        references_structured (backwards compatibility)."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="fallback_test",
            profile="LEGAL",
            prompt="Compare AI-Act and GDPR",
            expected=ExpectedBehavior(
                required_corpora=("ai_act", "gdpr"),
                min_corpora_cited=2,
            ),
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="aggregation",
            test_types=("corpus_coverage",),
        )

        config = EvalConfig(law="ai_act", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.references_structured = [
            {"corpus_id": "ai_act", "article": "5"},
            {"corpus_id": "gdpr", "article": "6"},
        ]
        # No references_structured_all available
        mock_result.retrieval_metrics = {}
        mock_result.answer = "Both laws..."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        # Falls back to cited refs, both present → passes
        assert result.scores["corpus_coverage"].passed is True



class TestRoutingPrecisionRetrievalLevel:
    """Fix: routing_precision should use retrieval-level refs (context_corpora)
    instead of alias-based answer text matching.

    Aligns with RAGAS IDBasedContextRecall — check structured corpus_id fields,
    not fragile text matching with an alias list.
    """

    def test_routing_precision_passes_when_retrieval_finds_expected_law(self):
        """routing_precision passes when retrieval found the expected corpus,
        regardless of what name the LLM uses in the answer text."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="routing_retrieval_test",
            profile="LEGAL",
            prompt="Which law covers digital operational resilience?",
            expected=ExpectedBehavior(
                required_corpora=("dora",),
            ),
            corpus_scope="discover",
            target_corpora=(),
            synthesis_mode="routing",
            test_types=("routing_precision",),
        )

        config = EvalConfig(law="dora", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.references_structured = [
            {"corpus_id": "dora", "article": "5"},
        ]
        # Retrieval found DORA
        mock_result.retrieval_metrics = {
            "references_structured_all": [
                {"corpus_id": "dora", "article": "5"},
                {"corpus_id": "dora", "article": "6"},
            ],
        }
        # Answer uses a name that aliases DON'T match — shouldn't matter anymore
        mock_result.answer = "The Digital Operational Resilience Act covers this."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        assert result.scores["routing_precision"].passed is True
        assert result.scores["routing_precision"].score == 1.0

    def test_routing_precision_fails_when_retrieval_misses_expected_law(self):
        """routing_precision fails when retrieval did NOT find the expected corpus."""
        from src.eval.types import GoldenCase, ExpectedBehavior
        from src.eval.eval_core import EvalConfig, _evaluate_single_case

        case = GoldenCase(
            id="routing_miss_test",
            profile="LEGAL",
            prompt="Which law covers cybersecurity for electricity?",
            expected=ExpectedBehavior(
                required_corpora=("eucs_dr_2024_1366",),
            ),
            corpus_scope="discover",
            target_corpora=(),
            synthesis_mode="routing",
            test_types=("routing_precision",),
        )

        config = EvalConfig(law="dora", run_mode="retrieval_only")

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.references_structured = []
        # Retrieval found DORA instead of EUCS
        mock_result.retrieval_metrics = {
            "references_structured_all": [
                {"corpus_id": "dora", "article": "5"},
            ],
        }
        mock_result.answer = "DORA covers this."

        with patch("src.eval.eval_core.ask.ask", return_value=mock_result):
            result = _evaluate_single_case(case, config, mock_engine, scorers=[])

        assert result.scores["routing_precision"].passed is False
        assert result.scores["routing_precision"].score == 0.0

    def test_routing_precision_no_corpus_resolver_needed(self):
        """routing_precision should work without CorpusResolver dependency."""
        from src.eval.cross_law_scorers import RoutingPrecisionScorer

        scorer = RoutingPrecisionScorer()
        result = scorer.score(
            context_corpora={"dora", "nis2"},
            expected_laws={"dora"},
            synthesis_mode="routing",
        )
        assert result.passed is True
        assert result.score == 1.0


class TestAnchorFormat:
    """Verify anchors use article:X format from generation."""

    def test_process_raw_cases_strips_whitespace(self):
        """_process_raw_cases strips whitespace from anchors."""
        from src.eval.eval_case_generator import _process_raw_cases
        raw = [
            {
                "id": "test",
                "prompt": "Q",
                "expected_corpora": ["ai_act"],
                "expected_anchors": ["  article:6 ", "article:13", "recital:1"],
            }
        ]
        cases = _process_raw_cases(raw, synthesis_mode="comparison")
        assert cases[0].expected_anchors == ("article:6", "article:13", "recital:1")



# ---------------------------------------------------------------------------
# Test TriggerCrossLawEvalRequest max_retries (R6)
# ---------------------------------------------------------------------------

class TestTriggerRequestMaxRetries:
    """TriggerCrossLawEvalRequest defaults to 0 retries and validates range."""

    def test_default_max_retries_is_zero(self):
        """Default max_retries is 0 (fast iteration)."""
        from ui_react.backend.routes.eval_cross_law import TriggerCrossLawEvalRequest
        req = TriggerCrossLawEvalRequest()
        assert req.max_retries == 0

    def test_max_retries_accepts_valid_values(self):
        """max_retries accepts 0, 1, 3, 5."""
        from ui_react.backend.routes.eval_cross_law import TriggerCrossLawEvalRequest
        for val in [0, 1, 3, 5]:
            req = TriggerCrossLawEvalRequest(max_retries=val)
            assert req.max_retries == val

    def test_max_retries_rejects_negative(self):
        """Negative max_retries is rejected by validation."""
        from pydantic import ValidationError
        from ui_react.backend.routes.eval_cross_law import TriggerCrossLawEvalRequest
        with pytest.raises(ValidationError):
            TriggerCrossLawEvalRequest(max_retries=-1)

    def test_max_retries_rejects_too_high(self):
        """max_retries > 5 is rejected by validation."""
        from pydantic import ValidationError
        from ui_react.backend.routes.eval_cross_law import TriggerCrossLawEvalRequest
        with pytest.raises(ValidationError):
            TriggerCrossLawEvalRequest(max_retries=6)


    def test_generation_prompt_instructs_empty_anchors(self):
        """The LLM prompt instructs leaving expected_anchors empty."""
        from src.eval.eval_case_generator import _build_generation_prompt, GenerationRequest
        request = GenerationRequest(
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            max_cases=5,
        )
        prompt = _build_generation_prompt(request, corpus_metadata={})
        assert '"expected_anchors": []' in prompt
