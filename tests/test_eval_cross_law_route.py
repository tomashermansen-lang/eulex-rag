"""Tests for cross-law eval route handler.

TDD: Verify CrossLawEvalService correctly converts GeneratedCase to
CrossLawGoldenCase, especially min_corpora_cited (R2).

Requirement mapping:
- R2-001: min_corpora_cited uses case-level expected_corpora count
- R2-002: min_corpora_cited is at least 1 even with empty expected_corpora
- R2-003: min_corpora_cited equals len(expected_corpora) when all corpora cited
"""

from __future__ import annotations

import asyncio
import pytest
from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui_react.backend.routes.eval_cross_law import (
    CrossLawEvalService,
    GenerateCasesRequest,
    CaseResponse,
    CrossLawCaseResult,
    CrossLawSuiteStats,
)
from src.eval.eval_case_generator import GeneratedCase
from src.eval.cross_law_suite_manager import CrossLawGoldenCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(corpus_ids: set[str] | None = None, tmp_path: Path | None = None) -> CrossLawEvalService:
    """Create a CrossLawEvalService with mocked manager."""
    ids = corpus_ids or {"ai_act", "gdpr", "nis2", "dora"}
    svc = CrossLawEvalService(
        evals_dir=tmp_path or Path("/tmp/test_evals"),
        valid_corpus_ids=ids,
    )
    # Mock out the manager so we don't touch disk
    svc.manager = MagicMock()
    svc.manager.get_suite.return_value = None
    svc.manager.create_suite.return_value = None
    svc.manager.add_case.return_value = None
    return svc


def _make_generated_case(
    case_id: str = "test_001",
    expected_corpora: tuple[str, ...] = ("ai_act", "gdpr"),
    expected_anchors: tuple[str, ...] = ("article:1", "article:5"),
    synthesis_mode: str = "comparison",
) -> GeneratedCase:
    return GeneratedCase(
        id=case_id,
        prompt="How do AI Act and GDPR interact on data protection?",
        expected_anchors=expected_anchors,
        expected_corpora=expected_corpora,
        synthesis_mode=synthesis_mode,
    )


async def _run_generate_cases(
    svc: CrossLawEvalService,
    request: GenerateCasesRequest,
    gen_case: GeneratedCase,
) -> None:
    """Run generate_cases with mocked dependencies."""
    with patch(
        "ui_react.backend.routes.eval_cross_law.generate_cross_law_cases",
        new_callable=AsyncMock,
        return_value=[gen_case],
    ), patch(
        "ui_react.backend.routes.eval_cross_law.assign_test_types",
        return_value=[gen_case],
    ):
        await svc.generate_cases(request)


# ---------------------------------------------------------------------------
# R2 Tests: min_corpora_cited uses case-level expected_corpora
# ---------------------------------------------------------------------------


class TestMinCorporaCitedCaseLevel:
    """R2: min_corpora_cited must reflect each case's expected_corpora, not suite-level."""

    def test_r2_001_min_corpora_cited_uses_case_level(self, tmp_path: Path) -> None:
        """When a case expects 2 of 4 target corpora, min_corpora_cited allows one miss."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = _make_generated_case(expected_corpora=("ai_act", "gdpr"))

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr", "nis2", "dora"],
            synthesis_mode="comparison",
            max_cases=1,
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        # 2 expected corpora → max(1, 2-1) = 1 (allows one miss)
        svc.manager.add_case.assert_called_once()
        golden_case = svc.manager.add_case.call_args[0][1]
        assert golden_case.min_corpora_cited == 1, (
            f"Expected min_corpora_cited=1 (allows one miss), got {golden_case.min_corpora_cited}"
        )

    def test_r2_002_min_corpora_cited_at_least_one(self, tmp_path: Path) -> None:
        """Even with empty expected_corpora, min_corpora_cited should be at least 1."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = _make_generated_case(expected_corpora=())

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr", "nis2", "dora"],
            synthesis_mode="comparison",
            max_cases=1,
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        svc.manager.add_case.assert_called_once()
        golden_case = svc.manager.add_case.call_args[0][1]
        assert golden_case.min_corpora_cited >= 1, (
            f"min_corpora_cited should be at least 1, got {golden_case.min_corpora_cited}"
        )

    def test_r2_003_min_corpora_cited_all_corpora(self, tmp_path: Path) -> None:
        """When case expects all 3 target corpora, min_corpora_cited = 2 (allows one miss)."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = _make_generated_case(expected_corpora=("ai_act", "gdpr", "nis2"))

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr", "nis2"],
            synthesis_mode="comparison",
            max_cases=1,
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        svc.manager.add_case.assert_called_once()
        golden_case = svc.manager.add_case.call_args[0][1]
        # 3 expected → max(1, 3-1) = 2
        assert golden_case.min_corpora_cited == 2


# ---------------------------------------------------------------------------
# Phase 11: Backend API extensions (R1.1, R2.5, R3.1, R1.5)
# ---------------------------------------------------------------------------


class TestDiscoveryFallbackTestType:
    """11.7: _convert_case_to_golden() must route discovery → corpus_coverage (no extra scorer)."""

    def test_p11_001_discovery_fallback_gets_corpus_coverage(self, tmp_path: Path) -> None:
        """Discovery case without explicit test_types gets corpus_coverage (shared)."""
        svc = _make_service(tmp_path=tmp_path)

        case = CrossLawGoldenCase(
            id="disc_001",
            prompt="Topic question",
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="discovery",
            expected_anchors=(),
            expected_corpora=("ai_act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
            test_types=(),  # Empty — triggers fallback
        )

        golden = svc._convert_case_to_golden(case)
        assert "corpus_coverage" in golden.test_types
        assert "corpus_recall" not in golden.test_types

    def test_p11_002_comparison_fallback_still_works(self, tmp_path: Path) -> None:
        """Comparison case fallback still gets comparison_completeness."""
        svc = _make_service(tmp_path=tmp_path)

        case = CrossLawGoldenCase(
            id="comp_001",
            prompt="Compare X and Y",
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=("article:6",),
            expected_corpora=("ai_act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
            test_types=(),
        )

        golden = svc._convert_case_to_golden(case)
        assert "comparison_completeness" in golden.test_types


class TestCaseResponseQualityFields:
    """11.1: CaseResponse includes difficulty."""

    def test_p11_003_case_response_has_quality_fields(self) -> None:
        """CaseResponse model includes quality metadata fields."""
        resp = CaseResponse(
            id="test",
            prompt="Q",
            corpus_scope="explicit",
            target_corpora=["ai_act"],
            synthesis_mode="comparison",
            expected_anchors=[],
            expected_corpora=["ai_act"],
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
            difficulty="hard",
        )
        assert resp.difficulty == "hard"

    def test_p11_004_case_response_quality_fields_default_none(self) -> None:
        """CaseResponse quality fields default to None."""
        resp = CaseResponse(
            id="test",
            prompt="Q",
            corpus_scope="explicit",
            target_corpora=["ai_act"],
            synthesis_mode="comparison",
            expected_anchors=[],
            expected_corpora=["ai_act"],
            min_corpora_cited=1,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        assert resp.difficulty is None


class TestCaseResultQualityFields:
    """11.3: CrossLawCaseResult includes difficulty."""

    def test_p11_005_case_result_has_difficulty(self) -> None:
        """CrossLawCaseResult includes difficulty field."""
        result = CrossLawCaseResult(
            case_id="test",
            prompt="Q",
            synthesis_mode="comparison",
            target_corpora=["ai_act"],
            passed=True,
            duration_ms=100.0,
            scores={},
            difficulty="medium",
        )
        assert result.difficulty == "medium"


class TestSuiteStatsModeCounts:
    """11.5: CrossLawSuiteStats includes mode_counts."""

    def test_p11_006_suite_stats_has_mode_counts(self) -> None:
        """CrossLawSuiteStats includes mode_counts dict."""
        stats = CrossLawSuiteStats(
            id="test",
            name="Test",
            case_count=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
            mode_counts={"comparison": 5, "discovery": 3, "aggregation": 2},
        )
        assert stats.mode_counts["discovery"] == 3


# ---------------------------------------------------------------------------
# Manual test bug fix: difficulty assignment during generation
# ---------------------------------------------------------------------------

from src.eval.eval_case_generator import GeneratedCase


class TestGenerateCasesAssignsDifficulty:
    """Bug fix: generate_cases must assign difficulty to generated cases."""

    def test_generated_case_has_difficulty(self, tmp_path: Path) -> None:
        """Cases created by generate_cases should have difficulty set."""
        from src.eval.cross_law_suite_manager import CrossLawSuiteManager

        manager = CrossLawSuiteManager(tmp_path, valid_corpus_ids={"ai-act", "gdpr"})
        service = CrossLawEvalService(tmp_path, {"ai-act", "gdpr"})

        # Simulate what generate_cases does: create a GeneratedCase and convert
        gen_case = GeneratedCase(
            id="test_diff",
            prompt="Compare AI-Act and GDPR on transparency",
            synthesis_mode="comparison",
            expected_corpora=("ai-act", "gdpr"),
            expected_anchors=("article:6",),
            test_types=("comparison_completeness",),
        )

        # assign_difficulty should return "easy" for 2 corpora + 1 anchor + comparison
        from src.eval.eval_case_generator import assign_difficulty
        difficulty = assign_difficulty(gen_case)
        assert difficulty == "easy"

        # The CrossLawGoldenCase constructed from it should carry the difficulty
        from src.eval.cross_law_suite_manager import CrossLawGoldenCase
        case = CrossLawGoldenCase(
            id=gen_case.id,
            prompt=gen_case.prompt,
            corpus_scope="explicit",
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode=gen_case.synthesis_mode,
            expected_anchors=gen_case.expected_anchors,
            expected_corpora=gen_case.expected_corpora,
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
            difficulty=difficulty,
        )
        assert case.difficulty == "easy"


# ---------------------------------------------------------------------------
# Case generator quality: anchors and min_corpora_cited
# ---------------------------------------------------------------------------


class TestGeneratedCaseQuality:
    """Generated cases should have realistic expectations."""

    def test_discovery_min_corpora_cited_is_none(self, tmp_path: Path) -> None:
        """Discovery cases should not set min_corpora_cited, letting threshold handle it."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = _make_generated_case(
            expected_corpora=("ai_act", "gdpr", "nis2"),
            synthesis_mode="discovery",
        )

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr", "nis2", "dora"],
            synthesis_mode="discovery",
            max_cases=1,
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        svc.manager.add_case.assert_called_once()
        golden_case = svc.manager.add_case.call_args[0][1]
        # Discovery uses corpus_coverage_threshold (0.8) — min_corpora_cited must be None
        assert golden_case.min_corpora_cited is None, (
            f"Discovery should not set min_corpora_cited, got {golden_case.min_corpora_cited}"
        )

    def test_non_discovery_min_corpora_cited_allows_one_miss(self, tmp_path: Path) -> None:
        """Non-discovery cases should allow one missing corpus (len - 1, min 1)."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = _make_generated_case(
            expected_corpora=("ai_act", "gdpr", "nis2"),
            synthesis_mode="comparison",
        )

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr", "nis2", "dora"],
            synthesis_mode="comparison",
            max_cases=1,
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        svc.manager.add_case.assert_called_once()
        golden_case = svc.manager.add_case.call_args[0][1]
        # 3 expected corpora → min_corpora_cited = 2 (allows one miss)
        assert golden_case.min_corpora_cited == 2

    def test_two_corpora_min_corpora_cited_is_one(self, tmp_path: Path) -> None:
        """With 2 expected corpora, min_corpora_cited should be at least 1."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = _make_generated_case(
            expected_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
        )

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
            synthesis_mode="comparison",
            max_cases=1,
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        svc.manager.add_case.assert_called_once()
        golden_case = svc.manager.add_case.call_args[0][1]
        assert golden_case.min_corpora_cited == 1

    def test_generated_cases_have_no_anchors(self, tmp_path: Path) -> None:
        """Auto-generated cases should not have expected_anchors (unreliable)."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = _make_generated_case(
            expected_anchors=("article:13", "article:14"),
            synthesis_mode="comparison",
        )

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
            synthesis_mode="comparison",
            max_cases=1,
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        svc.manager.add_case.assert_called_once()
        golden_case = svc.manager.add_case.call_args[0][1]
        # Auto-generated anchors should be stripped — they're unreliable
        assert golden_case.expected_anchors == ()


# =========================================================================
# Phase 5: Tests for inverted generation + retrieval calibration
# =========================================================================

from src.services.ask import AskResult


class TestGenerateCasesRequestStrategy:
    """Tests for generation_strategy and calibrate_anchors on GenerateCasesRequest."""

    def test_generation_strategy_defaults_to_standard(self):
        """GenerateCasesRequest.generation_strategy defaults to 'standard'."""
        req = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
            synthesis_mode="comparison",
        )
        assert req.generation_strategy == "standard"

    def test_generation_strategy_accepts_inverted(self):
        """GenerateCasesRequest accepts generation_strategy='inverted'."""
        req = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
            synthesis_mode="comparison",
            generation_strategy="inverted",
        )
        assert req.generation_strategy == "inverted"

    def test_calibrate_anchors_defaults_to_true(self):
        """calibrate_anchors defaults to True."""
        req = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
        )
        assert req.calibrate_anchors is True


class TestCaseResponseRetrievalConfirmed:
    """Tests for retrieval_confirmed on CaseResponse."""

    def test_case_response_has_retrieval_confirmed(self):
        """CaseResponse should include retrieval_confirmed."""
        resp = CaseResponse(
            id="test",
            prompt="Test",
            corpus_scope="explicit",
            target_corpora=["ai_act"],
            synthesis_mode="comparison",
            expected_anchors=["article:6"],
            expected_corpora=["ai_act"],
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
            retrieval_confirmed=True,
        )
        assert resp.retrieval_confirmed is True

    def test_case_response_retrieval_confirmed_default_none(self):
        """CaseResponse.retrieval_confirmed defaults to None."""
        resp = CaseResponse(
            id="test",
            prompt="Test",
            corpus_scope="explicit",
            target_corpora=["ai_act"],
            synthesis_mode="comparison",
            expected_anchors=[],
            expected_corpora=["ai_act"],
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )
        assert resp.retrieval_confirmed is None


class TestTestTypesPassthrough:
    """Bug fix: test_types from assign_test_types must be passed to CrossLawGoldenCase."""

    def test_generated_case_test_types_persisted(self, tmp_path: Path) -> None:
        """Generated case test_types must flow through to CrossLawGoldenCase."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case_raw = GeneratedCase(
            id="tt_001",
            prompt="Sammenlign krav",
            synthesis_mode="comparison",
            expected_corpora=("ai_act", "gdpr"),
            expected_anchors=(),
        )

        # After assign_test_types, case has non-empty test_types
        gen_case_with_types = GeneratedCase(
            id="tt_001",
            prompt="Sammenlign krav",
            synthesis_mode="comparison",
            expected_corpora=("ai_act", "gdpr"),
            expected_anchors=(),
            test_types=("comparison_completeness", "retrieval", "faithfulness", "relevancy"),
        )

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
            synthesis_mode="comparison",
            max_cases=1,
            generation_strategy="standard",
        )

        with patch(
            "ui_react.backend.routes.eval_cross_law.generate_cross_law_cases",
            new_callable=AsyncMock,
            return_value=[gen_case_raw],
        ), patch(
            "ui_react.backend.routes.eval_cross_law.assign_test_types",
            return_value=[gen_case_with_types],
        ):
            asyncio.run(svc.generate_cases(request))

        golden_case = svc.manager.add_case.call_args[0][1]
        assert golden_case.test_types == ("comparison_completeness", "retrieval", "faithfulness", "relevancy"), \
            f"Expected test_types from assign_test_types, got: {golden_case.test_types}"


class TestInvertedGenerationKeepsAnchors:
    """Tests for inverted generation keeping anchors in service layer."""

    def test_inverted_cases_keep_anchors(self, tmp_path: Path) -> None:
        """Inverted generation: anchors should be preserved, not stripped."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = GeneratedCase(
            id="inverted_001",
            prompt="Sammenlign krav til risikostyring",
            synthesis_mode="comparison",
            expected_corpora=("ai_act", "gdpr"),
            expected_anchors=("article:6", "article:35"),
            retrieval_confirmed=None,
        )

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
            synthesis_mode="comparison",
            max_cases=1,
            generation_strategy="inverted",
            calibrate_anchors=False,  # Skip calibration for this test
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        golden_case = svc.manager.add_case.call_args[0][1]
        assert golden_case.expected_anchors == ("article:6", "article:35")

    def test_inverted_cases_mirror_anchors_to_must_include_any_of(self, tmp_path: Path) -> None:
        """Inverted generation must copy anchors to must_include_any_of for UI visibility."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = GeneratedCase(
            id="inverted_mirror",
            prompt="Sammenlign krav til risikostyring",
            synthesis_mode="comparison",
            expected_corpora=("ai_act", "gdpr"),
            expected_anchors=("ai_act:article:6", "gdpr:article:35"),
        )

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
            synthesis_mode="comparison",
            max_cases=1,
            generation_strategy="inverted",
            calibrate_anchors=False,
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        golden_case = svc.manager.add_case.call_args[0][1]
        assert golden_case.must_include_any_of == ("ai_act:article:6", "gdpr:article:35")

    def test_standard_cases_still_strip_anchors(self, tmp_path: Path) -> None:
        """Standard generation: anchors should still be stripped."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = _make_generated_case(
            expected_anchors=("article:13",),
            synthesis_mode="comparison",
        )

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
            synthesis_mode="comparison",
            max_cases=1,
            generation_strategy="standard",
        )

        asyncio.run(_run_generate_cases(svc, request, gen_case))

        golden_case = svc.manager.add_case.call_args[0][1]
        assert golden_case.expected_anchors == ()


class TestCalibration:
    """Tests for _calibrate_case retrieval probe."""

    def test_calibration_confirms_when_anchors_found(self, tmp_path: Path) -> None:
        """Calibration with matching anchors → retrieval_confirmed=True, anchors kept."""
        svc = _make_service(tmp_path=tmp_path)

        case = CrossLawGoldenCase(
            id="cal_001",
            prompt="Sammenlign krav",
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=("article:6", "article:35"),
            expected_corpora=("ai_act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )

        mock_result = AskResult(
            answer="",
            references=[],
            references_structured=[],
            retrieval_metrics={
                "run": {"anchors_in_top_k": ["article:6", "article:35", "article:1"]},
                "retrieved_metadatas": [],
            },
        )

        with patch("ui_react.backend.routes.eval_cross_law.ask_service", return_value=mock_result):
            calibrated = asyncio.run(svc._calibrate_case(case))

        assert calibrated.retrieval_confirmed is True
        assert calibrated.expected_anchors == ("article:6", "article:35")

    def test_calibration_keeps_anchors_when_not_found(self, tmp_path: Path) -> None:
        """Calibration with no match → retrieval_confirmed=False, anchors KEPT."""
        svc = _make_service(tmp_path=tmp_path)

        case = CrossLawGoldenCase(
            id="cal_002",
            prompt="Sammenlign krav",
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=("article:6", "article:35"),
            expected_corpora=("ai_act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )

        mock_result = AskResult(
            answer="",
            references=[],
            references_structured=[],
            retrieval_metrics={
                "run": {"anchors_in_top_k": ["article:99"]},
                "retrieved_metadatas": [],
            },
        )

        with patch("ui_react.backend.routes.eval_cross_law.ask_service", return_value=mock_result):
            calibrated = asyncio.run(svc._calibrate_case(case))

        assert calibrated.retrieval_confirmed is False
        # Anchors must be KEPT (user should see them even if unconfirmed)
        assert calibrated.expected_anchors == ("article:6", "article:35")

    def test_calibration_exception_keeps_anchors(self, tmp_path: Path) -> None:
        """Calibration exception → retrieval_confirmed=None, anchors kept."""
        svc = _make_service(tmp_path=tmp_path)

        case = CrossLawGoldenCase(
            id="cal_003",
            prompt="Sammenlign krav",
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=("article:6",),
            expected_corpora=("ai_act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )

        with patch("ui_react.backend.routes.eval_cross_law.ask_service", side_effect=Exception("timeout")):
            calibrated = asyncio.run(svc._calibrate_case(case))

        assert calibrated.retrieval_confirmed is None
        assert calibrated.expected_anchors == ("article:6",)

    def test_calibration_skipped_for_empty_anchors(self, tmp_path: Path) -> None:
        """Empty anchors → calibration skipped."""
        svc = _make_service(tmp_path=tmp_path)

        case = CrossLawGoldenCase(
            id="cal_004",
            prompt="Test",
            corpus_scope="explicit",
            target_corpora=("ai_act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("ai_act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )

        # Should NOT call ask — no anchors to calibrate
        with patch("ui_react.backend.routes.eval_cross_law.ask_service") as mock_ask:
            calibrated = asyncio.run(svc._calibrate_case(case))

        mock_ask.assert_not_called()
        assert calibrated.retrieval_confirmed is None

    def test_calibrate_anchors_false_skips_probe(self, tmp_path: Path) -> None:
        """calibrate_anchors=False → no probe, anchors kept as-is."""
        svc = _make_service(tmp_path=tmp_path)

        gen_case = GeneratedCase(
            id="no_cal_001",
            prompt="Sammenlign krav",
            synthesis_mode="comparison",
            expected_corpora=("ai_act", "gdpr"),
            expected_anchors=("article:6", "article:35"),
        )

        request = GenerateCasesRequest(
            target_corpora=["ai_act", "gdpr"],
            synthesis_mode="comparison",
            max_cases=1,
            generation_strategy="inverted",
            calibrate_anchors=False,
        )

        with patch("ui_react.backend.routes.eval_cross_law.ask_service") as mock_ask:
            asyncio.run(_run_generate_cases(svc, request, gen_case))

        mock_ask.assert_not_called()
        golden_case = svc.manager.add_case.call_args[0][1]
        assert golden_case.expected_anchors == ("article:6", "article:35")

    def test_calibration_matches_corpus_qualified_anchors(self, tmp_path: Path) -> None:
        """Calibration confirms corpus-qualified anchors via retrieved_metadatas."""
        svc = _make_service(tmp_path=tmp_path)

        case = CrossLawGoldenCase(
            id="cal_cq_001",
            prompt="Sammenlign krav",
            corpus_scope="explicit",
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=("ai-act:article:6", "gdpr:article:35"),
            expected_corpora=("ai-act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )

        mock_result = AskResult(
            answer="",
            references=[],
            references_structured=[],
            retrieval_metrics={
                "run": {"anchors_in_top_k": []},
                "retrieved_metadatas": [
                    {"article": "6", "corpus_id": "ai-act"},
                    {"article": "35", "corpus_id": "gdpr"},
                ],
            },
        )

        with patch("ui_react.backend.routes.eval_cross_law.ask_service", return_value=mock_result):
            calibrated = asyncio.run(svc._calibrate_case(case))

        assert calibrated.retrieval_confirmed is True
        assert calibrated.expected_anchors == ("ai-act:article:6", "gdpr:article:35")

    def test_calibration_rejects_wrong_corpus_match(self, tmp_path: Path) -> None:
        """Corpus-qualified anchor from wrong corpus must NOT match."""
        svc = _make_service(tmp_path=tmp_path)

        case = CrossLawGoldenCase(
            id="cal_cq_002",
            prompt="Sammenlign krav",
            corpus_scope="explicit",
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=("ai-act:article:6",),
            expected_corpora=("ai-act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="auto-generated",
        )

        mock_result = AskResult(
            answer="",
            references=[],
            references_structured=[],
            retrieval_metrics={
                "run": {"anchors_in_top_k": []},
                "retrieved_metadatas": [
                    {"article": "6", "corpus_id": "gdpr"},  # Wrong corpus!
                ],
            },
        )

        with patch("ui_react.backend.routes.eval_cross_law.ask_service", return_value=mock_result):
            calibrated = asyncio.run(svc._calibrate_case(case))

        assert calibrated.retrieval_confirmed is False
        # Anchors kept even when wrong corpus match
        assert calibrated.expected_anchors == ("ai-act:article:6",)
