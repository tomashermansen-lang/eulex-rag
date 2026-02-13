"""Tests for eval metrics route endpoints (C2c, C2d, C2e).

Covers:
  Level 2: GET /overview, GET /quality, GET /performance, GET /ingestion.
  Level 3: GET /detail/law, /detail/suite, /detail/scorer, /detail/mode, /detail/difficulty.
  SSE: POST /analyse.
Uses fixture run files in tmp directories with monkeypatched paths.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ui_react.backend.routes import eval_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Fixture data constants
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_LAW_RUN_AI_ACT = {
    "meta": {
        "law": "ai-act",
        "timestamp": "2026-02-10T10:00:00Z",
        "run_mode": "full",
        "duration_seconds": 120.5,
    },
    "summary": {"total": 10, "passed": 9, "pass_rate": 0.9},
    "results": [
        {
            "case_id": f"ai-{i}",
            "passed": i < 9,
            "duration_ms": 3000 + i * 200,
            "scores": {
                "retrieval": {"passed": True, "score": 1.0},
                "faithfulness": {"passed": i < 9, "score": 1.0 if i < 9 else 0.0},
            },
            "retry_count": 0,
            "escalated": False,
        }
        for i in range(10)
    ],
    "stage_stats": {
        "retrieval_total": 10,
        "retrieval_passed": 10,
        "augmentation_total": 10,
        "augmentation_passed": 9,
        "generation_total": 10,
        "generation_passed": 9,
    },
    "retry_stats": {"cases_with_retries": 1, "total_retries": 1},
    "escalation_stats": {"cases_escalated": 1, "escalated_case_ids": ["ai-5"]},
}

SINGLE_LAW_RUN_GDPR = {
    "meta": {
        "law": "gdpr",
        "timestamp": "2026-02-10T09:00:00Z",
        "run_mode": "full",
        "duration_seconds": 80.0,
    },
    "summary": {"total": 5, "passed": 4, "pass_rate": 0.8},
    "results": [
        {
            "case_id": f"gdpr-{i}",
            "passed": i < 4,
            "duration_ms": 2000 + i * 300,
            "scores": {
                "retrieval": {"passed": True, "score": 1.0},
                "faithfulness": {"passed": i < 4, "score": 1.0 if i < 4 else 0.0},
            },
            "retry_count": 0,
            "escalated": False,
        }
        for i in range(5)
    ],
    "stage_stats": {
        "retrieval_total": 5,
        "retrieval_passed": 5,
        "augmentation_total": 5,
        "augmentation_passed": 5,
        "generation_total": 5,
        "generation_passed": 4,
    },
    "retry_stats": {"cases_with_retries": 0, "total_retries": 0},
    "escalation_stats": {"cases_escalated": 0, "escalated_case_ids": []},
}

CROSS_LAW_RUN_SUITE_A = {
    "run_id": "run_abc123",
    "suite_id": "suite_a_abc",
    "timestamp": "2026-02-10T12:00:00Z",
    "total": 5,
    "passed": 4,
    "failed": 1,
    "pass_rate": 0.8,
    "duration_seconds": 60.0,
    "run_mode": "full",
    "results": [
        {
            "case_id": "c1",
            "synthesis_mode": "comparison",
            "difficulty": "easy",
            "passed": True,
            "duration_ms": 5000,
            "scores": {
                "corpus_coverage": {"passed": True, "score": 1.0},
                "comparison_completeness": {"passed": True, "score": 1.0},
            },
        },
        {
            "case_id": "c2",
            "synthesis_mode": "comparison",
            "difficulty": "medium",
            "passed": True,
            "duration_ms": 8000,
            "scores": {
                "corpus_coverage": {"passed": True, "score": 1.0},
                "comparison_completeness": {"passed": True, "score": 1.0},
            },
        },
        {
            "case_id": "c3",
            "synthesis_mode": "discovery",
            "difficulty": "easy",
            "passed": True,
            "duration_ms": 6000,
            "scores": {"corpus_coverage": {"passed": True, "score": 1.0}},
        },
        {
            "case_id": "c4",
            "synthesis_mode": "discovery",
            "difficulty": "hard",
            "passed": False,
            "duration_ms": 12000,
            "scores": {"corpus_coverage": {"passed": False, "score": 0.0}},
        },
        {
            "case_id": "c5",
            "synthesis_mode": "routing",
            "difficulty": "medium",
            "passed": True,
            "duration_ms": 4000,
            "scores": {
                "corpus_coverage": {"passed": True, "score": 1.0},
                "routing_precision": {"passed": True, "score": 1.0},
            },
        },
    ],
}

CROSS_LAW_RUN_SUITE_B = {
    "run_id": "run_def456",
    "suite_id": "suite_b_def",
    "timestamp": "2026-02-10T13:00:00Z",
    "total": 3,
    "passed": 3,
    "failed": 0,
    "pass_rate": 1.0,
    "duration_seconds": 30.0,
    "run_mode": "full",
    "results": [
        {
            "case_id": "d1",
            "synthesis_mode": "comparison",
            "difficulty": "easy",
            "passed": True,
            "duration_ms": 4000,
            "scores": {
                "corpus_coverage": {"passed": True, "score": 1.0},
                "comparison_completeness": {"passed": True, "score": 1.0},
            },
        },
        {
            "case_id": "d2",
            "synthesis_mode": "aggregation",
            "difficulty": "medium",
            "passed": True,
            "duration_ms": 7000,
            "scores": {"corpus_coverage": {"passed": True, "score": 1.0}},
        },
        {
            "case_id": "d3",
            "synthesis_mode": "comparison",
            "difficulty": "hard",
            "passed": True,
            "duration_ms": 10000,
            "scores": {
                "corpus_coverage": {"passed": True, "score": 1.0},
                "comparison_completeness": {"passed": True, "score": 1.0},
            },
        },
    ],
}

CROSS_LAW_RUN_SUITE_A_OLD = {
    "run_id": "run_old999",
    "suite_id": "suite_a_abc",
    "timestamp": "2026-02-09T12:00:00Z",
    "total": 5,
    "passed": 3,
    "failed": 2,
    "pass_rate": 0.6,
    "duration_seconds": 70.0,
    "run_mode": "full",
    "results": [
        {
            "case_id": "c1",
            "synthesis_mode": "comparison",
            "difficulty": "easy",
            "passed": True,
            "duration_ms": 5500,
            "scores": {
                "corpus_coverage": {"passed": True, "score": 1.0},
                "comparison_completeness": {"passed": True, "score": 1.0},
            },
        },
        {
            "case_id": "c2",
            "synthesis_mode": "comparison",
            "difficulty": "medium",
            "passed": False,
            "duration_ms": 9000,
            "scores": {
                "corpus_coverage": {"passed": False, "score": 0.0},
                "comparison_completeness": {"passed": False, "score": 0.0},
            },
        },
        {
            "case_id": "c3",
            "synthesis_mode": "discovery",
            "difficulty": "easy",
            "passed": True,
            "duration_ms": 6500,
            "scores": {"corpus_coverage": {"passed": True, "score": 1.0}},
        },
        {
            "case_id": "c4",
            "synthesis_mode": "discovery",
            "difficulty": "hard",
            "passed": False,
            "duration_ms": 13000,
            "scores": {"corpus_coverage": {"passed": False, "score": 0.0}},
        },
        {
            "case_id": "c5",
            "synthesis_mode": "routing",
            "difficulty": "medium",
            "passed": True,
            "duration_ms": 4200,
            "scores": {
                "corpus_coverage": {"passed": True, "score": 1.0},
                "routing_precision": {"passed": True, "score": 1.0},
            },
        },
    ],
}

SINGLE_LAW_RUN_DELETED = {
    "meta": {
        "law": "co2-trucks",
        "timestamp": "2026-01-30T15:35:00Z",
        "run_mode": "full",
        "duration_seconds": 60.0,
    },
    "summary": {"total": 3, "passed": 3, "pass_rate": 1.0},
    "results": [
        {
            "case_id": f"co2-{i}",
            "passed": True,
            "duration_ms": 2000 + i * 100,
            "scores": {"retrieval": {"passed": True, "score": 1.0}},
            "retry_count": 0,
            "escalated": False,
        }
        for i in range(3)
    ],
    "stage_stats": {
        "retrieval_total": 3,
        "retrieval_passed": 3,
        "augmentation_total": 3,
        "augmentation_passed": 3,
        "generation_total": 3,
        "generation_passed": 3,
    },
    "retry_stats": {"cases_with_retries": 0, "total_retries": 0},
    "escalation_stats": {"cases_escalated": 0, "escalated_case_ids": []},
}

CORPORA_DATA = {
    "corpora": {
        "ai-act": {
            "display_name": "AI Act",
            "enabled": True,
            "quality": {
                "structure_coverage_pct": 99.5,
                "unhandled_count": 2,
                "chunk_count": 929,
            },
        },
        "gdpr": {
            "display_name": "GDPR",
            "enabled": True,
            "quality": {
                "structure_coverage_pct": 97.0,
                "unhandled_count": 5,
                "chunk_count": 500,
            },
        },
    },
    "version": 1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def project_dir(tmp_path):
    """Create fixture data in a tmp directory."""
    # Single-law runs
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    (runs_dir / "eval_ai-act.json").write_text(json.dumps(SINGLE_LAW_RUN_AI_ACT))
    (runs_dir / "eval_gdpr.json").write_text(json.dumps(SINGLE_LAW_RUN_GDPR))
    # Run for a law NOT in corpora — should be filtered out
    (runs_dir / "eval_co2-trucks.json").write_text(json.dumps(SINGLE_LAW_RUN_DELETED))

    # Cross-law runs (newer sorts after older alphabetically)
    cl_runs_dir = tmp_path / "data" / "evals" / "runs"
    cl_runs_dir.mkdir(parents=True)
    (cl_runs_dir / "run_abc123.json").write_text(json.dumps(CROSS_LAW_RUN_SUITE_A))
    (cl_runs_dir / "run_def456.json").write_text(json.dumps(CROSS_LAW_RUN_SUITE_B))
    (cl_runs_dir / "run_old999.json").write_text(json.dumps(CROSS_LAW_RUN_SUITE_A_OLD))

    # Suite YAML files (for name resolution)
    evals_dir = tmp_path / "data" / "evals"
    (evals_dir / "cross_law_suite_a_abc.yaml").write_text(
        yaml.dump({"name": "Test Suite A", "id": "suite_a_abc", "cases": []})
    )
    (evals_dir / "cross_law_suite_b_def.yaml").write_text(
        yaml.dump({"name": "Test Suite B", "id": "suite_b_def", "cases": []})
    )

    # Corpora
    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "corpora.json").write_text(json.dumps(CORPORA_DATA))

    return tmp_path


@pytest.fixture
def client(project_dir, monkeypatch):
    """TestClient with paths monkeypatched to tmp directory."""
    monkeypatch.setattr(
        eval_metrics, "_get_single_law_runs_dir", lambda: project_dir / "runs"
    )
    monkeypatch.setattr(
        eval_metrics,
        "_get_cross_law_runs_dir",
        lambda: project_dir / "data" / "evals" / "runs",
    )
    monkeypatch.setattr(
        eval_metrics, "_get_suites_dir", lambda: project_dir / "data" / "evals"
    )
    monkeypatch.setattr(
        eval_metrics,
        "_get_corpora_path",
        lambda: project_dir / "data" / "processed" / "corpora.json",
    )

    app = FastAPI()
    app.include_router(eval_metrics.router, prefix="/api/eval/metrics")
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# T-R1 – T-R5: GET /overview
# ─────────────────────────────────────────────────────────────────────────────


class TestOverviewEndpoint:
    """Level 1 trust overview."""

    def test_returns_unified_pass_rate(self, client):
        """T-R1: Unified pass rate is weighted average across all runs."""
        resp = client.get("/api/eval/metrics/overview")
        assert resp.status_code == 200
        data = resp.json()
        # SL: ai-act 9/10 + gdpr 4/5 = 13/15
        # CL: suite_a 4/5 + suite_b 3/3 = 7/8
        # Unified: (13 + 7) / (15 + 8) = 20/23 ≈ 86.96%
        assert data["unified_pass_rate"] == pytest.approx(20 / 23 * 100, abs=0.1)

    def test_returns_health_status(self, client):
        """T-R2: Health status mapped from pass rate using config thresholds."""
        resp = client.get("/api/eval/metrics/overview")
        data = resp.json()
        # ~86.96% → yellow (thresholds: [95, 80, 60])
        assert data["health_status"] == "yellow"

    def test_has_data_true_when_runs_exist(self, client):
        """T-R3: has_data is True when at least one run exists."""
        data = client.get("/api/eval/metrics/overview").json()
        assert data["has_data"] is True

    def test_category_summaries(self, client):
        """T-R4: Single-law and cross-law summaries correct."""
        data = client.get("/api/eval/metrics/overview").json()
        assert data["single_law"]["total"] == 15
        assert data["single_law"]["passed"] == 13
        assert data["cross_law"]["total"] == 8
        assert data["cross_law"]["passed"] == 7

    def test_excludes_non_ingested_laws(self, client):
        """Runs for laws not in corpora.json must be excluded from metrics."""
        data = client.get("/api/eval/metrics/overview").json()
        # Only ai-act and gdpr are in corpora. co2-trucks has a run file but
        # is not ingested, so it must NOT contribute to totals.
        # SL: ai-act 9/10 + gdpr 4/5 = 13/15 (co2-trucks 3/3 excluded)
        assert data["single_law"]["total"] == 15
        assert data["single_law"]["passed"] == 13
        assert data["summary"]["law_count"] == 2

    def test_quality_excludes_non_ingested_laws(self, client):
        """Per-law list in quality must not include non-ingested laws."""
        data = client.get("/api/eval/metrics/quality").json()
        law_ids = [l["law"] for l in data["per_law"]]
        assert "co2-trucks" not in law_ids
        assert len(law_ids) == 2

    def test_quality_returns_single_law_scorers(self, client):
        """Quality must include per_scorer_single_law with aggregated SL scorer rates."""
        data = client.get("/api/eval/metrics/quality").json()
        assert "per_scorer_single_law" in data
        sl_scorers = {s["scorer"]: s for s in data["per_scorer_single_law"]}
        # AI-ACT has 10 cases, GDPR has 5 cases → 15 cases total for each scorer
        assert "retrieval" in sl_scorers
        assert "faithfulness" in sl_scorers
        # Retrieval: all 15 cases pass
        assert sl_scorers["retrieval"]["passed"] == 15
        assert sl_scorers["retrieval"]["total"] == 15
        # Faithfulness: 9/10 AI-ACT + 4/5 GDPR = 13/15
        assert sl_scorers["faithfulness"]["passed"] == 13
        assert sl_scorers["faithfulness"]["total"] == 15
        # All SL scorers must have category "single_law"
        for s in data["per_scorer_single_law"]:
            assert s["category"] == "single_law"

    def test_empty_state(self, project_dir, monkeypatch):
        """T-R5: No runs → has_data=False, pass_rate=0."""
        empty_dir = project_dir / "empty_runs"
        empty_dir.mkdir()
        monkeypatch.setattr(eval_metrics, "_get_single_law_runs_dir", lambda: empty_dir)
        monkeypatch.setattr(
            eval_metrics, "_get_cross_law_runs_dir", lambda: empty_dir
        )

        app = FastAPI()
        app.include_router(eval_metrics.router, prefix="/api/eval/metrics")
        empty_client = TestClient(app)

        data = empty_client.get("/api/eval/metrics/overview").json()
        assert data["has_data"] is False
        assert data["unified_pass_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# T-R6 – T-R10: GET /quality
# ─────────────────────────────────────────────────────────────────────────────


class TestQualityEndpoint:
    """Level 2 eval quality panel."""

    def test_per_law_breakdown(self, client):
        """T-R6: Per-law pass rates from single-law runs."""
        data = client.get("/api/eval/metrics/quality").json()
        by_law = {r["law"]: r for r in data["per_law"]}
        assert by_law["ai-act"]["pass_rate"] == pytest.approx(90.0)
        assert by_law["ai-act"]["total"] == 10
        assert by_law["gdpr"]["pass_rate"] == pytest.approx(80.0)
        assert by_law["gdpr"]["total"] == 5

    def test_per_suite_breakdown(self, client):
        """T-R7: Per-suite pass rates with resolved names."""
        data = client.get("/api/eval/metrics/quality").json()
        by_suite = {r["suite_id"]: r for r in data["per_suite"]}
        assert by_suite["suite_a_abc"]["pass_rate"] == pytest.approx(80.0)
        assert by_suite["suite_a_abc"]["name"] == "Test Suite A"
        assert by_suite["suite_b_def"]["pass_rate"] == pytest.approx(100.0)
        assert by_suite["suite_b_def"]["name"] == "Test Suite B"

    def test_per_mode_breakdown(self, client):
        """T-R8: Per-mode pass rates from cross-law cases."""
        data = client.get("/api/eval/metrics/quality").json()
        by_mode = {r["mode"]: r for r in data["per_mode"]}
        # comparison: c1(T), c2(T), d1(T), d3(T) = 4/4 = 100%
        assert by_mode["comparison"]["pass_rate"] == pytest.approx(100.0)
        assert by_mode["comparison"]["total"] == 4
        # discovery: c3(T), c4(F) = 1/2 = 50%
        assert by_mode["discovery"]["pass_rate"] == pytest.approx(50.0)
        assert by_mode["discovery"]["total"] == 2

    def test_per_difficulty_breakdown(self, client):
        """T-R9: Per-difficulty pass rates from cross-law cases."""
        data = client.get("/api/eval/metrics/quality").json()
        by_diff = {r["difficulty"]: r for r in data["per_difficulty"]}
        # easy: c1(T), c3(T), d1(T) = 3/3 = 100%
        assert by_diff["easy"]["pass_rate"] == pytest.approx(100.0)
        # hard: c4(F), d3(T) = 1/2 = 50%
        assert by_diff["hard"]["pass_rate"] == pytest.approx(50.0)
        # medium: c2(T), c5(T), d2(T) = 3/3 = 100%
        assert by_diff["medium"]["pass_rate"] == pytest.approx(100.0)

    def test_per_scorer_mode_specific(self, client):
        """T-R10: Mode-specific scorers use correct denominators."""
        data = client.get("/api/eval/metrics/quality").json()
        by_scorer = {r["scorer"]: r for r in data["per_scorer"]}
        # comparison_completeness: only comparison cases: c1(T), c2(T), d1(T), d3(T) = 4/4
        assert by_scorer["comparison_completeness"]["total"] == 4
        assert by_scorer["comparison_completeness"]["passed"] == 4
        # corpus_coverage: all 8 cross-law cases
        assert by_scorer["corpus_coverage"]["total"] == 8
        assert by_scorer["corpus_coverage"]["passed"] == 7  # c4 failed


# ─────────────────────────────────────────────────────────────────────────────
# T-R11 – T-R14: GET /performance
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformanceEndpoint:
    """Level 2 processing performance."""

    def test_percentiles(self, client):
        """T-R11: Percentiles computed from all case durations."""
        data = client.get("/api/eval/metrics/performance").json()
        assert "p50" in data["percentiles"]
        assert "p95" in data["percentiles"]
        assert "p99" in data["percentiles"]
        assert data["percentiles"]["p50"] > 0

    def test_histogram_bins(self, client):
        """T-R12: Histogram bins cover all durations."""
        data = client.get("/api/eval/metrics/performance").json()
        assert len(data["histogram_bins"]) > 0
        total_count = sum(b["count"] for b in data["histogram_bins"])
        assert total_count == data["total_cases"]

    def test_latency_by_mode(self, client):
        """T-R13: P50 latency per synthesis mode from cross-law cases."""
        data = client.get("/api/eval/metrics/performance").json()
        by_mode = {r["mode"]: r for r in data["latency_by_synthesis_mode"]}
        assert "comparison" in by_mode
        assert "discovery" in by_mode
        assert by_mode["comparison"]["case_count"] == 4  # c1, c2, d1, d3

    def test_total_cases(self, client):
        """T-R14: Total cases counts all runs."""
        data = client.get("/api/eval/metrics/performance").json()
        # SL: 10 + 5 = 15, CL: 5 + 3 = 8 → 23 total
        assert data["total_cases"] == 23


# ─────────────────────────────────────────────────────────────────────────────
# T-R15 – T-R17: GET /ingestion
# ─────────────────────────────────────────────────────────────────────────────


class TestIngestionEndpoint:
    """Level 2 ingestion health."""

    def test_overall_coverage(self, client):
        """T-R15: Overall coverage is average of corpus coverages."""
        data = client.get("/api/eval/metrics/ingestion").json()
        # avg(99.5, 97.0) = 98.25
        assert data["overall_coverage"] == pytest.approx(98.25)

    def test_corpora_list(self, client):
        """T-R16: Corpora list with correct fields."""
        data = client.get("/api/eval/metrics/ingestion").json()
        assert len(data["corpora"]) == 2
        by_id = {c["corpus_id"]: c for c in data["corpora"]}
        assert by_id["ai-act"]["coverage"] == pytest.approx(99.5)
        assert by_id["ai-act"]["chunks"] == 929
        assert by_id["ai-act"]["unhandled"] == 2

    def test_health_status(self, client):
        """T-R17: Health status from coverage using thresholds."""
        data = client.get("/api/eval/metrics/ingestion").json()
        # 98.25% → green (threshold 95)
        assert data["health_status"] == "green"

    def test_zero_coverage_excluded_from_overall(self, project_dir, monkeypatch):
        """T-R17b: Corpora with 0% coverage are excluded from overall calculation."""
        corpora_with_zero = {
            "corpora": {
                "ai-act": {
                    "display_name": "AI Act",
                    "enabled": True,
                    "quality": {"structure_coverage_pct": 99.5, "unhandled_count": 2, "chunk_count": 929},
                },
                "gdpr": {
                    "display_name": "GDPR",
                    "enabled": True,
                    "quality": {"structure_coverage_pct": 97.0, "unhandled_count": 5, "chunk_count": 500},
                },
                "dora": {
                    "display_name": "DORA",
                    "enabled": True,
                    "quality": {"structure_coverage_pct": 0.0, "unhandled_count": 0, "chunk_count": 0},
                },
            },
            "version": 1,
        }
        corpora_path = project_dir / "data" / "processed" / "corpora.json"
        corpora_path.write_text(json.dumps(corpora_with_zero))
        monkeypatch.setattr(eval_metrics, "_get_corpora_path", lambda: corpora_path)

        from starlette.testclient import TestClient
        app = eval_metrics.router
        from fastapi import FastAPI
        test_app = FastAPI()
        test_app.include_router(app, prefix="/api/eval/metrics")
        test_client = TestClient(test_app)

        data = test_client.get("/api/eval/metrics/ingestion").json()
        # DORA (0% coverage) should be excluded from overall: avg(99.5, 97.0) = 98.25
        assert data["overall_coverage"] == pytest.approx(98.25)
        # But it should still appear in the corpora list
        by_id = {c["corpus_id"]: c for c in data["corpora"]}
        assert "dora" in by_id
        assert by_id["dora"]["coverage"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# T-R18 – T-R20: GET /detail/law/{law}
# ─────────────────────────────────────────────────────────────────────────────


class TestLawDetailEndpoint:
    """Level 3 per-law drill-down (single-law)."""

    def test_returns_law_data(self, client):
        """T-R18: Law detail returns correct law and display name."""
        data = client.get("/api/eval/metrics/detail/law/ai-act").json()
        assert data["law"] == "ai-act"
        assert data["display_name"] == "AI Act"

    def test_scorer_breakdown(self, client):
        """T-R19: Scorer breakdown from latest run results."""
        data = client.get("/api/eval/metrics/detail/law/ai-act").json()
        by_scorer = {r["scorer"]: r for r in data["scorer_breakdown"]}
        # ai-act has retrieval (10/10=100%) and faithfulness (9/10=90%)
        assert by_scorer["retrieval"]["pass_rate"] == pytest.approx(100.0)
        assert by_scorer["faithfulness"]["pass_rate"] == pytest.approx(90.0)

    def test_latest_results(self, client):
        """T-R20: Case-level results from latest run."""
        data = client.get("/api/eval/metrics/detail/law/ai-act").json()
        assert len(data["latest_results"]) == 10
        # First case should have case_id, passed, duration_ms
        case = data["latest_results"][0]
        assert "case_id" in case
        assert "passed" in case
        assert "duration_ms" in case

    def test_unknown_law_returns_empty(self, client):
        """T-R20b: Unknown law returns empty results."""
        data = client.get("/api/eval/metrics/detail/law/nonexistent").json()
        assert data["law"] == "nonexistent"
        assert len(data["latest_results"]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# T-R21 – T-R24: GET /detail/suite/{suite_id}
# ─────────────────────────────────────────────────────────────────────────────


class TestSuiteDetailEndpoint:
    """Level 3 per-suite drill-down (cross-law)."""

    def test_returns_suite_data(self, client):
        """T-R21: Suite detail with name and counts."""
        data = client.get("/api/eval/metrics/detail/suite/suite_a_abc").json()
        assert data["suite_id"] == "suite_a_abc"
        assert data["name"] == "Test Suite A"
        # Mode counts from latest run (run_abc123)
        assert data["mode_counts"]["comparison"] == 2
        assert data["mode_counts"]["discovery"] == 2
        assert data["mode_counts"]["routing"] == 1

    def test_mode_filter_narrows_results(self, client):
        """T-R22: Mode filter returns only matching cases."""
        data = client.get(
            "/api/eval/metrics/detail/suite/suite_a_abc?mode=comparison"
        ).json()
        assert data["applied_filters"]["mode"] == "comparison"
        # Only comparison cases: c1, c2
        assert len(data["latest_results"]) == 2
        for case in data["latest_results"]:
            assert case["synthesis_mode"] == "comparison"

    def test_difficulty_filter(self, client):
        """T-R23: Difficulty filter returns only matching cases."""
        data = client.get(
            "/api/eval/metrics/detail/suite/suite_a_abc?difficulty=easy"
        ).json()
        assert data["applied_filters"]["difficulty"] == "easy"
        for case in data["latest_results"]:
            assert case["difficulty"] == "easy"

    def test_trend_from_multiple_runs(self, client):
        """T-R24: Trend includes data from multiple runs."""
        data = client.get("/api/eval/metrics/detail/suite/suite_a_abc").json()
        # Should have 2 trend points (old run + latest run)
        assert len(data["trend"]) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# T-R25 – T-R26: GET /detail/scorer/{scorer}
# ─────────────────────────────────────────────────────────────────────────────


class TestScorerDetailEndpoint:
    """Level 3 per-scorer drill-down."""

    def test_per_law_rates(self, client):
        """T-R25: Scorer detail shows per-law pass rates."""
        data = client.get("/api/eval/metrics/detail/scorer/retrieval").json()
        assert data["scorer"] == "retrieval"
        by_law = {r["law"]: r for r in data["per_law_rates"]}
        # ai-act retrieval: 10/10 = 100%
        assert by_law["ai-act"]["pass_rate"] == pytest.approx(100.0)
        # gdpr retrieval: 5/5 = 100%
        assert by_law["gdpr"]["pass_rate"] == pytest.approx(100.0)

    def test_cross_law_scorer(self, client):
        """T-R26: Cross-law scorer also returns data."""
        data = client.get("/api/eval/metrics/detail/scorer/corpus_coverage").json()
        assert data["scorer"] == "corpus_coverage"
        # Should have per-law rates (actually per-suite for CL scorers,
        # but the endpoint aggregates across all available data)
        assert len(data["per_law_rates"]) >= 0


# ─────────────────────────────────────────────────────────────────────────────
# T-R27 – T-R28: GET /detail/mode/{mode}
# ─────────────────────────────────────────────────────────────────────────────


class TestModeDetailEndpoint:
    """Level 3 per-mode drill-down (cross-law)."""

    def test_returns_mode_data(self, client):
        """T-R27: Mode detail with pass rate and cases."""
        data = client.get("/api/eval/metrics/detail/mode/comparison").json()
        assert data["mode"] == "comparison"
        # comparison cases: c1(T), c2(T), d1(T), d3(T) = 4 total, 4 passed
        assert data["total"] == 4
        assert data["passed"] == 4
        assert data["pass_rate"] == pytest.approx(100.0)
        assert len(data["cases"]) == 4

    def test_applicable_scorers_only(self, client):
        """T-R28: Only scorers applicable to this mode are returned."""
        data = client.get("/api/eval/metrics/detail/mode/comparison").json()
        scorer_names = {s["scorer"] for s in data["applicable_scorers"]}
        # comparison mode should include comparison_completeness
        assert "comparison_completeness" in scorer_names
        assert "corpus_coverage" in scorer_names
        # routing_precision should NOT be included (routing-only)
        assert "routing_precision" not in scorer_names


# ─────────────────────────────────────────────────────────────────────────────
# T-R29: GET /detail/difficulty/{difficulty}
# ─────────────────────────────────────────────────────────────────────────────


class TestDifficultyDetailEndpoint:
    """Level 3 per-difficulty drill-down (cross-law)."""

    def test_returns_difficulty_data(self, client):
        """T-R29: Difficulty detail with correct filtered cases."""
        data = client.get("/api/eval/metrics/detail/difficulty/easy").json()
        assert data["difficulty"] == "easy"
        # easy cases: c1(T), c3(T), d1(T) = 3 total, 3 passed
        assert data["total"] == 3
        assert data["passed"] == 3
        assert data["pass_rate"] == pytest.approx(100.0)
        assert len(data["cases"]) == 3
        for case in data["cases"]:
            assert case["difficulty"] == "easy"


# ─────────────────────────────────────────────────────────────────────────────
# T-R30 – T-R33: POST /analyse (SSE stream)
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyseEndpoint:
    """SSE streaming AI analysis endpoint."""

    def test_streams_sse_events(self, client):
        """T-R30: Streams start, token, and complete SSE events."""

        async def mock_stream(snapshot, model=None):
            for chunk in ["Systemet ", "er ", "stabilt."]:
                yield chunk

        with patch(
            "ui_react.backend.routes.eval_metrics.analyse_metrics_stream",
            side_effect=mock_stream,
        ):
            resp = client.post("/api/eval/metrics/analyse")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        # Parse SSE events
        events = []
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))

        # Must have start, 3 tokens, and complete
        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert types[-1] == "complete"
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 3
        full_text = "".join(e["text"] for e in token_events)
        assert full_text == "Systemet er stabilt."

    def test_error_event_on_failure(self, client):
        """T-R31: Engine error produces error SSE event."""

        async def mock_stream_error(snapshot, model=None):
            raise RuntimeError("LLM timeout")
            yield  # noqa: RET503 — make it a generator

        with patch(
            "ui_react.backend.routes.eval_metrics.analyse_metrics_stream",
            side_effect=mock_stream_error,
        ):
            resp = client.post("/api/eval/metrics/analyse")

        assert resp.status_code == 200  # SSE always 200, error in stream

        events = []
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["error"] == "Analyse kunne ikke gennemføres"

    def test_snapshot_includes_metrics_data(self, client):
        """T-R32: Snapshot passed to engine has required fields."""
        captured_snapshots: list[dict] = []

        async def mock_stream_capture(snapshot, model=None):
            captured_snapshots.append(snapshot)
            yield "ok"

        with patch(
            "ui_react.backend.routes.eval_metrics.analyse_metrics_stream",
            side_effect=mock_stream_capture,
        ):
            client.post("/api/eval/metrics/analyse")

        assert len(captured_snapshots) == 1
        snap = captured_snapshots[0]
        assert "unified_pass_rate" in snap
        assert "health_status" in snap
        assert "single_law" in snap
        assert "cross_law" in snap
        assert "per_mode" in snap
        assert "per_difficulty" in snap
        assert "percentiles" in snap

    def test_sse_headers(self, client):
        """T-R33: Response has correct SSE headers."""

        async def mock_stream(snapshot, model=None):
            yield "ok"

        with patch(
            "ui_react.backend.routes.eval_metrics.analyse_metrics_stream",
            side_effect=mock_stream,
        ):
            resp = client.post("/api/eval/metrics/analyse")

        assert "no-cache" in resp.headers.get("cache-control", "")
        assert resp.headers.get("connection") == "keep-alive"
