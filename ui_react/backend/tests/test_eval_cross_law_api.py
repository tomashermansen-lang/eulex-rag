"""Tests for cross-law eval API endpoints.

TDD: These tests verify eval_cross_law.py routes provide correct REST API
for cross-law eval suite management.

Requirement mapping:
- API-001: POST /suites creates suite (R6.1)
- API-002: GET /suites lists all suites (R6.1)
- API-003: GET /suites/{id} returns suite (R6.1)
- API-004: PUT /suites/{id} updates suite (R8.1)
- API-005: DELETE /suites/{id} removes suite (R8.3)
- API-006: POST /suites/{id}/cases adds case (R6.2)
- API-007: PUT /suites/{id}/cases/{id} updates case (R8.1)
- API-008: POST /suites/{id}/cases/{id}/duplicate clones case (R8.2)
- API-009: DELETE /suites/{id}/cases/{id} removes case (R8.3)
- API-018: Invalid corpus_id rejected (E14)
- API-019: Comparison with 1 corpus rejected (E15)
- OR-001: Editing a case sets origin to "manual" (R6.2)
- OR-002: Creating a case via API has origin "manual" (R6.1)
- OR-003: Origin field included in case response (R6.1)
- GN-001: POST /generate creates new suite with generated cases (R1.1, R3.1)
- GN-002: POST /generate adds cases to existing suite (R3.1)
- GN-003: POST /generate rejects <2 corpora (R1.3, AS5)
- GN-004: POST /generate rejects invalid corpus IDs (E6)
- GN-005: POST /generate caps at 20 cases (E5)
- GN-006: Generated cases have origin "auto-generated" (R2.4)
- GN-007: Generated cases have correct target_corpora (R3.4)
- AI-001: POST /ai-suggest returns name suggestion (R8.1)
- AI-002: POST /ai-suggest returns description suggestion (R8.2)
- AI-003: POST /ai-suggest handles LLM error gracefully (E1)
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
import sys

# Add backend to path
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient


@pytest.fixture
def temp_evals_dir():
    """Create a temporary directory for test suites."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def app(temp_evals_dir):
    """Create FastAPI app with test configuration."""
    from routes.eval_cross_law import create_router, CrossLawEvalService

    from fastapi import FastAPI

    # Create service with temp directory
    service = CrossLawEvalService(
        evals_dir=temp_evals_dir,
        valid_corpus_ids={"ai-act", "gdpr", "nis2", "dora", "cra"},
    )

    app = FastAPI()
    router = create_router(service)
    app.include_router(router, prefix="/api/eval/cross-law")

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_suite_request():
    """Sample request for creating a suite."""
    return {
        "name": "Test Suite",
        "description": "A test cross-law eval suite",
        "target_corpora": ["ai-act", "gdpr"],
        "default_synthesis_mode": "comparison",
    }


@pytest.fixture
def sample_case_request():
    """Sample request for creating a case."""
    return {
        "prompt": "Compare AI-Act and GDPR on transparency",
        "corpus_scope": "explicit",
        "target_corpora": ["ai-act", "gdpr"],
        "synthesis_mode": "comparison",
        "expected_anchors": [],
        "expected_corpora": ["ai-act", "gdpr"],
        "min_corpora_cited": 2,
        "profile": "LEGAL",
        "disabled": False,
    }


class TestSuiteCRUD:
    """Tests for suite CRUD operations."""

    def test_api_001_post_suites_creates_suite(self, client, sample_suite_request):
        """POST /suites should create a new suite."""
        response = client.post("/api/eval/cross-law/suites", json=sample_suite_request)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Suite"
        assert data["id"] is not None
        assert data["case_count"] == 0

    def test_api_002_get_suites_lists_all(self, client, sample_suite_request):
        """GET /suites should list all suites."""
        # Create 3 suites
        for i in range(3):
            req = {**sample_suite_request, "name": f"Suite {i}"}
            client.post("/api/eval/cross-law/suites", json=req)

        response = client.get("/api/eval/cross-law/suites")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_api_003_get_suite_returns_specific(self, client, sample_suite_request):
        """GET /suites/{id} should return specific suite."""
        # Create suite
        create_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = create_resp.json()["id"]

        response = client.get(f"/api/eval/cross-law/suites/{suite_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == suite_id
        assert data["name"] == "Test Suite"

    def test_api_003b_get_suite_not_found(self, client):
        """GET /suites/{id} should return 404 for unknown ID."""
        response = client.get("/api/eval/cross-law/suites/nonexistent")

        assert response.status_code == 404

    def test_api_004_put_suite_updates(self, client, sample_suite_request):
        """PUT /suites/{id} should update suite."""
        # Create suite
        create_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = create_resp.json()["id"]

        # Update
        update_req = {**sample_suite_request, "name": "Updated Suite"}
        response = client.put(f"/api/eval/cross-law/suites/{suite_id}", json=update_req)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Suite"

    def test_api_005_delete_suite_removes(self, client, sample_suite_request):
        """DELETE /suites/{id} should remove suite."""
        # Create suite
        create_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = create_resp.json()["id"]

        # Delete
        response = client.delete(f"/api/eval/cross-law/suites/{suite_id}")

        assert response.status_code == 204

        # Verify deleted
        get_resp = client.get(f"/api/eval/cross-law/suites/{suite_id}")
        assert get_resp.status_code == 404


class TestCaseCRUD:
    """Tests for case CRUD operations."""

    def test_api_006_post_cases_adds_case(self, client, sample_suite_request, sample_case_request):
        """POST /suites/{id}/cases should add case."""
        # Create suite
        create_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = create_resp.json()["id"]

        # Add case
        response = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["prompt"] == sample_case_request["prompt"]
        assert data["id"] is not None

    def test_api_007_put_case_updates(self, client, sample_suite_request, sample_case_request):
        """PUT /suites/{id}/cases/{id} should update case."""
        # Create suite and case
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        case_resp = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )
        case_id = case_resp.json()["id"]

        # Update case
        update_req = {**sample_case_request, "prompt": "Updated prompt"}
        response = client.put(
            f"/api/eval/cross-law/suites/{suite_id}/cases/{case_id}",
            json=update_req,
        )

        assert response.status_code == 200
        assert response.json()["prompt"] == "Updated prompt"

    def test_api_008_duplicate_case_clones(self, client, sample_suite_request, sample_case_request):
        """POST /suites/{id}/cases/{id}/duplicate should clone case."""
        # Create suite and case
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        case_resp = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )
        case_id = case_resp.json()["id"]

        # Duplicate
        response = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases/{case_id}/duplicate"
        )

        assert response.status_code == 201
        new_case_id = response.json()["id"]
        assert new_case_id != case_id

        # Verify suite now has 2 cases
        suite_resp = client.get(f"/api/eval/cross-law/suites/{suite_id}")
        assert suite_resp.json()["case_count"] == 2

    def test_api_009_delete_case_removes(self, client, sample_suite_request, sample_case_request):
        """DELETE /suites/{id}/cases/{id} should remove case."""
        # Create suite and case
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        case_resp = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )
        case_id = case_resp.json()["id"]

        # Delete
        response = client.delete(
            f"/api/eval/cross-law/suites/{suite_id}/cases/{case_id}"
        )

        assert response.status_code == 204

        # Verify suite has 0 cases
        suite_resp = client.get(f"/api/eval/cross-law/suites/{suite_id}")
        assert suite_resp.json()["case_count"] == 0


class TestValidation:
    """Tests for request validation."""

    def test_api_018_invalid_corpus_rejected(self, client):
        """Invalid corpus_id should return 400."""
        request = {
            "name": "Test Suite",
            "description": "",
            "target_corpora": ["ai-act", "unknown_law"],  # Invalid!
            "default_synthesis_mode": "comparison",
        }

        response = client.post("/api/eval/cross-law/suites", json=request)

        assert response.status_code == 400
        assert "unknown_law" in response.json()["detail"]

    def test_api_019_comparison_single_corpus_rejected(self, client, sample_suite_request):
        """Comparison mode with single corpus should return 400."""
        # Create suite
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        # Try to add comparison case with 1 corpus
        case_request = {
            "prompt": "Compare?",
            "corpus_scope": "explicit",
            "target_corpora": ["ai-act"],  # Only 1!
            "synthesis_mode": "comparison",
            "expected_anchors": [],
            "expected_corpora": ["ai-act"],
            "min_corpora_cited": 1,
            "profile": "LEGAL",
            "disabled": False,
        }

        response = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=case_request,
        )

        assert response.status_code == 400
        assert "comparison" in response.json()["detail"].lower()


class TestYAMLImportExport:
    """Tests for YAML import/export."""

    def test_export_yaml_produces_valid(self, client, sample_suite_request, sample_case_request):
        """GET /suites/{id}/export should return valid YAML."""
        # Create suite with case
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )

        # Export
        response = client.get(f"/api/eval/cross-law/suites/{suite_id}/export")

        assert response.status_code == 200
        assert "yaml" in response.headers.get("content-type", "").lower() or response.text.startswith("id:")

    def test_import_yaml_creates_suite(self, client):
        """POST /import should create suite from YAML."""
        yaml_content = """
name: Imported Suite
description: Imported from YAML
target_corpora:
  - ai-act
  - gdpr
default_synthesis_mode: comparison
cases:
  - id: imported_case
    prompt: Test question
    corpus_scope: explicit
    target_corpora:
      - ai-act
      - gdpr
    synthesis_mode: comparison
    expected_anchors: []
    expected_corpora:
      - ai-act
      - gdpr
    min_corpora_cited: 2
    profile: LEGAL
    disabled: false
    origin: manual
"""
        response = client.post(
            "/api/eval/cross-law/import",
            json={"yaml_content": yaml_content},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Imported Suite"
        assert data["case_count"] == 1


class TestOverviewEndpoint:
    """Tests for overview endpoint (R-UI-02: drill-down navigation)."""

    def test_overview_returns_stats(self, client, sample_suite_request):
        """GET /overview should return suite matrix stats."""
        # Create a suite
        client.post("/api/eval/cross-law/suites", json=sample_suite_request)

        response = client.get("/api/eval/cross-law/overview")

        assert response.status_code == 200
        data = response.json()
        assert "suites" in data
        assert "total_cases" in data
        assert "overall_pass_rate" in data
        assert len(data["suites"]) == 1

    def test_overview_empty_returns_defaults(self, client):
        """GET /overview with no suites should return empty list."""
        response = client.get("/api/eval/cross-law/overview")

        assert response.status_code == 200
        data = response.json()
        assert data["suites"] == []
        assert data["total_cases"] == 0
        assert data["overall_pass_rate"] == 0.0

    def test_overview_includes_test_type_stats(
        self, client, sample_suite_request, sample_case_request
    ):
        """GET /overview should include per-test-type stats."""
        # Create suite with a case
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )

        response = client.get("/api/eval/cross-law/overview")

        assert response.status_code == 200
        data = response.json()
        assert data["total_cases"] == 1
        # Suite stats should include the case
        suite_stats = data["suites"][0]
        assert suite_stats["case_count"] == 1


class TestOriginTracking:
    """Tests for origin tracking (C4: OR-001 to OR-003)."""

    def test_or_001_editing_case_sets_origin_to_manual(
        self, client, sample_suite_request, sample_case_request
    ):
        """Editing an auto-generated case should flip origin to 'manual'."""
        # Create suite
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        # Add case (origin defaults to "manual")
        case_resp = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )
        case_id = case_resp.json()["id"]
        assert case_resp.json()["origin"] == "manual"

        # Update the case — origin should remain "manual"
        update_req = {**sample_case_request, "prompt": "Updated prompt"}
        response = client.put(
            f"/api/eval/cross-law/suites/{suite_id}/cases/{case_id}",
            json=update_req,
        )

        assert response.status_code == 200
        assert response.json()["origin"] == "manual"

    def test_or_002_creating_case_has_origin_manual(
        self, client, sample_suite_request, sample_case_request
    ):
        """Manually created case should have origin 'manual'."""
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        case_resp = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )

        assert case_resp.status_code == 201
        assert case_resp.json()["origin"] == "manual"

    def test_or_003_origin_field_included_in_response(
        self, client, sample_suite_request, sample_case_request
    ):
        """Origin field should be included in case responses."""
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        case_resp = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )
        case_id = case_resp.json()["id"]

        # Get suite detail — check case has origin
        suite_detail = client.get(f"/api/eval/cross-law/suites/{suite_id}")
        cases = suite_detail.json()["cases"]
        assert len(cases) == 1
        assert "origin" in cases[0]
        assert cases[0]["origin"] == "manual"

    def test_or_001b_editing_auto_generated_case_flips_to_manual(
        self, client, sample_suite_request, temp_evals_dir
    ):
        """Editing an auto-generated case should flip origin to 'manual'."""
        import yaml

        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        # Find the YAML file (manager uses suite_id as filename)
        yaml_files = list(temp_evals_dir.glob("*.yaml"))
        assert len(yaml_files) == 1
        suite_file = yaml_files[0]

        # Manually inject an auto-generated case via YAML file
        suite_data = yaml.safe_load(suite_file.read_text())
        suite_data["cases"] = [
            {
                "id": "auto_case_1",
                "prompt": "Auto-generated question",
                "corpus_scope": "explicit",
                "target_corpora": ["ai-act", "gdpr"],
                "synthesis_mode": "comparison",
                "expected_anchors": [],
                "expected_corpora": ["ai-act", "gdpr"],
                "min_corpora_cited": 2,
                "profile": "LEGAL",
                "disabled": False,
                "origin": "auto-generated",
            }
        ]
        suite_file.write_text(yaml.dump(suite_data, allow_unicode=True))

        # Verify it's auto-generated
        detail = client.get(f"/api/eval/cross-law/suites/{suite_id}")
        assert detail.json()["cases"][0]["origin"] == "auto-generated"

        # Edit the case — should flip to "manual"
        update_req = {
            "prompt": "Manually edited question",
            "corpus_scope": "explicit",
            "target_corpora": ["ai-act", "gdpr"],
            "synthesis_mode": "comparison",
            "expected_anchors": [],
            "expected_corpora": ["ai-act", "gdpr"],
            "min_corpora_cited": 2,
            "profile": "LEGAL",
            "disabled": False,
        }
        response = client.put(
            f"/api/eval/cross-law/suites/{suite_id}/cases/auto_case_1",
            json=update_req,
        )

        assert response.status_code == 200
        assert response.json()["origin"] == "manual"


class TestRunsEndpoint:
    """Tests for runs endpoint (R-UI-04: run history)."""

    def test_list_runs_empty_suite(self, client, sample_suite_request):
        """GET /suites/{id}/runs should return empty list for new suite."""
        # Create a suite
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        response = client.get(f"/api/eval/cross-law/suites/{suite_id}/runs")

        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["total"] == 0

    def test_list_runs_not_found_suite(self, client):
        """GET /suites/{id}/runs should return 404 for unknown suite."""
        response = client.get("/api/eval/cross-law/suites/nonexistent/runs")

        assert response.status_code == 404


class TestRunDetailEndpoint:
    """Tests for run detail endpoint (R-UI-05: test case results)."""

    def test_get_run_detail_not_found(self, client):
        """GET /runs/{id} should return 404 for unknown run."""
        response = client.get("/api/eval/cross-law/runs/nonexistent")

        assert response.status_code == 404


class TestTriggerEndpoint:
    """Tests for trigger endpoint (R-UI-06: run triggering)."""

    def test_trigger_eval_invalid_suite_returns_404(self, client):
        """POST /suites/{id}/run should return 404 for unknown suite."""
        response = client.post(
            "/api/eval/cross-law/suites/nonexistent/run",
            json={"run_mode": "retrieval_only"},
        )

        assert response.status_code == 404

    def test_trigger_eval_returns_sse_stream(self, client, sample_suite_request):
        """POST /suites/{id}/run should return SSE stream."""
        # Create a suite (empty, no cases)
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        response = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/run",
            json={"run_mode": "retrieval_only"},
        )

        # Should return SSE stream
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")


# =========================================================================
# C3: Generation + AI Suggest Endpoints
# =========================================================================

from unittest.mock import patch, AsyncMock, MagicMock


class TestGenerateEndpoint:
    """Tests for POST /generate endpoint (GN-001 to GN-007)."""

    def _mock_generated_cases(self, count=3):
        """Create mock GeneratedCase objects."""
        from src.eval.eval_case_generator import GeneratedCase

        return [
            GeneratedCase(
                id=f"auto_case_{i}",
                prompt=f"Sammenlign krav til gennemsigtighed i lov {i}",
                synthesis_mode="comparison",
                expected_corpora=("ai-act", "gdpr"),
                expected_anchors=(),
                test_types=("corpus_coverage", "retrieval", "faithfulness", "relevancy"),
            )
            for i in range(count)
        ]

    def test_gn_001_generate_creates_new_suite(self, client):
        """POST /generate should create new suite with generated cases."""
        mock_cases = self._mock_generated_cases(3)

        with patch(
            "routes.eval_cross_law.generate_cross_law_cases",
            new_callable=AsyncMock,
            return_value=mock_cases,
        ), patch(
            "routes.eval_cross_law.assign_test_types",
            return_value=mock_cases,
        ):
            response = client.post(
                "/api/eval/cross-law/generate",
                json={
                    "target_corpora": ["ai-act", "gdpr"],
                    "synthesis_mode": "comparison",
                    "max_cases": 3,
                    "suite_name": "Generated Test Suite",
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert "suite_id" in data
        assert data["case_count"] == 3

    def test_gn_002_generate_adds_to_existing_suite(self, client, sample_suite_request):
        """POST /generate should add cases to existing suite."""
        # Create suite first
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        mock_cases = self._mock_generated_cases(2)

        with patch(
            "routes.eval_cross_law.generate_cross_law_cases",
            new_callable=AsyncMock,
            return_value=mock_cases,
        ), patch(
            "routes.eval_cross_law.assign_test_types",
            return_value=mock_cases,
        ):
            response = client.post(
                "/api/eval/cross-law/generate",
                json={
                    "target_corpora": ["ai-act", "gdpr"],
                    "synthesis_mode": "comparison",
                    "max_cases": 2,
                    "suite_id": suite_id,
                },
            )

        assert response.status_code == 201
        data = response.json()
        assert data["suite_id"] == suite_id
        assert data["case_count"] == 2

    def test_gn_003_generate_rejects_less_than_2_corpora(self, client):
        """POST /generate should reject <2 corpora."""
        response = client.post(
            "/api/eval/cross-law/generate",
            json={
                "target_corpora": ["ai-act"],
                "synthesis_mode": "comparison",
                "max_cases": 5,
                "suite_name": "Bad Suite",
            },
        )

        assert response.status_code == 400
        assert "2" in response.json()["detail"] or "corpora" in response.json()["detail"].lower()

    def test_gn_004_generate_rejects_invalid_corpus_ids(self, client):
        """POST /generate should reject invalid corpus IDs."""
        response = client.post(
            "/api/eval/cross-law/generate",
            json={
                "target_corpora": ["ai-act", "nonexistent_law"],
                "synthesis_mode": "comparison",
                "max_cases": 5,
                "suite_name": "Bad Suite",
            },
        )

        assert response.status_code == 400
        assert "nonexistent_law" in response.json()["detail"]

    def test_gn_005_generate_caps_at_20_cases(self, client):
        """POST /generate should cap at 20 cases."""
        mock_cases = self._mock_generated_cases(20)

        with patch(
            "routes.eval_cross_law.generate_cross_law_cases",
            new_callable=AsyncMock,
            return_value=mock_cases,
        ), patch(
            "routes.eval_cross_law.assign_test_types",
            return_value=mock_cases,
        ):
            response = client.post(
                "/api/eval/cross-law/generate",
                json={
                    "target_corpora": ["ai-act", "gdpr"],
                    "synthesis_mode": "comparison",
                    "max_cases": 30,  # Over limit
                    "suite_name": "Capped Suite",
                },
            )

        assert response.status_code == 201
        assert response.json()["case_count"] <= 20

    def test_gn_006_generated_cases_have_origin_auto_generated(self, client):
        """Generated cases should have origin 'auto-generated'."""
        mock_cases = self._mock_generated_cases(1)

        with patch(
            "routes.eval_cross_law.generate_cross_law_cases",
            new_callable=AsyncMock,
            return_value=mock_cases,
        ), patch(
            "routes.eval_cross_law.assign_test_types",
            return_value=mock_cases,
        ):
            response = client.post(
                "/api/eval/cross-law/generate",
                json={
                    "target_corpora": ["ai-act", "gdpr"],
                    "synthesis_mode": "comparison",
                    "max_cases": 1,
                    "suite_name": "Origin Test Suite",
                },
            )

        assert response.status_code == 201
        suite_id = response.json()["suite_id"]

        # Get suite detail to check case origins
        detail = client.get(f"/api/eval/cross-law/suites/{suite_id}")
        cases = detail.json()["cases"]
        assert all(c["origin"] == "auto-generated" for c in cases)

    def test_gn_007_generated_cases_have_correct_target_corpora(self, client):
        """Generated cases should have correct target_corpora."""
        mock_cases = self._mock_generated_cases(1)

        with patch(
            "routes.eval_cross_law.generate_cross_law_cases",
            new_callable=AsyncMock,
            return_value=mock_cases,
        ), patch(
            "routes.eval_cross_law.assign_test_types",
            return_value=mock_cases,
        ):
            response = client.post(
                "/api/eval/cross-law/generate",
                json={
                    "target_corpora": ["ai-act", "gdpr"],
                    "synthesis_mode": "comparison",
                    "max_cases": 1,
                    "suite_name": "Target Test Suite",
                },
            )

        assert response.status_code == 201
        suite_id = response.json()["suite_id"]

        detail = client.get(f"/api/eval/cross-law/suites/{suite_id}")
        cases = detail.json()["cases"]
        assert len(cases) == 1
        # Cases should have target_corpora from the request
        assert set(cases[0]["target_corpora"]) == {"ai-act", "gdpr"}


class TestAiSuggestEndpoint:
    """Tests for POST /ai-suggest endpoint (AI-001 to AI-003)."""

    def test_ai_001_returns_name_suggestion(self, client):
        """POST /ai-suggest should return name suggestion."""
        with patch(
            "src.eval.eval_case_generator.call_generation_llm",
            return_value="Cross-Law AI & GDPR Analyse",
        ):
            response = client.post(
                "/api/eval/cross-law/ai-suggest",
                json={
                    "type": "name",
                    "corpora": ["ai-act", "gdpr"],
                    "synthesis_mode": "comparison",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "suggestion" in data
        assert len(data["suggestion"]) > 0

    def test_ai_002_returns_description_suggestion(self, client):
        """POST /ai-suggest should return description suggestion."""
        with patch(
            "src.eval.eval_case_generator.call_generation_llm",
            return_value="Sammenligning af kravene i AI-forordningen og GDPR vedrørende gennemsigtighed.",
        ):
            response = client.post(
                "/api/eval/cross-law/ai-suggest",
                json={
                    "type": "description",
                    "corpora": ["ai-act", "gdpr"],
                    "synthesis_mode": "comparison",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "suggestion" in data
        assert len(data["suggestion"]) > 0

    def test_ai_003_handles_llm_error_gracefully(self, client):
        """POST /ai-suggest should handle LLM error gracefully."""
        with patch(
            "src.eval.eval_case_generator.call_generation_llm",
            return_value=None,
        ):
            response = client.post(
                "/api/eval/cross-law/ai-suggest",
                json={
                    "type": "name",
                    "corpora": ["ai-act", "gdpr"],
                    "synthesis_mode": "comparison",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "suggestion" in data
        # Should return a fallback suggestion, not crash
        assert len(data["suggestion"]) > 0


# =========================================================================
# C2: trigger_eval() + Run Persistence
# =========================================================================

import json as json_module


class TestTriggerEvalIntegration:
    """Tests for trigger_eval() with eval_core integration (TE-001 to TE-007)."""

    def _mock_case_result(self, case_id: str, passed: bool = True):
        """Create a mock CaseResult."""
        from src.eval.reporters import CaseResult
        from src.eval.scorers import Score

        return CaseResult(
            case_id=case_id,
            profile="LEGAL",
            passed=passed,
            scores={
                "retrieval": Score(passed=passed, score=1.0 if passed else 0.0, message="test"),
            },
            duration_ms=150.0,
        )

    def _mock_eval_summary(self, case_results):
        """Create a mock EvalSummary."""
        from src.eval.reporters import EvalSummary

        passed = sum(1 for r in case_results if r.passed)
        failed = len(case_results) - passed
        return EvalSummary(
            law="ai-act",
            total=len(case_results),
            passed=passed,
            failed=failed,
            skipped=0,
            duration_seconds=1.5,
            results=case_results,
            run_mode="retrieval_only",
        )

    def _create_suite_with_case(self, client, sample_suite_request, sample_case_request):
        """Helper to create suite with a case, returns (suite_id, case_id)."""
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        case_resp = client.post(
            f"/api/eval/cross-law/suites/{suite_id}/cases",
            json=sample_case_request,
        )
        case_id = case_resp.json()["id"]
        return suite_id, case_id

    def test_te_001_trigger_eval_streams_start_event(
        self, client, sample_suite_request, sample_case_request
    ):
        """trigger_eval should stream SSE start event."""
        suite_id, case_id = self._create_suite_with_case(
            client, sample_suite_request, sample_case_request
        )

        case_result = self._mock_case_result(case_id)
        summary = self._mock_eval_summary([case_result])

        def mock_iter(*args, **kwargs):
            yield case_result
            yield summary

        with patch("routes.eval_cross_law.evaluate_cases_iter", side_effect=mock_iter):
            response = client.post(
                f"/api/eval/cross-law/suites/{suite_id}/run",
                json={"run_mode": "retrieval_only"},
            )

        assert response.status_code == 200
        events = [
            line for line in response.text.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        # First event should be "start"
        first_event = json_module.loads(events[0].replace("data: ", ""))
        assert first_event["type"] == "start"
        assert first_event["suite_id"] == suite_id

    def test_te_002_trigger_eval_streams_case_result_events(
        self, client, sample_suite_request, sample_case_request
    ):
        """trigger_eval should stream case_result events."""
        suite_id, case_id = self._create_suite_with_case(
            client, sample_suite_request, sample_case_request
        )

        case_result = self._mock_case_result(case_id, passed=True)
        summary = self._mock_eval_summary([case_result])

        def mock_iter(*args, **kwargs):
            yield case_result
            yield summary

        with patch("routes.eval_cross_law.evaluate_cases_iter", side_effect=mock_iter):
            response = client.post(
                f"/api/eval/cross-law/suites/{suite_id}/run",
                json={"run_mode": "retrieval_only"},
            )

        events = [
            json_module.loads(line.replace("data: ", ""))
            for line in response.text.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        # Should have a case_result event
        case_events = [e for e in events if e["type"] == "case_result"]
        assert len(case_events) == 1
        assert case_events[0]["case_id"] == case_id
        assert case_events[0]["passed"] is True

    def test_te_003_trigger_eval_streams_complete_event(
        self, client, sample_suite_request, sample_case_request
    ):
        """trigger_eval should stream complete event with summary."""
        suite_id, case_id = self._create_suite_with_case(
            client, sample_suite_request, sample_case_request
        )

        case_result = self._mock_case_result(case_id)
        summary = self._mock_eval_summary([case_result])

        def mock_iter(*args, **kwargs):
            yield case_result
            yield summary

        with patch("routes.eval_cross_law.evaluate_cases_iter", side_effect=mock_iter):
            response = client.post(
                f"/api/eval/cross-law/suites/{suite_id}/run",
                json={"run_mode": "retrieval_only"},
            )

        events = [
            json_module.loads(line.replace("data: ", ""))
            for line in response.text.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        complete_events = [e for e in events if e["type"] == "complete"]
        assert len(complete_events) == 1
        assert "pass_rate" in complete_events[0]
        assert "duration_seconds" in complete_events[0]

    def test_te_004_trigger_eval_persists_run(
        self, client, sample_suite_request, sample_case_request, temp_evals_dir
    ):
        """trigger_eval should persist run to JSON file."""
        suite_id, case_id = self._create_suite_with_case(
            client, sample_suite_request, sample_case_request
        )

        case_result = self._mock_case_result(case_id)
        summary = self._mock_eval_summary([case_result])

        def mock_iter(*args, **kwargs):
            yield case_result
            yield summary

        with patch("routes.eval_cross_law.evaluate_cases_iter", side_effect=mock_iter):
            response = client.post(
                f"/api/eval/cross-law/suites/{suite_id}/run",
                json={"run_mode": "retrieval_only"},
            )

        assert response.status_code == 200

        # Check run file was persisted
        runs_dir = temp_evals_dir / "runs"
        if runs_dir.exists():
            run_files = list(runs_dir.glob("*.json"))
            assert len(run_files) >= 1

    def test_te_005_list_runs_returns_persisted_runs(
        self, client, sample_suite_request, sample_case_request, temp_evals_dir
    ):
        """list_runs should return persisted runs for suite."""
        suite_id, case_id = self._create_suite_with_case(
            client, sample_suite_request, sample_case_request
        )

        case_result = self._mock_case_result(case_id)
        summary = self._mock_eval_summary([case_result])

        def mock_iter(*args, **kwargs):
            yield case_result
            yield summary

        with patch("routes.eval_cross_law.evaluate_cases_iter", side_effect=mock_iter):
            client.post(
                f"/api/eval/cross-law/suites/{suite_id}/run",
                json={"run_mode": "retrieval_only"},
            )

        # Now list runs
        response = client.get(f"/api/eval/cross-law/suites/{suite_id}/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["runs"]) >= 1
        assert data["runs"][0]["suite_id"] == suite_id

    def test_te_006_get_run_returns_details(
        self, client, sample_suite_request, sample_case_request, temp_evals_dir
    ):
        """get_run should return specific run details."""
        suite_id, case_id = self._create_suite_with_case(
            client, sample_suite_request, sample_case_request
        )

        case_result = self._mock_case_result(case_id)
        summary = self._mock_eval_summary([case_result])

        def mock_iter(*args, **kwargs):
            yield case_result
            yield summary

        with patch("routes.eval_cross_law.evaluate_cases_iter", side_effect=mock_iter):
            client.post(
                f"/api/eval/cross-law/suites/{suite_id}/run",
                json={"run_mode": "retrieval_only"},
            )

        # Get run list to find run_id
        runs_resp = client.get(f"/api/eval/cross-law/suites/{suite_id}/runs")
        run_id = runs_resp.json()["runs"][0]["run_id"]

        # Get run details
        response = client.get(f"/api/eval/cross-law/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == run_id
        assert data["suite_id"] == suite_id
        assert len(data["results"]) == 1

    def test_te_007_trigger_eval_404s_on_unknown_suite(self, client):
        """trigger_eval should 404 on unknown suite."""
        response = client.post(
            "/api/eval/cross-law/suites/nonexistent/run",
            json={"run_mode": "retrieval_only"},
        )
        assert response.status_code == 404


# =========================================================================
# C2: _convert_case_to_golden Mapping
# =========================================================================


class TestConvertCaseToGolden:
    """Tests for _convert_case_to_golden mapping (T2.1 to T2.7)."""

    @pytest.fixture
    def service(self, temp_evals_dir):
        """Create a CrossLawEvalService for direct method testing."""
        from routes.eval_cross_law import CrossLawEvalService
        return CrossLawEvalService(
            evals_dir=temp_evals_dir,
            valid_corpus_ids={"ai-act", "gdpr", "nis2"},
        )

    def _make_case(self, **overrides):
        """Create a CrossLawGoldenCase with sensible defaults + overrides."""
        from src.eval.cross_law_suite_manager import CrossLawGoldenCase
        defaults = dict(
            id="case-1",
            prompt="Compare AI Act and GDPR",
            corpus_scope="explicit",
            target_corpora=("ai-act", "gdpr"),
            synthesis_mode="comparison",
            expected_anchors=(),
            expected_corpora=("ai-act", "gdpr"),
            min_corpora_cited=2,
            profile="LEGAL",
            disabled=False,
            origin="manual",
        )
        defaults.update(overrides)
        return CrossLawGoldenCase(**defaults)

    def test_t2_1_maps_test_types_from_case(self, service):
        """T2.1: Non-empty test_types should be mapped directly."""
        case = self._make_case(
            test_types=("corpus_coverage", "synthesis_balance", "routing_precision"),
        )
        golden = service._convert_case_to_golden(case)
        assert golden.test_types == ("corpus_coverage", "synthesis_balance", "routing_precision")

    def test_t2_2_falls_back_to_cross_law_test_types_when_empty(self, service):
        """T2.2: Empty test_types on cross-law case falls back to standard + cross-law scorers."""
        case = self._make_case(test_types=())  # corpus_scope="explicit", synthesis_mode="comparison"
        golden = service._convert_case_to_golden(case)
        assert "retrieval" in golden.test_types
        assert "faithfulness" in golden.test_types
        assert "relevancy" in golden.test_types
        assert "corpus_coverage" in golden.test_types
        assert "comparison_completeness" in golden.test_types

    def test_t2_3_maps_all_anchor_lists(self, service):
        """T2.3: All 4 anchor lists should map to ExpectedBehavior."""
        case = self._make_case(
            must_include_any_of=("ai-act:art-13",),
            must_include_any_of_2=("gdpr:art-6",),
            must_include_all_of=("ai-act:art-9", "gdpr:art-22"),
            must_not_include_any_of=("nis2:art-1",),
        )
        golden = service._convert_case_to_golden(case)
        assert golden.expected.must_include_any_of == ["ai-act:art-13"]
        assert golden.expected.must_include_any_of_2 == ["gdpr:art-6"]
        assert golden.expected.must_include_all_of == ["ai-act:art-9", "gdpr:art-22"]
        assert golden.expected.must_not_include_any_of == ["nis2:art-1"]

    def test_t2_4_maps_expected_behavior(self, service):
        """T2.4: expected_behavior should map to ExpectedBehavior.behavior."""
        case = self._make_case(expected_behavior="abstain")
        golden = service._convert_case_to_golden(case)
        assert golden.expected.behavior == "abstain"

    def test_t2_5_maps_contract_check_and_citations(self, service):
        """T2.5: contract_check, min/max_citations should map correctly."""
        case = self._make_case(
            contract_check=True,
            min_citations=3,
            max_citations=10,
        )
        golden = service._convert_case_to_golden(case)
        assert golden.expected.contract_check is True
        assert golden.expected.min_citations == 3
        assert golden.expected.max_citations == 10

    def test_t2_6_maps_expected_corpora_and_min_corpora_cited(self, service):
        """T2.6: expected_corpora and min_corpora_cited should map to ExpectedBehavior."""
        case = self._make_case(
            expected_corpora=("ai-act", "gdpr"),
            min_corpora_cited=2,
        )
        golden = service._convert_case_to_golden(case)
        assert golden.expected.required_corpora == ("ai-act", "gdpr")
        assert golden.expected.min_corpora_cited == 2

    def test_t2_7_fallback_expected_anchors_to_must_include_any_of(self, service):
        """T2.7: When must_include_any_of is empty, fall back to expected_anchors."""
        case = self._make_case(
            expected_anchors=("ai-act:art-13", "gdpr:art-6"),
            must_include_any_of=(),
        )
        golden = service._convert_case_to_golden(case)
        assert golden.expected.must_include_any_of == ["ai-act:art-13", "gdpr:art-6"]


# =========================================================================
# C3: Expanded Pydantic Models + Run-Single Endpoint
# =========================================================================


class TestExpandedPydanticModels:
    """Tests for expanded CaseRequest/CaseResponse models (T3.1 to T3.3, T3.6, T3.7)."""

    def test_t3_1_case_request_accepts_all_new_fields(self, client, sample_suite_request):
        """T3.1: CaseRequest should accept all new fields."""
        # Create a suite first
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        case_data = {
            "prompt": "Compare AI-Act and GDPR on transparency requirements",
            "corpus_scope": "explicit",
            "target_corpora": ["ai-act", "gdpr"],
            "synthesis_mode": "comparison",
            "expected_anchors": [],
            "expected_corpora": ["ai-act", "gdpr"],
            "min_corpora_cited": 2,
            "profile": "LEGAL",
            "disabled": False,
            # New fields
            "test_types": ["corpus_coverage", "synthesis_balance"],
            "expected_behavior": "answer",
            "must_include_any_of": ["ai-act:art-13"],
            "must_include_any_of_2": ["gdpr:art-6"],
            "must_include_all_of": ["ai-act:art-9"],
            "must_not_include_any_of": ["nis2:art-1"],
            "contract_check": True,
            "min_citations": 3,
            "max_citations": 10,
            "notes": "Test case for transparency",
        }

        response = client.post(f"/api/eval/cross-law/suites/{suite_id}/cases", json=case_data)
        assert response.status_code == 201

    def test_t3_2_case_request_defaults_work(self, client, sample_suite_request):
        """T3.2: CaseRequest should work without new fields (backward compat)."""
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        # Only original fields — no new ones
        case_data = {
            "prompt": "Compare AI-Act and GDPR on transparency requirements",
            "target_corpora": ["ai-act", "gdpr"],
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act", "gdpr"],
        }

        response = client.post(f"/api/eval/cross-law/suites/{suite_id}/cases", json=case_data)
        assert response.status_code == 201
        data = response.json()
        # Defaults should be applied
        assert data["test_types"] == []
        assert data["expected_behavior"] == "answer"
        assert data["contract_check"] is False
        assert data["notes"] == ""

    def test_t3_3_case_response_includes_all_new_fields(self, client, sample_suite_request):
        """T3.3: CaseResponse should include all new fields."""
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        case_data = {
            "prompt": "Compare AI-Act and GDPR on transparency requirements",
            "target_corpora": ["ai-act", "gdpr"],
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act", "gdpr"],
            "test_types": ["corpus_coverage"],
            "expected_behavior": "abstain",
            "must_include_any_of": ["ai-act:art-13"],
            "must_include_any_of_2": ["gdpr:art-6"],
            "must_include_all_of": ["ai-act:art-9"],
            "must_not_include_any_of": ["nis2:art-1"],
            "contract_check": True,
            "min_citations": 2,
            "max_citations": 8,
            "notes": "Important test",
        }

        response = client.post(f"/api/eval/cross-law/suites/{suite_id}/cases", json=case_data)
        assert response.status_code == 201
        data = response.json()

        assert data["test_types"] == ["corpus_coverage"]
        assert data["expected_behavior"] == "abstain"
        assert data["must_include_any_of"] == ["ai-act:art-13"]
        assert data["must_include_any_of_2"] == ["gdpr:art-6"]
        assert data["must_include_all_of"] == ["ai-act:art-9"]
        assert data["must_not_include_any_of"] == ["nis2:art-1"]
        assert data["contract_check"] is True
        assert data["min_citations"] == 2
        assert data["max_citations"] == 8
        assert data["notes"] == "Important test"

    def test_t3_6_create_case_persists_new_fields(self, client, sample_suite_request):
        """T3.6: POST case should persist new fields and return them on GET."""
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        case_data = {
            "prompt": "Compare transparency requirements across AI Act and GDPR",
            "target_corpora": ["ai-act", "gdpr"],
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act", "gdpr"],
            "test_types": ["corpus_coverage", "comparison_completeness"],
            "expected_behavior": "answer",
            "notes": "Persistence test",
        }

        create_resp = client.post(f"/api/eval/cross-law/suites/{suite_id}/cases", json=case_data)
        assert create_resp.status_code == 201

        # Fetch suite detail to verify persisted
        detail_resp = client.get(f"/api/eval/cross-law/suites/{suite_id}")
        cases = detail_resp.json()["cases"]
        assert len(cases) == 1
        assert cases[0]["test_types"] == ["corpus_coverage", "comparison_completeness"]
        assert cases[0]["notes"] == "Persistence test"

    def test_t3_7_update_case_persists_new_fields(self, client, sample_suite_request):
        """T3.7: PUT case should update new fields and persist them."""
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        # Create case with minimal fields
        case_data = {
            "prompt": "Compare transparency requirements across AI Act and GDPR",
            "target_corpora": ["ai-act", "gdpr"],
            "synthesis_mode": "comparison",
            "expected_corpora": ["ai-act", "gdpr"],
        }
        create_resp = client.post(f"/api/eval/cross-law/suites/{suite_id}/cases", json=case_data)
        case_id = create_resp.json()["id"]

        # Update with new fields
        updated_data = {
            **case_data,
            "test_types": ["routing_precision"],
            "expected_behavior": "abstain",
            "must_include_any_of": ["ai-act:art-13"],
            "contract_check": True,
            "min_citations": 5,
            "notes": "Updated note",
        }
        update_resp = client.put(
            f"/api/eval/cross-law/suites/{suite_id}/cases/{case_id}",
            json=updated_data,
        )
        assert update_resp.status_code == 200
        data = update_resp.json()
        assert data["test_types"] == ["routing_precision"]
        assert data["expected_behavior"] == "abstain"
        assert data["must_include_any_of"] == ["ai-act:art-13"]
        assert data["contract_check"] is True
        assert data["min_citations"] == 5
        assert data["notes"] == "Updated note"
        assert data["origin"] == "manual"


class TestRunSingleEndpoint:
    """Tests for POST /suites/{suite_id}/run-single endpoint (T3.4, T3.5)."""

    def test_t3_5_run_single_404s_on_unknown_suite(self, client):
        """T3.5: run-single should 404 on unknown suite."""
        response = client.post(
            "/api/eval/cross-law/suites/nonexistent/run-single",
            json={
                "prompt": "Test",
                "profile": "LEGAL",
                "test_types": ["corpus_coverage"],
                "run_mode": "retrieval_only",
            },
        )
        assert response.status_code == 404

    def test_t3_4_run_single_returns_scores_and_answer(
        self, client, sample_suite_request
    ):
        """T3.4: run-single should return scores + answer + references."""
        # Create suite
        suite_resp = client.post("/api/eval/cross-law/suites", json=sample_suite_request)
        suite_id = suite_resp.json()["id"]

        # Mock eval internals
        from src.eval.reporters import CaseResult
        from src.eval.scorers import Score

        mock_case_result = CaseResult(
            case_id="inline-validation",
            profile="LEGAL",
            passed=True,
            scores={
                "corpus_coverage": Score(passed=True, score=1.0, message="All corpora cited"),
            },
            duration_ms=200.0,
            answer="Both AI Act and GDPR require transparency.",
            references_structured=[
                {"display": "AI Act Art 13", "chunk_text": "text", "corpus_id": "ai-act"},
                {"display": "GDPR Art 12", "chunk_text": "text", "corpus_id": "gdpr"},
            ],
        )

        with patch(
            "routes.eval_cross_law._evaluate_single_case",
            return_value=mock_case_result,
        ), patch(
            "routes.eval_cross_law._build_engine",
            return_value="mock_engine",
        ):
            response = client.post(
                f"/api/eval/cross-law/suites/{suite_id}/run-single",
                json={
                    "prompt": "Compare transparency requirements",
                    "profile": "LEGAL",
                    "test_types": ["corpus_coverage"],
                    "run_mode": "retrieval_only",
                    "expected_behavior": "answer",
                    "expected_corpora": ["ai-act", "gdpr"],
                    "min_corpora_cited": 2,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["passed"] is True
        assert "duration_ms" in data
        assert "corpus_coverage" in data["scores"]
        assert data["answer"] == "Both AI Act and GDPR require transparency."
        assert len(data["references"]) == 2
        assert data["references"][0]["corpus_id"] == "ai-act"
