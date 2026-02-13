"""Tests for eval dashboard API routes."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestEvalOverview:
    """Tests for /api/eval/overview endpoint."""

    def test_overview_returns_laws(self, client, tmp_path):
        """Should return stats for each law with golden cases."""
        # Create a mock golden cases file
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test_law.yaml"
        golden_file.write_text("""
- id: test-case-1
  profile: LEGAL
  prompt: Test question?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
""")

        # Mock the paths
        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("routes.eval._load_latest_run", return_value=None):

            response = client.get("/api/eval/overview")

        assert response.status_code == 200
        data = response.json()
        assert "laws" in data
        assert "test_types" in data
        assert "total_cases" in data

    def test_overview_includes_all_test_types(self, client):
        """Should include all valid test types in response."""
        with patch("routes.eval._get_evals_dir") as mock_dir:
            mock_dir.return_value = Path("/nonexistent")

            response = client.get("/api/eval/overview")

        assert response.status_code == 200
        data = response.json()

        expected_types = ["retrieval", "faithfulness", "relevancy", "abstention", "robustness", "multi_hop"]
        assert data["test_types"] == expected_types


class TestEvalRuns:
    """Tests for /api/eval/runs endpoint."""

    def test_runs_list_empty(self, client):
        """Should return empty list when no runs exist."""
        with patch("routes.eval._load_all_runs", return_value=[]):
            response = client.get("/api/eval/runs")

        assert response.status_code == 200
        data = response.json()
        assert data["runs"] == []
        assert data["total"] == 0

    def test_runs_list_with_filter(self, client):
        """Should filter runs by law parameter."""
        mock_runs = [
            ("run1", {"meta": {"law": "ai-act", "timestamp": "2026-01-01T00:00:00Z"}, "summary": {"total": 10, "passed": 10, "failed": 0, "pass_rate": 1.0}}),
            ("run2", {"meta": {"law": "gdpr", "timestamp": "2026-01-02T00:00:00Z"}, "summary": {"total": 5, "passed": 5, "failed": 0, "pass_rate": 1.0}}),
        ]

        with patch("routes.eval._load_all_runs", return_value=mock_runs):
            response = client.get("/api/eval/runs?law=ai-act")

        assert response.status_code == 200
        data = response.json()
        assert len(data["runs"]) == 1
        assert data["runs"][0]["law"] == "ai-act"


class TestEvalRunDetail:
    """Tests for /api/eval/runs/{run_id} endpoint."""

    def test_run_not_found(self, client):
        """Should return 404 for non-existent run."""
        with patch("routes.eval._get_runs_dir") as mock_dir:
            mock_dir.return_value = Path("/nonexistent")

            response = client.get("/api/eval/runs/nonexistent_run")

        assert response.status_code == 404


class TestTriggerEval:
    """Tests for /api/eval/trigger endpoint."""

    def test_trigger_missing_law(self, client):
        """Should return 404 for law without golden cases."""
        with patch("routes.eval._load_golden_cases", return_value=[]):
            response = client.post(
                "/api/eval/trigger",
                json={"law": "nonexistent", "run_mode": "retrieval_only"}
            )

        assert response.status_code == 404

    def test_trigger_valid_request(self, client):
        """Should accept valid trigger request."""
        mock_cases = [{"id": "test-1", "profile": "LEGAL", "prompt": "Test?"}]

        with patch("routes.eval._load_golden_cases", return_value=mock_cases):
            response = client.post(
                "/api/eval/trigger",
                json={"law": "test-law", "run_mode": "retrieval_only"}
            )

        # Should return streaming response
        assert response.status_code == 200
        assert response.headers.get("content-type") == "text/event-stream; charset=utf-8"


class TestEvalCasesCRUD:
    """Tests for /api/eval/cases CRUD endpoints."""

    def test_list_cases_returns_all_for_law(self, client, tmp_path):
        """Should return all cases for a law."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-basic
  profile: LEGAL
  prompt: What is the test?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
- id: test-law-02-second
  profile: ENGINEERING
  prompt: How does it work?
  test_types:
    - retrieval
    - faithfulness
  origin: manual
  expected:
    must_include_any_of:
      - article:2
""")

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.get("/api/eval/cases/test-law")

        assert response.status_code == 200
        data = response.json()
        assert "cases" in data
        assert len(data["cases"]) == 2
        assert data["cases"][0]["id"] == "test-law-01-basic"
        assert data["cases"][1]["id"] == "test-law-02-second"

    def test_list_cases_empty_for_nonexistent_law(self, client, tmp_path):
        """Should return empty list for non-existent law."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.get("/api/eval/cases/nonexistent")

        assert response.status_code == 200
        data = response.json()
        assert data["cases"] == []

    def test_get_case_returns_single(self, client, tmp_path):
        """Should return single case by ID."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-basic
  profile: LEGAL
  prompt: What is the test question that we want to ask?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
""")

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.get("/api/eval/cases/test-law/test-law-01-basic")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-law-01-basic"
        assert data["profile"] == "LEGAL"
        assert data["origin"] == "auto"

    def test_get_case_not_found(self, client, tmp_path):
        """Should return 404 for non-existent case."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.get("/api/eval/cases/test-law/nonexistent")

        assert response.status_code == 404

    def test_create_case_success(self, client, tmp_path):
        """Should create new case and return it."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        case_data = {
            "profile": "LEGAL",
            "prompt": "What are the new requirements for testing?",
            "test_types": ["retrieval"],
            "expected": {
                "must_include_any_of": ["article:5"],
                "behavior": "answer",
            },
        }

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.post("/api/eval/cases/test-law", json=case_data)

        assert response.status_code == 201
        data = response.json()
        assert data["profile"] == "LEGAL"
        assert data["origin"] == "manual"
        assert "id" in data

    def test_create_case_invalid_data_returns_422(self, client, tmp_path):
        """Should return 422 for invalid case data."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        # Missing required fields
        case_data = {
            "profile": "INVALID",
        }

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.post("/api/eval/cases/test-law", json=case_data)

        assert response.status_code == 422

    def test_update_case_success(self, client, tmp_path):
        """Should update case and return updated version."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-basic
  profile: LEGAL
  prompt: Original question that is long enough?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
""")

        update_data = {
            "prompt": "Updated question that is definitely long enough to pass validation?",
        }

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.put("/api/eval/cases/test-law/test-law-01-basic", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["prompt"] == "Updated question that is definitely long enough to pass validation?"
        assert data["origin"] == "manual"  # Changed from auto to manual

    def test_update_case_not_found_returns_404(self, client, tmp_path):
        """Should return 404 for non-existent case."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.put(
                "/api/eval/cases/test-law/nonexistent",
                json={"prompt": "Updated question that is long enough?"}
            )

        assert response.status_code == 404

    def test_delete_case_success(self, client, tmp_path):
        """Should delete case and return 204."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-to-delete
  profile: LEGAL
  prompt: Delete this question?
  test_types:
    - retrieval
  origin: manual
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
""")

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.delete("/api/eval/cases/test-law/test-law-01-to-delete")

        assert response.status_code == 204

    def test_delete_case_not_found_returns_404(self, client, tmp_path):
        """Should return 404 for non-existent case."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.delete("/api/eval/cases/test-law/nonexistent")

        assert response.status_code == 404

    def test_duplicate_case_creates_copy_with_new_id(self, client, tmp_path):
        """Should duplicate case with new ID."""
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-original
  profile: LEGAL
  prompt: Original question to duplicate for testing purposes?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
""")

        with patch("routes.eval._get_evals_dir", return_value=evals_dir), \
             patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            response = client.post("/api/eval/cases/test-law/test-law-01-original/duplicate")

        assert response.status_code == 201
        data = response.json()
        assert data["id"] != "test-law-01-original"
        assert data["prompt"] == "Original question to duplicate for testing purposes?"
        assert data["origin"] == "manual"
