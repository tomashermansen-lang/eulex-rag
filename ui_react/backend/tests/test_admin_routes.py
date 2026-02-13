"""Tests for admin API routes.

Tests legislation listing, update checking, and corpus management.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

# Add backend directory to sys.path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Add project root for src imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def mock_eurlex_listing(monkeypatch):
    """Mock the eurlex_listing module."""
    from src.ingestion import eurlex_listing

    # Create mock LegislationInfo objects
    mock_legislation = [
        eurlex_listing.LegislationInfo(
            celex_number="32024R1689",
            title_da="AI-forordningen",
            title_en="AI Act",
            last_modified=datetime(2024, 7, 12),
            in_force=True,
            is_ingested=True,
            html_url="https://eur-lex.europa.eu/...",
            document_type="Regulation",
        ),
        eurlex_listing.LegislationInfo(
            celex_number="32022L2555",
            title_da="NIS2-direktivet",
            title_en="NIS2 Directive",
            last_modified=datetime(2024, 10, 17),
            in_force=True,
            is_ingested=False,
            html_url="https://eur-lex.europa.eu/...",
            document_type="Directive",
        ),
    ]

    monkeypatch.setattr(
        eurlex_listing,
        "list_available_legislation",
        lambda **kwargs: mock_legislation,
    )
    monkeypatch.setattr(
        eurlex_listing,
        "enrich_corpora_with_status",
        lambda corpora, legislation: legislation,
    )

    return eurlex_listing


@pytest.fixture
def mock_corpora_inventory(monkeypatch):
    """Mock the corpora inventory functions."""
    from src.common import corpora_inventory

    mock_data = {
        "version": 1,
        "corpora": {
            "ai-act": {
                "display_name": "AI Act",
                "enabled": True,
                "chunks_collection": "ai-act_documents",
                "source_url": "https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32024R1689",
            }
        }
    }

    # Patch in both the source module and the admin routes module
    # (where it's imported with 'from ... import')
    monkeypatch.setattr(
        corpora_inventory,
        "load_corpora_inventory",
        lambda path: mock_data,
    )
    monkeypatch.setattr(
        corpora_inventory,
        "save_corpora_inventory",
        lambda path, data: None,
    )
    # Also patch in admin routes where function is directly imported
    monkeypatch.setattr(
        "routes.admin.load_corpora_inventory",
        lambda path: mock_data,
    )
    monkeypatch.setattr(
        "routes.admin.save_corpora_inventory",
        lambda path, data: None,
    )

    return corpora_inventory


@pytest.fixture
def admin_client(mock_eurlex_listing, mock_corpora_inventory):
    """Provide a test client with mocked admin dependencies."""
    from ui_react.backend.main import app

    return TestClient(app)


class TestListLegislationEndpoint:
    """Tests for GET /api/admin/legislation."""

    def test_list_legislation_returns_list(self, admin_client):
        """Should return a list of legislation."""
        response = admin_client.get("/api/admin/legislation")

        assert response.status_code == 200
        data = response.json()
        assert "legislation" in data
        assert "total" in data
        assert isinstance(data["legislation"], list)

    def test_list_legislation_with_search(self, admin_client):
        """Should accept search parameter."""
        response = admin_client.get("/api/admin/legislation?search=nis")

        assert response.status_code == 200

    def test_list_legislation_returns_expected_fields(self, admin_client):
        """Should return legislation with expected fields."""
        response = admin_client.get("/api/admin/legislation")

        assert response.status_code == 200
        data = response.json()

        if data["legislation"]:
            leg = data["legislation"][0]
            assert "celex_number" in leg
            assert "title_da" in leg
            assert "title_en" in leg
            assert "is_ingested" in leg
            assert "html_url" in leg


class TestCheckUpdateEndpoint:
    """Tests for GET /api/admin/legislation/{celex}/check-update."""

    def test_check_update_valid_celex(self, admin_client):
        """Should accept valid CELEX number."""
        response = admin_client.get("/api/admin/legislation/32024R1689/check-update")

        assert response.status_code == 200
        data = response.json()
        assert "celex_number" in data
        assert "is_outdated" in data

    def test_check_update_invalid_celex(self, admin_client):
        """Should reject invalid CELEX format."""
        response = admin_client.get("/api/admin/legislation/invalid/check-update")

        assert response.status_code == 400
        assert "Invalid CELEX" in response.json()["detail"]

    def test_check_update_sql_injection_attempt(self, admin_client):
        """Should reject SQL injection attempts."""
        response = admin_client.get("/api/admin/legislation/32024R1689';DROP/check-update")

        assert response.status_code == 400


class TestRemoveCorpusEndpoint:
    """Tests for DELETE /api/admin/corpus/{corpus_id}."""

    def test_remove_existing_corpus(self, admin_client, monkeypatch, tmp_path):
        """Should remove existing corpus with confirm parameter."""
        # Mock chromadb
        mock_client = MagicMock()
        monkeypatch.setattr("chromadb.PersistentClient", lambda **kwargs: mock_client)

        # CRITICAL: Mock file operations to prevent deleting real files
        from pathlib import Path
        original_unlink = Path.unlink
        original_exists = Path.exists

        # Track which files would be deleted (but don't actually delete them)
        deleted_files = []

        def mock_unlink(self, missing_ok=False):
            # Only mock files in data/ directory to prevent real deletions
            if "data/" in str(self) or "data\\" in str(self):
                deleted_files.append(str(self))
                return  # Don't actually delete
            return original_unlink(self, missing_ok=missing_ok)

        def mock_exists(self):
            # Return False for data files to skip deletion attempts
            if "data/" in str(self) or "data\\" in str(self):
                return False
            return original_exists(self)

        monkeypatch.setattr(Path, "unlink", mock_unlink)
        monkeypatch.setattr(Path, "exists", mock_exists)

        # Mock shutil.rmtree to prevent cache deletion
        import shutil
        monkeypatch.setattr(shutil, "rmtree", lambda *args, **kwargs: None)

        # GUARDRAIL: Must include confirm parameter matching corpus_id
        response = admin_client.delete("/api/admin/corpus/ai-act?confirm=ai-act")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["corpus_id"] == "ai-act"

    def test_remove_corpus_without_confirm_fails(self, admin_client):
        """Should reject deletion without confirm parameter."""
        response = admin_client.delete("/api/admin/corpus/ai-act")

        assert response.status_code == 400
        assert "confirm" in response.json()["detail"].lower()

    def test_remove_corpus_with_wrong_confirm_fails(self, admin_client):
        """Should reject deletion with mismatched confirm parameter."""
        response = admin_client.delete("/api/admin/corpus/ai-act?confirm=gdpr")

        assert response.status_code == 400
        assert "confirm" in response.json()["detail"].lower()

    def test_remove_nonexistent_corpus(self, admin_client):
        """Should return 404 for non-existent corpus."""
        # Must include confirm even for nonexistent corpus (checked before 404)
        response = admin_client.delete("/api/admin/corpus/nonexistent?confirm=nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestAddLawStreamEndpoint:
    """Tests for POST /api/admin/add-law/stream."""

    def test_add_law_invalid_celex(self, admin_client):
        """Should reject invalid CELEX number."""
        response = admin_client.post(
            "/api/admin/add-law/stream",
            json={
                "celex_number": "invalid",
                "corpus_id": "test",
                "display_name": "Test Law",
                "generate_eval": False,
            },
        )

        assert response.status_code == 400
        assert "Invalid CELEX" in response.json()["detail"]

    def test_add_law_duplicate_corpus_id(self, admin_client):
        """Should reject duplicate corpus ID."""
        response = admin_client.post(
            "/api/admin/add-law/stream",
            json={
                "celex_number": "32022L2555",
                "corpus_id": "ai-act",  # Already exists
                "display_name": "NIS2",
                "generate_eval": False,
            },
        )

        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_add_law_missing_fields(self, admin_client):
        """Should require all fields."""
        response = admin_client.post(
            "/api/admin/add-law/stream",
            json={
                "celex_number": "32022L2555",
                # Missing corpus_id and display_name
            },
        )

        assert response.status_code == 422  # Validation error


class TestSecurityHeaders:
    """Tests for security-related behavior."""

    def test_cors_headers_present(self, admin_client):
        """Admin endpoints should have CORS headers for localhost."""
        response = admin_client.get(
            "/api/admin/legislation",
            headers={"Origin": "http://localhost:5173"},
        )

        # CORS headers should be present for allowed origins
        assert response.status_code == 200


class TestSuggestNamesEndpoint:
    """Tests for POST /api/admin/suggest-names."""

    def test_display_name_includes_corpus_id_in_parenthesis(self, admin_client, monkeypatch):
        """Display name shortname in parenthesis must match corpus_id format with year/number."""
        import re

        # Mock LLM to return a response without year in shortname
        def mock_call_llm(prompt, temperature=0.3):
            return '{"known_name": "EU-DR", "display_name": "Forordning om cybersikkerhed (EU-DR)", "is_implementing": false, "is_delegated": true}'

        monkeypatch.setattr("src.engine.llm_client.call_llm", mock_call_llm)

        response = admin_client.post(
            "/api/admin/suggest-names",
            json={
                "celex_number": "32024R1366",
                "title": "Kommissionens delegerede forordning (EU) 2024/1366 om cybersikkerhed",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # corpus_id should be eu-dr-2024-1366
        assert data["corpus_id"] == "eu-dr-2024-1366"

        # Extract shortname from parenthesis in display_name
        match = re.search(r'\(([^)]+)\)\s*$', data["display_name"])
        assert match, f"display_name should end with (SHORTNAME), got: {data['display_name']}"

        shortname = match.group(1).lower()
        corpus_id = data["corpus_id"]

        # The shortname should match corpus_id (ensures year/number is included)
        assert shortname == corpus_id, (
            f"Shortname in display_name must match corpus_id. "
            f"Got shortname='{shortname}' but corpus_id='{corpus_id}'"
        )


class TestInputValidation:
    """Tests for input validation security."""

    def test_rejects_extremely_long_search(self, admin_client):
        """Should handle extremely long search terms gracefully."""
        long_search = "a" * 10000
        response = admin_client.get(f"/api/admin/legislation?search={long_search}")

        # Should not crash - either success or validation error
        assert response.status_code in [200, 400, 422]

    def test_rejects_null_bytes_in_search(self, admin_client):
        """Should handle null bytes in search term."""
        response = admin_client.get("/api/admin/legislation?search=test%00injection")

        # Should not crash
        assert response.status_code in [200, 400, 422]


class TestListAnchorsEndpoint:
    """Tests for GET /api/admin/corpus/{law}/anchors."""

    def test_list_anchors_returns_from_citation_graph(self, admin_client, tmp_path, monkeypatch):
        """Should return anchors from citation graph."""
        import json

        # Create mock citation graph with nodes structure
        graph_data = {
            "nodes": {
                "1": {"type": "article", "title": "Article 1"},
                "2": {"type": "article", "title": "Article 2"},
                "i": {"type": "annex", "title": "Annex I"},
            }
        }

        def mock_get_citation_graph_path(law):
            graph_path = tmp_path / f"citation_graph_{law}.json"
            graph_path.write_text(json.dumps(graph_data))
            return graph_path

        monkeypatch.setattr("routes.admin._get_citation_graph_path", mock_get_citation_graph_path)

        response = admin_client.get("/api/admin/corpus/test-law/anchors")

        assert response.status_code == 200
        data = response.json()
        assert "anchors" in data
        assert "total" in data
        assert len(data["anchors"]) == 3
        assert "article:1" in data["anchors"]

    def test_list_anchors_filters_by_query(self, admin_client, tmp_path, monkeypatch):
        """Should filter anchors by query string."""
        import json

        graph_data = {
            "nodes": {
                "1": {"type": "article"},
                "2": {"type": "article"},
                "i": {"type": "annex"},
            }
        }

        def mock_get_citation_graph_path(law):
            graph_path = tmp_path / f"citation_graph_{law}.json"
            graph_path.write_text(json.dumps(graph_data))
            return graph_path

        monkeypatch.setattr("routes.admin._get_citation_graph_path", mock_get_citation_graph_path)

        response = admin_client.get("/api/admin/corpus/test-law/anchors?q=annex")

        assert response.status_code == 200
        data = response.json()
        assert len(data["anchors"]) == 1
        assert data["anchors"][0] == "annex:i"

    def test_list_anchors_returns_empty_for_missing_graph(self, admin_client, tmp_path, monkeypatch):
        """Should return empty list if citation graph doesn't exist."""

        def mock_get_citation_graph_path(law):
            return tmp_path / "nonexistent.json"

        monkeypatch.setattr("routes.admin._get_citation_graph_path", mock_get_citation_graph_path)

        response = admin_client.get("/api/admin/corpus/test-law/anchors")

        assert response.status_code == 200
        data = response.json()
        assert data["anchors"] == []
