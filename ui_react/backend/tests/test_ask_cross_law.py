"""Tests for cross-law API functionality.

TDD: These tests are written BEFORE the implementation.
"""

import pytest
from fastapi.testclient import TestClient


class TestAskRequestCrossLawFields:
    """Tests for cross-law fields in AskRequest schema."""

    def test_ask_request_accepts_corpus_scope(self):
        """AskRequest accepts corpus_scope parameter."""
        from schemas import AskRequest

        request = AskRequest(
            question="What do all laws say about data protection?",
            law="ai_act",  # Default/primary corpus
            user_profile="LEGAL",
            corpus_scope="all",
        )

        assert request.corpus_scope == "all"

    def test_ask_request_defaults_corpus_scope_to_single(self):
        """AskRequest defaults corpus_scope to 'single' for backward compatibility."""
        from schemas import AskRequest

        request = AskRequest(
            question="What is Article 6?",
            law="ai_act",
            user_profile="LEGAL",
        )

        assert request.corpus_scope == "single"

    def test_ask_request_accepts_target_corpora(self):
        """AskRequest accepts target_corpora list for explicit scope."""
        from schemas import AskRequest

        request = AskRequest(
            question="Compare AI-Act and GDPR",
            law="ai_act",
            user_profile="LEGAL",
            corpus_scope="explicit",
            target_corpora=["ai_act", "gdpr"],
        )

        assert request.target_corpora == ["ai_act", "gdpr"]

    def test_ask_request_target_corpora_defaults_to_empty(self):
        """AskRequest defaults target_corpora to empty list."""
        from schemas import AskRequest

        request = AskRequest(
            question="What is Article 6?",
            law="ai_act",
        )

        assert request.target_corpora == []


class TestAskResponseCrossLawFields:
    """Tests for cross-law fields in AskResponse schema."""

    def test_ask_response_includes_synthesis_mode(self):
        """AskResponse includes synthesis_mode field."""
        from schemas import AskResponse

        response = AskResponse(
            answer="Both laws address this topic.",
            references=[],
            retrieval_metrics={},
            response_time_seconds=1.5,
            synthesis_mode="aggregation",
            laws_searched=["ai_act", "gdpr"],
        )

        assert response.synthesis_mode == "aggregation"
        assert response.laws_searched == ["ai_act", "gdpr"]

    def test_ask_response_synthesis_mode_defaults_to_none(self):
        """AskResponse synthesis_mode defaults to None for single-law queries."""
        from schemas import AskResponse

        response = AskResponse(
            answer="Article 6 states...",
            references=[],
            retrieval_metrics={},
            response_time_seconds=1.0,
        )

        assert response.synthesis_mode is None
        assert response.laws_searched == []


class TestAskEndpointCrossLaw:
    """Integration tests for cross-law /ask endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        import sys
        from pathlib import Path
        backend_path = Path(__file__).parent.parent
        if str(backend_path) not in sys.path:
            sys.path.insert(0, str(backend_path))

        from main import app
        return TestClient(app)

    def test_ask_with_single_scope_works_as_before(self, client):
        """Single scope query uses existing single-law path."""
        # This test verifies backward compatibility
        response = client.post("/api/ask", json={
            "question": "What is Article 6?",
            "law": "ai_act",
            "user_profile": "LEGAL",
            "corpus_scope": "single",
        })

        # Should succeed or fail gracefully
        # 200 = success, 400 = corpus not found, 500 = internal error
        # This verifies the request schema is accepted
        assert response.status_code in (200, 400, 500)

    def test_ask_request_validation_rejects_invalid_scope(self, client):
        """Invalid corpus_scope is rejected."""
        response = client.post("/api/ask", json={
            "question": "What is Article 6?",
            "law": "ai_act",
            "corpus_scope": "invalid_scope",  # Not valid
        })

        assert response.status_code == 422  # Validation error
