"""Tests for API routes."""

from __future__ import annotations

import json


class TestAskEndpoint:
    """Tests for /api/ask endpoint."""

    def test_ask_returns_valid_response(self, client):
        """Test that ask endpoint returns expected response structure."""
        response = client.post(
            "/api/ask",
            json={"question": "What is prohibited?", "law": "ai-act"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "references" in data
        assert "retrieval_metrics" in data
        assert "response_time_seconds" in data

        assert isinstance(data["answer"], str)
        assert isinstance(data["references"], list)
        assert isinstance(data["response_time_seconds"], float)

    def test_ask_with_all_parameters(self, client):
        """Test ask endpoint with all optional parameters."""
        response = client.post(
            "/api/ask",
            json={
                "question": "What is prohibited?",
                "law": "ai-act",
                "user_profile": "ENGINEERING",
            }
        )

        assert response.status_code == 200

    def test_ask_references_structure(self, client):
        """Test that references have expected fields."""
        response = client.post(
            "/api/ask",
            json={"question": "Test?", "law": "ai-act"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["references"]) > 0
        ref = data["references"][0]

        assert "idx" in ref
        assert "display" in ref
        assert "chunk_text" in ref

    def test_ask_missing_question_fails(self, client):
        """Test that missing question returns 422."""
        response = client.post(
            "/api/ask",
            json={"law": "ai-act"}
        )

        assert response.status_code == 422

    def test_ask_missing_law_fails(self, client):
        """Test that missing law returns 422."""
        response = client.post(
            "/api/ask",
            json={"question": "Test?"}
        )

        assert response.status_code == 422


class TestAskStreamEndpoint:
    """Tests for /api/ask/stream endpoint."""

    def test_stream_returns_sse(self, client):
        """Test that stream endpoint returns SSE format."""
        response = client.post(
            "/api/ask/stream",
            json={"question": "What is prohibited?", "law": "ai-act"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

    def test_stream_yields_chunks_then_result(self, client):
        """Test that stream yields chunk events followed by result."""
        response = client.post(
            "/api/ask/stream",
            json={"question": "Test?", "law": "ai-act"}
        )

        content = response.text
        lines = [l for l in content.split("\n") if l.startswith("data: ")]

        # Should have multiple events
        assert len(lines) >= 2

        # Parse events
        events = []
        for line in lines:
            data = line[6:]  # Remove "data: " prefix
            if data != "[DONE]":
                events.append(json.loads(data))

        # Should have chunk events
        chunk_events = [e for e in events if e.get("type") == "chunk"]
        assert len(chunk_events) > 0

        # Should have exactly one result event
        result_events = [e for e in events if e.get("type") == "result"]
        assert len(result_events) == 1

        # Result should have expected structure
        result = result_events[0]["data"]
        assert "answer" in result
        assert "references" in result


class TestConfigEndpoints:
    """Tests for config endpoints."""

    def test_get_corpora(self, client):
        """Test /api/corpora endpoint."""
        response = client.get("/api/corpora")

        assert response.status_code == 200
        data = response.json()

        assert "corpora" in data
        assert len(data["corpora"]) > 0

        corpus = data["corpora"][0]
        assert "id" in corpus
        assert "name" in corpus

    def test_get_examples(self, client):
        """Test /api/examples endpoint."""
        response = client.get("/api/examples")

        assert response.status_code == 200
        data = response.json()

        assert "examples" in data
        assert isinstance(data["examples"], dict)

    def test_health_check(self, client):
        """Test /api/health endpoint."""
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ok"
        assert "version" in data


class TestApiRoot:
    """Tests for API root."""

    def test_api_root(self, client):
        """Test /api endpoint returns info."""
        response = client.get("/api")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data
