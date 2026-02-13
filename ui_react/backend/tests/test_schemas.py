"""Tests for Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ui_react.backend.schemas import (
    AskRequest,
    AskResponse,
    Reference,
    StreamChunk,
    CorporaResponse,
    CorpusInfo,
    ExamplesResponse,
    HealthResponse,
)


class TestAskRequest:
    """Tests for AskRequest schema."""

    def test_valid_minimal_request(self):
        """Test minimal valid request."""
        request = AskRequest(question="What is AI?", law="ai-act")

        assert request.question == "What is AI?"
        assert request.law == "ai-act"
        assert request.user_profile == "LEGAL"  # default

    def test_valid_full_request(self):
        """Test request with all fields."""
        request = AskRequest(
            question="What is prohibited?",
            law="gdpr",
            user_profile="ENGINEERING",
        )

        assert request.question == "What is prohibited?"
        assert request.law == "gdpr"
        assert request.user_profile == "ENGINEERING"

    def test_empty_question_fails(self):
        """Test that empty question is rejected."""
        with pytest.raises(ValidationError):
            AskRequest(question="", law="ai-act")


class TestReference:
    """Tests for Reference schema."""

    def test_valid_reference(self):
        """Test valid reference."""
        ref = Reference(
            idx=1,
            display="Article 5",
            chunk_text="Some text",
            corpus_id="ai-act",
            article="5",
        )

        assert ref.idx == 1
        assert ref.display == "Article 5"
        assert ref.article == "5"

    def test_reference_with_string_idx(self):
        """Test reference with string index."""
        ref = Reference(idx="1a", display="Test", chunk_text="Text")
        assert ref.idx == "1a"

    def test_reference_optional_fields(self):
        """Test that optional fields default to None."""
        ref = Reference(idx=1, display="Test", chunk_text="Text")

        assert ref.corpus_id is None
        assert ref.article is None
        assert ref.recital is None
        assert ref.annex is None


class TestAskResponse:
    """Tests for AskResponse schema."""

    def test_valid_response(self):
        """Test valid response."""
        response = AskResponse(
            answer="This is the answer.",
            references=[
                Reference(idx=1, display="Source 1", chunk_text="Text 1")
            ],
            retrieval_metrics={"best_distance": 0.25},
            response_time_seconds=1.5,
        )

        assert response.answer == "This is the answer."
        assert len(response.references) == 1
        assert response.response_time_seconds == 1.5

    def test_response_defaults(self):
        """Test response with defaults."""
        response = AskResponse(
            answer="Answer",
            response_time_seconds=1.0,
        )

        assert response.references == []
        assert response.retrieval_metrics == {}


class TestStreamChunk:
    """Tests for StreamChunk schema."""

    def test_text_chunk(self):
        """Test text chunk event."""
        chunk = StreamChunk(type="chunk", content="Hello ")
        assert chunk.type == "chunk"
        assert chunk.content == "Hello "
        assert chunk.data is None

    def test_result_chunk(self):
        """Test result event."""
        response = AskResponse(answer="Done", response_time_seconds=1.0)
        chunk = StreamChunk(type="result", data=response)

        assert chunk.type == "result"
        assert chunk.content is None
        assert chunk.data.answer == "Done"


class TestCorporaResponse:
    """Tests for CorporaResponse schema."""

    def test_valid_corpora(self):
        """Test valid corpora response."""
        response = CorporaResponse(
            corpora=[
                CorpusInfo(id="ai-act", name="AI Act", source_url="https://example.com"),
                CorpusInfo(id="gdpr", name="GDPR"),
            ]
        )

        assert len(response.corpora) == 2
        assert response.corpora[0].id == "ai-act"
        assert response.corpora[1].source_url is None


class TestExamplesResponse:
    """Tests for ExamplesResponse schema."""

    def test_valid_examples(self):
        """Test valid examples response."""
        response = ExamplesResponse(
            examples={
                "ai-act": {
                    "LEGAL": ["Q1?", "Q2?"],
                    "ENGINEERING": ["Q3?"],
                }
            }
        )

        assert "ai-act" in response.examples
        assert len(response.examples["ai-act"]["LEGAL"]) == 2


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_defaults(self):
        """Test default values."""
        response = HealthResponse()

        assert response.status == "ok"
        assert response.version == "1.0.0"

    def test_custom_values(self):
        """Test custom values."""
        response = HealthResponse(status="degraded", version="2.0.0")

        assert response.status == "degraded"
        assert response.version == "2.0.0"
