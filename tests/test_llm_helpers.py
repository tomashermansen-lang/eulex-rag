"""Tests for common LLM helper functions.

TDD: These tests verify src/common/llm_helpers.py provides correct
shared LLM generation utilities used by both ingestion and eval.

Requirement mapping:
- LH-001: call_generation_llm returns string on success (R2.1)
- LH-002: call_generation_llm returns None on API error (R2.1)
- LH-003: parse_json_response extracts JSON from plain text (R2.1)
- LH-004: parse_json_response handles markdown-wrapped JSON (R2.1)
- LH-005: parse_json_response returns None on invalid JSON (R2.1)
- LH-006: load_article_content returns formatted string (R2.5)
- LH-007: load_article_content returns placeholder for missing corpus (R2.5)
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.llm_helpers import (
    call_generation_llm,
    parse_json_response,
    load_article_content,
)


class TestCallGenerationLlm:
    """Tests for call_generation_llm function."""

    def test_lh_001_returns_string_on_success(self):
        """Successful LLM call should return response content string."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated text"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch(
            "src.common.llm_helpers._get_openai_client",
            return_value=mock_client,
        ):
            result = call_generation_llm(
                "Test prompt",
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=100,
            )

        assert result == "Generated text"
        mock_client.chat.completions.create.assert_called_once()

    def test_lh_002_returns_none_on_api_error(self):
        """LLM API error should return None, not raise."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with patch(
            "src.common.llm_helpers._get_openai_client",
            return_value=mock_client,
        ):
            result = call_generation_llm(
                "Test prompt",
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=100,
            )

        assert result is None


class TestParseJsonResponse:
    """Tests for parse_json_response function."""

    def test_lh_003_extracts_json_from_plain_text(self):
        """Should parse plain JSON string."""
        content = '{"cases": [{"id": "test"}]}'
        result = parse_json_response(content)

        assert result == {"cases": [{"id": "test"}]}

    def test_lh_004_handles_markdown_wrapped_json(self):
        """Should extract JSON from markdown code blocks."""
        content = '```json\n{"cases": [{"id": "test"}]}\n```'
        result = parse_json_response(content)

        assert result == {"cases": [{"id": "test"}]}

    def test_lh_004b_handles_plain_markdown_wrapper(self):
        """Should extract JSON from plain ``` code blocks."""
        content = '```\n{"key": "value"}\n```'
        result = parse_json_response(content)

        assert result == {"key": "value"}

    def test_lh_005_returns_none_on_invalid_json(self):
        """Invalid JSON should return None."""
        result = parse_json_response("not json at all")
        assert result is None

    def test_lh_005b_returns_none_on_empty_content(self):
        """Empty content should return None."""
        result = parse_json_response("")
        assert result is None

    def test_lh_005c_returns_none_on_none(self):
        """None content should return None."""
        result = parse_json_response(None)
        assert result is None


class TestLoadArticleContent:
    """Tests for load_article_content function."""

    def test_lh_006_returns_formatted_string_for_existing_corpus(self, tmp_path):
        """Should return formatted article content from chunks JSONL."""
        # Create test chunks file
        chunks_dir = tmp_path / "data" / "processed"
        chunks_dir.mkdir(parents=True)
        chunks_file = chunks_dir / "test-corpus_chunks.jsonl"

        chunks = [
            {
                "text": "Article 1 content that is long enough to pass the 50 char filter for test purposes.",
                "metadata": {"article": "1", "article_title": "Definitions"},
            },
            {
                "text": "Article 2 content that is also long enough to pass the 50 character filter check.",
                "metadata": {"article": "2", "article_title": "Scope"},
            },
        ]
        with open(chunks_file, "w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")

        with patch(
            "src.common.llm_helpers.PROJECT_ROOT", tmp_path,
        ):
            result = load_article_content("test-corpus")

        assert "Artikel 1" in result
        assert "Definitions" in result
        assert "Artikel 2" in result
        assert "Scope" in result

    def test_lh_007_returns_placeholder_for_missing_corpus(self, tmp_path):
        """Should return placeholder when chunks file doesn't exist."""
        with patch(
            "src.common.llm_helpers.PROJECT_ROOT", tmp_path,
        ):
            result = load_article_content("nonexistent-corpus")

        assert "ikke tilgængelig" in result.lower() or "not available" in result.lower()
