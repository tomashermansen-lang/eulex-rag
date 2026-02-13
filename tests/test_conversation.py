"""Tests for conversation module.

Tests conversation context management for multi-turn RAG.
"""

from __future__ import annotations

import pytest

from src.engine.conversation import (
    HistoryMessage,
    truncate_history,
    format_history_for_prompt,
    rewrite_query_for_retrieval,
    _build_history_summary_for_rewrite,
)


class TestHistoryMessage:
    """Tests for HistoryMessage dataclass."""

    def test_creates_user_message(self):
        msg = HistoryMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_creates_assistant_message(self):
        msg = HistoryMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"

    def test_is_immutable(self):
        msg = HistoryMessage(role="user", content="Test")
        with pytest.raises(AttributeError):
            msg.content = "Changed"


class TestTruncateHistory:
    """Tests for truncate_history function."""

    def test_returns_empty_list_for_none(self):
        result = truncate_history(None)
        assert result == []

    def test_returns_empty_list_for_empty_input(self):
        result = truncate_history([])
        assert result == []

    def test_keeps_all_when_under_limit(self):
        history = [
            HistoryMessage("user", "Question 1"),
            HistoryMessage("assistant", "Answer 1"),
        ]
        result = truncate_history(history, max_messages=10)
        assert len(result) == 2
        assert result[0].content == "Question 1"

    def test_truncates_oldest_messages_when_over_limit(self):
        history = [
            HistoryMessage("user", f"Message {i}")
            for i in range(15)
        ]
        result = truncate_history(history, max_messages=10)
        assert len(result) == 10
        # Should keep the 10 most recent (messages 5-14)
        assert result[0].content == "Message 5"
        assert result[-1].content == "Message 14"

    def test_respects_custom_max_messages(self):
        history = [
            HistoryMessage("user", f"Msg {i}")
            for i in range(10)
        ]
        result = truncate_history(history, max_messages=3)
        assert len(result) == 3
        assert result[0].content == "Msg 7"


class TestFormatHistoryForPrompt:
    """Tests for format_history_for_prompt function."""

    def test_returns_empty_string_for_none(self):
        result = format_history_for_prompt(None)
        assert result == ""

    def test_returns_empty_string_for_empty_list(self):
        result = format_history_for_prompt([])
        assert result == ""

    def test_formats_single_user_message(self):
        history = [HistoryMessage("user", "Hvad er artikel 5?")]
        result = format_history_for_prompt(history)
        assert "TIDLIGERE SAMTALE:" in result
        assert "Bruger:" in result
        assert "Hvad er artikel 5?" in result

    def test_formats_single_exchange(self):
        history = [
            HistoryMessage("user", "Hvad er artikel 5?"),
            HistoryMessage("assistant", "Artikel 5 handler om forbudte AI-praksisser."),
        ]
        result = format_history_for_prompt(history)
        assert "TIDLIGERE SAMTALE:" in result
        assert "Bruger:" in result
        assert "Assistent:" in result
        assert "Hvad er artikel 5?" in result
        assert "forbudte AI-praksisser" in result

    def test_formats_multiple_exchanges(self):
        history = [
            HistoryMessage("user", "Første spørgsmål"),
            HistoryMessage("assistant", "Første svar"),
            HistoryMessage("user", "Andet spørgsmål"),
            HistoryMessage("assistant", "Andet svar"),
        ]
        result = format_history_for_prompt(history)
        assert result.count("Bruger:") == 2
        assert result.count("Assistent:") == 2

    def test_truncates_long_messages(self):
        long_content = "x" * 1000
        history = [HistoryMessage("user", long_content)]
        result = format_history_for_prompt(history, max_chars_per_message=100)
        # Should be truncated with ellipsis
        assert len(result) < 200
        assert "..." in result

    def test_preserves_short_messages(self):
        history = [HistoryMessage("user", "Short message")]
        result = format_history_for_prompt(history, max_chars_per_message=100)
        assert "Short message" in result
        assert "..." not in result

    def test_includes_end_marker(self):
        history = [HistoryMessage("user", "Test")]
        result = format_history_for_prompt(history)
        assert "---" in result  # End of history section marker


class TestBuildHistorySummaryForRewrite:
    """Tests for _build_history_summary_for_rewrite helper."""

    def test_returns_empty_for_empty_history(self):
        result = _build_history_summary_for_rewrite([])
        assert result == ""

    def test_formats_user_messages(self):
        history = [HistoryMessage("user", "Hvad er artikel 9?")]
        result = _build_history_summary_for_rewrite(history)
        assert "Bruger:" in result
        assert "artikel 9" in result

    def test_truncates_long_assistant_messages(self):
        history = [
            HistoryMessage("user", "Spørgsmål"),
            HistoryMessage("assistant", "x" * 1000),
        ]
        result = _build_history_summary_for_rewrite(history)
        # Assistant message should be truncated to 500 chars
        assert len(result) < 800

    def test_removes_kilder_section(self):
        history = [
            HistoryMessage("assistant", "Svaret er X. Kilder: [1], [2], [3]"),
        ]
        result = _build_history_summary_for_rewrite(history)
        assert "Kilder:" not in result
        assert "Svaret er X" in result

    def test_removes_referencer_section(self):
        history = [
            HistoryMessage("assistant", "Svaret er Y. Referencer: [1], [2]"),
        ]
        result = _build_history_summary_for_rewrite(history)
        assert "Referencer:" not in result

    def test_limits_to_last_6_messages(self):
        history = [
            HistoryMessage("user", f"Spørgsmål {i}")
            for i in range(10)
        ]
        result = _build_history_summary_for_rewrite(history)
        # Should only include last 6 messages (4-9)
        assert "Spørgsmål 4" in result
        assert "Spørgsmål 9" in result
        assert "Spørgsmål 3" not in result


class TestRewriteQueryForRetrieval:
    """Tests for rewrite_query_for_retrieval function."""

    def test_returns_original_when_no_history(self):
        result = rewrite_query_for_retrieval("Hvad er stk 4?", None)
        assert result == "Hvad er stk 4?"

    def test_returns_original_when_empty_history(self):
        result = rewrite_query_for_retrieval("Hvad er stk 4?", [])
        assert result == "Hvad er stk 4?"

    def test_returns_original_for_self_contained_question(self):
        # Question with specific article reference and substantial length
        question = "Hvad siger artikel 9 i GDPR om behandling af særlige kategorier af personoplysninger?"
        history = [HistoryMessage("user", "Tidligere spørgsmål")]
        result = rewrite_query_for_retrieval(question, history)
        assert result == question

    def test_returns_original_for_bilag_reference(self):
        question = "Hvad indeholder bilag 1 om høj-risiko AI-systemer? Giv mig alle detaljer."
        history = [HistoryMessage("user", "Noget")]
        result = rewrite_query_for_retrieval(question, history)
        assert result == question
