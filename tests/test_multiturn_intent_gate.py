"""Tests for multi-turn intent gate feature.

Tests context-aware intent classification for multi-turn conversations.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.engine.conversation import (
    HistoryMessage,
    last_exchange,
    needs_context_augmentation,
    SHORT_QUERY_THRESHOLD,
)
from src.engine.intent_router import (
    _ROUTER_PROMPT,
    _ROUTER_PROMPT_WITH_CONTEXT,
    _INTENT_ROUTER_CACHE,
    _call_router_llm,
    disambiguate_intent,
)
from src.engine.policy import classify_question_intent_with_router
from src.engine.types import ClaimIntent


# ---------------------------------------------------------------------------
# C1: conversation.py — last_exchange()
# ---------------------------------------------------------------------------


class TestLastExchange:
    """Tests for last_exchange() helper."""

    def test_returns_empty_for_none(self):
        assert last_exchange(None) == []

    def test_returns_empty_for_empty_list(self):
        assert last_exchange([]) == []

    def test_returns_last_user_and_assistant(self):
        history = [
            HistoryMessage("user", "First question"),
            HistoryMessage("assistant", "First answer"),
            HistoryMessage("user", "Second question"),
            HistoryMessage("assistant", "Second answer"),
        ]
        result = last_exchange(history)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].content == "Second question"
        assert result[1].role == "assistant"
        assert result[1].content == "Second answer"

    def test_returns_only_last_user_when_no_assistant_follows(self):
        history = [
            HistoryMessage("user", "First question"),
            HistoryMessage("assistant", "First answer"),
            HistoryMessage("user", "Second question"),
        ]
        result = last_exchange(history)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].content == "First question"
        assert result[1].role == "assistant"
        assert result[1].content == "First answer"

    def test_returns_max_two_messages(self):
        history = [
            HistoryMessage("user", "Q1"),
            HistoryMessage("assistant", "A1"),
            HistoryMessage("user", "Q2"),
            HistoryMessage("assistant", "A2"),
            HistoryMessage("user", "Q3"),
            HistoryMessage("assistant", "A3"),
        ]
        result = last_exchange(history)
        assert len(result) == 2

    def test_handles_only_assistant_messages(self):
        history = [
            HistoryMessage("assistant", "Welcome message"),
        ]
        result = last_exchange(history)
        assert result == []


# ---------------------------------------------------------------------------
# C1: conversation.py — needs_context_augmentation()
# ---------------------------------------------------------------------------


class TestNeedsContextAugmentation:
    """Tests for needs_context_augmentation() helper."""

    def test_true_when_rewritten_equals_original(self):
        assert needs_context_augmentation("kort spørgsmål", "kort spørgsmål") is True

    def test_true_when_rewritten_is_short(self):
        short = "hvad med stk. 4?"
        assert len(short) < SHORT_QUERY_THRESHOLD
        assert needs_context_augmentation("original", short) is True

    def test_false_when_rewritten_differs_and_long(self):
        original = "hvad med stk. 4?"
        rewritten = "Hvad siger artikel 5, stk. 4 i AI-forordningen om forbudte praksisser?"
        assert len(rewritten) >= SHORT_QUERY_THRESHOLD
        assert needs_context_augmentation(original, rewritten) is False

    def test_false_when_original_is_none(self):
        """First turn: original_query is None, no augmentation needed."""
        assert needs_context_augmentation(None, "Any question here that is long enough to pass") is False


# ---------------------------------------------------------------------------
# C1: conversation.py — SHORT_QUERY_THRESHOLD
# ---------------------------------------------------------------------------


class TestShortQueryThreshold:
    """Tests for SHORT_QUERY_THRESHOLD constant."""

    def test_threshold_is_40(self):
        assert SHORT_QUERY_THRESHOLD == 40


# ---------------------------------------------------------------------------
# C2: intent_router.py — _ROUTER_PROMPT_WITH_CONTEXT
# ---------------------------------------------------------------------------


class TestRouterPromptWithContext:
    """Tests for _ROUTER_PROMPT_WITH_CONTEXT template."""

    def test_contains_question_and_context_placeholders(self):
        assert "{question}" in _ROUTER_PROMPT_WITH_CONTEXT
        assert "{context}" in _ROUTER_PROMPT_WITH_CONTEXT

    def test_produces_valid_prompt(self):
        result = _ROUTER_PROMPT_WITH_CONTEXT.format(
            question="Is my system prohibited?",
            context="User: Is my chatbot high-risk?\nAssistant: It depends on...",
        )
        assert "Is my system prohibited?" in result
        assert "chatbot high-risk" in result


# ---------------------------------------------------------------------------
# C2: intent_router.py — _call_router_llm(question, context)
# ---------------------------------------------------------------------------


class TestCallRouterLlmWithContext:
    """Tests for _call_router_llm with optional context parameter."""

    def setup_method(self):
        _INTENT_ROUTER_CACHE.clear()

    @patch("src.engine.intent_router.OpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_without_context_uses_original_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="LAW_CONTENT"))
        ]

        _call_router_llm("What is Article 6?")

        call_args = mock_client.chat.completions.create.call_args
        prompt_sent = call_args[1]["messages"][0]["content"]
        # Original prompt does NOT contain context section
        assert "CONVERSATION CONTEXT" not in prompt_sent

    @patch("src.engine.intent_router.OpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_with_context_uses_context_prompt(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="USER_SYSTEM"))
        ]

        _call_router_llm("og for mit system?", context="User: Is my chatbot high-risk?")

        call_args = mock_client.chat.completions.create.call_args
        prompt_sent = call_args[1]["messages"][0]["content"]
        assert "chatbot high-risk" in prompt_sent

    @patch("src.engine.intent_router.OpenAI")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    def test_cache_key_differs_with_and_without_context(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="LAW_CONTENT"))
        ]

        _call_router_llm("same question")
        result1_cache_size = len(_INTENT_ROUTER_CACHE)

        _call_router_llm("same question", context="some context")
        result2_cache_size = len(_INTENT_ROUTER_CACHE)

        # Should be 2 separate cache entries
        assert result2_cache_size == result1_cache_size + 1


# ---------------------------------------------------------------------------
# C2: intent_router.py — disambiguate_intent(..., last_exchange, query_was_rewritten)
# ---------------------------------------------------------------------------


class TestDisambiguateIntentWithContext:
    """Tests for disambiguate_intent with multi-turn context."""

    def setup_method(self):
        _INTENT_ROUTER_CACHE.clear()

    @patch("src.engine.intent_router._call_router_llm")
    def test_no_last_exchange_no_augmentation(self, mock_llm):
        mock_llm.return_value = "USER_SYSTEM"
        intent, debug = disambiguate_intent(
            "Is my system prohibited?",
            ClaimIntent.CLASSIFICATION,
            enable_router=True,
        )
        # Should be called without context
        mock_llm.assert_called_once_with("Is my system prohibited?", context=None)
        assert debug["context_augmented"] is False

    @patch("src.engine.intent_router._call_router_llm")
    def test_short_query_with_exchange_triggers_augmentation(self, mock_llm):
        mock_llm.return_value = "USER_SYSTEM"
        exchange = [
            HistoryMessage("user", "Er min chatbot højrisiko?"),
            HistoryMessage("assistant", "Det afhænger af systemets formål."),
        ]
        intent, debug = disambiguate_intent(
            "og GDPR?",  # Short query, unchanged by rewriter
            ClaimIntent.SCOPE,
            enable_router=True,
            last_exchange=exchange,
            query_was_rewritten=False,
        )
        # Should be called WITH context
        call_args = mock_llm.call_args
        assert call_args[1]["context"] is not None
        assert "højrisiko" in call_args[1]["context"]
        assert debug["context_augmented"] is True

    @patch("src.engine.intent_router._call_router_llm")
    def test_long_rewritten_query_no_augmentation(self, mock_llm):
        mock_llm.return_value = "LAW_CONTENT"
        exchange = [
            HistoryMessage("user", "Previous question"),
            HistoryMessage("assistant", "Previous answer"),
        ]
        long_query = "Hvad siger artikel 5, stk. 4 i AI-forordningen om forbudte praksisser?"
        intent, debug = disambiguate_intent(
            long_query,
            ClaimIntent.REQUIREMENTS,
            enable_router=True,
            last_exchange=exchange,
            query_was_rewritten=True,
        )
        # Long rewritten query = no augmentation
        mock_llm.assert_called_once_with(long_query, context=None)
        assert debug["context_augmented"] is False

    @patch("src.engine.intent_router._call_router_llm")
    def test_debug_contains_new_keys(self, mock_llm):
        mock_llm.return_value = "USER_SYSTEM"
        _, debug = disambiguate_intent(
            "test",
            ClaimIntent.SCOPE,
            enable_router=True,
        )
        assert "context_augmented" in debug
        assert "query_was_rewritten" in debug

    @patch("src.engine.intent_router._call_router_llm")
    def test_general_without_context_skips_router(self, mock_llm):
        """GENERAL without context (first turn) should not call router."""
        intent, debug = disambiguate_intent(
            "Hvad siger artikel 5?",
            ClaimIntent.GENERAL,
            enable_router=True,
        )
        mock_llm.assert_not_called()
        assert intent == ClaimIntent.GENERAL
        assert debug["context_augmented"] is False

    @patch("src.engine.intent_router._call_router_llm")
    def test_general_follow_up_with_user_system_context_overrides_to_classification(self, mock_llm):
        """GENERAL follow-up where router detects user_system should override to CLASSIFICATION."""
        mock_llm.return_value = "USER_SYSTEM"
        exchange = [
            HistoryMessage("user", "Er min chatbot et højrisiko-system?"),
            HistoryMessage("assistant", "Det afhænger af din chatbots formål."),
        ]
        intent, debug = disambiguate_intent(
            "og hvad med GDPR?",
            ClaimIntent.GENERAL,
            enable_router=True,
            last_exchange=exchange,
            query_was_rewritten=False,
        )
        mock_llm.assert_called_once()
        assert intent == ClaimIntent.CLASSIFICATION
        assert debug["context_augmented"] is True
        assert debug["override_applied"] is True
        assert debug["router_called"] is True

    @patch("src.engine.intent_router._call_router_llm")
    def test_general_follow_up_about_law_content_stays_general(self, mock_llm):
        """GENERAL follow-up where router detects law_content should stay GENERAL (no leakage)."""
        mock_llm.return_value = "LAW_CONTENT"
        exchange = [
            HistoryMessage("user", "Er min chatbot et højrisiko-system?"),
            HistoryMessage("assistant", "Det afhænger af din chatbots formål."),
        ]
        intent, debug = disambiguate_intent(
            "Hvad definerer loven som AI?",
            ClaimIntent.GENERAL,
            enable_router=True,
            last_exchange=exchange,
            query_was_rewritten=False,
        )
        mock_llm.assert_called_once()
        assert intent == ClaimIntent.GENERAL
        assert debug["override_applied"] is False


# ---------------------------------------------------------------------------
# C3: policy.py — classify_question_intent_with_router(last_exchange, query_was_rewritten)
# ---------------------------------------------------------------------------


class TestPolicyThreadsContext:
    """Tests for classify_question_intent_with_router passing context through."""

    def setup_method(self):
        _INTENT_ROUTER_CACHE.clear()

    @patch("src.engine.policy.disambiguate_intent")
    def test_passes_last_exchange_to_disambiguate(self, mock_disambiguate):
        mock_disambiguate.return_value = (ClaimIntent.SCOPE, {"context_augmented": True})
        exchange = [
            HistoryMessage("user", "Er min chatbot højrisiko?"),
            HistoryMessage("assistant", "Det afhænger af systemets formål."),
        ]
        classify_question_intent_with_router(
            "og GDPR?",
            enable_router=True,
            last_exchange=exchange,
            query_was_rewritten=False,
        )
        call_kwargs = mock_disambiguate.call_args[1]
        assert call_kwargs["last_exchange"] is exchange
        assert call_kwargs["query_was_rewritten"] is False

    @patch("src.engine.policy.disambiguate_intent")
    def test_passes_query_was_rewritten_true(self, mock_disambiguate):
        mock_disambiguate.return_value = (ClaimIntent.GENERAL, {"context_augmented": False})
        classify_question_intent_with_router(
            "long rewritten question about article 5 stk 4",
            enable_router=True,
            query_was_rewritten=True,
        )
        call_kwargs = mock_disambiguate.call_args[1]
        assert call_kwargs["query_was_rewritten"] is True
        assert call_kwargs["last_exchange"] is None

    @patch("src.engine.policy.disambiguate_intent")
    def test_defaults_to_no_context(self, mock_disambiguate):
        mock_disambiguate.return_value = (ClaimIntent.GENERAL, {"context_augmented": False})
        classify_question_intent_with_router(
            "What is Article 6?",
            enable_router=True,
        )
        call_kwargs = mock_disambiguate.call_args[1]
        assert call_kwargs["last_exchange"] is None
        assert call_kwargs["query_was_rewritten"] is False


# ---------------------------------------------------------------------------
# C4: planning.py — prepare_answer_context threads last_exchange/original_query
# ---------------------------------------------------------------------------


class TestPrepareAnswerContextThreadsContext:
    """Tests for prepare_answer_context passing context to classify_intent_fn."""

    def test_passes_last_exchange_and_original_query(self):
        """classify_intent_fn receives last_exchange and query_was_rewritten kwargs."""
        from src.engine.planning import prepare_answer_context, UserProfile

        exchange = [
            HistoryMessage("user", "Previous"),
            HistoryMessage("assistant", "Answer"),
        ]
        mock_classify = MagicMock(
            return_value=(ClaimIntent.GENERAL, {"context_augmented": False})
        )
        mock_policy_fn = MagicMock()
        mock_policy_fn.return_value = (
            ClaimIntent.GENERAL,
            {"policy_applied": False},
        )
        mock_effective_policy = MagicMock()
        mock_effective_policy.intent_keys_effective = []
        mock_effective_policy.contributors = {}
        mock_effective_policy.normative_guard = None
        mock_effective_policy.answer_policy = None

        prepare_answer_context(
            question="og GDPR?",
            corpus_id="test",
            resolved_profile=UserProfile.LEGAL,
            top_k=3,
            get_effective_policy_fn=MagicMock(return_value=mock_effective_policy),
            classify_intent_fn=mock_classify,
            apply_policy_to_intent_fn=mock_policy_fn,
            is_debug_corpus_fn=MagicMock(return_value=False),
            iso_utc_now_fn=MagicMock(return_value="2025-01-01T00:00:00Z"),
            git_commit_fn=MagicMock(return_value="abc123"),
            last_exchange=exchange,
            original_query="og GDPR?",
        )

        call_kwargs = mock_classify.call_args
        # The first positional arg is the question
        assert call_kwargs[0][0] == "og GDPR?"
        # Check keyword args passed through
        assert call_kwargs[1]["last_exchange"] is exchange
        assert call_kwargs[1]["query_was_rewritten"] is False

    def test_query_was_rewritten_true_when_queries_differ(self):
        """When original_query != question, query_was_rewritten=True."""
        from src.engine.planning import prepare_answer_context, UserProfile

        mock_classify = MagicMock(
            return_value=(ClaimIntent.GENERAL, {"context_augmented": False})
        )
        mock_policy_fn = MagicMock()
        mock_policy_fn.return_value = (
            ClaimIntent.GENERAL,
            {"policy_applied": False},
        )
        mock_effective_policy = MagicMock()
        mock_effective_policy.intent_keys_effective = []
        mock_effective_policy.contributors = {}
        mock_effective_policy.normative_guard = None
        mock_effective_policy.answer_policy = None

        prepare_answer_context(
            question="Hvad siger artikel 5, stk. 4 om forbudte praksisser?",
            corpus_id="test",
            resolved_profile=UserProfile.LEGAL,
            top_k=3,
            get_effective_policy_fn=MagicMock(return_value=mock_effective_policy),
            classify_intent_fn=mock_classify,
            apply_policy_to_intent_fn=mock_policy_fn,
            is_debug_corpus_fn=MagicMock(return_value=False),
            iso_utc_now_fn=MagicMock(return_value="2025-01-01T00:00:00Z"),
            git_commit_fn=MagicMock(return_value="abc123"),
            original_query="hvad med stk. 4?",
        )

        call_kwargs = mock_classify.call_args
        assert call_kwargs[1]["query_was_rewritten"] is True

    def test_no_context_when_no_original_query(self):
        """First turn: no original_query → no context kwargs."""
        from src.engine.planning import prepare_answer_context, UserProfile

        mock_classify = MagicMock(
            return_value=(ClaimIntent.GENERAL, {"context_augmented": False})
        )
        mock_policy_fn = MagicMock()
        mock_policy_fn.return_value = (
            ClaimIntent.GENERAL,
            {"policy_applied": False},
        )
        mock_effective_policy = MagicMock()
        mock_effective_policy.intent_keys_effective = []
        mock_effective_policy.contributors = {}
        mock_effective_policy.normative_guard = None
        mock_effective_policy.answer_policy = None

        prepare_answer_context(
            question="What is Article 6?",
            corpus_id="test",
            resolved_profile=UserProfile.LEGAL,
            top_k=3,
            get_effective_policy_fn=MagicMock(return_value=mock_effective_policy),
            classify_intent_fn=mock_classify,
            apply_policy_to_intent_fn=mock_policy_fn,
            is_debug_corpus_fn=MagicMock(return_value=False),
            iso_utc_now_fn=MagicMock(return_value="2025-01-01T00:00:00Z"),
            git_commit_fn=MagicMock(return_value="abc123"),
        )

        call_kwargs = mock_classify.call_args
        assert call_kwargs[1]["last_exchange"] is None
        assert call_kwargs[1]["query_was_rewritten"] is False


# ---------------------------------------------------------------------------
# C5: conversation.py — rewrite_query_for_retrieval preserves law name
# ---------------------------------------------------------------------------


class TestRewriterPreservesLawName:
    """Tests for query rewriter anchoring follow-ups to the correct corpus."""

    @patch("src.engine.llm_client.call_llm")
    def test_rewrite_prompt_contains_law_anchoring_rule(self, mock_llm):
        """Rewrite prompt must instruct LLM to preserve law name from history."""
        from src.engine.conversation import rewrite_query_for_retrieval

        mock_llm.return_value = "Hvad siger stk. 4 i artikel 5 i AI-forordningen?"
        history = [
            HistoryMessage("user", "Hvad siger artikel 5 i AI-forordningen?"),
            HistoryMessage("assistant", "AI-forordningen artikel 5 handler om forbudte praksisser."),
        ]
        rewrite_query_for_retrieval("hvad med stk 4?", history)

        prompt_sent = mock_llm.call_args[0][0]
        # The prompt must contain an explicit rule about preserving law names
        assert "lov" in prompt_sent.lower() or "forordning" in prompt_sent.lower()
        assert "SKAL" in prompt_sent or "skal" in prompt_sent
