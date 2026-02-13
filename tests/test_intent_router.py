"""Tests for src/engine/intent_router.py - Intent routing logic."""

import time
import pytest
from unittest.mock import patch, MagicMock

from src.engine.intent_router import (
    _get_cache_key,
    _get_cached_result,
    _cache_result,
    disambiguate_intent,
    clear_cache,
    _INTENT_ROUTER_CACHE,
)
from src.engine.types import ClaimIntent


class TestCacheKey:
    """Tests for _get_cache_key function."""

    def test_same_question_same_key(self):
        """Same question produces same cache key."""
        key1 = _get_cache_key("What is Article 6?")
        key2 = _get_cache_key("What is Article 6?")
        assert key1 == key2

    def test_case_insensitive(self):
        """Cache key is case-insensitive."""
        key1 = _get_cache_key("What is Article 6?")
        key2 = _get_cache_key("WHAT IS ARTICLE 6?")
        assert key1 == key2

    def test_strips_whitespace(self):
        """Cache key strips leading/trailing whitespace."""
        key1 = _get_cache_key("test question")
        key2 = _get_cache_key("  test question  ")
        assert key1 == key2


class TestCachedResult:
    """Tests for cache retrieval functions."""

    def setup_method(self):
        """Clear cache before each test."""
        _INTENT_ROUTER_CACHE.clear()

    def test_returns_none_when_not_cached(self):
        """Returns None when question not in cache."""
        result = _get_cached_result("uncached question")
        assert result is None

    def test_returns_cached_value(self):
        """Returns cached value when present."""
        _cache_result("test question", "LAW_CONTENT")
        result = _get_cached_result("test question")
        assert result == "LAW_CONTENT"

    def test_cache_is_case_insensitive(self):
        """Cache lookup is case-insensitive."""
        _cache_result("test question", "USER_SYSTEM")
        result = _get_cached_result("TEST QUESTION")
        assert result == "USER_SYSTEM"


class TestClearCache:
    """Tests for clear_cache function."""

    def setup_method(self):
        """Clear cache before each test."""
        _INTENT_ROUTER_CACHE.clear()

    def test_returns_count_cleared(self):
        """Returns count of entries cleared."""
        _cache_result("q1", "LAW_CONTENT")
        _cache_result("q2", "USER_SYSTEM")
        count = clear_cache()
        assert count == 2

    def test_cache_empty_after_clear(self):
        """Cache is empty after clearing."""
        _cache_result("test", "LAW_CONTENT")
        clear_cache()
        assert _get_cached_result("test") is None


class TestDisambiguateIntent:
    """Tests for disambiguate_intent function."""

    def setup_method(self):
        """Clear cache before each test."""
        _INTENT_ROUTER_CACHE.clear()

    def test_general_intent_unchanged(self):
        """GENERAL intent is not routed."""
        intent, debug = disambiguate_intent(
            "What is the weather?",
            ClaimIntent.GENERAL,
            enable_router=True,
        )
        assert intent == ClaimIntent.GENERAL
        assert debug["router_called"] is False

    def test_non_gated_intents_not_routed(self):
        """Non-gated intents (GENERAL) are not routed."""
        # GENERAL is the only non-gated intent
        intent, debug = disambiguate_intent(
            "What is AI?",
            ClaimIntent.GENERAL,
            enable_router=True,
        )
        assert intent == ClaimIntent.GENERAL
        assert debug["router_called"] is False

    def test_router_disabled_returns_candidate(self):
        """When router disabled, returns candidate as-is."""
        intent, debug = disambiguate_intent(
            "Is my system prohibited?",
            ClaimIntent.CLASSIFICATION,
            enable_router=False,
        )
        assert intent == ClaimIntent.CLASSIFICATION
        assert debug["router_enabled"] is False
        assert debug["router_called"] is False

    @patch.dict("os.environ", {"INTENT_ROUTER_DISABLED": "true"})
    def test_env_disabled_returns_candidate(self):
        """When INTENT_ROUTER_DISABLED=true, returns candidate."""
        intent, debug = disambiguate_intent(
            "Is my system prohibited?",
            ClaimIntent.CLASSIFICATION,
            enable_router=True,
        )
        assert intent == ClaimIntent.CLASSIFICATION
        assert debug["router_enabled"] is False

    @patch("src.engine.intent_router._call_router_llm")
    def test_law_content_overrides_to_general(self, mock_llm):
        """LAW_CONTENT result overrides to GENERAL."""
        mock_llm.return_value = "LAW_CONTENT"
        intent, debug = disambiguate_intent(
            "What does Article 6 say?",
            ClaimIntent.CLASSIFICATION,
            enable_router=True,
        )
        assert intent == ClaimIntent.GENERAL
        assert debug["override_applied"] is True
        assert debug["router_result"] == "LAW_CONTENT"

    @patch("src.engine.intent_router._call_router_llm")
    def test_user_system_keeps_gated_intent(self, mock_llm):
        """USER_SYSTEM result keeps the gated intent."""
        mock_llm.return_value = "USER_SYSTEM"
        intent, debug = disambiguate_intent(
            "Is my system prohibited?",
            ClaimIntent.CLASSIFICATION,
            enable_router=True,
        )
        assert intent == ClaimIntent.CLASSIFICATION
        assert debug["override_applied"] is False
        assert debug["router_result"] == "USER_SYSTEM"

    @patch("src.engine.intent_router._call_router_llm")
    def test_routes_all_gated_intents(self, mock_llm):
        """All gated intents are routed."""
        mock_llm.return_value = "USER_SYSTEM"
        gated = [
            ClaimIntent.CLASSIFICATION,
            ClaimIntent.ENFORCEMENT,
            ClaimIntent.REQUIREMENTS,
            ClaimIntent.SCOPE,
        ]
        for candidate in gated:
            intent, debug = disambiguate_intent(
                "test question",
                candidate,
                enable_router=True,
            )
            assert debug["router_called"] is True, f"{candidate} should be routed"

    def test_debug_info_contains_expected_keys(self):
        """Debug info contains all expected keys."""
        _, debug = disambiguate_intent(
            "test",
            ClaimIntent.GENERAL,
            enable_router=True,
        )
        expected_keys = [
            "router_enabled",
            "candidate_intent",
            "router_called",
            "router_result",
            "override_applied",
            "final_intent",
        ]
        for key in expected_keys:
            assert key in debug, f"Missing key: {key}"
