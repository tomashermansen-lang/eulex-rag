"""Tests for performance configuration settings.

TDD: Tests written BEFORE implementation of get_performance_settings().
"""

import os
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# PC-01: Performance section exists with defaults
# ---------------------------------------------------------------------------


class TestPerformanceConfigDefaults:
    """Verify get_performance_settings() returns correct defaults."""

    def test_pc01_performance_section_exists_with_defaults(self):
        """R4.1: All 6 keys present with correct default values."""
        from src.common.config_loader import get_performance_settings

        settings = get_performance_settings()
        assert settings["max_retrieval_workers"] == 16
        assert settings["max_llm_concurrency"] == 5
        assert settings["retrieval_timeout_secs"] == 3.0
        assert settings["async_enabled"] is True
        assert settings["connection_pool_size"] == 100
        assert settings["keepalive_connections"] == 50

    def test_pc06_missing_performance_section_returns_defaults(self):
        """R4.1: If yaml has no performance key, return sensible defaults."""
        from src.common.config_loader import get_performance_settings

        with patch("src.common.config_loader.get_settings_yaml", return_value={}):
            settings = get_performance_settings()
        assert settings["max_retrieval_workers"] == 16
        assert settings["max_llm_concurrency"] == 5
        assert settings["retrieval_timeout_secs"] == 3.0
        assert settings["async_enabled"] is True
        assert settings["connection_pool_size"] == 100
        assert settings["keepalive_connections"] == 50


# ---------------------------------------------------------------------------
# PC-02 to PC-05: Environment variable overrides
# ---------------------------------------------------------------------------


class TestPerformanceConfigEnvOverrides:
    """Verify environment variables override yaml values."""

    def test_pc02_env_override_max_retrieval_workers(self):
        """R4.3: RAG_MAX_RETRIEVAL_WORKERS overrides yaml."""
        from src.common.config_loader import get_performance_settings

        with patch.dict(os.environ, {"RAG_MAX_RETRIEVAL_WORKERS": "8"}):
            settings = get_performance_settings()
        assert settings["max_retrieval_workers"] == 8

    def test_pc03_env_override_max_llm_concurrency(self):
        """R4.3: RAG_MAX_LLM_CONCURRENCY overrides yaml."""
        from src.common.config_loader import get_performance_settings

        with patch.dict(os.environ, {"RAG_MAX_LLM_CONCURRENCY": "3"}):
            settings = get_performance_settings()
        assert settings["max_llm_concurrency"] == 3

    def test_pc04_env_override_retrieval_timeout(self):
        """R4.3: RAG_RETRIEVAL_TIMEOUT_SECS overrides yaml."""
        from src.common.config_loader import get_performance_settings

        with patch.dict(os.environ, {"RAG_RETRIEVAL_TIMEOUT_SECS": "5.0"}):
            settings = get_performance_settings()
        assert settings["retrieval_timeout_secs"] == 5.0

    def test_pc05_env_override_async_enabled(self):
        """R4.3: RAG_ASYNC_ENABLED=false overrides yaml."""
        from src.common.config_loader import get_performance_settings

        with patch.dict(os.environ, {"RAG_ASYNC_ENABLED": "false"}):
            settings = get_performance_settings()
        assert settings["async_enabled"] is False
