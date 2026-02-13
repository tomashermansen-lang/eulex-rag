"""Tests for src/engine/llm_client.py - LLM API communication layer.

Covers:
- Client creation with timeout/retry config
- Model capability detection (reasoning, no-temperature)
- Rate limit retry logic with exponential backoff
- Streaming functionality
- Error handling and client cleanup
- Singleton client management (connection pooling)
- Async LLM client
"""

import asyncio
import re
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.engine.llm_client import (
    make_openai_client,
    call_llm,
    call_llm_stream,
    get_sync_client,
    get_async_client,
    reset_clients,
    call_llm_async,
    call_llm_stream_async,
    _get_model_capabilities,
    _model_uses_reasoning,
    _model_skips_temperature,
)
from src.engine.types import RAGEngineError


# ─────────────────────────────────────────────────────────────────────────────
# Model Capability Detection
# ─────────────────────────────────────────────────────────────────────────────


class TestGetModelCapabilities:
    """Tests for _get_model_capabilities function."""

    def test_returns_defaults_when_no_config(self):
        """Empty settings returns empty lists and default effort."""
        reasoning, no_temp, effort = _get_model_capabilities({})
        assert reasoning == []
        assert no_temp == []
        assert effort == "low"

    def test_parses_reasoning_models_from_config(self):
        """Extracts reasoning_models list from config."""
        settings = {
            "model_capabilities": {
                "reasoning_models": ["o1", "o3"],
            }
        }
        reasoning, no_temp, effort = _get_model_capabilities(settings)
        assert reasoning == ["o1", "o3"]
        assert no_temp == []

    def test_parses_no_temperature_models_from_config(self):
        """Extracts no_temperature_models list from config."""
        settings = {
            "model_capabilities": {
                "no_temperature_models": ["o1", "o1-mini"],
            }
        }
        reasoning, no_temp, effort = _get_model_capabilities(settings)
        assert no_temp == ["o1", "o1-mini"]

    def test_parses_reasoning_effort_from_config(self):
        """Extracts reasoning_effort setting."""
        settings = {
            "model_capabilities": {
                "reasoning_effort": "high",
            }
        }
        _, _, effort = _get_model_capabilities(settings)
        assert effort == "high"

    def test_handles_none_values_gracefully(self):
        """None values in config are handled as empty lists."""
        settings = {
            "model_capabilities": {
                "reasoning_models": None,
                "no_temperature_models": None,
                "reasoning_effort": None,
            }
        }
        reasoning, no_temp, effort = _get_model_capabilities(settings)
        assert reasoning == []
        assert no_temp == []
        assert effort == "low"


class TestModelUsesReasoning:
    """Tests for _model_uses_reasoning function."""

    def test_returns_false_for_empty_model(self):
        """Empty model string returns False."""
        assert _model_uses_reasoning("", ["o1"]) is False
        assert _model_uses_reasoning(None, ["o1"]) is False

    def test_returns_false_for_empty_list(self):
        """Empty reasoning models list returns False."""
        assert _model_uses_reasoning("o1-preview", []) is False
        assert _model_uses_reasoning("o1-preview", None) is False

    def test_matches_prefix(self):
        """Model matching a prefix returns True."""
        reasoning_models = ["o1", "o3"]
        assert _model_uses_reasoning("o1-preview", reasoning_models) is True
        assert _model_uses_reasoning("o1-mini", reasoning_models) is True
        assert _model_uses_reasoning("o3-mini", reasoning_models) is True

    def test_non_matching_model_returns_false(self):
        """Model not matching any prefix returns False."""
        reasoning_models = ["o1", "o3"]
        assert _model_uses_reasoning("gpt-4o", reasoning_models) is False
        assert _model_uses_reasoning("gpt-4o-mini", reasoning_models) is False


class TestModelSkipsTemperature:
    """Tests for _model_skips_temperature function."""

    def test_returns_false_for_empty_model(self):
        """Empty model string returns False."""
        assert _model_skips_temperature("", ["o1"]) is False
        assert _model_skips_temperature(None, ["o1"]) is False

    def test_returns_false_for_empty_list(self):
        """Empty no-temp models list returns False."""
        assert _model_skips_temperature("o1-preview", []) is False
        assert _model_skips_temperature("o1-preview", None) is False

    def test_matches_prefix(self):
        """Model matching a prefix returns True."""
        no_temp_models = ["o1", "o3"]
        assert _model_skips_temperature("o1-preview", no_temp_models) is True
        assert _model_skips_temperature("o1-mini", no_temp_models) is True

    def test_non_matching_model_returns_false(self):
        """Model not matching any prefix returns False."""
        no_temp_models = ["o1"]
        assert _model_skips_temperature("gpt-4o", no_temp_models) is False


# ─────────────────────────────────────────────────────────────────────────────
# Client Creation
# ─────────────────────────────────────────────────────────────────────────────


class TestMakeOpenaiClient:
    """Tests for make_openai_client function."""

    def setup_method(self):
        reset_clients()

    def teardown_method(self):
        reset_clients()

    def test_creates_client_with_config_values(self, monkeypatch):
        """Client is created with timeout and retries from config."""
        mock_settings = {
            "openai": {
                "timeout_secs": 60,
                "max_retries": 5,
            }
        }

        mock_client = MagicMock()
        mock_openai_class = MagicMock(return_value=mock_client)

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr("src.engine.llm_client.OpenAI", mock_openai_class)

        result = make_openai_client()

        mock_openai_class.assert_called_once_with(timeout=60.0, max_retries=5)
        assert result == mock_client

    def test_env_vars_override_config(self, monkeypatch):
        """Environment variables override YAML config."""
        mock_settings = {
            "openai": {
                "timeout_secs": 60,
                "max_retries": 3,
            }
        }

        mock_client = MagicMock()
        mock_openai_class = MagicMock(return_value=mock_client)

        monkeypatch.setenv("RAG_OPENAI_TIMEOUT_SECS", "30")
        monkeypatch.setenv("RAG_OPENAI_MAX_RETRIES", "10")
        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr("src.engine.llm_client.OpenAI", mock_openai_class)

        make_openai_client()

        mock_openai_class.assert_called_once_with(timeout=30.0, max_retries=10)

    def test_falls_back_to_defaults(self, monkeypatch):
        """Uses default values when config is empty."""
        mock_client = MagicMock()
        mock_openai_class = MagicMock(return_value=mock_client)

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: {})
        monkeypatch.setattr("src.engine.llm_client.OpenAI", mock_openai_class)

        make_openai_client()

        mock_openai_class.assert_called_once_with(timeout=120.0, max_retries=3)

    def test_falls_back_on_typeerror(self, monkeypatch):
        """Falls back to simple OpenAI() if kwargs not accepted."""
        mock_client = MagicMock()

        def mock_openai_class(*args, **kwargs):
            if kwargs:
                raise TypeError("unexpected keyword argument")
            return mock_client

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: {})
        monkeypatch.setattr("src.engine.llm_client.OpenAI", mock_openai_class)

        result = make_openai_client()
        assert result == mock_client


# ─────────────────────────────────────────────────────────────────────────────
# call_llm Function
# ─────────────────────────────────────────────────────────────────────────────


class TestCallLlm:
    """Tests for call_llm function."""

    def _make_mock_response(self, content: str):
        """Create a mock OpenAI response."""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def test_calls_standard_model_with_temperature(self, monkeypatch):
        """Standard models are called with temperature parameter."""
        mock_settings = {
            "openai": {
                "chat_model": "gpt-4o-mini",
                "temperature": 0.7,
            },
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Hello world"
        )

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        result = call_llm("Test prompt")

        assert result == "Hello world"
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o-mini"
        assert call_args.kwargs["temperature"] == 0.7
        assert "reasoning" not in call_args.kwargs

    def test_calls_reasoning_model_with_reasoning_param(self, monkeypatch):
        """Reasoning models are called with reasoning parameter instead of temperature."""
        mock_settings = {
            "openai": {
                "chat_model": "o1-preview",
                "temperature": 0.7,
            },
            "model_capabilities": {
                "reasoning_models": ["o1"],
                "reasoning_effort": "medium",
            },
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Reasoning response"
        )

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        result = call_llm("Test prompt")

        assert result == "Reasoning response"
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "o1-preview"
        assert call_args.kwargs["reasoning"] == {"effort": "medium"}
        assert "temperature" not in call_args.kwargs

    def test_calls_no_temp_model_without_temperature(self, monkeypatch):
        """No-temperature models are called without temperature parameter."""
        mock_settings = {
            "openai": {
                "chat_model": "o1-mini",
                "temperature": 0.7,
            },
            "model_capabilities": {
                "no_temperature_models": ["o1"],
            },
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "No temp response"
        )

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        result = call_llm("Test prompt")

        assert result == "No temp response"
        call_args = mock_client.chat.completions.create.call_args
        assert "temperature" not in call_args.kwargs
        assert "reasoning" not in call_args.kwargs

    def test_explicit_model_overrides_config(self, monkeypatch):
        """Explicitly passed model overrides config default."""
        mock_settings = {
            "openai": {
                "chat_model": "gpt-4o-mini",
                "temperature": 0.5,
            },
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Custom model"
        )

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        call_llm("Test", model="gpt-4-turbo")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4-turbo"

    def test_explicit_temperature_overrides_config(self, monkeypatch):
        """Explicitly passed temperature overrides config default."""
        mock_settings = {
            "openai": {
                "chat_model": "gpt-4o-mini",
                "temperature": 0.5,
            },
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Custom temp"
        )

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        call_llm("Test", temperature=0.9)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.9

    def test_env_var_overrides_model(self, monkeypatch):
        """OPENAI_CHAT_MODEL env var overrides config."""
        mock_settings = {
            "openai": {
                "chat_model": "gpt-4o-mini",
                "temperature": 0.5,
            },
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Env model"
        )

        monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4-turbo")
        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        call_llm("Test")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4-turbo"

    def test_does_not_close_client_per_call(self, monkeypatch):
        """Singleton client is NOT closed after each call (connection reuse)."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "Success"
        )

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        call_llm("Test")

        mock_client.close.assert_not_called()

    def test_does_not_close_client_after_error(self, monkeypatch):
        """Singleton client is NOT closed after error (connection reuse)."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        with pytest.raises(RAGEngineError, match="OpenAI request failed"):
            call_llm("Test")

        mock_client.close.assert_not_called()

    def test_strips_whitespace_from_response(self, monkeypatch):
        """Response content is stripped of leading/trailing whitespace."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "  Hello world  \n"
        )

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        result = call_llm("Test")

        assert result == "Hello world"


class TestCallLlmRateLimitRetry:
    """Tests for rate limit retry logic in call_llm."""

    def _make_mock_response(self, content: str):
        """Create a mock OpenAI response."""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def test_retries_on_rate_limit_and_succeeds(self, monkeypatch):
        """Retries on RateLimitError and succeeds on retry."""
        from openai import RateLimitError

        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        call_count = [0]

        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RateLimitError(
                    message="Rate limit exceeded",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            return self._make_mock_response("Success after retry")

        mock_client.chat.completions.create.side_effect = mock_create

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )
        monkeypatch.setattr("src.engine.llm_client.time.sleep", lambda x: None)

        result = call_llm("Test")

        assert result == "Success after retry"
        assert call_count[0] == 3

    def test_extracts_wait_time_from_error_message(self, monkeypatch):
        """Extracts wait time from 'try again in Xs' error message."""
        from openai import RateLimitError

        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        call_count = [0]
        sleep_times = []

        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RateLimitError(
                    message="Rate limit exceeded, try again in 2.5s",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            return self._make_mock_response("Success")

        def mock_sleep(t):
            sleep_times.append(t)

        mock_client.chat.completions.create.side_effect = mock_create

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )
        monkeypatch.setattr("src.engine.llm_client.time.sleep", mock_sleep)

        call_llm("Test")

        assert len(sleep_times) == 1
        assert sleep_times[0] == 3.0  # 2.5 + 0.5 buffer

    def test_uses_exponential_backoff_when_no_wait_time(self, monkeypatch):
        """Uses exponential backoff when error doesn't contain wait time."""
        from openai import RateLimitError

        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        call_count = [0]
        sleep_times = []

        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] < 4:
                raise RateLimitError(
                    message="Rate limit exceeded",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            return self._make_mock_response("Success")

        def mock_sleep(t):
            sleep_times.append(t)

        mock_client.chat.completions.create.side_effect = mock_create

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )
        monkeypatch.setattr("src.engine.llm_client.time.sleep", mock_sleep)

        call_llm("Test")

        # Exponential backoff: 2^0, 2^1, 2^2
        assert sleep_times == [1, 2, 4]

    def test_raises_after_max_retries(self, monkeypatch):
        """Raises RAGEngineError after max retries exceeded."""
        from openai import RateLimitError

        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None,
        )

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )
        monkeypatch.setattr("src.engine.llm_client.time.sleep", lambda x: None)

        # The outer exception handler wraps the rate limit error
        with pytest.raises(RAGEngineError, match="OpenAI request failed"):
            call_llm("Test")


class TestCallLlmTypeerrorFallback:
    """Tests for TypeError fallback in call_llm."""

    def _make_mock_response(self, content: str):
        """Create a mock OpenAI response."""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def test_falls_back_on_typeerror(self, monkeypatch):
        """Falls back to basic call when reasoning/temperature causes TypeError."""
        mock_settings = {
            "openai": {"chat_model": "o1-preview", "temperature": 0.5},
            "model_capabilities": {"reasoning_models": ["o1"]},
        }

        mock_client = MagicMock()
        call_count = [0]

        def mock_create(**kwargs):
            call_count[0] += 1
            if "reasoning" in kwargs:
                raise TypeError("unexpected keyword argument 'reasoning'")
            return self._make_mock_response("Fallback response")

        mock_client.chat.completions.create.side_effect = mock_create

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        result = call_llm("Test")

        assert result == "Fallback response"
        assert call_count[0] == 2  # First with reasoning, then fallback


# ─────────────────────────────────────────────────────────────────────────────
# call_llm_stream Function
# ─────────────────────────────────────────────────────────────────────────────


class TestCallLlmStream:
    """Tests for call_llm_stream function."""

    def test_yields_chunks_from_stream(self, monkeypatch):
        """Yields text chunks as they arrive from stream."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="!"))]
            ),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        result = list(call_llm_stream("Test"))

        assert result == ["Hello ", "world", "!"]

    def test_skips_empty_chunks(self, monkeypatch):
        """Skips chunks with no content."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello"))]
            ),
            SimpleNamespace(choices=[]),  # Empty choices
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            ),  # None content
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))]
            ),
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter(chunks)

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        result = list(call_llm_stream("Test"))

        assert result == ["Hello", "world"]

    def test_streaming_uses_stream_param(self, monkeypatch):
        """Streaming calls use stream=True parameter."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        list(call_llm_stream("Test"))

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["stream"] is True

    def test_stream_with_reasoning_model(self, monkeypatch):
        """Streaming with reasoning model uses reasoning parameter."""
        mock_settings = {
            "openai": {"chat_model": "o1-preview", "temperature": 0.5},
            "model_capabilities": {
                "reasoning_models": ["o1"],
                "reasoning_effort": "high",
            },
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        list(call_llm_stream("Test"))

        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["reasoning"] == {"effort": "high"}
        assert call_args.kwargs["stream"] is True

    def test_stream_does_not_close_client(self, monkeypatch):
        """Singleton client is NOT closed after streaming (connection reuse)."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        list(call_llm_stream("Test"))

        mock_client.close.assert_not_called()

    def test_stream_falls_back_on_typeerror(self, monkeypatch):
        """Falls back to non-streaming on TypeError."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        call_count = [0]

        def mock_create(**kwargs):
            call_count[0] += 1
            if kwargs.get("stream"):
                raise TypeError("unexpected keyword argument 'stream'")
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Fallback"))]
            )

        mock_client.chat.completions.create.side_effect = mock_create

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )

        result = list(call_llm_stream("Test"))

        assert result == ["Fallback"]

    def test_stream_retries_on_rate_limit(self, monkeypatch):
        """Streaming retries on rate limit errors."""
        from openai import RateLimitError

        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        call_count = [0]

        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Success"))]
            ),
        ]

        def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RateLimitError(
                    message="Rate limit",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            return iter(chunks)

        mock_client.chat.completions.create.side_effect = mock_create

        monkeypatch.setattr(
            "src.common.config_loader.get_settings_yaml",
            lambda: mock_settings,
        )
        monkeypatch.setattr(
            "src.engine.llm_client.make_openai_client",
            lambda: mock_client,
        )
        monkeypatch.setattr("src.engine.llm_client.time.sleep", lambda x: None)

        result = list(call_llm_stream("Test"))

        assert result == ["Success"]
        assert call_count[0] == 2


# ─────────────────────────────────────────────────────────────────────────────
# Singleton Client Management
# ─────────────────────────────────────────────────────────────────────────────


class TestGetSyncClient:
    """Tests for get_sync_client() singleton. (LC-01, LC-03, LC-06)"""

    def setup_method(self):
        reset_clients()

    def teardown_method(self):
        reset_clients()

    def test_returns_singleton(self, monkeypatch):
        """LC-01: Two calls to get_sync_client() return the same object."""
        mock_client = MagicMock()
        mock_openai_class = MagicMock(return_value=mock_client)

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: {})
        monkeypatch.setattr("src.engine.llm_client.OpenAI", mock_openai_class)

        client1 = get_sync_client()
        client2 = get_sync_client()

        assert client1 is client2
        mock_openai_class.assert_called_once()

    def test_reset_clears_singleton(self, monkeypatch):
        """LC-03: After reset_clients(), next get_sync_client() creates new instance."""
        call_count = [0]

        def mock_openai_class(*args, **kwargs):
            call_count[0] += 1
            return MagicMock(name=f"client-{call_count[0]}")

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: {})
        monkeypatch.setattr("src.engine.llm_client.OpenAI", mock_openai_class)

        client1 = get_sync_client()
        reset_clients()
        client2 = get_sync_client()

        assert client1 is not client2
        assert call_count[0] == 2

    def test_make_openai_client_delegates_to_singleton(self, monkeypatch):
        """LC-06: make_openai_client() returns the singleton (backward compat)."""
        mock_client = MagicMock()
        mock_openai_class = MagicMock(return_value=mock_client)

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: {})
        monkeypatch.setattr("src.engine.llm_client.OpenAI", mock_openai_class)

        singleton = get_sync_client()
        compat = make_openai_client()

        assert compat is singleton


class TestGetAsyncClient:
    """Tests for get_async_client() singleton. (LC-02)"""

    def setup_method(self):
        reset_clients()

    def teardown_method(self):
        reset_clients()

    def test_returns_singleton(self, monkeypatch):
        """LC-02: Two calls to get_async_client() return the same object."""
        mock_client = MagicMock()
        mock_async_class = MagicMock(return_value=mock_client)

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: {})
        monkeypatch.setattr("src.engine.llm_client.AsyncOpenAI", mock_async_class)

        client1 = get_async_client()
        client2 = get_async_client()

        assert client1 is client2
        mock_async_class.assert_called_once()


class TestCallLlmUsesSingleton:
    """Tests that call_llm and call_llm_stream use the singleton. (LC-04, LC-05)"""

    def setup_method(self):
        reset_clients()

    def teardown_method(self):
        reset_clients()

    def _make_mock_response(self, content: str):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def test_call_llm_uses_singleton(self, monkeypatch):
        """LC-04: call_llm() does not create a new client per call."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = self._make_mock_response("ok")
        creation_count = [0]

        def mock_openai(*args, **kwargs):
            creation_count[0] += 1
            return mock_client

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: mock_settings)
        monkeypatch.setattr("src.engine.llm_client.OpenAI", mock_openai)

        call_llm("First call")
        call_llm("Second call")

        assert creation_count[0] == 1  # Only one client created

    def test_call_llm_stream_uses_singleton(self, monkeypatch):
        """LC-05: call_llm_stream() does not create a new client per call."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])
        creation_count = [0]

        def mock_openai(*args, **kwargs):
            creation_count[0] += 1
            return mock_client

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: mock_settings)
        monkeypatch.setattr("src.engine.llm_client.OpenAI", mock_openai)

        list(call_llm_stream("First call"))
        list(call_llm_stream("Second call"))

        assert creation_count[0] == 1  # Only one client created


class TestCallLlmAsync:
    """Tests for call_llm_async(). (LC-07, LC-08, LC-10)"""

    def setup_method(self):
        reset_clients()

    def teardown_method(self):
        reset_clients()

    def _make_mock_async_response(self, content: str):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def test_returns_response(self, monkeypatch):
        """LC-07: call_llm_async() returns string response from mock."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
            "performance": {"max_llm_concurrency": 5},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._make_mock_async_response("Async response")
        )

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: mock_settings)
        monkeypatch.setattr("src.engine.llm_client.AsyncOpenAI", MagicMock(return_value=mock_client))

        result = asyncio.run(call_llm_async("Test prompt"))

        assert result == "Async response"

    def test_uses_model_capabilities(self, monkeypatch):
        """LC-08: Async client respects reasoning/no-temp model config."""
        mock_settings = {
            "openai": {"chat_model": "o1-preview", "temperature": 0.7},
            "model_capabilities": {
                "reasoning_models": ["o1"],
                "reasoning_effort": "medium",
            },
            "performance": {"max_llm_concurrency": 5},
        }

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=self._make_mock_async_response("Reasoning async")
        )

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: mock_settings)
        monkeypatch.setattr("src.engine.llm_client.AsyncOpenAI", MagicMock(return_value=mock_client))

        result = asyncio.run(call_llm_async("Test prompt"))

        assert result == "Reasoning async"
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["reasoning"] == {"effort": "medium"}
        assert "temperature" not in call_args.kwargs

    def test_rate_limit_retry(self, monkeypatch):
        """LC-10: Async call retries on RateLimitError with backoff."""
        from openai import RateLimitError

        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
            "performance": {"max_llm_concurrency": 5},
        }

        mock_client = MagicMock()
        call_count = [0]

        async def mock_create(**kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                raise RateLimitError(
                    message="Rate limit exceeded",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            return self._make_mock_async_response("Success after retry")

        mock_client.chat.completions.create = mock_create

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: mock_settings)
        monkeypatch.setattr("src.engine.llm_client.AsyncOpenAI", MagicMock(return_value=mock_client))
        monkeypatch.setattr("src.engine.llm_client.asyncio.sleep", AsyncMock())

        result = asyncio.run(call_llm_async("Test"))

        assert result == "Success after retry"
        assert call_count[0] == 3


class TestCallLlmStreamAsync:
    """Tests for call_llm_stream_async(). (LC-09)"""

    def setup_method(self):
        reset_clients()

    def teardown_method(self):
        reset_clients()

    def test_yields_chunks(self, monkeypatch):
        """LC-09: call_llm_stream_async() yields chunks from mock."""
        mock_settings = {
            "openai": {"chat_model": "gpt-4o-mini", "temperature": 0.5},
            "model_capabilities": {},
            "performance": {"max_llm_concurrency": 5},
        }

        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="async"))]
            ),
        ]

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: mock_settings)
        monkeypatch.setattr("src.engine.llm_client.AsyncOpenAI", MagicMock(return_value=mock_client))

        async def collect():
            result = []
            async for chunk in call_llm_stream_async("Test"):
                result.append(chunk)
            return result

        result = asyncio.run(collect())

        assert result == ["Hello ", "async"]

    def test_stream_async_retries_streaming_without_reasoning_on_typeerror(self, monkeypatch):
        """call_llm_stream_async retries with stream=True (no reasoning) on TypeError."""
        mock_settings = {
            "openai": {"chat_model": "gpt-5", "temperature": 0.5},
            "model_capabilities": {
                "reasoning_models": ["gpt-5"],
            },
            "performance": {"max_llm_concurrency": 5},
        }

        chunks = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Streamed "))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="result"))]
            ),
        ]

        call_log: list[dict] = []

        async def mock_stream():
            for chunk in chunks:
                yield chunk

        async def mock_create(**kwargs):
            call_log.append(kwargs)
            if "reasoning" in kwargs:
                raise TypeError("unexpected keyword argument 'reasoning'")
            # Second call: stream=True without reasoning → return async iterator
            return mock_stream()

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: mock_settings)
        monkeypatch.setattr("src.engine.llm_client.AsyncOpenAI", MagicMock(return_value=mock_client))

        async def collect():
            result = []
            async for chunk in call_llm_stream_async("Test"):
                result.append(chunk)
            return result

        result = asyncio.run(collect())

        # Should have streamed chunks, not a single non-streaming response
        assert result == ["Streamed ", "result"]
        # First call had reasoning + stream, second had stream but no reasoning
        assert "reasoning" in call_log[0]
        assert call_log[0]["stream"] is True
        assert "reasoning" not in call_log[1]
        assert call_log[1]["stream"] is True

    def test_stream_async_falls_back_to_nonstreaming_on_double_typeerror(self, monkeypatch):
        """Falls back to non-streaming when streaming retry also raises TypeError."""
        mock_settings = {
            "openai": {"chat_model": "gpt-5", "temperature": 0.5},
            "model_capabilities": {
                "reasoning_models": ["gpt-5"],
            },
            "performance": {"max_llm_concurrency": 5},
        }

        call_log: list[dict] = []

        async def mock_create(**kwargs):
            call_log.append(kwargs)
            if kwargs.get("stream"):
                raise TypeError("stream not supported")
            # Non-streaming fallback
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Non-streamed"))]
            )

        mock_client = MagicMock()
        mock_client.chat.completions.create = mock_create

        monkeypatch.setattr("src.common.config_loader.get_settings_yaml", lambda: mock_settings)
        monkeypatch.setattr("src.engine.llm_client.AsyncOpenAI", MagicMock(return_value=mock_client))

        async def collect():
            result = []
            async for chunk in call_llm_stream_async("Test"):
                result.append(chunk)
            return result

        result = asyncio.run(collect())

        assert result == ["Non-streamed"]
        # Three calls: reasoning+stream, stream-only, non-streaming
        assert len(call_log) == 3


class TestSyncApiUnchanged:
    """Tests that existing sync API is unchanged. (LC-11)"""

    def test_call_llm_signature_unchanged(self):
        """LC-11: call_llm() signature and return type unchanged."""
        import inspect

        sig = inspect.signature(call_llm)
        params = list(sig.parameters.keys())
        assert params == ["prompt", "model", "temperature"]

    def test_call_llm_stream_signature_unchanged(self):
        """LC-11: call_llm_stream() signature and return type unchanged."""
        import inspect

        sig = inspect.signature(call_llm_stream)
        params = list(sig.parameters.keys())
        assert params == ["prompt", "model", "temperature"]
