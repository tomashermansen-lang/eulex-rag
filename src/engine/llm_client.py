"""LLM client for OpenAI API communication.

Single Responsibility: Handle all LLM API calls (sync, streaming, async).
No prompt construction, no JSON parsing, no business logic.

Clients are lazily initialized as singletons (connection pooling).
Use reset_clients() in test teardown to clear singletons.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time

logger = logging.getLogger(__name__)

from openai import AsyncOpenAI, OpenAI, RateLimitError

from .types import RAGEngineError


__all__ = [
    "make_openai_client",
    "get_sync_client",
    "get_async_client",
    "reset_clients",
    "call_llm",
    "call_llm_stream",
    "call_llm_async",
    "call_llm_stream_async",
]

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_sync_client: OpenAI | None = None
_async_client: AsyncOpenAI | None = None
_sync_lock = threading.Lock()
_async_lock = threading.Lock()


def _build_sync_client() -> OpenAI:
    """Create an OpenAI client with settings from config.

    Uses timeouts/retries to avoid 'silent stalls' on network issues.
    Falls back to OpenAI() when running under test fakes that don't accept kwargs.
    """
    from ..common.config_loader import get_settings_yaml

    settings = get_settings_yaml()
    openai_settings = settings.get("openai", {})

    timeout_secs = float(
        os.getenv("RAG_OPENAI_TIMEOUT_SECS")
        or openai_settings.get("timeout_secs", 120)
    )
    max_retries = int(
        os.getenv("RAG_OPENAI_MAX_RETRIES")
        or openai_settings.get("max_retries", 3)
    )

    try:
        return OpenAI(timeout=timeout_secs, max_retries=max_retries)
    except TypeError:
        return OpenAI()


def _build_async_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client with connection pooling from config."""
    from ..common.config_loader import get_settings_yaml

    settings = get_settings_yaml()
    openai_settings = settings.get("openai", {})
    perf = settings.get("performance", {})

    timeout_secs = float(
        os.getenv("RAG_OPENAI_TIMEOUT_SECS")
        or openai_settings.get("timeout_secs", 120)
    )
    max_retries = int(
        os.getenv("RAG_OPENAI_MAX_RETRIES")
        or openai_settings.get("max_retries", 3)
    )

    try:
        import httpx
        pool_size = int(perf.get("connection_pool_size", 100))
        keepalive = int(perf.get("keepalive_connections", 50))
        return AsyncOpenAI(
            timeout=timeout_secs,
            max_retries=max_retries,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=pool_size,
                    max_keepalive_connections=keepalive,
                ),
            ),
        )
    except (TypeError, ImportError):
        try:
            return AsyncOpenAI(timeout=timeout_secs, max_retries=max_retries)
        except TypeError:
            return AsyncOpenAI()


def get_sync_client() -> OpenAI:
    """Return the singleton sync OpenAI client (lazy, thread-safe)."""
    global _sync_client  # noqa: PLW0603
    if _sync_client is None:
        with _sync_lock:
            if _sync_client is None:
                _sync_client = _build_sync_client()
    return _sync_client


def get_async_client() -> AsyncOpenAI:
    """Return the singleton async OpenAI client (lazy, thread-safe)."""
    global _async_client  # noqa: PLW0603
    if _async_client is None:
        with _async_lock:
            if _async_client is None:
                _async_client = _build_async_client()
    return _async_client


def reset_clients() -> None:
    """Close and clear both singletons. Call in test teardown."""
    global _sync_client, _async_client  # noqa: PLW0603
    with _sync_lock:
        if _sync_client is not None:
            close_fn = getattr(_sync_client, "close", None)
            if callable(close_fn):
                close_fn()
            _sync_client = None
    with _async_lock:
        if _async_client is not None:
            close_fn = getattr(_async_client, "close", None)
            if callable(close_fn):
                close_fn()
            _async_client = None


def make_openai_client() -> OpenAI:
    """Backward-compatible: delegates to singleton get_sync_client()."""
    return get_sync_client()


def _get_model_capabilities(settings: dict) -> tuple[list[str], list[str], str]:
    """Get model capability lists from config/settings.yaml.

    Returns:
        Tuple of (reasoning_models, no_temperature_models, reasoning_effort)
    """
    caps = settings.get("model_capabilities", {})
    reasoning_models = caps.get("reasoning_models") or []
    no_temp_models = caps.get("no_temperature_models") or []
    reasoning_effort = caps.get("reasoning_effort") or "low"
    return reasoning_models, no_temp_models, reasoning_effort


def _model_uses_reasoning(model: str, reasoning_models: list[str]) -> bool:
    """Check if model uses reasoning parameter based on config."""
    if not model or not reasoning_models:
        return False
    return any(model.startswith(prefix) for prefix in reasoning_models)


def _model_skips_temperature(model: str, no_temp_models: list[str]) -> bool:
    """Check if model doesn't support temperature based on config."""
    if not model or not no_temp_models:
        return False
    return any(model.startswith(prefix) for prefix in no_temp_models)


def call_llm(prompt: str, model: str | None = None, temperature: float | None = None) -> str:
    """Call the LLM with appropriate parameters based on model capabilities.

    Model-specific behavior is driven by config/settings.yaml, not hardcoded.
    Uses the singleton sync client (no per-call creation overhead).
    """
    from ..common.config_loader import get_settings_yaml

    client = make_openai_client()
    try:
        settings = get_settings_yaml()
        openai_settings = settings.get("openai", {})

        # Get model and temperature from config (env can override)
        eff_model = model or os.getenv("OPENAI_CHAT_MODEL") or openai_settings.get("chat_model")
        eff_temp = temperature if temperature is not None else openai_settings.get("temperature")

        # Get model capabilities from config
        reasoning_models, no_temp_models, reasoning_effort = _get_model_capabilities(settings)

        # Determine how to call this model based on config
        uses_reasoning = _model_uses_reasoning(eff_model, reasoning_models)
        skips_temperature = _model_skips_temperature(eff_model, no_temp_models)

        # Rate limit retry settings
        max_retries = 5
        last_exc = None

        for attempt in range(max_retries):
            try:
                if uses_reasoning:
                    response = client.chat.completions.create(
                        model=eff_model,
                        messages=[{"role": "user", "content": prompt}],
                        reasoning={"effort": reasoning_effort},
                    )
                elif skips_temperature:
                    response = client.chat.completions.create(
                        model=eff_model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                else:
                    response = client.chat.completions.create(
                        model=eff_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=eff_temp,
                    )
                return response.choices[0].message.content.strip()
            except TypeError:
                response = client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content.strip()
            except RateLimitError as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait_match = re.search(r"try again in (\d+\.?\d*)s", str(exc))
                    wait_time = float(wait_match.group(1)) + 0.5 if wait_match else (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                raise RAGEngineError(f"OpenAI rate limit exceeded after {max_retries} retries.") from exc

        if last_exc:
            raise RAGEngineError("OpenAI request failed after retries.") from last_exc
        raise RAGEngineError("OpenAI request failed after retries.")
    except Exception as exc:  # noqa: BLE001
        raise RAGEngineError("OpenAI request failed.") from exc


def call_llm_stream(prompt: str, model: str | None = None, temperature: float | None = None):
    """Call the LLM with streaming enabled. Yields chunks of text as they arrive.

    Uses the singleton sync client (no per-call creation overhead).

    Yields:
        str: Text chunks as they are received from the LLM.
    """
    from ..common.config_loader import get_settings_yaml

    client = make_openai_client()
    try:
        settings = get_settings_yaml()
        openai_settings = settings.get("openai", {})

        eff_model = model or os.getenv("OPENAI_CHAT_MODEL") or openai_settings.get("chat_model")
        eff_temp = temperature if temperature is not None else openai_settings.get("temperature")

        reasoning_models, no_temp_models, reasoning_effort = _get_model_capabilities(settings)

        uses_reasoning = _model_uses_reasoning(eff_model, reasoning_models)
        skips_temperature = _model_skips_temperature(eff_model, no_temp_models)

        max_retries = 5
        last_exc = None

        for attempt in range(max_retries):
            try:
                if uses_reasoning:
                    stream = client.chat.completions.create(
                        model=eff_model,
                        messages=[{"role": "user", "content": prompt}],
                        reasoning={"effort": reasoning_effort},
                        stream=True,
                    )
                elif skips_temperature:
                    stream = client.chat.completions.create(
                        model=eff_model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                    )
                else:
                    stream = client.chat.completions.create(
                        model=eff_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=eff_temp,
                        stream=True,
                    )

                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return

            except TypeError:
                # Reasoning param may be unsupported; retry streaming without it
                try:
                    stream = client.chat.completions.create(
                        model=eff_model,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                    return
                except TypeError:
                    # Streaming itself unsupported; fall back to non-streaming
                    response = client.chat.completions.create(
                        model=eff_model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    yield response.choices[0].message.content.strip()
                    return

            except RateLimitError as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    wait_match = re.search(r"try again in (\d+\.?\d*)s", str(exc))
                    wait_time = float(wait_match.group(1)) + 0.5 if wait_match else (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                raise RAGEngineError(f"OpenAI rate limit exceeded after {max_retries} retries.") from exc

        if last_exc:
            raise RAGEngineError("OpenAI request failed after retries.") from last_exc
        raise RAGEngineError("OpenAI request failed after retries.")
    except Exception as exc:  # noqa: BLE001
        raise RAGEngineError("OpenAI request failed.") from exc


# ---------------------------------------------------------------------------
# Async API
# ---------------------------------------------------------------------------


def _get_llm_semaphore() -> asyncio.Semaphore:
    """Get the LLM concurrency semaphore (lazy, from config)."""
    from ..common.config_loader import get_performance_settings

    perf = get_performance_settings()
    return asyncio.Semaphore(perf["max_llm_concurrency"])


async def call_llm_async(
    prompt: str, model: str | None = None, temperature: float | None = None,
) -> str:
    """Async mirror of call_llm(). Uses singleton AsyncOpenAI client."""
    from ..common.config_loader import get_settings_yaml

    client = get_async_client()
    settings = get_settings_yaml()
    openai_settings = settings.get("openai", {})

    eff_model = model or os.getenv("OPENAI_CHAT_MODEL") or openai_settings.get("chat_model")
    eff_temp = temperature if temperature is not None else openai_settings.get("temperature")

    reasoning_models, no_temp_models, reasoning_effort = _get_model_capabilities(settings)
    uses_reasoning = _model_uses_reasoning(eff_model, reasoning_models)
    skips_temperature = _model_skips_temperature(eff_model, no_temp_models)

    max_retries = 5
    last_exc = None

    for attempt in range(max_retries):
        try:
            if uses_reasoning:
                response = await client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning={"effort": reasoning_effort},
                )
            elif skips_temperature:
                response = await client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                )
            else:
                response = await client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=eff_temp,
                )
            return response.choices[0].message.content.strip()
        except TypeError:
            response = await client.chat.completions.create(
                model=eff_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait_match = re.search(r"try again in (\\d+\\.?\\d*)s", str(exc))
                wait_time = float(wait_match.group(1)) + 0.5 if wait_match else (2 ** attempt)
                await asyncio.sleep(wait_time)
                continue
            raise RAGEngineError(
                f"OpenAI rate limit exceeded after {max_retries} retries."
            ) from exc

    if last_exc:
        raise RAGEngineError("OpenAI request failed after retries.") from last_exc
    raise RAGEngineError("OpenAI request failed after retries.")


async def call_llm_stream_async(
    prompt: str, model: str | None = None, temperature: float | None = None,
):
    """Async mirror of call_llm_stream(). Yields chunks via AsyncOpenAI client.

    Yields:
        str: Text chunks as they are received from the LLM.
    """
    from ..common.config_loader import get_settings_yaml

    client = get_async_client()
    settings = get_settings_yaml()
    openai_settings = settings.get("openai", {})

    eff_model = model or os.getenv("OPENAI_CHAT_MODEL") or openai_settings.get("chat_model")
    eff_temp = temperature if temperature is not None else openai_settings.get("temperature")

    reasoning_models, no_temp_models, reasoning_effort = _get_model_capabilities(settings)
    uses_reasoning = _model_uses_reasoning(eff_model, reasoning_models)
    skips_temperature = _model_skips_temperature(eff_model, no_temp_models)

    max_retries = 5
    last_exc = None

    for attempt in range(max_retries):
        try:
            if uses_reasoning:
                stream = await client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning={"effort": reasoning_effort},
                    stream=True,
                )
            elif skips_temperature:
                stream = await client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
            else:
                stream = await client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=eff_temp,
                    stream=True,
                )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            return

        except TypeError as te:
            logger.warning("Stream TypeError (attempt %d): %s — retrying without reasoning", attempt, te)
            # Reasoning param may be unsupported; retry streaming without it
            try:
                stream = await client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except TypeError:
                # Streaming itself unsupported; fall back to non-streaming
                response = await client.chat.completions.create(
                    model=eff_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                yield response.choices[0].message.content.strip()
                return

        except RateLimitError as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait_match = re.search(r"try again in (\\d+\\.?\\d*)s", str(exc))
                wait_time = float(wait_match.group(1)) + 0.5 if wait_match else (2 ** attempt)
                await asyncio.sleep(wait_time)
                continue
            raise RAGEngineError(
                f"OpenAI rate limit exceeded after {max_retries} retries."
            ) from exc

    if last_exc:
        raise RAGEngineError("OpenAI request failed after retries.") from last_exc
    raise RAGEngineError("OpenAI request failed after retries.")
