"""Tests for metrics analysis engine module (C3).

Covers: _build_analysis_prompt(), analyse_metrics_stream().
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest

from src.engine.metrics_analysis import (
    _build_analysis_prompt,
    analyse_metrics_stream,
)


# ─────────────────────────────────────────────────────────────────────────────
# Sample metrics snapshot
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_SNAPSHOT = {
    "unified_pass_rate": 86.96,
    "health_status": "yellow",
    "single_law": {"total": 15, "passed": 13, "pass_rate": 86.67},
    "cross_law": {"total": 8, "passed": 7, "pass_rate": 87.5},
    "per_mode": [
        {"mode": "comparison", "pass_rate": 100.0, "total": 4, "passed": 4},
        {"mode": "discovery", "pass_rate": 50.0, "total": 2, "passed": 1},
    ],
    "per_difficulty": [
        {"difficulty": "easy", "pass_rate": 100.0, "total": 3, "passed": 3},
        {"difficulty": "hard", "pass_rate": 50.0, "total": 2, "passed": 1},
    ],
    "per_scorer": [
        {"scorer": "corpus_coverage", "pass_rate": 87.5, "total": 8, "passed": 7},
    ],
    "ingestion_coverage": 98.25,
    "percentiles": {"p50": 4.5, "p95": 10.2, "p99": 12.8},
}


# ─────────────────────────────────────────────────────────────────────────────
# T-E1 – T-E3: _build_analysis_prompt
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildAnalysisPrompt:
    """Prompt construction from metrics snapshot."""

    def test_contains_pass_rate(self):
        """T-E1: Prompt includes the unified pass rate."""
        prompt = _build_analysis_prompt(SAMPLE_SNAPSHOT)
        assert "86.96" in prompt

    def test_contains_mode_data(self):
        """T-E2: Prompt includes per-mode pass rates."""
        prompt = _build_analysis_prompt(SAMPLE_SNAPSHOT)
        assert "comparison" in prompt.lower()
        assert "discovery" in prompt.lower()
        assert "50.0" in prompt or "50%" in prompt

    def test_contains_difficulty_data(self):
        """T-E3: Prompt includes per-difficulty data."""
        prompt = _build_analysis_prompt(SAMPLE_SNAPSHOT)
        assert "easy" in prompt.lower()
        assert "hard" in prompt.lower()

    def test_requests_danish_output(self):
        """T-E4: Prompt instructs Danish-language output."""
        prompt = _build_analysis_prompt(SAMPLE_SNAPSHOT)
        assert "dansk" in prompt.lower() or "danish" in prompt.lower()

    def test_prompt_requests_structured_list_format(self):
        """Prompt instructs model to use numbered/structured lists for output."""
        prompt = _build_analysis_prompt(SAMPLE_SNAPSHOT)
        # Prompt should instruct the model to format points as a numbered list
        assert "nummereret liste" in prompt.lower() or "numbered list" in prompt.lower() or "brug nummerering" in prompt.lower()


# ─────────────────────────────────────────────────────────────────────────────
# T-E5 – T-E6: analyse_metrics_stream
# ─────────────────────────────────────────────────────────────────────────────


class TestAnalyseMetricsStream:
    """Streaming LLM analysis."""

    def test_yields_tokens(self):
        """T-E5: Stream yields text chunks from mock LLM."""

        async def mock_stream(*args, **kwargs) -> AsyncGenerator[str, None]:
            for chunk in ["Analyse: ", "systemet ", "er stabilt."]:
                yield chunk

        async def _run():
            with patch(
                "src.engine.metrics_analysis.call_llm_stream_async",
                side_effect=mock_stream,
            ):
                chunks = []
                async for chunk in analyse_metrics_stream(SAMPLE_SNAPSHOT):
                    chunks.append(chunk)
                return chunks

        chunks = asyncio.run(_run())
        assert len(chunks) == 3
        assert "".join(chunks) == "Analyse: systemet er stabilt."

    def test_uses_config_model(self):
        """T-E6: Uses model from config when not overridden."""
        called_with_model: list[str | None] = []

        async def mock_stream(prompt: str, model: str | None = None, **kwargs):
            called_with_model.append(model)
            yield "ok"

        async def _run():
            with patch(
                "src.engine.metrics_analysis.call_llm_stream_async",
                side_effect=mock_stream,
            ):
                async for _ in analyse_metrics_stream(SAMPLE_SNAPSHOT):
                    pass

        asyncio.run(_run())
        assert called_with_model[0] == "gpt-4o"
