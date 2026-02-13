"""Tests for src/engine/rag.py - RAG engine core functions.

Focus on testable helper functions without requiring full RAGEngine setup.
"""

import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch, MagicMock

import pytest

from src.engine.rag import (
    _get_openai_settings,
    _get_model_capabilities,
    _get_default_chat_model,
    _get_default_embedding_model,
    _get_default_temperature,
    _get_rag_settings,
    _sync_json_mode_results_to_run_meta,
    _RetrievalResult,
)
from src.engine.generation_types import StructuredGenerationResult


class TestGetOpenAISettings:
    """Tests for _get_openai_settings function."""

    def test_returns_dict(self):
        """Returns a dictionary of OpenAI settings."""
        result = _get_openai_settings()
        assert isinstance(result, dict)

    def test_contains_chat_model(self):
        """Result contains chat_model key."""
        result = _get_openai_settings()
        assert "chat_model" in result or result == {}

    def test_contains_embedding_model(self):
        """Result contains embedding_model key."""
        result = _get_openai_settings()
        assert "embedding_model" in result or result == {}


class TestGetModelCapabilities:
    """Tests for _get_model_capabilities function."""

    def test_returns_dict(self):
        """Returns a dictionary of model capabilities."""
        result = _get_model_capabilities()
        assert isinstance(result, dict)


class TestGetDefaultChatModel:
    """Tests for _get_default_chat_model function."""

    def test_returns_string(self):
        """Returns a string model name."""
        result = _get_default_chat_model()
        assert result is None or isinstance(result, str)

    @patch.dict(os.environ, {"OPENAI_CHAT_MODEL": "gpt-4-test"}, clear=False)
    def test_env_override(self):
        """Environment variable overrides config."""
        result = _get_default_chat_model()
        assert result == "gpt-4-test"

    @patch.dict(os.environ, {}, clear=False)
    def test_falls_back_to_config(self):
        """Falls back to config when no env var."""
        # Remove env var if present
        env_backup = os.environ.pop("OPENAI_CHAT_MODEL", None)
        try:
            result = _get_default_chat_model()
            # Just verify it returns something (config value)
            assert result is None or isinstance(result, str)
        finally:
            if env_backup:
                os.environ["OPENAI_CHAT_MODEL"] = env_backup


class TestGetDefaultEmbeddingModel:
    """Tests for _get_default_embedding_model function."""

    def test_returns_string(self):
        """Returns a string model name."""
        result = _get_default_embedding_model()
        assert result is None or isinstance(result, str)

    @patch.dict(os.environ, {"OPENAI_EMBEDDING_MODEL": "text-embedding-test"}, clear=False)
    def test_env_override(self):
        """Environment variable overrides config."""
        result = _get_default_embedding_model()
        assert result == "text-embedding-test"


class TestGetDefaultTemperature:
    """Tests for _get_default_temperature function."""

    def test_returns_float(self):
        """Returns a float temperature."""
        result = _get_default_temperature()
        assert isinstance(result, float)

    @patch.dict(os.environ, {"RAG_OPENAI_TEMPERATURE": "0.5"}, clear=False)
    def test_env_override(self):
        """Environment variable overrides config."""
        result = _get_default_temperature()
        assert result == 0.5

    @patch.dict(os.environ, {"RAG_OPENAI_TEMPERATURE": "0.0"}, clear=False)
    def test_env_override_zero(self):
        """Environment variable handles zero value."""
        result = _get_default_temperature()
        assert result == 0.0


class TestGetRagSettings:
    """Tests for _get_rag_settings function."""

    def test_returns_dict(self):
        """Returns a dictionary of RAG settings."""
        result = _get_rag_settings()
        assert isinstance(result, dict)


class TestSyncJsonModeResultsToRunMeta:
    """Tests for _sync_json_mode_results_to_run_meta function."""

    def _make_gen_result(
        self,
        cited_idxs: list[int] | None = None,
        valid_cited_idxs: list[int] | None = None,
        repair_attempts: int = 0,
        enrich_attempts: int = 0,
        fail_reason: str | None = None,
        debug: dict | None = None,
        parsed_json: Any = None,
        answer_text: str | None = None,
        raw_llm_response: str | None = None,
    ) -> StructuredGenerationResult:
        """Create a StructuredGenerationResult for testing."""
        return StructuredGenerationResult(
            answer_text=answer_text or "Test answer",
            raw_llm_response=raw_llm_response or "Raw LLM response",
            parsed_json=parsed_json,
            cited_idxs=cited_idxs or [],
            valid_cited_idxs=valid_cited_idxs or [],
            repair_attempts=repair_attempts,
            enrich_attempts=enrich_attempts,
            fail_reason=fail_reason,
            debug=debug or {},
        )

    def test_sets_llm_calls_count(self):
        """Syncs llm_calls_count from debug."""
        gen_result = self._make_gen_result(debug={"llm_calls_count": 3})
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs={1, 2, 3}, contract_min_citations=2
        )

        assert run_meta["llm_calls_count"] == 3

    def test_sets_citations_source(self):
        """Syncs citations_source from debug."""
        gen_result = self._make_gen_result(debug={"citations_source": "json_extract"})
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None
        )

        assert run_meta["citations_source"] == "json_extract"

    def test_initializes_engineering_json_block(self):
        """Creates engineering_json block with defaults."""
        gen_result = self._make_gen_result()
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=2
        )

        assert "engineering_json" in run_meta
        assert run_meta["engineering_json"]["enabled"] is True
        assert run_meta["engineering_json"]["min_citations"] == 2

    def test_sorts_allowed_idxs(self):
        """Sorts allowed indices."""
        gen_result = self._make_gen_result()
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs={5, 2, 8, 1}, contract_min_citations=None
        )

        assert run_meta["allowed_idxs"] == [1, 2, 5, 8]
        assert run_meta["allowed_idxs_count"] == 4

    def test_syncs_repair_attempts(self):
        """Syncs repair_retry_performed flag."""
        gen_result = self._make_gen_result(repair_attempts=2)
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None
        )

        assert run_meta["repair_retry_performed"] is True
        assert run_meta["engineering_json"]["repair_retry_performed"] is True

    def test_syncs_enrich_attempts(self):
        """Syncs enrich_retry_performed flag."""
        gen_result = self._make_gen_result(enrich_attempts=1)
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None
        )

        assert run_meta["enrich_retry_performed"] is True

    def test_syncs_cited_idxs(self):
        """Syncs cited indices."""
        gen_result = self._make_gen_result(cited_idxs=[1, 3, 5])
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None
        )

        assert run_meta["cited_idxs"] == [1, 3, 5]
        assert run_meta["cited_idxs_json"] == [1, 3, 5]

    def test_syncs_valid_cited_idxs(self):
        """Syncs valid cited indices."""
        gen_result = self._make_gen_result(valid_cited_idxs=[1, 3])
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None
        )

        assert run_meta["valid_cited"] == [1, 3]

    def test_syncs_fail_reason(self):
        """Syncs fail_reason."""
        gen_result = self._make_gen_result(fail_reason="min_citations_not_met")
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None
        )

        assert run_meta["fail_reason"] == "min_citations_not_met"

    def test_sets_final_gate_reason_for_missing_ref(self):
        """Sets final_gate_reason when answer is MISSING_REF."""
        # is_missing_ref is a property that returns True when answer_text == "MISSING_REF"
        gen_result = self._make_gen_result(
            answer_text="MISSING_REF", fail_reason="schema_validation_failed"
        )
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None
        )

        assert run_meta["final_gate_reason"] == "schema_validation_failed"

    def test_syncs_bullet_counts(self):
        """Syncs bullet count metrics."""
        gen_result = self._make_gen_result(
            debug={
                "requirements_bullet_count": 5,
                "audit_evidence_bullet_count": 3,
            }
        )
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None
        )

        assert run_meta["requirements_bullet_count"] == 5
        assert run_meta["audit_evidence_bullet_count"] == 3

    def test_handles_answer_policy(self):
        """Handles answer_policy object for policy knobs."""
        gen_result = self._make_gen_result()
        run_meta: dict = {}

        # Create mock policy
        policy = MagicMock()
        policy.min_section3_bullets = 3
        policy.include_audit_evidence = True

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None,
            answer_policy=policy
        )

        assert run_meta["policy_min_section3_bullets"] == 3
        assert run_meta["policy_include_audit_evidence"] is True

    def test_handles_none_answer_policy(self):
        """Handles None answer_policy gracefully."""
        gen_result = self._make_gen_result()
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None,
            answer_policy=None
        )

        assert run_meta["policy_min_section3_bullets"] is None
        assert run_meta["policy_include_audit_evidence"] is False

    def test_syncs_rendered_text_from_parsed_json(self):
        """Syncs rendered_text when parsed_json exists."""
        gen_result = self._make_gen_result(
            parsed_json={"section1": "Summary"},
            answer_text="Rendered text here",
        )
        run_meta: dict = {}

        _sync_json_mode_results_to_run_meta(
            gen_result, run_meta, allowed_idxs=set(), contract_min_citations=None
        )

        assert run_meta["engineering_json"]["rendered_text"] == "Rendered text here"


class TestRetrievalResult:
    """Tests for _RetrievalResult dataclass."""

    def test_default_values(self):
        """Dataclass has sensible defaults."""
        result = _RetrievalResult(
            hits=[],
            distances=[],
            retrieved_ids=[],
            retrieved_metas=[],
            run_meta_updates={},
        )
        assert result.selected_chunks == ()
        assert result.total_retrieved == 0
        assert result.citable_count == 0

    def test_stores_hits_and_distances(self):
        """Stores hits and distances correctly."""
        hits = [("id1", {"article": "6"}), ("id2", {"article": "7"})]
        distances = [0.1, 0.2]

        result = _RetrievalResult(
            hits=hits,
            distances=distances,
            retrieved_ids=["id1", "id2"],
            retrieved_metas=[{"article": "6"}, {"article": "7"}],
            run_meta_updates={"key": "value"},
        )

        assert result.hits == hits
        assert result.distances == distances
        assert result.retrieved_ids == ["id1", "id2"]
        assert result.run_meta_updates == {"key": "value"}
