"""Tests for src/engine/rag_config.py — config helpers extracted from rag.py.

Step 7.1: Verifies that config helpers are importable from their new location.
"""

from src.engine.rag_config import (
    _get_openai_settings,
    _get_model_capabilities,
    _get_default_chat_model,
    _get_default_embedding_model,
    _get_default_temperature,
    _get_rag_settings,
    _RetrievalResult,
)


class TestRagConfigImports:
    """Verify all config helpers are importable from rag_config."""

    def test_get_openai_settings_importable(self):
        assert callable(_get_openai_settings)

    def test_get_model_capabilities_importable(self):
        assert callable(_get_model_capabilities)

    def test_get_default_chat_model_importable(self):
        assert callable(_get_default_chat_model)

    def test_get_default_embedding_model_importable(self):
        assert callable(_get_default_embedding_model)

    def test_get_default_temperature_importable(self):
        assert callable(_get_default_temperature)

    def test_get_rag_settings_importable(self):
        assert callable(_get_rag_settings)

    def test_retrieval_result_is_dataclass(self):
        from dataclasses import fields

        field_names = [f.name for f in fields(_RetrievalResult)]
        assert "hits" in field_names
        assert "distances" in field_names
