"""Tests for pipeline_context module — PipelineContext frozen dataclass.

Step 6.1: Verifies freeze semantics, defaults, and construction.
"""

import pytest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock

from src.engine.pipeline_context import PipelineContext


# ---------------------------------------------------------------------------
# PipelineContext: frozen dataclass tests
# ---------------------------------------------------------------------------


def _make_context(**overrides) -> PipelineContext:
    """Helper: build a PipelineContext with sensible defaults."""
    defaults = {
        "corpus_id": "ai-act",
        "top_k": 3,
        "max_distance": None,
        "hard_max_distance": None,
        "chat_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-3-small",
        "enable_hybrid_rerank": True,
        "hybrid_vec_k": 30,
        "ranking_weights": SimpleNamespace(
            alpha_vec=0.4, beta_bm25=0.3, gamma_cite=0.2, delta_role=0.1
        ),
        "retriever": SimpleNamespace(),
        "ranker": SimpleNamespace(),
        "collection": SimpleNamespace(name="ai-act_documents"),
        "chroma": SimpleNamespace(),
        "project_root": Path("/tmp/test"),
        "resolver_fn": lambda: None,
        "collection_name": "ai-act_documents",
    }
    defaults.update(overrides)
    return PipelineContext(**defaults)


class TestPipelineContextFreeze:
    """PipelineContext must be immutable after creation."""

    def test_frozen_rejects_attribute_mutation(self):
        ctx = _make_context()
        with pytest.raises(FrozenInstanceError):
            ctx.corpus_id = "changed"

    def test_frozen_rejects_top_k_mutation(self):
        ctx = _make_context()
        with pytest.raises(FrozenInstanceError):
            ctx.top_k = 999

    def test_frozen_rejects_new_attribute(self):
        ctx = _make_context()
        with pytest.raises(FrozenInstanceError):
            ctx.new_field = "bad"


class TestPipelineContextConstruction:
    """PipelineContext stores all config, services, and computed fields."""

    def test_stores_config_fields(self):
        ctx = _make_context(corpus_id="dora", top_k=5, max_distance=1.5)
        assert ctx.corpus_id == "dora"
        assert ctx.top_k == 5
        assert ctx.max_distance == 1.5

    def test_stores_service_fields(self):
        retriever = SimpleNamespace(name="test-retriever")
        ranker = SimpleNamespace(name="test-ranker")
        ctx = _make_context(retriever=retriever, ranker=ranker)
        assert ctx.retriever.name == "test-retriever"
        assert ctx.ranker.name == "test-ranker"

    def test_stores_computed_fields(self):
        ctx = _make_context(collection_name="dora_documents")
        assert ctx.collection_name == "dora_documents"

    def test_resolver_fn_is_callable(self):
        resolver = MagicMock(return_value="resolved")
        ctx = _make_context(resolver_fn=resolver)
        result = ctx.resolver_fn()
        assert result == "resolved"
        resolver.assert_called_once()


# ---------------------------------------------------------------------------
# Step 6.2b: normalize_user_profile moved to planning.py
# ---------------------------------------------------------------------------

from src.engine.planning import normalize_user_profile
from src.engine.types import UserProfile


class TestNormalizeUserProfile:
    """normalize_user_profile converts strings/None to UserProfile enum."""

    def test_passthrough_enum(self):
        assert normalize_user_profile(UserProfile.LEGAL) is UserProfile.LEGAL

    def test_engineering_string(self):
        assert normalize_user_profile("ENGINEERING") is UserProfile.ENGINEERING

    def test_dev_alias(self):
        assert normalize_user_profile("DEV") is UserProfile.ENGINEERING

    def test_legal_string(self):
        assert normalize_user_profile("legal") is UserProfile.LEGAL

    def test_none_returns_any(self):
        assert normalize_user_profile(None) is UserProfile.ANY

    def test_empty_returns_any(self):
        assert normalize_user_profile("") is UserProfile.ANY
