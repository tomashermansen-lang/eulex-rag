"""Tests for src/common/config_loader.py - Unified configuration loader.

Covers:
- Settings dataclass creation
- YAML loading and parsing
- Environment variable overrides
- Validation logic
- Corpora discovery and loading
- Concept configuration loading
- Schema validation with Pydantic
- Various getter functions
"""

import os
from pathlib import Path

import pytest
import yaml

from src.common.config_loader import (
    Settings,
    CorpusSettings,
    RankingWeights,
    load_settings,
    clear_config_cache,
    _slugify,
    _validate_settings,
    _load_settings_yaml,
    get_settings_yaml,
    validate_corpus_config,
    load_corpus_config,
    load_routing_config,
    get_concept_keywords,
    get_all_keywords_for_corpus,
    get_concept_bump_hints,
    get_default_bump_hints,
    get_concept_answer_policy,
    get_concept_normative_guard,
    get_concept_bias_rules,
    get_detection_settings,
    get_eval_settings,
    get_sibling_expansion_settings,
    ConceptSchema,
    CorpusConfigSchema,
    RescueRuleSchema,
)
import src.common.config_loader as config_loader


# ─────────────────────────────────────────────────────────────────────────────
# Slugify Function
# ─────────────────────────────────────────────────────────────────────────────


class TestSlugify:
    """Tests for _slugify function."""

    def test_lowercases_input(self):
        """Input is lowercased."""
        assert _slugify("AI-ACT") == "ai-act"
        assert _slugify("GDPR") == "gdpr"

    def test_replaces_spaces_with_hyphens(self):
        """Spaces are replaced with hyphens."""
        assert _slugify("AI Act") == "ai-act"
        assert _slugify("Data Protection") == "data-protection"

    def test_removes_special_characters(self):
        """Special characters are removed."""
        assert _slugify("AI Act (EU)") == "ai-act-eu"
        assert _slugify("Test@#$%File") == "testfile"

    def test_collapses_multiple_hyphens(self):
        """Multiple hyphens are collapsed to one."""
        assert _slugify("AI--Act") == "ai-act"
        assert _slugify("test---file") == "test-file"

    def test_strips_leading_trailing_hyphens(self):
        """Leading and trailing hyphens are stripped."""
        assert _slugify("-test-") == "test"
        assert _slugify("---test---") == "test"

    def test_returns_doc_for_empty_input(self):
        """Empty input returns 'doc'."""
        assert _slugify("") == "doc"
        assert _slugify("   ") == "doc"
        assert _slugify("@#$%") == "doc"


# ─────────────────────────────────────────────────────────────────────────────
# Settings Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


class TestRankingWeights:
    """Tests for RankingWeights dataclass."""

    def test_default_values(self):
        """Default values sum to 1.0."""
        weights = RankingWeights()
        total = weights.alpha_vec + weights.beta_bm25 + weights.gamma_cite + weights.delta_role
        assert total == 1.0

    def test_custom_values(self):
        """Custom values are accepted."""
        weights = RankingWeights(
            alpha_vec=0.4,
            beta_bm25=0.3,
            gamma_cite=0.2,
            delta_role=0.1,
        )
        assert weights.alpha_vec == 0.4
        assert weights.beta_bm25 == 0.3


class TestCorpusSettings:
    """Tests for CorpusSettings dataclass."""

    def test_required_fields(self):
        """Required fields are set correctly."""
        corpus = CorpusSettings(
            id="ai-act",
            display_name="AI Act",
            chunks_collection="ai-act_documents",
        )
        assert corpus.id == "ai-act"
        assert corpus.display_name == "AI Act"
        assert corpus.chunks_collection == "ai-act_documents"

    def test_optional_fields_default_to_none(self):
        """Optional fields default to None."""
        corpus = CorpusSettings(
            id="test",
            display_name="Test",
            chunks_collection="test_docs",
        )
        assert corpus.max_distance is None
        assert corpus.source_url is None


# ─────────────────────────────────────────────────────────────────────────────
# Settings Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestValidateSettings:
    """Tests for _validate_settings function."""

    def test_valid_settings_pass(self):
        """Valid settings pass validation."""
        settings = Settings(
            retrieval_pool_size=50,
            rag_max_distance=1.25,
            eurlex_chunk_tokens=500,
            eurlex_overlap=100,
        )
        _validate_settings(settings)  # Should not raise

    def test_invalid_retrieval_pool_size_raises(self):
        """Zero or negative retrieval_pool_size raises ValueError."""
        settings = Settings(retrieval_pool_size=0)
        with pytest.raises(ValueError, match="retrieval_pool_size must be >= 1"):
            _validate_settings(settings)

    def test_invalid_max_distance_raises(self):
        """Zero or negative max_distance raises ValueError."""
        settings = Settings(rag_max_distance=0)
        with pytest.raises(ValueError, match="max_distance must be > 0"):
            _validate_settings(settings)

        settings = Settings(rag_max_distance=-1.0)
        with pytest.raises(ValueError, match="max_distance must be > 0"):
            _validate_settings(settings)

    def test_none_max_distance_is_valid(self):
        """None max_distance is valid (no filtering)."""
        settings = Settings(rag_max_distance=None)
        _validate_settings(settings)  # Should not raise

    def test_overlap_must_be_less_than_chunk_tokens(self):
        """overlap must be less than chunk_tokens."""
        settings = Settings(eurlex_chunk_tokens=100, eurlex_overlap=100)
        with pytest.raises(ValueError, match="overlap must be < eurlex.chunk_tokens"):
            _validate_settings(settings)

        settings = Settings(eurlex_chunk_tokens=100, eurlex_overlap=150)
        with pytest.raises(ValueError, match="overlap must be < eurlex.chunk_tokens"):
            _validate_settings(settings)

    def test_ranking_weights_must_sum_to_one(self):
        """Ranking weights must sum to approximately 1.0."""
        weights = RankingWeights(
            alpha_vec=0.5,
            beta_bm25=0.5,
            gamma_cite=0.5,
            delta_role=0.5,
        )
        settings = Settings(ranking_weights=weights)
        with pytest.raises(ValueError, match="ranking_weights must sum to ~1.0"):
            _validate_settings(settings)


# ─────────────────────────────────────────────────────────────────────────────
# Settings Loading
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadSettingsYaml:
    """Tests for _load_settings_yaml function."""

    def test_returns_empty_dict_for_missing_file(self, monkeypatch, tmp_path):
        """Returns empty dict when settings.yaml doesn't exist."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        # No settings.yaml file

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)

        result = _load_settings_yaml()
        assert result == {}

    def test_loads_yaml_content(self, monkeypatch, tmp_path):
        """Loads and parses YAML content."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            """
openai:
  chat_model: gpt-4o-mini
  temperature: 0.7
""",
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)

        result = _load_settings_yaml()
        assert result["openai"]["chat_model"] == "gpt-4o-mini"
        assert result["openai"]["temperature"] == 0.7


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_loads_openai_settings(self, monkeypatch, tmp_path):
        """Loads OpenAI settings from YAML."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            """
openai:
  chat_model: gpt-4-turbo
  embedding_model: text-embedding-3-small
""",
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()

        settings = load_settings()
        assert settings.chat_model == "gpt-4-turbo"
        assert settings.embedding_model == "text-embedding-3-small"

    def test_env_vars_override_yaml(self, monkeypatch, tmp_path):
        """Environment variables override YAML values."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            """
openai:
  chat_model: gpt-4o-mini
rag:
  retrieval_pool_size: 50
  max_distance: 1.25
""",
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4-turbo")
        monkeypatch.setenv("RAG_RETRIEVAL_POOL_SIZE", "100")
        monkeypatch.setenv("RAG_MAX_DISTANCE", "1.5")
        clear_config_cache()

        settings = load_settings()
        assert settings.chat_model == "gpt-4-turbo"
        assert settings.retrieval_pool_size == 100
        assert settings.rag_max_distance == 1.5

    def test_loads_ranking_weights(self, monkeypatch, tmp_path):
        """Loads ranking weights from YAML."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            """
rag:
  ranking_weights:
    alpha_vec: 0.3
    beta_bm25: 0.3
    gamma_cite: 0.3
    delta_role: 0.1
""",
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()

        settings = load_settings()
        assert settings.ranking_weights.alpha_vec == 0.3
        assert settings.ranking_weights.beta_bm25 == 0.3
        assert settings.ranking_weights.gamma_cite == 0.3
        assert settings.ranking_weights.delta_role == 0.1

    def test_resolves_relative_paths(self, monkeypatch, tmp_path):
        """Resolves relative paths to absolute."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            """
paths:
  docs_path: data/docs
  vector_store_path: data/vectors
""",
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()

        settings = load_settings()
        assert settings.docs_path == tmp_path / "data/docs"
        assert settings.vector_store_path == tmp_path / "data/vectors"

    def test_caches_result(self, monkeypatch, tmp_path):
        """Result is cached after first load."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "openai:\n  chat_model: test\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()

        settings1 = load_settings()
        settings2 = load_settings()

        assert settings1 is settings2


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schema Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestRescueRuleSchema:
    """Tests for RescueRuleSchema."""

    def test_ensure_list_from_string(self):
        """Converts single string to list."""
        schema = RescueRuleSchema(if_present="article:6")
        assert schema.if_present == ["article:6"]

    def test_ensure_list_from_none(self):
        """Converts None to empty list."""
        schema = RescueRuleSchema(if_present=None)
        assert schema.if_present == []


class TestConceptSchema:
    """Tests for ConceptSchema."""

    def test_default_values(self):
        """Default values are empty lists."""
        schema = ConceptSchema()
        assert schema.keywords == []
        assert schema.toc_contains == []
        assert schema.bump_hints == []

    def test_ensure_list_from_string(self):
        """Converts single string to list."""
        schema = ConceptSchema(keywords="logging")
        assert schema.keywords == ["logging"]


class TestCorpusConfigSchema:
    """Tests for CorpusConfigSchema."""

    def test_default_values(self):
        """Default values are empty."""
        schema = CorpusConfigSchema()
        assert schema.concepts == {}
        assert schema.default == {}

    def test_parses_concepts(self):
        """Parses concepts dict."""
        schema = CorpusConfigSchema(
            concepts={
                "logging": {"keywords": ["log", "record"]},
            }
        )
        assert "logging" in schema.concepts
        assert schema.concepts["logging"].keywords == ["log", "record"]


class TestValidateCorpusConfig:
    """Tests for validate_corpus_config function."""

    def test_valid_config_returns_schema(self):
        """Valid config returns CorpusConfigSchema."""
        config = {
            "concepts": {
                "logging": {"keywords": ["log"]},
            },
            "default": {"bump_hints": ["article:1"]},
        }
        result = validate_corpus_config(config, "test")
        assert result is not None
        assert isinstance(result, CorpusConfigSchema)

    def test_invalid_config_returns_none(self):
        """Invalid config returns None."""
        # This should be invalid because concepts should be a dict
        config = {
            "concepts": "not a dict",
        }
        result = validate_corpus_config(config, "test")
        # Pydantic rejects invalid structure and returns None
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Corpus Config Loading
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadCorpusConfig:
    """Tests for load_corpus_config function."""

    def test_returns_empty_for_missing_file(self, monkeypatch, tmp_path):
        """Returns empty dict for missing config file."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        monkeypatch.setenv("CONCEPTS_DIR", str(concepts_dir))
        load_corpus_config.cache_clear()

        result = load_corpus_config("nonexistent")
        assert result == {}

    def test_skips_template_files(self, monkeypatch, tmp_path):
        """Skips files starting with underscore."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "_template.yaml").write_text(
            "concepts:\n  test:\n    keywords: [test]\n"
        )

        monkeypatch.setenv("CONCEPTS_DIR", str(concepts_dir))
        load_corpus_config.cache_clear()

        result = load_corpus_config("_template")
        assert result == {}

    def test_loads_valid_config(self, monkeypatch, tmp_path):
        """Loads valid config file."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "test.yaml").write_text(
            """
concepts:
  logging:
    keywords:
      - log
      - record
    bump_hints:
      - article:6
default:
  bump_hints:
    - article:1
""",
            encoding="utf-8",
        )

        monkeypatch.setenv("CONCEPTS_DIR", str(concepts_dir))
        load_corpus_config.cache_clear()

        result = load_corpus_config("test")
        assert "concepts" in result
        assert "logging" in result["concepts"]
        assert result["concepts"]["logging"]["keywords"] == ["log", "record"]


# ─────────────────────────────────────────────────────────────────────────────
# Getter Functions
# ─────────────────────────────────────────────────────────────────────────────


class TestGetConceptKeywords:
    """Tests for get_concept_keywords function."""

    def test_returns_keywords_for_concept(self, monkeypatch, tmp_path):
        """Returns keywords for existing concept."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "test.yaml").write_text(
            """
concepts:
  logging:
    keywords:
      - log
      - record
""",
            encoding="utf-8",
        )

        monkeypatch.setenv("CONCEPTS_DIR", str(concepts_dir))
        load_corpus_config.cache_clear()

        result = get_concept_keywords("test", "logging")
        assert result == ("log", "record")

    def test_returns_empty_for_missing_concept(self, monkeypatch, tmp_path):
        """Returns empty tuple for missing concept."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "test.yaml").write_text("concepts: {}\n")

        monkeypatch.setenv("CONCEPTS_DIR", str(concepts_dir))
        load_corpus_config.cache_clear()

        result = get_concept_keywords("test", "nonexistent")
        assert result == ()


class TestGetConceptBumpHints:
    """Tests for get_concept_bump_hints function."""

    def test_returns_bump_hints_for_concept(self, monkeypatch, tmp_path):
        """Returns bump hints for existing concept."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "test.yaml").write_text(
            """
concepts:
  logging:
    bump_hints:
      - article:6
      - article:7
""",
            encoding="utf-8",
        )

        monkeypatch.setenv("CONCEPTS_DIR", str(concepts_dir))
        load_corpus_config.cache_clear()

        result = get_concept_bump_hints("test", "logging")
        assert result == ["article:6", "article:7"]


class TestGetDefaultBumpHints:
    """Tests for get_default_bump_hints function."""

    def test_returns_default_bump_hints(self, monkeypatch, tmp_path):
        """Returns default bump hints for corpus."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "test.yaml").write_text(
            """
default:
  bump_hints:
    - article:1
    - article:2
""",
            encoding="utf-8",
        )

        monkeypatch.setenv("CONCEPTS_DIR", str(concepts_dir))
        load_corpus_config.cache_clear()

        result = get_default_bump_hints("test")
        assert result == ["article:1", "article:2"]


class TestGetSiblingExpansionSettings:
    """Tests for get_sibling_expansion_settings function."""

    def test_returns_settings_from_yaml(self, monkeypatch, tmp_path):
        """Returns sibling expansion settings from YAML."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            """
rag:
  sibling_expansion:
    enabled: true
    max_siblings: 3
""",
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)

        result = get_sibling_expansion_settings()
        assert result["enabled"] is True
        assert result["max_siblings"] == 3

    def test_returns_defaults_when_not_configured(self, monkeypatch, tmp_path):
        """Returns defaults when not configured."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text("rag: {}\n")

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)

        result = get_sibling_expansion_settings()
        assert result["enabled"] is False
        assert result["max_siblings"] == 2


class TestGetEvalSettings:
    """Tests for get_eval_settings function."""

    def test_returns_eval_settings(self, monkeypatch, tmp_path):
        """Returns eval settings from YAML."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            """
eval:
  primary_model: gpt-4o-mini
  fallback_model: gpt-4-turbo
""",
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)

        result = get_eval_settings()
        assert result["primary_model"] == "gpt-4o-mini"
        assert result["fallback_model"] == "gpt-4-turbo"


class TestClearConfigCache:
    """Tests for clear_config_cache function."""

    def test_clears_all_caches(self, monkeypatch, tmp_path):
        """Clears all configuration caches."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "openai:\n  chat_model: initial\n"
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)

        # Load initial settings
        clear_config_cache()
        settings1 = load_settings()
        assert settings1.chat_model == "initial"

        # Update file
        (config_dir / "settings.yaml").write_text(
            "openai:\n  chat_model: updated\n"
        )

        # Without clearing, should still return cached
        settings2 = load_settings()
        assert settings2.chat_model == "initial"

        # After clearing, should return updated
        clear_config_cache()
        settings3 = load_settings()
        assert settings3.chat_model == "updated"


class TestGetDiscoverySettings:
    """Tests for get_discovery_settings function."""

    def test_returns_discovery_settings(self, monkeypatch, tmp_path):
        """Returns discovery settings from YAML."""
        from src.common.config_loader import get_discovery_settings

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            """
discovery:
  enabled: true
  probe_top_k: 15
  auto_threshold: 0.80
  suggest_threshold: 0.55
  ambiguity_margin: 0.12
  max_corpora: 3
  llm_disambiguation: false
  scoring_weights:
    w_similarity: 0.60
    w_best: 0.40
""",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)

        result = get_discovery_settings()
        assert result["enabled"] is True
        assert result["probe_top_k"] == 15
        assert result["auto_threshold"] == 0.80
        assert result["suggest_threshold"] == 0.55
        assert result["ambiguity_margin"] == 0.12
        assert result["max_corpora"] == 3
        assert result["llm_disambiguation"] is False
        assert result["scoring_weights"]["w_similarity"] == 0.60
        assert result["scoring_weights"]["w_best"] == 0.40

    def test_returns_defaults_when_not_configured(self, monkeypatch, tmp_path):
        """Returns empty dict when discovery section is absent."""
        from src.common.config_loader import get_discovery_settings

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text("rag: {}\n")
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)

        result = get_discovery_settings()
        assert result == {}

    def test_partial_config(self, monkeypatch, tmp_path):
        """Returns partial discovery settings when only some fields present."""
        from src.common.config_loader import get_discovery_settings

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            """
discovery:
  enabled: false
  probe_top_k: 5
""",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)

        result = get_discovery_settings()
        assert result["enabled"] is False
        assert result["probe_top_k"] == 5
        assert "auto_threshold" not in result


# ─────────────────────────────────────────────────────────────────────────────
# T-A1: Dashboard config section loads correctly
# ─────────────────────────────────────────────────────────────────────────────


class TestDashboardConfig:
    """Tests for the dashboard configuration section (R10)."""

    def test_dashboard_config_loads_defaults(self):
        """T-A1: Dashboard config section loads with expected defaults."""
        settings = get_settings_yaml()
        dashboard = settings["dashboard"]

        assert dashboard["trend_window"] == 5
        assert dashboard["trend_threshold"] == 2
        assert dashboard["health_thresholds"] == [95, 80, 60]
        assert dashboard["ai_analysis_model"] == "gpt-4o"
        assert dashboard["max_runs_scan"] == 50
