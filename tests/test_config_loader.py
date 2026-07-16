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

from pathlib import Path

import pytest

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
    get_concept_keywords,
    get_concept_bump_hints,
    get_default_bump_hints,
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
        total = (
            weights.alpha_vec
            + weights.beta_bm25
            + weights.gamma_cite
            + weights.delta_role
        )
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
        (config_dir / "settings.yaml").write_text("openai:\n  chat_model: initial\n")

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)

        # Load initial settings
        clear_config_cache()
        settings1 = load_settings()
        assert settings1.chat_model == "initial"

        # Update file
        (config_dir / "settings.yaml").write_text("openai:\n  chat_model: updated\n")

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


class TestGetCaseLawSettings:
    """Tests for get_case_law_settings function."""

    def setup_method(self):
        clear_config_cache()

    def teardown_method(self):
        clear_config_cache()

    def test_loads_case_law_settings(self, monkeypatch):
        """Returns CaseLawSettings with correct fields from settings.yaml."""
        monkeypatch.setattr(
            config_loader,
            "_load_settings_yaml",
            lambda: {
                "case_law": {
                    "parser": {
                        "section_heading_patterns": {
                            "grounds": ["Grounds"],
                            "operative_part": ["Operative part"],
                        },
                        "article_reference_patterns": [
                            r"Article\s+(\d+)\s+of\s+Regulation\s+(\d{4}/\d{1,5})"
                        ],
                        "corpus_celex_mapping": {"32016R0679": "gdpr"},
                    }
                }
            },
        )
        from src.common.config_loader import (
            get_case_law_settings,
            CaseLawParserSettings,
        )

        result = get_case_law_settings()
        assert isinstance(result, CaseLawParserSettings)
        assert result.section_heading_patterns == {
            "grounds": ["Grounds"],
            "operative_part": ["Operative part"],
        }
        assert result.corpus_celex_mapping == {"32016R0679": "gdpr"}
        assert len(result.article_reference_patterns) == 1

    def test_default_when_missing(self, monkeypatch):
        """No case_law key returns empty defaults."""
        monkeypatch.setattr(config_loader, "_load_settings_yaml", lambda: {})
        from src.common.config_loader import (
            get_case_law_settings,
            CaseLawParserSettings,
        )

        result = get_case_law_settings()
        assert isinstance(result, CaseLawParserSettings)
        assert result.section_heading_patterns == {}
        assert result.article_reference_patterns == ()
        assert result.corpus_celex_mapping == {}

    def test_patterns_pre_compiled(self, monkeypatch):
        """article_reference_patterns are re.Pattern instances."""
        import re

        monkeypatch.setattr(
            config_loader,
            "_load_settings_yaml",
            lambda: {
                "case_law": {
                    "parser": {
                        "article_reference_patterns": [
                            r"Article\s+(\d+)\s+of\s+Regulation\s+(\d{4}/\d{1,5})"
                        ],
                    }
                }
            },
        )
        from src.common.config_loader import get_case_law_settings

        result = get_case_law_settings()
        assert all(isinstance(p, re.Pattern) for p in result.article_reference_patterns)

    def test_cache_clear_integration(self, monkeypatch):
        """clear_config_cache() clears get_case_law_settings cache."""
        call_count = 0

        def counting_loader():
            nonlocal call_count
            call_count += 1
            return {
                "case_law": {
                    "parser": {"corpus_celex_mapping": {"key": f"val{call_count}"}}
                }
            }

        monkeypatch.setattr(config_loader, "_load_settings_yaml", counting_loader)
        from src.common.config_loader import get_case_law_settings

        result1 = get_case_law_settings()
        result2 = get_case_law_settings()
        assert result1 is result2  # cached

        clear_config_cache()
        result3 = get_case_law_settings()
        assert result3 is not result1  # cache was cleared

    def test_frozen(self, monkeypatch):
        """CaseLawSettings is immutable."""
        monkeypatch.setattr(
            config_loader,
            "_load_settings_yaml",
            lambda: {
                "case_law": {"parser": {"corpus_celex_mapping": {"32016R0679": "gdpr"}}}
            },
        )
        from src.common.config_loader import get_case_law_settings

        result = get_case_law_settings()
        with pytest.raises(AttributeError):
            result.corpus_celex_mapping = {}


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


# ─────────────────────────────────────────────────────────────────────────────
# CaseLawSettings Loading and Validation (REQ-9, Edge 7)
# ─────────────────────────────────────────────────────────────────────────────


class TestCaseLawConfigLoading:
    """Tests for CaseLawSettings parsing, defaults, and validation."""

    def test_case_law_section_parsed(self, monkeypatch, tmp_path):
        """REQ-9: case_law YAML section is parsed into CaseLawSettings on Settings."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n"
            "  enabled: true\n"
            "  client:\n"
            "    sparql_endpoint: 'https://example.com/sparql'\n"
            "    request_timeout_secs: 60\n"
            "    max_retries: 5\n"
            "    request_delay_secs: 2.0\n"
            "    base_retry_delay_secs: 0.5\n"
            '    user_agent: "TestAgent/1.0"\n',
            encoding="utf-8",
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()

        settings = load_settings()
        cl = settings.case_law
        assert cl.enabled is True
        assert cl.sparql_endpoint == "https://example.com/sparql"
        assert cl.request_timeout_secs == 60
        assert cl.max_retries == 5
        assert cl.request_delay_secs == 2.0
        assert cl.base_retry_delay_secs == 0.5
        assert cl.user_agent == "TestAgent/1.0"

    def test_case_law_missing_defaults_disabled(self, monkeypatch, tmp_path):
        """Edge 7: Missing case_law section defaults to enabled=false."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "openai:\n  chat_model: test\n", encoding="utf-8"
        )

        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()

        settings = load_settings()
        cl = settings.case_law
        assert cl.enabled is False
        assert cl.sparql_endpoint == "https://publications.europa.eu/webapi/rdf/sparql"
        assert cl.max_retries == 3
        assert cl.request_delay_secs == 1.0

    def test_case_law_validate_delay_positive(self):
        """request_delay_secs <= 0 raises ValueError."""
        from src.common.config_loader import CaseLawSettings

        settings = Settings(case_law=CaseLawSettings(request_delay_secs=0))
        with pytest.raises(ValueError, match="request_delay_secs must be > 0"):
            _validate_settings(settings)

        settings = Settings(case_law=CaseLawSettings(request_delay_secs=-1.0))
        with pytest.raises(ValueError, match="request_delay_secs must be > 0"):
            _validate_settings(settings)

    def test_case_law_validate_retries_nonnegative(self):
        """max_retries < 0 raises ValueError."""
        from src.common.config_loader import CaseLawSettings

        settings = Settings(case_law=CaseLawSettings(max_retries=-1))
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            _validate_settings(settings)

    def test_case_law_validate_timeout_positive(self):
        """request_timeout_secs <= 0 raises ValueError."""
        from src.common.config_loader import CaseLawSettings

        settings = Settings(case_law=CaseLawSettings(request_timeout_secs=0))
        with pytest.raises(ValueError, match="request_timeout_secs must be > 0"):
            _validate_settings(settings)

    def test_case_law_validate_retry_delay_positive(self):
        """base_retry_delay_secs <= 0 raises ValueError."""
        from src.common.config_loader import CaseLawSettings

        settings = Settings(case_law=CaseLawSettings(base_retry_delay_secs=0))
        with pytest.raises(ValueError, match="base_retry_delay_secs must be > 0"):
            _validate_settings(settings)

    def test_case_law_eurlex_html_base_url_default(self):
        """eurlex_html_base_url defaults to EUR-Lex HTML content URL."""
        from src.common.config_loader import CaseLawSettings

        cl = CaseLawSettings()
        assert (
            cl.eurlex_html_base_url
            == "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:"
        )

    def test_case_law_eurlex_html_base_url_parsed(self, monkeypatch, tmp_path):
        """eurlex_html_base_url is parsed from YAML."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n"
            "  client:\n"
            "    eurlex_html_base_url: 'https://custom.example.com/html?celex='\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()

        settings = load_settings()
        assert (
            settings.case_law.eurlex_html_base_url
            == "https://custom.example.com/html?celex="
        )

    def test_case_law_max_backoff_secs_default(self):
        """max_backoff_secs defaults to 60.0."""
        from src.common.config_loader import CaseLawSettings

        cl = CaseLawSettings()
        assert cl.max_backoff_secs == 60.0

    def test_case_law_max_backoff_secs_parsed(self, monkeypatch, tmp_path):
        """max_backoff_secs is parsed from YAML."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n  client:\n    max_backoff_secs: 120.0\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()

        settings = load_settings()
        assert settings.case_law.max_backoff_secs == 120.0

    def test_case_law_validate_max_backoff_positive(self):
        """max_backoff_secs <= 0 raises ValueError."""
        from src.common.config_loader import CaseLawSettings

        settings = Settings(case_law=CaseLawSettings(max_backoff_secs=0))
        with pytest.raises(ValueError, match="max_backoff_secs must be > 0"):
            _validate_settings(settings)


# ─────────────────────────────────────────────────────────────────────────────
# C3: CaseLawChunkingSettings, CaseLawDomainSettings, nested path reads
# ─────────────────────────────────────────────────────────────────────────────


class TestCaseLawChunkingSettings:
    def test_defaults(self):
        from src.common.config_loader import CaseLawChunkingSettings

        s = CaseLawChunkingSettings()
        assert s.chunk_tokens == 500
        assert s.min_chunk_tokens == 100
        assert s.overlap_tokens == 50
        assert s.non_breaking_abbreviations == ["Art.", "No.", "p.", "cf.", "para."]

    def test_missing_chunking_subsection_uses_defaults(self, monkeypatch, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n  enabled: false\n", encoding="utf-8"
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()
        settings = load_settings()
        assert settings.case_law.chunking.chunk_tokens == 500

    def test_chunk_tokens_zero_raises(self):
        from src.common.config_loader import CaseLawChunkingSettings, CaseLawSettings

        settings = Settings(
            case_law=CaseLawSettings(chunking=CaseLawChunkingSettings(chunk_tokens=0))
        )
        with pytest.raises(ValueError, match="chunk_tokens"):
            _validate_settings(settings)

    def test_overlap_ge_chunk_raises(self):
        from src.common.config_loader import CaseLawChunkingSettings, CaseLawSettings

        settings = Settings(
            case_law=CaseLawSettings(
                chunking=CaseLawChunkingSettings(overlap_tokens=500, chunk_tokens=500)
            )
        )
        with pytest.raises(ValueError, match="overlap_tokens"):
            _validate_settings(settings)

    def test_frozen(self):
        from src.common.config_loader import CaseLawChunkingSettings

        s = CaseLawChunkingSettings()
        with pytest.raises(AttributeError):
            s.chunk_tokens = 300


class TestCaseLawDomainSettings:
    def test_court_mapping_from_yaml(self, monkeypatch, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n  court_mapping:\n    C: CJEU\n    T: General Court\n"
            "  section_groups:\n    procedural: [parties]\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()
        settings = load_settings()
        assert settings.case_law.domain.court_mapping == {
            "C": "CJEU",
            "T": "General Court",
        }

    def test_missing_court_mapping_defaults_to_empty_dict(self, monkeypatch, tmp_path):
        """W1: Missing court_mapping YAML key defaults to empty dict, not None."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n  enabled: false\n", encoding="utf-8"
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()
        settings = load_settings()
        assert settings.case_law.domain.court_mapping == {}
        assert isinstance(settings.case_law.domain.court_mapping, dict)

    def test_missing_section_groups_defaults_to_empty_dict(self, monkeypatch, tmp_path):
        """W1: Missing section_groups YAML key defaults to empty dict, not None."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n  enabled: false\n", encoding="utf-8"
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()
        settings = load_settings()
        assert settings.case_law.domain.section_groups == {}
        assert isinstance(settings.case_law.domain.section_groups, dict)

    def test_missing_abbreviations_defaults_to_list(self, monkeypatch, tmp_path):
        """W1: Missing non_breaking_abbreviations YAML key defaults to list, not None."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n  enabled: false\n", encoding="utf-8"
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()
        settings = load_settings()
        assert settings.case_law.chunking.non_breaking_abbreviations == [
            "Art.",
            "No.",
            "p.",
            "cf.",
            "para.",
        ]
        assert isinstance(settings.case_law.chunking.non_breaking_abbreviations, list)

    def test_section_groups_all_four_groups(self, monkeypatch, tmp_path):
        """W7: Extend YAML fixture to include all 4 groups and verify."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n"
            "  section_groups:\n"
            "    procedural: [parties, subject, keywords]\n"
            "    substantive: [summary, grounds, costs]\n"
            "    dispositive: [operative_part]\n"
            "    fallback: [full_text]\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()
        settings = load_settings()
        groups = settings.case_law.domain.section_groups
        assert len(groups) == 4
        assert groups["procedural"] == ["parties", "subject", "keywords"]
        assert groups["substantive"] == ["summary", "grounds", "costs"]
        assert groups["dispositive"] == ["operative_part"]
        assert groups["fallback"] == ["full_text"]

    def test_frozen(self):
        from src.common.config_loader import CaseLawDomainSettings

        s = CaseLawDomainSettings(court_mapping={}, section_groups={})
        with pytest.raises(AttributeError):
            s.court_mapping = {"X": "Y"}


class TestNestedPathReads:
    def test_client_settings_from_nested_path(self, monkeypatch, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "settings.yaml").write_text(
            "case_law:\n  client:\n    sparql_endpoint: https://custom.example.com/sparql\n    max_retries: 5\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config_loader, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_loader, "_REPO_ROOT", tmp_path)
        clear_config_cache()
        settings = load_settings()
        assert settings.case_law.sparql_endpoint == "https://custom.example.com/sparql"
        assert settings.case_law.max_retries == 5

    def test_parser_settings_from_nested_path(self, monkeypatch):
        monkeypatch.setattr(
            config_loader,
            "_load_settings_yaml",
            lambda: {
                "case_law": {
                    "parser": {
                        "section_heading_patterns": {"grounds": ["Grounds"]},
                        "corpus_celex_mapping": {"32016R0679": "gdpr"},
                    }
                }
            },
        )
        from src.common.config_loader import get_case_law_settings

        clear_config_cache()
        result = get_case_law_settings()
        assert result.section_heading_patterns == {"grounds": ["Grounds"]}

    def test_no_duplicate_yaml_keys(self):
        import re as re_mod

        settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
        content = settings_path.read_text()
        matches = re_mod.findall(r"^case_law:", content, re_mod.MULTILINE)
        assert len(matches) == 1


class TestCaseLawDocTypeMapping:
    """T4.1-T4.4: celex_document_type_mapping and max_response_bytes."""

    def setup_method(self):
        clear_config_cache()

    def teardown_method(self):
        clear_config_cache()

    def test_celex_document_type_mapping_default(self):
        """T4.1: Default mapping has CJ, CC, CO."""
        from src.common.config_loader import CaseLawParserSettings

        ps = CaseLawParserSettings(
            section_heading_patterns={},
            article_reference_patterns=(),
            corpus_celex_mapping={},
        )
        assert ps.celex_document_type_mapping == {
            "CJ": "judgment",
            "CC": "ag_opinion",
            "CO": "order",
        }

    def test_celex_document_type_mapping_loaded_from_yaml(self, monkeypatch):
        """T4.2: get_case_law_settings loads mapping from YAML."""
        monkeypatch.setattr(
            config_loader,
            "_load_settings_yaml",
            lambda: {
                "case_law": {
                    "celex_document_type_mapping": {"CJ": "judgment", "XX": "custom"},
                    "parser": {
                        "corpus_celex_mapping": {"32016R0679": "gdpr"},
                    },
                }
            },
        )
        from src.common.config_loader import get_case_law_settings

        result = get_case_law_settings()
        assert result.celex_document_type_mapping == {"CJ": "judgment", "XX": "custom"}

    def test_celex_document_type_mapping_absent_uses_default(self, monkeypatch):
        """T4.3: When YAML key absent, default is used."""
        monkeypatch.setattr(
            config_loader,
            "_load_settings_yaml",
            lambda: {
                "case_law": {
                    "parser": {
                        "corpus_celex_mapping": {"32016R0679": "gdpr"},
                    },
                }
            },
        )
        from src.common.config_loader import get_case_law_settings

        result = get_case_law_settings()
        assert result.celex_document_type_mapping == {
            "CJ": "judgment",
            "CC": "ag_opinion",
            "CO": "order",
        }

    def test_max_response_bytes_default(self):
        """T4.4: CaseLawSettings.max_response_bytes default is 10 MB."""
        from src.common.config_loader import CaseLawSettings

        cl = CaseLawSettings()
        assert cl.max_response_bytes == 10_485_760
