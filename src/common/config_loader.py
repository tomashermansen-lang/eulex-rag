"""
Unified configuration loader for the RAG framework.

This module is the single source of truth for all configuration:
- Settings dataclasses (Settings, CorpusSettings)
- Loading settings from config/settings.yaml with env var overrides
- Concept routing configuration from config/concepts/
- Pydantic schema validation for production-ready error reporting

All code should import configuration from this module, not from settings.yaml directly.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ValidationError, field_validator

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Core Settings Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CorpusSettings:
    """Configuration for a single corpus/law."""

    id: str
    display_name: str
    chunks_collection: str
    max_distance: float | None = None
    source_url: str | None = None


@dataclass(frozen=True)
class RankingWeights:
    """Configurable weights for hybrid reranking.

    score = α*vec_sim + β*bm25 + γ*citation + δ*role
    All weights should sum to 1.0
    """

    alpha_vec: float = 0.25  # Vector similarity weight
    beta_bm25: float = 0.25  # BM25 lexical match weight
    gamma_cite: float = 0.35  # Citation graph boost weight
    delta_role: float = 0.15  # Role alignment weight


@dataclass(frozen=True)
class CaseLawChunkingSettings:
    """Chunking parameters for case law sections."""

    chunk_tokens: int = 500
    min_chunk_tokens: int = 100
    overlap_tokens: int = 50
    non_breaking_abbreviations: list[str] = field(
        default_factory=lambda: ["Art.", "No.", "p.", "cf.", "para."]
    )


@dataclass(frozen=True)
class CaseLawDomainSettings:
    """Domain-specific mappings for case law (court codes, section groups)."""

    court_mapping: dict[str, str] = field(default_factory=dict)
    section_groups: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseLawSettings:
    """Configuration for CJEU case law integration (CELLAR SPARQL client)."""

    enabled: bool = False
    sparql_endpoint: str = "https://publications.europa.eu/webapi/rdf/sparql"
    request_timeout_secs: int = 30
    max_retries: int = 3
    request_delay_secs: float = 1.0
    base_retry_delay_secs: float = 1.0
    max_backoff_secs: float = 60.0
    user_agent: str = "EuLex-RAG-Framework/1.0 (CellarClient)"
    eurlex_html_base_url: str = (
        "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:"
    )
    max_response_bytes: int = 10_485_760  # 10 MB
    default_output_dir: str = "data/processed/case_law"
    default_language: str = "en"
    chunking: CaseLawChunkingSettings = CaseLawChunkingSettings()
    domain: CaseLawDomainSettings = CaseLawDomainSettings()


@dataclass(frozen=True)
class Settings:
    """Application settings - all values loaded from config/settings.yaml.

    This is a frozen dataclass to ensure immutability after loading.
    All values are set at load time from the YAML config with env var overrides.
    """

    # OpenAI settings
    chat_model: str = ""
    embedding_model: str = ""

    # RAG settings - simplified pipeline
    retrieval_pool_size: int = 50  # Candidates to fetch before reranking
    max_context_legal: int = 20  # Final cap for LEGAL profile
    max_context_engineering: int = 15  # Final cap for ENGINEERING profile
    rag_max_distance: float | None = (
        1.25  # Soft threshold - triggers low-evidence warning
    )
    rag_hard_max_distance: float | None = (
        1.3  # Hard threshold - abstains if best exceeds
    )

    # Hybrid retrieval with 4-factor scoring (always enabled)
    hybrid_vec_k: int = 30
    ranking_weights: RankingWeights = RankingWeights()

    # Default corpus selection
    default_corpus: str = ""

    # Paths
    docs_path: Path = Path("data/sample_docs")
    vector_store_path: Path = Path("data/vector_store")
    raw_html_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    # Chunking parameters
    eurlex_chunk_tokens: int = 500
    eurlex_overlap: int = 100

    # Context positioning (sandwich pattern for "Lost in the Middle" mitigation)
    context_positioning: str = "sandwich"  # "sandwich" | "relevance" | "none"

    # Case law integration
    case_law: CaseLawSettings = CaseLawSettings()

    # Corpus registry
    corpora: dict[str, "CorpusSettings"] | None = None


def _get_concepts_dir() -> Path:
    """Get concepts directory. Can be overridden via CONCEPTS_DIR env var for testing."""
    override = os.getenv("CONCEPTS_DIR")
    if override:
        return Path(override)
    return _CONFIG_DIR / "concepts"


# ─────────────────────────────────────────────────────────────────────────────
# Settings Loading Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _slugify(value: str) -> str:
    """Convert a string to a URL-safe slug."""
    value = (value or "").strip().lower().replace(" ", "-")
    value = re.sub(r"[^a-z0-9\-]+", "", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "doc"


def _discover_corpora_from_raw_html(
    *, root: Path, raw_html_dir: Path
) -> dict[str, CorpusSettings]:
    """Fallback: discover corpora from HTML files in raw directory."""
    raw_dir = raw_html_dir if raw_html_dir.is_absolute() else (root / raw_html_dir)

    corpora: dict[str, CorpusSettings] = {}
    for html_path in sorted(raw_dir.glob("*.html")):
        corpus_id = _slugify(html_path.stem)
        if corpus_id in corpora:
            raise ValueError(
                f"Duplicate corpus id {corpus_id!r} from HTML files '{html_path.name}' and '{corpora[corpus_id].display_name}.html'"
            )
        corpora[corpus_id] = CorpusSettings(
            id=corpus_id,
            display_name=html_path.stem,
            chunks_collection=f"{corpus_id}_documents",
            max_distance=None,
        )
    return corpora


def _validate_settings(settings: Settings) -> None:
    """Validate settings values."""
    if settings.retrieval_pool_size < 1:
        raise ValueError(
            f"rag.retrieval_pool_size must be >= 1 (got {settings.retrieval_pool_size})"
        )

    if settings.rag_max_distance is not None and settings.rag_max_distance <= 0:
        raise ValueError(
            f"rag.max_distance must be > 0 when set (got {settings.rag_max_distance})"
        )

    if (
        settings.rag_hard_max_distance is not None
        and settings.rag_hard_max_distance <= 0
    ):
        raise ValueError(
            f"rag.hard_max_distance must be > 0 when set (got {settings.rag_hard_max_distance})"
        )

    # Validate ranking weights sum to approximately 1.0
    w = settings.ranking_weights
    weight_sum = w.alpha_vec + w.beta_bm25 + w.gamma_cite + w.delta_role
    if not (0.95 <= weight_sum <= 1.05):
        raise ValueError(
            f"ranking_weights must sum to ~1.0 (got {weight_sum:.2f}: "
            f"α={w.alpha_vec}, β={w.beta_bm25}, γ={w.gamma_cite}, δ={w.delta_role})"
        )

    if settings.eurlex_overlap >= settings.eurlex_chunk_tokens:
        raise ValueError(
            f"eurlex.overlap must be < eurlex.chunk_tokens "
            f"(got {settings.eurlex_overlap} >= {settings.eurlex_chunk_tokens})"
        )

    _validate_case_law_settings(settings.case_law)


def _validate_case_law_settings(cl: CaseLawSettings) -> None:
    """Validate case law settings values."""
    if cl.request_delay_secs <= 0:
        raise ValueError(
            f"case_law.request_delay_secs must be > 0 (got {cl.request_delay_secs})"
        )
    if cl.max_retries < 0:
        raise ValueError(f"case_law.max_retries must be >= 0 (got {cl.max_retries})")
    if cl.request_timeout_secs <= 0:
        raise ValueError(
            f"case_law.request_timeout_secs must be > 0 (got {cl.request_timeout_secs})"
        )
    if cl.base_retry_delay_secs <= 0:
        raise ValueError(
            f"case_law.base_retry_delay_secs must be > 0 (got {cl.base_retry_delay_secs})"
        )
    if cl.max_backoff_secs <= 0:
        raise ValueError(
            f"case_law.max_backoff_secs must be > 0 (got {cl.max_backoff_secs})"
        )

    ch = cl.chunking
    if ch.chunk_tokens <= 0:
        raise ValueError(
            f"case_law.chunking.chunk_tokens must be > 0 (got {ch.chunk_tokens})"
        )
    if ch.min_chunk_tokens <= 0:
        raise ValueError(
            f"case_law.chunking.min_chunk_tokens must be > 0 (got {ch.min_chunk_tokens})"
        )
    if ch.overlap_tokens < 0:
        raise ValueError(
            f"case_law.chunking.overlap_tokens must be >= 0 (got {ch.overlap_tokens})"
        )
    if ch.overlap_tokens >= ch.chunk_tokens:
        raise ValueError(
            f"case_law.chunking.overlap_tokens must be < chunk_tokens (got {ch.overlap_tokens} >= {ch.chunk_tokens})"
        )


def _load_settings_yaml() -> dict[str, Any]:
    """Load raw settings from config/settings.yaml."""
    settings_path = _CONFIG_DIR / "settings.yaml"
    if not settings_path.exists():
        return {}
    with open(settings_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _env_int(key: str, default: int) -> int:
    """Read an integer from an environment variable, falling back to default."""
    val = os.getenv(key)
    return int(val) if val and val.strip() else default


def _env_float(key: str, default: float | None) -> float | None:
    """Read a float from an environment variable, falling back to default."""
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return float(val)


def _resolve_path(path: Path, root: Path) -> Path:
    """Resolve a path relative to root if it is not absolute."""
    return path if path.is_absolute() else root / path


def _parse_corpus_payload(
    corpus_id: str, payload: dict[str, Any]
) -> CorpusSettings | None:
    """Parse a single corpus entry from inventory or YAML into CorpusSettings."""
    if not isinstance(corpus_id, str) or not isinstance(payload, dict):
        return None
    if payload.get("enabled") is False:
        return None

    display_name = str(payload.get("display_name") or corpus_id)
    chunks_collection = str(
        payload.get("chunks_collection") or f"{corpus_id}_documents"
    )

    source_url = payload.get("source_url")
    if source_url is not None:
        source_url = str(source_url).strip() or None

    corpus_max_distance = _safe_float(payload.get("max_distance"))

    return CorpusSettings(
        id=corpus_id,
        display_name=display_name,
        chunks_collection=chunks_collection,
        max_distance=corpus_max_distance,
        source_url=source_url,
    )


def _load_corpora_from_inventory(corpora_path: Path) -> dict[str, CorpusSettings]:
    """Load corpora from the JSON inventory file."""
    from .corpora_inventory import load_corpora_inventory

    inventory = load_corpora_inventory(corpora_path)
    inv_corpora = inventory.get("corpora") if isinstance(inventory, dict) else {}
    if not isinstance(inv_corpora, Mapping):
        raise SystemExit(
            f"Invalid format in corpora inventory '{corpora_path}': 'corpora' must be a JSON object"
        )

    corpora: dict[str, CorpusSettings] = {}
    for corpus_id, payload in inv_corpora.items():
        parsed = _parse_corpus_payload(corpus_id, payload)
        if parsed is not None:
            corpora[corpus_id] = parsed
    return corpora


def _safe_float(value: Any) -> float | None:
    """Try to convert value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def _merge_corpus_override(
    corpus_id: str,
    payload: dict[str, Any],
    base: CorpusSettings | None,
) -> CorpusSettings:
    """Build a CorpusSettings by merging YAML override with existing base."""
    display_name = str(payload.get("display_name") or corpus_id)
    chunks_collection = str(payload.get("chunks_collection") or "").strip()
    corpus_max_distance = _safe_float(payload.get("max_distance"))

    if base is not None and not chunks_collection:
        chunks_collection = base.chunks_collection
    if base is not None and corpus_max_distance is None:
        corpus_max_distance = base.max_distance

    source_url = str(payload.get("source_url") or "").strip() or (
        base.source_url if base else None
    )

    return CorpusSettings(
        id=corpus_id,
        display_name=display_name,
        chunks_collection=chunks_collection,
        max_distance=corpus_max_distance,
        source_url=source_url,
    )


def _apply_yaml_corpora_overrides(
    corpora: dict[str, CorpusSettings],
    corpora_raw: Any,
) -> None:
    """Merge YAML corpus overrides into the corpora dict in-place."""
    if not isinstance(corpora_raw, Mapping):
        return
    for corpus_id, payload in corpora_raw.items():
        if not isinstance(corpus_id, str) or not isinstance(payload, dict):
            continue
        corpora[corpus_id] = _merge_corpus_override(
            corpus_id, payload, corpora.get(corpus_id)
        )


def _resolve_corpora(
    root: Path,
    raw_html_dir: Path,
    corpora_raw: Any,
) -> dict[str, CorpusSettings]:
    """Load corpora from inventory file, with YAML overrides and HTML fallback."""
    from .corpora_inventory import default_corpora_path

    corpora_path_raw = os.getenv("RAG_CORPORA_PATH")
    if corpora_path_raw is not None and corpora_path_raw.strip() == "":
        corpora_path_raw = None
    corpora_path = (
        Path(corpora_path_raw) if corpora_path_raw else default_corpora_path(root)
    )

    if corpora_path.exists():
        corpora = _load_corpora_from_inventory(corpora_path)
    else:
        logger.info("corpora.json missing, using fallback defaults")
        corpora = _discover_corpora_from_raw_html(root=root, raw_html_dir=raw_html_dir)

    _apply_yaml_corpora_overrides(corpora, corpora_raw)
    return corpora


@functools.lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Load and validate application settings from config/settings.yaml.

    Environment variables override YAML values:
    - OPENAI_CHAT_MODEL, OPENAI_EMBEDDING_MODEL
    - RAG_RETRIEVAL_POOL_SIZE, RAG_MAX_CONTEXT_LEGAL, RAG_MAX_CONTEXT_ENGINEERING
    - RAG_MAX_DISTANCE, RAG_ANCHOR_WEIGHT
    - RAG_ENABLE_HYBRID_RERANK, RAG_HYBRID_ALPHA, RAG_HYBRID_VEC_K
    - RAG_DEFAULT_CORPUS
    - RAG_CORPORA_PATH

    Returns:
        Settings: Validated, frozen settings object
    """
    if load_dotenv is not None:
        load_dotenv()

    config = _load_settings_yaml()
    root = _REPO_ROOT

    openai_cfg = config.get("openai", {})
    rag_cfg = config.get("rag", {})
    paths_cfg = config.get("paths", {})
    eurlex_cfg = config.get("eurlex", {})
    case_law_cfg = config.get("case_law", {})

    chat_model = os.getenv("OPENAI_CHAT_MODEL") or openai_cfg.get("chat_model", "")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL") or openai_cfg.get(
        "embedding_model", ""
    )

    weights_cfg = rag_cfg.get("ranking_weights", {})
    ranking_weights = RankingWeights(
        alpha_vec=_env_float("RAG_ALPHA_VEC", weights_cfg.get("alpha_vec", 0.25))
        or 0.25,
        beta_bm25=_env_float("RAG_BETA_BM25", weights_cfg.get("beta_bm25", 0.25))
        or 0.25,
        gamma_cite=_env_float("RAG_GAMMA_CITE", weights_cfg.get("gamma_cite", 0.35))
        or 0.35,
        delta_role=_env_float("RAG_DELTA_ROLE", weights_cfg.get("delta_role", 0.15))
        or 0.15,
    )

    raw_html_dir = _resolve_path(Path(paths_cfg.get("raw_html_dir", "data/raw")), root)
    default_corpus = os.getenv("RAG_DEFAULT_CORPUS") or rag_cfg.get(
        "default_corpus", ""
    )
    corpora = _resolve_corpora(root, raw_html_dir, config.get("corpora", {}))

    if not default_corpus or default_corpus not in corpora:
        default_corpus = next(iter(sorted(corpora.keys())), "")

    # Case law settings — reads from case_law.client.* subsection
    client_cfg = case_law_cfg.get("client", {})
    chunking_cfg = case_law_cfg.get("chunking", {})
    chunking_abbrevs = chunking_cfg.get("non_breaking_abbreviations") or [
        "Art.",
        "No.",
        "p.",
        "cf.",
        "para.",
    ]
    chunking_settings = CaseLawChunkingSettings(
        chunk_tokens=int(
            chunking_cfg.get("chunk_tokens", CaseLawChunkingSettings.chunk_tokens)
        ),
        min_chunk_tokens=int(
            chunking_cfg.get(
                "min_chunk_tokens", CaseLawChunkingSettings.min_chunk_tokens
            )
        ),
        overlap_tokens=int(
            chunking_cfg.get("overlap_tokens", CaseLawChunkingSettings.overlap_tokens)
        ),
        non_breaking_abbreviations=chunking_abbrevs,
    )
    domain_settings = CaseLawDomainSettings(
        court_mapping=case_law_cfg.get("court_mapping") or {},
        section_groups=case_law_cfg.get("section_groups") or {},
    )
    case_law = CaseLawSettings(
        enabled=bool(case_law_cfg.get("enabled", False)),
        sparql_endpoint=str(
            client_cfg.get("sparql_endpoint", CaseLawSettings.sparql_endpoint)
        ),
        request_timeout_secs=int(
            client_cfg.get("request_timeout_secs", CaseLawSettings.request_timeout_secs)
        ),
        max_retries=int(client_cfg.get("max_retries", CaseLawSettings.max_retries)),
        request_delay_secs=float(
            client_cfg.get("request_delay_secs", CaseLawSettings.request_delay_secs)
        ),
        base_retry_delay_secs=float(
            client_cfg.get(
                "base_retry_delay_secs", CaseLawSettings.base_retry_delay_secs
            )
        ),
        max_backoff_secs=float(
            client_cfg.get("max_backoff_secs", CaseLawSettings.max_backoff_secs)
        ),
        user_agent=str(client_cfg.get("user_agent", CaseLawSettings.user_agent)),
        eurlex_html_base_url=str(
            client_cfg.get("eurlex_html_base_url", CaseLawSettings.eurlex_html_base_url)
        ),
        max_response_bytes=int(
            client_cfg.get("max_response_bytes", CaseLawSettings.max_response_bytes)
        ),
        default_output_dir=str(
            case_law_cfg.get("default_output_dir", CaseLawSettings.default_output_dir)
        ),
        default_language=str(
            case_law_cfg.get("default_language", CaseLawSettings.default_language)
        ),
        chunking=chunking_settings,
        domain=domain_settings,
    )

    context_positioning = os.getenv("RAG_CONTEXT_POSITIONING") or rag_cfg.get(
        "context_positioning", "sandwich"
    )

    settings = Settings(
        chat_model=str(chat_model),
        embedding_model=str(embedding_model),
        retrieval_pool_size=_env_int(
            "RAG_RETRIEVAL_POOL_SIZE", int(rag_cfg.get("retrieval_pool_size", 50))
        ),
        max_context_legal=_env_int(
            "RAG_MAX_CONTEXT_LEGAL", int(rag_cfg.get("max_context_legal", 20))
        ),
        max_context_engineering=_env_int(
            "RAG_MAX_CONTEXT_ENGINEERING",
            int(rag_cfg.get("max_context_engineering", 15)),
        ),
        rag_max_distance=_env_float(
            "RAG_MAX_DISTANCE", rag_cfg.get("max_distance", 1.25)
        ),
        rag_hard_max_distance=_env_float(
            "RAG_HARD_MAX_DISTANCE", rag_cfg.get("hard_max_distance", 1.3)
        ),
        hybrid_vec_k=_env_int("RAG_HYBRID_VEC_K", int(rag_cfg.get("hybrid_vec_k", 30))),
        ranking_weights=ranking_weights,
        default_corpus=str(default_corpus),
        docs_path=_resolve_path(
            Path(paths_cfg.get("docs_path", "data/sample_docs")), root
        ),
        vector_store_path=_resolve_path(
            Path(paths_cfg.get("vector_store_path", "data/vector_store")), root
        ),
        raw_html_dir=raw_html_dir,
        processed_dir=_resolve_path(
            Path(paths_cfg.get("processed_dir", "data/processed")), root
        ),
        eurlex_chunk_tokens=int(eurlex_cfg.get("chunk_tokens", 500)),
        eurlex_overlap=int(eurlex_cfg.get("overlap", 100)),
        context_positioning=str(context_positioning),
        case_law=case_law,
        corpora=corpora,
    )

    _validate_settings(settings)
    return settings


def get_settings_yaml() -> dict[str, Any]:
    """Get raw settings dict from YAML (for backwards compatibility)."""
    return _load_settings_yaml()


@dataclass(frozen=True)
class CaseLawParserSettings:
    """Configuration for case law parsing from settings.yaml case_law section."""

    section_heading_patterns: dict[str, list[str]]
    article_reference_patterns: tuple[re.Pattern[str], ...]
    corpus_celex_mapping: dict[str, str]
    celex_document_type_mapping: dict[str, str] = field(
        default_factory=lambda: {"CJ": "judgment", "CC": "ag_opinion", "CO": "order"}
    )


@functools.lru_cache(maxsize=1)
def get_case_law_settings() -> CaseLawParserSettings:
    """Get case law settings from settings.yaml.

    Returns a frozen typed dataclass. Cached after first call.
    Call get_case_law_settings.cache_clear() to invalidate
    (clear_config_cache() includes this).
    """
    settings = get_settings_yaml()
    raw = settings.get("case_law", {})
    parser_raw = raw.get("parser", {})
    # celex_document_type_mapping is a sibling of parser, not nested under it
    default_type_mapping = {"CJ": "judgment", "CC": "ag_opinion", "CO": "order"}
    return CaseLawParserSettings(
        section_heading_patterns=parser_raw.get("section_heading_patterns", {}),
        article_reference_patterns=tuple(
            re.compile(p) for p in parser_raw.get("article_reference_patterns", [])
        ),
        corpus_celex_mapping=parser_raw.get("corpus_celex_mapping", {}),
        celex_document_type_mapping=raw.get(
            "celex_document_type_mapping", default_type_mapping
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schema for Concept Configuration
# ─────────────────────────────────────────────────────────────────────────────


class RescueRuleSchema(BaseModel):
    """Schema for a rescue rule."""

    if_present: list[str] = []
    must_include_one_of: list[str] = []
    action: str = "anchor_lookup_inject"
    profiles: list[str] = ["ANY"]

    @field_validator("if_present", "must_include_one_of", "profiles", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


class ConceptSchema(BaseModel):
    """Schema for a single concept definition."""

    keywords: list[str] = []
    toc_contains: list[str] = []
    heading_contains: list[str] = []
    bump_hints: list[str] = []
    rescue_rules: list[RescueRuleSchema] = []
    answer_policy: dict[str, Any] | None = None
    normative_guard: dict[str, Any] | None = None

    @field_validator(
        "keywords", "toc_contains", "heading_contains", "bump_hints", mode="before"
    )
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


class CorpusConfigSchema(BaseModel):
    """Schema for a corpus concept configuration file."""

    concepts: dict[str, ConceptSchema] = {}
    default: dict[str, Any] = {}

    @field_validator("concepts", mode="before")
    @classmethod
    def ensure_concepts_dict(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        return dict(v)


def validate_corpus_config(
    config: dict[str, Any], corpus_id: str
) -> CorpusConfigSchema | None:
    """Validate a corpus config against the schema.

    Returns the validated config or None if validation fails.
    Logs detailed error messages for malformed configs.
    """
    try:
        return CorpusConfigSchema.model_validate(config)
    except ValidationError as e:
        logger.warning(
            "Invalid config for corpus '%s': %s",
            corpus_id,
            e.errors(),
        )
        return None


@functools.lru_cache(maxsize=1)
def load_routing_config() -> dict[str, Any]:
    """Load routing configuration from config/settings.yaml routing section."""
    settings = get_settings_yaml()
    return settings.get("routing", {})


@functools.lru_cache(maxsize=16)
def load_corpus_config(corpus_id: str) -> dict[str, Any]:
    """Load and validate concept configuration for a specific corpus.

    Args:
        corpus_id: The corpus identifier (e.g., 'ai-act', 'gdpr', 'dora')

    Returns:
        Validated configuration dict, or empty dict if file missing/invalid.
    """
    # Skip template files
    if corpus_id.startswith("_"):
        return {}

    config_path = _get_concepts_dir() / f"{corpus_id}.yaml"
    if not config_path.exists():
        return {}

    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    # Validate against schema
    validated = validate_corpus_config(raw_config, corpus_id)
    if validated is None:
        logger.error(
            "Config validation failed for '%s'. Using empty config. "
            "See config/concepts/_template.yaml for correct structure.",
            corpus_id,
        )
        return {}

    return raw_config


def get_concept_keywords(corpus_id: str, concept_name: str) -> tuple[str, ...]:
    """Get keywords for a specific concept in a corpus.

    Args:
        corpus_id: The corpus identifier (e.g., 'ai-act', 'gdpr', 'dora')
        concept_name: The concept name (e.g., 'LOGGING_AND_RECORD_KEEPING')

    Returns:
        Tuple of keyword strings for matching
    """
    config = load_corpus_config(corpus_id)
    concepts = config.get("concepts", {})
    concept_config = concepts.get(concept_name, {})
    keywords = concept_config.get("keywords", [])
    return tuple(keywords)


def get_all_keywords_for_corpus(corpus_id: str) -> dict[str, tuple[str, ...]]:
    """Get all concept keywords for a corpus.

    Returns:
        Dict mapping concept_name -> tuple of keywords
    """
    config = load_corpus_config(corpus_id)
    concepts = config.get("concepts", {})
    return {name: tuple(cfg.get("keywords", [])) for name, cfg in concepts.items()}


def get_all_keywords_merged() -> dict[str, tuple[str, ...]]:
    """Get merged keywords across all corpora (union of keywords per concept).

    This is used for corpus-agnostic concept detection.
    Skips template files (starting with '_').
    """
    merged: dict[str, set[str]] = {}

    for corpus_file in _get_concepts_dir().glob("*.yaml"):
        corpus_id = corpus_file.stem
        # Skip template files
        if corpus_id.startswith("_"):
            continue
        config = load_corpus_config(corpus_id)
        concepts = config.get("concepts", {})

        for concept_name, concept_cfg in concepts.items():
            if concept_name not in merged:
                merged[concept_name] = set()
            merged[concept_name].update(concept_cfg.get("keywords", []))

    return {name: tuple(sorted(kws)) for name, kws in merged.items()}


def get_concept_bump_hints(corpus_id: str, concept_name: str) -> list[str]:
    """Get bump hints for a specific concept in a corpus."""
    config = load_corpus_config(corpus_id)
    concepts = config.get("concepts", {})
    concept_config = concepts.get(concept_name, {})
    return list(concept_config.get("bump_hints", []))


def get_default_bump_hints(corpus_id: str) -> list[str]:
    """Get default bump hints for a corpus (when no specific concept matches)."""
    config = load_corpus_config(corpus_id)
    default_config = config.get("default", {})
    return list(default_config.get("bump_hints", []))


def get_concept_answer_policy(
    corpus_id: str, concept_name: str
) -> dict[str, Any] | None:
    """Get answer policy for a specific concept."""
    config = load_corpus_config(corpus_id)
    concepts = config.get("concepts", {})
    concept_config = concepts.get(concept_name, {})
    return concept_config.get("answer_policy")


def get_concept_normative_guard(
    corpus_id: str, concept_name: str
) -> dict[str, Any] | None:
    """Get normative guard settings for a specific concept."""
    config = load_corpus_config(corpus_id)
    concepts = config.get("concepts", {})
    concept_config = concepts.get(concept_name, {})
    return concept_config.get("normative_guard")


def get_concept_bias_rules(
    corpus_id: str, concept_name: str
) -> dict[str, tuple[str, ...]]:
    """Get bias rules (toc_contains, heading_contains) for a specific concept.

    Returns:
        Dict with 'toc_contains' and 'heading_contains' as tuples of keywords.
    """
    config = load_corpus_config(corpus_id)
    concepts = config.get("concepts", {})
    concept_config = concepts.get(concept_name, {})
    return {
        "toc_contains": tuple(concept_config.get("toc_contains", [])),
        "heading_contains": tuple(concept_config.get("heading_contains", [])),
    }


def get_all_bias_rules_merged() -> dict[str, dict[str, tuple[str, ...]]]:
    """Get merged bias rules across all corpora (union per concept).

    Skips template files (starting with '_').

    Returns:
        Dict mapping concept_name -> {'toc_contains': tuple, 'heading_contains': tuple}
    """
    merged: dict[str, dict[str, set[str]]] = {}

    for corpus_file in _get_concepts_dir().glob("*.yaml"):
        corpus_id = corpus_file.stem
        # Skip template files
        if corpus_id.startswith("_"):
            continue
        config = load_corpus_config(corpus_id)
        concepts = config.get("concepts", {})

        for concept_name, concept_cfg in concepts.items():
            if concept_name not in merged:
                merged[concept_name] = {
                    "toc_contains": set(),
                    "heading_contains": set(),
                }
            merged[concept_name]["toc_contains"].update(
                concept_cfg.get("toc_contains", [])
            )
            merged[concept_name]["heading_contains"].update(
                concept_cfg.get("heading_contains", [])
            )

    return {
        name: {
            "toc_contains": tuple(sorted(rules["toc_contains"])),
            "heading_contains": tuple(sorted(rules["heading_contains"])),
        }
        for name, rules in merged.items()
    }


def get_detection_settings() -> dict[str, Any]:
    """Get concept detection settings from routing config."""
    routing = load_routing_config()
    return routing.get("detection", {})


def get_eval_settings() -> dict[str, Any]:
    """Get eval settings from settings.yaml."""
    settings = get_settings_yaml()
    return settings.get("eval", {})


def get_sibling_expansion_settings() -> dict[str, Any]:
    """Get sibling expansion settings from settings.yaml."""
    settings = get_settings_yaml()
    rag = settings.get("rag", {})
    sibling = rag.get("sibling_expansion", {})
    return {
        "enabled": sibling.get("enabled", False),
        "max_siblings": sibling.get("max_siblings", 2),
    }


def get_performance_settings() -> dict[str, Any]:
    """Get performance tuning settings from settings.yaml with env overrides.

    Returns dict with keys: max_retrieval_workers, max_llm_concurrency,
    retrieval_timeout_secs, async_enabled, connection_pool_size, keepalive_connections.
    """
    settings = get_settings_yaml()
    perf = settings.get("performance", {})

    return {
        "max_retrieval_workers": int(
            os.getenv("RAG_MAX_RETRIEVAL_WORKERS")
            or perf.get("max_retrieval_workers", 16)
        ),
        "max_llm_concurrency": int(
            os.getenv("RAG_MAX_LLM_CONCURRENCY") or perf.get("max_llm_concurrency", 5)
        ),
        "retrieval_timeout_secs": float(
            os.getenv("RAG_RETRIEVAL_TIMEOUT_SECS")
            or perf.get("retrieval_timeout_secs", 3.0)
        ),
        "async_enabled": _parse_bool_env(
            "RAG_ASYNC_ENABLED",
            perf.get("async_enabled", True),
        ),
        "connection_pool_size": int(perf.get("connection_pool_size", 100)),
        "keepalive_connections": int(perf.get("keepalive_connections", 50)),
    }


def _parse_bool_env(env_key: str, default: bool) -> bool:
    """Parse a boolean from environment variable, falling back to default."""
    val = os.getenv(env_key)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def get_discovery_settings() -> dict[str, Any]:
    """Get discovery settings from settings.yaml."""
    settings = get_settings_yaml()
    return settings.get("discovery", {})


def clear_config_cache() -> None:
    """Clear all cached configurations (useful for testing)."""
    load_settings.cache_clear()
    load_corpus_config.cache_clear()
    load_routing_config.cache_clear()
    get_case_law_settings.cache_clear()
    # Note: get_settings_yaml is not cached (calls _load_settings_yaml which isn't cached)
