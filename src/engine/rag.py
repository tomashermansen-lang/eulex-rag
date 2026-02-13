import json
from pathlib import Path
import os
import re
import subprocess
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import chromadb
from chromadb.errors import InvalidArgumentError
from openai import OpenAI

from .types import ClaimIntent
from .prompt_builder import build_prompt_context, PromptContext
from . import helpers

from ..common.corpus_registry import normalize_alias, normalize_corpus_id
from ..common.corpus_registry import default_registry_path, load_registry
from .corpus_resolver import load_resolver_for_project_root
from .corpus_discovery import discover_corpora, DiscoveryConfig

from .planning import (
    AnswerContext,
    FocusSelection,
    FocusType,
    Intent,
    QueryContext,
    RetrievalPlan,
    UserProfile,
    build_retrieval_plan,
    prepare_answer_context,
    refine_retrieval_plan,
)

from dotenv import load_dotenv

from .engineering_json_mode import (
    EngineeringJSONValidationError,
    bullet_counts as _engineering_json_bullet_counts,
    extract_cited_idxs as _engineering_json_extract_cited_idxs,
    render_engineering_answer_text as _engineering_json_render_text,
    validate_engineering_answer_json as _engineering_json_validate,
    validate_engineering_answer_json_schema_only as _engineering_json_validate_schema_only,
    validate_engineering_answer_json_policy as _engineering_json_validate_policy,
)
from .concept_config import (
    get_effective_policy,
    Policy as AnchorPolicy,
)
from ..common.config_loader import load_settings

from . import citations
from . import policy as policy_engine
from .helpers import _truthy_env
from .constants import contains_normative_claim
# Import from refactored generation modules (SOLID - Fase 12: rag.py is now the ONLY orchestrator)
from .generation_types import GenerationConfig, StructuredGenerationResult
from .generation_strategies import (
    execute_structured_generation,
    execute_citation_retry_if_needed,
    build_engineering_answer,
)
from .prompt_builder import (
    build_prompt,
    build_disclaimer,
    focus_block_for_prompt,
    build_answer_policy_suffix,
    build_citation_requirement_suffix,
    build_multi_corpus_prompt,
)
from .synthesis_router import detect_synthesis_mode, SynthesisMode
from .summary_generation import generate_chapter_summary_from_chunks
from .llm_client import call_llm
from . import instrumentation
from . import indexing
from .types import RAGEngineError, ClaimIntent, EvidenceType, LegalClaimGateResult
from .retrieval import (
    Retriever,
    RetrievalPassTracker,
)
from .ranking import Ranker, execute_ranking_pipeline
from .retrieval_pipeline import (
    PipelineConfig,
    PipelineInput,
    PipelineResult,
    RetrievedChunk,
    SelectedChunk,
    ScoredChunk,
    execute_pipeline,
)
from .multi_corpus_retrieval import (
    MultiCorpusInput,
    MultiCorpusConfig,
    MultiCorpusResult,
    execute_multi_corpus_retrieval,
)

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")




@dataclass
class _RetrievalResult:
    """Result from retrieval stage (modular pipeline)."""
    hits: List[Tuple[str, Dict[str, Any]]]
    distances: List[float]
    retrieved_ids: List[str]
    retrieved_metas: List[Dict[str, Any]]
    run_meta_updates: Dict[str, Any]  # Updates to merge into run_meta
    # New fields for prompt building
    selected_chunks: Tuple[Any, ...] = ()  # Tuple[SelectedChunk, ...]
    total_retrieved: int = 0  # Total chunks before citable filtering
    citable_count: int = 0  # Citable chunks before cap


# ---------------------------------------------------------------------------
# Config helpers - all defaults come from config/settings.yaml
# ---------------------------------------------------------------------------

def _get_openai_settings() -> dict:
    """Get OpenAI settings from config. Environment variables can override."""
    from ..common.config_loader import get_settings_yaml
    settings = get_settings_yaml()
    return settings.get("openai", {})


def _get_model_capabilities() -> dict:
    """Get model capabilities from config."""
    from ..common.config_loader import get_settings_yaml
    settings = get_settings_yaml()
    return settings.get("model_capabilities", {})


def _get_default_chat_model() -> str:
    """Get default chat model from config. Env OPENAI_CHAT_MODEL can override."""
    openai_settings = _get_openai_settings()
    return os.getenv("OPENAI_CHAT_MODEL") or openai_settings.get("chat_model")


def _get_default_embedding_model() -> str:
    """Get default embedding model from config. Env OPENAI_EMBEDDING_MODEL can override."""
    openai_settings = _get_openai_settings()
    return os.getenv("OPENAI_EMBEDDING_MODEL") or openai_settings.get("embedding_model")


def _get_default_temperature() -> float:
    """Get default temperature from config. Env RAG_OPENAI_TEMPERATURE can override."""
    openai_settings = _get_openai_settings()
    temp = os.getenv("RAG_OPENAI_TEMPERATURE")
    if temp is not None:
        return float(temp)
    return float(openai_settings.get("temperature"))


def _get_rag_settings() -> dict:
    """Get RAG pipeline settings from config."""
    from ..common.config_loader import get_settings_yaml
    settings = get_settings_yaml()
    return settings.get("rag", {})



# Static helpers delegated to helpers.py (Phase F deduplication)
_derive_structural_fields_from_location_id = helpers._derive_structural_fields_from_location_id
_extract_raw_anchors_from_chunk = helpers._extract_raw_anchors_from_chunk


def _sync_json_mode_results_to_run_meta(
    gen_result: StructuredGenerationResult,
    run_meta: dict[str, Any],
    allowed_idxs: set[int],
    contract_min_citations: int | None,
    answer_policy: Any | None = None,
) -> None:
    """Sync JSON mode generation result to run_meta for backwards compatibility.

    This function consolidates the metadata bookkeeping that was previously
    inline in answer_structured() (lines 1914-1989).

    Args:
        gen_result: The structured generation result.
        run_meta: The run metadata dict to update (mutated in-place).
        allowed_idxs: Set of allowed citation indices.
        contract_min_citations: Minimum citations required by contract.
        answer_policy: Optional answer policy with min_section3_bullets etc.
    """
    # Common fields for all modes
    run_meta["llm_calls_count"] = gen_result.debug.get("llm_calls_count", 0)
    run_meta["citations_source"] = gen_result.debug.get("citations_source", "text_parse")

    # Initialize engineering_json block
    run_meta.setdefault(
        "engineering_json",
        {
            "enabled": True,
            "json_parse_ok": None,
            "repair_retry_performed": False,
            "enrich_retry_performed": False,
            "allowed_idxs": [],
            "cited_idxs": [],
            "valid_cited": [],
            "min_citations": contract_min_citations or 0,
            "fail_reason": None,
        },
    )

    # Sync allowed indices
    sorted_allowed = sorted(allowed_idxs)
    run_meta["allowed_idxs"] = sorted_allowed
    run_meta["engineering_json"]["allowed_idxs"] = sorted_allowed
    run_meta["allowed_idxs_count"] = len(sorted_allowed)
    run_meta["engineering_json"]["allowed_idxs_count"] = len(sorted_allowed)

    # Sync parse/repair/enrich status
    run_meta["json_parse_ok"] = gen_result.debug.get("json_parse_ok")
    run_meta["engineering_json"]["json_parse_ok"] = gen_result.debug.get("json_parse_ok")

    run_meta["repair_retry_performed"] = gen_result.repair_attempts > 0
    run_meta["engineering_json"]["repair_retry_performed"] = gen_result.repair_attempts > 0

    run_meta["enrich_retry_performed"] = gen_result.enrich_attempts > 0
    run_meta["engineering_json"]["enrich_retry_performed"] = gen_result.enrich_attempts > 0

    run_meta["enrich_success"] = gen_result.debug.get("enrich_success")

    # Sync citation indices
    run_meta["cited_idxs"] = gen_result.cited_idxs
    run_meta["engineering_json"]["cited_idxs"] = gen_result.cited_idxs
    run_meta["cited_idxs_json"] = list(gen_result.cited_idxs)

    run_meta["valid_cited"] = gen_result.valid_cited_idxs
    run_meta["engineering_json"]["valid_cited"] = gen_result.valid_cited_idxs

    # Sync failure reasons
    run_meta["min_citations"] = contract_min_citations or 0
    run_meta["fail_reason"] = gen_result.fail_reason
    run_meta["final_fail_reason"] = gen_result.debug.get("final_fail_reason")
    run_meta["engineering_json"]["fail_reason"] = gen_result.fail_reason

    run_meta["strict_validation_failed_code"] = gen_result.debug.get("strict_validation_failed_code")
    run_meta["schema_only_fallback_used"] = gen_result.debug.get("schema_only_fallback_used", False)

    run_meta["json_policy_enforced"] = True
    run_meta["engineering_json"]["json_policy_enforced"] = True

    # Surface policy knobs
    try:
        run_meta["policy_min_section3_bullets"] = getattr(answer_policy, "min_section3_bullets", None) if answer_policy is not None else None
        run_meta["policy_include_audit_evidence"] = bool(getattr(answer_policy, "include_audit_evidence", False)) if answer_policy is not None else False
    except Exception:  # noqa: BLE001
        run_meta["policy_min_section3_bullets"] = None
        run_meta["policy_include_audit_evidence"] = False

    # Sync bullet counts
    run_meta["requirements_bullet_count"] = gen_result.debug.get("requirements_bullet_count", 0)
    run_meta["audit_evidence_bullet_count"] = gen_result.debug.get("audit_evidence_bullet_count", 0)
    run_meta["engineering_json"]["requirements_bullet_count"] = gen_result.debug.get("requirements_bullet_count", 0)
    run_meta["engineering_json"]["audit_evidence_bullet_count"] = gen_result.debug.get("audit_evidence_bullet_count", 0)

    # Sync final gate reason if MISSING_REF
    if gen_result.is_missing_ref:
        run_meta["final_gate_reason"] = gen_result.fail_reason or "json_parse_or_schema_fail"

    # Sync rendered text
    if gen_result.parsed_json is not None:
        try:
            run_meta["engineering_json"]["rendered_text"] = str(gen_result.answer_text or "")
        except Exception:  # noqa: BLE001
            pass


class RAGEngine:
    def __init__(
        self,
        docs_path: str,
        *,
        corpus_id: str = "ai_act",
        chunks_collection: str | None = None,
        embedding_model: str | None = None,
        chat_model: str | None = None,
        top_k: int | None = None,
        vector_store_path: str | None = None,
        max_distance: float | None = None,
        hybrid_vec_k: int | None = None,
        ranking_weights: "RankingWeights | None" = None,
    ):
        from ..common.config_loader import RankingWeights

        self.docs_path = docs_path
        self.corpus_id = corpus_id
        self._project_root = self._infer_project_root()
        if vector_store_path:
            self.db_path = Path(vector_store_path).resolve()
        else:
            self.db_path = Path(docs_path).resolve().parent / "vector_store"
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chroma = chromadb.PersistentClient(path=str(self.db_path))

        # Collections are corpus-specific: {corpus_id}_documents
        # Normalize corpus_id for collection naming (replace _ with -)
        corpus_slug = corpus_id.replace("_", "-")
        self.collection_name = chunks_collection or f"{corpus_slug}_documents"
        self.collection = self.chroma.get_or_create_collection(self.collection_name)

        self.embedding_model = embedding_model or _get_default_embedding_model()
        self.chat_model = chat_model or _get_default_chat_model()
        self.top_k = int(top_k) if top_k is not None else int(os.getenv("RAG_TOP_K", "3"))
        max_distance_env = os.getenv("RAG_MAX_DISTANCE")
        if max_distance is not None:
            self.max_distance = float(max_distance)
        elif max_distance_env is not None and max_distance_env.strip() != "":
            self.max_distance = float(max_distance_env)
        else:
            self.max_distance = None

        # Hard max distance - abstains if best result exceeds this threshold
        # Load from settings (which supports env var override)
        from ..common.config_loader import load_settings
        _settings = load_settings()
        self.hard_max_distance = _settings.rag_hard_max_distance

        self._last_distances: list[float] = []
        self._last_query_collection_name: str | None = None
        self._last_query_where: dict[str, Any] | None = None

        # Hybrid retrieval / reranking with 4-factor scoring (always enabled):
        # score = α*vec_sim + β*bm25 + γ*citation + δ*role
        rag_settings = _get_rag_settings()
        self.enable_hybrid_rerank = True  # 4-factor hybrid rerank always enabled

        vec_k_env = os.getenv("RAG_HYBRID_VEC_K", "").strip()
        if hybrid_vec_k is not None:
            self.hybrid_vec_k = int(hybrid_vec_k)
        elif vec_k_env:
            self.hybrid_vec_k = int(vec_k_env)
        else:
            self.hybrid_vec_k = int(rag_settings.get("hybrid_vec_k", 30))

        if self.hybrid_vec_k < 5:
            self.hybrid_vec_k = 5

        # Ranking weights from config or constructor
        if ranking_weights is not None:
            self.ranking_weights = ranking_weights
        else:
            # Load from settings.yaml via config_loader
            from ..common.config_loader import load_settings
            settings = load_settings()
            self.ranking_weights = settings.ranking_weights

        self._retriever = Retriever(
            collection=self.collection,
            embedding_model=self.embedding_model,
        )
        self._ranker = Ranker()

    @property
    def retriever(self) -> Retriever:
        # Lazy initialization supports RAGEngine.__new__ usage in tests,
        # where __init__ is bypassed but properties are accessed.
        if not hasattr(self, "_retriever"):
            self._retriever = Retriever(
                collection=getattr(self, "collection", None),
                embedding_model=getattr(self, "embedding_model", _get_default_embedding_model()),
            )
        return self._retriever

    @property
    def ranker(self) -> Ranker:
        # Lazy initialization supports RAGEngine.__new__ usage in tests.
        if not hasattr(self, "_ranker"):
            self._ranker = Ranker()
        return self._ranker

    @property
    def _last_retrieved_ids(self) -> list[str]:
        return self.retriever._last_retrieved_ids

    @_last_retrieved_ids.setter
    def _last_retrieved_ids(self, value: list[str]):
        self.retriever._last_retrieved_ids = value

    @property
    def _last_retrieved_metadatas(self) -> list[dict[str, Any]]:
        return self.retriever._last_retrieved_metadatas

    @_last_retrieved_metadatas.setter
    def _last_retrieved_metadatas(self, value: list[dict[str, Any]]):
        self.retriever._last_retrieved_metadatas = value

    @property
    def _last_effective_where(self) -> dict[str, Any] | None:
        return self.retriever._last_effective_where

    @_last_effective_where.setter
    def _last_effective_where(self, value: dict[str, Any] | None):
        self.retriever._last_effective_where = value

    @property
    def _last_effective_collection_name(self) -> str | None:
        return self.retriever._last_effective_collection_name

    @_last_effective_collection_name.setter
    def _last_effective_collection_name(self, value: str | None):
        self.retriever._last_effective_collection_name = value

    @property
    def _last_effective_collection_type(self) -> str | None:
        return self.retriever._last_effective_collection_type

    @_last_effective_collection_type.setter
    def _last_effective_collection_type(self, value: str | None):
        self.retriever._last_effective_collection_type = value

    @property
    def _last_sibling_expansion(self) -> dict[str, Any]:
        return getattr(self.retriever, "_last_sibling_expansion", {})

    def _infer_project_root(self) -> Path:
        """Best-effort repo root discovery.

        Prefer a parent directory that contains data/processed/corpus_registry.json.
        Falls back to the parent of docs_path.
        """

        try:
            docs = Path(self.docs_path).resolve()
        except Exception:  # noqa: BLE001
            return Path.cwd()

        for p in [docs] + list(docs.parents)[:6]:
            candidate = p
            try:
                # docs_path is typically <root>/data/sample_docs
                if candidate.name == "sample_docs" and candidate.parent.name == "data":
                    candidate = candidate.parent.parent
            except Exception:  # noqa: BLE001
                pass
            if (candidate / "data" / "processed" / "corpus_registry.json").exists():
                return candidate

        # Heuristic fallback.
        try:
            return docs.parent.parent
        except Exception:  # noqa: BLE001
            return docs.parent

    def _resolver(self):
        project_root = getattr(self, "_project_root", None)
        if not isinstance(project_root, Path):
            try:
                docs_path = getattr(self, "docs_path", None)
                if docs_path:
                    project_root = self._infer_project_root()
                else:
                    project_root = Path(__file__).resolve().parents[2]
            except Exception:  # noqa: BLE001
                project_root = Path(__file__).resolve().parents[2]
            try:
                self._project_root = project_root
            except Exception:  # noqa: BLE001
                pass

        return load_resolver_for_project_root(str(project_root))

    def _best_effort_source_label(self, meta: Dict[str, Any] | None) -> str:
        m = dict(meta or {})
        src = str(m.get("source") or "").strip()
        if src:
            return src

        cid = str(m.get("corpus_id") or "").strip() or str(getattr(self, "corpus_id", "") or "").strip()
        try:
            dn = self._resolver().display_name_for(cid)
        except Exception:  # noqa: BLE001
            dn = None
        return dn or cid or "Unknown source"

    # Phase 2 refactoring: anchor_score and rank_retrieved_chunks_anchor_aware moved to Ranker
    # Thin wrappers preserved for backward compatibility with existing call sites

    # NOTE: anchor_score, rank_retrieved_chunks_anchor_aware, _select_references_used_in_answer
    # removed in Phase 8 - use Ranker.* and citations.* directly

    @staticmethod
    def _iso_utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _best_effort_git_commit_short() -> str | None:
        env = str(os.getenv("GIT_COMMIT", "") or "").strip()
        if env:
            return env[:12]
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            return out.decode("utf-8", errors="ignore").strip() or None
        except Exception:  # noqa: BLE001
            return None

    def _collection_name_best_effort(self, collection: Any) -> str | None:
        try:
            if collection is getattr(self, "collection", None):
                return str(getattr(self, "collection_name", None) or "") or None
        except Exception:  # noqa: BLE001
            pass
        name = getattr(collection, "name", None)
        try:
            return str(name) if name else None
        except Exception:  # noqa: BLE001
            return None

    def _maybe_log_intent_event(self, *, question: str, intent: ClaimIntent, profile: UserProfile | str) -> None:
        """Delegate to instrumentation.log_intent_event."""
        instrumentation.log_intent_event(
            question=question,
            intent=intent,
            profile=profile,
            corpus_id=str(getattr(self, "corpus_id", "") or ""),
            project_root=getattr(self, "_project_root", None),
        )

    # Removed: _is_citable_metadata, _extract_precise_ref_from_text, _is_citable_chunk
    # These are now called directly from citations module

    def load_documents(self):
        """Load .txt files into vectorstore. Delegates to indexing module."""
        return indexing.index_documents(self)

    def ingest_jsonl(self, jsonl_path: str, batch_size: int = 32):
        """Ingest JSONL chunks into vectorstore. Delegates to indexing module."""
        return indexing.index_jsonl(self, jsonl_path, batch_size)

    # Removed: _upsert_with_embeddings, _upsert_with_embeddings_to, _reset_collection
    # These methods have been moved to src/engine/indexing.py (Phase B)
    # Removed: _looks_like_structure_question, _looks_like_substantive_question
    # These were thin wrappers - call helpers._looks_like_*() directly (Phase 1 refactoring)

    def _should_abstain(
        self,
        question: str,
        hits: list[tuple[str, dict[str, Any]]],
        distances: list[float] | None = None,
        *,
        allow_low_evidence_answer: bool = False,
        references_structured: list[dict[str, Any]] | None = None,
        corpus_scope: str = "single",
    ) -> str | None:
        """Delegate to policy_engine.should_abstain(). See policy.py for implementation.

        This method gathers instance attributes and passes them as explicit parameters.
        """
        # Get resolver (may be None if not available)
        resolver = None
        try:
            resolver = self._resolver()
        except Exception:  # noqa: BLE001
            pass

        # Use instance distances if not provided
        used_distances = distances if distances is not None else getattr(self, "_last_distances", [])

        return policy_engine.should_abstain(
            question=question,
            hits=hits,
            distances=used_distances,
            corpus_id=str(getattr(self, "corpus_id", "") or "").strip(),
            resolver=resolver,
            max_distance=getattr(self, "max_distance", None),
            hard_max_distance=getattr(self, "hard_max_distance", None),
            allow_low_evidence_answer=allow_low_evidence_answer,
            references_structured=references_structured,
            corpus_scope=corpus_scope,
        )

    # Removed in Phase 1 refactoring - call helpers directly:
    # _extract_article_ref, _extract_article_refs, _looks_like_multi_part_question,
    # _extract_annex_refs, _extract_recital_ref, _looks_like_recital_quote_question,
    # _extract_chapter_ref, _normalize_abstain_text, _extract_section_ref, _roman_to_int
    # _looks_like_chapter_overview_question, _looks_like_chapter_summary_question -> helpers.py

    def _answer_chapter_summary_from_chunks(self, question: str) -> dict[str, Any] | None:
        if not helpers._looks_like_chapter_summary_question(question):
            return None

        chapter_ref = helpers._extract_chapter_ref(question)
        if not chapter_ref:
            return None

        # Use chapter_ref directly (uppercase)
        canonical = chapter_ref.upper()

        # Retrieve chunks scoped to this chapter.
        try:
            k = max(8, int(getattr(self, "top_k", 3)) * 4)
        except Exception:  # noqa: BLE001
            k = 12

        try:
            hits, distances = self._query_collection_with_distances(
                collection=self.collection,
                question=f"Sammenfat Kapitel {canonical}.",
                k=min(20, k),
                where={"chapter": canonical},
            )
        except Exception:  # noqa: BLE001
            return None

        if not hits:
            return None

        references: List[str] = []
        context_blocks: List[str] = []
        references_structured: list[dict[str, Any]] = []

        filtered_hits: list[tuple[str, dict[str, Any], str | None]] = []
        for doc, metadata in hits:
            meta_dict = dict(metadata or {})
            doc_str = str(doc or "")
            
            if citations._is_citable_metadata(meta_dict):
                filtered_hits.append((doc_str, meta_dict, None))
                continue
                
            extracted = citations.extract_precise_ref_from_text(doc_str)
            if extracted:
                src = self._best_effort_source_label(meta_dict)
                filtered_hits.append((doc_str, meta_dict, f"{src}, {extracted}"))

        if not filtered_hits:
            return None

        for idx, (doc, metadata, precise_override) in enumerate(filtered_hits, start=1):
            display = citations._format_metadata(metadata)
            # Determine the most precise token available: metadata -> text extract -> missing
            precise = precise_override or citations.extract_precise_token_from_meta(dict(metadata or {}))
            missing = False
            if not precise:
                # Try extracting from the chunk text
                extracted = citations.extract_precise_ref_from_text(doc or "")
                if extracted:
                    src = citations.best_effort_source_label(dict(metadata or {}), fallback=self._best_effort_source_label(dict(metadata or {})))
                    precise = f"{src}, {extracted}"
                else:
                    # Mark as missing precision
                    src = citations.best_effort_source_label(dict(metadata or {}), fallback=self._best_effort_source_label(dict(metadata or {})))
                    precise = f"MISSING_REF — {src} (kilden mangler artikel/bilag i metadata og tekst)"
                    missing = True

            references.append(f"[{idx}] {precise}")
            context_blocks.append(f"[{idx}] {display}\n{doc}")

            references_structured.append(
                {
                    "idx": idx,
                    "chunk_id": getattr(metadata or {}, "get", lambda k, d=None: None)("chunk_id", None) or f"hit-{idx}",
                    "display": display,
                    "precise_ref": precise,
                    "missing_ref": missing,
                    "source": (metadata or {}).get("source"),
                    **dict(metadata or {}),
                }
            )

        context = "\n\n".join(context_blocks)

        # For chapter summaries we intentionally do not apply the distance abstention guardrail;
        # we already constrained retrieval by chapter metadata.
        answer_text = generate_chapter_summary_from_chunks(
            client=self.client,
            model=getattr(self, "chat_model", _get_default_chat_model()),
            context=context,
            question=question,
            chapter_ref=canonical,
            temperature=_get_default_temperature(),
        )

        self._last_distances = list(distances or [])
        self._last_query_collection_name = getattr(self, "collection_name", None)
        self._last_query_where = {"chapter": canonical}

        return {
            "answer": answer_text,
            "references": references,
            "retrieval": {
                "distances": list(distances or []),
                "query_collection": getattr(self, "collection_name", None),
                "query_where": {"chapter": canonical},
                "retrieved_ids": list(getattr(self, "_last_retrieved_ids", []) or []),
                "retrieved_metadatas": list(getattr(self, "_last_retrieved_metadatas", []) or []),
                "hybrid_rerank": {
                    "enabled": bool(getattr(self, "enable_hybrid_rerank", True)),
                    "vec_k": int(getattr(self, "hybrid_vec_k", 30)),
                    "weights": {
                        "alpha_vec": getattr(self.ranking_weights, "alpha_vec", 0.25) if hasattr(self, "ranking_weights") else 0.25,
                        "beta_bm25": getattr(self.ranking_weights, "beta_bm25", 0.25) if hasattr(self, "ranking_weights") else 0.25,
                        "gamma_cite": getattr(self.ranking_weights, "gamma_cite", 0.35) if hasattr(self, "ranking_weights") else 0.35,
                        "delta_role": getattr(self.ranking_weights, "delta_role", 0.15) if hasattr(self, "ranking_weights") else 0.15,
                    },
                },
            },
        }

    def _query_collection_with_distances(
        self,
        *,
        collection: Any,
        question: str,
        k: int,
        where: Dict[str, Any] | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[float]]:
        return self.retriever._query_collection_with_distances(
            collection=collection, question=question, k=k, where=where
        )

    @staticmethod
    def _normalize_chroma_where(where: Dict[str, Any] | None) -> Dict[str, Any] | None:
        """Normalize a where-clause to a single top-level operator for Chroma.

        Some Chroma versions require `where` to contain exactly one operator at the top level.
        This helper rewrites multi-key dicts into a deterministic `$and`.

        Examples:
        - {"article": "6"} -> unchanged
        - {"corpus_id": "ai-act", "article": "6"} -> {"$and": [{"corpus_id": "ai-act"}, {"article": "6"}]}
        - {"corpus_id": "ai-act", "$or": [{"article": "2"}, {"article": "3"}]} ->
            {"$and": [{"corpus_id": "ai-act"}, {"$or": [...]}]}
        """
        return Retriever._normalize_chroma_where(where)

    def _query_collection_raw(
        self,
        *,
        collection: Any,
        question: str,
        k: int,
        where: Dict[str, Any] | None = None,
        track_state: bool = True,
    ) -> tuple[list[str], list[str], list[dict[str, Any]], list[float]]:
        return self.retriever._query_collection_raw(
            collection=collection, question=question, k=k, where=where, track_state=track_state
        )

    def _hybrid_rerank_hits(
        self,
        *,
        question: str,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        distances: list[float],
        k: int,
        hint_anchors: set[str] | None = None,
        citation_boost: float = 0.0,
        user_profile: "UserProfile | None" = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[float], list[str]]:
        return self.ranker._hybrid_rerank_hits(
            question=question,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            k=k,
            hint_anchors=hint_anchors,
            citation_boost=citation_boost,
            weights=getattr(self, "ranking_weights", None),
            user_profile=user_profile,
        )

    def run_retrieval_pipeline(
        self,
        question: str,
        user_profile: "UserProfile | str | None" = None,
        where_filter: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Run the new modular retrieval pipeline (standalone).

        This is a standalone method for testing the new pipeline without
        affecting the existing answer_structured flow.

        The pipeline has 4 stages:
            1. VectorRetrieval -> Initial embedding search
            2. CitationExpansion -> Graph-based article discovery + chunk injection
            3. HybridRerank -> 4-factor scoring (vec + bm25 + citation + role)
            4. ContextSelection -> Citable filter + diversity + top-k cap

        Args:
            question: User's question
            user_profile: LEGAL or ENGINEERING profile
            where_filter: Optional retrieval filter

        Returns:
            Dict with pipeline results and debug info
        """
        from .citations import _is_citable_chunk

        # Resolve profile
        resolved_profile = self._normalize_user_profile(user_profile)

        # Load config
        corpus_id = str(getattr(self, "corpus_id", "") or "").strip()
        config = PipelineConfig.from_settings(corpus_id=corpus_id)

        # Wrap _query_collection_raw as query_fn
        def query_fn(
            q: str, k: int, where: Dict[str, Any] | None
        ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[float]]:
            """Query function adapter for pipeline."""
            ids, docs, metas, dists = self._query_collection_raw(
                collection=self.collection,
                question=q,
                k=k,
                where=where,
            )
            hits = list(zip(ids, docs, metas, strict=False))
            return hits, dists

        # Wrap anchor injection as inject_fn
        def inject_fn(
            q: str, anchor: str, k: int
        ) -> List[Tuple[str, str, Dict[str, Any], float]]:
            """Inject function for fetching chunks by anchor."""
            if ":" not in anchor:
                return []
            kind, value = anchor.split(":", 1)
            kind = kind.strip().lower()
            value = value.strip()
            if kind not in {"article", "recital", "annex"}:
                return []

            where_hint: Dict[str, Any] = {"corpus_id": corpus_id}
            if kind == "annex" and ":" in value:
                # Handle annex:iii:5 format
                parts = value.split(":")
                where_hint["annex"] = helpers.normalize_annex_for_chroma(parts[0])
                if len(parts) > 1:
                    where_hint["annex_point"] = parts[1]
            elif kind == "annex":
                where_hint["annex"] = helpers.normalize_annex_for_chroma(value)
            else:
                where_hint[kind] = value

            # track_state=False: Don't update _last_effective_where for injection queries
            ids, docs, metas, dists = self._query_collection_raw(
                collection=self.collection,
                question=q,
                k=k,
                where=where_hint,
                track_state=False,
            )
            return list(zip(ids, docs, metas, dists, strict=False))

        # Build input
        profile_str = (
            "ENGINEERING" if resolved_profile == UserProfile.ENGINEERING else "LEGAL"
        )
        pipeline_input = PipelineInput(
            question=question,
            corpus_id=corpus_id,
            user_profile=profile_str,
            where_filter=where_filter,
        )

        # Execute pipeline
        result = execute_pipeline(
            input=pipeline_input,
            config=config,
            query_fn=query_fn,
            inject_fn=inject_fn,
            is_citable_fn=_is_citable_chunk,
        )

        # Return in a format compatible with existing tools
        return {
            "hits": result.get_hits(),
            "distances": result.get_distances(),
            "ids": result.get_ids(),
            "metadatas": result.get_metadatas(),
            "pipeline_debug": result.debug_summary,
            "context_selection": {
                "citable_count": result.context_result.citable_count,
                "unique_anchors": list(result.context_result.unique_anchors),
            },
            "expansion": {
                "hint_anchors": list(result.expansion_result.hint_anchors),
                "chunks_injected": result.expansion_result.chunks_injected,
            },
            "rerank": {
                "query_intent": result.rerank_result.query_intent,
                "top_10": [
                    {
                        "anchor": sc.chunk.anchor_key(),
                        "final_score": round(sc.final_score, 3),
                    }
                    for sc in result.rerank_result.scored_chunks[:10]
                ],
            },
        }

    def _modular_retrieval(
        self,
        question: str,
        resolved_profile: "UserProfile",
        where_for_retrieval: Dict[str, Any] | None,
        ctx: "QueryContext",
    ) -> _RetrievalResult:
        """Run the new modular retrieval pipeline.

        This replaces the legacy retrieval/rerank spaghetti with a clean 4-stage pipeline.
        """
        from .citations import _is_citable_chunk

        # Load config
        corpus_id = str(ctx.corpus_id)
        config = PipelineConfig.from_settings(corpus_id=corpus_id)

        # Create query function adapter
        def query_fn(
            q: str, k: int, where: Dict[str, Any] | None
        ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[float]]:
            ids, docs, metas, dists = self._query_collection_raw(
                collection=self.collection,
                question=q,
                k=k,
                where=where,
            )
            hits = list(zip(ids, docs, metas, strict=False))
            return hits, dists

        # Create inject function for citation expansion
        def inject_fn(
            q: str, anchor: str, k: int
        ) -> List[Tuple[str, str, Dict[str, Any], float]]:
            if ":" not in anchor:
                return []
            kind, value = anchor.split(":", 1)
            kind = kind.strip().lower()
            value = value.strip()
            if kind not in {"article", "recital", "annex"}:
                return []

            where_hint: Dict[str, Any] = {"corpus_id": corpus_id}
            if kind == "annex" and ":" in value:
                # Handle annex:iii:5 format
                parts = value.split(":")
                where_hint["annex"] = helpers.normalize_annex_for_chroma(parts[0])
                if len(parts) > 1:
                    where_hint["annex_point"] = parts[1]
            elif kind == "annex":
                where_hint["annex"] = helpers.normalize_annex_for_chroma(value)
            else:
                where_hint[kind] = value

            # track_state=False: Don't update _last_effective_where for injection queries
            ids, docs, metas, dists = self._query_collection_raw(
                collection=self.collection,
                question=q,
                k=k,
                where=where_hint,
                track_state=False,
            )
            return list(zip(ids, docs, metas, dists, strict=False))

        # Build pipeline input
        profile_str = "ENGINEERING" if resolved_profile == UserProfile.ENGINEERING else "LEGAL"
        pipeline_input = PipelineInput(
            question=question,
            corpus_id=corpus_id,
            user_profile=profile_str,
            where_filter=where_for_retrieval,
        )

        # Execute the modular pipeline
        pipeline_result = execute_pipeline(
            input=pipeline_input,
            config=config,
            query_fn=query_fn,
            inject_fn=inject_fn,
            is_citable_fn=_is_citable_chunk,
        )

        # Extract results
        hits = pipeline_result.get_hits()
        distances = pipeline_result.get_distances()
        retrieved_ids = pipeline_result.get_ids()
        retrieved_metas = pipeline_result.get_metadatas()

        # Update internal state for compatibility with downstream code
        self._last_retrieved_ids = list(retrieved_ids)
        self._last_retrieved_metadatas = list(retrieved_metas)
        self._last_distances = list(distances)

        # Build run_meta updates
        run_meta_updates = {
            "modular_pipeline": pipeline_result.debug_summary,
            "anchor_hints": {
                "hint_anchors": list(pipeline_result.expansion_result.hint_anchors),
                "citation_expansion_articles": list(pipeline_result.expansion_result.hint_anchors),
                "anchor_hints_applied": True,
                "injected_added": pipeline_result.expansion_result.chunks_injected,
                "citation_source": "modular_pipeline",
            },
            "hybrid_rerank": {
                "enabled": True,
                "query_intent": pipeline_result.rerank_result.query_intent,
                "top_scores": [
                    {
                        "anchor": sc.chunk.anchor_key(),
                        "vec": round(sc.vec_score, 3),
                        "bm25": round(sc.bm25_score, 3),
                        "cite": round(sc.citation_score, 3),
                        "role": round(sc.role_score, 3),
                        "final": round(sc.final_score, 3),
                    }
                    for sc in pipeline_result.rerank_result.scored_chunks[:10]
                ],
            },
        }

        return _RetrievalResult(
            hits=list(hits),
            distances=list(distances),
            retrieved_ids=list(retrieved_ids),
            retrieved_metas=list(retrieved_metas),
            run_meta_updates=run_meta_updates,
            selected_chunks=pipeline_result.context_result.selected,
            total_retrieved=len(pipeline_result.vector_result.chunks),
            citable_count=pipeline_result.context_result.citable_count,
        )

    def _available_corpora(self) -> List[str]:
        """Get list of all available corpus IDs from the config.

        Returns config keys (e.g. 'ai-act') which match the corpus_id values
        stored in ChromaDB document metadata. The registry uses normalized
        (underscore) keys which differ from the indexed format.
        """
        settings = load_settings()
        if settings.corpora:
            return list(settings.corpora.keys())
        # Fallback to registry if no config
        project_root = Path(__file__).resolve().parents[2]
        registry_path = default_registry_path(project_root)
        registry = load_registry(registry_path)
        return list(registry.keys())

    def _execute_cross_law_retrieval(
        self,
        question: str,
        corpus_ids: Tuple[str, ...],
        resolved_profile: "UserProfile",
        where_filter: Dict[str, Any] | None,
    ) -> _RetrievalResult:
        """Execute multi-corpus retrieval with RRF fusion.

        Coordinates parallel retrieval across multiple corpora and fuses
        results using Reciprocal Rank Fusion.

        Args:
            question: The question to answer
            corpus_ids: Tuple of corpus IDs to query
            resolved_profile: User profile for context sizing
            where_filter: Optional where filter for retrieval

        Returns:
            _RetrievalResult with fused chunks from all corpora
        """
        from .citations import _is_citable_chunk

        # Build input for multi-corpus retrieval
        multi_input = MultiCorpusInput(
            question=question,
            corpus_ids=corpus_ids,
            user_profile=str(resolved_profile.value) if hasattr(resolved_profile, "value") else str(resolved_profile),
            where_filter=where_filter,
        )

        # Use default config
        multi_config = MultiCorpusConfig()

        # Create factory functions for each corpus
        def query_fn_factory(corpus_id: str):
            """Create query function for a specific corpus."""
            collection = self._get_collection_for_corpus(corpus_id)

            def query_fn(
                q: str, k: int, where: Dict[str, Any] | None
            ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[float]]:
                # Build where filter with corpus_id
                effective_where = dict(where or {})
                effective_where["corpus_id"] = corpus_id

                ids, docs, metas, dists = self._query_collection_raw(
                    collection=collection,
                    question=q,
                    k=k,
                    where=effective_where,
                )
                hits = list(zip(ids, docs, metas, strict=False))
                return hits, dists

            return query_fn

        def inject_fn_factory(corpus_id: str):
            """Create inject function for citation expansion for a specific corpus."""
            collection = self._get_collection_for_corpus(corpus_id)

            def inject_fn(
                q: str, anchor: str, k: int
            ) -> List[Tuple[str, str, Dict[str, Any], float]]:
                if ":" not in anchor:
                    return []
                kind, value = anchor.split(":", 1)
                kind = kind.strip().lower()
                value = value.strip()
                if kind not in {"article", "recital", "annex"}:
                    return []

                where_hint: Dict[str, Any] = {"corpus_id": corpus_id}
                if kind == "annex" and ":" in value:
                    parts = value.split(":")
                    where_hint["annex"] = helpers.normalize_annex_for_chroma(parts[0])
                    if len(parts) > 1:
                        where_hint["annex_point"] = parts[1]
                elif kind == "annex":
                    where_hint["annex"] = helpers.normalize_annex_for_chroma(value)
                else:
                    where_hint[kind] = value

                ids, docs, metas, dists = self._query_collection_raw(
                    collection=collection,
                    question=q,
                    k=k,
                    where=where_hint,
                )
                return list(zip(ids, docs, metas, dists, strict=False))

            return inject_fn

        # Execute multi-corpus retrieval
        multi_result = execute_multi_corpus_retrieval(
            input=multi_input,
            config=multi_config,
            query_fn_factory=query_fn_factory,
            inject_fn_factory=inject_fn_factory,
            is_citable_fn=_is_citable_chunk,
        )

        # Convert MultiCorpusResult to _RetrievalResult format
        return self._convert_multi_corpus_result(multi_result, resolved_profile)

    def _get_collection_for_corpus(self, corpus_id: str):
        """Get ChromaDB collection for a specific corpus.

        Each corpus has its own collection named {corpus_id}_documents.
        """
        collection_name = f"{corpus_id}_documents"
        return self.chroma.get_or_create_collection(collection_name)

    def _convert_multi_corpus_result(
        self,
        multi_result: MultiCorpusResult,
        resolved_profile: "UserProfile",
    ) -> _RetrievalResult:
        """Convert MultiCorpusResult to _RetrievalResult format.

        Converts the fused chunks to the format expected by downstream code.

        Args:
            multi_result: Result from multi-corpus retrieval
            resolved_profile: User profile for context sizing

        Returns:
            _RetrievalResult with hits, distances, and selected chunks
        """
        # Extract hits and distances from fused chunks
        hits: List[Tuple[str, Dict[str, Any]]] = []
        distances: List[float] = []
        retrieved_ids: List[str] = []
        retrieved_metas: List[Dict[str, Any]] = []
        selected_chunks: List[SelectedChunk] = []

        for i, scored_chunk in enumerate(multi_result.fused_chunks):
            chunk = scored_chunk.chunk
            meta = dict(chunk.metadata)

            hits.append((chunk.document, meta))
            distances.append(chunk.distance)
            retrieved_ids.append(chunk.chunk_id)
            retrieved_metas.append(meta)

            # Create SelectedChunk for prompt building
            # Determine citability from metadata
            is_citable = bool(
                meta.get("article")
                or meta.get("recital")
                or meta.get("annex")
            )
            selected_chunks.append(
                SelectedChunk(
                    chunk=chunk,
                    is_citable=is_citable,
                    precise_ref=None,
                    rank=i,
                )
            )

        # Determine context cap based on profile
        settings = load_settings()
        context_cap = (
            settings.max_context_legal
            if resolved_profile == UserProfile.LEGAL
            else settings.max_context_engineering
        )

        # Apply context cap to selected chunks
        capped_selected = tuple(selected_chunks[:context_cap])

        # Build run_meta updates
        run_meta_updates = {
            "multi_corpus_retrieval": {
                "enabled": True,
                "per_corpus_hits": multi_result.per_corpus_hits,
                "duration_ms": round(multi_result.duration_ms, 2),
                "fused_count": len(multi_result.fused_chunks),
            },
            "laws_searched": list(multi_result.per_corpus_hits.keys()),
        }

        return _RetrievalResult(
            hits=hits,
            distances=distances,
            retrieved_ids=retrieved_ids,
            retrieved_metas=retrieved_metas,
            run_meta_updates=run_meta_updates,
            selected_chunks=capped_selected,
            total_retrieved=len(multi_result.fused_chunks),
            citable_count=sum(1 for sc in selected_chunks if sc.is_citable),
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.retriever._embed(texts)

    def query(self, question: str, k: int | None = None):
        return self.query_with_where(question, k=k, where=None)

    def query_with_where(
        self,
        question: str,
        k: int | None = None,
        *,
        where: Dict[str, Any] | None = None,
    ):
        if k is None:
            k = getattr(self, "top_k", 3)

        # Reset last distances for this query.
        self._last_distances = []
        self._last_query_collection_name = None
        self._last_query_where = None
        self._last_retrieved_ids = []
        self._last_retrieved_metadatas = []

        # Optional: if the user references exactly ONE article, try filtering chunks by metadata.
        # IMPORTANT: multi-article prompts (e.g. "Artikel 2/3/6") must NOT be hard-scoped to the first match.
        article_refs = helpers._extract_article_refs(question)
        article = article_refs[0] if len(article_refs) == 1 else None
        if article:
            try:
                where_article = dict(where or {})
                where_article["article"] = article
                filtered, filtered_distances = self._query_collection_with_distances(
                    collection=self.collection, question=question, k=k, where=where_article
                )
                if filtered:
                    self._last_distances = filtered_distances
                    self._last_query_collection_name = getattr(self, "collection_name", None)
                    self._last_query_where = where_article
                    return filtered
            except Exception:  # noqa: BLE001
                pass

        # Default retrieval: embeddings. Optional hybrid reranking over top-N.
        if getattr(self, "enable_hybrid_rerank", True):
            vec_k = max(int(getattr(self, "hybrid_vec_k", 30)), int(k))
            ids, docs, metas, distances = self._query_collection_raw(
                collection=self.collection,
                question=question,
                k=vec_k,
                where=where,
            )
            
            # Apply sibling expansion before hybrid rerank
            from ..common.config_loader import get_sibling_expansion_settings
            sibling_settings = get_sibling_expansion_settings()
            expand_fn = getattr(self.retriever, "_expand_to_siblings", None)
            if sibling_settings.get("enabled", False) and ids and callable(expand_fn):
                original_count = len(ids)
                max_siblings = sibling_settings.get("max_siblings", 2)
                ids, docs, metas, distances = expand_fn(
                    collection=self.collection,
                    ids=ids,
                    documents=docs,
                    metadatas=metas,
                    distances=distances,
                    max_siblings=max_siblings,
                )
                # Track sibling expansion on retriever
                self.retriever._last_sibling_expansion = {
                    "enabled": True,
                    "original_count": original_count,
                    "expanded_count": len(ids),
                    "siblings_added": len(ids) - original_count,
                    "max_siblings": max_siblings,
                }
            else:
                # Track that expansion was disabled
                self.retriever._last_sibling_expansion = {
                    "enabled": False,
                    "original_count": len(ids) if ids else 0,
                    "expanded_count": len(ids) if ids else 0,
                    "siblings_added": 0,
                }
            
            hits, out_distances, out_ids = self._hybrid_rerank_hits(
                question=question,
                ids=ids,
                documents=docs,
                metadatas=metas,
                distances=distances,
                k=k,
            )
            # Post-retrieval filter: prioritize precise hits (have article/annex/chapter in metadata)
            # Delegates to Retriever.split_precise (Phase C extraction)
            p, pd, pid, ip, ipd, ipid = Retriever.split_precise(hits, out_distances, out_ids)
            if len(p) >= int(k):
                selected = p[: int(k)]
                sel_d = pd[: int(k)]
                sel_ids = pid[: int(k)]
            else:
                need = int(k) - len(p)
                selected = p + ip[:need]
                sel_d = pd + ipd[:need]
                sel_ids = pid + ipid[:need]

            self._last_distances = list(sel_d)
            self._last_query_collection_name = getattr(self, "collection_name", None)
            self._last_query_where = where
            self._last_retrieved_ids = list(sel_ids)
            self._last_retrieved_metadatas = [dict(m or {}) for _, m in selected]
            return selected

        hits, distances = self._query_collection_with_distances(
            collection=self.collection,
            question=question,
            k=k,
            where=where,
        )

        # Post-retrieval filter: prioritize precise hits (have article/annex/chapter in metadata)
        # Delegates to Retriever.split_precise_simple (Phase C extraction)
        p, pd, pid, ip, ipd, ipid = Retriever.split_precise_simple(hits, distances)
        if len(p) >= int(k):
            selected = p[: int(k)]
            sel_d = pd[: int(k)]
            sel_ids = pid[: int(k)]
        else:
            need = int(k) - len(p)
            selected = p + ip[:need]
            sel_d = pd + ipd[:need]
            sel_ids = pid + ipid[:need]

        self._last_distances = list(sel_d)
        self._last_query_collection_name = getattr(self, "collection_name", None)
        self._last_query_where = where
        self._last_retrieved_ids = list(sel_ids)
        self._last_retrieved_metadatas = [dict(m or {}) for _, m in selected]
        return selected

    @staticmethod
    def _normalize_user_profile(user_profile: UserProfile | str | None) -> UserProfile:
        if isinstance(user_profile, UserProfile):
            return user_profile
        raw = str(user_profile or "").strip().upper()
        if raw in {"ENGINEERING", "DEV", "DEVELOPER"}:
            return UserProfile.ENGINEERING
        if raw == "LEGAL":
            return UserProfile.LEGAL
        return UserProfile.ANY

    def _retrieve_representative_chapter_chunks(
        self,
        *,
        question: str,
        where: Dict[str, Any],
        top_k: int,
        per_article: int = 2,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[float]]:
        return self.retriever._retrieve_representative_chapter_chunks(
            question=question, where=where, top_k=top_k, per_article=per_article
        )

    def _prepare_for_streaming(
        self,
        question: str,
        user_profile: UserProfile | str | None = None,
    ) -> dict[str, Any]:
        """Prepare retrieval context and prompt for streaming.

        This performs retrieval and prompt building, returning everything
        needed for streaming the LLM response.

        Returns a dict with:
            - prompt: str - The full prompt to send to the LLM
            - references_structured: list[dict] - Reference metadata
            - reference_lines: list[str] - Formatted reference strings
            - retrieval: dict - Retrieval metadata
        """
        from .planning import QueryContext, build_retrieval_plan
        
        if not question.strip():
            raise RAGEngineError("Question cannot be empty.")

        # Normalize user profile
        resolved_profile = self._normalize_user_profile(user_profile)
        
        # Build query context
        ctx = QueryContext(
            corpus_id=str(getattr(self, "corpus_id", "") or "").strip(),
            user_profile=resolved_profile,
            focus=None,  # Simplified for streaming
            top_k=int(getattr(self, "top_k", 3)),
            question=question,
        )
        plan = build_retrieval_plan(ctx)
        
        # Perform retrieval - fetch extra for citation expansion
        where_filter: dict[str, Any] = {"corpus_id": ctx.corpus_id}
        hits, distances = self._query_collection_with_distances(
            collection=self.collection,
            question=question,
            k=ctx.top_k * 2,  # Fetch extra to allow for expansion
            where=where_filter,
        )
        
        # Apply citation expansion to get scope articles/annexes
        expansion_hits: list[tuple[str, dict[str, Any]]] = []
        expansion_distances: list[float] = []
        try:
            from .citation_expansion import get_citation_expansion_for_query, is_citation_expansion_enabled, get_max_expansion
            
            if is_citation_expansion_enabled():
                expansion_articles = get_citation_expansion_for_query(
                    question=question,
                    corpus_id=ctx.corpus_id,
                    retrieved_metadatas=[h[1] for h in hits],
                )
                
                if expansion_articles:
                    # Fetch expansion chunks - use config-based limit
                    max_exp = get_max_expansion()
                    for exp_ref in expansion_articles[:max_exp]:
                        exp_ref_str = str(exp_ref).strip()
                        if exp_ref_str.upper().startswith("ANNEX:"):
                            annex_val = exp_ref_str.split(":", 1)[1] if ":" in exp_ref_str else exp_ref_str
                            exp_where = {"corpus_id": ctx.corpus_id, "annex": helpers.normalize_annex_for_chroma(annex_val)}
                        else:
                            exp_where = {"corpus_id": ctx.corpus_id, "article": exp_ref_str}
                        
                        try:
                            exp_hits_result, exp_dist = self._query_collection_with_distances(
                                collection=self.collection,
                                question=question,
                                k=1,
                                where=exp_where,
                            )
                            if exp_hits_result:
                                # Check if already in original hits
                                exp_meta = exp_hits_result[0][1]
                                exp_art = exp_meta.get("article")
                                exp_ann = exp_meta.get("annex")
                                already_present = False
                                for _, existing_meta in hits:
                                    ex_art = existing_meta.get("article")
                                    ex_ann = existing_meta.get("annex")
                                    # Match on article OR annex (not both being None)
                                    if exp_art and ex_art and exp_art == ex_art:
                                        already_present = True
                                        break
                                    if exp_ann and ex_ann and exp_ann == ex_ann:
                                        already_present = True
                                        break
                                if not already_present:
                                    expansion_hits.append(exp_hits_result[0])
                                    expansion_distances.append(exp_dist[0] if exp_dist else 1.0)
                        except Exception:
                            pass
        except Exception:
            pass  # Citation expansion is optional
        
        # Merge: take first few original hits, then expansion hits, then remaining original hits
        # This ensures expansion hits are included in top_k
        merged_hits: list[tuple[str, dict[str, Any]]] = []
        merged_distances: list[float] = []
        
        # Take first 5 original hits
        for i, h in enumerate(hits[:5]):
            merged_hits.append(h)
            merged_distances.append(distances[i] if i < len(distances) else 1.0)
        
        # Add expansion hits
        for i, h in enumerate(expansion_hits):
            merged_hits.append(h)
            merged_distances.append(expansion_distances[i] if i < len(expansion_distances) else 1.0)
        
        # Fill remaining slots with original hits
        for i, h in enumerate(hits[5:]):
            if len(merged_hits) >= ctx.top_k:
                break
            merged_hits.append(h)
            merged_distances.append(distances[5 + i] if (5 + i) < len(distances) else 1.0)
        
        # Limit to top_k
        hits = merged_hits[:ctx.top_k]
        distances = merged_distances[:ctx.top_k]
        
        # Build context blocks and references (simplified version)
        context_blocks: list[str] = []
        references_structured: list[dict[str, Any]] = []
        reference_lines: list[str] = []
        kilder_lines: list[str] = ["KILDER:"]
        
        # hits is list of (doc, meta) tuples
        for idx, (doc, meta) in enumerate(hits, start=1):
            # Build reference entry
            corpus = str(meta.get("corpus_id") or ctx.corpus_id).strip()
            article = meta.get("article")
            recital = meta.get("recital")
            annex = meta.get("annex")
            
            anchor_parts = []
            if article:
                anchor_parts.append(f"Artikel {article}")
            if recital:
                anchor_parts.append(f"Betragtning {recital}")
            if annex:
                anchor_parts.append(f"Bilag {annex}")
            anchor_label = ", ".join(anchor_parts) if anchor_parts else "Ukendt reference"
            
            display = f"{corpus} / {anchor_label}"
            excerpt = str(doc or "")[:150].strip().replace("\n", " ")
            
            ref = {
                "idx": idx,
                "display": display,
                "corpus_id": corpus,
                "article": article,
                "recital": recital,
                "annex": annex,
                "chunk_text": str(doc or ""),
            }
            references_structured.append(ref)
            reference_lines.append(f"[{idx}] {display}")
            kilder_lines.append(f"- [{idx}] {corpus} / {anchor_label} — {excerpt}")
            
            # Build context block
            context_blocks.append(f"[{idx}] {anchor_label}:\n{doc}")
        
        kilder_block = "\n".join(kilder_lines).strip()
        if kilder_block and context_blocks:
            context = f"{kilder_block}\n\n" + "\n\n".join(context_blocks)
        else:
            context = "\n\n".join(context_blocks) if context_blocks else ""
        
        # Build prompt
        prompt = build_prompt(ctx=ctx, plan=plan, context=context, focus_block="")
        
        return {
            "prompt": prompt,
            "references_structured": references_structured,
            "reference_lines": reference_lines,
            "retrieval": {
                "distances": distances if distances else [],
                "retrieved_ids": [],  # Simplified - no chunk_ids in this format
                "retrieved_metadatas": [h[1] for h in hits] if hits else [],
                "query_collection": getattr(self.collection, "name", None),
                "query_where": where_filter,
            },
        }

    def answer_structured(
        self,
        question: str,
        user_profile: UserProfile | str | None = None,
        contract_min_citations: int | None = None,
        dry_run: bool = False,
        history_context: str = "",
        corpus_scope: str = "single",
        target_corpora: List[str] | None = None,
        original_query: str | None = None,
        last_exchange: list | None = None,
    ) -> dict[str, Any]:
        """Generate structured answer with references.

        Args:
            question: The question to answer
            user_profile: User profile (LEGAL/ENGINEERING)
            contract_min_citations: Minimum citations required (ENGINEERING)
            dry_run: If True, run full pre-LLM pipeline but skip LLM call.
                     Returns run_meta with all pipeline state for analysis.
                     EVAL = PROD: uses exact same code path, just stops before LLM.
            history_context: Formatted conversation history (optional).
            corpus_scope: Corpus search scope: "single" (default), "explicit", or "all".
            target_corpora: List of corpus IDs for "explicit" scope.
            original_query: Original query before rewriting (None for first turn).
            last_exchange: Last user+assistant exchange for intent context augmentation.
        """
        
        if not question.strip():
            raise RAGEngineError("Question cannot be empty.")

        resolved_profile = self._normalize_user_profile(user_profile)

        # ===================================================================
        # STAGE 1: SETUP (delegated to planning module)
        # ===================================================================
        answer_ctx = prepare_answer_context(
            question=question,
            corpus_id=str(getattr(self, "corpus_id", "") or "").strip(),
            resolved_profile=resolved_profile,
            top_k=int(getattr(self, "top_k", 3)),
            get_effective_policy_fn=get_effective_policy,
            classify_intent_fn=policy_engine.classify_question_intent_with_router,
            apply_policy_to_intent_fn=policy_engine._apply_answer_policy_to_claim_intent,
            is_debug_corpus_fn=instrumentation.is_debug_corpus_enabled,
            iso_utc_now_fn=self._iso_utc_now,
            git_commit_fn=self._best_effort_git_commit_short,
            resolver_fn=self._resolver,
            required_anchors_payload=None,  # DEPRECATED: eval must not influence retrieval
            contract_min_citations=contract_min_citations,
            last_exchange=last_exchange,
            original_query=original_query,
        )

        # Extract from AnswerContext for compatibility with existing code
        ctx = answer_ctx.ctx
        plan = answer_ctx.plan
        effective_plan = answer_ctx.effective_plan
        effective_policy = answer_ctx.effective_policy
        run_meta = answer_ctx.run_meta
        claim_intent_final = answer_ctx.claim_intent_final
        corpus_debug_on = answer_ctx.corpus_debug_on
        explicit_article_refs = answer_ctx.explicit_article_refs
        explicit_annex_refs = answer_ctx.explicit_annex_refs
        multi_anchor_question = answer_ctx.multi_anchor_question
        policy_intent_keys = answer_ctx.policy_intent_keys
        where_for_retrieval = answer_ctx.where_for_retrieval
        focus = ctx.focus  # Extract focus from ctx for downstream use
        required_anchors_payload = None  # DEPRECATED: eval must not influence retrieval
        planner_payload: dict[str, Any] = {}  # Reserved for future planner output

        # Use RetrievalPassTracker from retrieval.py to track retrieval passes
        pass_tracker = RetrievalPassTracker(self.retriever)

        def _record_pass(*, pass_name: str, planned_where: dict[str, Any] | None, planned_collection_type: str) -> None:
            pass_tracker.record_pass(
                pass_name=pass_name,
                planned_where=planned_where,
                planned_collection_type=planned_collection_type,
            )

        # ===================================================================
        # STAGE 1.5: CORPUS DISCOVERY (if corpus_scope == "discover")
        # ===================================================================
        if corpus_scope == "discover":
            from ..common.config_loader import get_discovery_settings

            disc_raw = get_discovery_settings()
            disc_config = DiscoveryConfig(
                enabled=disc_raw.get("enabled", True),
                probe_top_k=int(disc_raw.get("probe_top_k", 10)),
                auto_threshold=float(disc_raw.get("auto_threshold", 0.75)),
                suggest_threshold=float(disc_raw.get("suggest_threshold", 0.65)),
                ambiguity_margin=float(disc_raw.get("ambiguity_margin", 0.10)),
                max_corpora=int(disc_raw.get("max_corpora", 5)),
                llm_disambiguation=bool(disc_raw.get("llm_disambiguation", True)),
                w_similarity=float(disc_raw.get("scoring_weights", {}).get("w_similarity", 0.50)),
                w_best=float(disc_raw.get("scoring_weights", {}).get("w_best", 0.50)),
                max_suggest_corpora=int(disc_raw.get("max_suggest_corpora", 3)),
            )

            all_corpus_ids = self._available_corpora()
            project_root = str(Path(__file__).resolve().parents[2])
            resolver = load_resolver_for_project_root(project_root)

            # The resolver returns normalized (underscore) keys from the
            # registry (e.g. "ai_act"), but documents are indexed with config
            # keys (e.g. "ai-act"). Build a mapping so alias detection returns
            # the same key format as corpus_ids and probe results.
            _norm_to_cfg: dict[str, str] = {
                normalize_corpus_id(k): k for k in all_corpus_ids
            }

            class _ConfigKeyResolver:
                """Adapter: maps resolver's normalized keys to config keys."""

                def __init__(self, inner: object, key_map: dict[str, str]) -> None:
                    self._inner = inner
                    self._map = key_map

                def mentioned_corpus_keys(self, text: str) -> list[str]:
                    norm_keys = self._inner.mentioned_corpus_keys(text)  # type: ignore[attr-defined]
                    return [self._map.get(k, k) for k in norm_keys]

            adapted_resolver = _ConfigKeyResolver(resolver, _norm_to_cfg)

            def _probe_fn(q: str, cid: str, k: int) -> list[tuple[dict, float]]:
                coll = self._get_collection_for_corpus(cid)
                _ids, _docs, metas, dists = self.retriever._query_collection_raw(
                    collection=coll, question=q, k=k,
                    where={"corpus_id": cid}, track_state=False,
                )
                return list(zip(metas, dists, strict=False))

            from .llm_client import call_llm as _call_llm

            disc_result = discover_corpora(
                question=question,
                corpus_ids=all_corpus_ids,
                probe_fn=_probe_fn,
                resolver=adapted_resolver,
                config=disc_config,
                llm_fn=_call_llm if disc_config.llm_disambiguation else None,
            )

            run_meta["discovery"] = {
                "matches": [
                    {
                        "corpus_id": m.corpus_id,
                        "confidence": m.confidence,
                        "reason": m.reason,
                        "display_name": resolver.display_name_for(m.corpus_id) or m.corpus_id,
                    }
                    for m in disc_result.matches
                ],
                "resolved_scope": disc_result.resolved_scope,
                "resolved_corpora": list(disc_result.resolved_corpora),
                "gate": disc_result.gate,
            }

            if disc_result.gate == "ABSTAIN":
                run_meta["abstain"] = {
                    "abstained": True,
                    "reason": "Kan ikke med sikkerhed identificere den relevante lovgivning. Vælg venligst en eller flere love manuelt.",
                }
                return {
                    "answer": "",
                    "references": [],
                    "reference_lines": [],
                    "retrieval": {},
                    "run": run_meta,
                    "prompt": "",
                }

            # Resolve discovery to effective scope
            corpus_scope = disc_result.resolved_scope
            target_corpora = list(disc_result.resolved_corpora)

        # ===================================================================
        # STAGE 2: RETRIEVAL PIPELINE (with cross-law support)
        # ===================================================================
        # Determine if multi-corpus retrieval is needed
        effective_target_corpora = target_corpora or []
        use_multi_corpus = (
            corpus_scope == "all"
            or (corpus_scope == "explicit" and len(effective_target_corpora) > 0)
        )
        laws_searched: List[str] = []

        # Track synthesis mode for multi-corpus queries
        synthesis_context = None

        if use_multi_corpus:
            # Multi-corpus retrieval
            if corpus_scope == "all":
                # Get all available corpora
                corpus_ids = tuple(self._available_corpora())
            else:
                # Use explicitly specified corpora
                corpus_ids = tuple(effective_target_corpora)

            # Detect synthesis mode for multi-corpus queries
            # Note: We pass corpus_ids as selected_corpora since we've already resolved them
            # This avoids requiring resolver.get_all_corpus_ids() in synthesis_router
            synth_resolver = self._resolver()
            synthesis_context = detect_synthesis_mode(
                question=question,
                corpus_scope="explicit",  # Use explicit since we already have corpus_ids
                selected_corpora=list(corpus_ids),
                resolver=synth_resolver,
            )
            run_meta["synthesis_mode"] = synthesis_context.mode.name

            multi_result = self._execute_cross_law_retrieval(
                question=question,
                corpus_ids=corpus_ids,
                resolved_profile=resolved_profile,
                where_filter=where_for_retrieval,
            )
            retrieval_result = multi_result
            # Set _last_retrieved_metadatas for multi-corpus (needed for eval scorers)
            self._last_retrieved_metadatas = list(multi_result.retrieved_metas)
            laws_searched = list(multi_result.run_meta_updates.get("laws_searched", []))
            run_meta["laws_searched"] = laws_searched
            run_meta["corpus_scope"] = corpus_scope
        else:
            # Single corpus retrieval (default behavior)
            retrieval_result = self._modular_retrieval(
                question=question,
                resolved_profile=resolved_profile,
                where_for_retrieval=where_for_retrieval,
                ctx=ctx,
            )
            laws_searched = [str(getattr(self, "corpus_id", "") or "")]
            run_meta["laws_searched"] = laws_searched
            run_meta["corpus_scope"] = "single"

        hits = retrieval_result.hits
        distances = retrieval_result.distances

        # Track planned filter state for debugging/payload
        final_planned_where = deepcopy(effective_plan.where) if effective_plan.where is not None else None
        final_planned_collection_type = "chunk"

        # Record pass for debugging/tracing
        _record_pass(
            pass_name="multi_corpus_pipeline" if use_multi_corpus else "modular_pipeline",
            planned_where=final_planned_where,
            planned_collection_type=final_planned_collection_type,
        )

        # Merge run_meta updates from pipeline
        for key, value in retrieval_result.run_meta_updates.items():
            if isinstance(value, dict) and isinstance(run_meta.get(key), dict):
                run_meta[key].update(value)
            else:
                run_meta[key] = value

        # Extract ranking debug info from pipeline run_meta_updates
        ranking_debug_payload = retrieval_result.run_meta_updates.get("hybrid_rerank")

        # Prompt building: build references, kilder_block, and context_string from selected chunks
        # Note: citable filtering and context selection are already done in the modular pipeline
        retrieved_ids = retrieval_result.retrieved_ids

        # Check if raw anchor logging is enabled
        enable_raw_anchor_log = resolved_profile == UserProfile.ENGINEERING and (
            os.environ.get("ENABLE_CONTEXT_RAW_ANCHOR_LOG", "true").lower() == "true"
            or os.environ.get("ENABLE_REQUIRED_ANCHOR_DIVERSITY_GUARD", "").lower() == "true"
        )

        # Load settings for citation improvement features
        settings = load_settings()

        # Build prompt context from selected chunks (Stage 5 of pipeline)
        # Use context positioning from settings (default: "sandwich" for Lost-in-the-Middle mitigation)
        prompt_ctx = build_prompt_context(
            selected=retrieval_result.selected_chunks,
            format_metadata_fn=citations._format_metadata_audit_safe,
            corpus_id=str(getattr(self, "corpus_id", "") or ""),
            enable_raw_anchor_log=enable_raw_anchor_log,
            context_positioning=settings.context_positioning,
        )

        # Extract results (legacy variable names for compatibility)
        included = prompt_ctx.included
        references = prompt_ctx.references
        context_blocks = prompt_ctx.context_blocks
        references_structured_all = prompt_ctx.references_structured
        citable_count_total = retrieval_result.citable_count
        citable_count_context = len(prompt_ctx.included)
        total_retrieved = retrieval_result.total_retrieved
        non_citable_debug: List[Dict[str, Any]] = []  # No longer tracked separately

        # Update run_meta with raw anchor log (if enabled)
        if enable_raw_anchor_log and prompt_ctx.raw_context_anchors:
            run_meta["context_raw_chunks_count"] = int(len(included))
            run_meta["context_raw_unique_anchors_count"] = int(len(prompt_ctx.raw_context_anchors))
            run_meta["context_raw_unique_anchors_top"] = prompt_ctx.raw_context_anchors[:20]

        if instrumentation.is_debug_enabled():
            print(
                f"[rag_debug] retrieved_total={total_retrieved} citable={citable_count_total} context_capped={citable_count_context} non_citable={len(non_citable_debug)}"
            )
            if non_citable_debug:
                print("[rag_debug] non_citable_items (chunk_id, location_id, heading_path):")
                for item in non_citable_debug[:10]:
                    print(
                        "- "
                        + str(item.get("chunk_id") or "")
                        + " "
                        + str(item.get("location_id") or "")
                        + " "
                        + str(item.get("heading_path") or "")
                    )

        # Debug: high-level retrieval evidence summary.
        try:
            anchors_in_top_k: list[str] = []
            for r in list(references_structured_all or []):
                if not isinstance(r, dict):
                    continue
                if r.get("article"):
                    anchors_in_top_k.append(f"article:{str(r.get('article')).strip().lower()}")
                if r.get("recital"):
                    anchors_in_top_k.append(f"recital:{str(r.get('recital')).strip().lower()}")
                if r.get("annex"):
                    # Annex values are often roman numerals; normalize to lowercase for debug.
                    anchors_in_top_k.append(f"annex:{str(r.get('annex')).strip().lower()}")
            anchors_in_top_k = sorted(set([a for a in anchors_in_top_k if a and str(a).strip()]))
            run_meta["citable_refs_count"] = int(len(references_structured_all or []))
            run_meta["references_structured_all_count"] = int(len(references_structured_all or []))
            run_meta["candidates_count"] = int(total_retrieved)
            run_meta["anchors_in_top_k"] = anchors_in_top_k

            # Mirror anchor state into anchor_rescue when enabled.
            if required_anchors_payload and isinstance(run_meta.get("anchor_rescue"), dict):
                run_meta["anchor_rescue"]["anchors_in_top_k"] = list(anchors_in_top_k)
                try:
                    req_any_1 = set([
                        re.sub(r"\s+", "", str(a)).strip().lower()
                        for a in list(required_anchors_payload.get("must_include_any_of") or [])
                        if isinstance(a, str) and a.strip() and ":" in a
                    ])
                    req_any_2 = set([
                        re.sub(r"\s+", "", str(a)).strip().lower()
                        for a in list(required_anchors_payload.get("must_include_any_of_2") or [])
                        if isinstance(a, str) and a.strip() and ":" in a
                    ])
                    req_all = set([
                        re.sub(r"\s+", "", str(a)).strip().lower()
                        for a in list(required_anchors_payload.get("must_include_all_of") or [])
                        if isinstance(a, str) and a.strip() and ":" in a
                    ])
                    present = set(anchors_in_top_k)
                    if req_any_1 and not (req_any_1 & present):
                        run_meta["anchor_rescue"]["missing_required_anchor_any_of"] = sorted(req_any_1)
                    if req_any_2 and not (req_any_2 & present):
                        run_meta["anchor_rescue"]["missing_required_anchor_any_of_2"] = sorted(req_any_2)
                    if req_all:
                        run_meta["anchor_rescue"]["missing_required_anchor_all_of"] = sorted(req_all - present)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            run_meta.setdefault("citable_refs_count", int(len(references_structured_all or [])))
            run_meta.setdefault("references_structured_all_count", int(len(references_structured_all or [])))
            try:
                run_meta.setdefault("candidates_count", int(total_retrieved))
            except Exception:  # noqa: BLE001
                run_meta.setdefault("candidates_count", None)
            run_meta.setdefault("anchors_in_top_k", [])

        # Snapshot A: after retrieval + candidate/reference construction.
        try:
            run_meta["effective_where"] = deepcopy(getattr(self, "_last_effective_where", None))
        except Exception:  # noqa: BLE001
            run_meta["effective_where"] = getattr(self, "_last_effective_where", None)
        instrumentation._debug_dump_run_meta(
            run_meta=run_meta,
            stage="after_retrieval_candidates",
            extra={
                "retrieved_ids_count": int(len(getattr(self, "_last_retrieved_ids", []) or [])),
                "retrieved_metadatas_count": int(len(getattr(self, "_last_retrieved_metadatas", []) or [])),
                "citable_count_total": int(citable_count_total),
            },
        )

        # Phase 4: kilder_block and context string built by prompt_builder
        kilder_block = prompt_ctx.kilder_block
        context = prompt_ctx.context_string

        # If the user explicitly asks what a specific recital says and we have it as a citable chunk,
        # return the source text directly (audit-safe, avoids LLM + distance noise).
        recital_ref = helpers._extract_recital_ref(question)
        if recital_ref and helpers._looks_like_recital_quote_question(question):
            for doc, meta, _chunk_id, _precise_override in included:
                if str((meta or {}).get("recital") or "").strip() == recital_ref:
                    answer_text = str(doc or "").strip()
                    answer_text = helpers._normalize_abstain_text(answer_text)
                    reference_lines = [f"[{r['idx']}] {r.get('precise_ref') or r.get('display')}" for r in references_structured_all]
                    return helpers.build_answer_response_payload(
                        run_meta=run_meta,
                        user_profile_value=resolved_profile.value,
                        focus=focus,
                        intent_value=plan.intent.value,
                        answer_text=answer_text,
                        references=references_structured_all,
                        reference_lines=reference_lines,
                        distances=distances,
                        retrieval_state={
                            "query_collection": getattr(self, "_last_query_collection_name", None),
                            "query_where": getattr(self, "_last_query_where", None),
                            "planned_where": deepcopy(final_planned_where) if final_planned_where is not None else None,
                            "effective_where": deepcopy(getattr(self, "_last_effective_where", None)),
                            "planned_collection_type": str(final_planned_collection_type),
                            "effective_collection": getattr(self, "_last_effective_collection_name", None),
                            "effective_collection_type": getattr(self, "_last_effective_collection_type", None),
                            "retrieved_ids": list(getattr(self, "_last_retrieved_ids", []) or []),
                            "retrieved_metadatas": list(getattr(self, "_last_retrieved_metadatas", []) or []),
                        },
                        effective_plan=effective_plan,
                        where_for_retrieval=where_for_retrieval,
                        pass_tracker_passes=pass_tracker.get_passes(),
                        hybrid_rerank={
                            "enabled": bool(getattr(self, "enable_hybrid_rerank", True)),
                            "weights": {"alpha_vec": self.ranking_weights.alpha_vec, "beta_bm25": self.ranking_weights.beta_bm25, "gamma_cite": self.ranking_weights.gamma_cite, "delta_role": self.ranking_weights.delta_role},
                            "vec_k": int(getattr(self, "hybrid_vec_k", 30)),
                        },
                        sibling_expansion=self._last_sibling_expansion,
                    )

        # Engineering evidence gating threshold is computed above (second-pass retrieval). Default 2.
        min_citable_required = 2
        try:
            scoped = bool(
                focus
                or helpers._looks_like_structure_question(question)
                or helpers._extract_article_ref(question)
                or helpers._extract_chapter_ref(question)
            )
            if scoped:
                min_citable_required = 1
        except Exception:  # noqa: BLE001
            pass

        # Multi-turn: still run abstain check but with relaxed soft threshold.
        # allow_low_evidence_answer=True bypasses the soft distance check while
        # keeping the hard max distance guardrail (catches truly out-of-scope queries).
        has_history = bool(history_context and history_context.strip())
        abstain_reason = self._should_abstain(
            question,
            hits,
            distances,
            allow_low_evidence_answer=effective_plan.allow_low_evidence_answer or has_history,
            references_structured=references_structured_all,
            corpus_scope=corpus_scope,
        )
        did_abstain = False
        bypass_required_support_gate = False
        if abstain_reason:
            did_abstain = True
            run_meta.setdefault("abstain", {})
            run_meta["abstain"].update(
                {
                    "abstained": True,
                    "reason": str(abstain_reason),
                }
            )
            answer_text = abstain_reason
            # If dry_run and abstained, return early with the abstain message
            # (streaming should show the abstain reason, not call LLM)
            if dry_run:
                run_meta["dry_run"] = True
                run_meta["dry_run_stage"] = "abstained"
                return helpers.build_answer_response_payload(
                    run_meta=run_meta,
                    user_profile_value=resolved_profile.value,
                    focus=focus,
                    intent_value=plan.intent.value,
                    answer_text=str(abstain_reason),
                    references=references_structured_all,
                    reference_lines=[f"[{r['idx']}] {r.get('display')}" for r in references_structured_all],
                    distances=distances,
                    retrieval_state={
                        "query_collection": getattr(self, "_last_query_collection_name", None),
                        "query_where": getattr(self, "_last_query_where", None),
                        "planned_where": deepcopy(final_planned_where) if final_planned_where is not None else None,
                        "effective_where": deepcopy(getattr(self, "_last_effective_where", None)),
                        "planned_collection_type": str(final_planned_collection_type),
                        "effective_collection": getattr(self, "_last_effective_collection_name", None),
                        "effective_collection_type": getattr(self, "_last_effective_collection_type", None),
                        "retrieved_ids": list(getattr(self, "_last_retrieved_ids", []) or []),
                        "retrieved_metadatas": list(getattr(self, "_last_retrieved_metadatas", []) or []),
                    },
                    effective_plan=effective_plan,
                    where_for_retrieval=where_for_retrieval,
                    pass_tracker_passes=pass_tracker.get_passes(),
                    dry_run=True,
                    prompt="",  # No prompt needed - we're abstaining
                    ranking_debug=ranking_debug_payload,
                    sibling_expansion=self._last_sibling_expansion,
                )
        else:
            run_meta.setdefault("abstain", {})
            run_meta["abstain"].update({"abstained": False})
            # If ENGINEERING and insufficient citable evidence, do not call the LLM.
            if resolved_profile == UserProfile.ENGINEERING and citable_count_total < min_citable_required:
                run_meta["final_gate_reason"] = "insufficient_citable_evidence_pre_llm"
                answer_text = "MISSING_REF"
            elif dry_run:
                # DRY RUN MODE: Return pre-LLM pipeline state without calling LLM.
                # EVAL = PROD: This uses the exact same code path, just stops before LLM.
                # Also build prompt for streaming use case.
                run_meta["dry_run"] = True
                run_meta["dry_run_stage"] = "pre_llm_complete"

                # Build prompt (needed for streaming to use same retrieval as non-streaming)
                focus_block_str = focus_block_for_prompt(focus)
                dry_run_prompt = build_prompt(
                    ctx=ctx,
                    plan=effective_plan,
                    context=context,
                    focus_block=focus_block_str,
                    contract_min_citations=contract_min_citations,
                    history_context=history_context,
                )

                return helpers.build_answer_response_payload(
                    run_meta=run_meta,
                    user_profile_value=resolved_profile.value,
                    focus=focus,
                    intent_value=plan.intent.value,
                    answer_text="[DRY_RUN - LLM not called]",
                    references=references_structured_all,
                    reference_lines=[f"[{r['idx']}] {r.get('display')}" for r in references_structured_all],
                    distances=distances,
                    retrieval_state={
                        "query_collection": getattr(self, "_last_query_collection_name", None),
                        "query_where": getattr(self, "_last_query_where", None),
                        "planned_where": deepcopy(final_planned_where) if final_planned_where is not None else None,
                        "effective_where": deepcopy(getattr(self, "_last_effective_where", None)),
                        "planned_collection_type": str(final_planned_collection_type),
                        "effective_collection": getattr(self, "_last_effective_collection_name", None),
                        "effective_collection_type": getattr(self, "_last_effective_collection_type", None),
                        "retrieved_ids": list(getattr(self, "_last_retrieved_ids", []) or []),
                        "retrieved_metadatas": list(getattr(self, "_last_retrieved_metadatas", []) or []),
                    },
                    effective_plan=effective_plan,
                    where_for_retrieval=where_for_retrieval,
                    pass_tracker_passes=pass_tracker.get_passes(),
                    dry_run=True,
                    prompt=dry_run_prompt,
                    ranking_debug=ranking_debug_payload,
                    sibling_expansion=self._last_sibling_expansion,
                )
            else:
                # ---------------------------------------------------------------
                # STAGE 3: GENERATION (inlined from generation.py - SOLID Fase 12)
                # ---------------------------------------------------------------
                focus_block = focus_block_for_prompt(focus)

                # Determine JSON mode settings per profile
                engineering_json_mode = bool(
                    resolved_profile == UserProfile.ENGINEERING
                    and instrumentation.is_engineering_json_mode_enabled()
                )
                legal_json_mode = bool(
                    resolved_profile == UserProfile.LEGAL
                    and instrumentation.is_legal_json_mode_enabled()
                )

                # Build base prompt - use multi-corpus prompt if synthesis mode detected
                if synthesis_context is not None and synthesis_context.mode != SynthesisMode.SINGLE:
                    # Multi-corpus prompt with grounding rules
                    prompt = build_multi_corpus_prompt(
                        mode=synthesis_context.mode,
                        question=question,
                        context=context,
                        kilder_block=kilder_block,
                        references_structured=list(references_structured_all or []),
                        user_profile=resolved_profile.name if hasattr(resolved_profile, "name") else str(resolved_profile),
                        resolver=self._resolver(),
                    )
                else:
                    # Standard single-corpus prompt
                    prompt = build_prompt(
                        ctx=ctx,
                        plan=effective_plan,
                        context=context,
                        focus_block=focus_block,
                        contract_min_citations=contract_min_citations,
                        legal_json_mode=legal_json_mode,
                        history_context=history_context,
                    )

                # Policy-driven answer shaping (ENGINEERING only)
                answer_policy = getattr(effective_policy, "answer_policy", None)
                if os.getenv("DISABLE_ANSWER_POLICY", "").strip().lower() in ("1", "true", "yes"):
                    answer_policy = None
                prompt += build_answer_policy_suffix(answer_policy, resolved_profile)

                # Track LLM calls
                if "llm_calls_count" not in run_meta:
                    run_meta["llm_calls_count"] = 0

                def _call_llm_counted(p: str) -> str:
                    run_meta["llm_calls_count"] = int(run_meta.get("llm_calls_count") or 0) + 1
                    return self._call_llm(p)

                # Citation requirement suffix (ENGINEERING only)
                prompt += build_citation_requirement_suffix(
                    user_profile=resolved_profile,
                    contract_min_citations=contract_min_citations,
                    references_structured_all=references_structured_all,
                    json_mode=engineering_json_mode,
                )

                # Build allowed_idxs set from references
                allowed_idxs: set[int] = set()
                for r in list(references_structured_all or []):
                    if not isinstance(r, dict):
                        continue
                    try:
                        allowed_idxs.add(int(r.get("idx")))
                    except Exception:  # noqa: BLE001
                        continue

                # Create generation config based on profile and JSON mode setting
                profile_json_mode = (
                    engineering_json_mode if resolved_profile == UserProfile.ENGINEERING
                    else legal_json_mode
                )
                gen_config = GenerationConfig.for_profile(
                    resolved_profile,
                    contract_min_citations=contract_min_citations,
                    json_mode_enabled=profile_json_mode,
                )

                # Execute unified generation pipeline
                gen_result = execute_structured_generation(
                    prompt=prompt,
                    llm_fn=_call_llm_counted,
                    config=gen_config,
                    allowed_idxs=allowed_idxs,
                    references_structured_all=list(references_structured_all or []),
                    answer_policy=answer_policy,
                    claim_intent=claim_intent_final,
                )

                answer_text = gen_result.answer_text

                # Update run_meta with JSON mode flags
                run_meta["engineering_json_mode"] = bool(engineering_json_mode)
                run_meta["legal_json_mode"] = bool(legal_json_mode)

                # Sync generation result to run_meta for backwards compatibility
                run_meta["llm_calls_count"] = int(run_meta.get("llm_calls_count") or 0)
                run_meta["citations_source"] = gen_result.debug.get("citations_source", "text_parse")

                if engineering_json_mode and gen_result is not None:
                    # Sync JSON mode results to run_meta (local private function)
                    _sync_json_mode_results_to_run_meta(
                        gen_result=gen_result,
                        run_meta=run_meta,
                        allowed_idxs=allowed_idxs,
                        contract_min_citations=contract_min_citations,
                        answer_policy=answer_policy,
                    )

                    # Snapshot B: after JSON parse/validate/repair/enrich (before claim-stage gates).
                    instrumentation._debug_dump_run_meta(
                        run_meta=run_meta,
                        stage="after_engineering_json_mode",
                        extra={
                            "answer_preview": str(answer_text or "")[:160],
                            "allowed_idxs_count": int(run_meta.get("allowed_idxs_count") or 0),
                            "cited_idxs_json": list(run_meta.get("cited_idxs") or []),
                            "valid_cited": list(run_meta.get("valid_cited") or []),
                            "llm_calls_count": int(run_meta.get("llm_calls_count") or 0),
                        },
                    )
                else:
                    # Non-JSON mode: ensure contract/debug fields exist
                    run_meta.setdefault("allowed_idxs", sorted(allowed_idxs))
                    run_meta.setdefault("allowed_idxs_count", len(allowed_idxs))

                # Debug-only: force a deterministic answer body to reproduce gating behavior.
                # (Only applies to non-JSON mode; JSON mode handles this internally)
                forced_answer = None
                if not engineering_json_mode and corpus_debug_on:
                    raw_forced = str(os.getenv("RAG_DEBUG_FORCE_ANSWER", "") or "").strip()
                    if raw_forced:
                        forced_answer = raw_forced
                        answer_text = forced_answer
                        run_meta.setdefault("corpus_debug", {})
                        run_meta["corpus_debug"].update({"forced_answer_used": True})

                # Ensure contract/debug fields exist for non-JSON mode too.
                if not bool(run_meta.get("engineering_json_mode")):
                    run_meta.setdefault("allowed_idxs", [])
                    run_meta.setdefault("allowed_idxs_count", int(len(run_meta.get("allowed_idxs") or [])))

                # ENGINEERING: deterministic 1x retry if citations are missing/insufficient but sources exist.
                # Delegated to generation module for cleaner orchestration.
                if (
                    resolved_profile == UserProfile.ENGINEERING
                    and contract_min_citations is not None
                    and forced_answer is None
                    and not bool(run_meta.get("engineering_json_mode"))
                ):
                    try:
                        min_cit = int(contract_min_citations)
                    except Exception:  # noqa: BLE001
                        min_cit = 0

                    retry_allowed_idxs: set[int] = set()
                    for r in list(references_structured_all or []):
                        if not isinstance(r, dict):
                            continue
                        try:
                            retry_allowed_idxs.add(int(r.get("idx")))
                        except Exception:  # noqa: BLE001
                            continue

                    def _call_llm_counted_retry(p: str) -> str:
                        run_meta["llm_calls_count"] = int(run_meta.get("llm_calls_count") or 0) + 1
                        return self._call_llm(p)

                    answer_text = execute_citation_retry_if_needed(
                        answer_text=str(answer_text or ""),
                        prompt=prompt,
                        llm_fn=_call_llm_counted_retry,
                        allowed_idxs=retry_allowed_idxs,
                        min_citations=min_cit,
                        run_meta=run_meta,
                        strip_references_fn=helpers._strip_trailing_references_section,
                    )

        answer_text = helpers._normalize_abstain_text(answer_text)

        # ---------------------------------------------------------------
        # STAGE 4a-pre: POLICY GATES (pre-engineering)
        # ---------------------------------------------------------------
        # Apply LEGAL gate + intent determination BEFORE engineering answer building
        pre_policy_result = policy_engine.apply_pre_engineering_policy_gates(
            answer_text=answer_text,
            question=question,
            resolved_profile=resolved_profile,
            references_structured_all=list(references_structured_all or []),
            claim_intent_from_run_meta=(run_meta.get("claim_intent") or {}).get("final"),
            classify_intent_fn=policy_engine.classify_question_intent,
            run_meta=run_meta,
            corpus_debug_on=corpus_debug_on,
        )

        # Extract pre-engineering results
        answer_text = pre_policy_result.answer_text
        references_structured_all = pre_policy_result.references_structured_all
        legal_allow_reference_fallback = pre_policy_result.legal_allow_reference_fallback
        intent_used = pre_policy_result.intent_used or claim_intent_final

        self._maybe_log_intent_event(question=question, intent=intent_used, profile=resolved_profile)

        # Disclaimer and low-evidence check
        max_distance = getattr(self, "max_distance", None)
        low_evidence = False
        if max_distance is not None and distances:
            try:
                effective_max_distance = float(max_distance)
                if effective_plan.allow_low_evidence_answer:
                    effective_max_distance = max(effective_max_distance, 1.35)
                low_evidence = min(distances) > effective_max_distance
            except Exception:  # noqa: BLE001
                low_evidence = False
        disclaimer = build_disclaimer(ctx=ctx, low_evidence=low_evidence)

        # Preserve internal fail-closed marker: do not decorate it.
        if str(answer_text or "").strip() != "MISSING_REF":
            if disclaimer:
                answer_text = f"{answer_text}\n\nBemærk: {disclaimer}"

        # Build the simple reference lines shown in the UI reference box.
        reference_lines_all = [
            f"[{r['idx']}] {r.get('precise_ref') or r.get('display')}" for r in references_structured_all
        ]

        # If ENGINEERING profile, enforce the Engineering Answer Contract
        if resolved_profile == UserProfile.ENGINEERING:
            answer_text = self._build_engineering_answer(
                raw_interpretation=answer_text,
                ctx=ctx,
                references_structured=references_structured_all,
                reference_lines=reference_lines_all,
                distances=distances,
                total_retrieved=total_retrieved,
                citable_count=citable_count_total,
                min_citable_required=min_citable_required,
            )

            # Post-process to enforce requirements-oriented structure when policy requests it.
            try:
                ap = getattr(effective_policy, "answer_policy", None)
                if ap is not None:
                    answer_text, ap_dbg = policy_engine._engineering_apply_answer_policy_requirements_enforcement(
                        answer_text=str(answer_text or ""),
                        answer_policy=ap,
                        engineering_json_mode=bool(run_meta.get("engineering_json_mode") is True),
                    )
                    run_meta.setdefault("answer_policy", {})
                    run_meta["answer_policy"].update(ap_dbg)
            except Exception:  # noqa: BLE001
                pass

        # ---------------------------------------------------------------
        # STAGE 4a-post: POLICY GATES (post-engineering)
        # ---------------------------------------------------------------
        # Apply claim-stage gates AFTER engineering answer building
        # These gates analyze the final answer text for high-risk mentions
        # and normative claims without proper evidence support.
        post_policy_result = policy_engine.apply_post_engineering_policy_gates(
            answer_text=answer_text,
            question=question,
            resolved_profile=resolved_profile,
            references_structured_all=list(references_structured_all or []),
            intent_used=intent_used,
            classify_evidence_type_fn=policy_engine.classify_evidence_type_from_metadata,
            select_references_used_fn=citations.select_references_used_in_answer,
            inject_enforcement_citations_fn=citations._engineering_inject_neutral_hjemmel_citations_for_enforcement,
            run_meta=run_meta,
            corpus_debug_on=corpus_debug_on,
        )

        # Extract post-engineering results
        answer_text = post_policy_result.answer_text
        if post_policy_result.did_abstain:
            did_abstain = True
        if post_policy_result.bypass_required_support_gate:
            bypass_required_support_gate = True

        # ---------------------------------------------------------------
        # STAGE 4b: CITATION PROCESSING (delegated to citations module)
        # ---------------------------------------------------------------
        # Apply all citation processing via citations module
        citation_processing_result = citations.apply_all_citation_processing(
            answer_text=answer_text,
            question=question,
            references_structured_all=list(references_structured_all or []),
            resolved_profile=resolved_profile,
            intent_used=intent_used,
            did_abstain=did_abstain,
            required_anchors_payload=required_anchors_payload,
            contract_min_citations=contract_min_citations,
            is_legal_profile=(resolved_profile == UserProfile.LEGAL),
            legal_allow_reference_fallback=bool(legal_allow_reference_fallback),
            run_meta=run_meta,
            corpus_debug_on=corpus_debug_on,
            is_debug_enabled_fn=instrumentation.is_debug_enabled,
        )

        # Extract results
        answer_text = citation_processing_result.answer_text
        references_structured: list[dict[str, Any]] = citation_processing_result.references_structured
        reference_lines = citation_processing_result.reference_lines
        used_chunk_ids = citation_processing_result.used_chunk_ids

        # Snapshot D: after citation processing
        instrumentation._debug_dump_run_meta(
            run_meta=run_meta,
            stage="after_citation_processing",
            extra={
                "answer_preview": str(answer_text or "")[:160],
                "references_structured_count": int(len(references_structured or [])),
                "reference_lines_count": int(len(reference_lines or [])),
                "citations_source": run_meta.get("citations_source"),
                "parsed_citations_raw": list(run_meta.get("parsed_citations_raw") or []),
            },
        )

        # Required-support / normative guard (policy-aware).
        claim_intent_final = intent_used
        answer_text, references_structured = policy_engine._apply_required_support_guard(
            resolved_profile=resolved_profile,
            claim_intent_final=claim_intent_final,
            answer_text=str(answer_text or ""),
            references_structured=list(references_structured or []),
            bypass_required_support_gate=bool(bypass_required_support_gate),
            policy=(effective_policy if "effective_policy" in locals() else None),
            run_meta=run_meta,
        )
        if str(answer_text or "").strip() == "MISSING_REF":
            references_structured = []
            reference_lines = []
            used_chunk_ids = []

        # Danish-first deterministic normalization (modal verbs + standalone YES/NO lines).
        # This must not change citation format or reference gating.
        # Delegates to helpers._normalize_modals_to_danish (Phase D deduplication)
        answer_text = helpers._normalize_modals_to_danish(answer_text)

        # --- SCOPE display-only post-processing (after hard reference gating) ---
        # IMPORTANT:
        # - Only applies when intent_used == SCOPE
        # - Only mutates answer_text + reference_lines (UI display)
        # - Must not change [n] markers or reference selection
        if intent_used == ClaimIntent.SCOPE:
            if resolved_profile == UserProfile.ENGINEERING:
                answer_text = policy_engine._engineering_remove_normative_bullets_from_systemkrav_section_for_scope(answer_text)

            answer_text, reference_lines = policy_engine._scope_apply_litra_consistency_to_display(
                answer_text=answer_text,
                reference_lines=list(reference_lines or []),
            )

        if corpus_debug_on:
            # Collect and emit corpus debug telemetry (delegated to instrumentation module)
            instrumentation.collect_corpus_debug_telemetry(
                answer_text=str(answer_text or ""),
                references_structured_all=list(references_structured_all or []),
                contract_min_citations=contract_min_citations,
                run_meta=run_meta,
                total_retrieved=total_retrieved,
                citable_count=citable_count_total,
                strip_references_fn=helpers._strip_trailing_references_section,
                count_normative_fn=helpers._count_normative_sentences,
            )

        # Snapshot FINAL: end-state payload summary (for eval before/after diagnostics).
        # Keep counters in run_meta aligned with the final state.
        try:
            run_meta["references_structured_all_count"] = int(len(references_structured_all or []))
            run_meta["references_structured_count"] = int(len(references_structured or []))
            run_meta["reference_lines_count"] = int(len(reference_lines or []))
            run_meta["answer_is_missing_ref"] = bool(str(answer_text or "").strip() == "MISSING_REF")
        except Exception:  # noqa: BLE001
            pass

        instrumentation._debug_dump_run_meta(
            run_meta=run_meta,
            stage="final_payload",
            extra={
                "answer_preview": str(answer_text or "")[:160],
                "answer_is_missing_ref": bool(str(answer_text or "").strip() == "MISSING_REF"),
                "references_structured_all_count": int(len(references_structured_all or [])),
                "references_structured_count": int(len(references_structured or [])),
                "reference_lines_count": int(len(reference_lines or [])),
                "final_fail_reason": run_meta.get("final_fail_reason"),
                "fail_reason": run_meta.get("fail_reason"),
            },
        )

        return helpers.build_answer_response_payload(
            run_meta=run_meta,
            user_profile_value=resolved_profile.value,
            focus=focus,
            intent_value=plan.intent.value,
            answer_text=answer_text,
            references=references_structured,
            reference_lines=reference_lines,
            distances=distances,
            retrieval_state={
                "query_collection": getattr(self, "_last_query_collection_name", None),
                "query_where": getattr(self, "_last_query_where", None),
                "planned_where": final_planned_where,
                "effective_where": getattr(self, "_last_effective_where", None),
                "planned_collection_type": final_planned_collection_type,
                "effective_collection": getattr(self, "_last_effective_collection_name", None),
                "effective_collection_type": getattr(self, "_last_effective_collection_type", None),
                "retrieved_ids": getattr(self, "_last_retrieved_ids", []),
                "retrieved_metadatas": getattr(self, "_last_retrieved_metadatas", []),
            },
            effective_plan=effective_plan,
            where_for_retrieval=where_for_retrieval,
            pass_tracker_passes=pass_tracker.get_passes(),
            planner=planner_payload,
            references_structured_all=references_structured_all,
            used_chunk_ids=used_chunk_ids,
            hybrid_rerank={
                "enabled": bool(getattr(self, "enable_hybrid_rerank", True)),
                "weights": {"alpha_vec": self.ranking_weights.alpha_vec, "beta_bm25": self.ranking_weights.beta_bm25, "gamma_cite": self.ranking_weights.gamma_cite, "delta_role": self.ranking_weights.delta_role},
                "vec_k": int(getattr(self, "hybrid_vec_k", 30)),
            },
            sibling_expansion=self._last_sibling_expansion,
            ranking_debug=ranking_debug_payload,
        )

    def answer(self, question: str) -> str:
        payload = self.answer_structured(question)
        answer_text = str(payload.get("answer") or "")
        reference_lines = payload.get("reference_lines") or []
        if reference_lines:
            answer_text = f"{answer_text}\n\nReferencer:\n" + "\n".join(reference_lines)
        else:
            answer_text = f"{answer_text}\n\nReferencer:\n(ingen)"
        return answer_text

    def _call_llm(self, prompt: str) -> str:
        """Single call boundary for LLM chat completion (mock-friendly)."""
        return self._call_openai(prompt)

    def _call_openai(self, prompt: str) -> str:
        model = getattr(self, "chat_model", _get_default_chat_model())
        temperature = _get_default_temperature()
        return call_llm(prompt, model=model, temperature=temperature)

    def _build_engineering_answer(
        self,
        raw_interpretation: str,
        ctx: "QueryContext",
        references_structured: list[dict[str, Any]],
        reference_lines: list[str],
        distances: list[float],
        total_retrieved: int = 0,
        citable_count: int = 0,
        min_citable_required: int = 2,
    ) -> str:
        """Build engineering answer (mock-friendly delegation to generation_strategies)."""
        return build_engineering_answer(
            raw_interpretation=raw_interpretation,
            ctx=ctx,
            references_structured=references_structured,
            reference_lines=reference_lines,
            distances=distances,
            total_retrieved=total_retrieved,
            citable_count=citable_count,
            min_citable_required=min_citable_required,
        )
