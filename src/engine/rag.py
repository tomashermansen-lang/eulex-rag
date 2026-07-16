from pathlib import Path
import os
from copy import deepcopy
from typing import Any, Dict, List

import chromadb

from .types import RAGEngineError
from .prompt_builder import build_prompt_context
from . import metadata_helpers
from . import query_helpers
from . import text_transforms
from . import payload_builders
from .rag_config import iso_utc_now, best_effort_git_commit_short

from .corpus_resolver import (
    load_resolver_for_project_root,
    infer_project_root,
    available_corpora,
)

from .planning import (
    UserProfile,
    normalize_user_profile,
    prepare_answer_context,
)

from dotenv import load_dotenv

from .concept_config import get_effective_policy
from ..common.config_loader import load_settings

from . import citations
from . import policy as policy_engine
from . import retrieval_orchestration as ro
from .llm_client import call_llm
from . import instrumentation
from . import indexing
from .retrieval import Retriever, RetrievalPassTracker
from .ranking import Ranker
from .rag_config import (
    _get_default_chat_model,
    _get_default_embedding_model,
    _get_default_temperature,
    _get_rag_settings,
)

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Static helpers delegated to metadata_helpers.py (Phase F deduplication)
_derive_structural_fields_from_location_id = (
    metadata_helpers._derive_structural_fields_from_location_id
)
_extract_raw_anchors_from_chunk = metadata_helpers._extract_raw_anchors_from_chunk


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

        self.docs_path = docs_path
        self.corpus_id = corpus_id
        self._project_root = infer_project_root(docs_path)
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
        rag_settings = _get_rag_settings()
        default_top_k = int(rag_settings.get("default_top_k", 3))
        self.top_k = (
            int(top_k)
            if top_k is not None
            else int(os.getenv("RAG_TOP_K", str(default_top_k)))
        )
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
                embedding_model=getattr(
                    self, "embedding_model", _get_default_embedding_model()
                ),
            )
        return self._retriever

    @property
    def ranker(self) -> Ranker:
        # Lazy initialization supports RAGEngine.__new__ usage in tests.
        if not hasattr(self, "_ranker"):
            self._ranker = Ranker()
        return self._ranker

    @property
    def _last_sibling_expansion(self) -> dict[str, Any]:
        return getattr(self.retriever, "_last_sibling_expansion", {})

    def _resolver(self):
        project_root = getattr(self, "_project_root", None)
        if not isinstance(project_root, Path):
            try:
                docs_path = getattr(self, "docs_path", None)
                if docs_path:
                    project_root = infer_project_root(getattr(self, "docs_path", None))
                else:
                    project_root = Path(__file__).resolve().parents[2]
            except Exception:  # noqa: BLE001
                project_root = Path(__file__).resolve().parents[2]
            try:
                self._project_root = project_root
            except Exception:  # noqa: BLE001
                pass

        return load_resolver_for_project_root(str(project_root))

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
    # These were thin wrappers - call query_helpers._looks_like_*() directly (Phase 1 refactoring)

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
        """Delegator → policy_engine.should_abstain_gate."""
        return policy_engine.should_abstain_gate(
            question=question,
            hits=hits,
            distances=distances,
            fallback_distances=getattr(self, "_last_distances", []),
            corpus_id=str(getattr(self, "corpus_id", "") or "").strip(),
            max_distance=getattr(self, "max_distance", None),
            hard_max_distance=getattr(self, "hard_max_distance", None),
            resolver_fn=self._resolver,
            allow_low_evidence_answer=allow_low_evidence_answer,
            references_structured=references_structured,
            corpus_scope=corpus_scope,
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

        # Reset state for this query
        self._last_distances = []
        self._last_query_collection_name = None
        self._last_query_where = None
        self.retriever._last_retrieved_ids = []
        self.retriever._last_retrieved_metadatas = []

        result = ro.execute_query_with_where(
            question=question,
            k=k,
            where=where,
            collection=self.collection,
            collection_name=getattr(self, "collection_name", None),
            retriever=self.retriever,
            ranker=self.ranker,
            enable_hybrid_rerank=bool(getattr(self, "enable_hybrid_rerank", True)),
            hybrid_vec_k=int(getattr(self, "hybrid_vec_k", 30)),
            ranking_weights=getattr(self, "ranking_weights", None),
        )

        # Sync engine state from result
        self._last_distances = list(result.distances)
        self._last_query_collection_name = result.collection_name
        self._last_query_where = result.query_where
        self.retriever._last_retrieved_ids = list(result.ids)
        self.retriever._last_retrieved_metadatas = list(result.metadatas)
        self.retriever._last_sibling_expansion = result.sibling_expansion
        return result.hits

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

        resolved_profile = normalize_user_profile(user_profile)

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
            iso_utc_now_fn=iso_utc_now,
            git_commit_fn=best_effort_git_commit_short,
            resolver_fn=self._resolver,
            required_anchors_payload=None,  # DEPRECATED: eval must not influence retrieval
            contract_min_citations=contract_min_citations,
            last_exchange=last_exchange,
            original_query=original_query,
        )

        # Extract from AnswerContext
        ctx = answer_ctx.ctx
        plan = answer_ctx.plan
        effective_plan = answer_ctx.effective_plan
        effective_policy = answer_ctx.effective_policy
        run_meta = answer_ctx.run_meta
        claim_intent_final = answer_ctx.claim_intent_final
        corpus_debug_on = answer_ctx.corpus_debug_on
        where_for_retrieval = answer_ctx.where_for_retrieval
        focus = ctx.focus
        required_anchors_payload = None  # DEPRECATED
        planner_payload: dict[str, Any] = {}
        pass_tracker = RetrievalPassTracker(self.retriever)

        # STAGE 1.5: Corpus discovery
        if corpus_scope == "discover":
            from .corpus_discovery import execute_corpus_discovery_stage

            corpus_scope, target_corpora, early_return = execute_corpus_discovery_stage(
                question=question,
                run_meta=run_meta,
                available_corpora_fn=available_corpora,
                get_collection_for_corpus_fn=lambda cid: ro.get_collection_for_corpus(
                    self.chroma, cid
                ),
                retriever=self.retriever,
                llm_fn=call_llm,
            )
            if early_return is not None:
                return early_return

        # STAGE 2: Retrieval pipeline
        from .answer_stages import (
            execute_answer_retrieval_stage,
            ANSWER_COLLECTION_TYPE,
        )

        retrieval_stage = execute_answer_retrieval_stage(
            question=question,
            corpus_scope=corpus_scope,
            target_corpora=target_corpora,
            resolved_profile=resolved_profile,
            where_for_retrieval=where_for_retrieval,
            ctx=ctx,
            run_meta=run_meta,
            effective_plan=effective_plan,
            pass_tracker=pass_tracker,
            corpus_id=str(getattr(self, "corpus_id", "") or ""),
            retriever=self.retriever,
            chroma=self.chroma,
            collection=self.collection,
            available_corpora_fn=available_corpora,
            resolver_fn=self._resolver,
        )

        hits = retrieval_stage.hits
        distances = retrieval_stage.distances
        self._last_distances = list(distances)
        retrieval_result = retrieval_stage.retrieval_result
        synthesis_context = retrieval_stage.synthesis_context
        ranking_debug_payload = retrieval_stage.ranking_debug_payload
        final_planned_where = retrieval_stage.final_planned_where

        # Pre-compute retrieval state + hybrid rerank dicts (used in 3 payload sites)
        retrieval_state = payload_builders.build_retrieval_state_dict(
            retriever=self.retriever,
            query_collection_name=getattr(self, "_last_query_collection_name", None),
            query_where=getattr(self, "_last_query_where", None),
            planned_where=final_planned_where,
            planned_collection_type=ANSWER_COLLECTION_TYPE,
        )
        hybrid_rerank = payload_builders.build_hybrid_rerank_dict(
            enable_hybrid_rerank=bool(getattr(self, "enable_hybrid_rerank", True)),
            ranking_weights=self.ranking_weights,
            hybrid_vec_k=int(getattr(self, "hybrid_vec_k", 30)),
        )

        # Build prompt context from selected chunks (Stage 5 of pipeline)
        enable_raw_anchor_log = resolved_profile == UserProfile.ENGINEERING and (
            os.environ.get("ENABLE_CONTEXT_RAW_ANCHOR_LOG", "true").lower() == "true"
            or os.environ.get("ENABLE_REQUIRED_ANCHOR_DIVERSITY_GUARD", "").lower()
            == "true"
        )
        settings = load_settings()
        prompt_ctx = build_prompt_context(
            selected=retrieval_result.selected_chunks,
            format_metadata_fn=citations._format_metadata_audit_safe,
            corpus_id=str(getattr(self, "corpus_id", "") or ""),
            enable_raw_anchor_log=enable_raw_anchor_log,
            context_positioning=settings.context_positioning,
        )

        included = prompt_ctx.included
        references_structured_all = prompt_ctx.references_structured
        citable_count_total = retrieval_result.citable_count
        total_retrieved = retrieval_result.total_retrieved

        if enable_raw_anchor_log and prompt_ctx.raw_context_anchors:
            run_meta["context_raw_chunks_count"] = int(len(included))
            run_meta["context_raw_unique_anchors_count"] = int(
                len(prompt_ctx.raw_context_anchors)
            )
            run_meta["context_raw_unique_anchors_top"] = prompt_ctx.raw_context_anchors[
                :20
            ]

        instrumentation.log_retrieval_debug(
            total_retrieved=total_retrieved,
            citable_count_total=citable_count_total,
            citable_count_context=len(included),
            non_citable_debug=[],
        )
        instrumentation.update_retrieval_evidence_metadata(
            run_meta=run_meta,
            references_structured_all=references_structured_all,
            total_retrieved=total_retrieved,
            required_anchors_payload=required_anchors_payload,
        )

        # Snapshot A: after retrieval + candidate/reference construction.
        try:
            run_meta["effective_where"] = deepcopy(
                getattr(self.retriever, "_last_effective_where", None)
            )
        except Exception:  # noqa: BLE001
            run_meta["effective_where"] = getattr(
                self.retriever, "_last_effective_where", None
            )
        instrumentation._debug_dump_run_meta(
            run_meta=run_meta,
            stage="after_retrieval_candidates",
            extra={
                "retrieved_ids_count": int(
                    len(getattr(self.retriever, "_last_retrieved_ids", []) or [])
                ),
                "retrieved_metadatas_count": int(
                    len(getattr(self.retriever, "_last_retrieved_metadatas", []) or [])
                ),
                "citable_count_total": int(citable_count_total),
            },
        )

        # Phase 4: kilder_block and context string built by prompt_builder
        kilder_block = prompt_ctx.kilder_block
        context = prompt_ctx.context_string

        # If the user explicitly asks what a specific recital says and we have it as a citable chunk,
        # return the source text directly (audit-safe, avoids LLM + distance noise).
        recital_ref = query_helpers._extract_recital_ref(question)
        if recital_ref and query_helpers._looks_like_recital_quote_question(question):
            for doc, meta, _chunk_id, _precise_override in included:
                if str((meta or {}).get("recital") or "").strip() == recital_ref:
                    answer_text = str(doc or "").strip()
                    answer_text = text_transforms._normalize_abstain_text(answer_text)
                    reference_lines = [
                        f"[{r['idx']}] {r.get('precise_ref') or r.get('display')}"
                        for r in references_structured_all
                    ]
                    return payload_builders.build_answer_response_payload(
                        run_meta=run_meta,
                        user_profile_value=resolved_profile.value,
                        focus=focus,
                        intent_value=plan.intent.value,
                        answer_text=answer_text,
                        references=references_structured_all,
                        reference_lines=reference_lines,
                        distances=distances,
                        retrieval_state=retrieval_state,
                        effective_plan=effective_plan,
                        where_for_retrieval=where_for_retrieval,
                        pass_tracker_passes=pass_tracker.get_passes(),
                        hybrid_rerank=hybrid_rerank,
                        sibling_expansion=self._last_sibling_expansion,
                    )

        # STAGE 3: Evidence gating + generation
        from .answer_stages import execute_evidence_gate_stage

        evidence_gate = execute_evidence_gate_stage(
            question=question,
            hits=hits,
            distances=distances,
            focus=focus,
            history_context=history_context,
            dry_run=dry_run,
            corpus_scope=corpus_scope,
            resolved_profile=resolved_profile,
            effective_plan=effective_plan,
            ctx=ctx,
            plan=plan,
            references_structured_all=references_structured_all,
            citable_count_total=citable_count_total,
            context=context,
            kilder_block=kilder_block,
            synthesis_context=synthesis_context,
            effective_policy=effective_policy,
            claim_intent_final=claim_intent_final,
            corpus_debug_on=corpus_debug_on,
            contract_min_citations=contract_min_citations,
            run_meta=run_meta,
            payload_context={
                "retrieval_state": retrieval_state,
                "where_for_retrieval": where_for_retrieval,
                "pass_tracker_passes": pass_tracker.get_passes(),
                "ranking_debug": ranking_debug_payload,
                "sibling_expansion": self._last_sibling_expansion,
            },
            should_abstain_fn=self._should_abstain,
            llm_fn=self._call_openai,
            resolver_fn=self._resolver,
        )
        if evidence_gate.early_return is not None:
            return evidence_gate.early_return

        answer_text = evidence_gate.answer_text
        did_abstain = evidence_gate.did_abstain
        bypass_required_support_gate = evidence_gate.bypass_required_support_gate
        min_citable_required = evidence_gate.min_citable_required

        # STAGE 4: Post-generation pipeline (policy gates, citations, normalization)
        from .answer_stages import execute_post_generation_stage

        post_gen = execute_post_generation_stage(
            answer_text=answer_text,
            question=question,
            resolved_profile=resolved_profile,
            run_meta=run_meta,
            references_structured_all=references_structured_all,
            distances=distances,
            effective_plan=effective_plan,
            effective_policy=effective_policy,
            claim_intent_final=claim_intent_final,
            did_abstain=did_abstain,
            bypass_required_support_gate=bypass_required_support_gate,
            min_citable_required=min_citable_required,
            ctx=ctx,
            total_retrieved=total_retrieved,
            citable_count_total=citable_count_total,
            contract_min_citations=contract_min_citations,
            corpus_debug_on=corpus_debug_on,
            max_distance=getattr(self, "max_distance", None),
            corpus_id=str(getattr(self, "corpus_id", "") or ""),
            project_root=getattr(self, "_project_root", None),
        )

        answer_text = post_gen.answer_text
        references_structured = post_gen.references_structured
        reference_lines = post_gen.reference_lines
        used_chunk_ids = post_gen.used_chunk_ids

        # Final snapshot
        try:
            run_meta.update(
                {
                    "references_structured_all_count": int(
                        len(references_structured_all or [])
                    ),
                    "references_structured_count": int(
                        len(references_structured or [])
                    ),
                    "reference_lines_count": int(len(reference_lines or [])),
                    "answer_is_missing_ref": bool(
                        str(answer_text or "").strip() == "MISSING_REF"
                    ),
                }
            )
        except Exception:  # noqa: BLE001
            pass
        instrumentation._debug_dump_run_meta(
            run_meta=run_meta,
            stage="final_payload",
            extra={
                "answer_preview": str(answer_text or "")[:160],
                "answer_is_missing_ref": run_meta.get("answer_is_missing_ref"),
            },
        )

        return payload_builders.build_answer_response_payload(
            run_meta=run_meta,
            user_profile_value=resolved_profile.value,
            focus=focus,
            intent_value=plan.intent.value,
            answer_text=answer_text,
            references=references_structured,
            reference_lines=reference_lines,
            distances=distances,
            retrieval_state=retrieval_state,
            effective_plan=effective_plan,
            where_for_retrieval=where_for_retrieval,
            pass_tracker_passes=pass_tracker.get_passes(),
            planner=planner_payload,
            references_structured_all=references_structured_all,
            used_chunk_ids=used_chunk_ids,
            hybrid_rerank=hybrid_rerank,
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

    def _call_openai(self, prompt: str) -> str:
        model = getattr(self, "chat_model", _get_default_chat_model())
        temperature = _get_default_temperature()
        return call_llm(prompt, model=model, temperature=temperature)
