from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from ..engine.rag import RAGEngine
from ..engine.planning import UserProfile
from ..engine.conversation import HistoryMessage, format_history_for_prompt, rewrite_query_for_retrieval, last_exchange
from ..common.config_loader import Settings, load_settings


@dataclass(frozen=True)
class AskResult:
    answer: str
    references: list[str]
    references_structured: list[dict[str, Any]]
    retrieval_metrics: dict[str, Any]


@dataclass(frozen=True)
class RetrievalOnlyResult:
    """Result from retrieval-only mode (no LLM generation)."""
    retrieved_ids: list[str]
    retrieved_metadatas: list[dict[str, Any]]
    distances: list[float]
    top_k: int
    corpus_id: str
    user_profile: str


@dataclass(frozen=True)
class RetrievalAugmentedResult:
    """Result from retrieval + augmentation mode (no LLM generation).
    
    Shows what the LLM would receive after citation graph expansion.
    """
    retrieved_ids: list[str]
    retrieved_metadatas: list[dict[str, Any]]
    distances: list[float]
    top_k: int
    corpus_id: str
    user_profile: str
    # Augmentation details
    citation_expansion_articles: list[str]
    augmented_metadatas: list[dict[str, Any]]  # After expansion


def build_engine(*, law: str, top_k: int | None = None, settings: Settings | None = None, corpus_scope: str = "single") -> RAGEngine:
    resolved_settings = settings or load_settings()
    corpora = resolved_settings.corpora or {}

    # For discover scope, use first available corpus as bootstrap engine
    # (the actual corpus is determined by the discovery classifier in rag.py)
    if corpus_scope == "discover" and law not in corpora:
        if not corpora:
            raise ValueError("No corpora configured.")
        law = next(iter(corpora))

    if law not in corpora:
        raise ValueError(f"Unknown law/corpus '{law}'. Available: {', '.join(sorted(corpora.keys()))}")

    corpus = corpora[law]
    return RAGEngine(
        docs_path=str(resolved_settings.docs_path),
        corpus_id=law,
        chunks_collection=corpus.chunks_collection,
        embedding_model=resolved_settings.embedding_model,
        chat_model=resolved_settings.chat_model,
        # Note: top_k is now dynamic via retrieval_pool_size and max_context_* from config
        top_k=top_k,  # Only pass if explicitly provided, otherwise engine uses config defaults
        vector_store_path=str(resolved_settings.vector_store_path),
        max_distance=(corpus.max_distance if corpus.max_distance is not None else resolved_settings.rag_max_distance),
        hybrid_vec_k=resolved_settings.hybrid_vec_k,
        ranking_weights=resolved_settings.ranking_weights,
    )


def ask(
    *,
    question: str,
    law: str,
    user_profile: str | UserProfile = UserProfile.LEGAL,
    top_k: int | None = None,
    contract_min_citations: int | None = None,
    engine: Optional[RAGEngine] = None,
    settings: Settings | None = None,
    dry_run: bool = False,
    history: list[HistoryMessage] | None = None,
    corpus_scope: str = "single",
    target_corpora: list[str] | None = None,
) -> AskResult:
    """Ask a question and get an answer with references.

    Args:
        dry_run: If True, run full pre-LLM pipeline but skip LLM call.
                 EVAL = PROD: uses exact same code path, just stops before LLM.
        history: Conversation history for context (optional).
        corpus_scope: Corpus search scope: "single" (default), "explicit", or "all".
        target_corpora: List of corpus IDs for "explicit" scope.
    """
    resolved_engine = engine or build_engine(law=law, top_k=top_k, settings=settings, corpus_scope=corpus_scope)

    resolved_profile: UserProfile
    if isinstance(user_profile, UserProfile):
        resolved_profile = user_profile
    else:
        raw = str(user_profile or "").strip().upper()
        resolved_profile = UserProfile.ENGINEERING if raw in {"ENGINEERING", "DEV", "DEVELOPER"} else UserProfile.LEGAL

    # Format conversation history for prompt injection
    history_context = format_history_for_prompt(history)

    # Rewrite query for better retrieval (industry standard for conversational RAG)
    retrieval_question = rewrite_query_for_retrieval(question, history)

    # Compute context for intent classification
    exchange = last_exchange(history)

    try:
        payload: Dict[str, Any] = resolved_engine.answer_structured(
            retrieval_question,  # Use rewritten query for retrieval
            user_profile=resolved_profile,
            contract_min_citations=contract_min_citations,
            dry_run=dry_run,
            history_context=history_context,
            corpus_scope=corpus_scope,
            target_corpora=target_corpora,
            original_query=question,
            last_exchange=exchange if exchange else None,
        )
    except TypeError:
        # Some tests and external callers use a fake engine with the old signature.
        payload = resolved_engine.answer_structured(question)

    distances = payload.get("retrieval", {}).get("distances", []) or []
    best_distance = min(distances) if distances else None

    used_collection = payload.get("retrieval", {}).get("query_collection")
    used_where = payload.get("retrieval", {}).get("query_where")
    retrieved_ids = payload.get("retrieval", {}).get("retrieved_ids", []) or []
    retrieved_metadatas = payload.get("retrieval", {}).get("retrieved_metadatas", []) or []
    role_rerank = payload.get("retrieval", {}).get("role_rerank")
    planned_where = payload.get("retrieval", {}).get("planned_where")
    effective_where = payload.get("retrieval", {}).get("effective_where")
    planned_collection_type = payload.get("retrieval", {}).get("planned_collection_type")
    effective_collection = payload.get("retrieval", {}).get("effective_collection")
    effective_collection_type = payload.get("retrieval", {}).get("effective_collection_type")
    passes = payload.get("retrieval", {}).get("passes")
    run_meta = payload.get("run")

    retrieval_metrics: dict[str, Any] = {
        "run": run_meta,
        "best_distance": best_distance,
        "distances": distances,
        "top_k": getattr(resolved_engine, "top_k", top_k),
        "used_law": getattr(resolved_engine, "corpus_id", law),
        "user_profile": str(resolved_profile.value),
        "used_collection": used_collection,
        "used_where": used_where,
        "retrieved_ids": retrieved_ids,
        "retrieved_metadatas": retrieved_metadatas,
        "planned_where": planned_where,
        "effective_where": effective_where,
        "planned_collection_type": planned_collection_type,
        "effective_collection": effective_collection,
        "effective_collection_type": effective_collection_type,
        "passes": passes,
        "role_rerank": role_rerank,
        "anchor_ranking": payload.get("retrieval", {}).get("anchor_ranking"),
        "references_used_in_answer": payload.get("retrieval", {}).get("references_used_in_answer"),
        "references_structured_all": payload.get("retrieval", {}).get("references_structured_all"),
        "hybrid_rerank": payload.get("retrieval", {}).get("hybrid_rerank"),
        "planner": payload.get("retrieval", {}).get("planner"),
        "sibling_expansion": payload.get("retrieval", {}).get("sibling_expansion"),
    }

    return AskResult(
        answer=str(payload.get("answer") or ""),
        references=list(payload.get("reference_lines") or []),
        references_structured=list(payload.get("references") or []),
        retrieval_metrics=retrieval_metrics,
    )


def ask_stream(
    *,
    question: str,
    law: str,
    user_profile: str | UserProfile = UserProfile.LEGAL,
    top_k: int | None = None,
    engine: Optional[RAGEngine] = None,
    settings: Settings | None = None,
    history: list[HistoryMessage] | None = None,
    corpus_scope: str = "single",
    target_corpora: list[str] | None = None,
) -> Iterator[str | AskResult]:
    """Stream the answer while generating, then yield final AskResult.

    This function provides a streaming interface for better UX:
    - First yields str chunks as the LLM generates the answer
    - Finally yields an AskResult with the complete answer and references

    IMPORTANT: Uses answer_structured(dry_run=True) for retrieval to ensure
    identical retrieval pipeline as non-streaming (eval, CLI). Only the LLM
    generation is streamed.

    Args:
        corpus_scope: Corpus search scope: "single" (default), "explicit", or "all".
        target_corpora: List of corpus IDs for "explicit" scope.

    Usage:
        accumulated = ""
        result = None
        for chunk in ask_stream(question=q, law=l):
            if isinstance(chunk, str):
                accumulated += chunk
                # Display chunk to user
            else:
                result = chunk  # Final AskResult
    """
    from ..engine.llm_client import call_llm, call_llm_stream

    resolved_engine = engine or build_engine(law=law, top_k=top_k, settings=settings, corpus_scope=corpus_scope)

    resolved_profile: UserProfile
    if isinstance(user_profile, UserProfile):
        resolved_profile = user_profile
    else:
        raw = str(user_profile or "").strip().upper()
        resolved_profile = UserProfile.ENGINEERING if raw in {"ENGINEERING", "DEV", "DEVELOPER"} else UserProfile.LEGAL

    # Format conversation history for prompt injection
    history_context = format_history_for_prompt(history)

    # Rewrite query for better retrieval (industry standard for conversational RAG)
    retrieval_question = rewrite_query_for_retrieval(question, history)

    # Compute context for intent classification
    exchange = last_exchange(history)

    # Step 1: Run full retrieval pipeline with dry_run=True (skips LLM call)
    # This ensures streaming uses IDENTICAL retrieval as eval/CLI (citation boost,
    # hybrid rerank, role rerank, etc.)
    try:
        retrieval_payload: Dict[str, Any] = resolved_engine.answer_structured(
            retrieval_question,  # Use rewritten query for retrieval
            user_profile=resolved_profile,
            dry_run=True,  # Full pipeline, no LLM call
            history_context=history_context,
            corpus_scope=corpus_scope,
            target_corpora=target_corpora,
            original_query=question,
            last_exchange=exchange if exchange else None,
        )
    except (AttributeError, TypeError):
        # Fallback: engine doesn't support dry_run, use regular ask
        yield ask(
            question=question,
            law=law,
            user_profile=user_profile,
            top_k=top_k,
            engine=engine,
            settings=settings,
            history=history,
            corpus_scope=corpus_scope,
            target_corpora=target_corpora,
        )
        return

    prompt = retrieval_payload.get("prompt", "")
    references_structured = retrieval_payload.get("references", [])
    reference_lines = retrieval_payload.get("reference_lines", [])
    retrieval_data = retrieval_payload.get("retrieval", {})
    run_meta = retrieval_payload.get("run", {})

    # Check if we abstained (no LLM call needed)
    # BUT: If we have conversation history, don't abstain - let LLM use history context
    abstain_info = run_meta.get("abstain", {})
    has_history = bool(history and len(history) > 0)

    if abstain_info.get("abstained") and not has_history:
        # Don't call LLM - use the abstain reason as the answer
        accumulated_answer = str(abstain_info.get("reason", "Kan ikke besvare spørgsmålet baseret på de tilgængelige kilder."))
        yield accumulated_answer
    else:
        # Step 2: Stream the LLM answer
        accumulated_answer = ""
        try:
            for chunk in call_llm_stream(prompt):
                accumulated_answer += chunk
                yield chunk
        except Exception:
            # Fallback to non-streaming if streaming fails
            accumulated_answer = call_llm(prompt)
            yield accumulated_answer

    # Step 3: Build and yield final AskResult with full retrieval metrics
    distances = retrieval_data.get("distances", []) or []
    best_distance = min(distances) if distances else None

    retrieval_metrics: dict[str, Any] = {
        "run": run_meta,  # Full run metadata from answer_structured
        "best_distance": best_distance,
        "distances": distances,
        "top_k": getattr(resolved_engine, "top_k", top_k),
        "used_law": getattr(resolved_engine, "corpus_id", law),
        "user_profile": str(resolved_profile.value),
        "used_collection": retrieval_data.get("query_collection"),
        "used_where": retrieval_data.get("query_where"),
        "retrieved_ids": retrieval_data.get("retrieved_ids", []),
        "retrieved_metadatas": retrieval_data.get("retrieved_metadatas", []),
        "planned_where": retrieval_data.get("planned_where"),
        "effective_where": retrieval_data.get("effective_where"),
        "planned_collection_type": retrieval_data.get("planned_collection_type"),
        "effective_collection": retrieval_data.get("effective_collection"),
        "effective_collection_type": retrieval_data.get("effective_collection_type"),
        "passes": retrieval_data.get("passes"),
        "role_rerank": retrieval_data.get("role_rerank"),
        "anchor_ranking": retrieval_data.get("anchor_ranking"),
        "hybrid_rerank": retrieval_data.get("hybrid_rerank"),
        "sibling_expansion": retrieval_data.get("sibling_expansion"),
        "streaming": True,
    }

    yield AskResult(
        answer=accumulated_answer.strip(),
        references=list(reference_lines or []),
        references_structured=list(references_structured or []),
        retrieval_metrics=retrieval_metrics,
    )


def retrieve_only(
    *,
    question: str,
    law: str,
    user_profile: str | UserProfile = UserProfile.LEGAL,
    top_k: int | None = None,
    engine: Optional[RAGEngine] = None,
    settings: Settings | None = None,
) -> RetrievalOnlyResult:
    """Run retrieval only (no LLM generation). Much faster for debugging retrieval issues."""
    resolved_engine = engine or build_engine(law=law, top_k=top_k, settings=settings)

    resolved_profile: UserProfile
    if isinstance(user_profile, UserProfile):
        resolved_profile = user_profile
    else:
        raw = str(user_profile or "").strip().upper()
        resolved_profile = UserProfile.ENGINEERING if raw in {"ENGINEERING", "DEV", "DEVELOPER"} else UserProfile.LEGAL

    # Use the retriever directly to get chunks without LLM
    effective_top_k = top_k if top_k is not None else int(getattr(resolved_engine, "top_k", 10))
    
    # Build where clause for corpus
    where: Dict[str, Any] = {"corpus_id": law}
    
    # Query the collection
    hits, distances = resolved_engine._query_collection_with_distances(
        collection=resolved_engine.collection,
        question=question,
        k=effective_top_k,
        where=where,
    )
    
    retrieved_ids = [h[0] for h in hits] if hits else []
    retrieved_metadatas = [h[1] for h in hits] if hits else []

    return RetrievalOnlyResult(
        retrieved_ids=retrieved_ids,
        retrieved_metadatas=retrieved_metadatas,
        distances=distances,
        top_k=effective_top_k,
        corpus_id=law,
        user_profile=str(resolved_profile.value),
    )


def retrieve_augmented(
    *,
    question: str,
    law: str,
    user_profile: str | UserProfile = UserProfile.LEGAL,
    top_k: int | None = None,
    engine: Optional[RAGEngine] = None,
    settings: Settings | None = None,
) -> RetrievalAugmentedResult:
    """Run retrieval + citation graph augmentation (no LLM generation).
    
    Shows exactly what the LLM would receive after all post-retrieval processing.
    Faster than full eval for debugging augmentation issues.
    """
    from ..engine.citation_expansion import get_citation_expansion_for_query, is_citation_expansion_enabled
    
    resolved_engine = engine or build_engine(law=law, top_k=top_k, settings=settings)

    resolved_profile: UserProfile
    if isinstance(user_profile, UserProfile):
        resolved_profile = user_profile
    else:
        raw = str(user_profile or "").strip().upper()
        resolved_profile = UserProfile.ENGINEERING if raw in {"ENGINEERING", "DEV", "DEVELOPER"} else UserProfile.LEGAL

    effective_top_k = top_k if top_k is not None else int(getattr(resolved_engine, "top_k", 10))
    
    # Build where clause for corpus
    where: Dict[str, Any] = {"corpus_id": law}
    
    # Query the collection (initial retrieval)
    hits, distances = resolved_engine._query_collection_with_distances(
        collection=resolved_engine.collection,
        question=question,
        k=effective_top_k,
        where=where,
    )
    
    retrieved_ids = [h[0] for h in hits] if hits else []
    retrieved_metadatas = [h[1] for h in hits] if hits else []
    
    # Apply citation graph expansion (same as in answer_structured)
    citation_expansion_articles: list[str] = []
    augmented_metadatas = list(retrieved_metadatas)  # Start with original
    
    if is_citation_expansion_enabled() and retrieved_metadatas:
        try:
            citation_expansion_articles = list(get_citation_expansion_for_query(
                corpus_id=law,
                question=question,
                retrieved_metadatas=retrieved_metadatas,
            ) or [])
            
            # Inject expanded articles as additional "virtual" metadatas
            # (This simulates what the full pipeline does)
            for art in citation_expansion_articles:
                art_str = str(art)
                # Check if already in retrieved metadatas
                already_present = False
                for m in augmented_metadatas:
                    if not isinstance(m, dict):
                        continue
                    m_art = str(m.get("article") or "").strip()
                    m_anx = str(m.get("annex") or "").strip()
                    if art_str.upper().startswith("ANNEX:"):
                        anx_val = art_str.split(":", 1)[1] if ":" in art_str else ""
                        if m_anx.upper() == anx_val.upper():
                            already_present = True
                            break
                    elif m_art == art_str:
                        already_present = True
                        break
                
                if not already_present:
                    # Add a synthetic metadata entry to show expansion
                    if art_str.upper().startswith("ANNEX:"):
                        anx_val = art_str.split(":", 1)[1] if ":" in art_str else art_str
                        augmented_metadatas.append({
                            "annex": anx_val,
                            "_augmented": True,
                            "_source": "citation_graph",
                        })
                    else:
                        augmented_metadatas.append({
                            "article": art_str,
                            "_augmented": True,
                            "_source": "citation_graph",
                        })
        except Exception:
            pass
    
    return RetrievalAugmentedResult(
        retrieved_ids=retrieved_ids,
        retrieved_metadatas=retrieved_metadatas,
        distances=distances,
        top_k=effective_top_k,
        corpus_id=law,
        user_profile=str(resolved_profile.value),
        citation_expansion_articles=citation_expansion_articles,
        augmented_metadatas=augmented_metadatas,
    )
