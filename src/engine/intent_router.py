"""LLM-based intent router for semantic disambiguation.

This module provides a lightweight LLM call to disambiguate between:
- USER_SYSTEM: Questions about the user's own system/situation (needs strict gating)
- LAW_CONTENT: Questions about the law's content itself (general RAG retrieval)

Best practices (per OpenAI Guardrails Cookbook):
- Uses gpt-4o-mini for speed/cost optimization
- Simple, focused prompt
- Designed to run async parallel with retrieval (no latency impact)
"""

from __future__ import annotations

import os
import hashlib
import time
from typing import Optional
from functools import lru_cache

from openai import OpenAI, RateLimitError

from .conversation import HistoryMessage
from .types import ClaimIntent


# In-memory cache for intent routing (question_hash -> result)
_INTENT_ROUTER_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour


def _get_cache_key(question: str, context: str | None = None) -> str:
    """Generate cache key from normalized question and optional context."""
    normalized = question.strip().lower()
    if context is not None:
        normalized += "\n---\n" + context.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()


def _get_cached_result(question: str, context: str | None = None) -> Optional[str]:
    """Get cached result if still valid."""
    key = _get_cache_key(question, context)
    if key in _INTENT_ROUTER_CACHE:
        result, timestamp = _INTENT_ROUTER_CACHE[key]
        if time.time() - timestamp < _CACHE_TTL_SECONDS:
            return result
        # Expired - remove
        del _INTENT_ROUTER_CACHE[key]
    return None


def _cache_result(question: str, result: str, context: str | None = None) -> None:
    """Cache result with timestamp."""
    key = _get_cache_key(question, context)
    _INTENT_ROUTER_CACHE[key] = (result, time.time())


# Router prompt - simple and focused per best practices
_ROUTER_PROMPT = """Classify this legal question's intent type.

QUESTION: {question}

Determine if the user is asking:
A) About THEIR OWN system, product, or situation - needs assessment of whether THEY comply/apply
   Examples: "Is my system prohibited?", "Does GDPR apply to us?", "What must we comply with?"
   
B) About the LAW'S CONTENT itself - asking what the law says, defines, or requires in general
   Examples: "What is prohibited under Article 5?", "What are the GDPR penalties?", "How does the law define AI?"

Reply with EXACTLY one word:
- "USER_SYSTEM" if asking about their own situation
- "LAW_CONTENT" if asking about what the law says

Answer:"""


# Context-augmented version of the router prompt for multi-turn conversations.
# Used when the rewriter could not fully resolve ambiguity (short/unchanged query).
_ROUTER_PROMPT_WITH_CONTEXT = """Classify this legal question's intent type.

CONVERSATION CONTEXT (recent exchange for reference):
{context}

QUESTION: {question}

Determine if the user is asking:
A) About THEIR OWN system, product, or situation - needs assessment of whether THEY comply/apply
   Examples: "Is my system prohibited?", "Does GDPR apply to us?", "What must we comply with?"

B) About the LAW'S CONTENT itself - asking what the law says, defines, or requires in general
   Examples: "What is prohibited under Article 5?", "What are the GDPR penalties?", "How does the law define AI?"

Reply with EXACTLY one word:
- "USER_SYSTEM" if asking about their own situation
- "LAW_CONTENT" if asking about what the law says

Answer:"""


def _get_router_model() -> str:
    """Get model for intent routing from env or default."""
    return os.getenv("INTENT_ROUTER_MODEL", "gpt-4o-mini")


def _call_router_llm(question: str, context: str | None = None) -> str:
    """Make LLM call to classify question intent type.

    Args:
        question: The user's question.
        context: Optional conversation context (last exchange) for ambiguous queries.

    Returns: "USER_SYSTEM" or "LAW_CONTENT"
    """
    # Check cache first
    cached = _get_cached_result(question, context)
    if cached is not None:
        return cached

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # No API key - fallback to heuristic (no override)
        return "USER_SYSTEM"

    client = OpenAI(api_key=api_key)
    model = _get_router_model()
    if context is not None:
        prompt = _ROUTER_PROMPT_WITH_CONTEXT.format(question=question, context=context)
    else:
        prompt = _ROUTER_PROMPT.format(question=question)
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            result = response.choices[0].message.content.strip().upper()
            
            # Normalize to expected values
            if "LAW" in result or "CONTENT" in result:
                result = "LAW_CONTENT"
            else:
                result = "USER_SYSTEM"
            
            # Cache the result
            _cache_result(question, result, context)
            return result
            
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            # Rate limit - conservative fallback
            return "USER_SYSTEM"
        except Exception:  # noqa: BLE001
            # Any error - conservative fallback (keep gated intent)
            return "USER_SYSTEM"
        finally:
            close_callable = getattr(client, "close", None)
            if callable(close_callable):
                try:
                    close_callable()
                except Exception:  # noqa: BLE001
                    pass
    
    return "USER_SYSTEM"


def _format_exchange_as_context(exchange: list[HistoryMessage]) -> str:
    """Format a last-exchange list as a context string for the router prompt."""
    lines = []
    for msg in exchange:
        label = "User" if msg.role == "user" else "Assistant"
        lines.append(f"{label}: {msg.content}")
    return "\n".join(lines)


def disambiguate_intent(
    question: str,
    candidate_intent: ClaimIntent,
    *,
    enable_router: bool = True,
    last_exchange: list[HistoryMessage] | None = None,
    query_was_rewritten: bool = False,
) -> tuple[ClaimIntent, dict]:
    """Disambiguate a gated intent using LLM semantic understanding.

    If the keyword heuristic matched a gated intent (CLASSIFICATION, ENFORCEMENT,
    REQUIREMENTS, SCOPE) but the question is actually about LAW CONTENT rather than
    the user's own system, we override to GENERAL.

    Args:
        question: The user's question (possibly rewritten).
        candidate_intent: The intent from keyword heuristics.
        enable_router: If False, skip LLM call and return candidate as-is.
        last_exchange: Optional last user+assistant exchange for context augmentation.
        query_was_rewritten: Whether the query was changed by the rewriter.

    Returns:
        Tuple of (final_intent, debug_info)
    """
    # Determine if context augmentation is needed
    original_query = question if not query_was_rewritten else None
    # If query_was_rewritten is True, the rewriter changed it → original differs.
    # If query_was_rewritten is False AND last_exchange is provided, it's a follow-up
    # that the rewriter could not resolve → needs augmentation.
    use_context = False
    if last_exchange:
        if not query_was_rewritten:
            # Rewriter returned unchanged → needs context
            use_context = True
        elif len(question) < 40:
            # Rewritten but still short → needs context
            use_context = True

    debug_info = {
        "router_enabled": enable_router,
        "candidate_intent": str(candidate_intent.value),
        "router_called": False,
        "router_result": None,
        "override_applied": False,
        "final_intent": str(candidate_intent.value),
        "context_augmented": False,
        "query_was_rewritten": query_was_rewritten,
    }
    
    # Route all gated intents - they all can have false positives when
    # the question is about law content rather than user's own system
    gated_intents = {
        ClaimIntent.CLASSIFICATION,
        ClaimIntent.ENFORCEMENT,
        ClaimIntent.REQUIREMENTS,
        ClaimIntent.SCOPE,
    }

    # Also route GENERAL when context suggests a user_system follow-up.
    # Example: "og hvad med GDPR?" after "Er min chatbot et højrisiko-system?"
    # has no user_system keywords (→ GENERAL), but context reveals it IS about
    # the user's system in a new legal domain.
    needs_router = candidate_intent in gated_intents or (
        candidate_intent == ClaimIntent.GENERAL and use_context
    )

    if not needs_router:
        return candidate_intent, debug_info

    if not enable_router:
        return candidate_intent, debug_info

    # Check if router is disabled via env
    if os.getenv("INTENT_ROUTER_DISABLED", "").lower() in {"1", "true", "yes"}:
        debug_info["router_enabled"] = False
        return candidate_intent, debug_info

    # Call LLM router
    debug_info["router_called"] = True
    context_str: str | None = None
    if use_context and last_exchange:
        context_str = _format_exchange_as_context(last_exchange)
        debug_info["context_augmented"] = True
    router_result = _call_router_llm(question, context=context_str)
    debug_info["router_result"] = router_result

    if candidate_intent in gated_intents:
        # Gated intent: override to GENERAL if law content
        if router_result == "LAW_CONTENT":
            debug_info["override_applied"] = True
            debug_info["final_intent"] = "GENERAL"
            return ClaimIntent.GENERAL, debug_info
        # Keep the gated intent
        return candidate_intent, debug_info

    # GENERAL with context: override to CLASSIFICATION if user_system
    if router_result == "USER_SYSTEM":
        debug_info["override_applied"] = True
        debug_info["final_intent"] = str(ClaimIntent.CLASSIFICATION.value)
        return ClaimIntent.CLASSIFICATION, debug_info

    # Keep GENERAL
    return candidate_intent, debug_info


def clear_cache() -> int:
    """Clear the intent router cache. Returns number of entries cleared."""
    count = len(_INTENT_ROUTER_CACHE)
    _INTENT_ROUTER_CACHE.clear()
    return count
