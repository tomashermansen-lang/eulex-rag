"""Intent classification and LLM-based disambiguation.

This module consolidates all intent classification logic:
- Deterministic keyword heuristics (classify_question_intent)
- LLM-based semantic disambiguation (disambiguate_intent)
- Config-driven policy overrides (_apply_answer_policy_to_claim_intent)

Best practices (per OpenAI Guardrails Cookbook):
- Uses gpt-4o-mini for speed/cost optimization
- Simple, focused prompt
- Designed to run async parallel with retrieval (no latency impact)
"""

from __future__ import annotations

import os
import re
import hashlib
import time
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI, RateLimitError

from .constants import (
    _INTENT_ENFORCEMENT_KEYWORDS_SUBSTR,
    _INTENT_ENFORCEMENT_KEYWORDS_EXACT,
    _INTENT_REQUIREMENTS_KEYWORDS_STRONG_SUBSTR,
    _INTENT_REQUIREMENTS_KEYWORDS_WEAK_SUBSTR,
    _INTENT_REQUIREMENTS_KEYWORDS_VERBS,
    _INTENT_CLASSIFICATION_KEYWORDS_SUBSTR,
    _INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR,
)
from .conversation import HistoryMessage
from .types import ClaimIntent, UserProfile
from .concept_config import Policy as AnchorPolicy
from ..common.corpus_registry import normalize_alias
from .corpus_resolver import load_resolver_for_project_root


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
                time.sleep(2**attempt)
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


# ---------------------------------------------------------------------------
# Keyword-based intent classification (moved from policy.py, Phase 8a)
# ---------------------------------------------------------------------------


def _intent_match_signals(prompt_text: str) -> dict[str, bool]:
    q = str(prompt_text or "").strip().lower()
    if not q:
        return {
            "enforcement": False,
            "requirements": False,
            "classification": False,
            "scope": False,
        }

    enforcement = any(k in q for k in _INTENT_ENFORCEMENT_KEYWORDS_SUBSTR) or any(
        re.search(rf"(?i)\\b{re.escape(w)}\\b", q)
        for w in _INTENT_ENFORCEMENT_KEYWORDS_EXACT
    )
    requirements = any(k in q for k in _INTENT_REQUIREMENTS_KEYWORDS_STRONG_SUBSTR) or (
        any(k in q for k in _INTENT_REQUIREMENTS_KEYWORDS_WEAK_SUBSTR)
        and any(
            k in q
            for k in (
                "krav",
                "kræver",
                "kræves",
                "skal",
                "must",
                "should",
                "hvordan",
                "overhold",
                "efterlev",
                "implement",
            )
        )
    )
    classification = any(k in q for k in _INTENT_CLASSIFICATION_KEYWORDS_SUBSTR)
    scope = any(k in q for k in _INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR)
    return {
        "enforcement": bool(enforcement),
        "requirements": bool(requirements),
        "classification": bool(classification),
        "scope": bool(scope),
    }


def _detect_intent_cues(prompt_text: str) -> dict[str, Any]:
    """Return matched cue tokens for observability.

    This is *not* a new classifier; it mirrors existing deterministic heuristics
    but provides explainability (matched tokens) for audit/debug.
    """

    q = str(prompt_text or "").strip().lower()
    if not q:
        return {
            "requirements_cues_detected": False,
            "requirements_cues_matched": [],
            "enforcement_cues_detected": False,
            "enforcement_cues_matched": [],
        }

    enforcement_matched = [k for k in _INTENT_ENFORCEMENT_KEYWORDS_SUBSTR if k in q]
    enforcement_word_matched: list[str] = []
    for w in _INTENT_ENFORCEMENT_KEYWORDS_EXACT:
        try:
            if re.search(rf"(?i)\b{re.escape(w)}\b", q):
                enforcement_word_matched.append(str(w))
        except Exception:  # noqa: BLE001
            if str(w).lower() in q:
                enforcement_word_matched.append(str(w))
    enforcement_matched = sorted(set([*enforcement_matched, *enforcement_word_matched]))

    req_strong = [k for k in _INTENT_REQUIREMENTS_KEYWORDS_STRONG_SUBSTR if k in q]
    req_weak = [k for k in _INTENT_REQUIREMENTS_KEYWORDS_WEAK_SUBSTR if k in q]
    req_verbs = [k for k in _INTENT_REQUIREMENTS_KEYWORDS_VERBS if k in q]
    requirements_detected = bool(req_strong) or (bool(req_weak) and bool(req_verbs))
    requirements_matched = sorted(set([*req_strong, *req_weak, *req_verbs]))

    return {
        "requirements_cues_detected": bool(requirements_detected),
        "requirements_cues_matched": list(requirements_matched),
        "enforcement_cues_detected": bool(enforcement_matched),
        "enforcement_cues_matched": list(enforcement_matched),
    }


def classify_question_intent(prompt_text: str) -> ClaimIntent:
    """Classify the user's intent for claim-stage gating.

    Deterministic, heuristic-only, and intentionally small.
    """

    q = str(prompt_text or "").strip().lower()
    if not q:
        return ClaimIntent.GENERAL

    # NOTE: Order matters. We prefer the most safety-sensitive intents first.
    signals = _intent_match_signals(q)
    if signals["enforcement"]:
        return ClaimIntent.ENFORCEMENT

    if signals["requirements"]:
        return ClaimIntent.REQUIREMENTS

    if signals["classification"]:
        return ClaimIntent.CLASSIFICATION

    # SCOPE is specifically about the law's applicability/anvendelsesområde.
    # Avoid treating phrases like "Hvornår gælder retten til ..." as scope.
    if any(k in q for k in _INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR):
        return ClaimIntent.SCOPE

    if "gælder" in q:
        # Generic: treat "gælder <law/corpus>" as scope when the question explicitly
        # mentions any known corpus alias/display name from the registry.
        try:
            project_root = Path(__file__).resolve().parents[2]
            resolver = load_resolver_for_project_root(str(project_root))
            if resolver.any_alias_in(normalize_alias(q)):
                return ClaimIntent.SCOPE
        except Exception:  # noqa: BLE001
            pass

    return ClaimIntent.GENERAL


def classify_question_intent_with_router(
    prompt_text: str,
    *,
    enable_router: bool = True,
    last_exchange: list | None = None,
    query_was_rewritten: bool = False,
) -> tuple[ClaimIntent, dict]:
    """Classify intent using keyword heuristics + LLM router for disambiguation.

    This is the recommended function to use. It:
    1. Uses fast keyword heuristics to get candidate intent
    2. If candidate is a gated intent (CLASSIFICATION, ENFORCEMENT, REQUIREMENTS, SCOPE),
       calls LLM router to check if question is about LAW_CONTENT vs USER_SYSTEM
    3. Overrides to GENERAL if question is about law content (not user's own system)

    Args:
        prompt_text: The user's question
        enable_router: If False, skip LLM call (useful for testing)
        last_exchange: Optional last user+assistant exchange for context augmentation.
        query_was_rewritten: Whether the query was changed by the rewriter.

    Returns:
        Tuple of (final_intent, debug_info)
    """
    # First: fast keyword heuristics
    candidate = classify_question_intent(prompt_text)

    # Then: LLM disambiguation if gated
    return disambiguate_intent(
        prompt_text,
        candidate,
        enable_router=enable_router,
        last_exchange=last_exchange,
        query_was_rewritten=query_was_rewritten,
    )


def _apply_answer_policy_to_claim_intent(
    *,
    resolved_profile: UserProfile,
    classifier_intent: ClaimIntent,
    policy: AnchorPolicy | None,
    question: str | None = None,
) -> tuple[ClaimIntent, dict[str, Any]]:
    """Apply config-driven answer_policy to claim-stage intent.

    This is used to prevent off-topic ENGINEERING answers when retrieval is good but
    heuristic enforcement signals would otherwise override requirements-oriented planning.
    """

    dbg: dict[str, Any] = {
        "classifier_intent": str(
            getattr(classifier_intent, "value", classifier_intent) or ""
        ),
        "policy_present": bool(policy is not None),
        "policy_intent_category": None,
        "final_intent": str(
            getattr(classifier_intent, "value", classifier_intent) or ""
        ),
        "override_applied": False,
    }

    if policy is None:
        return classifier_intent, dbg

    # If policy explicitly sets intent_category, we may override the classifier.
    # Currently only supports overriding ENFORCEMENT -> REQUIREMENTS if the policy says so.
    ap = getattr(policy, "answer_policy", None)
    if ap is None:
        return classifier_intent, dbg

    policy_intent = str(getattr(ap, "intent_category", "") or "").strip().upper()
    dbg["policy_intent_category"] = policy_intent

    if not policy_intent:
        return classifier_intent, dbg

    if classifier_intent == ClaimIntent.ENFORCEMENT and policy_intent == "REQUIREMENTS":
        # Override: The user asked about enforcement (e.g. "bøde"), but the policy
        # dictates this is a requirements question (e.g. "Hvad er kravene?").
        # This happens when enforcement keywords appear in a requirements context.
        dbg["override_applied"] = True
        dbg["final_intent"] = "REQUIREMENTS"
        dbg["requirements_cues_detected"] = True
        return ClaimIntent.REQUIREMENTS, dbg

    return classifier_intent, dbg


# ---------------------------------------------------------------------------
# LLM-based disambiguation
# ---------------------------------------------------------------------------


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
