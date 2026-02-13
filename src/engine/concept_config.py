from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..common.config_loader import (
    load_corpus_config,
    get_concept_bump_hints,
    get_concept_answer_policy,
    get_concept_normative_guard,
    get_default_bump_hints,
)

logger = logging.getLogger(__name__)


_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _normalize_anchor(anchor: str) -> str:
    raw = str(anchor or "").strip().lower()
    raw = re.sub(r"\s+", "", raw)
    return raw


def _normalize_intent_key(intent_key: str) -> str:
    raw = str(intent_key or "").strip().lower()
    # Strip legacy prefixes for backwards compatibility
    if raw.startswith("legalconcept."):
        raw = raw[len("legalconcept."):]
    raw = re.sub(r"\s+", "", raw)
    return raw


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in list(items or []):
        sx = str(x or "")
        if not sx:
            continue
        if sx in seen:
            continue
        seen.add(sx)
        out.append(sx)
    return out


def _normalize_anchor_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for a in value:
        if not isinstance(a, str):
            continue
        na = _normalize_anchor(a)
        if na and ":" in na:
            out.append(na)
    return _dedupe_preserve_order(out)


def _normalize_profile_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("ANY",)
    if not isinstance(value, list):
        return ("ANY",)
    out: list[str] = []
    for p in value:
        if not isinstance(p, str):
            continue
        pp = str(p).strip().upper()
        if not pp:
            continue
        if pp not in {"ANY", "LEGAL", "ENGINEERING"}:
            continue
        out.append(pp)
    out = _dedupe_preserve_order(out)
    return tuple(out) if out else ("ANY",)


@dataclass(frozen=True)
class NormativeGuardPolicy:
    # required_support governs what evidence must exist in references_structured.
    # - "article": require at least one article-backed reference
    # - "article_or_annex": allow annex-only support
    # - "any": disable this guard (always satisfied)
    required_support: str
    profiles: tuple[str, ...] = ("ANY",)

    def applies_to_profile(self, profile: str) -> bool:
        p = str(profile or "").strip().upper()
        return ("ANY" in self.profiles) or (p in self.profiles)


@dataclass(frozen=True)
class RescueRule:
    if_present: tuple[str, ...]
    must_include_one_of: tuple[str, ...]
    action: str
    profiles: tuple[str, ...] = ("ANY",)

    def applies_to_profile(self, profile: str) -> bool:
        p = str(profile or "").strip().upper()
        return ("ANY" in self.profiles) or (p in self.profiles)


@dataclass(frozen=True)
class AnswerPolicy:
    # Categorizes an intent key for ENGINEERING planning/generation.
    # - REQUIREMENTS: prefer requirements-style output structure
    # - ENFORCEMENT: prefer enforcement/sanctions output
    # - OTHER: no special override
    intent_category: str
    requirements_first: bool = False
    include_audit_evidence: bool = False
    min_section3_bullets: int | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "intent_category": str(self.intent_category),
            "requirements_first": bool(self.requirements_first),
            "include_audit_evidence": bool(self.include_audit_evidence),
            "min_section3_bullets": (int(self.min_section3_bullets) if self.min_section3_bullets is not None else None),
        }


@dataclass(frozen=True)
class Policy:
    bump_hints: tuple[str, ...] = ()
    normative_guard: NormativeGuardPolicy | None = None
    rescue_rules: tuple[RescueRule, ...] = ()
    answer_policy: AnswerPolicy | None = None
    # Debug surface: which intent keys contributed.
    intent_keys_requested: tuple[str, ...] = ()
    intent_keys_effective: tuple[str, ...] = ()
    contributors: dict[str, Any] = field(default_factory=dict)


def _parse_v2_normative_guard(value: Any) -> NormativeGuardPolicy | None:
    if not isinstance(value, dict):
        return None
    required_support = str(value.get("required_support") or "").strip().lower()
    if required_support not in {"article", "article_or_annex", "any"}:
        return None
    profiles = _normalize_profile_list(value.get("profiles"))
    return NormativeGuardPolicy(required_support=required_support, profiles=profiles)


def _parse_v2_rescue_rules(value: Any) -> list[RescueRule]:
    if not isinstance(value, list):
        return []
    out: list[RescueRule] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        if_present = tuple(_normalize_anchor_list(item.get("if_present")))
        must_one = tuple(_normalize_anchor_list(item.get("must_include_one_of")))
        action = str(item.get("action") or "").strip().lower()
        if action not in {"anchor_lookup_inject", "pool_expand_and_select"}:
            continue
        profiles = _normalize_profile_list(item.get("profiles"))
        out.append(
            RescueRule(
                if_present=if_present,
                must_include_one_of=must_one,
                action=action,
                profiles=profiles,
            )
        )
    return out


def _parse_v2_answer_policy(value: Any) -> AnswerPolicy | None:
    if not isinstance(value, dict):
        return None

    intent_category = str(value.get("intent_category") or "").strip().upper()
    if intent_category not in {"REQUIREMENTS", "ENFORCEMENT", "OTHER"}:
        return None

    requirements_first = bool(value.get("requirements_first") is True)
    include_audit_evidence = bool(value.get("include_audit_evidence") is True)

    min_bullets_raw = value.get("min_section3_bullets")
    min_section3_bullets: int | None = None
    if min_bullets_raw is not None:
        try:
            min_section3_bullets = int(min_bullets_raw)
        except Exception:  # noqa: BLE001
            min_section3_bullets = None
        if min_section3_bullets is not None and min_section3_bullets < 0:
            min_section3_bullets = None

    return AnswerPolicy(
        intent_category=intent_category,
        requirements_first=requirements_first,
        include_audit_evidence=include_audit_evidence,
        min_section3_bullets=min_section3_bullets,
    )


def _parse_intent_entry(value: Any) -> dict[str, Any] | None:
    """Parse an intent entry. Supports:

    - v1: list[str] => bump_hints
    - v2: { bump_hints?: list[str], normative_guard?: {...}, rescue_rules?: [...] }
    """

    if isinstance(value, list):
        return {
            "bump_hints": _normalize_anchor_list(value),
            "normative_guard": None,
            "rescue_rules": [],
            "answer_policy": None,
        }

    if not isinstance(value, dict):
        return None

    bump_hints = _normalize_anchor_list(value.get("bump_hints"))
    normative_guard = _parse_v2_normative_guard(value.get("normative_guard"))
    rescue_rules = _parse_v2_rescue_rules(value.get("rescue_rules"))
    answer_policy = _parse_v2_answer_policy(value.get("answer_policy"))
    return {
        "bump_hints": bump_hints,
        "normative_guard": normative_guard,
        "rescue_rules": rescue_rules,
        "answer_policy": answer_policy,
    }


def extract_anchors_from_metadata(meta: dict[str, Any]) -> set[str]:
    """Extract anchors from chunk metadata for citation matching.

    Returns normalized anchors like: {'article:6', 'annex:iii', 'annex:iii:5'}

    Supports punkt-level granularity for annexes (e.g., ANNEX:III:5) to enable
    citation boost propagation from parent annexes to specific points.
    """
    if not isinstance(meta, dict):
        return set()

    article = str(meta.get("article") or "").strip()
    recital = str(meta.get("recital") or "").strip()
    annex = str(meta.get("annex") or "").strip()
    annex_point = str(meta.get("annex_point") or "").strip()
    annex_section = str(meta.get("annex_section") or "").strip()

    anchors: set[str] = set()
    if article:
        anchors.add(_normalize_anchor(f"article:{article}"))
    if recital:
        anchors.add(_normalize_anchor(f"recital:{recital}"))
    if annex:
        # Add annex-level anchor (e.g., annex:iii)
        anchors.add(_normalize_anchor(f"annex:{annex}"))

        # Add punkt-level anchor if present (e.g., annex:iii:5)
        # This enables citation boost from graph nodes like ANNEX:III:5
        if annex_point:
            if annex_section:
                # Full path: annex:iii:a:5 (section + point)
                anchors.add(_normalize_anchor(f"annex:{annex}:{annex_section}:{annex_point}"))
            else:
                # Direct point: annex:iii:5
                anchors.add(_normalize_anchor(f"annex:{annex}:{annex_point}"))

        # Add section-level anchor if present (e.g., annex:viii:a)
        if annex_section:
            anchors.add(_normalize_anchor(f"annex:{annex}:{annex_section}"))

    return anchors


def _repo_root() -> Path:
    # File is src/engine/concept_config.py
    return Path(__file__).resolve().parents[2]


def _concepts_dir() -> Path:
    """Get concepts directory. Can be overridden via CONCEPTS_DIR env var for testing."""
    override = os.getenv("CONCEPTS_DIR")
    if override:
        return Path(override)
    root = _repo_root()
    return root / "config" / "concepts"


@lru_cache(maxsize=1)
def load_concept_config() -> dict[str, dict[str, dict[str, Any]]]:
    """Load concept configuration from YAML files.

    Reads from config/concepts/*.yaml (single source of truth).
    
    Format per (corpus_id, concept_name):
    {
        "bump_hints": [..],  # deprecated - use citation_expansion instead
        "normative_guard": {"required_support": "article"|"article_or_annex"|"any", "profiles": [...]},
        "answer_policy": {...}
    }

    NOTE: cached per-process for determinism.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    
    concepts_dir = _concepts_dir()
    if not concepts_dir.exists():
        return out
    
    for yaml_file in concepts_dir.glob("*.yaml"):
        corpus_id = yaml_file.stem
        # Skip template files
        if corpus_id.startswith("_"):
            continue
            
        try:
            config = load_corpus_config(corpus_id)
        except Exception:
            continue
            
        concepts = config.get("concepts", {})
        if not isinstance(concepts, dict):
            continue
            
        cleaned: dict[str, dict[str, Any]] = {}
        
        for concept_name, concept_cfg in concepts.items():
            if not isinstance(concept_name, str) or not concept_name.strip():
                continue
            if not isinstance(concept_cfg, dict):
                continue
                
            bump_hints = _normalize_anchor_list(concept_cfg.get("bump_hints", []))
            normative_guard = _parse_v2_normative_guard(concept_cfg.get("normative_guard"))
            answer_policy = _parse_v2_answer_policy(concept_cfg.get("answer_policy"))
            rescue_rules = _parse_v2_rescue_rules(concept_cfg.get("rescue_rules"))
            
            # Use concept name as intent key (normalized)
            intent_key = _normalize_intent_key(concept_name)
            cleaned[intent_key] = {
                "bump_hints": bump_hints,
                "normative_guard": normative_guard,
                "rescue_rules": rescue_rules,
                "answer_policy": answer_policy,
            }
        
        # Add default config if present
        default_cfg = config.get("default", {})
        if isinstance(default_cfg, dict):
            default_hints = _normalize_anchor_list(default_cfg.get("bump_hints", []))
            default_ng = _parse_v2_normative_guard(default_cfg.get("normative_guard"))
            default_ap = _parse_v2_answer_policy(default_cfg.get("answer_policy"))
            default_rr = _parse_v2_rescue_rules(default_cfg.get("rescue_rules"))
            
            # Only add default if there's something to configure
            if default_hints or default_ng or default_ap or default_rr:
                cleaned["default"] = {
                    "bump_hints": default_hints,
                    "normative_guard": default_ng,
                    "rescue_rules": default_rr,
                    "answer_policy": default_ap,
                }
        
        out[corpus_id] = cleaned

    return out


def anchor_hint_bumping_enabled() -> bool:
    return str(os.getenv("ANCHOR_HINT_BUMPING", "") or "").strip().lower() in _TRUTHY_ENV_VALUES


def get_hint_anchors(
    *,
    corpus_id: str,
    intent_keys: Iterable[str],
) -> set[str]:
    cfg = load_concept_config()
    corpus_map = cfg.get(str(corpus_id or "").strip(), {})
    if not isinstance(corpus_map, dict):
        return set()

    out: set[str] = set()
    for k in list(intent_keys or []):
        kk = _normalize_intent_key(str(k or ""))
        if not kk:
            continue
        entry = corpus_map.get(kk)
        if isinstance(entry, dict):
            anchors = entry.get("bump_hints")
            if isinstance(anchors, list):
                out |= {_normalize_anchor(str(a)) for a in anchors if isinstance(a, str) and str(a).strip()}
    return out


def get_effective_policy(*, corpus_id: str, intent_keys: list[str]) -> Policy:
    """Merge config into an effective policy.

    Merge order:
      - always start with "default"
      - then each intent_key in the given order
    Rules:
      - bump_hints: union (stable order)
      - rescue_rules: append (stable order)
      - normative_guard.required_support: last-wins (if profile match is evaluated later)
    """

    requested_raw = [str(k or "").strip() for k in list(intent_keys or []) if str(k or "").strip()]
    requested_norm = [_normalize_intent_key(k) for k in requested_raw]
    requested_norm = [k for k in requested_norm if k]

    # Preserve the caller's ordering, but always include default first.
    merge_keys = _dedupe_preserve_order(["default", *requested_norm])

    cfg = load_concept_config()
    corpus_map = cfg.get(str(corpus_id or "").strip(), {})
    if not isinstance(corpus_map, dict):
        corpus_map = {}

    bump_out: list[str] = []
    bump_sources: list[dict[str, Any]] = []
    rescue_out: list[RescueRule] = []
    rescue_sources: list[dict[str, Any]] = []
    normative_guard: NormativeGuardPolicy | None = None
    normative_source: str | None = None
    answer_policy: AnswerPolicy | None = None
    answer_policy_sources: list[dict[str, Any]] = []
    requirements_first_lock = False
    requirements_first_source: str | None = None
    requirements_first_policy: AnswerPolicy | None = None
    effective_keys: list[str] = []

    for k in merge_keys:
        entry = corpus_map.get(k)
        if not isinstance(entry, dict):
            continue
        effective_keys.append(k)

        bump = entry.get("bump_hints")
        if isinstance(bump, list) and bump:
            before = set(bump_out)
            for a in bump:
                aa = _normalize_anchor(str(a))
                if aa and aa not in before and ":" in aa:
                    bump_out.append(aa)
                    before.add(aa)
            bump_sources.append({"intent_key": k, "bump_hints": list(bump)})

        ng = entry.get("normative_guard")
        if isinstance(ng, NormativeGuardPolicy):
            normative_guard = ng
            normative_source = k

        ap = entry.get("answer_policy")
        if isinstance(ap, AnswerPolicy):
            # Deterministic merge: last-wins for scalar fields.
            # Special rule: if ANY contributing policy declares REQUIREMENTS + requirements_first=true,
            # it cannot be overridden by later entries (fail-closed towards REQUIREMENTS).
            answer_policy = ap
            answer_policy_sources.append({"intent_key": k, "answer_policy": ap.to_debug_dict()})
            if ap.intent_category == "REQUIREMENTS" and bool(ap.requirements_first):
                requirements_first_lock = True
                requirements_first_source = k
                requirements_first_policy = ap

        rr = entry.get("rescue_rules")
        if isinstance(rr, list) and rr:
            parsed_rr: list[RescueRule] = [r for r in rr if isinstance(r, RescueRule)]
            if parsed_rr:
                rescue_out.extend(parsed_rr)
                rescue_sources.append({"intent_key": k, "count": int(len(parsed_rr))})

    contributors = {
        "merge_keys": list(merge_keys),
        "bump_hints_sources": bump_sources,
        "normative_guard_source": normative_source,
        "rescue_rules_sources": rescue_sources,
        "answer_policy_sources": answer_policy_sources,
        "requirements_first_lock_source": requirements_first_source,
    }

    # Apply REQUIREMENTS-first lock after the full merge (order-independent).
    if requirements_first_lock:
        src = requirements_first_policy
        answer_policy = AnswerPolicy(
            intent_category="REQUIREMENTS",
            requirements_first=True,
            include_audit_evidence=(bool(getattr(src, "include_audit_evidence", False)) if src is not None else False),
            min_section3_bullets=(getattr(src, "min_section3_bullets", None) if src is not None else None),
        )

    return Policy(
        bump_hints=tuple(bump_out),
        normative_guard=normative_guard,
        rescue_rules=tuple(rescue_out),
        answer_policy=answer_policy,
        intent_keys_requested=tuple(requested_raw),
        intent_keys_effective=tuple(effective_keys),
        contributors=contributors,
    )


@dataclass(frozen=True)
class AnchorHintBumpResult:
    order: list[int]
    applied: bool
    bonus: float
    matched_indices: list[int]


def compute_anchor_hint_bump_order(
    *,
    metadatas: list[dict[str, Any]],
    distances: list[float],
    hint_anchors: set[str],
    bonus: float,
) -> AnchorHintBumpResult:
    """Compute a deterministic re-order based on hint anchors.

    Scoring rule (minimal and audit-friendly):
      score = -distance + (bonus if candidate has any hinted anchor)

    Tie-break (deterministic):
      score desc, anchor_id asc, chunk_id asc, original_index asc

    NOTE: This is ranking-only; caller may cut back to top_k.
    """

    if not hint_anchors or bonus <= 0.0:
        return AnchorHintBumpResult(order=list(range(min(len(metadatas), len(distances)))), applied=False, bonus=float(bonus), matched_indices=[])

    n = min(len(metadatas), len(distances))
    scored: list[tuple[float, str, str, int]] = []
    matched: list[int] = []

    for i in range(n):
        meta = dict(metadatas[i] or {})
        anchors = extract_anchors_from_metadata(meta)
        hit = bool(set(anchors) & set(hint_anchors))
        if hit:
            matched.append(i)

        base = -float(distances[i])
        final = float(base + (float(bonus) if hit else 0.0))
        # Pick a stable anchor id for tie-break: smallest lex anchor among present anchors.
        anchor_id = sorted(anchors)[0] if anchors else ""
        chunk_id = str(meta.get("chunk_id") or "").strip()
        scored.append((final, anchor_id, chunk_id, i))

    # Sort: final desc, anchor_id asc, chunk_id asc, original index asc
    scored.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))
    order = [i for _s, _a, _c, i in scored]

    return AnchorHintBumpResult(order=order, applied=True, bonus=float(bonus), matched_indices=matched)
