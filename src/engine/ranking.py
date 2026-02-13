from typing import Any, Dict, List, Tuple
import re
import math
from .planning import UserProfile
from .helpers import classify_query_intent
from ..common.config_loader import RankingWeights

class Ranker:
    # --- Deterministic role taxonomy (do not change values) ---
    # These constants are used for reranking weights and are referenced by RAGEngine.
    # They are kept here to maintain the ranking domain logic.
    ROLE_DEFINITION_SCOPE = "DEFINITION_SCOPE"
    ROLE_PROHIBITIONS = "PROHIBITIONS"
    ROLE_OBLIGATIONS_NORMATIVE = "OBLIGATIONS_NORMATIVE"
    ROLE_TRANSPARENCY = "TRANSPARENCY"
    ROLE_GOVERNANCE_PROCESS = "GOVERNANCE_PROCESS"
    ROLE_ENFORCEMENT_SUPERVISION = "ENFORCEMENT_SUPERVISION"
    ROLE_EXCEPTIONS_SANDBOX = "EXCEPTIONS_SANDBOX"
    ROLE_RECITAL_CONTEXT = "RECITAL_CONTEXT"

    def __init__(self):
        pass

    @staticmethod
    def _tokenize_for_lexical(text: str) -> list[str]:
        # Keep it simple and Unicode-friendly for Danish legal text.
        if not text:
            return []
        return re.findall(r"[0-9A-Za-zÆØÅæøå]+", text.lower())

    @classmethod
    def _bm25_scores(cls, *, query: str, documents: list[str]) -> list[float]:
        """Compute BM25-like lexical scores over a candidate set.

        This is intended as a reranker over top-N embedding candidates.
        """

        q_terms = cls._tokenize_for_lexical(query)
        if not q_terms or not documents:
            return [0.0 for _ in documents]

        # Use unique query terms to reduce overweighting repeated question tokens.
        q_terms = list(dict.fromkeys(q_terms))

        N = len(documents)
        doc_tokens = [cls._tokenize_for_lexical(d) for d in documents]
        doc_lens = [len(toks) for toks in doc_tokens]
        avgdl = (sum(doc_lens) / N) if N else 1.0
        if avgdl <= 0:
            avgdl = 1.0

        # df per query term
        dfs: dict[str, int] = {t: 0 for t in q_terms}
        tfs_per_doc: list[dict[str, int]] = []
        for toks in doc_tokens:
            tf: dict[str, int] = {}
            for tok in toks:
                if tok in dfs:
                    tf[tok] = tf.get(tok, 0) + 1
            tfs_per_doc.append(tf)
            for term in tf.keys():
                dfs[term] += 1

        def idf(df: int) -> float:
            # Standard BM25 idf with +1 to keep it non-negative.
            return math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        k1 = 1.2
        b = 0.75
        scores: list[float] = []
        for dl, tf in zip(doc_lens, tfs_per_doc):
            s = 0.0
            norm = k1 * (1.0 - b + b * (dl / avgdl))
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                term_idf = idf(dfs.get(term, 0))
                s += term_idf * (f * (k1 + 1.0)) / (f + norm)
            scores.append(s)

        return scores

    @staticmethod
    def _normalize_scores(values: list[float]) -> list[float]:
        if not values:
            return []
        vmin = min(values)
        vmax = max(values)
        if vmax <= vmin:
            return [0.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]

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
        weights: RankingWeights | None = None,
        user_profile: UserProfile | str | None = None,
    ) -> tuple[list[tuple[str, dict[str, Any]]], list[float], list[str]]:
        """Hybrid rerank combining 4 factors: vector similarity, BM25, citation, and role.

        Args:
            question: Query text for BM25 scoring and intent classification.
            ids: Chunk IDs.
            documents: Chunk text content.
            metadatas: Chunk metadata dicts.
            distances: Embedding distances (lower = more similar).
            k: Number of results to return.
            hint_anchors: Set of anchors from citation graph (e.g. {'article:6', 'annex:iii'}).
                         Chunks matching these get citation boost.
            citation_boost: Score bonus for chunks matching hint_anchors (default: 0.0).
                           Typically 0.15 from config bump_bonus.
            weights: RankingWeights with α, β, γ, δ values. If None, uses defaults.
            user_profile: User profile for role-based scoring (LEGAL/ENGINEERING).

        Returns:
            Tuple of (hits, distances, ids) reordered by combined score.

        Scoring formula:
            score = α*vec_sim + β*bm25 + γ*citation_signal + δ*role_signal
            where weights come from config (default: α=0.25, β=0.25, γ=0.35, δ=0.15)
        """
        if not ids or not documents:
            return [], [], []

        # Use provided weights or defaults
        w = weights or RankingWeights()

        # 1. Vector similarity (lower distance => higher similarity)
        vec_sims = [1.0 / (1.0 + float(d)) for d in distances]
        vec_norm = self._normalize_scores(vec_sims)

        # 2. BM25 lexical scoring
        bm25 = self._bm25_scores(query=question, documents=documents)
        bm25_norm = self._normalize_scores(bm25)

        # 3. Citation signal: 1.0 if chunk matches any hint anchor, else 0.0
        citation_signals: list[float] = []
        if hint_anchors and citation_boost > 0.0:
            from src.engine.concept_config import extract_anchors_from_metadata
            for meta in metadatas:
                chunk_anchors = extract_anchors_from_metadata(meta)
                has_match = bool(chunk_anchors & hint_anchors)
                citation_signals.append(1.0 if has_match else 0.0)
        else:
            citation_signals = [0.0] * len(ids)

        # 4. Role signal: alignment with query intent and user profile
        role_signals: list[float] = []
        if user_profile is not None and w.delta_role > 0:
            intent = classify_query_intent(question)
            for i, (meta, doc) in enumerate(zip(metadatas, documents)):
                role_info = self.classify_role(meta, doc)
                role = str(role_info.get("role") or self.ROLE_OBLIGATIONS_NORMATIVE)
                role_conf = float(role_info.get("confidence") or 0.0)
                # Get role delta and normalize to [0, 1] range
                delta = self._rerank_delta_for_role(
                    role=role,
                    role_confidence=role_conf,
                    intent=intent,
                    user_profile=user_profile,
                )
                # Delta ranges roughly from -0.15 to +0.18, normalize to [0, 1]
                # Using sigmoid-like normalization: (delta + 0.15) / 0.33
                normalized_delta = max(0.0, min(1.0, (delta + 0.15) / 0.33))
                role_signals.append(normalized_delta)
        else:
            role_signals = [0.5] * len(ids)  # Neutral role signal

        # Combined 4-factor scoring
        combined = [
            w.alpha_vec * v + w.beta_bm25 * b + w.gamma_cite * c + w.delta_role * r
            for v, b, c, r in zip(vec_norm, bm25_norm, citation_signals, role_signals)
        ]

        order = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)
        order = order[: max(1, int(k))]

        hits = [(documents[i], metadatas[i]) for i in order]
        out_distances = [distances[i] for i in order]
        out_ids = [ids[i] for i in order]
        return hits, out_distances, out_ids

    @classmethod
    def classify_role(cls, meta: dict[str, Any] | None, text: str) -> dict[str, Any]:
        """Classify an article chunk into a generic role.

        Deterministic + explainable: returns role, confidence, and the signals used.
        Never inspects article numbers.
        """

        m = dict(meta or {})
        signals: list[str] = []

        # High-priority structural signals: recitals/preamble.
        doc_type = str(m.get("doc_type") or m.get("source_type") or "").strip().lower()
        location_id = str(m.get("location_id") or "").strip().lower()
        heading_display = str(m.get("heading_path_display") or "").strip().lower()

        is_recital_structural = False
        if m.get("recital"):
            is_recital_structural = True
            signals.append("meta.recital_present")
        if doc_type and "recital" in doc_type:
            is_recital_structural = True
            signals.append("meta.doc_type_recital")
        if "recital:" in location_id or "preamble" in location_id:
            is_recital_structural = True
            signals.append("location_id_recital_or_preamble")
        if "recital" in heading_display or "betragtning" in heading_display or "preamble" in heading_display:
            is_recital_structural = True
            signals.append("heading_recital_or_preamble")

        if is_recital_structural:
            return {
                "role": cls.ROLE_RECITAL_CONTEXT,
                "confidence": 0.95,
                "signals": signals,
            }

        # Medium-weight heading/title keyword signals.
        title = str(m.get("title") or "").strip().lower()
        heading_blob = " ".join([s for s in [heading_display, title] if s]).strip()

        # Low-weight textual signals.
        body = str(text or "")
        body_lower = body.lower()

        scores: dict[str, float] = {
            cls.ROLE_DEFINITION_SCOPE: 0.0,
            cls.ROLE_PROHIBITIONS: 0.0,
            cls.ROLE_OBLIGATIONS_NORMATIVE: 0.0,
            cls.ROLE_TRANSPARENCY: 0.0,
            cls.ROLE_GOVERNANCE_PROCESS: 0.0,
            cls.ROLE_ENFORCEMENT_SUPERVISION: 0.0,
            cls.ROLE_EXCEPTIONS_SANDBOX: 0.0,
            cls.ROLE_RECITAL_CONTEXT: 0.0,
        }

        def add_heading(role: str, why: str) -> None:
            scores[role] += 2.0
            signals.append(f"heading:{why}")

        def add_text(role: str, why: str) -> None:
            scores[role] += 1.0
            signals.append(f"text:{why}")

        # Heading/title keyword mapping.
        if heading_blob:
            if re.search(r"\b(definition|definitions|scope|subject\s+matter)\b", heading_blob):
                add_heading(cls.ROLE_DEFINITION_SCOPE, "definition_or_scope")
            if re.search(r"\b(prohibited|prohibition|forbidden)\b", heading_blob):
                add_heading(cls.ROLE_PROHIBITIONS, "prohibition")
            if re.search(r"\b(transparency|inform|information|disclosure)\b", heading_blob):
                add_heading(cls.ROLE_TRANSPARENCY, "transparency_or_information")
            if re.search(r"\b(authority|penalt(y|ies)|sanction(s)?|supervision|enforcement)\b", heading_blob):
                add_heading(cls.ROLE_ENFORCEMENT_SUPERVISION, "enforcement_supervision")
            if re.search(r"\b(sandbox)\b", heading_blob):
                add_heading(cls.ROLE_EXCEPTIONS_SANDBOX, "sandbox")
            if re.search(
                r"\b(risk\s+management|quality\s+management|conformity|monitoring|post-?market)\b",
                heading_blob,
            ):
                add_heading(cls.ROLE_GOVERNANCE_PROCESS, "governance_process")

        # Textual signals.
        if re.search(r"\b(must|shall|required)\b", body_lower):
            add_text(cls.ROLE_OBLIGATIONS_NORMATIVE, "modal_must_shall_required")
        if re.search(r"\bskal\b", body_lower):
            add_text(cls.ROLE_OBLIGATIONS_NORMATIVE, "modal_skal")
        if re.search(r"\b(inform|information|disclose|tell\s+user|notify)\b", body_lower):
            add_text(cls.ROLE_TRANSPARENCY, "inform_user")
        if re.search(r"\b(transparency|gennemsigtighed)\b", body_lower):
            add_text(cls.ROLE_TRANSPARENCY, "transparency")

        best_role = cls.ROLE_OBLIGATIONS_NORMATIVE
        best_score = -1.0
        for r, sc in scores.items():
            if sc > best_score:
                best_role, best_score = r, sc

        # Confidence is deterministic based on score separation.
        top_roles = [r for r, sc in scores.items() if sc == best_score]
        if best_score <= 0.0:
            return {
                "role": cls.ROLE_OBLIGATIONS_NORMATIVE,
                "confidence": 0.20,
                "signals": ["fallback:no_signals"],
            }

        if best_score >= 2.0:
            confidence = 0.75 if len(top_roles) == 1 else 0.65
        else:
            confidence = 0.55 if len(top_roles) == 1 else 0.45

        return {
            "role": best_role,
            "confidence": float(confidence),
            "signals": signals,
        }

    @classmethod
    def _rerank_delta_for_role(
        cls,
        *,
        role: str,
        role_confidence: float,
        intent: str,
        user_profile: UserProfile | str,
    ) -> float:
        """Compute a deterministic rerank delta for a chunk role.

        Returned value is added to (-distance). Positive boosts relevance.
        """
        delta = 0.0

        # Intent-based deltas.
        if intent == "TRANSPARENCY":
            if role == cls.ROLE_TRANSPARENCY:
                delta += 0.18
            if role == cls.ROLE_OBLIGATIONS_NORMATIVE:
                delta += 0.10
            if role == cls.ROLE_ENFORCEMENT_SUPERVISION:
                delta -= 0.08
            if role == cls.ROLE_RECITAL_CONTEXT:
                delta -= 0.12
        elif intent == "OBLIGATIONS":
            if role == cls.ROLE_OBLIGATIONS_NORMATIVE:
                delta += 0.15
            if role == cls.ROLE_GOVERNANCE_PROCESS:
                delta += 0.10
        elif intent == "ENFORCEMENT":
            if role == cls.ROLE_ENFORCEMENT_SUPERVISION:
                delta += 0.18

        # Profile-based deltas.
        profile = user_profile
        if isinstance(profile, str):
            p = profile.strip().upper()
        else:
            p = str(getattr(profile, "value", profile)).strip().upper()

        if p == "ENGINEERING":
            if role == cls.ROLE_RECITAL_CONTEXT:
                delta -= 0.20
            if role == cls.ROLE_ENFORCEMENT_SUPERVISION and intent != "ENFORCEMENT":
                delta -= 0.10
        elif p == "LEGAL":
            if role == cls.ROLE_RECITAL_CONTEXT:
                delta -= 0.05

        # Scale by confidence to keep the rerank explainable and conservative.
        conf = float(role_confidence)
        if conf < 0.0:
            conf = 0.0
        if conf > 1.0:
            conf = 1.0
        return float(delta * max(0.20, conf))

    @classmethod
    def rerank_retrieved_chunks(
        cls,
        *,
        metadatas: list[dict[str, Any]],
        documents: list[str],
        distances: list[float],
        question: str,
        user_profile: UserProfile | str,
    ) -> list[int]:
        """Return reordered indices for the retrieved set.

        Deterministic rerank only: final_score = (-distance) + rerank_delta.
        """
        intent = classify_query_intent(question)
        n = min(len(metadatas), len(documents), len(distances))
        scored: list[tuple[float, int]] = []
        for i in range(n):
            role_info = cls.classify_role(metadatas[i], documents[i])
            delta = cls._rerank_delta_for_role(
                role=str(role_info.get("role") or cls.ROLE_OBLIGATIONS_NORMATIVE),
                role_confidence=float(role_info.get("confidence") or 0.0),
                intent=intent,
                user_profile=user_profile,
            )
            score = (-float(distances[i])) + float(delta)
            scored.append((score, i))

        # Stable: keep original order on ties.
        scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
        return [i for _, i in scored]

    @staticmethod
    def anchor_score(metadata: dict[str, Any] | None, hint_anchors: set[str] | None = None) -> int:
        """Return a deterministic anchor precision score.

        +5 = matches a hint_anchor (explicitly mentioned in query)
        +4 = article + paragraph
        +3 = article
        +2 = recital
        +1 = annex
        +0 = chapter/section only

        When hint_anchors is provided, chunks matching those anchors get top priority.
        """

        m = dict(metadata or {})
        article = str(m.get("article") or "").strip()
        paragraph = str(m.get("paragraph") or "").strip()
        recital = str(m.get("recital") or "").strip()
        annex = str(m.get("annex") or "").strip()

        # Check if this chunk matches any hint anchor (highest priority)
        if hint_anchors:
            # Build normalized anchor identifiers for this chunk
            chunk_anchors = set()
            if article:
                chunk_anchors.add(f"article:{article}".lower())
            if annex:
                chunk_anchors.add(f"annex:{annex}".lower())
            if recital:
                chunk_anchors.add(f"recital:{recital}".lower())

            # If any match, this is a hint anchor match
            if chunk_anchors & hint_anchors:
                return 5

        if article and paragraph:
            return 4
        if article:
            return 3
        if recital:
            return 2
        if annex:
            return 1
        return 0

# ---------------------------------------------------------------------------
# Ranking Pipeline (extracted from RAGEngine.answer_structured)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class RankingPipelineResult:
    """Result of the ranking pipeline pass-through."""
    ranked_hits: List[Tuple[str, Dict[str, Any]]]
    ranked_distances: List[float]
    ranked_ids: List[str]
    ranked_metas: List[Dict[str, Any]]
    debug_info: Dict[str, Any] | None = None


def execute_ranking_pipeline(
    *,
    hits: List[Tuple[str, Dict[str, Any]]],
    distances: List[float],
    retrieved_ids: List[str],
    retrieved_metas: List[Dict[str, Any]],
    question: str,
    user_profile: "UserProfile | None" = None,
) -> RankingPipelineResult:
    """Pass through hybrid_rerank results with optional debug info.

    All ranking is done by hybrid_rerank (4-factor scoring):
    score = α*vec_sim + β*bm25 + γ*citation + δ*role

    This function preserves the order from hybrid_rerank and generates
    debug info for UI visualization.
    """
    if not hits or not distances:
        return RankingPipelineResult(
            ranked_hits=list(hits),
            ranked_distances=list(distances),
            ranked_ids=list(retrieved_ids),
            ranked_metas=list(retrieved_metas),
        )

    # Ensure alignment
    n = min(len(hits), len(distances), len(retrieved_ids), len(retrieved_metas))
    hits = list(hits[:n])
    distances = list(distances[:n])
    retrieved_ids = list(retrieved_ids[:n])
    retrieved_metas = list(retrieved_metas[:n])
    docs = [str(d or "") for d, _m in hits]

    # Generate debug info for UI (role classification info per chunk)
    debug_info: Dict[str, Any] | None = None

    if user_profile is not None:
        try:
            intent = classify_query_intent(question)
            items: List[Dict[str, Any]] = []

            for i in range(len(hits)):
                role_info = Ranker.classify_role(retrieved_metas[i], docs[i])
                role = str(role_info.get("role") or Ranker.ROLE_OBLIGATIONS_NORMATIVE)
                role_conf = float(role_info.get("confidence") or 0.0)
                role_signals = list(role_info.get("signals") or [])

                items.append({
                    "chunk_id": str(retrieved_ids[i]),
                    "distance": float(distances[i]),
                    "role": role,
                    "role_confidence": float(role_conf),
                    "role_signals": role_signals,
                    "query_intent": intent,
                    "rank": i + 1,
                    "heading_path_display": str(retrieved_metas[i].get("heading_path_display", "")),
                })

            debug_info = {
                "query_intent": intent,
                "items": items,
            }
        except Exception:  # noqa: BLE001
            debug_info = {"error": "debug_info_failed"}

    return RankingPipelineResult(
        ranked_hits=hits,
        ranked_distances=distances,
        ranked_ids=retrieved_ids,
        ranked_metas=retrieved_metas,
        debug_info=debug_info,
    )
