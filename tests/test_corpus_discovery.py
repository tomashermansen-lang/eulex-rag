"""Tests for corpus_discovery module — AI Law Discovery.

TDD: These tests are written BEFORE the implementation.
"""

from __future__ import annotations

import dataclasses
import pytest

from src.engine.corpus_discovery import (
    DiscoveryConfig,
    DiscoveryMatch,
    DiscoveryResult,
    discover_corpora,
    distance_to_similarity,
    _stage_alias_detection,
    _stage_retrieval_probe,
    _score_corpus,
    _apply_confidence_gating,
    _merge_stages,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeCorpusResolver:
    """Minimal fake matching the CorpusResolver interface for tests."""

    def __init__(self, alias_map: dict[str, list[str]]) -> None:
        self._alias_map = alias_map

    def mentioned_corpus_keys(self, text: str) -> list[str]:
        text_lower = text.lower()
        found = []
        for key, aliases in self._alias_map.items():
            for alias in aliases:
                if alias.lower() in text_lower:
                    found.append(key)
                    break
        return sorted(found)


@pytest.fixture
def resolver_with_gdpr_and_ai_act() -> FakeCorpusResolver:
    return FakeCorpusResolver({
        "gdpr": ["gdpr", "databeskyttelsesforordningen"],
        "ai_act": ["ai-act", "ai act", "ai-forordningen"],
    })


@pytest.fixture
def resolver_empty() -> FakeCorpusResolver:
    return FakeCorpusResolver({})


@pytest.fixture
def default_config() -> DiscoveryConfig:
    return DiscoveryConfig()


def make_probe_fn(corpus_results: dict[str, list[tuple[dict, float]]]):
    """Build a mock probe_fn returning pre-defined per-corpus results.

    corpus_results: {corpus_id: [(metadata_dict, distance), ...]}
    """

    def probe_fn(question: str, corpus_id: str, k: int) -> list[tuple[dict, float]]:
        return corpus_results.get(corpus_id, [])

    return probe_fn


def make_failing_probe_fn():
    """Build a probe_fn that always raises."""

    def probe_fn(question: str, corpus_id: str, k: int) -> list[tuple[dict, float]]:
        raise RuntimeError("Embedding API failure")

    return probe_fn


# ===========================================================================
# 1A: Data Types
# ===========================================================================


class TestDataTypes:
    def test_discovery_match_frozen(self) -> None:
        m = DiscoveryMatch(corpus_id="gdpr", confidence=0.9, reason="alias_match")
        with pytest.raises(dataclasses.FrozenInstanceError):
            m.confidence = 0.5  # type: ignore[misc]

    def test_discovery_result_frozen(self) -> None:
        r = DiscoveryResult(
            matches=(),
            resolved_scope="abstain",
            resolved_corpora=(),
            gate="ABSTAIN",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.gate = "AUTO"  # type: ignore[misc]

    def test_discovery_config_defaults(self, default_config: DiscoveryConfig) -> None:
        assert default_config.enabled is True
        assert default_config.probe_top_k == 10
        assert default_config.auto_threshold == 0.75
        assert default_config.suggest_threshold == 0.65
        assert default_config.ambiguity_margin == 0.10
        assert default_config.max_corpora == 5
        assert default_config.llm_disambiguation is True
        assert default_config.w_similarity == 0.50
        assert default_config.w_best == 0.50
        assert default_config.max_suggest_corpora == 2


# ===========================================================================
# 1B: Stage 1 — Alias Detection
# ===========================================================================


class TestAliasDetection:
    def test_single_match(self, resolver_with_gdpr_and_ai_act) -> None:
        matches = _stage_alias_detection("Hvad siger GDPR om databehandleraftaler?", resolver_with_gdpr_and_ai_act)
        assert len(matches) == 1
        assert matches[0].corpus_id == "gdpr"

    def test_multiple_matches(self, resolver_with_gdpr_and_ai_act) -> None:
        matches = _stage_alias_detection("Sammenlign GDPR og AI-Act", resolver_with_gdpr_and_ai_act)
        corpus_ids = {m.corpus_id for m in matches}
        assert corpus_ids == {"gdpr", "ai_act"}

    def test_no_match_returns_empty(self, resolver_with_gdpr_and_ai_act) -> None:
        matches = _stage_alias_detection("Hvad er reglerne for datadeling?", resolver_with_gdpr_and_ai_act)
        assert matches == []

    def test_confidence_is_one(self, resolver_with_gdpr_and_ai_act) -> None:
        matches = _stage_alias_detection("GDPR regler", resolver_with_gdpr_and_ai_act)
        for m in matches:
            assert m.confidence == 1.0

    def test_reason_is_alias_match(self, resolver_with_gdpr_and_ai_act) -> None:
        matches = _stage_alias_detection("GDPR regler", resolver_with_gdpr_and_ai_act)
        for m in matches:
            assert m.reason == "alias_match"


# ===========================================================================
# 1C: Stage 2 — Retrieval Probe
# ===========================================================================


class TestRetrievalProbe:
    def test_single_dominant_corpus(self, default_config: DiscoveryConfig) -> None:
        # ai_act has very close results, gdpr has distant results
        probe_fn = make_probe_fn({
            "ai_act": [
                ({"chunk_id": "a1"}, 0.2),  # sim=0.833
                ({"chunk_id": "a2"}, 0.3),  # sim=0.769
                ({"chunk_id": "a3"}, 0.4),  # sim=0.714
            ],
            "gdpr": [
                ({"chunk_id": "g1"}, 2.0),  # sim=0.333
                ({"chunk_id": "g2"}, 3.0),  # sim=0.250
            ],
        })
        matches = _stage_retrieval_probe(
            "facial recognition rules",
            ["ai_act", "gdpr"],
            probe_fn,
            default_config,
        )
        # ai_act should have higher confidence than gdpr
        ai_match = next(m for m in matches if m.corpus_id == "ai_act")
        gdpr_match = next(m for m in matches if m.corpus_id == "gdpr")
        assert ai_match.confidence > gdpr_match.confidence

    def test_multi_corpus_balanced(self, default_config: DiscoveryConfig) -> None:
        # Both corpora have similar close results
        probe_fn = make_probe_fn({
            "nis2": [
                ({"chunk_id": "n1"}, 0.3),
                ({"chunk_id": "n2"}, 0.4),
            ],
            "dora": [
                ({"chunk_id": "d1"}, 0.35),
                ({"chunk_id": "d2"}, 0.45),
            ],
        })
        matches = _stage_retrieval_probe(
            "incident reporting",
            ["nis2", "dora"],
            probe_fn,
            default_config,
        )
        assert len(matches) == 2
        # Scores should be close to each other
        scores = [m.confidence for m in matches]
        assert abs(scores[0] - scores[1]) < 0.2

    def test_all_low_similarity(self, default_config: DiscoveryConfig) -> None:
        # All results very distant
        probe_fn = make_probe_fn({
            "ai_act": [({"chunk_id": "a1"}, 10.0)],  # sim=0.091
            "gdpr": [({"chunk_id": "g1"}, 10.0)],     # sim=0.091
        })
        matches = _stage_retrieval_probe(
            "something vague",
            ["ai_act", "gdpr"],
            probe_fn,
            default_config,
        )
        for m in matches:
            assert m.confidence < 0.50

    def test_empty_results(self, default_config: DiscoveryConfig) -> None:
        probe_fn = make_probe_fn({})
        matches = _stage_retrieval_probe(
            "anything",
            ["ai_act", "gdpr"],
            probe_fn,
            default_config,
        )
        assert matches == []

    def test_100_pct_dominance(self, default_config: DiscoveryConfig) -> None:
        # Only one corpus returns results
        probe_fn = make_probe_fn({
            "ai_act": [
                ({"chunk_id": "a1"}, 0.2),
                ({"chunk_id": "a2"}, 0.3),
            ],
            "gdpr": [],
        })
        matches = _stage_retrieval_probe(
            "facial recognition",
            ["ai_act", "gdpr"],
            probe_fn,
            default_config,
        )
        # Only ai_act should have a match (gdpr returned nothing)
        corpus_ids = [m.corpus_id for m in matches]
        assert "ai_act" in corpus_ids

    def test_reason_is_retrieval_probe(self, default_config: DiscoveryConfig) -> None:
        probe_fn = make_probe_fn({
            "ai_act": [({"chunk_id": "a1"}, 0.3)],
        })
        matches = _stage_retrieval_probe(
            "facial recognition",
            ["ai_act"],
            probe_fn,
            default_config,
        )
        for m in matches:
            assert m.reason == "retrieval_probe"


# ===========================================================================
# 1D: Scoring
# ===========================================================================


class TestScoring:
    def test_distance_to_similarity_zero(self) -> None:
        assert distance_to_similarity(0.0) == 1.0

    def test_distance_to_similarity_close_is_high(self) -> None:
        """Close distances (d < 0.5) should map to high similarity (> 0.85)."""
        sim = distance_to_similarity(0.3)
        assert sim > 0.85

    def test_distance_to_similarity_moderate(self) -> None:
        """Moderate distance (d=1.0) should give moderate similarity."""
        sim = distance_to_similarity(1.0)
        assert 0.50 < sim < 0.70

    def test_distance_to_similarity_large(self) -> None:
        # Very distant should give low similarity
        sim = distance_to_similarity(100.0)
        assert sim < 0.02

    def test_distance_to_similarity_negative_clamps(self) -> None:
        # Negative distances shouldn't happen but should be safe
        sim = distance_to_similarity(-1.0)
        assert sim >= 0.0
        assert sim <= 1.0

    def test_distance_to_similarity_spreads_useful_range(self) -> None:
        """Similarity function should spread [0.2, 1.5] across a wide range.

        Good discrimination requires close matches scoring significantly
        higher than moderate matches. The useful range [0.2, 1.5] should
        map to at least a 0.50 spread, not compressed into a narrow band.
        """
        close = distance_to_similarity(0.2)
        far = distance_to_similarity(1.5)
        spread = close - far
        assert spread > 0.50

    def test_score_corpus_basic(self, default_config: DiscoveryConfig) -> None:
        # 3 results with known distances
        distances = [0.2, 0.3, 0.4]
        score = _score_corpus(distances, default_config)
        assert 0.0 <= score <= 1.0

    def test_score_corpus_empty_returns_zero(self, default_config: DiscoveryConfig) -> None:
        score = _score_corpus([], default_config)
        assert score == 0.0

    def test_score_corpus_close_results_high(self, default_config: DiscoveryConfig) -> None:
        # Very close results
        close_score = _score_corpus([0.1, 0.15, 0.2], default_config)
        # Distant results
        far_score = _score_corpus([5.0, 6.0, 7.0], default_config)
        assert close_score > far_score


# ===========================================================================
# 1E: LLM Disambiguation
# ===========================================================================


class TestLLMDisambiguation:
    def test_not_triggered_clear_winner(self, default_config: DiscoveryConfig) -> None:
        # When top candidate is clearly ahead, LLM should not be called
        candidates = [
            DiscoveryMatch(corpus_id="ai_act", confidence=0.85, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="gdpr", confidence=0.40, reason="retrieval_probe"),
        ]
        call_count = 0

        def llm_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "ai_act"

        result = discover_corpora(
            question="facial recognition rules",
            corpus_ids=["ai_act", "gdpr"],
            probe_fn=make_probe_fn({
                "ai_act": [({"chunk_id": "a1"}, 0.1)] * 5,
                "gdpr": [({"chunk_id": "g1"}, 2.0)] * 2,
            }),
            resolver=FakeCorpusResolver({}),
            config=default_config,
            llm_fn=llm_fn,
        )
        # LLM should not have been called (clear winner)
        assert call_count == 0

    def test_disabled_by_config(self) -> None:
        config = DiscoveryConfig(llm_disambiguation=False)
        call_count = 0

        def llm_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "ai_act"

        result = discover_corpora(
            question="ambiguous question",
            corpus_ids=["ai_act", "gdpr"],
            probe_fn=make_probe_fn({
                "ai_act": [({"chunk_id": "a1"}, 0.5)] * 3,
                "gdpr": [({"chunk_id": "g1"}, 0.55)] * 3,
            }),
            resolver=FakeCorpusResolver({}),
            config=config,
            llm_fn=llm_fn,
        )
        assert call_count == 0

    def test_llm_error_falls_back(self, default_config: DiscoveryConfig) -> None:
        def llm_fn(prompt: str) -> str:
            raise RuntimeError("LLM timeout")

        # Should not raise, should return probe results
        result = discover_corpora(
            question="ambiguous question",
            corpus_ids=["ai_act", "gdpr"],
            probe_fn=make_probe_fn({
                "ai_act": [({"chunk_id": "a1"}, 0.5)] * 3,
                "gdpr": [({"chunk_id": "g1"}, 0.55)] * 3,
            }),
            resolver=FakeCorpusResolver({}),
            config=default_config,
            llm_fn=llm_fn,
        )
        assert isinstance(result, DiscoveryResult)


# ===========================================================================
# 1F: Stage Merging
# ===========================================================================


class TestMergeStages:
    def test_alias_takes_precedence(self) -> None:
        alias = [DiscoveryMatch(corpus_id="gdpr", confidence=1.0, reason="alias_match")]
        probe = [
            DiscoveryMatch(corpus_id="gdpr", confidence=0.7, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="ai_act", confidence=0.6, reason="retrieval_probe"),
        ]
        merged = _merge_stages(alias, probe, [])
        gdpr = next(m for m in merged if m.corpus_id == "gdpr")
        assert gdpr.confidence == 1.0
        assert gdpr.reason == "alias_match"

    def test_deduplicates(self) -> None:
        alias = [DiscoveryMatch(corpus_id="gdpr", confidence=1.0, reason="alias_match")]
        probe = [DiscoveryMatch(corpus_id="gdpr", confidence=0.7, reason="retrieval_probe")]
        merged = _merge_stages(alias, probe, [])
        assert len([m for m in merged if m.corpus_id == "gdpr"]) == 1

    def test_sorted_by_confidence_desc(self) -> None:
        probe = [
            DiscoveryMatch(corpus_id="ai_act", confidence=0.6, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="gdpr", confidence=0.9, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="dora", confidence=0.3, reason="retrieval_probe"),
        ]
        merged = _merge_stages([], probe, [])
        confs = [m.confidence for m in merged]
        assert confs == sorted(confs, reverse=True)


# ===========================================================================
# 1G: Confidence Gating
# ===========================================================================


class TestConfidenceGating:
    def test_auto_single_no_suggest(self, default_config: DiscoveryConfig) -> None:
        """One AUTO corpus, others below SUGGEST → only AUTO corpus used."""
        matches = [
            DiscoveryMatch(corpus_id="ai_act", confidence=0.85, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="gdpr", confidence=0.30, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "AUTO"
        assert result.resolved_scope == "explicit"
        assert result.resolved_corpora == ("ai_act",)

    def test_auto_only_includes_auto_tier_in_resolved(self, default_config: DiscoveryConfig) -> None:
        """AUTO gate should only include AUTO-tier corpora in resolved_corpora.

        Scenario: alias match for GDPR (1.0), retrieval probe finds NIS2 (0.55)
        and DORA (0.55) below suggest_threshold (0.65).
        Expected: only GDPR in resolved_corpora — sub-threshold corpora are
        visible in matches but not in resolved_corpora (reduces banner noise).
        """
        matches = [
            DiscoveryMatch(corpus_id="gdpr", confidence=1.0, reason="alias_match"),
            DiscoveryMatch(corpus_id="nis2", confidence=0.55, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="dora", confidence=0.55, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="elan", confidence=0.30, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "AUTO"
        assert result.resolved_scope == "explicit"
        # Only AUTO-tier (≥0.75) in resolved_corpora
        assert result.resolved_corpora == ("gdpr",)
        # SUGGEST-tier still visible in matches for sidebar display
        all_ids = {m.corpus_id for m in result.matches}
        assert "nis2" in all_ids
        assert "dora" in all_ids

    def test_auto_multiple(self, default_config: DiscoveryConfig) -> None:
        matches = [
            DiscoveryMatch(corpus_id="nis2", confidence=0.89, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="dora", confidence=0.82, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="gdpr", confidence=0.76, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "AUTO"
        assert result.resolved_scope == "explicit"
        assert set(result.resolved_corpora) == {"nis2", "dora", "gdpr"}

    def test_suggest(self, default_config: DiscoveryConfig) -> None:
        matches = [
            DiscoveryMatch(corpus_id="data_act", confidence=0.68, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="gdpr", confidence=0.55, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "SUGGEST"

    def test_abstain(self, default_config: DiscoveryConfig) -> None:
        matches = [
            DiscoveryMatch(corpus_id="ai_act", confidence=0.38, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="gdpr", confidence=0.31, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "ABSTAIN"
        assert result.resolved_scope == "abstain"
        assert result.resolved_corpora == ()

    def test_at_auto_boundary(self, default_config: DiscoveryConfig) -> None:
        matches = [
            DiscoveryMatch(corpus_id="ai_act", confidence=0.75, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "AUTO"
        assert "ai_act" in result.resolved_corpora

    def test_at_suggest_boundary(self, default_config: DiscoveryConfig) -> None:
        matches = [
            DiscoveryMatch(corpus_id="ai_act", confidence=0.65, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "SUGGEST"

    def test_max_corpora_limit(self) -> None:
        config = DiscoveryConfig(max_corpora=2)
        matches = [
            DiscoveryMatch(corpus_id="a", confidence=0.90, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="b", confidence=0.85, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="c", confidence=0.80, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, config)
        assert len(result.resolved_corpora) <= 2

    def test_all_suggest_no_auto(self, default_config: DiscoveryConfig) -> None:
        matches = [
            DiscoveryMatch(corpus_id="a", confidence=0.70, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="b", confidence=0.68, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "SUGGEST"
        assert result.resolved_scope == "explicit"
        assert len(result.resolved_corpora) == 2

    def test_empty_matches(self, default_config: DiscoveryConfig) -> None:
        result = _apply_confidence_gating([], default_config)
        assert result.gate == "ABSTAIN"
        assert result.resolved_corpora == ()

    def test_abstain_on_vague_query_many_suggest(self, default_config: DiscoveryConfig) -> None:
        """3+ corpora above SUGGEST but none reaching AUTO → ABSTAIN.

        Vague queries like 'hvad er loven' match every legal corpus moderately.
        When >2 corpora pass the suggest threshold without any reaching AUTO,
        the query lacks specificity and the system should abstain.
        """
        matches = [
            DiscoveryMatch(corpus_id="gdpr", confidence=0.70, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="nis2", confidence=0.69, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="ai_act", confidence=0.68, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="data_act", confidence=0.55, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "ABSTAIN"
        assert result.resolved_corpora == ()

    def test_suggest_allowed_with_few_corpora(self, default_config: DiscoveryConfig) -> None:
        """At most 2 corpora above SUGGEST (<=max_suggest_corpora) is legitimate."""
        matches = [
            DiscoveryMatch(corpus_id="data_act", confidence=0.70, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="gdpr", confidence=0.68, reason="retrieval_probe"),
            DiscoveryMatch(corpus_id="ai_act", confidence=0.40, reason="retrieval_probe"),
        ]
        result = _apply_confidence_gating(matches, default_config)
        assert result.gate == "SUGGEST"


# ===========================================================================
# 1H: End-to-End
# ===========================================================================


class TestDiscoverCorporaE2E:
    def test_auto_via_alias(self, default_config: DiscoveryConfig) -> None:
        result = discover_corpora(
            question="Hvad siger GDPR om databehandleraftaler?",
            corpus_ids=["gdpr", "ai_act"],
            probe_fn=make_probe_fn({}),
            resolver=FakeCorpusResolver({"gdpr": ["gdpr"]}),
            config=default_config,
        )
        assert result.gate == "AUTO"
        assert result.resolved_corpora == ("gdpr",)
        assert result.matches[0].reason == "alias_match"

    def test_abstain_on_vague_query(self, default_config: DiscoveryConfig) -> None:
        result = discover_corpora(
            question="Hvad er loven?",
            corpus_ids=["ai_act", "gdpr"],
            probe_fn=make_probe_fn({
                "ai_act": [({"chunk_id": "a1"}, 10.0)],
                "gdpr": [({"chunk_id": "g1"}, 10.0)],
            }),
            resolver=FakeCorpusResolver({}),
            config=default_config,
        )
        assert result.gate == "ABSTAIN"

    def test_empty_corpus_list(self, default_config: DiscoveryConfig) -> None:
        result = discover_corpora(
            question="anything",
            corpus_ids=[],
            probe_fn=make_probe_fn({}),
            resolver=FakeCorpusResolver({}),
            config=default_config,
        )
        assert result.gate == "ABSTAIN"
        assert result.resolved_corpora == ()

    def test_probe_fn_error_returns_abstain(self, default_config: DiscoveryConfig) -> None:
        result = discover_corpora(
            question="facial recognition",
            corpus_ids=["ai_act"],
            probe_fn=make_failing_probe_fn(),
            resolver=FakeCorpusResolver({}),
            config=default_config,
        )
        assert result.gate == "ABSTAIN"

    def test_results_sorted_by_confidence(self, default_config: DiscoveryConfig) -> None:
        result = discover_corpora(
            question="something about laws",
            corpus_ids=["a", "b", "c"],
            probe_fn=make_probe_fn({
                "a": [({"chunk_id": "x"}, 0.5)] * 3,
                "b": [({"chunk_id": "x"}, 0.2)] * 5,
                "c": [({"chunk_id": "x"}, 1.5)] * 2,
            }),
            resolver=FakeCorpusResolver({}),
            config=default_config,
        )
        confs = [m.confidence for m in result.matches]
        assert confs == sorted(confs, reverse=True)

    def test_hyphenated_corpus_ids_work_with_alias_and_probe(
        self, default_config: DiscoveryConfig,
    ) -> None:
        """Config-format corpus IDs (hyphenated) work correctly in discovery.

        When corpus_ids use hyphens (as in corpora.json config) and the resolver
        also returns hyphenated keys, alias matches and probe results should
        merge correctly — no duplicate entries for the same corpus.
        """
        result = discover_corpora(
            question="Hvad siger AI-Act om højrisiko?",
            corpus_ids=["ai-act", "gdpr"],
            probe_fn=make_probe_fn({
                "ai-act": [({"chunk_id": "a1"}, 0.3)] * 5,  # good similarity
                "gdpr": [({"chunk_id": "g1"}, 1.5)],  # poor similarity
            }),
            resolver=FakeCorpusResolver({"ai-act": ["ai-act", "ai act"]}),
            config=default_config,
        )
        assert result.gate == "AUTO"
        # ai-act should appear ONCE (alias wins over probe), not duplicated
        ai_act_matches = [m for m in result.matches if m.corpus_id == "ai-act"]
        assert len(ai_act_matches) == 1
        assert ai_act_matches[0].confidence == 1.0  # alias match
        assert ai_act_matches[0].reason == "alias_match"
        assert "ai-act" in result.resolved_corpora

    def test_stateless_same_input_same_output(self, default_config: DiscoveryConfig) -> None:
        kwargs = dict(
            question="GDPR databehandleraftaler",
            corpus_ids=["gdpr", "ai_act"],
            probe_fn=make_probe_fn({
                "gdpr": [({"chunk_id": "g1"}, 0.3)],
                "ai_act": [({"chunk_id": "a1"}, 1.5)],
            }),
            resolver=FakeCorpusResolver({"gdpr": ["gdpr"]}),
            config=default_config,
        )
        r1 = discover_corpora(**kwargs)
        r2 = discover_corpora(**kwargs)
        assert r1 == r2
