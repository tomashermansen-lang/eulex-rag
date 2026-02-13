"""Tests for src/engine/concept_config.py - Policy and anchor configuration.

Covers:
- Anchor normalization functions
- Intent key normalization
- Profile list handling
- Policy dataclasses
- Parsing functions for v2 config format
- Anchor extraction from metadata
- Policy merging and effective policy computation
- Anchor hint bump order calculation
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.engine.concept_config import (
    _normalize_anchor,
    _normalize_intent_key,
    _dedupe_preserve_order,
    _normalize_anchor_list,
    _normalize_profile_list,
    NormativeGuardPolicy,
    RescueRule,
    AnswerPolicy,
    Policy,
    _parse_v2_normative_guard,
    _parse_v2_rescue_rules,
    _parse_v2_answer_policy,
    _parse_intent_entry,
    extract_anchors_from_metadata,
    load_concept_config,
    anchor_hint_bumping_enabled,
    get_hint_anchors,
    get_effective_policy,
    compute_anchor_hint_bump_order,
    AnchorHintBumpResult,
)


# ─────────────────────────────────────────────────────────────────────────────
# Normalization Functions
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeAnchor:
    """Tests for _normalize_anchor function."""

    def test_lowercases_and_strips(self):
        """Anchor is lowercased and whitespace stripped."""
        assert _normalize_anchor("  ARTICLE:6  ") == "article:6"
        assert _normalize_anchor("Article:42") == "article:42"

    def test_removes_internal_whitespace(self):
        """Internal whitespace is removed."""
        assert _normalize_anchor("article : 6") == "article:6"
        assert _normalize_anchor("annex : iii : 5") == "annex:iii:5"

    def test_handles_empty_and_none(self):
        """Empty strings and None return empty string."""
        assert _normalize_anchor("") == ""
        assert _normalize_anchor(None) == ""
        assert _normalize_anchor("   ") == ""


class TestNormalizeIntentKey:
    """Tests for _normalize_intent_key function."""

    def test_lowercases_and_strips(self):
        """Intent key is lowercased and stripped."""
        assert _normalize_intent_key("  LOGGING  ") == "logging"

    def test_strips_legacy_prefix(self):
        """Legacy 'legalconcept.' prefix is stripped."""
        assert _normalize_intent_key("legalconcept.logging") == "logging"
        assert _normalize_intent_key("LegalConcept.LOGGING") == "logging"

    def test_removes_internal_whitespace(self):
        """Internal whitespace is removed."""
        assert _normalize_intent_key("logging and records") == "loggingandrecords"

    def test_handles_empty_and_none(self):
        """Empty strings and None return empty string."""
        assert _normalize_intent_key("") == ""
        assert _normalize_intent_key(None) == ""


class TestDedupePreserveOrder:
    """Tests for _dedupe_preserve_order function."""

    def test_removes_duplicates(self):
        """Duplicates are removed."""
        assert _dedupe_preserve_order(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_preserves_order(self):
        """Original order is preserved."""
        assert _dedupe_preserve_order(["c", "b", "a"]) == ["c", "b", "a"]

    def test_handles_empty_input(self):
        """Empty input returns empty list."""
        assert _dedupe_preserve_order([]) == []
        assert _dedupe_preserve_order(None) == []

    def test_filters_empty_strings(self):
        """Empty strings are filtered out."""
        assert _dedupe_preserve_order(["a", "", "b", None, "c"]) == ["a", "b", "c"]


class TestNormalizeAnchorList:
    """Tests for _normalize_anchor_list function."""

    def test_normalizes_anchors(self):
        """Anchors are normalized."""
        result = _normalize_anchor_list(["ARTICLE:6", "Annex:III"])
        assert result == ["article:6", "annex:iii"]

    def test_filters_invalid_anchors(self):
        """Anchors without colons are filtered."""
        result = _normalize_anchor_list(["article:6", "invalid", "annex:iii"])
        assert result == ["article:6", "annex:iii"]

    def test_deduplicates(self):
        """Duplicate anchors are removed."""
        result = _normalize_anchor_list(["article:6", "ARTICLE:6", "article:6"])
        assert result == ["article:6"]

    def test_handles_non_list_input(self):
        """Non-list input returns empty list."""
        assert _normalize_anchor_list("article:6") == []
        assert _normalize_anchor_list(None) == []
        assert _normalize_anchor_list(123) == []

    def test_filters_non_string_items(self):
        """Non-string items are filtered."""
        result = _normalize_anchor_list(["article:6", 123, None, "annex:iii"])
        assert result == ["article:6", "annex:iii"]


class TestNormalizeProfileList:
    """Tests for _normalize_profile_list function."""

    def test_normalizes_to_uppercase(self):
        """Profiles are normalized to uppercase."""
        result = _normalize_profile_list(["legal", "Engineering"])
        assert result == ("LEGAL", "ENGINEERING")

    def test_filters_invalid_profiles(self):
        """Invalid profiles are filtered."""
        result = _normalize_profile_list(["LEGAL", "invalid", "ANY"])
        assert result == ("LEGAL", "ANY")

    def test_returns_any_for_empty(self):
        """Empty list returns ('ANY',)."""
        assert _normalize_profile_list([]) == ("ANY",)
        assert _normalize_profile_list(None) == ("ANY",)

    def test_deduplicates(self):
        """Duplicate profiles are removed."""
        result = _normalize_profile_list(["LEGAL", "legal", "LEGAL"])
        assert result == ("LEGAL",)


# ─────────────────────────────────────────────────────────────────────────────
# Policy Dataclasses
# ─────────────────────────────────────────────────────────────────────────────


class TestNormativeGuardPolicy:
    """Tests for NormativeGuardPolicy dataclass."""

    def test_applies_to_any_profile(self):
        """Profile matching with ANY."""
        policy = NormativeGuardPolicy(required_support="article", profiles=("ANY",))
        assert policy.applies_to_profile("LEGAL") is True
        assert policy.applies_to_profile("ENGINEERING") is True
        assert policy.applies_to_profile("") is True

    def test_applies_to_specific_profile(self):
        """Profile matching with specific profiles."""
        policy = NormativeGuardPolicy(
            required_support="article", profiles=("LEGAL",)
        )
        assert policy.applies_to_profile("LEGAL") is True
        assert policy.applies_to_profile("ENGINEERING") is False

    def test_case_insensitive_profile_matching(self):
        """Profile matching is case-insensitive."""
        policy = NormativeGuardPolicy(
            required_support="article", profiles=("LEGAL",)
        )
        assert policy.applies_to_profile("legal") is True
        assert policy.applies_to_profile("Legal") is True


class TestRescueRule:
    """Tests for RescueRule dataclass."""

    def test_applies_to_any_profile(self):
        """Profile matching with ANY."""
        rule = RescueRule(
            if_present=("article:6",),
            must_include_one_of=("article:7",),
            action="anchor_lookup_inject",
            profiles=("ANY",),
        )
        assert rule.applies_to_profile("LEGAL") is True
        assert rule.applies_to_profile("ENGINEERING") is True

    def test_applies_to_specific_profile(self):
        """Profile matching with specific profiles."""
        rule = RescueRule(
            if_present=("article:6",),
            must_include_one_of=("article:7",),
            action="anchor_lookup_inject",
            profiles=("ENGINEERING",),
        )
        assert rule.applies_to_profile("ENGINEERING") is True
        assert rule.applies_to_profile("LEGAL") is False


class TestAnswerPolicy:
    """Tests for AnswerPolicy dataclass."""

    def test_to_debug_dict(self):
        """Debug dict contains all fields."""
        policy = AnswerPolicy(
            intent_category="REQUIREMENTS",
            requirements_first=True,
            include_audit_evidence=True,
            min_section3_bullets=3,
        )
        debug = policy.to_debug_dict()
        assert debug == {
            "intent_category": "REQUIREMENTS",
            "requirements_first": True,
            "include_audit_evidence": True,
            "min_section3_bullets": 3,
        }

    def test_to_debug_dict_with_none_bullets(self):
        """Debug dict handles None min_section3_bullets."""
        policy = AnswerPolicy(
            intent_category="OTHER",
            requirements_first=False,
            include_audit_evidence=False,
            min_section3_bullets=None,
        )
        debug = policy.to_debug_dict()
        assert debug["min_section3_bullets"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Parsing Functions
# ─────────────────────────────────────────────────────────────────────────────


class TestParseV2NormativeGuard:
    """Tests for _parse_v2_normative_guard function."""

    def test_parses_valid_config(self):
        """Parses valid normative guard config."""
        config = {
            "required_support": "article",
            "profiles": ["LEGAL"],
        }
        result = _parse_v2_normative_guard(config)
        assert result is not None
        assert result.required_support == "article"
        assert result.profiles == ("LEGAL",)

    def test_accepts_valid_support_types(self):
        """Accepts article, article_or_annex, any."""
        for support in ["article", "article_or_annex", "any"]:
            result = _parse_v2_normative_guard({"required_support": support})
            assert result is not None
            assert result.required_support == support

    def test_rejects_invalid_support_type(self):
        """Rejects invalid required_support values."""
        result = _parse_v2_normative_guard({"required_support": "invalid"})
        assert result is None

    def test_returns_none_for_non_dict(self):
        """Returns None for non-dict input."""
        assert _parse_v2_normative_guard("string") is None
        assert _parse_v2_normative_guard(None) is None
        assert _parse_v2_normative_guard([]) is None


class TestParseV2RescueRules:
    """Tests for _parse_v2_rescue_rules function."""

    def test_parses_valid_rules(self):
        """Parses valid rescue rules."""
        config = [
            {
                "if_present": ["article:6"],
                "must_include_one_of": ["article:7"],
                "action": "anchor_lookup_inject",
                "profiles": ["LEGAL"],
            }
        ]
        result = _parse_v2_rescue_rules(config)
        assert len(result) == 1
        assert result[0].if_present == ("article:6",)
        assert result[0].must_include_one_of == ("article:7",)
        assert result[0].action == "anchor_lookup_inject"

    def test_accepts_valid_actions(self):
        """Accepts anchor_lookup_inject and pool_expand_and_select."""
        for action in ["anchor_lookup_inject", "pool_expand_and_select"]:
            config = [{"action": action}]
            result = _parse_v2_rescue_rules(config)
            assert len(result) == 1
            assert result[0].action == action

    def test_rejects_invalid_action(self):
        """Rejects invalid action values."""
        config = [{"action": "invalid_action"}]
        result = _parse_v2_rescue_rules(config)
        assert result == []

    def test_returns_empty_for_non_list(self):
        """Returns empty list for non-list input."""
        assert _parse_v2_rescue_rules("string") == []
        assert _parse_v2_rescue_rules(None) == []
        assert _parse_v2_rescue_rules({}) == []


class TestParseV2AnswerPolicy:
    """Tests for _parse_v2_answer_policy function."""

    def test_parses_valid_config(self):
        """Parses valid answer policy config."""
        config = {
            "intent_category": "REQUIREMENTS",
            "requirements_first": True,
            "include_audit_evidence": True,
            "min_section3_bullets": 3,
        }
        result = _parse_v2_answer_policy(config)
        assert result is not None
        assert result.intent_category == "REQUIREMENTS"
        assert result.requirements_first is True
        assert result.include_audit_evidence is True
        assert result.min_section3_bullets == 3

    def test_accepts_valid_categories(self):
        """Accepts REQUIREMENTS, ENFORCEMENT, OTHER."""
        for category in ["REQUIREMENTS", "ENFORCEMENT", "OTHER"]:
            result = _parse_v2_answer_policy({"intent_category": category})
            assert result is not None
            assert result.intent_category == category

    def test_rejects_invalid_category(self):
        """Rejects invalid intent_category values."""
        result = _parse_v2_answer_policy({"intent_category": "INVALID"})
        assert result is None

    def test_handles_invalid_min_bullets(self):
        """Handles invalid min_section3_bullets values."""
        # Negative value
        result = _parse_v2_answer_policy({
            "intent_category": "REQUIREMENTS",
            "min_section3_bullets": -1,
        })
        assert result.min_section3_bullets is None

        # Non-numeric value
        result = _parse_v2_answer_policy({
            "intent_category": "REQUIREMENTS",
            "min_section3_bullets": "not a number",
        })
        assert result.min_section3_bullets is None


class TestParseIntentEntry:
    """Tests for _parse_intent_entry function."""

    def test_parses_v1_list_format(self):
        """Parses v1 format (list of bump hints)."""
        result = _parse_intent_entry(["article:6", "article:7"])
        assert result is not None
        assert result["bump_hints"] == ["article:6", "article:7"]
        assert result["normative_guard"] is None
        assert result["rescue_rules"] == []

    def test_parses_v2_dict_format(self):
        """Parses v2 format (dict with multiple fields)."""
        config = {
            "bump_hints": ["article:6"],
            "normative_guard": {"required_support": "article"},
            "rescue_rules": [{"action": "anchor_lookup_inject"}],
            "answer_policy": {"intent_category": "REQUIREMENTS"},
        }
        result = _parse_intent_entry(config)
        assert result is not None
        assert result["bump_hints"] == ["article:6"]
        assert result["normative_guard"].required_support == "article"
        assert len(result["rescue_rules"]) == 1
        assert result["answer_policy"].intent_category == "REQUIREMENTS"

    def test_returns_none_for_invalid_input(self):
        """Returns None for invalid input types."""
        assert _parse_intent_entry("string") is None
        assert _parse_intent_entry(123) is None
        assert _parse_intent_entry(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# Anchor Extraction
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractAnchorsFromMetadata:
    """Tests for extract_anchors_from_metadata function."""

    def test_extracts_article_anchor(self):
        """Extracts article anchor from metadata."""
        meta = {"article": "6"}
        anchors = extract_anchors_from_metadata(meta)
        assert "article:6" in anchors

    def test_extracts_recital_anchor(self):
        """Extracts recital anchor from metadata."""
        meta = {"recital": "42"}
        anchors = extract_anchors_from_metadata(meta)
        assert "recital:42" in anchors

    def test_extracts_annex_anchor(self):
        """Extracts annex anchor from metadata."""
        meta = {"annex": "III"}
        anchors = extract_anchors_from_metadata(meta)
        assert "annex:iii" in anchors

    def test_extracts_annex_with_point(self):
        """Extracts annex anchor with point."""
        meta = {"annex": "III", "annex_point": "5"}
        anchors = extract_anchors_from_metadata(meta)
        assert "annex:iii" in anchors
        assert "annex:iii:5" in anchors

    def test_extracts_annex_with_section(self):
        """Extracts annex anchor with section."""
        meta = {"annex": "VIII", "annex_section": "A"}
        anchors = extract_anchors_from_metadata(meta)
        assert "annex:viii" in anchors
        assert "annex:viii:a" in anchors

    def test_extracts_annex_with_section_and_point(self):
        """Extracts full annex path with section and point."""
        meta = {"annex": "III", "annex_section": "A", "annex_point": "5"}
        anchors = extract_anchors_from_metadata(meta)
        assert "annex:iii" in anchors
        assert "annex:iii:a" in anchors
        assert "annex:iii:a:5" in anchors

    def test_handles_empty_metadata(self):
        """Handles empty or missing metadata."""
        assert extract_anchors_from_metadata({}) == set()
        assert extract_anchors_from_metadata(None) == set()

    def test_handles_missing_fields(self):
        """Handles metadata with missing optional fields."""
        meta = {"article": "6", "recital": ""}
        anchors = extract_anchors_from_metadata(meta)
        assert "article:6" in anchors
        assert len([a for a in anchors if "recital" in a]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Anchor Hint Bumping
# ─────────────────────────────────────────────────────────────────────────────


class TestAnchorHintBumpingEnabled:
    """Tests for anchor_hint_bumping_enabled function."""

    def test_returns_false_when_not_set(self, monkeypatch):
        """Returns False when env var not set."""
        monkeypatch.delenv("ANCHOR_HINT_BUMPING", raising=False)
        assert anchor_hint_bumping_enabled() is False

    def test_returns_true_for_truthy_values(self, monkeypatch):
        """Returns True for truthy env values."""
        for value in ["1", "true", "yes", "on", "TRUE", "Yes"]:
            monkeypatch.setenv("ANCHOR_HINT_BUMPING", value)
            assert anchor_hint_bumping_enabled() is True

    def test_returns_false_for_falsy_values(self, monkeypatch):
        """Returns False for non-truthy env values."""
        for value in ["0", "false", "no", "off", ""]:
            monkeypatch.setenv("ANCHOR_HINT_BUMPING", value)
            assert anchor_hint_bumping_enabled() is False


class TestComputeAnchorHintBumpOrder:
    """Tests for compute_anchor_hint_bump_order function."""

    def test_returns_original_order_when_no_hints(self):
        """Returns original order when no hint anchors."""
        metadatas = [{"article": "1"}, {"article": "2"}, {"article": "3"}]
        distances = [0.1, 0.2, 0.3]

        result = compute_anchor_hint_bump_order(
            metadatas=metadatas,
            distances=distances,
            hint_anchors=set(),
            bonus=0.5,
        )

        assert result.order == [0, 1, 2]
        assert result.applied is False

    def test_returns_original_order_when_zero_bonus(self):
        """Returns original order when bonus is zero."""
        metadatas = [{"article": "1"}, {"article": "2"}]
        distances = [0.1, 0.2]

        result = compute_anchor_hint_bump_order(
            metadatas=metadatas,
            distances=distances,
            hint_anchors={"article:2"},
            bonus=0.0,
        )

        assert result.applied is False

    def test_bumps_matching_anchors(self):
        """Bumps chunks matching hint anchors to top."""
        metadatas = [
            {"article": "1"},  # index 0, distance 0.1
            {"article": "2"},  # index 1, distance 0.2, matches hint
            {"article": "3"},  # index 2, distance 0.3
        ]
        distances = [0.1, 0.2, 0.3]

        result = compute_anchor_hint_bump_order(
            metadatas=metadatas,
            distances=distances,
            hint_anchors={"article:2"},
            bonus=0.5,
        )

        assert result.applied is True
        assert result.order[0] == 1  # article:2 bumped to first
        assert result.matched_indices == [1]

    def test_preserves_relative_order_among_matched(self):
        """Preserves relative order among matched chunks."""
        metadatas = [
            {"article": "1"},  # index 0, no match
            {"article": "2"},  # index 1, matches, distance 0.2
            {"article": "3"},  # index 2, matches, distance 0.3
        ]
        distances = [0.1, 0.2, 0.3]

        result = compute_anchor_hint_bump_order(
            metadatas=metadatas,
            distances=distances,
            hint_anchors={"article:2", "article:3"},
            bonus=0.5,
        )

        # Both matched chunks should come before unmatched
        assert 1 in result.order[:2]
        assert 2 in result.order[:2]
        assert result.matched_indices == [1, 2]

    def test_deterministic_tiebreak(self):
        """Uses deterministic tiebreak (anchor_id, chunk_id, original index)."""
        metadatas = [
            {"article": "2", "chunk_id": "c1"},
            {"article": "1", "chunk_id": "c2"},
        ]
        distances = [0.1, 0.1]  # Same distance

        result = compute_anchor_hint_bump_order(
            metadatas=metadatas,
            distances=distances,
            hint_anchors={"article:1", "article:2"},
            bonus=0.5,
        )

        # Both have same score after bump, tiebreak by anchor_id (alphabetically)
        # article:1 < article:2, so index 1 should come first
        assert result.order[0] == 1

    def test_records_bonus_in_result(self):
        """Records applied bonus in result."""
        result = compute_anchor_hint_bump_order(
            metadatas=[{"article": "1"}],
            distances=[0.1],
            hint_anchors={"article:1"},
            bonus=0.75,
        )

        assert result.bonus == 0.75


# ─────────────────────────────────────────────────────────────────────────────
# Config Loading (with temp directory)
# ─────────────────────────────────────────────────────────────────────────────


class TestLoadConceptConfig:
    """Tests for load_concept_config function."""

    def test_returns_empty_for_missing_dir(self, monkeypatch, tmp_path):
        """Returns empty dict when concepts dir doesn't exist."""
        monkeypatch.setenv("CONCEPTS_DIR", str(tmp_path / "nonexistent"))

        # Clear the cache to pick up new env var
        load_concept_config.cache_clear()

        result = load_concept_config()
        assert result == {}

    def test_skips_template_files(self, monkeypatch, tmp_path):
        """Skips files starting with underscore."""
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        # Create a template file
        (concepts_dir / "_template.yaml").write_text(
            "concepts:\n  test:\n    keywords: [test]\n"
        )

        monkeypatch.setenv("CONCEPTS_DIR", str(concepts_dir))
        load_concept_config.cache_clear()

        result = load_concept_config()
        assert "_template" not in result


class TestGetEffectivePolicy:
    """Tests for get_effective_policy function."""

    def test_returns_empty_policy_for_missing_corpus(self, monkeypatch, tmp_path):
        """Returns empty policy for non-existent corpus."""
        monkeypatch.setenv("CONCEPTS_DIR", str(tmp_path / "nonexistent"))
        load_concept_config.cache_clear()

        policy = get_effective_policy(corpus_id="nonexistent", intent_keys=[])

        assert policy.bump_hints == ()
        assert policy.normative_guard is None
        assert policy.rescue_rules == ()

    def test_normalizes_intent_keys(self, monkeypatch, tmp_path):
        """Intent keys are normalized before lookup."""
        # This tests that legalconcept. prefix is stripped
        concepts_dir = tmp_path / "concepts"
        concepts_dir.mkdir()

        monkeypatch.setenv("CONCEPTS_DIR", str(concepts_dir))
        load_concept_config.cache_clear()

        policy = get_effective_policy(
            corpus_id="test",
            intent_keys=["legalconcept.LOGGING", "  spaces  "],
        )

        # Should normalize to ["logging", "spaces"]
        assert "logging" in policy.intent_keys_effective or policy.intent_keys_effective == ()

    def test_records_requested_intent_keys(self, monkeypatch, tmp_path):
        """Records original requested intent keys."""
        monkeypatch.setenv("CONCEPTS_DIR", str(tmp_path / "nonexistent"))
        load_concept_config.cache_clear()

        policy = get_effective_policy(
            corpus_id="test",
            intent_keys=["LOGGING", "ENFORCEMENT"],
        )

        assert policy.intent_keys_requested == ("LOGGING", "ENFORCEMENT")
