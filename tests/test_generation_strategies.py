"""Tests for src/engine/generation_strategies.py - Strategy pattern generation."""

import pytest
from unittest.mock import MagicMock, patch

from src.engine.generation_types import (
    GenerationConfig,
    StructuredGenerationResult,
    UserProfile,
)
from src.engine.generation_strategies import (
    execute_prose_generation,
    execute_legal_json_generation,
    execute_engineering_json_generation,
    get_strategy,
    execute_structured_generation,
    execute_citation_retry_if_needed,
    build_engineering_answer,
)


# ---------------------------------------------------------------------------
# Test: execute_prose_generation
# ---------------------------------------------------------------------------


class TestExecuteProseGeneration:
    def test_returns_structured_result(self):
        config = GenerationConfig.for_legal(json_mode_enabled=False)
        llm_fn = MagicMock(return_value="This is a prose answer.")

        result = execute_prose_generation(
            prompt="test prompt",
            llm_fn=llm_fn,
            config=config,
            allowed_idxs={1, 2, 3},
        )

        assert isinstance(result, StructuredGenerationResult)
        assert result.answer_text == "This is a prose answer."
        assert result.raw_llm_response == "This is a prose answer."

    def test_extracts_citations_in_soft_mode(self):
        config = GenerationConfig(
            profile=UserProfile.LEGAL,
            citation_mode="soft",
        )
        llm_fn = MagicMock(return_value="See [1] and [2] for details.")

        result = execute_prose_generation(
            prompt="test",
            llm_fn=llm_fn,
            config=config,
            allowed_idxs={1, 2, 3},
        )

        assert 1 in result.cited_idxs
        assert 2 in result.cited_idxs
        assert result.valid_cited_idxs == [1, 2]

    def test_tracks_llm_calls(self):
        config = GenerationConfig.for_legal(json_mode_enabled=False)
        llm_fn = MagicMock(return_value="Answer")

        result = execute_prose_generation(
            prompt="test",
            llm_fn=llm_fn,
            config=config,
            allowed_idxs=set(),
        )

        assert result.debug.get("llm_calls_count") == 1
        llm_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Test: execute_legal_json_generation
# ---------------------------------------------------------------------------


class TestExecuteLegalJsonGeneration:
    def test_valid_json_response(self):
        config = GenerationConfig.for_legal(json_mode_enabled=True)
        valid_json = '{"summary": "Test summary", "key_points": ["Point 1 [1]"]}'
        llm_fn = MagicMock(return_value=valid_json)

        with patch("src.engine.generation_strategies.execute_legal_json_generation") as mock:
            mock.return_value = StructuredGenerationResult(
                answer_text="Test summary",
                raw_llm_response=valid_json,
                cited_idxs=[1],
                valid_cited_idxs=[1],
                debug={"json_parse_ok": True},
            )
            result = mock(
                prompt="test",
                llm_fn=llm_fn,
                config=config,
                allowed_idxs={1, 2},
            )

        assert result.debug.get("json_parse_ok") is True
        assert "Test summary" in result.answer_text

    def test_fallback_to_prose_on_json_failure(self):
        config = GenerationConfig.for_legal(json_mode_enabled=True)
        config.soft_json_fallback = True
        llm_fn = MagicMock(return_value="Not valid JSON, just prose with [1].")

        result = execute_legal_json_generation(
            prompt="test",
            llm_fn=llm_fn,
            config=config,
            allowed_idxs={1, 2},
        )

        # Should fall back to prose
        assert result.debug.get("json_parse_ok") is False or result.debug.get("legal_json_fallback_to_prose") is True
        assert "[1]" in result.answer_text or "Not valid JSON" in result.answer_text

    def test_fails_without_fallback(self):
        config = GenerationConfig.for_legal(json_mode_enabled=True)
        config.soft_json_fallback = False
        llm_fn = MagicMock(return_value="Invalid JSON")

        result = execute_legal_json_generation(
            prompt="test",
            llm_fn=llm_fn,
            config=config,
            allowed_idxs={1},
        )

        assert result.failed is True or result.answer_text == "MISSING_REF"


# ---------------------------------------------------------------------------
# Test: get_strategy
# ---------------------------------------------------------------------------


class TestGetStrategy:
    def test_returns_prose_for_no_json_schema(self):
        config = GenerationConfig(
            profile=UserProfile.LEGAL,
            require_json_schema=False,
        )
        strategy = get_strategy(config)
        assert strategy == execute_prose_generation

    def test_returns_legal_for_legal_schema(self):
        config = GenerationConfig(
            profile=UserProfile.LEGAL,
            require_json_schema=True,
            json_schema_type="legal",
        )
        strategy = get_strategy(config)
        assert strategy == execute_legal_json_generation

    def test_returns_engineering_for_engineering_schema(self):
        config = GenerationConfig(
            profile=UserProfile.ENGINEERING,
            require_json_schema=True,
            json_schema_type="engineering",
        )
        strategy = get_strategy(config)
        assert strategy == execute_engineering_json_generation


# ---------------------------------------------------------------------------
# Test: execute_structured_generation
# ---------------------------------------------------------------------------


class TestExecuteStructuredGeneration:
    def test_uses_auto_selected_strategy(self):
        config = GenerationConfig.for_legal(json_mode_enabled=False)
        llm_fn = MagicMock(return_value="Auto selected prose")

        result = execute_structured_generation(
            prompt="test",
            llm_fn=llm_fn,
            config=config,
            allowed_idxs={1},
        )

        assert result.answer_text == "Auto selected prose"

    def test_uses_injected_strategy(self):
        config = GenerationConfig.for_legal()
        llm_fn = MagicMock(return_value="Should not be called")

        custom_strategy = MagicMock(return_value=StructuredGenerationResult(
            answer_text="Custom strategy result",
            raw_llm_response="custom",
        ))

        result = execute_structured_generation(
            prompt="test",
            llm_fn=llm_fn,
            config=config,
            allowed_idxs={1},
            strategy=custom_strategy,
        )

        assert result.answer_text == "Custom strategy result"
        custom_strategy.assert_called_once()


# ---------------------------------------------------------------------------
# Test: execute_citation_retry_if_needed
# ---------------------------------------------------------------------------


class TestExecuteCitationRetryIfNeeded:
    def test_no_retry_when_sufficient_citations(self):
        run_meta = {}
        llm_fn = MagicMock(return_value="Should not be called")

        result = execute_citation_retry_if_needed(
            answer_text="Answer with [1] and [2] citations.",
            prompt="test",
            llm_fn=llm_fn,
            allowed_idxs={1, 2, 3},
            min_citations=2,
            run_meta=run_meta,
            strip_references_fn=lambda x: x,
        )

        assert result == "Answer with [1] and [2] citations."
        assert run_meta["llm_retry"]["retry_performed"] is False
        llm_fn.assert_not_called()

    def test_retry_when_insufficient_citations(self):
        run_meta = {}
        llm_fn = MagicMock(return_value="Improved answer with [1] and [2].")

        result = execute_citation_retry_if_needed(
            answer_text="Answer with only [1].",
            prompt="test",
            llm_fn=llm_fn,
            allowed_idxs={1, 2, 3},
            min_citations=2,
            run_meta=run_meta,
            strip_references_fn=lambda x: x,
        )

        assert run_meta["llm_retry"]["retry_performed"] is True
        assert "Improved answer" in result
        llm_fn.assert_called_once()

    def test_no_retry_when_not_enough_allowed(self):
        run_meta = {}
        llm_fn = MagicMock()

        result = execute_citation_retry_if_needed(
            answer_text="Answer with [1].",
            prompt="test",
            llm_fn=llm_fn,
            allowed_idxs={1},  # Only 1 allowed, but need 2
            min_citations=2,
            run_meta=run_meta,
            strip_references_fn=lambda x: x,
        )

        # Should not retry since there aren't enough allowed citations
        assert run_meta["llm_retry"]["retry_performed"] is False


# ---------------------------------------------------------------------------
# Test: build_engineering_answer
# ---------------------------------------------------------------------------


class TestBuildEngineeringAnswer:
    def test_preserves_missing_ref(self):
        result = build_engineering_answer(
            raw_interpretation="MISSING_REF",
            ctx=None,
            references_structured=[],
            reference_lines=[],
            distances=[],
        )
        assert result == "MISSING_REF"

    def test_insufficient_evidence_gating(self):
        result = build_engineering_answer(
            raw_interpretation="Some answer",
            ctx=None,
            references_structured=[],
            reference_lines=[],
            distances=[],
            total_retrieved=5,
            citable_count=1,
            min_citable_required=2,
        )

        assert "UTILSTRÆKKELIG_EVIDENS" in result
        assert "citerbart: 1" in result

    def test_appends_evidence_stats(self):
        result = build_engineering_answer(
            raw_interpretation="Good answer with evidence",
            ctx=None,
            references_structured=[],
            reference_lines=[],
            distances=[],
            total_retrieved=10,
            citable_count=5,
            min_citable_required=2,
        )

        assert "Good answer with evidence" in result
        assert "Evidens: 5 citerbart" in result


# ---------------------------------------------------------------------------
# Test: GenerationConfig factories
# ---------------------------------------------------------------------------


class TestGenerationConfigFactories:
    def test_engineering_config(self):
        config = GenerationConfig.for_engineering(contract_min_citations=3)

        assert config.profile == UserProfile.ENGINEERING
        assert config.require_json_schema is True
        assert config.min_citations == 3
        assert config.citation_mode == "strict"
        assert config.json_schema_type == "engineering"

    def test_legal_config(self):
        config = GenerationConfig.for_legal()

        assert config.profile == UserProfile.LEGAL
        assert config.citation_mode == "soft"
        assert config.soft_json_fallback is True
        assert config.json_schema_type == "legal"

    def test_for_profile_routing(self):
        eng_config = GenerationConfig.for_profile(UserProfile.ENGINEERING)
        legal_config = GenerationConfig.for_profile(UserProfile.LEGAL)

        assert eng_config.profile == UserProfile.ENGINEERING
        assert legal_config.profile == UserProfile.LEGAL


# ---------------------------------------------------------------------------
# Test: StructuredGenerationResult
# ---------------------------------------------------------------------------


class TestStructuredGenerationResult:
    def test_is_missing_ref_true(self):
        result = StructuredGenerationResult(
            answer_text="MISSING_REF",
            raw_llm_response="",
        )
        assert result.is_missing_ref is True

    def test_is_missing_ref_false(self):
        result = StructuredGenerationResult(
            answer_text="A real answer",
            raw_llm_response="",
        )
        assert result.is_missing_ref is False

    def test_default_values(self):
        result = StructuredGenerationResult(
            answer_text="test",
            raw_llm_response="test",
        )
        assert result.cited_idxs == []
        assert result.valid_cited_idxs == []
        assert result.repair_attempts == 0
        assert result.enrich_attempts == 0
        assert result.failed is False
