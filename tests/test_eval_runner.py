"""Tests for src/eval/eval_runner.py - Evaluation runner logic."""

import json
import tempfile
from pathlib import Path

import pytest

from src.eval.eval_runner import (
    _derive_run_mode,
    _normalize_anchor,
    _build_context_text,
    _parse_case_ids,
    load_golden_cases,
    _load_yaml_or_json,
)


class TestDeriveRunMode:
    """Tests for _derive_run_mode function."""

    def test_skip_llm_returns_retrieval_only(self):
        """skip_llm=True returns 'retrieval_only' mode."""
        result = _derive_run_mode(skip_llm=True, llm_judge=None)
        assert result == "retrieval_only"

    def test_skip_llm_ignores_llm_judge(self):
        """skip_llm=True returns 'retrieval_only' even if llm_judge=True."""
        result = _derive_run_mode(skip_llm=True, llm_judge=True)
        assert result == "retrieval_only"

    def test_no_skip_llm_judge_none_returns_full_with_judge(self):
        """Default (llm_judge=None) returns 'full_with_judge'."""
        result = _derive_run_mode(skip_llm=False, llm_judge=None)
        assert result == "full_with_judge"

    def test_llm_judge_true_returns_full_with_judge(self):
        """llm_judge=True returns 'full_with_judge'."""
        result = _derive_run_mode(skip_llm=False, llm_judge=True)
        assert result == "full_with_judge"

    def test_llm_judge_false_returns_full(self):
        """llm_judge=False returns 'full' (no judge)."""
        result = _derive_run_mode(skip_llm=False, llm_judge=False)
        assert result == "full"


class TestNormalizeAnchor:
    """Tests for _normalize_anchor function."""

    def test_strips_whitespace(self):
        """Strips leading and trailing whitespace."""
        result = _normalize_anchor("  Article 6  ")
        assert result == "article6"

    def test_lowercases(self):
        """Converts to lowercase."""
        result = _normalize_anchor("ARTICLE 6")
        assert result == "article6"

    def test_removes_internal_whitespace(self):
        """Removes all internal whitespace."""
        result = _normalize_anchor("Article  6 (1) (a)")
        assert result == "article6(1)(a)"

    def test_handles_none(self):
        """Handles None input."""
        result = _normalize_anchor(None)
        assert result == ""

    def test_handles_empty_string(self):
        """Handles empty string."""
        result = _normalize_anchor("")
        assert result == ""

    def test_handles_numbers(self):
        """Handles numeric input."""
        result = _normalize_anchor(6)
        assert result == "6"


class TestBuildContextText:
    """Tests for _build_context_text function."""

    def test_builds_from_chunk_text(self):
        """Extracts text from 'chunk_text' field."""
        refs = [{"chunk_text": "This is article 6 content."}]
        result = _build_context_text(refs)
        assert "This is article 6 content." in result

    def test_builds_from_text_field(self):
        """Falls back to 'text' field."""
        refs = [{"text": "Alternative text field."}]
        result = _build_context_text(refs)
        assert "Alternative text field." in result

    def test_builds_from_content_field(self):
        """Falls back to 'content' field."""
        refs = [{"content": "Content field text."}]
        result = _build_context_text(refs)
        assert "Content field text." in result

    def test_adds_anchor_prefix(self):
        """Adds anchor info as prefix."""
        refs = [{"chunk_text": "Content", "article": "6"}]
        result = _build_context_text(refs)
        assert "[Article 6]" in result
        assert "Content" in result

    def test_combines_multiple_anchors(self):
        """Combines multiple anchor types."""
        refs = [{"chunk_text": "Content", "article": "6", "recital": "42"}]
        result = _build_context_text(refs)
        assert "Article 6" in result
        assert "Recital 42" in result

    def test_handles_annex(self):
        """Includes annex in anchor prefix."""
        refs = [{"chunk_text": "Annex content", "annex": "III"}]
        result = _build_context_text(refs)
        assert "[Annex III]" in result

    def test_joins_multiple_refs(self):
        """Joins multiple references with double newline."""
        refs = [
            {"chunk_text": "First ref"},
            {"chunk_text": "Second ref"},
        ]
        result = _build_context_text(refs)
        assert "First ref" in result
        assert "Second ref" in result
        assert "\n\n" in result

    def test_skips_non_dict_refs(self):
        """Skips non-dict references."""
        refs = [{"chunk_text": "Valid"}, "invalid", None]
        result = _build_context_text(refs)
        assert "Valid" in result
        assert "invalid" not in result

    def test_empty_refs_returns_empty(self):
        """Empty refs list returns empty string."""
        result = _build_context_text([])
        assert result == ""

    def test_refs_without_text_skipped(self):
        """References without text content are skipped."""
        refs = [{"article": "6"}]  # No text field
        result = _build_context_text(refs)
        assert result == ""


class TestParseCaseIds:
    """Tests for _parse_case_ids function."""

    def test_parses_single_id(self):
        """Parses single case ID."""
        result = _parse_case_ids("case_1")
        assert result == {"case_1"}

    def test_parses_multiple_ids(self):
        """Parses comma-separated IDs."""
        result = _parse_case_ids("case_1,case_2,case_3")
        assert result == {"case_1", "case_2", "case_3"}

    def test_strips_whitespace(self):
        """Strips whitespace from IDs."""
        result = _parse_case_ids("case_1 , case_2 , case_3")
        assert result == {"case_1", "case_2", "case_3"}

    def test_empty_string_returns_empty_set(self):
        """Empty string returns empty set."""
        result = _parse_case_ids("")
        assert result == set()

    def test_ignores_empty_parts(self):
        """Ignores empty parts from trailing commas."""
        result = _parse_case_ids("case_1,,case_2,")
        assert result == {"case_1", "case_2"}


class TestLoadYamlOrJson:
    """Tests for _load_yaml_or_json function."""

    def test_loads_json_file(self):
        """Loads valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"id": "test", "value": 42}], f)
            f.flush()
            path = Path(f.name)

        try:
            result = _load_yaml_or_json(path)
            assert result == [{"id": "test", "value": 42}]
        finally:
            path.unlink()

    def test_loads_yaml_file(self):
        """Loads valid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- id: test\n  value: 42\n")
            f.flush()
            path = Path(f.name)

        try:
            result = _load_yaml_or_json(path)
            assert result == [{"id": "test", "value": 42}]
        finally:
            path.unlink()


class TestLoadGoldenCases:
    """Tests for load_golden_cases function."""

    def _create_temp_yaml(self, content: str) -> Path:
        """Helper to create temp YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()
            return Path(f.name)

    def test_loads_valid_case(self):
        """Loads valid golden case."""
        yaml_content = """
- id: test_case_1
  profile: LEGAL
  prompt: "What does Article 6 require?"
  expected:
    must_include_any_of:
      - Article 6
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            assert len(cases) == 1
            assert cases[0].id == "test_case_1"
            assert cases[0].profile == "LEGAL"
            assert cases[0].prompt == "What does Article 6 require?"
            assert "article6" in cases[0].expected.must_include_any_of
        finally:
            path.unlink()

    def test_normalizes_anchors(self):
        """Normalizes anchor strings in expected."""
        yaml_content = """
- id: test_case
  profile: LEGAL
  prompt: "Test"
  expected:
    must_include_any_of:
      - "  Article 6  "
    must_include_all_of:
      - "ANNEX III"
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            assert "article6" in cases[0].expected.must_include_any_of
            assert "annexiii" in cases[0].expected.must_include_all_of
        finally:
            path.unlink()

    def test_engineering_profile_defaults_contract_check(self):
        """ENGINEERING profile defaults contract_check to True."""
        yaml_content = """
- id: eng_case
  profile: ENGINEERING
  prompt: "Test"
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            assert cases[0].expected.contract_check is True
            assert cases[0].expected.min_citations == 2
            assert cases[0].expected.max_citations == 8
        finally:
            path.unlink()

    def test_legal_profile_no_contract_check_default(self):
        """LEGAL profile defaults contract_check to False."""
        yaml_content = """
- id: legal_case
  profile: LEGAL
  prompt: "Test"
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            assert cases[0].expected.contract_check is False
        finally:
            path.unlink()

    def test_behavior_defaults_to_answer(self):
        """Behavior defaults to 'answer'."""
        yaml_content = """
- id: test_case
  profile: LEGAL
  prompt: "Test"
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            assert cases[0].expected.behavior == "answer"
        finally:
            path.unlink()

    def test_abstain_behavior(self):
        """Parses abstain behavior."""
        yaml_content = """
- id: abstain_case
  profile: LEGAL
  prompt: "Is my system prohibited?"
  expected:
    behavior: abstain
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            assert cases[0].expected.behavior == "abstain"
        finally:
            path.unlink()

    def test_sorts_by_id(self):
        """Cases are sorted by ID."""
        yaml_content = """
- id: z_case
  profile: LEGAL
  prompt: "Test Z"
- id: a_case
  profile: LEGAL
  prompt: "Test A"
- id: m_case
  profile: LEGAL
  prompt: "Test M"
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            ids = [c.id for c in cases]
            assert ids == ["a_case", "m_case", "z_case"]
        finally:
            path.unlink()

    def test_raises_on_missing_id(self):
        """Raises ValueError when case missing ID."""
        yaml_content = """
- profile: LEGAL
  prompt: "No ID"
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="missing 'id'"):
                load_golden_cases(path)
        finally:
            path.unlink()

    def test_raises_on_invalid_profile(self):
        """Raises ValueError on invalid profile."""
        yaml_content = """
- id: test
  profile: INVALID
  prompt: "Test"
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="invalid profile"):
                load_golden_cases(path)
        finally:
            path.unlink()

    def test_raises_on_missing_prompt(self):
        """Raises ValueError when case missing prompt."""
        yaml_content = """
- id: test
  profile: LEGAL
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="missing 'prompt'"):
                load_golden_cases(path)
        finally:
            path.unlink()

    def test_raises_on_invalid_behavior(self):
        """Raises ValueError on invalid behavior."""
        yaml_content = """
- id: test
  profile: LEGAL
  prompt: "Test"
  expected:
    behavior: invalid
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="invalid behavior"):
                load_golden_cases(path)
        finally:
            path.unlink()

    def test_raises_on_non_list(self):
        """Raises ValueError when file doesn't contain a list."""
        yaml_content = """
id: single_case
profile: LEGAL
prompt: "Not a list"
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="expected a list"):
                load_golden_cases(path)
        finally:
            path.unlink()

    def test_parses_test_types(self):
        """Parses test_types field."""
        yaml_content = """
- id: test_case
  profile: LEGAL
  prompt: "Test"
  test_types:
    - retrieval
    - generation
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            assert cases[0].test_types == ("retrieval", "generation")
        finally:
            path.unlink()

    def test_parses_origin(self):
        """Parses origin field."""
        yaml_content = """
- id: test_case
  profile: LEGAL
  prompt: "Test"
  origin: manual
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            assert cases[0].origin == "manual"
        finally:
            path.unlink()

    def test_origin_defaults_to_auto(self):
        """Origin defaults to 'auto'."""
        yaml_content = """
- id: test_case
  profile: LEGAL
  prompt: "Test"
"""
        path = self._create_temp_yaml(yaml_content)
        try:
            cases = load_golden_cases(path)
            assert cases[0].origin == "auto"
        finally:
            path.unlink()
