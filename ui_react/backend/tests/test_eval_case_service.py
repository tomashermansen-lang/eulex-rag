"""Tests for eval case CRUD service.

TDD: These tests are written BEFORE the implementation.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch


class TestLoadCases:
    """Tests for loading eval cases."""

    def test_load_cases_for_law_returns_list(self, tmp_path):
        """Should return list of cases for a law."""
        from services.eval_cases import load_cases_for_law

        # Create test golden cases file
        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-basic
  profile: LEGAL
  prompt: What is the test?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
- id: test-law-02-second
  profile: ENGINEERING
  prompt: How does it work?
  test_types:
    - retrieval
    - faithfulness
  origin: manual
  expected:
    must_include_any_of:
      - article:2
    behavior: answer
""")

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            cases = load_cases_for_law("test-law")

        assert len(cases) == 2
        assert cases[0]["id"] == "test-law-01-basic"
        assert cases[0]["profile"] == "LEGAL"
        assert cases[0]["origin"] == "auto"
        assert cases[1]["id"] == "test-law-02-second"
        assert cases[1]["origin"] == "manual"

    def test_load_cases_for_nonexistent_law_returns_empty(self, tmp_path):
        """Should return empty list for non-existent law."""
        from services.eval_cases import load_cases_for_law

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            cases = load_cases_for_law("nonexistent")

        assert cases == []

    def test_get_case_by_id(self, tmp_path):
        """Should return single case by ID."""
        from services.eval_cases import get_case_by_id

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-basic
  profile: LEGAL
  prompt: What is the test?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
""")

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            case = get_case_by_id("test-law", "test-law-01-basic")

        assert case is not None
        assert case["id"] == "test-law-01-basic"
        assert case["profile"] == "LEGAL"

    def test_get_case_by_id_not_found(self, tmp_path):
        """Should return None for non-existent case."""
        from services.eval_cases import get_case_by_id

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-basic
  profile: LEGAL
  prompt: What is the test?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
""")

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            case = get_case_by_id("test-law", "nonexistent-case")

        assert case is None

    def test_load_cases_with_underscore_filename_and_hyphen_law(self, tmp_path):
        """Should find ai_act file when law is ai-act (hyphen to underscore)."""
        from services.eval_cases import load_cases_for_law

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        # File uses underscore (like real ai_act file)
        golden_file = evals_dir / "golden_cases_ai_act.yaml"
        golden_file.write_text("""
- id: ai-act-01-test
  profile: LEGAL
  prompt: What are the AI Act requirements?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:5
    behavior: answer
""")

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            # Law parameter uses hyphen
            cases = load_cases_for_law("ai-act")

        assert len(cases) == 1
        assert cases[0]["id"] == "ai-act-01-test"


class TestCreateCase:
    """Tests for creating eval cases."""

    def test_create_case_sets_manual_origin(self, tmp_path):
        """Created cases should always have origin='manual'."""
        from services.eval_cases import create_case

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        # Create empty golden cases file
        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        case_data = {
            "profile": "LEGAL",
            "prompt": "New test question?",
            "test_types": ["retrieval"],
            "expected": {
                "must_include_any_of": ["article:5"],
                "behavior": "answer",
            },
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            created = create_case("test-law", case_data)

        assert created["origin"] == "manual"
        assert created["profile"] == "LEGAL"
        assert created["prompt"] == "New test question?"

    def test_create_case_generates_id(self, tmp_path):
        """Should auto-generate case ID from prompt if not provided."""
        from services.eval_cases import create_case

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        case_data = {
            "profile": "LEGAL",
            "prompt": "What are the requirements for data governance?",
            "test_types": ["retrieval"],
            "expected": {
                "must_include_any_of": ["article:10"],
                "behavior": "answer",
            },
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            created = create_case("test-law", case_data)

        # ID should be generated: {law}-{nn}-{slug}
        assert created["id"].startswith("test-law-")
        assert "requirements" in created["id"].lower() or "data" in created["id"].lower()

    def test_create_case_persists_to_file(self, tmp_path):
        """Created case should be saved to YAML file."""
        from services.eval_cases import create_case, load_cases_for_law

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        case_data = {
            "profile": "ENGINEERING",
            "prompt": "How to implement logging?",
            "test_types": ["retrieval", "faithfulness"],
            "expected": {
                "must_include_any_of": ["article:12"],
                "behavior": "answer",
            },
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            created = create_case("test-law", case_data)
            # Reload from file
            cases = load_cases_for_law("test-law")

        assert len(cases) == 1
        assert cases[0]["id"] == created["id"]

    def test_create_case_validates_required_fields(self, tmp_path):
        """Should raise ValueError for missing required fields."""
        from services.eval_cases import create_case, ValidationError

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        # Missing prompt
        case_data = {
            "profile": "LEGAL",
            "test_types": ["retrieval"],
            "expected": {"behavior": "answer"},
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            with pytest.raises(ValidationError) as exc_info:
                create_case("test-law", case_data)

        assert "prompt" in str(exc_info.value).lower()

    def test_create_case_validates_prompt_min_length(self, tmp_path):
        """Should reject prompts shorter than 10 characters."""
        from services.eval_cases import create_case, ValidationError

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        case_data = {
            "profile": "LEGAL",
            "prompt": "Short?",  # Too short
            "test_types": ["retrieval"],
            "expected": {"behavior": "answer"},
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            with pytest.raises(ValidationError) as exc_info:
                create_case("test-law", case_data)

        assert "10" in str(exc_info.value) or "length" in str(exc_info.value).lower()

    def test_create_case_validates_test_types(self, tmp_path):
        """Should require at least one test type."""
        from services.eval_cases import create_case, ValidationError

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        case_data = {
            "profile": "LEGAL",
            "prompt": "What are the requirements?",
            "test_types": [],  # Empty
            "expected": {"behavior": "answer"},
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            with pytest.raises(ValidationError) as exc_info:
                create_case("test-law", case_data)

        assert "test_type" in str(exc_info.value).lower()

    def test_create_case_with_explicit_id(self, tmp_path):
        """Should use provided ID if given."""
        from services.eval_cases import create_case

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        case_data = {
            "id": "test-law-99-custom-id",
            "profile": "LEGAL",
            "prompt": "Custom ID test question?",
            "test_types": ["retrieval"],
            "expected": {"behavior": "answer"},
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            created = create_case("test-law", case_data)

        assert created["id"] == "test-law-99-custom-id"


class TestUpdateCase:
    """Tests for updating eval cases."""

    def test_update_case_changes_origin_to_manual(self, tmp_path):
        """Updating any case should change origin to 'manual'."""
        from services.eval_cases import update_case, load_cases_for_law

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-auto
  profile: LEGAL
  prompt: Original question?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
""")

        update_data = {
            "prompt": "Updated question text?",
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            updated = update_case("test-law", "test-law-01-auto", update_data)
            # Verify in file
            cases = load_cases_for_law("test-law")

        assert updated["origin"] == "manual"
        assert updated["prompt"] == "Updated question text?"
        assert cases[0]["origin"] == "manual"

    def test_update_case_preserves_id(self, tmp_path):
        """Case ID should remain unchanged after update."""
        from services.eval_cases import update_case

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-original
  profile: LEGAL
  prompt: Original question?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
""")

        update_data = {
            "id": "test-law-99-hacked",  # Attempt to change ID
            "prompt": "New question?",
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            updated = update_case("test-law", "test-law-01-original", update_data)

        # ID should NOT change
        assert updated["id"] == "test-law-01-original"

    def test_update_case_not_found(self, tmp_path):
        """Should raise error for non-existent case."""
        from services.eval_cases import update_case, NotFoundError

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            with pytest.raises(NotFoundError):
                update_case("test-law", "nonexistent", {"prompt": "This is a valid prompt that is long enough"})

    def test_update_case_partial_update(self, tmp_path):
        """Should only update provided fields."""
        from services.eval_cases import update_case

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-basic
  profile: LEGAL
  prompt: Original question?
  test_types:
    - retrieval
    - faithfulness
  origin: auto
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
""")

        # Only update profile
        update_data = {
            "profile": "ENGINEERING",
        }

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            updated = update_case("test-law", "test-law-01-basic", update_data)

        assert updated["profile"] == "ENGINEERING"
        assert updated["prompt"] == "Original question?"  # Unchanged
        assert updated["test_types"] == ["retrieval", "faithfulness"]  # Unchanged


class TestDeleteCase:
    """Tests for deleting eval cases."""

    def test_delete_case_removes_from_file(self, tmp_path):
        """Deleted case should be removed from YAML file."""
        from services.eval_cases import delete_case, load_cases_for_law

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-keep
  profile: LEGAL
  prompt: Keep this question?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
- id: test-law-02-delete
  profile: LEGAL
  prompt: Delete this question?
  test_types:
    - retrieval
  origin: manual
  expected:
    must_include_any_of:
      - article:2
    behavior: answer
""")

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            delete_case("test-law", "test-law-02-delete")
            cases = load_cases_for_law("test-law")

        assert len(cases) == 1
        assert cases[0]["id"] == "test-law-01-keep"

    def test_delete_case_not_found(self, tmp_path):
        """Should raise error for non-existent case."""
        from services.eval_cases import delete_case, NotFoundError

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            with pytest.raises(NotFoundError):
                delete_case("test-law", "nonexistent")


class TestDuplicateCase:
    """Tests for duplicating eval cases."""

    def test_duplicate_case_generates_new_id(self, tmp_path):
        """Duplicated case should have a new unique ID."""
        from services.eval_cases import duplicate_case, load_cases_for_law

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("""
- id: test-law-01-original
  profile: LEGAL
  prompt: Original question to duplicate?
  test_types:
    - retrieval
  origin: auto
  expected:
    must_include_any_of:
      - article:1
    behavior: answer
""")

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            duplicated = duplicate_case("test-law", "test-law-01-original")
            cases = load_cases_for_law("test-law")

        assert duplicated["id"] != "test-law-01-original"
        assert duplicated["prompt"] == "Original question to duplicate?"
        assert duplicated["origin"] == "manual"  # Duplicates are manual
        assert len(cases) == 2

    def test_duplicate_case_not_found(self, tmp_path):
        """Should raise error for non-existent source case."""
        from services.eval_cases import duplicate_case, NotFoundError

        evals_dir = tmp_path / "data" / "evals"
        evals_dir.mkdir(parents=True)

        golden_file = evals_dir / "golden_cases_test-law.yaml"
        golden_file.write_text("[]")

        with patch("services.eval_cases._get_evals_dir", return_value=evals_dir):
            with pytest.raises(NotFoundError):
                duplicate_case("test-law", "nonexistent")


class TestGenerateCaseId:
    """Tests for case ID generation."""

    def test_generate_case_id_format(self):
        """Generated ID should follow {law}-{nn}-{slug} format."""
        from services.eval_cases import generate_case_id
        import re

        case_id = generate_case_id(
            law="ai-act",
            prompt="What are the requirements for high-risk AI systems?",
            existing_ids=set(),
        )

        assert case_id.startswith("ai-act-")
        # Format: ai-act-NN-slug where NN is zero-padded number
        # Pattern: {law}-{digits}-{slug}
        match = re.match(r"^ai-act-(\d+)-(.+)$", case_id)
        assert match is not None, f"Case ID '{case_id}' does not match expected format"
        assert match.group(1).isdigit()  # Number part
        assert len(match.group(2)) > 0  # Slug part exists

    def test_generate_case_id_avoids_collisions(self):
        """Generated ID should be unique among existing IDs."""
        from services.eval_cases import generate_case_id

        existing = {"ai-act-01-requirements-for-high", "ai-act-02-requirements-for-high"}

        case_id = generate_case_id(
            law="ai-act",
            prompt="What are the requirements for high-risk?",
            existing_ids=existing,
        )

        assert case_id not in existing
        assert case_id.startswith("ai-act-")

    def test_generate_case_id_slugifies_prompt(self):
        """Slug should be URL-safe version of prompt."""
        from services.eval_cases import generate_case_id

        case_id = generate_case_id(
            law="gdpr",
            prompt="What are the GDPR requirements for data processing?",
            existing_ids=set(),
        )

        # Should not contain special characters
        assert " " not in case_id
        assert "?" not in case_id
        # Should be lowercase
        assert case_id == case_id.lower()


class TestValidateAnchor:
    """Tests for anchor validation."""

    def test_valid_article_anchor(self):
        """Should accept article:N format."""
        from services.eval_cases import validate_anchor

        assert validate_anchor("article:5") is True
        assert validate_anchor("article:14") is True

    def test_valid_section_anchor(self):
        """Should accept section:N format."""
        from services.eval_cases import validate_anchor

        assert validate_anchor("section:3") is True

    def test_valid_annex_anchor(self):
        """Should accept annex:N and annex:iii formats."""
        from services.eval_cases import validate_anchor

        assert validate_anchor("annex:iii") is True
        assert validate_anchor("annex:viii") is True
        assert validate_anchor("annex:3") is True

    def test_valid_recital_anchor(self):
        """Should accept recital:N format."""
        from services.eval_cases import validate_anchor

        assert validate_anchor("recital:42") is True

    def test_invalid_anchor_format(self):
        """Should reject invalid formats."""
        from services.eval_cases import validate_anchor

        assert validate_anchor("invalid") is False
        assert validate_anchor("article5") is False
        assert validate_anchor("article:") is False
        assert validate_anchor(":5") is False
        assert validate_anchor("") is False
