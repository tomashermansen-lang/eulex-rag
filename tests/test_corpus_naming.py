"""Tests for corpus naming conventions.

TDD: These tests define expected behavior before implementation.
"""

import pytest
from pathlib import Path


class TestNamingConventionsConfig:
    """Test that naming conventions config is loaded correctly."""

    def test_config_file_exists(self):
        """naming_conventions.yaml exists in config directory."""
        config_path = Path(__file__).parent.parent / "config" / "naming_conventions.yaml"
        assert config_path.exists(), "config/naming_conventions.yaml must exist"

    def test_config_has_required_keys(self):
        """Config has all required keys."""
        import yaml

        config_path = Path(__file__).parent.parent / "config" / "naming_conventions.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "celex_type_mapping" in config
        assert "corpus_id_format" in config
        assert "L" in config["celex_type_mapping"]
        assert "R" in config["celex_type_mapping"]


class TestCelexParsing:
    """Test CELEX number parsing utilities."""

    def test_parse_celex_directive(self):
        """Parse directive CELEX number."""
        from src.common.corpus_naming import parse_celex

        result = parse_celex("32022L2555")
        assert result["type_char"] == "L"
        assert result["year"] == 2022
        assert result["number"] == 2555
        assert result["type_code"] == "dir"

    def test_parse_celex_regulation(self):
        """Parse regulation CELEX number."""
        from src.common.corpus_naming import parse_celex

        result = parse_celex("32016R0679")
        assert result["type_char"] == "R"
        assert result["year"] == 2016
        assert result["number"] == 679
        assert result["type_code"] == "reg"

    def test_parse_celex_with_leading_zeros(self):
        """Parse CELEX with leading zeros in number."""
        from src.common.corpus_naming import parse_celex

        result = parse_celex("32024R1689")
        assert result["year"] == 2024
        assert result["number"] == 1689

    def test_parse_invalid_celex_raises(self):
        """Invalid CELEX raises ValueError."""
        from src.common.corpus_naming import parse_celex

        with pytest.raises(ValueError):
            parse_celex("invalid")

        with pytest.raises(ValueError):
            parse_celex("12345")


class TestCorpusIdGeneration:
    """Test corpus_id generation from components."""

    def test_generate_corpus_id_directive(self):
        """Generate corpus_id for directive."""
        from src.common.corpus_naming import generate_corpus_id

        result = generate_corpus_id(
            known_name="NIS2",
            celex_number="32022L2555"
        )
        assert result == "nis2-dir-2022-2555"

    def test_generate_corpus_id_regulation(self):
        """Generate corpus_id for regulation."""
        from src.common.corpus_naming import generate_corpus_id

        result = generate_corpus_id(
            known_name="GDPR",
            celex_number="32016R0679"
        )
        assert result == "gdpr-reg-2016-679"

    def test_generate_corpus_id_implementing_regulation(self):
        """Generate corpus_id for implementing regulation."""
        from src.common.corpus_naming import generate_corpus_id

        result = generate_corpus_id(
            known_name="NIS2-CIR",
            celex_number="32024R2690",
            is_implementing=True
        )
        assert result == "nis2-cir-cir-2024-2690"

    def test_generate_corpus_id_lowercase(self):
        """corpus_id is always lowercase."""
        from src.common.corpus_naming import generate_corpus_id

        result = generate_corpus_id(
            known_name="AI-ACT",
            celex_number="32024R1689"
        )
        assert result == result.lower()
        assert result == "ai-act-reg-2024-1689"

    def test_generate_corpus_id_no_special_chars(self):
        """corpus_id contains only allowed characters."""
        from src.common.corpus_naming import generate_corpus_id

        result = generate_corpus_id(
            known_name="DATA ACT",  # Space should be converted
            celex_number="32023R2854"
        )
        # Only lowercase letters, numbers, and hyphens allowed
        assert all(c.islower() or c.isdigit() or c == "-" for c in result)


class TestFullnameInCorpora:
    """Test fullname field in corpora inventory."""

    def test_upsert_corpus_with_fullname(self):
        """upsert_corpus_inventory accepts fullname in extra dict."""
        from src.common.corpora_inventory import upsert_corpus_inventory

        data = {"version": 1, "corpora": {}}
        result = upsert_corpus_inventory(
            data,
            corpus_id="test-corpus",
            display_name="Test Corpus",
            extra={
                "fullname": "Full Official Legal Title Here",
                "celex_number": "32024R1234"
            }
        )

        assert "test-corpus" in result["corpora"]
        assert result["corpora"]["test-corpus"]["fullname"] == "Full Official Legal Title Here"

    def test_fullname_preserved_on_update(self):
        """fullname is preserved when updating other fields."""
        from src.common.corpora_inventory import upsert_corpus_inventory

        data = {
            "version": 1,
            "corpora": {
                "test-corpus": {
                    "display_name": "Test",
                    "fullname": "Original Full Name",
                    "enabled": True
                }
            }
        }

        result = upsert_corpus_inventory(
            data,
            corpus_id="test-corpus",
            display_name="Updated Test",
            extra={"fullname": "Original Full Name"}  # Preserve fullname
        )

        assert result["corpora"]["test-corpus"]["fullname"] == "Original Full Name"


class TestSuggestNamesResponse:
    """Test that SuggestNamesResponse includes fullname."""

    def test_response_has_fullname_field(self):
        """SuggestNamesResponse schema includes fullname."""
        from ui_react.backend.schemas import SuggestNamesResponse

        # Check that fullname is a valid field
        fields = SuggestNamesResponse.model_fields
        assert "fullname" in fields, "SuggestNamesResponse must have fullname field"

    def test_response_can_be_created_with_fullname(self):
        """SuggestNamesResponse can be instantiated with fullname."""
        from ui_react.backend.schemas import SuggestNamesResponse

        response = SuggestNamesResponse(
            corpus_id="test-reg-2024-123",
            display_name="Test Regulation",
            fullname="Full Official Title of Test Regulation (EU) 2024/123"
        )

        assert response.fullname == "Full Official Title of Test Regulation (EU) 2024/123"


class TestCorpusInfoResponse:
    """Test that CorpusInfo includes fullname."""

    def test_corpus_info_has_fullname_field(self):
        """CorpusInfo schema includes fullname."""
        from ui_react.backend.schemas import CorpusInfo

        fields = CorpusInfo.model_fields
        assert "fullname" in fields, "CorpusInfo must have fullname field"
