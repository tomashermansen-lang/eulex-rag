import json
import re

import pytest

from src.common.metadata_schema import (
    CASE_LAW_REQUIRED_FIELDS,
    CaseLawChunkMetadata,
    CommonMetadata,
    build_chunk_id,
    compute_text_hash,
    make_heading_path,
    make_location_id,
    normalize_case_number_for_id,
    validate_articles_interpreted,
    validate_metadata_primitives,
    validate_required_fields,
)


def test_make_location_id_deterministic_and_normalized():
    ref_state = {
        "chapter": "IV",
        "section": "2",
        "article": "10",
        "paragraph": " 3 ",
        "litra": "B",
        "annex": None,
    }

    loc1 = make_location_id(reference_state=ref_state)
    loc2 = make_location_id(reference_state=ref_state)
    assert loc1 == loc2
    assert loc1.startswith("loc:v1/")
    assert "chapter:iv" in loc1
    assert "section:2" in loc1
    assert "article:10" in loc1
    assert "paragraph:3" in loc1
    assert "litra:b" in loc1


def test_make_heading_path_json_is_stable():
    ref_state = {"chapter": "I", "article": "1", "section": None}
    heading_json, display = make_heading_path(reference_state=ref_state)
    segments = json.loads(heading_json)
    assert segments == ["chapter:i", "article:1"]
    assert "Chapter" in display


def test_build_chunk_id_uses_text_hash():
    text_hash = compute_text_hash("Hello   world")
    cid = build_chunk_id(
        doc_id="GDPR",
        location_id="loc:v1/article:1",
        chunk_index=0,
        text_hash=text_hash,
    )
    assert cid.startswith("chunk:v1/")
    assert cid.endswith(text_hash[:12])


def test_validate_metadata_primitives_rejects_dict():
    with pytest.raises(ValueError):
        validate_metadata_primitives({"ok": "x", "bad": {"nested": True}})

    validate_metadata_primitives({"ok": "x", "n": 1, "f": 1.0, "b": True, "none": None})


# ─────────────────────────────────────────────────────────────────────────────
# C1: CaseLawChunkMetadata, CASE_LAW_REQUIRED_FIELDS, normalize, validate
# ─────────────────────────────────────────────────────────────────────────────


class TestCaseLawChunkMetadata:
    def test_importable_and_extends_common(self):
        assert issubclass(CaseLawChunkMetadata, dict)
        common_keys = set(CommonMetadata.__annotations__)
        case_keys = set(CaseLawChunkMetadata.__annotations__)
        assert common_keys.issubset(case_keys)

    def test_all_required_fields_present(self):
        expected = {
            "ecli",
            "case_number",
            "court",
            "decision_date",
            "section_type",
            "paragraph_range",
            "articles_interpreted",
            "case_name",
            "heading_path",
            "heading_path_display",
            "source_type",
            "doc_type",
            "chunk_index",
            "chunk_id",
            "text_hash",
        }
        assert expected.issubset(set(CaseLawChunkMetadata.__annotations__))


class TestCaseLawRequiredFields:
    def test_is_tuple_with_expected_fields(self):
        assert isinstance(CASE_LAW_REQUIRED_FIELDS, tuple)
        assert set(CASE_LAW_REQUIRED_FIELDS) == {
            "ecli",
            "case_number",
            "court",
            "decision_date",
            "source_type",
            "articles_interpreted",
        }

    def test_validate_raises_on_missing_ecli(self):
        meta = {
            "case_number": "C-311/18",
            "court": "CJEU",
            "decision_date": "2020-07-16",
            "source_type": "cjeu_case_law",
            "articles_interpreted": "[]",
        }
        with pytest.raises(ValueError, match="ecli"):
            validate_required_fields(meta, CASE_LAW_REQUIRED_FIELDS)

    def test_validate_raises_on_empty_court(self):
        meta = {
            "ecli": "ECLI:EU:C:2020:790",
            "case_number": "C-311/18",
            "court": "",
            "decision_date": "2020-07-16",
            "source_type": "cjeu_case_law",
            "articles_interpreted": "[]",
        }
        with pytest.raises(ValueError, match="court"):
            validate_required_fields(meta, CASE_LAW_REQUIRED_FIELDS)

    def test_validate_passes_all_present(self):
        meta = {
            "ecli": "ECLI:EU:C:2020:790",
            "case_number": "C-311/18",
            "court": "CJEU",
            "decision_date": "2020-07-16",
            "source_type": "cjeu_case_law",
            "articles_interpreted": '["gdpr/article:46"]',
        }
        validate_required_fields(meta, CASE_LAW_REQUIRED_FIELDS)


class TestNormalizeCaseNumberForId:
    def test_standard_case(self):
        assert normalize_case_number_for_id("C-311/18") == "c-311-18"

    def test_already_lowercase(self):
        assert normalize_case_number_for_id("c-311/18") == "c-311-18"

    def test_joined_cases(self):
        assert (
            normalize_case_number_for_id("C-293/12 and C-594/12")
            == "c-293-12-and-c-594-12"
        )

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            normalize_case_number_for_id("")

    def test_result_slug_safe(self):
        assert re.fullmatch(r"[a-z0-9\-]+", normalize_case_number_for_id("C-311/18"))


class TestValidateArticlesInterpreted:
    def test_empty_list_valid(self):
        validate_articles_interpreted([])

    def test_valid_entries(self):
        validate_articles_interpreted(["gdpr/article:46", "ai_act/article:3"])

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="invalid_format"):
            validate_articles_interpreted(["invalid_format"])

    def test_missing_article_number_raises(self):
        with pytest.raises(ValueError, match="gdpr/article:"):
            validate_articles_interpreted(["gdpr/article:"])

    def test_uppercase_corpus_raises(self):
        with pytest.raises(ValueError, match="GDPR/article:6"):
            validate_articles_interpreted(["GDPR/article:6"])

    def test_over_100_chars_raises(self):
        long_entry = "a" * 91 + "/article:1"
        with pytest.raises(ValueError):
            validate_articles_interpreted([long_entry])


class TestValidateCaseName:
    """W2: case_name validation — non-empty, max 200 chars, UTF-8."""

    def test_valid_case_name(self):
        from src.common.metadata_schema import validate_case_name

        validate_case_name("Schrems II")  # Should not raise

    def test_empty_raises(self):
        from src.common.metadata_schema import validate_case_name

        with pytest.raises(ValueError, match="case_name must be non-empty"):
            validate_case_name("")

    def test_whitespace_only_raises(self):
        from src.common.metadata_schema import validate_case_name

        with pytest.raises(ValueError, match="case_name must be non-empty"):
            validate_case_name("   ")

    def test_over_200_chars_raises(self):
        from src.common.metadata_schema import validate_case_name

        with pytest.raises(ValueError, match="case_name exceeds 200 characters"):
            validate_case_name("A" * 201)

    def test_exactly_200_chars_valid(self):
        from src.common.metadata_schema import validate_case_name

        validate_case_name("A" * 200)  # Should not raise

    def test_utf8_valid(self):
        from src.common.metadata_schema import validate_case_name

        validate_case_name("Schrems II — Datenschutz")  # Should not raise


class TestBackwardCompatibility:
    def test_existing_validate_required_fields_unchanged(self):
        meta = {"corpus_id": "gdpr", "doc_id": "GDPR", "chunk_id": "chunk:v1/x/y/0/abc"}
        validate_required_fields(meta, ("corpus_id", "doc_id", "chunk_id"))
        with pytest.raises(ValueError, match="corpus_id"):
            validate_required_fields({"doc_id": "x"}, ("corpus_id", "doc_id"))
