import json

import pytest

from src.common.metadata_schema import (
    build_chunk_id,
    compute_text_hash,
    make_heading_path,
    make_location_id,
    validate_metadata_primitives,
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
    cid = build_chunk_id(doc_id="GDPR", location_id="loc:v1/article:1", chunk_index=0, text_hash=text_hash)
    assert cid.startswith("chunk:v1/")
    assert cid.endswith(text_hash[:12])


def test_validate_metadata_primitives_rejects_dict():
    with pytest.raises(ValueError):
        validate_metadata_primitives({"ok": "x", "bad": {"nested": True}})

    validate_metadata_primitives({"ok": "x", "n": 1, "f": 1.0, "b": True, "none": None})
