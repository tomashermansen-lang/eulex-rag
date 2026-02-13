import pytest
from src.engine.rag import _extract_raw_anchors_from_chunk

def test_extract_raw_anchors_from_chunk():
    # Test 1: Article only
    meta1 = {"article": "5"}
    assert _extract_raw_anchors_from_chunk(meta1) == ["article:5"]

    # Test 2: Article and Recital
    meta2 = {"article": "5", "recital": "10"}
    # Order depends on implementation: article, recital, annex
    assert _extract_raw_anchors_from_chunk(meta2) == ["article:5", "recital:10"]

    # Test 3: From location_id
    meta3 = {"location_id": "ai-act/article:6/paragraph:1"}
    assert _extract_raw_anchors_from_chunk(meta3) == ["article:6"]

    # Test 4: Mixed (meta overrides location_id if present, or merges? Implementation uses meta.get OR derived.get)
    # If meta has article, it uses it.
    meta4 = {"article": "7", "location_id": "ai-act/article:6"}
    assert _extract_raw_anchors_from_chunk(meta4) == ["article:7"]

    # Test 5: Normalization (spaces, case)
    meta5 = {"article": " 5 "}
    assert _extract_raw_anchors_from_chunk(meta5) == ["article:5"]
    
    meta6 = {"annex": "III"}
    assert _extract_raw_anchors_from_chunk(meta6) == ["annex:iii"]

def test_extract_raw_anchors_empty():
    assert _extract_raw_anchors_from_chunk({}) == []
    assert _extract_raw_anchors_from_chunk(None) == []
