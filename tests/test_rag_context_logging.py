import pytest
from src.engine.rag import _extract_raw_anchors_from_chunk

def test_extract_raw_anchors_simple():
    meta = {"article": "5", "recital": "10"}
    anchors = _extract_raw_anchors_from_chunk(meta)
    assert "article:5" in anchors
    assert "recital:10" in anchors
    assert len(anchors) == 2

def test_extract_raw_anchors_derived():
    # Simulate location_id derivation
    meta = {"location_id": "ai-act/title-1/article:5"}
    anchors = _extract_raw_anchors_from_chunk(meta)
    assert "article:5" in anchors
    assert len(anchors) == 1

def test_extract_raw_anchors_mixed():
    meta = {
        "location_id": "ai-act/annex:III",
        "article": "6" # Explicit override or additional info
    }
    anchors = _extract_raw_anchors_from_chunk(meta)
    assert "annex:iii" in anchors
    assert "article:6" in anchors
    assert len(anchors) == 2

def test_extract_raw_anchors_normalization():
    meta = {"article": "  5  ", "recital": " 10 "}
    anchors = _extract_raw_anchors_from_chunk(meta)
    assert "article:5" in anchors
    assert "recital:10" in anchors

def test_extract_raw_anchors_empty():
    meta = {}
    anchors = _extract_raw_anchors_from_chunk(meta)
    assert anchors == []
