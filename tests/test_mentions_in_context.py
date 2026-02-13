"""Test that mentions (cross-references) are included in LLM context."""
import pytest


def test_format_mentions_for_context_with_articles():
    """Verify article mentions are formatted correctly."""
    from src.engine.citations import format_mentions_for_context
    
    mentions_json = '{"article": ["16", "6"]}'
    
    result = format_mentions_for_context(mentions_json)
    
    assert result is not None
    assert "Artikel 16" in result
    assert "Artikel 6" in result


def test_format_mentions_for_context_mixed():
    """Verify mixed mentions (article, recital, annex) are formatted."""
    from src.engine.citations import format_mentions_for_context
    
    mentions_json = '{"article": ["53"], "recital": ["107"], "annex": ["III"]}'
    
    result = format_mentions_for_context(mentions_json)
    
    assert result is not None
    assert "Artikel 53" in result
    assert "Betragtning 107" in result
    assert "Bilag III" in result


def test_format_mentions_for_context_none():
    """Verify None input returns None."""
    from src.engine.citations import format_mentions_for_context
    
    result = format_mentions_for_context(None)
    assert result is None


def test_format_mentions_for_context_empty_string():
    """Verify empty string returns None."""
    from src.engine.citations import format_mentions_for_context
    
    result = format_mentions_for_context("")
    assert result is None


def test_format_mentions_for_context_empty_dict():
    """Verify empty dict JSON returns None."""
    from src.engine.citations import format_mentions_for_context
    
    result = format_mentions_for_context("{}")
    assert result is None


def test_format_mentions_for_context_empty_lists():
    """Verify dict with empty lists returns None."""
    from src.engine.citations import format_mentions_for_context
    
    result = format_mentions_for_context('{"article": [], "recital": []}')
    assert result is None


def test_format_mentions_for_context_invalid_json():
    """Verify invalid JSON is handled gracefully."""
    from src.engine.citations import format_mentions_for_context
    
    result = format_mentions_for_context("not valid json")
    assert result is None


def test_format_mentions_for_context_dict_input():
    """Verify dict input (not just string) is handled."""
    from src.engine.citations import format_mentions_for_context
    
    mentions_dict = {"article": ["16"]}
    
    result = format_mentions_for_context(mentions_dict)
    
    assert result is not None
    assert "Artikel 16" in result


def test_format_metadata_includes_mentions():
    """Verify _format_metadata includes mentions when present."""
    from src.engine.citations import _format_metadata
    
    meta = {
        "source": "AI ACT",
        "article": "53",
        "mentions": '{"article": ["16"]}'
    }
    
    result = _format_metadata(meta)
    
    # Should include the cross-reference indicator
    assert "16" in result
    assert "→" in result or "Ref" in result or "Artikel 16" in result
