"""Tests for citation_expansion.py - automatic article discovery from citation graph."""

from __future__ import annotations

import pytest

from src.engine.citation_expansion import (
    extract_mentioned_articles,
    is_citation_expansion_enabled,
    get_max_expansion,
    get_min_weight,
    get_bump_bonus,
    should_include_scope_articles,
)


class TestExtractMentionedArticles:
    """Test article/annex extraction from question text."""

    def test_extracts_danish_artikel(self) -> None:
        result = extract_mentioned_articles("Hvad siger artikel 6 om dette?")
        assert "6" in result

    def test_extracts_english_article(self) -> None:
        result = extract_mentioned_articles("What does article 10 say?")
        assert "10" in result

    def test_extracts_abbreviated_art(self) -> None:
        result = extract_mentioned_articles("Se art. 5 for mere info")
        assert "5" in result

    def test_extracts_jf_reference(self) -> None:
        result = extract_mentioned_articles("jf. artikel 3")
        assert "3" in result

    def test_extracts_danish_bilag(self) -> None:
        result = extract_mentioned_articles("Bilag III beskriver dette")
        assert "ANNEX:III" in result

    def test_extracts_english_annex(self) -> None:
        result = extract_mentioned_articles("See Annex III for details")
        assert "ANNEX:III" in result

    def test_extracts_lowercase_annex(self) -> None:
        result = extract_mentioned_articles("refer to annex iv")
        assert "ANNEX:IV" in result

    def test_extracts_multiple_articles(self) -> None:
        question = "Hvordan hænger artikel 5 og artikel 12 sammen med bilag III?"
        result = extract_mentioned_articles(question)
        assert "5" in result
        assert "12" in result
        assert "ANNEX:III" in result

    def test_handles_no_references(self) -> None:
        result = extract_mentioned_articles("Hvad er AI-forordningen?")
        assert result == []

    def test_handles_empty_string(self) -> None:
        result = extract_mentioned_articles("")
        assert result == []

    def test_handles_article_with_stk(self) -> None:
        result = extract_mentioned_articles("artikel 6, stk. 2")
        assert "6" in result

    def test_includes_duplicates_by_design(self) -> None:
        # Note: current implementation doesn't deduplicate
        question = "Artikel 5 siger at artikel 5 skal følges"
        result = extract_mentioned_articles(question)
        assert "5" in result


class TestShouldIncludeScopeArticles:
    """Test detection of scope/applicability questions."""

    def test_danish_falder_ind_under(self) -> None:
        assert should_include_scope_articles("Falder ind under AI-forordningen?")

    def test_danish_falder_under_exact(self) -> None:
        assert should_include_scope_articles("Falder under disse regler?")

    def test_danish_omfattet_af(self) -> None:
        assert should_include_scope_articles("Er vi omfattet af reglerne?")

    def test_danish_anvendelsesomraade(self) -> None:
        assert should_include_scope_articles("Hvad er anvendelsesområdet?")

    def test_english_scope(self) -> None:
        assert should_include_scope_articles("What is the scope of the AI Act?")

    def test_english_applies_to(self) -> None:
        assert should_include_scope_articles("Does the regulation apply? It applies to us.")

    def test_normal_question_returns_false(self) -> None:
        assert not should_include_scope_articles("Hvad er kravene til logging?")

    def test_empty_question_returns_false(self) -> None:
        assert not should_include_scope_articles("")


class TestConfigLoading:
    """Test configuration loading from settings.yaml."""

    def test_is_citation_expansion_enabled_returns_bool(self) -> None:
        result = is_citation_expansion_enabled()
        assert isinstance(result, bool)

    def test_get_max_expansion_returns_int(self) -> None:
        result = get_max_expansion()
        assert isinstance(result, int)
        assert result > 0

    def test_get_min_weight_returns_float(self) -> None:
        result = get_min_weight()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_get_bump_bonus_returns_float(self) -> None:
        result = get_bump_bonus()
        assert isinstance(result, float)
        assert result >= 0.0
