import pytest
from src.engine.rag import RAGEngine
from src.engine import helpers

def test_strip_trailing_references_section():
    # Case 1: No references section
    text = "This is a normal answer."
    assert helpers._strip_trailing_references_section(text) == text

    # Case 2: With references section
    text_with_ref = "This is an answer.\n\nReferencer:\n[1] Source 1"
    expected = "This is an answer."
    assert helpers._strip_trailing_references_section(text_with_ref) == expected

    # Case 3: Empty string
    assert helpers._strip_trailing_references_section("") == ""

    # Case 4: None (defensive)
    assert helpers._strip_trailing_references_section(None) == ""

def test_extract_anchor_mentions_from_answer():
    # Case 1: Articles with and without paragraph
    text = "See Artikel 5 and Article 6, stk. 2."
    mentions = helpers._extract_anchor_mentions_from_answer(text)
    articles = mentions["articles"]
    # Note: The regex might capture "5" and None, "6" and "2"
    # Let's check if "5" is present (case insensitive normalization in method?)
    # The method upper-cases the article number/letter.
    
    # "Artikel 5" -> ("5", None)
    # "Article 6, stk. 2" -> ("6", "2")
    
    assert ("5", None) in articles
    assert ("6", "2") in articles

    # Case 2: Recitals
    text_rec = "Consider betragtning 10 and recital 42."
    mentions = helpers._extract_anchor_mentions_from_answer(text_rec)
    recitals = mentions["recitals"]
    assert "10" in recitals
    assert "42" in recitals

    # Case 3: Annexes
    text_annex = "Refer to Bilag II and Annex IV."
    mentions = helpers._extract_anchor_mentions_from_answer(text_annex)
    annexes = mentions["annexes"]
    assert "II" in annexes
    assert "IV" in annexes

    # Case 4: Mixed
    text_mixed = "Artikel 10 requires compliance with Bilag III."
    mentions = helpers._extract_anchor_mentions_from_answer(text_mixed)
    assert ("10", None) in mentions["articles"]
    assert "III" in mentions["annexes"]
