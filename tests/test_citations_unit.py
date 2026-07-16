"""Unit tests for src/engine/citations.py - Citation processing logic."""

import pytest
from src.engine.citations import (
    extract_bracket_citations,
    count_valid_citations,
    _format_metadata,
    _is_citable_metadata,
    format_mentions_for_context,
    extract_precise_ref_from_text,
    extract_precise_token_from_meta,
    best_effort_source_label,
    select_references_used_in_answer,
    _cosine_similarity,
    _find_better_match,
    verify_citation_similarity,
)


class TestExtractBracketCitations:
    """Tests for extract_bracket_citations function."""

    def test_extracts_all_citations(self):
        """Extracts all bracket citations from text."""
        text = "See [1] and also [2], plus [3]."
        result = extract_bracket_citations(text)
        assert result == {1, 2, 3}

    def test_empty_text_returns_empty_set(self):
        """Empty text returns empty set."""
        assert extract_bracket_citations("") == set()
        assert extract_bracket_citations(None) == set()

    def test_no_citations_returns_empty_set(self):
        """Text without citations returns empty set."""
        text = "This has no citations at all."
        assert extract_bracket_citations(text) == set()

    def test_handles_repeated_citations(self):
        """Repeated citations are deduplicated."""
        text = "See [1] and [1] again."
        result = extract_bracket_citations(text)
        assert result == {1}

    def test_handles_three_digit_citations(self):
        """Handles citations up to 3 digits."""
        text = "See [1], [12], [123]."
        result = extract_bracket_citations(text)
        assert result == {1, 12, 123}


class TestCountValidCitations:
    """Tests for count_valid_citations function."""

    def test_counts_valid_only(self):
        """Only counts citations that are in allowed set."""
        text = "See [1], [2], [3]."
        allowed = {1, 2}
        assert count_valid_citations(text, allowed) == 2

    def test_empty_allowed_returns_zero(self):
        """Empty allowed set returns zero."""
        text = "See [1], [2], [3]."
        assert count_valid_citations(text, set()) == 0

    def test_empty_text_returns_zero(self):
        """Empty text returns zero."""
        assert count_valid_citations("", {1, 2, 3}) == 0


class TestFormatMetadata:
    """Tests for _format_metadata function."""

    def test_empty_metadata_returns_ukendt(self):
        """Empty metadata returns 'Ukendt kilde'."""
        assert _format_metadata({}) == "Ukendt kilde"
        assert _format_metadata(None) == "Ukendt kilde"

    def test_formats_article(self):
        """Formats article metadata."""
        meta = {"article": "6"}
        result = _format_metadata(meta)
        assert "Artikel 6" in result

    def test_formats_article_with_paragraph(self):
        """Formats article with paragraph."""
        meta = {"article": "6", "paragraph": "1"}
        result = _format_metadata(meta)
        assert "Artikel 6" in result
        assert "stk. 1" in result

    def test_formats_annex(self):
        """Formats annex metadata."""
        meta = {"annex": "III"}
        result = _format_metadata(meta)
        assert "Bilag III" in result

    def test_formats_recital(self):
        """Formats recital metadata."""
        meta = {"recital": "42"}
        result = _format_metadata(meta)
        assert "Betragtning 42" in result

    def test_formats_chapter(self):
        """Formats chapter metadata."""
        meta = {"chapter": "IV"}
        result = _format_metadata(meta)
        assert "Kapitel IV" in result

    def test_formats_source(self):
        """Includes source in output."""
        meta = {"source": "AI Act", "article": "6"}
        result = _format_metadata(meta)
        assert "AI Act" in result


class TestIsCitableMetadata:
    """Tests for _is_citable_metadata function."""

    def test_article_is_citable(self):
        """Metadata with article is citable."""
        assert _is_citable_metadata({"article": "6"}) is True

    def test_annex_is_citable(self):
        """Metadata with annex is citable."""
        assert _is_citable_metadata({"annex": "III"}) is True

    def test_chapter_is_citable(self):
        """Metadata with chapter is citable."""
        assert _is_citable_metadata({"chapter": "IV"}) is True

    def test_recital_is_citable(self):
        """Metadata with recital is citable."""
        assert _is_citable_metadata({"recital": "42"}) is True

    def test_empty_is_not_citable(self):
        """Empty metadata is not citable."""
        assert _is_citable_metadata({}) is False
        assert _is_citable_metadata(None) is False

    def test_location_id_with_article_is_citable(self):
        """location_id containing article: is citable."""
        meta = {"location_id": "loc:v1/corpus:ai-act/article:6"}
        assert _is_citable_metadata(meta) is True

    def test_location_id_without_anchor_is_not_citable(self):
        """location_id without anchor is not citable."""
        meta = {"location_id": "loc:v1/corpus:ai-act/chunk:1"}
        assert _is_citable_metadata(meta) is False


class TestFormatMentionsForContext:
    """Tests for format_mentions_for_context function."""

    def test_formats_article_mentions(self):
        """Formats article mentions."""
        mentions = {"article": ["6", "7"]}
        result = format_mentions_for_context(mentions)
        assert "Artikel 6" in result
        assert "Artikel 7" in result

    def test_formats_recital_mentions(self):
        """Formats recital mentions."""
        mentions = {"recital": ["42"]}
        result = format_mentions_for_context(mentions)
        assert "Betragtning 42" in result

    def test_formats_annex_mentions(self):
        """Formats annex mentions."""
        mentions = {"annex": ["III"]}
        result = format_mentions_for_context(mentions)
        assert "Bilag III" in result

    def test_empty_mentions_returns_none(self):
        """Empty mentions returns None."""
        assert format_mentions_for_context({}) is None
        assert format_mentions_for_context(None) is None

    def test_handles_json_string(self):
        """Handles JSON string input."""
        mentions_json = '{"article": ["6"]}'
        result = format_mentions_for_context(mentions_json)
        assert "Artikel 6" in result


class TestExtractPreciseRefFromText:
    """Tests for extract_precise_ref_from_text function."""

    def test_extracts_article(self):
        """Extracts article reference from text."""
        text = "According to Article 6, systems must..."
        result = extract_precise_ref_from_text(text)
        assert result is not None
        assert "6" in result

    def test_extracts_artikel(self):
        """Extracts Danish artikel reference."""
        text = "I henhold til Artikel 6, skal systemer..."
        result = extract_precise_ref_from_text(text)
        assert result is not None

    def test_extracts_chapter(self):
        """Extracts chapter reference."""
        text = "Chapter III describes..."
        result = extract_precise_ref_from_text(text)
        assert result is not None

    def test_extracts_annex(self):
        """Extracts annex reference."""
        text = "See Annex III for details..."
        result = extract_precise_ref_from_text(text)
        assert result is not None

    def test_returns_none_for_no_ref(self):
        """Returns None when no reference found."""
        text = "This text has no legal references."
        assert extract_precise_ref_from_text(text) is None

    def test_empty_text_returns_none(self):
        """Empty text returns None."""
        assert extract_precise_ref_from_text("") is None
        assert extract_precise_ref_from_text(None) is None


class TestExtractPreciseTokenFromMeta:
    """Tests for extract_precise_token_from_meta function."""

    def test_extracts_article_token(self):
        """Extracts token with article."""
        meta = {"source": "AI Act", "article": "6"}
        result = extract_precise_token_from_meta(meta)
        assert "Artikel 6" in result

    def test_extracts_article_with_paragraph(self):
        """Extracts token with article and paragraph."""
        meta = {"article": "6", "paragraph": "1"}
        result = extract_precise_token_from_meta(meta)
        assert "Artikel 6(1)" in result

    def test_returns_none_for_imprecise(self):
        """Returns None for metadata without precise ref."""
        meta = {"source": "AI Act"}
        assert extract_precise_token_from_meta(meta) is None

    def test_returns_none_for_empty(self):
        """Returns None for empty metadata."""
        assert extract_precise_token_from_meta({}) is None
        assert extract_precise_token_from_meta(None) is None


class TestBestEffortSourceLabel:
    """Tests for best_effort_source_label function."""

    def test_returns_source_when_present(self):
        """Returns source when present in metadata."""
        meta = {"source": "AI Act"}
        assert best_effort_source_label(meta) == "AI Act"

    def test_returns_fallback_when_missing(self):
        """Returns fallback when source missing."""
        meta = {"article": "6"}
        result = best_effort_source_label(meta, fallback="Default")
        assert result == "Default"

    def test_returns_unknown_when_no_fallback(self):
        """Returns 'Unknown source' when no fallback provided."""
        meta = {}
        result = best_effort_source_label(meta)
        assert result == "Unknown source"


class TestSelectReferencesUsedInAnswer:
    """Tests for select_references_used_in_answer function."""

    def test_selects_by_bracket_citation(self):
        """Selects references by bracket citation."""
        answer = "See [1] for details."
        refs = [
            {"idx": 1, "chunk_id": "c1", "article": "6"},
            {"idx": 2, "chunk_id": "c2", "article": "7"},
        ]
        result = select_references_used_in_answer(
            answer_text=answer,
            references_structured=refs,
        )
        assert "c1" in result
        assert "c2" not in result

    def test_selects_by_article_mention(self):
        """Selects references by article mention in answer."""
        answer = "Article 6 requires..."
        refs = [
            {"idx": 1, "chunk_id": "c1", "article": "6"},
            {"idx": 2, "chunk_id": "c2", "article": "7"},
        ]
        result = select_references_used_in_answer(
            answer_text=answer,
            references_structured=refs,
        )
        assert "c1" in result

    def test_handles_empty_answer(self):
        """Handles empty answer text."""
        result = select_references_used_in_answer(
            answer_text="",
            references_structured=[{"idx": 1, "chunk_id": "c1"}],
        )
        assert result == []

    def test_deduplicates_results(self):
        """Results are deduplicated."""
        answer = "See [1] and Article 6 (same thing)."
        refs = [{"idx": 1, "chunk_id": "c1", "article": "6"}]
        result = select_references_used_in_answer(
            answer_text=answer,
            references_structured=refs,
        )
        assert result == ["c1"]  # Not duplicated


# ──────────────────────────────────────────────────────────────────────
# Safety-net tests for citation verification functions (Phase 1)
# ──────────────────────────────────────────────────────────────────────


class TestCosineSimilarity:
    """Tests for _cosine_similarity - vector math used in citation verification."""

    def test_identical_vectors_return_one(self):
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 0.0, 0.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors have similarity 0.0."""
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_negative_one(self):
        """Opposite vectors have similarity -1.0."""
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_empty_vectors_return_zero(self):
        """Empty vectors return 0.0."""
        assert _cosine_similarity([], []) == 0.0

    def test_zero_vector_returns_zero(self):
        """Zero-magnitude vector returns 0.0 (no division by zero)."""
        assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_mismatched_lengths_return_zero(self):
        """Vectors of different length return 0.0."""
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_single_element_vectors(self):
        """Single-element vectors work correctly."""
        assert _cosine_similarity([3.0], [3.0]) == pytest.approx(1.0)
        assert _cosine_similarity([3.0], [-3.0]) == pytest.approx(-1.0)


class TestFindBetterMatch:
    """Tests for _find_better_match - finds best-matching reference for a claim context."""

    @staticmethod
    def _constant_embedding(text: str) -> list[float]:
        """Deterministic embedding: returns a vector based on text length modulo."""
        # Simplistic mock: orthogonal vectors for different text patterns.
        if "article 6" in text.lower():
            return [1.0, 0.0, 0.0]
        if "article 7" in text.lower():
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def test_finds_best_matching_reference(self):
        """Returns idx of the most similar reference."""
        refs = [
            {"idx": 1, "chunk_text": "Article 6 requires risk assessment"},
            {"idx": 2, "chunk_text": "Article 7 defines obligations"},
        ]
        result = _find_better_match(
            "article 6 compliance", refs, self._constant_embedding
        )
        assert result == 1

    def test_empty_claim_returns_none(self):
        """Empty claim context returns None."""
        refs = [{"idx": 1, "chunk_text": "Some text"}]
        assert _find_better_match("", refs, self._constant_embedding) is None
        assert _find_better_match("   ", refs, self._constant_embedding) is None

    def test_no_chunk_text_returns_none(self):
        """References without chunk_text are skipped."""
        refs = [{"idx": 1}, {"idx": 2, "chunk_text": ""}]
        assert _find_better_match("some claim", refs, self._constant_embedding) is None

    def test_embedding_failure_returns_none(self):
        """Returns None when embedding function returns empty."""
        refs = [{"idx": 1, "chunk_text": "text"}]
        assert _find_better_match("claim", refs, lambda _: []) is None


class TestVerifyCitationSimilarity:
    """Tests for verify_citation_similarity - end-to-end citation verification."""

    @staticmethod
    def _mock_embedding(text: str) -> list[float]:
        """Mock embedding that returns similar vectors for similar content."""
        # Texts containing "risk" get one direction, "obligation" another
        if "risk" in text.lower():
            return [0.9, 0.1, 0.0]
        if "obligation" in text.lower():
            return [0.1, 0.9, 0.0]
        return [0.0, 0.0, 1.0]

    def test_verified_citation_high_similarity(self):
        """Citation is verified when claim context matches chunk text."""
        answer = "Risk assessment is required [1] for high-risk systems."
        refs = [
            {"idx": 1, "chunk_text": "Risk assessment procedures must be established."}
        ]
        result = verify_citation_similarity(
            answer, refs, self._mock_embedding, similarity_threshold=0.5
        )
        assert 1 in result.verified
        assert len(result.suspicious) == 0

    def test_suspicious_citation_low_similarity(self):
        """Citation is flagged when claim context does not match chunk text."""
        answer = "Risk assessment is required [1] for compliance."
        refs = [{"idx": 1, "chunk_text": "Obligations for provider transparency."}]
        result = verify_citation_similarity(
            answer, refs, self._mock_embedding, similarity_threshold=0.99
        )
        assert len(result.suspicious) == 1
        assert result.suspicious[0]["idx"] == 1

    def test_nonexistent_citation_flagged(self):
        """Citation referencing non-existent source is flagged."""
        answer = "See [99] for details."
        refs = [{"idx": 1, "chunk_text": "Some text"}]
        result = verify_citation_similarity(answer, refs, self._mock_embedding)
        assert len(result.suspicious) == 1
        assert result.suspicious[0]["reason"] == "citation_not_in_references"

    def test_empty_answer_returns_clean_result(self):
        """Empty answer returns empty verified list with overall_score 1.0."""
        result = verify_citation_similarity("", [], self._mock_embedding)
        assert result.verified == []
        assert result.suspicious == []
        assert result.overall_score == 1.0

    def test_no_chunk_text_auto_verified(self):
        """Citation with no chunk_text in reference is auto-verified."""
        answer = "See [1] for details."
        refs = [{"idx": 1}]  # No chunk_text
        result = verify_citation_similarity(answer, refs, self._mock_embedding)
        assert 1 in result.verified

    def test_overall_score_is_average(self):
        """Overall score is the average of individual scores."""
        answer = "Risk [1] and obligation [2]."
        refs = [
            {"idx": 1, "chunk_text": "Risk assessment required."},
            {
                "idx": 2,
                "chunk_text": "Risk assessment required.",
            },  # Mismatch with "obligation" context
        ]
        result = verify_citation_similarity(
            answer, refs, self._mock_embedding, similarity_threshold=0.5
        )
        assert len(result.scores) == 2
        assert result.overall_score == pytest.approx(sum(result.scores) / 2, abs=0.01)
