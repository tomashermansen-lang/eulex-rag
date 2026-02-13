"""Test that embeddings include heading_path_display and LLM-generated terms for semantic retrieval."""
import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.slow
def test_enrich_document_for_embedding_with_heading():
    """Verify that heading_path_display is prepended to document."""
    from src.engine.indexing import _enrich_document_for_embedding
    
    doc = "Udbydere af AI-modeller skal overholde..."
    meta = {"heading_path_display": "Chapter V (AI-MODELLER TIL ALMEN BRUG) > Article 53"}
    
    result = _enrich_document_for_embedding(doc, meta)
    
    assert "Chapter V (AI-MODELLER TIL ALMEN BRUG)" in result
    assert "Article 53" in result
    assert "Udbydere af AI-modeller" in result
    # Heading should come before document
    assert result.index("Chapter V") < result.index("Udbydere")


@patch("src.ingestion.embedding_enrichment.is_enrichment_enabled", return_value=False)
def test_enrich_document_for_embedding_no_heading(_mock_enabled):
    """Verify graceful handling when no heading_path_display exists."""
    from src.engine.indexing import _enrich_document_for_embedding
    
    doc = "Chunk uden heading."
    meta = {}
    
    result = _enrich_document_for_embedding(doc, meta)
    
    assert result == doc


@patch("src.ingestion.embedding_enrichment.is_enrichment_enabled", return_value=False)
def test_enrich_document_for_embedding_empty_heading(_mock_enabled):
    """Verify graceful handling when heading_path_display is empty string."""
    from src.engine.indexing import _enrich_document_for_embedding
    
    doc = "Chunk med tom heading."
    meta = {"heading_path_display": ""}
    
    result = _enrich_document_for_embedding(doc, meta)
    
    assert result == doc


@patch("src.ingestion.embedding_enrichment.is_enrichment_enabled", return_value=False)
def test_enrich_document_for_embedding_whitespace_heading(_mock_enabled):
    """Verify graceful handling when heading_path_display is whitespace."""
    from src.engine.indexing import _enrich_document_for_embedding
    
    doc = "Chunk med whitespace heading."
    meta = {"heading_path_display": "   "}
    
    result = _enrich_document_for_embedding(doc, meta)
    
    assert result == doc


@pytest.mark.slow
def test_upsert_uses_enriched_documents_for_embedding():
    """Verify that _upsert_with_embeddings uses enriched text for embedding."""
    from src.engine.indexing import _upsert_with_embeddings
    
    engine = MagicMock()
    captured_embed_input = []
    def capture_embed(docs):
        captured_embed_input.extend(docs)
        return [[0.1] * 10 for _ in docs]
    engine._embed = capture_embed
    engine.collection = MagicMock()
    
    ids = ["chunk-1"]
    documents = ["Original chunk text."]
    metadatas = [{"heading_path_display": "Chapter I > Article 1 (Formål)"}]
    
    _upsert_with_embeddings(engine, ids=ids, documents=documents, metadatas=metadatas)
    
    # Embedding should be called with enriched text
    assert len(captured_embed_input) == 1
    assert "Chapter I" in captured_embed_input[0]
    assert "Formål" in captured_embed_input[0]
    assert "Original chunk text." in captured_embed_input[0]
    
    # But ChromaDB should store original document (not enriched)
    upsert_call = engine.collection.upsert.call_args
    stored_docs = upsert_call.kwargs.get("documents") or upsert_call[1].get("documents")
    assert stored_docs == ["Original chunk text."]


@patch("src.ingestion.embedding_enrichment.is_enrichment_enabled", return_value=False)
def test_upsert_without_heading_still_works(_mock_enabled):
    """Verify that upsert works when no heading_path_display exists."""
    from src.engine.indexing import _upsert_with_embeddings
    
    engine = MagicMock()
    captured_embed_input = []
    def capture_embed(docs):
        captured_embed_input.extend(docs)
        return [[0.1] * 10 for _ in docs]
    engine._embed = capture_embed
    engine.collection = MagicMock()
    
    ids = ["chunk-1"]
    documents = ["Original chunk text."]
    metadatas = [{}]  # No heading_path_display
    
    _upsert_with_embeddings(engine, ids=ids, documents=documents, metadatas=metadatas)
    
    # Embedding should be called with original text (no enrichment)
    assert len(captured_embed_input) == 1
    assert captured_embed_input[0] == "Original chunk text."


@pytest.mark.slow
def test_enrich_annex_point_without_title_uses_first_line():
    """Verify that annex points without titles use first line as fallback."""
    from src.engine.indexing import _enrich_document_for_embedding
    
    doc = "Beskæftigelse, forvaltning af arbejdstagere og adgang til selvstændig virksomhed:\na) AI-systemer der..."
    meta = {
        "heading_path_display": "Annex III (Højrisiko-AI-systemer) > Point 4",
        "annex": "III",
        "annex_point": "4",
        # annex_point_title is missing (None)
    }
    
    result = _enrich_document_for_embedding(doc, meta)
    
    # Should include the first line as inferred title
    assert "Beskæftigelse" in result
    assert "forvaltning af arbejdstagere" in result
    # Original content should still be there
    assert "AI-systemer der" in result


@pytest.mark.slow
def test_enrich_annex_point_with_title_does_not_use_fallback():
    """Verify that annex points with titles don't use first-line fallback."""
    from src.engine.indexing import _enrich_document_for_embedding
    
    doc = "a) AI-systemer der..."
    meta = {
        "heading_path_display": "Annex III > Point 4 (Beskæftigelse)",
        "annex": "III",
        "annex_point": "4",
        "annex_point_title": "Beskæftigelse",  # Title exists
    }
    
    result = _enrich_document_for_embedding(doc, meta)
    
    # Should use the existing heading, not modify it
    assert "Annex III > Point 4 (Beskæftigelse)" in result
    # Should NOT duplicate title
    assert result.count("Beskæftigelse") == 1


# =============================================================================
# LLM-based Embedding Enrichment Tests
# =============================================================================


class TestEnrichmentConfig:
    """Test configuration loading for LLM enrichment."""
    
    def test_is_enrichment_enabled_respects_config(self):
        """Enrichment enabled state comes from config."""
        import src.ingestion.embedding_enrichment as mod
        
        # Test with disabled config
        mod._config_cache = {"enabled": False}
        from src.ingestion.embedding_enrichment import is_enrichment_enabled
        assert is_enrichment_enabled() is False
        
        # Test with enabled config
        mod._config_cache = {"enabled": True}
        assert is_enrichment_enabled() is True
        
        mod._config_cache = None
    
    def test_get_enrichment_model_default(self):
        """Default model is gpt-4o-mini."""
        import src.ingestion.embedding_enrichment as mod
        mod._config_cache = {}
        
        from src.ingestion.embedding_enrichment import get_enrichment_model
        assert get_enrichment_model() == "gpt-4o-mini"
        mod._config_cache = None
    
    def test_get_max_terms_default(self):
        """Default max terms is 5."""
        import src.ingestion.embedding_enrichment as mod
        mod._config_cache = {}
        
        from src.ingestion.embedding_enrichment import get_max_terms
        assert get_max_terms() == 5
        mod._config_cache = None


class TestCacheKeyGeneration:
    """Test cache key generation."""
    
    def test_cache_key_deterministic(self):
        """Same input produces same cache key."""
        from src.ingestion.embedding_enrichment import _get_cache_key
        
        key1 = _get_cache_key("chunk text", "Artikel 50", "ai-act")
        key2 = _get_cache_key("chunk text", "Artikel 50", "ai-act")
        
        assert key1 == key2
        assert len(key1) == 16  # Truncated hash
    
    def test_cache_key_differs_for_different_input(self):
        """Different input produces different cache key."""
        from src.ingestion.embedding_enrichment import _get_cache_key
        
        key1 = _get_cache_key("chunk text 1", "Artikel 50", "ai-act")
        key2 = _get_cache_key("chunk text 2", "Artikel 50", "ai-act")
        
        assert key1 != key2


class TestGenerateEnrichmentTerms:
    """Test term generation."""
    
    def test_returns_empty_when_disabled(self):
        """Returns empty list when enrichment is disabled."""
        import src.ingestion.embedding_enrichment as mod
        mod._config_cache = {"enabled": False}
        
        from src.ingestion.embedding_enrichment import generate_enrichment_terms
        
        result = generate_enrichment_terms(
            "Udbydere af AI-systemer skal sikre syntetisk indhold er mærket.",
            article_title="Artikel 50",
            corpus_id="ai-act",
        )
        
        assert result == []
        mod._config_cache = None
    
    @patch("openai.OpenAI")
    def test_returns_terms_for_short_chunks_via_llm(self, mock_openai_class):
        """Returns terms for short chunks via LLM (per Anthropic best practice - all chunks get enrichment)."""
        import src.ingestion.embedding_enrichment as mod
        mod._config_cache = {"enabled": True, "cache_enabled": False}
        
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "KONTEKST: Kort tekst om AI.\nSØGETERMER: AI test | kort tekst"
        mock_client.chat.completions.create.return_value = mock_response
        
        from src.ingestion.embedding_enrichment import generate_enrichment_terms
        
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = generate_enrichment_terms(
                "Short text",  # Under 100 chars - but still gets processed
                article_title="Artikel 50",
                corpus_id="ai-act",
            )
        
        # All chunks get enrichment per Anthropic best practice
        assert len(result) >= 1
        mock_client.chat.completions.create.assert_called_once()
        mod._config_cache = None
    
    @patch("openai.OpenAI")
    def test_generates_terms_via_llm(self, mock_openai_class):
        """Generates terms via LLM when enabled."""
        import src.ingestion.embedding_enrichment as mod
        mod._config_cache = {"enabled": True, "cache_enabled": False}
        
        # Mock OpenAI response with new format (KONTEKST + SØGETERMER)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "KONTEKST: Denne bestemmelse fastsætter krav om mærkning af AI-genereret indhold.\nSØGETERMER: musik AI | generere billeder | AI-genereret indhold"
        mock_client.chat.completions.create.return_value = mock_response
        
        from src.ingestion.embedding_enrichment import generate_enrichment_terms
        
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = generate_enrichment_terms(
                "Udbydere af AI-systemer der genererer syntetisk lyd-, billed-, video- eller tekstindhold skal sikre at indholdet er mærket i et maskinlæsbart format.",
                article_title="Artikel 50 - Gennemsigtighed",
                corpus_id="ai-act",
            )
        
        assert len(result) == 3
        assert "musik AI" in result
        assert "generere billeder" in result
        mod._config_cache = None

    @patch("openai.OpenAI")
    def test_generate_enrichment_returns_full_result(self, mock_openai_class):
        """generate_enrichment returns EnrichmentResult with description and terms."""
        import src.ingestion.embedding_enrichment as mod
        mod._config_cache = {"enabled": True, "cache_enabled": False}
        
        # Mock OpenAI response with new format
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "KONTEKST: Denne bestemmelse fastsætter krav om mærkning af AI-genereret indhold.\nSØGETERMER: musik AI | generere billeder | AI-genereret indhold"
        mock_client.chat.completions.create.return_value = mock_response
        
        from src.ingestion.embedding_enrichment import generate_enrichment, EnrichmentResult
        
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = generate_enrichment(
                "Udbydere af AI-systemer der genererer syntetisk lyd-, billed-, video- eller tekstindhold skal sikre at indholdet er mærket i et maskinlæsbart format og kan detekteres som kunstigt genereret.",
                article_title="Artikel 50",
                corpus_id="ai-act",
            )
        
        assert result is not None
        assert isinstance(result, EnrichmentResult)
        assert "mærkning" in result.contextual_description
        assert len(result.search_terms) == 3
        assert result.terms == result.search_terms  # Backward compat alias
        mod._config_cache = None

    @patch("openai.OpenAI")
    def test_generate_enrichment_parses_single_line_format(self, mock_openai_class):
        """Parser handles when LLM puts KONTEKST and SØGETERMER on same line."""
        import src.ingestion.embedding_enrichment as mod
        mod._config_cache = {"enabled": True, "cache_enabled": False}
        
        # Mock OpenAI response with single-line format (problematic case)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # This is the problematic format we saw in production
        mock_response.choices[0].message.content = "KONTEKST: Denne forordning fastsætter regler for AI i EU. SØGETERMER: kunstig intelligens | AI regler | EU lovgivning"
        mock_client.chat.completions.create.return_value = mock_response
        
        from src.ingestion.embedding_enrichment import generate_enrichment
        
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            result = generate_enrichment(
                "Udbydere af AI-systemer der genererer syntetisk lyd-, billed-, video- eller tekstindhold skal sikre at indholdet er mærket i et maskinlæsbart format og kan detekteres som kunstigt genereret.",
                article_title="Artikel 50",
                corpus_id="ai-act",
            )
        
        assert result is not None
        # Description should NOT contain SØGETERMER
        assert "SØGETERMER" not in result.contextual_description
        assert "kunstig intelligens" not in result.contextual_description
        assert "forordning" in result.contextual_description
        # Terms should be parsed correctly
        assert len(result.search_terms) == 3
        assert "kunstig intelligens" in result.search_terms
        mod._config_cache = None

class TestEnrichTextForEmbedding:
    """Test the main enrichment entry point."""
    
    def test_returns_original_when_disabled(self):
        """Returns original doc when enrichment is disabled."""
        import src.ingestion.embedding_enrichment as mod
        mod._config_cache = {"enabled": False}
        
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding
        
        doc = "Original document text"
        meta = {"article": "50", "article_title": "Gennemsigtighed"}
        
        result = enrich_text_for_embedding(doc, meta, corpus_id="ai-act")
        
        assert result == doc
        mod._config_cache = None
    
    @patch("src.ingestion.embedding_enrichment.generate_enrichment_terms")
    def test_prepends_terms_when_generated(self, mock_generate):
        """Prepends terms block when terms are generated."""
        import src.ingestion.embedding_enrichment as mod
        mod._config_cache = {"enabled": True}
        
        mock_generate.return_value = ["musik AI", "generere billeder"]
        
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding
        
        doc = "Original document text about synthetic content"
        meta = {"article": "50", "article_title": "Gennemsigtighed"}
        
        result = enrich_text_for_embedding(doc, meta, corpus_id="ai-act")
        
        assert "[Søgetermer: musik AI | generere billeder]" in result
        assert "Original document text" in result
        assert meta.get("_enrichment_terms") == ["musik AI", "generere billeder"]
        mod._config_cache = None


class TestClearCache:
    """Test cache clearing."""
    
    def test_clear_enrichment_cache_returns_count(self):
        """clear_enrichment_cache returns number of files deleted."""
        from src.ingestion.embedding_enrichment import clear_enrichment_cache
        
        # Should not raise, returns 0 if no cache
        count = clear_enrichment_cache()
        assert isinstance(count, int)
        assert count >= 0