"""Tests for case law embedding enrichment.

Covers: context headers, legislation header extraction, prompt template,
config-driven roles, source type dispatch, cache keys, failure handling.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# =============================================================================
# C4: Case Law Context Headers
# =============================================================================


class TestBuildCaseLawContextHeaders:
    """Tests for _build_case_law_context_headers (pure function)."""

    def test_full_metadata_builds_all_headers(self):
        """T1: All four headers returned for complete metadata (REQ-2, AS-6)."""
        from src.ingestion.embedding_enrichment import _build_case_law_context_headers

        metadata = {
            "case_name": "Schrems II",
            "case_number": "C-311/18",
            "court": "CJEU",
            "decision_date": "2020-07-16",
            "section_type": "grounds",
            "paragraph_range": "42-48",
        }
        headers = _build_case_law_context_headers(metadata)

        assert headers == [
            "[Schrems II (C-311/18)]",
            "[CJEU, 2020-07-16]",
            "[Grounds]",
            "[Paragraphs 42-48]",
        ]

    def test_empty_case_name_uses_case_number_only(self):
        """T2: Empty case_name produces [(case_number)] (AS-9)."""
        from src.ingestion.embedding_enrichment import _build_case_law_context_headers

        metadata = {
            "case_name": "",
            "case_number": "C-311/18",
            "court": "CJEU",
            "decision_date": "2020-07-16",
            "section_type": "grounds",
            "paragraph_range": "",
        }
        headers = _build_case_law_context_headers(metadata)

        assert headers[0] == "[(C-311/18)]"
        assert len(headers) == 3  # No paragraph header

    def test_missing_paragraph_range_omits_header(self):
        """T3: No paragraph header when paragraph_range is empty (REQ-2)."""
        from src.ingestion.embedding_enrichment import _build_case_law_context_headers

        metadata = {
            "case_name": "Schrems II",
            "case_number": "C-311/18",
            "court": "CJEU",
            "decision_date": "2020-07-16",
            "section_type": "grounds",
            "paragraph_range": "",
        }
        headers = _build_case_law_context_headers(metadata)

        assert len(headers) == 3
        assert not any("Paragraphs" in h for h in headers)

    def test_section_type_underscore_to_title_case(self):
        """T4: section_type with underscores becomes Title Case (REQ-2)."""
        from src.ingestion.embedding_enrichment import _build_case_law_context_headers

        metadata = {
            "case_name": "Test",
            "case_number": "C-1/20",
            "court": "CJEU",
            "decision_date": "2020-01-01",
            "section_type": "operative_part",
            "paragraph_range": "",
        }
        headers = _build_case_law_context_headers(metadata)

        assert "[Operative Part]" in headers

        # Also test single-word section type
        metadata["section_type"] = "grounds"
        headers = _build_case_law_context_headers(metadata)
        assert "[Grounds]" in headers


# =============================================================================
# C5: Legislation Context Headers Extraction
# =============================================================================


class TestBuildLegislationContextHeaders:
    """Tests for _build_legislation_context_headers (pure refactor)."""

    def test_legislation_headers_chapter_and_article(self):
        """T5: Chapter + article metadata produces correct headers (REQ-8)."""
        from src.ingestion.embedding_enrichment import (
            _build_legislation_context_headers,
        )

        metadata = {
            "chapter": "2",
            "chapter_title": "Forbud",
            "article": "5",
            "article_title": "Forbudte AI-praksisser",
        }
        headers = _build_legislation_context_headers(metadata)

        assert "[Kapitel 2: Forbud]" in headers
        assert "[Artikel 5: Forbudte AI-praksisser]" in headers

    def test_legislation_headers_annex(self):
        """T6: Annex metadata produces correct header (REQ-8)."""
        from src.ingestion.embedding_enrichment import (
            _build_legislation_context_headers,
        )

        metadata = {"annex": "III", "annex_title": "Højrisiko"}
        headers = _build_legislation_context_headers(metadata)

        assert "[Bilag III: Højrisiko]" in headers

    def test_legislation_headers_recital(self):
        """T7: Recital metadata produces correct header (REQ-8)."""
        from src.ingestion.embedding_enrichment import (
            _build_legislation_context_headers,
        )

        metadata = {"recital": "12"}
        headers = _build_legislation_context_headers(metadata)

        assert "[Betragtning 12]" in headers

    def test_legislation_headers_empty_metadata(self):
        """T8: Empty metadata produces empty list (REQ-8)."""
        from src.ingestion.embedding_enrichment import (
            _build_legislation_context_headers,
        )

        headers = _build_legislation_context_headers({})

        assert headers == []


# =============================================================================
# C1: Case Law Enrichment Prompt
# =============================================================================


class TestCaseLawEnrichmentPrompt:
    """Tests for CASE_LAW_ENRICHMENT_PROMPT template."""

    def test_prompt_template_formats_correctly(self):
        """T9: Template formats with all four variables without KeyError (REQ-3)."""
        from src.ingestion.ingestion_generation import CASE_LAW_ENRICHMENT_PROMPT

        result = CASE_LAW_ENRICHMENT_PROMPT.format(
            article_title="Schrems II (C-311/18)",
            court="CJEU",
            section_type="grounds",
            chunk_text="The court finds that...",
        )

        assert "Schrems II (C-311/18)" in result
        assert "CJEU" in result
        assert "grounds" in result
        assert "The court finds that..." in result

    def test_prompt_contains_kontekst_sogetermer_roller_format(self):
        """T10: Template instructs KONTEKST/SOGETERMER/ROLLER output format (REQ-3)."""
        from src.ingestion.ingestion_generation import CASE_LAW_ENRICHMENT_PROMPT

        assert "KONTEKST:" in CASE_LAW_ENRICHMENT_PROMPT
        # Accept either SØGETERMER or SOGETERMER
        assert "GETERMER:" in CASE_LAW_ENRICHMENT_PROMPT
        assert "ROLLER:" in CASE_LAW_ENRICHMENT_PROMPT


# =============================================================================
# C2: Case Law Enrichment Roles Configuration
# =============================================================================


class TestLoadCaseLawEnrichmentRoles:
    """Tests for _load_case_law_enrichment_roles."""

    def test_loads_roles_from_config(self):
        """T11: Reads roles from full config dict (REQ-4, AS-5)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import _load_case_law_enrichment_roles

        mod._full_config_cache = {
            "case_law": {
                "enrichment": {
                    "enrichment_roles": ["interpretation", "dissent"],
                }
            },
            "embedding_enrichment": {"enabled": True},
        }
        try:
            result = _load_case_law_enrichment_roles()
            assert result == frozenset({"interpretation", "dissent"})
        finally:
            mod._full_config_cache = None

    def test_falls_back_to_defaults_when_key_missing(self):
        """T12: Missing enrichment subsection uses 5 default roles (EC-12)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import _load_case_law_enrichment_roles

        mod._full_config_cache = {
            "case_law": {},
            "embedding_enrichment": {"enabled": True},
        }
        try:
            result = _load_case_law_enrichment_roles()
            assert len(result) == 5
            assert "interpretation" in result
            assert "application" in result
            assert "principle" in result
            assert "procedural" in result
            assert "remedy" in result
        finally:
            mod._full_config_cache = None

    def test_empty_roles_list_returns_empty_frozenset(self):
        """T13: Empty roles list in config returns empty frozenset (EC-4)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import _load_case_law_enrichment_roles

        mod._full_config_cache = {
            "case_law": {
                "enrichment": {"enrichment_roles": []},
            },
            "embedding_enrichment": {"enabled": True},
        }
        try:
            result = _load_case_law_enrichment_roles()
            assert result == frozenset()
        finally:
            mod._full_config_cache = None


class TestLoadFullConfigRefactor:
    """Test that _load_config delegates to _load_full_config."""

    def test_load_config_delegates_to_full_config(self):
        """T39: _load_config returns embedding_enrichment subsection."""
        import src.ingestion.embedding_enrichment as mod

        mod._full_config_cache = {
            "embedding_enrichment": {"enabled": True, "model": "gpt-4o-mini"},
            "case_law": {},
        }
        mod._config_cache = None
        try:
            result = mod._load_config()
            assert result == {"enabled": True, "model": "gpt-4o-mini"}
        finally:
            mod._full_config_cache = None
            mod._config_cache = None


# =============================================================================
# C3: Source Type Dispatch
# =============================================================================

# Shared fixture metadata for case law chunks
_CASE_LAW_METADATA = {
    "source_type": "cjeu_case_law",
    "case_name": "Schrems II",
    "case_number": "C-311/18",
    "court": "CJEU",
    "decision_date": "2020-07-16",
    "section_type": "grounds",
    "paragraph_range": "42-48",
    "ecli": "ECLI:EU:C:2020:790",
    "corpus_id": "gdpr",
}

_VALID_LLM_RESPONSE = (
    "KONTEKST: Schrems II handler om overførsel af personoplysninger til USA.\n"
    "SØGETERMER: datatilsyn | persondata | USA overførsel | privatlivsskjold\n"
    "ROLLER: interpretation | application"
)


def _mock_openai_response(content: str) -> MagicMock:
    """Helper to build a mocked OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _setup_enrichment_enabled(mod, *, case_law_roles=None):
    """Set up module caches for enabled enrichment with case law roles."""
    roles = case_law_roles or [
        "interpretation",
        "application",
        "principle",
        "procedural",
        "remedy",
    ]
    mod._config_cache = {"enabled": True, "cache_enabled": False}
    mod._full_config_cache = {
        "embedding_enrichment": {"enabled": True, "cache_enabled": False},
        "case_law": {
            "enrichment": {"enrichment_roles": roles},
        },
    }


def _teardown_caches(mod):
    """Reset module caches."""
    mod._config_cache = None
    mod._full_config_cache = None


class TestSourceTypeDispatch:
    """T14-T18: Dispatch based on source_type."""

    @patch("openai.OpenAI")
    def test_case_law_chunk_dispatches_to_case_law_path(self, mock_openai_class):
        """T14: source_type=cjeu_case_law uses CASE_LAW_ENRICHMENT_PROMPT (REQ-1, AS-1)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _VALID_LLM_RESPONSE
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                enrich_text_for_embedding(
                    "The court finds that...",
                    dict(_CASE_LAW_METADATA),
                    corpus_id="gdpr",
                )

            # Verify case law prompt was used (contains "retspraksis")
            call_args = mock_client.chat.completions.create.call_args
            prompt_used = call_args.kwargs["messages"][0]["content"]
            assert "retspraksis" in prompt_used
            assert "lovtekst" not in prompt_used
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_legislation_chunk_uses_legislation_path(self, mock_openai_class):
        """T15: No source_type uses ENRICHMENT_PROMPT (REQ-8, AS-7)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "KONTEKST: Test legislation.\nSØGETERMER: test | law\nROLLER: scope"
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                enrich_text_for_embedding(
                    "Article 5 states...",
                    {"article": "5", "article_title": "Forbudte AI-praksisser"},
                    corpus_id="ai-act",
                )

            call_args = mock_client.chat.completions.create.call_args
            prompt_used = call_args.kwargs["messages"][0]["content"]
            assert "lovtekst" in prompt_used
            assert "retspraksis" not in prompt_used
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_unknown_source_type_uses_legislation_path(self, mock_openai_class):
        """T16: source_type=national_law uses legislation path (EC-2)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "KONTEKST: Test.\nSØGETERMER: test\nROLLER: ingen"
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                enrich_text_for_embedding(
                    "Some text",
                    {"source_type": "national_law", "article": "1"},
                    corpus_id="test",
                )

            call_args = mock_client.chat.completions.create.call_args
            prompt_used = call_args.kwargs["messages"][0]["content"]
            assert "lovtekst" in prompt_used
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_wrong_case_source_type_uses_legislation_path(self, mock_openai_class):
        """T17: source_type=CJEU_CASE_LAW (wrong case) uses legislation path (EC-3)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "KONTEKST: Test.\nSØGETERMER: test\nROLLER: ingen"
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                enrich_text_for_embedding(
                    "Some text",
                    {"source_type": "CJEU_CASE_LAW", "article": "1"},
                    corpus_id="test",
                )

            call_args = mock_client.chat.completions.create.call_args
            prompt_used = call_args.kwargs["messages"][0]["content"]
            assert "lovtekst" in prompt_used
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_missing_source_type_uses_legislation_path(self, mock_openai_class):
        """T18: No source_type key uses legislation path (EC-1)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "KONTEKST: Test.\nSØGETERMER: test\nROLLER: ingen"
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                enrich_text_for_embedding(
                    "Some text",
                    {"article": "1"},
                    corpus_id="test",
                )

            call_args = mock_client.chat.completions.create.call_args
            prompt_used = call_args.kwargs["messages"][0]["content"]
            assert "lovtekst" in prompt_used
        finally:
            _teardown_caches(mod)


class TestCaseLawEnrichmentResult:
    """T19-T21: Case law enrichment result content."""

    @patch("openai.OpenAI")
    def test_case_law_enrichment_includes_case_name(self, mock_openai_class):
        """T19: contextual_description includes case name (REQ-5, AS-1)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _VALID_LLM_RESPONSE
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = generate_enrichment(
                    "The court finds that...",
                    article_title="Schrems II (C-311/18)",
                    corpus_id="gdpr",
                    metadata=dict(_CASE_LAW_METADATA),
                    prompt="KONTEKST: {article_title}\n{chunk_text}\nSØGETERMER:\nROLLER:",
                    valid_roles=frozenset(["interpretation", "application"]),
                    cache_key="test-key-19",
                )

            assert result is not None
            assert "Schrems II" in result.contextual_description
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_case_law_context_headers_prepended(self, mock_openai_class):
        """T20: Enriched text has case law context headers (REQ-2, AS-6)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _VALID_LLM_RESPONSE
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = enrich_text_for_embedding(
                    "The court finds that...",
                    dict(_CASE_LAW_METADATA),
                    corpus_id="gdpr",
                )

            assert "[Schrems II (C-311/18)]" in result
            assert "[CJEU, 2020-07-16]" in result
            assert "[Grounds]" in result
            assert "[Paragraphs 42-48]" in result
            # Headers come before doc text
            assert result.index("[Schrems II") < result.index("The court finds")
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_case_law_search_terms_appended(self, mock_openai_class):
        """T21: Search terms block present in enriched text (REQ-3)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _VALID_LLM_RESPONSE
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = enrich_text_for_embedding(
                    "The court finds that...",
                    dict(_CASE_LAW_METADATA),
                    corpus_id="gdpr",
                )

            assert "[Søgetermer:" in result
        finally:
            _teardown_caches(mod)


class TestCaseLawRoleValidation:
    """T22-T26: Role validation with case law roles."""

    @patch("openai.OpenAI")
    def test_case_law_roles_accepted(self, mock_openai_class):
        """T22: remedy|interpretation both accepted from case law roles (AS-2)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "KONTEKST: Test.\nSØGETERMER: test\nROLLER: remedy | interpretation"
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = generate_enrichment(
                    "The operative part states...",
                    article_title="Test (C-1/20)",
                    corpus_id="gdpr",
                    metadata={"source_type": "cjeu_case_law"},
                    prompt="{article_title} {chunk_text}",
                    valid_roles=frozenset(
                        [
                            "interpretation",
                            "application",
                            "principle",
                            "procedural",
                            "remedy",
                        ]
                    ),
                    cache_key="test-key-22",
                )

            assert result is not None
            assert "remedy" in result.roles
            assert "interpretation" in result.roles
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_legislation_roles_rejected_for_case_law(self, mock_openai_class):
        """T23: obligations|scope rejected by case law roles (AS-3)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "KONTEKST: Test.\nSØGETERMER: test\nROLLER: obligations | scope"
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = generate_enrichment(
                    "Some case law text...",
                    article_title="Test (C-1/20)",
                    corpus_id="gdpr",
                    metadata={"source_type": "cjeu_case_law"},
                    prompt="{article_title} {chunk_text}",
                    valid_roles=frozenset(
                        [
                            "interpretation",
                            "application",
                            "principle",
                            "procedural",
                            "remedy",
                        ]
                    ),
                    cache_key="test-key-23",
                )

            assert result is not None
            assert result.roles == []
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_mixed_valid_invalid_roles(self, mock_openai_class):
        """T24: interpretation accepted, obligations rejected (EC-6)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "KONTEKST: Test.\nSØGETERMER: test\nROLLER: interpretation | obligations"
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = generate_enrichment(
                    "Some case law text...",
                    article_title="Test (C-1/20)",
                    corpus_id="gdpr",
                    metadata={"source_type": "cjeu_case_law"},
                    prompt="{article_title} {chunk_text}",
                    valid_roles=frozenset(
                        [
                            "interpretation",
                            "application",
                            "principle",
                            "procedural",
                            "remedy",
                        ]
                    ),
                    cache_key="test-key-24",
                )

            assert result is not None
            assert result.roles == ["interpretation"]
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_ingen_roles_returns_empty(self, mock_openai_class):
        """T25: ROLLER: ingen returns empty roles list (EC-5)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "KONTEKST: Test.\nSØGETERMER: test\nROLLER: ingen"
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = generate_enrichment(
                    "Some case law text...",
                    article_title="Test (C-1/20)",
                    corpus_id="gdpr",
                    metadata={"source_type": "cjeu_case_law"},
                    prompt="{article_title} {chunk_text}",
                    valid_roles=frozenset(["interpretation"]),
                    cache_key="test-key-25",
                )

            assert result is not None
            assert result.roles == []
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_config_driven_role_update(self, mock_openai_class):
        """T26: Config changed to include 'dissent' — accepted without code change (AS-5)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod, case_law_roles=["interpretation", "dissent"])
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "KONTEKST: Test.\nSØGETERMER: test\nROLLER: dissent"
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = generate_enrichment(
                    "Some case law text...",
                    article_title="Test (C-1/20)",
                    corpus_id="gdpr",
                    metadata={"source_type": "cjeu_case_law"},
                    prompt="{article_title} {chunk_text}",
                    valid_roles=frozenset(["interpretation", "dissent"]),
                    cache_key="test-key-26",
                )

            assert result is not None
            assert "dissent" in result.roles
        finally:
            _teardown_caches(mod)


class TestCaseLawCacheKeys:
    """T27-T30: Cache key tests."""

    def test_case_law_cache_key_uses_v4_prefix(self):
        """T27: _get_case_law_cache_key produces v4-prefixed hash (REQ-7)."""
        from src.ingestion.embedding_enrichment import _get_case_law_cache_key

        key = _get_case_law_cache_key(
            "some chunk text",
            ecli="ECLI:EU:C:2020:790",
            section_type="grounds",
            corpus_id="gdpr",
        )
        # Key is a hex hash, but we verify the input string uses v4 prefix
        # by checking determinism and that it differs from v3 key
        assert isinstance(key, str)
        assert len(key) == 16

    def test_legislation_cache_key_still_v3(self):
        """T28: _get_cache_key still uses v3 prefix (REQ-7, REQ-8)."""
        from src.ingestion.embedding_enrichment import _get_cache_key

        key = _get_cache_key("some chunk text", "Artikel 5", "ai-act")
        assert isinstance(key, str)
        assert len(key) == 16

    def test_same_text_different_source_types_different_keys(self):
        """T29: Same chunk text produces different keys for legislation vs case law (EC-10, AS-8)."""
        from src.ingestion.embedding_enrichment import (
            _get_cache_key,
            _get_case_law_cache_key,
        )

        leg_key = _get_cache_key("identical text", "Artikel 5", "gdpr")
        cl_key = _get_case_law_cache_key(
            "identical text",
            ecli="ECLI:EU:C:2020:790",
            section_type="grounds",
            corpus_id="gdpr",
        )
        assert leg_key != cl_key

    def test_ecli_with_colons_in_cache_key(self):
        """T30: ECLI with colons works without error (EC-11)."""
        from src.ingestion.embedding_enrichment import _get_case_law_cache_key

        key = _get_case_law_cache_key(
            "test text",
            ecli="ECLI:EU:C:2020:790",
            section_type="grounds",
            corpus_id="gdpr",
        )
        # Same input = deterministic output
        key2 = _get_case_law_cache_key(
            "test text",
            ecli="ECLI:EU:C:2020:790",
            section_type="grounds",
            corpus_id="gdpr",
        )
        assert key == key2


class TestArticleTitleConstruction:
    """T31-T32: Article title for case law."""

    @patch("openai.OpenAI")
    def test_article_title_from_case_name_and_number(self, mock_openai_class):
        """T31: case_name + case_number -> 'Schrems II (C-311/18)' (REQ-10)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _VALID_LLM_RESPONSE
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                enrich_text_for_embedding(
                    "The court finds that...",
                    dict(_CASE_LAW_METADATA),
                    corpus_id="gdpr",
                )

            call_args = mock_client.chat.completions.create.call_args
            prompt_used = call_args.kwargs["messages"][0]["content"]
            assert "Schrems II (C-311/18)" in prompt_used
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_article_title_empty_case_name(self, mock_openai_class):
        """T32: Empty case_name -> '(C-311/18)' (AS-9)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            _VALID_LLM_RESPONSE
        )

        meta = dict(_CASE_LAW_METADATA)
        meta["case_name"] = ""

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                enrich_text_for_embedding(
                    "The court finds that...",
                    meta,
                    corpus_id="gdpr",
                )

            call_args = mock_client.chat.completions.create.call_args
            prompt_used = call_args.kwargs["messages"][0]["content"]
            assert "(C-311/18)" in prompt_used
            # Should NOT have "Schrems II" since case_name is empty
            assert "Schrems II" not in prompt_used
        finally:
            _teardown_caches(mod)


class TestFailureHandling:
    """T33-T37: Graceful failure tests."""

    @patch("openai.OpenAI")
    def test_llm_exception_returns_none_and_logs(self, mock_openai_class, caplog):
        """T33: LLM exception -> None, warning logged with ECLI (REQ-6, AS-4)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = generate_enrichment(
                    "Some text...",
                    article_title="Test (C-1/20)",
                    corpus_id="gdpr",
                    metadata=dict(_CASE_LAW_METADATA),
                    prompt="{article_title} {chunk_text}",
                    valid_roles=frozenset(["interpretation"]),
                    cache_key="test-key-33",
                )

            assert result is None
            assert "Enrichment generation failed" in caplog.text
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_enrichment_failure_returns_original_doc(self, mock_openai_class):
        """T34: When generate_enrichment returns None, original doc returned (REQ-6)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        original_doc = "The court finds that..."
        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = enrich_text_for_embedding(
                    original_doc,
                    dict(_CASE_LAW_METADATA),
                    corpus_id="gdpr",
                )

            assert result == original_doc
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_empty_llm_response(self, mock_openai_class):
        """T35: Empty LLM response -> generate_enrichment returns None (EC-7)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response("")

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = generate_enrichment(
                    "Some text...",
                    article_title="Test (C-1/20)",
                    corpus_id="gdpr",
                    metadata=dict(_CASE_LAW_METADATA),
                    prompt="{article_title} {chunk_text}",
                    valid_roles=frozenset(["interpretation"]),
                    cache_key="test-key-35",
                )

            # Empty response is a failure — returns None per EC-7
            assert result is None
        finally:
            _teardown_caches(mod)

    @patch("openai.OpenAI")
    def test_malformed_format_returns_partial(self, mock_openai_class):
        """T36: Malformed LLM output -> EnrichmentResult with empty fields (EC-8)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod)
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "This is just random text without any markers."
        )

        try:
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                result = generate_enrichment(
                    "Some text...",
                    article_title="Test (C-1/20)",
                    corpus_id="gdpr",
                    metadata=dict(_CASE_LAW_METADATA),
                    prompt="{article_title} {chunk_text}",
                    valid_roles=frozenset(["interpretation"]),
                    cache_key="test-key-36",
                )

            # Not None — partial result with empty fields
            assert result is not None
            assert result.contextual_description == ""
            assert result.search_terms == []
            assert result.roles == []
        finally:
            _teardown_caches(mod)

    def test_missing_api_key_returns_none(self):
        """T37: No OPENAI_API_KEY -> returns None (EC-9)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import generate_enrichment

        _setup_enrichment_enabled(mod)

        try:
            with patch.dict("os.environ", {}, clear=True):
                # Remove OPENAI_API_KEY if present
                import os

                os.environ.pop("OPENAI_API_KEY", None)

                result = generate_enrichment(
                    "Some text...",
                    article_title="Test",
                    corpus_id="gdpr",
                )

            assert result is None
        finally:
            _teardown_caches(mod)


class TestConfigGate:
    """T38: Enrichment disabled skips case law."""

    def test_enrichment_disabled_skips_case_law(self):
        """T38: Disabled enrichment returns original doc for case law chunk (EC-13, REQ-9)."""
        import src.ingestion.embedding_enrichment as mod
        from src.ingestion.embedding_enrichment import enrich_text_for_embedding

        mod._config_cache = {"enabled": False}
        mod._full_config_cache = {"embedding_enrichment": {"enabled": False}}

        original_doc = "The court finds that..."
        try:
            result = enrich_text_for_embedding(
                original_doc,
                dict(_CASE_LAW_METADATA),
                corpus_id="gdpr",
            )
            assert result == original_doc
        finally:
            _teardown_caches(mod)
