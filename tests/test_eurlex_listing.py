"""Tests for EUR-Lex listing service.

Tests security functions and data validation.
"""

import pytest
from datetime import datetime

from src.ingestion.eurlex_listing import (
    validate_eurlex_url,
    validate_celex,
    extract_celex_from_url,
    build_html_url,
    get_document_type,
    LegislationInfo,
    EurLexSecurityError,
    EurLexValidationError,
    ALLOWED_DOMAINS,
)


class TestValidateEurlexUrl:
    """Tests for URL validation security."""

    def test_valid_eurlex_url(self):
        """Should accept valid EUR-Lex URLs."""
        url = "https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32024R1689"
        assert validate_eurlex_url(url) is True

    def test_valid_publications_url(self):
        """Should accept valid publications.europa.eu URLs."""
        url = "https://publications.europa.eu/webapi/rdf/sparql"
        assert validate_eurlex_url(url) is True

    def test_rejects_http_url(self):
        """Should reject HTTP (non-HTTPS) URLs."""
        url = "http://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32024R1689"
        with pytest.raises(EurLexSecurityError, match="Only HTTPS"):
            validate_eurlex_url(url)

    def test_rejects_non_eurlex_domain(self):
        """Should reject URLs from non-EUR-Lex domains."""
        url = "https://example.com/malicious"
        with pytest.raises(EurLexSecurityError, match="Domain not allowed"):
            validate_eurlex_url(url)

    def test_rejects_internal_network_url(self):
        """Should reject internal network URLs (SSRF protection)."""
        urls = [
            "https://localhost/admin",
            "https://127.0.0.1/secret",
            "https://192.168.1.1/internal",
            "https://10.0.0.1/private",
        ]
        for url in urls:
            with pytest.raises(EurLexSecurityError, match="Domain not allowed"):
                validate_eurlex_url(url)

    def test_rejects_empty_url(self):
        """Should reject empty URL."""
        with pytest.raises(EurLexSecurityError, match="cannot be empty"):
            validate_eurlex_url("")

    def test_allowed_domains_is_frozen(self):
        """Allowed domains should be immutable (frozenset)."""
        assert isinstance(ALLOWED_DOMAINS, frozenset)
        assert "eur-lex.europa.eu" in ALLOWED_DOMAINS
        assert "publications.europa.eu" in ALLOWED_DOMAINS


class TestValidateCelex:
    """Tests for CELEX number validation."""

    def test_valid_regulation_celex(self):
        """Should accept valid regulation CELEX numbers."""
        assert validate_celex("32024R1689") is True  # AI Act
        assert validate_celex("32016R0679") is True  # GDPR

    def test_valid_directive_celex(self):
        """Should accept valid directive CELEX numbers."""
        assert validate_celex("32022L2555") is True  # NIS2

    def test_valid_decision_celex(self):
        """Should accept valid decision CELEX numbers."""
        assert validate_celex("32022D1234") is True

    def test_case_insensitive(self):
        """Should handle lowercase input."""
        assert validate_celex("32024r1689") is True

    def test_rejects_invalid_format(self):
        """Should reject invalid CELEX formats."""
        invalid_celex = [
            "12345",           # Too short
            "1234567890",      # No letter
            "ABCDE12345",      # Letters in wrong position
            "3202XR1689",      # X instead of digit
            "32024-1689",      # Hyphen instead of letter
            "",                # Empty
        ]
        for celex in invalid_celex:
            with pytest.raises(EurLexValidationError):
                validate_celex(celex)

    def test_rejects_sql_injection_attempt(self):
        """Should reject SQL injection attempts."""
        with pytest.raises(EurLexValidationError):
            validate_celex("32024R1689'; DROP TABLE--")

    def test_rejects_empty_celex(self):
        """Should reject empty CELEX number."""
        with pytest.raises(EurLexValidationError, match="cannot be empty"):
            validate_celex("")


class TestExtractCelexFromUrl:
    """Tests for CELEX extraction from URLs."""

    def test_extract_from_standard_url(self):
        """Should extract CELEX from standard EUR-Lex URL."""
        url = "https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32024R1689"
        assert extract_celex_from_url(url) == "32024R1689"

    def test_extract_from_encoded_url(self):
        """Should extract CELEX from URL-encoded format."""
        url = "https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX%3A32016R0679"
        assert extract_celex_from_url(url) == "32016R0679"

    def test_returns_none_for_invalid_url(self):
        """Should return None for URLs without CELEX."""
        url = "https://eur-lex.europa.eu/homepage.html"
        assert extract_celex_from_url(url) is None

    def test_returns_uppercase_celex(self):
        """Should return uppercase CELEX."""
        url = "https://eur-lex.europa.eu/?uri=celex:32024r1689"
        assert extract_celex_from_url(url) == "32024R1689"


class TestBuildHtmlUrl:
    """Tests for HTML URL construction."""

    def test_build_danish_url(self):
        """Should build Danish HTML URL by default."""
        url = build_html_url("32024R1689")
        assert url == "https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32024R1689"

    def test_build_english_url(self):
        """Should build English HTML URL when specified."""
        url = build_html_url("32024R1689", language="EN")
        assert url == "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689"

    def test_validates_celex_before_building(self):
        """Should validate CELEX before building URL."""
        with pytest.raises(EurLexValidationError):
            build_html_url("invalid-celex")


class TestGetDocumentType:
    """Tests for document type detection."""

    def test_regulation_type(self):
        """Should detect regulations."""
        assert get_document_type("32024R1689") == "Regulation"

    def test_directive_type(self):
        """Should detect directives."""
        assert get_document_type("32022L2555") == "Directive"

    def test_decision_type(self):
        """Should detect decisions."""
        assert get_document_type("32022D1234") == "Decision"

    def test_unknown_type(self):
        """Should return Unknown for unrecognized types."""
        assert get_document_type("32024X1234") == "Unknown"

    def test_empty_celex(self):
        """Should handle empty CELEX."""
        assert get_document_type("") == "Unknown"


class TestLegislationInfo:
    """Tests for LegislationInfo dataclass."""

    def test_to_dict_serialization(self):
        """Should serialize to dictionary correctly."""
        info = LegislationInfo(
            celex_number="32024R1689",
            title_da="AI-forordningen",
            title_en="AI Act",
            last_modified=datetime(2024, 7, 12),
            in_force=True,
            amended_by=["32024R0001"],
            is_ingested=True,
            local_version_date=datetime(2025, 1, 15),
            is_outdated=False,
            html_url="https://eur-lex.europa.eu/...",
            document_type="Regulation",
        )

        d = info.to_dict()

        assert d["celex_number"] == "32024R1689"
        assert d["title_da"] == "AI-forordningen"
        assert d["title_en"] == "AI Act"
        assert d["last_modified"] == "2024-07-12T00:00:00"
        assert d["in_force"] is True
        assert d["amended_by"] == ["32024R0001"]
        assert d["is_ingested"] is True
        assert d["local_version_date"] == "2025-01-15T00:00:00"
        assert d["is_outdated"] is False
        assert d["document_type"] == "Regulation"

    def test_to_dict_handles_none_dates(self):
        """Should handle None dates in serialization."""
        info = LegislationInfo(
            celex_number="32024R1689",
            title_da="",
            title_en="",
            last_modified=None,
            in_force=True,
        )

        d = info.to_dict()

        assert d["last_modified"] is None
        assert d["local_version_date"] is None

    def test_default_values(self):
        """Should have correct default values."""
        info = LegislationInfo(
            celex_number="32024R1689",
            title_da="",
            title_en="",
            last_modified=None,
            in_force=True,
        )

        assert info.amended_by == []
        assert info.is_ingested is False
        assert info.local_version_date is None
        assert info.is_outdated is False
        assert info.html_url == ""
        assert info.document_type == ""
