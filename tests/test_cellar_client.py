"""Tests for cellar_client — SPARQL client for CELLAR endpoint.

Covers: input validation, feature flag, retry logic, rate limiting,
response parsing, parameterized queries, and all public API functions.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import pytest

from src.common.config_loader import CaseLawSettings, Settings
from src.ingestion.cellar_models import CaseMetadata, CellarQueryError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_settings(*, enabled: bool = True, **overrides) -> Settings:
    """Create a Settings object with case_law config for testing."""
    cl_kwargs = {
        "enabled": enabled,
        "sparql_endpoint": "https://publications.europa.eu/webapi/rdf/sparql",
        "request_timeout_secs": 30,
        "max_retries": 3,
        "request_delay_secs": 1.0,
        "base_retry_delay_secs": 1.0,
        "user_agent": "TestAgent/1.0",
    }
    cl_kwargs.update(overrides)
    return Settings(case_law=CaseLawSettings(**cl_kwargs))


def _sparql_json(bindings: list[dict]) -> dict:
    """Build a SPARQL JSON results dict from a list of binding dicts."""
    return {
        "results": {
            "bindings": [
                {k: {"type": "literal", "value": v} for k, v in b.items()}
                for b in bindings
            ]
        }
    }


CASE_BINDING_1 = {
    "ecli": "ECLI:EU:C:2020:790",
    "case_number": "C-311/18",
    "celex_id": "62018CJ0311",
    "date": "2020-07-16",
    "court": "CJEU",
    "title": "Schrems II",
    "articles_interpreted": "Art. 46, Art. 49",
}

CASE_BINDING_2 = {
    "ecli": "ECLI:EU:C:2022:258",
    "case_number": "C-252/21",
    "celex_id": "62021CJ0252",
    "date": "2022-04-28",
    "court": "CJEU",
    "title": "Meta Platforms",
    "articles_interpreted": "Art. 6",
}

CASE_BINDING_3 = {
    "ecli": "ECLI:EU:C:2023:100",
    "case_number": "C-100/22",
    "celex_id": "62022CJ0100",
    "date": "2023-03-15",
    "court": "CJEU",
    "title": "Test Case",
}


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset module-level rate limiter state between tests."""
    import src.ingestion.cellar_client as mod

    mod._last_request_time = None
    yield
    mod._last_request_time = None


@pytest.fixture()
def mock_settings(monkeypatch):
    """Patch load_settings to return test settings with case_law enabled."""
    settings = _make_settings(enabled=True)
    monkeypatch.setattr("src.ingestion.cellar_client.load_settings", lambda: settings)
    return settings


@pytest.fixture()
def mock_sparql(monkeypatch):
    """Patch SPARQLWrapper to return controlled results. Returns the mock instance."""
    mock_wrapper_instance = MagicMock()
    mock_wrapper_class = MagicMock(return_value=mock_wrapper_instance)
    monkeypatch.setattr("src.ingestion.cellar_client.SPARQLWrapper", mock_wrapper_class)
    # Suppress rate-limit sleeps
    monkeypatch.setattr("src.ingestion.cellar_client.time.sleep", lambda _: None)
    return mock_wrapper_instance


# ─────────────────────────────────────────────────────────────────────────────
# Input Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestInputValidation:
    """REQ-11, AS-8, AS-9, Edge 10-12: Input validation functions."""

    def test_validate_legislation_celex_valid(self):
        from src.ingestion.cellar_client import _validate_legislation_celex

        assert _validate_legislation_celex("32016R0679") == "32016R0679"

    def test_validate_legislation_celex_strips_and_uppercases(self):
        from src.ingestion.cellar_client import _validate_legislation_celex

        assert _validate_legislation_celex("  32016r0679  ") == "32016R0679"

    def test_validate_legislation_celex_invalid(self):
        from src.ingestion.cellar_client import _validate_legislation_celex

        with pytest.raises(ValueError, match="CELEX"):
            _validate_legislation_celex("invalid")

    def test_validate_legislation_celex_empty(self):
        from src.ingestion.cellar_client import _validate_legislation_celex

        with pytest.raises(ValueError, match="CELEX"):
            _validate_legislation_celex("")

    def test_validate_case_law_celex_valid(self):
        from src.ingestion.cellar_client import _validate_case_law_celex

        assert _validate_case_law_celex("62020CJ0790") == "62020CJ0790"

    def test_validate_case_law_celex_invalid(self):
        from src.ingestion.cellar_client import _validate_case_law_celex

        with pytest.raises(ValueError, match="case law CELEX"):
            _validate_case_law_celex("invalid")

    def test_validate_ecli_valid(self):
        from src.ingestion.cellar_client import _validate_ecli

        assert _validate_ecli("ECLI:EU:C:2020:790") == "ECLI:EU:C:2020:790"

    def test_validate_ecli_invalid(self):
        from src.ingestion.cellar_client import _validate_ecli

        with pytest.raises(ValueError, match="ECLI"):
            _validate_ecli("not-an-ecli")

    def test_validate_ecli_case_insensitive(self):
        """Edge 12: Lowercase ECLI normalized to uppercase."""
        from src.ingestion.cellar_client import _validate_ecli

        assert _validate_ecli("ecli:eu:c:2020:790") == "ECLI:EU:C:2020:790"

    def test_validate_ecli_with_dots(self):
        """Edge 11: ECLI with dots in ordinal accepted."""
        from src.ingestion.cellar_client import _validate_ecli

        assert _validate_ecli("ECLI:EU:C:2020:790.1") == "ECLI:EU:C:2020:790.1"

    def test_validate_celex_special_chars(self):
        """Edge 10, AS-10: URI-special chars rejected by pattern."""
        from src.ingestion.cellar_client import _validate_legislation_celex

        with pytest.raises(ValueError):
            _validate_legislation_celex("32016R0679'; DROP")

    @pytest.mark.parametrize(
        "payload",
        [
            'ECLI:EU:C:2020:790"} OPTIONAL {',  # SPARQL double-quote injection
            "ECLI:EU:C:2020:790} DELETE {",  # closing brace injection
            "ECLI:EU:C:2020:790\\n",  # newline injection
            "ECLI:EU:C:2020:790\\",  # backslash injection
        ],
    )
    def test_ecli_sparql_injection_rejected(self, payload):
        """SPARQL metacharacters in ECLI input are rejected by validation."""
        from src.ingestion.cellar_client import _validate_ecli

        with pytest.raises(ValueError):
            _validate_ecli(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            '62020CJ0790"} OPTIONAL {',  # SPARQL double-quote injection
            "62020CJ0790} DELETE {",  # closing brace injection
            "62020CJ0790\\n",  # newline injection
            "62020CJ0790\\",  # backslash injection
        ],
    )
    def test_case_law_celex_sparql_injection_rejected(self, payload):
        """SPARQL metacharacters in case law CELEX input are rejected by validation."""
        from src.ingestion.cellar_client import _validate_case_law_celex

        with pytest.raises(ValueError):
            _validate_case_law_celex(payload)


# ─────────────────────────────────────────────────────────────────────────────
# Parameterized Queries
# ─────────────────────────────────────────────────────────────────────────────


class TestParameterizedQueries:
    """AS-10, REQ-5: SPARQL templates use named placeholders only."""

    def test_sparql_templates_have_expected_placeholders(self):
        import re
        from src.ingestion.cellar_client import (
            _SPARQL_CASES_FOR_LEGISLATION,
            _SPARQL_CASE_DOCUMENT_URL,
            _SPARQL_CASE_METADATA,
        )

        # Each template should contain exactly its expected placeholder(s)
        cases_placeholders = set(
            re.findall(r"\{(\w+)\}", _SPARQL_CASES_FOR_LEGISLATION)
        )
        assert cases_placeholders == {"celex_id"}

        meta_placeholders = set(re.findall(r"\{(\w+)\}", _SPARQL_CASE_METADATA))
        assert meta_placeholders == {"ecli"}

        doc_placeholders = set(re.findall(r"\{(\w+)\}", _SPARQL_CASE_DOCUMENT_URL))
        assert doc_placeholders == {"celex_id", "eurlex_html_base_url"}

    def test_no_unvalidated_interpolation(self, mock_settings, mock_sparql):
        """Validated input used in .format(), not raw string concat."""
        from src.ingestion.cellar_client import fetch_cases_for_legislation

        mock_sparql.queryAndConvert.return_value = _sparql_json([])
        fetch_cases_for_legislation("32016R0679")

        # Verify setQuery was called with the exact formatted template
        from src.ingestion.cellar_client import _SPARQL_CASES_FOR_LEGISLATION

        call_args = mock_sparql.setQuery.call_args[0][0]
        expected = _SPARQL_CASES_FOR_LEGISLATION.format(celex_id="32016R0679")
        assert call_args == expected


# ─────────────────────────────────────────────────────────────────────────────
# Feature Flag
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureFlag:
    """AS-6, Edge 7, Edge 9: Feature flag gating."""

    def test_feature_disabled_returns_empty_list(self, monkeypatch):
        settings = _make_settings(enabled=False)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        result = fetch_cases_for_legislation("32016R0679")
        assert result == []

    def test_feature_disabled_returns_none_metadata(self, monkeypatch):
        settings = _make_settings(enabled=False)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        from src.ingestion.cellar_client import fetch_case_metadata

        result = fetch_case_metadata("ECLI:EU:C:2020:790")
        assert result is None

    def test_feature_disabled_returns_none_doc_url(self, monkeypatch):
        settings = _make_settings(enabled=False)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        from src.ingestion.cellar_client import fetch_case_document_url

        result = fetch_case_document_url("62020CJ0790")
        assert result is None

    def test_feature_disabled_no_network(self, monkeypatch):
        """No SPARQLWrapper calls when disabled."""
        settings = _make_settings(enabled=False)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        mock_class = MagicMock()
        monkeypatch.setattr("src.ingestion.cellar_client.SPARQLWrapper", mock_class)

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        fetch_cases_for_legislation("32016R0679")
        mock_class.assert_not_called()

    def test_config_missing_defaults_disabled(self, monkeypatch):
        """Edge 7: Default CaseLawSettings has enabled=False."""
        settings = Settings()  # Uses CaseLawSettings() defaults
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        result = fetch_cases_for_legislation("32016R0679")
        assert result == []

    def test_guard_checked_at_entry(self, monkeypatch):
        """Edge 9: Feature check at function entry."""
        call_count = 0
        original_settings = _make_settings(enabled=True)

        def counting_load():
            nonlocal call_count
            call_count += 1
            return original_settings

        monkeypatch.setattr("src.ingestion.cellar_client.load_settings", counting_load)
        monkeypatch.setattr("src.ingestion.cellar_client.time.sleep", lambda _: None)

        mock_wrapper = MagicMock()
        mock_wrapper.queryAndConvert.return_value = _sparql_json([CASE_BINDING_1])
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        fetch_cases_for_legislation("32016R0679")
        # load_settings called for feature guard + rate limiter + execute_sparql
        # Must be bounded (not called per-query-iteration in a loop)
        assert 1 <= call_count <= 5


# ─────────────────────────────────────────────────────────────────────────────
# Fetch Cases for Legislation
# ─────────────────────────────────────────────────────────────────────────────


class TestFetchCasesForLegislation:
    """REQ-1, AS-1, AS-3, Edge 1-3."""

    def test_happy_path_three_cases(self, mock_settings, mock_sparql):
        """AS-1: 3 mocked bindings -> 3 CaseMetadata objects."""
        mock_sparql.queryAndConvert.return_value = _sparql_json(
            [CASE_BINDING_1, CASE_BINDING_2, CASE_BINDING_3]
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        result = fetch_cases_for_legislation("32016R0679")

        assert len(result) == 3
        assert all(isinstance(m, CaseMetadata) for m in result)
        assert result[0].ecli == "ECLI:EU:C:2020:790"
        assert result[0].case_number == "C-311/18"
        assert result[0].articles_interpreted == ["Art. 46", "Art. 49"]

    def test_empty_results(self, mock_settings, mock_sparql):
        """AS-3, REQ-10: Zero bindings -> empty list."""
        mock_sparql.queryAndConvert.return_value = _sparql_json([])

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        result = fetch_cases_for_legislation("32016R0679")
        assert result == []

    def test_dedup_by_ecli(self, mock_settings, mock_sparql):
        """Edge 1: Duplicate ECLIs deduplicated (keep first)."""
        dup = dict(CASE_BINDING_1)
        dup["title"] = "Duplicate"
        mock_sparql.queryAndConvert.return_value = _sparql_json([CASE_BINDING_1, dup])

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        result = fetch_cases_for_legislation("32016R0679")
        assert len(result) == 1
        assert result[0].title == "Schrems II"

    def test_missing_fields_skipped(self, mock_settings, mock_sparql, caplog):
        """Edge 2: Binding missing required field -> skipped + warning."""
        incomplete = {"ecli": "ECLI:EU:C:2020:790"}  # missing case_number, etc.
        mock_sparql.queryAndConvert.return_value = _sparql_json(
            [incomplete, CASE_BINDING_2]
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        with caplog.at_level(logging.WARNING):
            result = fetch_cases_for_legislation("32016R0679")

        assert len(result) == 1
        assert result[0].ecli == "ECLI:EU:C:2022:258"
        assert any(
            "missing required field" in r.message.lower()
            or "skipping" in r.message.lower()
            for r in caplog.records
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fetch Case Metadata
# ─────────────────────────────────────────────────────────────────────────────


class TestFetchCaseMetadata:
    """REQ-2, AS-2."""

    def test_happy_path(self, mock_settings, mock_sparql):
        """AS-2: Single case metadata returned."""
        mock_sparql.queryAndConvert.return_value = _sparql_json([CASE_BINDING_1])

        from src.ingestion.cellar_client import fetch_case_metadata

        result = fetch_case_metadata("ECLI:EU:C:2020:790")

        assert result is not None
        assert result.case_number == "C-311/18"
        assert result.court == "CJEU"

    def test_not_found_returns_none(self, mock_settings, mock_sparql):
        mock_sparql.queryAndConvert.return_value = _sparql_json([])

        from src.ingestion.cellar_client import fetch_case_metadata

        result = fetch_case_metadata("ECLI:EU:C:2020:790")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Fetch Case Document URL
# ─────────────────────────────────────────────────────────────────────────────


class TestFetchCaseDocumentUrl:
    """REQ-3, AS-11."""

    def test_happy_path(self, mock_settings, mock_sparql):
        """AS-11: Returns validated EUR-Lex URL."""
        url = (
            "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:62020CJ0790"
        )
        mock_sparql.queryAndConvert.return_value = _sparql_json([{"url": url}])

        from src.ingestion.cellar_client import fetch_case_document_url

        result = fetch_case_document_url("62020CJ0790")
        assert result == url

    def test_not_found_returns_none(self, mock_settings, mock_sparql):
        mock_sparql.queryAndConvert.return_value = _sparql_json([])

        from src.ingestion.cellar_client import fetch_case_document_url

        result = fetch_case_document_url("62020CJ0790")
        assert result is None

    def test_non_eurlex_url_returns_none(self, mock_settings, mock_sparql):
        """Non-EUR-Lex URL from SPARQL is rejected, returns None."""
        mock_sparql.queryAndConvert.return_value = _sparql_json(
            [{"url": "https://evil.com/doc"}]
        )

        from src.ingestion.cellar_client import fetch_case_document_url

        result = fetch_case_document_url("62020CJ0790")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Retry Logic
# ─────────────────────────────────────────────────────────────────────────────


class TestRetryLogic:
    """REQ-6, AS-4, AS-5, Edge 4-6."""

    def test_503_twice_then_success(self, mock_settings, monkeypatch):
        """AS-4: Two 503s then success."""
        from SPARQLWrapper.SPARQLExceptions import EndPointInternalError

        mock_wrapper = MagicMock()
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Simulate HTTP 503
                resp = MagicMock()
                resp.code = 503
                resp.info.return_value = {}
                raise EndPointInternalError(resp)
            return _sparql_json([CASE_BINDING_1])

        mock_wrapper.queryAndConvert.side_effect = side_effect
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )
        sleep_calls = []
        monkeypatch.setattr(
            "src.ingestion.cellar_client.time.sleep", lambda d: sleep_calls.append(d)
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        result = fetch_cases_for_legislation("32016R0679")
        assert len(result) == 1
        assert result[0].ecli == "ECLI:EU:C:2020:790"

    def test_max_retries_exhausted(self, monkeypatch):
        """AS-5: All attempts fail -> CellarQueryError."""
        from SPARQLWrapper.SPARQLExceptions import EndPointInternalError

        settings = _make_settings(enabled=True, max_retries=2)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        mock_wrapper = MagicMock()
        resp = MagicMock()
        resp.code = 503
        resp.info.return_value = {}
        mock_wrapper.queryAndConvert.side_effect = EndPointInternalError(resp)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )
        monkeypatch.setattr("src.ingestion.cellar_client.time.sleep", lambda _: None)

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        with pytest.raises(CellarQueryError, match="retries exhausted|failed after"):
            fetch_cases_for_legislation("32016R0679")

    def test_timeout_counts_as_attempt(self, mock_settings, monkeypatch):
        """Edge 4: Timeout counts as failed attempt."""
        from urllib.error import URLError

        mock_wrapper = MagicMock()
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise URLError("timed out")
            return _sparql_json([CASE_BINDING_1])

        mock_wrapper.queryAndConvert.side_effect = side_effect
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )
        monkeypatch.setattr("src.ingestion.cellar_client.time.sleep", lambda _: None)

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        result = fetch_cases_for_legislation("32016R0679")
        assert len(result) == 1

    def test_400_no_retry(self, mock_settings, monkeypatch):
        """Edge 5: HTTP 400 -> CellarQueryError immediately (no retry)."""
        from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

        mock_wrapper = MagicMock()
        mock_wrapper.queryAndConvert.side_effect = QueryBadFormed(MagicMock())
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )
        monkeypatch.setattr("src.ingestion.cellar_client.time.sleep", lambda _: None)

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        with pytest.raises(CellarQueryError):
            fetch_cases_for_legislation("32016R0679")
        # Only called once — no retry
        assert mock_wrapper.queryAndConvert.call_count == 1

    def test_401_403_404_no_retry(self, mock_settings, monkeypatch):
        """Auth/not-found errors -> CellarQueryError immediately."""
        from SPARQLWrapper.SPARQLExceptions import Unauthorized

        mock_wrapper = MagicMock()
        mock_wrapper.queryAndConvert.side_effect = Unauthorized(MagicMock())
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )
        monkeypatch.setattr("src.ingestion.cellar_client.time.sleep", lambda _: None)

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        with pytest.raises(CellarQueryError):
            fetch_cases_for_legislation("32016R0679")
        assert mock_wrapper.queryAndConvert.call_count == 1

    def test_backoff_timing(self, monkeypatch):
        """Edge 6: sleep called with base * 2^attempt."""
        from SPARQLWrapper.SPARQLExceptions import EndPointInternalError

        settings = _make_settings(
            enabled=True, max_retries=3, base_retry_delay_secs=1.0
        )
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        mock_wrapper = MagicMock()
        resp = MagicMock()
        resp.code = 503
        resp.info.return_value = {}
        mock_wrapper.queryAndConvert.side_effect = EndPointInternalError(resp)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )

        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "src.ingestion.cellar_client.time.sleep", lambda d: sleep_calls.append(d)
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        with pytest.raises(CellarQueryError):
            fetch_cases_for_legislation("32016R0679")

        # Filter out rate-limit sleeps (small values near 0-1s) vs retry sleeps
        # Retry sleeps should be: 1.0, 2.0, 4.0
        retry_sleeps = [s for s in sleep_calls if s >= 1.0]
        assert len(retry_sleeps) >= 2
        # Verify exponential pattern: each should be roughly double the previous
        for i in range(1, len(retry_sleeps)):
            assert retry_sleeps[i] >= retry_sleeps[i - 1] * 1.5  # Allow some tolerance

    def test_429_with_retry_after(self, mock_settings, monkeypatch):
        """429 with Retry-After header uses header value."""
        from io import BytesIO
        from urllib.error import HTTPError

        mock_wrapper = MagicMock()
        call_count = 0

        def side_effect():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                err = HTTPError(
                    url="https://example.com/sparql",
                    code=429,
                    msg="Too Many Requests",
                    hdrs={"Retry-After": "5"},
                    fp=BytesIO(b""),
                )
                raise err
            return _sparql_json([CASE_BINDING_1])

        mock_wrapper.queryAndConvert.side_effect = side_effect
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )

        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "src.ingestion.cellar_client.time.sleep", lambda d: sleep_calls.append(d)
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        result = fetch_cases_for_legislation("32016R0679")
        assert len(result) == 1
        # Should have used Retry-After value of 5
        assert any(s == 5.0 or s == 5 for s in sleep_calls)


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────


class TestRateLimiting:
    """REQ-7, AS-7: Rate limiting between requests."""

    def test_delay_enforced(self, monkeypatch):
        """AS-7: sleep called when elapsed < delay."""
        import src.ingestion.cellar_client as mod

        settings = _make_settings(enabled=True, request_delay_secs=2.0)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        # Simulate a recent request 0.5s ago
        monkeypatch.setattr(mod, "_last_request_time", time.time() - 0.5)

        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "src.ingestion.cellar_client.time.sleep", lambda d: sleep_calls.append(d)
        )

        mock_wrapper = MagicMock()
        mock_wrapper.queryAndConvert.return_value = _sparql_json([CASE_BINDING_1])
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        fetch_cases_for_legislation("32016R0679")

        # Should have slept approximately 1.5s (2.0 - 0.5)
        rate_limit_sleeps = [s for s in sleep_calls if s > 0.5]
        assert len(rate_limit_sleeps) >= 1
        assert rate_limit_sleeps[0] > 1.0  # at least 1s remaining

    def test_no_delay_when_enough_time_passed(self, monkeypatch):
        """No sleep when elapsed >= delay."""
        import src.ingestion.cellar_client as mod

        settings = _make_settings(enabled=True, request_delay_secs=1.0)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        # Simulate a request 5s ago (well past the 1s delay)
        monkeypatch.setattr(mod, "_last_request_time", time.time() - 5.0)

        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "src.ingestion.cellar_client.time.sleep", lambda d: sleep_calls.append(d)
        )

        mock_wrapper = MagicMock()
        mock_wrapper.queryAndConvert.return_value = _sparql_json([CASE_BINDING_1])
        monkeypatch.setattr(
            "src.ingestion.cellar_client.SPARQLWrapper",
            MagicMock(return_value=mock_wrapper),
        )

        from src.ingestion.cellar_client import fetch_cases_for_legislation

        fetch_cases_for_legislation("32016R0679")

        # No rate-limit sleep (>0.1s) should have occurred
        rate_limit_sleeps = [s for s in sleep_calls if s > 0.1]
        assert len(rate_limit_sleeps) == 0


class TestWaitForRateLimit:
    """REQ-12: Public rate limiter for shared use by HTML downloads."""

    def test_delegates_to_internal_rate_limit(self, monkeypatch):
        """wait_for_rate_limit() shares state with SPARQL _rate_limit()."""
        import src.ingestion.cellar_client as mod

        settings = _make_settings(enabled=True, request_delay_secs=2.0)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        # Simulate a recent request 0.5s ago
        monkeypatch.setattr(mod, "_last_request_time", time.time() - 0.5)

        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "src.ingestion.cellar_client.time.sleep",
            lambda d: sleep_calls.append(d),
        )

        from src.ingestion.cellar_client import wait_for_rate_limit

        wait_for_rate_limit()

        # Should have slept approximately 1.5s (2.0 - 0.5)
        assert len(sleep_calls) == 1
        assert sleep_calls[0] > 1.0

    def test_two_rapid_calls_respect_delay(self, monkeypatch):
        """Two rapid wait_for_rate_limit() calls enforce delay."""
        import src.ingestion.cellar_client as mod

        settings = _make_settings(enabled=True, request_delay_secs=1.0)
        monkeypatch.setattr(
            "src.ingestion.cellar_client.load_settings", lambda: settings
        )

        # Start fresh
        monkeypatch.setattr(mod, "_last_request_time", None)

        sleep_calls: list[float] = []
        monkeypatch.setattr(
            "src.ingestion.cellar_client.time.sleep",
            lambda d: sleep_calls.append(d),
        )

        from src.ingestion.cellar_client import wait_for_rate_limit

        wait_for_rate_limit()  # First call — no sleep, stamps time
        wait_for_rate_limit()  # Second call — should sleep ~1.0s

        # Second call should have triggered a sleep
        assert len(sleep_calls) >= 1
        assert sleep_calls[-1] > 0.5
