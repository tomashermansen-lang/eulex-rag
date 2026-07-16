"""CELLAR SPARQL client for querying CJEU case law metadata.

Public API:
- fetch_cases_for_legislation(celex_id) -> list[CaseMetadata]
- fetch_case_metadata(ecli) -> CaseMetadata | None
- fetch_case_document_url(celex_id) -> str | None

All functions are gated by settings.case_law.enabled (REQ-8).
All inputs validated before query construction (REQ-5, REQ-11).
Retry with exponential backoff on transient errors (REQ-6).
Rate limiting between requests (REQ-7).
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError

from SPARQLWrapper import JSON as SPARQL_JSON
from SPARQLWrapper import SPARQLWrapper
from SPARQLWrapper.SPARQLExceptions import (
    EndPointInternalError,
    EndPointNotFound,
    QueryBadFormed,
    Unauthorized,
)

from src.common.config_loader import load_settings
from src.ingestion.cellar_models import (
    CASE_LAW_CELEX_PATTERN,
    ECLI_PATTERN,
    CaseMetadata,
    CellarQueryError,
)
from src.ingestion.eurlex_listing import (
    CELEX_PATTERN,
    EurLexSecurityError,
    validate_eurlex_url,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# SPARQL Query Templates
# ─────────────────────────────────────────────────────────────────────────────

_SPARQL_CASES_FOR_LEGISLATION = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>

SELECT DISTINCT ?ecli ?case_number ?celex_id ?date ?court ?title ?articles_interpreted
WHERE {{
  ?case cdm:resource_legal_cites_resource_legal ?legislation .
  ?legislation cdm:resource_legal_id_celex "{celex_id}" .
  ?case cdm:case-law_identifier_ecli ?ecli .
  ?case cdm:resource_legal_id_celex ?celex_id .
  OPTIONAL {{ ?case cdm:work_date_document ?date . }}
  OPTIONAL {{ ?case cdm:case-law_delivered_by_court ?court . }}
  OPTIONAL {{
    ?expr cdm:expression_belongs_to_work ?case .
    ?expr cdm:expression_title ?title .
    FILTER(LANG(?title) = "en" || LANG(?title) = "")
  }}
  OPTIONAL {{ ?case cdm:case-law_article_or_provision_interpreted ?articles_interpreted . }}
  BIND(REPLACE(STR(?ecli), "^.*/(C|T)-", "$1-") AS ?case_number)
}}
"""

_SPARQL_CASE_METADATA = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>

SELECT DISTINCT ?ecli ?case_number ?celex_id ?date ?court ?title ?articles_interpreted
WHERE {{
  ?case cdm:case-law_identifier_ecli "{ecli}" .
  ?case cdm:resource_legal_id_celex ?celex_id .
  BIND("{ecli}" AS ?ecli)
  OPTIONAL {{ ?case cdm:work_date_document ?date . }}
  OPTIONAL {{ ?case cdm:case-law_delivered_by_court ?court . }}
  OPTIONAL {{
    ?expr cdm:expression_belongs_to_work ?case .
    ?expr cdm:expression_title ?title .
    FILTER(LANG(?title) = "en" || LANG(?title) = "")
  }}
  OPTIONAL {{ ?case cdm:case-law_article_or_provision_interpreted ?articles_interpreted . }}
  BIND(REPLACE(STR(?ecli), "^.*/(C|T)-", "$1-") AS ?case_number)
}}
"""

_SPARQL_CASE_DOCUMENT_URL = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>

SELECT DISTINCT ?url
WHERE {{
  ?work cdm:resource_legal_id_celex "{celex_id}" .
  ?expr cdm:expression_belongs_to_work ?work .
  ?manif cdm:manifestation_manifests_expression ?expr .
  ?manif cdm:manifestation_type <http://publications.europa.eu/resource/authority/manifestation-type/html> .
  BIND(CONCAT("{eurlex_html_base_url}", "{celex_id}") AS ?url)
}}
LIMIT 1
"""

# ─────────────────────────────────────────────────────────────────────────────
# Input Validation (REQ-5, REQ-11)
# SECURITY INVARIANT: Validation patterns (CELEX_PATTERN, CASE_LAW_CELEX_PATTERN,
# ECLI_PATTERN) must never accept characters " { } \ as these would enable SPARQL
# injection via the str.format() template approach used below.
# ─────────────────────────────────────────────────────────────────────────────

# Non-retryable exception types (raise CellarQueryError immediately)
_NON_RETRYABLE = (QueryBadFormed, Unauthorized, EndPointNotFound)


def _validate_legislation_celex(celex_id: str) -> str:
    """Validate and normalize a legislation CELEX number.

    Returns normalized (stripped, uppercased) value.
    Raises ValueError if format is invalid.
    """
    normalized = celex_id.strip().upper()
    if not CELEX_PATTERN.match(normalized):
        raise ValueError(
            f"Invalid legislation CELEX format: {celex_id!r}. "
            f"Expected: 5 digits + letter + 1-5 digits (e.g., 32016R0679)"
        )
    return normalized


def _validate_case_law_celex(celex_id: str) -> str:
    """Validate and normalize a case law CELEX number.

    Returns normalized (stripped, uppercased) value.
    Raises ValueError if format is invalid.
    """
    normalized = celex_id.strip().upper()
    if not CASE_LAW_CELEX_PATTERN.match(normalized):
        raise ValueError(
            f"Invalid case law CELEX format: {celex_id!r}. "
            f"Expected: 5 digits + 1-2 letter court code + 1-5 digits (e.g., 62020CJ0790)"
        )
    return normalized


def _validate_ecli(ecli: str) -> str:
    """Validate and normalize an ECLI identifier.

    Normalizes to uppercase (Edge 12: case-insensitive).
    Allows dots in ordinal (Edge 11).
    Returns normalized value.
    Raises ValueError if format is invalid.
    """
    normalized = ecli.strip().upper()
    if not ECLI_PATTERN.match(normalized):
        raise ValueError(
            f"Invalid ECLI format: {ecli!r}. "
            f"Expected: ECLI:EU:<C|T>:<year>:<ordinal> (e.g., ECLI:EU:C:2020:790)"
        )
    return normalized


# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiter (REQ-7)
# ─────────────────────────────────────────────────────────────────────────────

_last_request_time: float | None = None


def _rate_limit() -> None:
    """Enforce minimum delay between SPARQL requests."""
    global _last_request_time

    settings = load_settings()
    delay = settings.case_law.request_delay_secs

    if _last_request_time is not None:
        elapsed = time.time() - _last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)

    _last_request_time = time.time()


def _reset_rate_limit() -> None:
    """Reset rate limiter state (for testing)."""
    global _last_request_time
    _last_request_time = None


def wait_for_rate_limit() -> None:
    """Enforce minimum delay between requests (shared with HTML downloads).

    Delegates to internal _rate_limit() to share state with SPARQL queries.
    """
    _rate_limit()


# ─────────────────────────────────────────────────────────────────────────────
# SPARQL Execution with Retry (REQ-6)
# ─────────────────────────────────────────────────────────────────────────────


def _execute_sparql(query: str) -> dict:
    """Execute a SPARQL query against the CELLAR endpoint with retry.

    Returns parsed SPARQL JSON results dict.
    Raises CellarQueryError on non-retryable errors or after retries exhausted.
    """
    settings = load_settings()
    cl = settings.case_law

    last_error: Exception | None = None

    for attempt in range(cl.max_retries + 1):
        _rate_limit()

        try:
            wrapper = SPARQLWrapper(cl.sparql_endpoint)
            wrapper.setQuery(query)
            wrapper.setReturnFormat(SPARQL_JSON)
            wrapper.addCustomHttpHeader("User-Agent", cl.user_agent)
            wrapper.setTimeout(cl.request_timeout_secs)
            result: dict = wrapper.queryAndConvert()  # type: ignore[assignment]
            return result

        except _NON_RETRYABLE as exc:
            raise CellarQueryError(f"Non-retryable SPARQL error: {exc}") from exc

        except HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = _get_retry_after_from_http_error(exc)
                logger.warning(
                    "CELLAR returned 429, retrying after %ss (attempt %d/%d)",
                    retry_after,
                    attempt + 1,
                    cl.max_retries + 1,
                )
                time.sleep(retry_after)
            elif 400 <= exc.code < 500:
                raise CellarQueryError(f"Non-retryable HTTP {exc.code}: {exc}") from exc
            else:
                delay = min(
                    cl.base_retry_delay_secs * (2**attempt), cl.max_backoff_secs
                )
                logger.warning(
                    "CELLAR returned HTTP %d, retrying in %.1fs (attempt %d/%d)",
                    exc.code,
                    delay,
                    attempt + 1,
                    cl.max_retries + 1,
                )
                time.sleep(delay)

        except EndPointInternalError as exc:
            last_error = exc
            delay = min(cl.base_retry_delay_secs * (2**attempt), cl.max_backoff_secs)
            logger.warning(
                "CELLAR returned server error, retrying in %.1fs (attempt %d/%d)",
                delay,
                attempt + 1,
                cl.max_retries + 1,
            )
            time.sleep(delay)

        except URLError as exc:
            last_error = exc
            delay = min(cl.base_retry_delay_secs * (2**attempt), cl.max_backoff_secs)
            logger.warning(
                "CELLAR connection error: %s, retrying in %.1fs (attempt %d/%d)",
                exc,
                delay,
                attempt + 1,
                cl.max_retries + 1,
            )
            time.sleep(delay)

    raise CellarQueryError(
        f"CELLAR query failed after {cl.max_retries + 1} attempts"
    ) from last_error


def _get_retry_after_from_http_error(exc: HTTPError) -> float:
    """Extract Retry-After header from HTTPError, fallback to default backoff."""
    try:
        retry_after = exc.headers.get("Retry-After") if exc.headers else None
        if retry_after is not None:
            return float(retry_after)
    except (TypeError, ValueError, AttributeError):
        pass
    settings = load_settings()
    return settings.case_law.base_retry_delay_secs


# ─────────────────────────────────────────────────────────────────────────────
# Response Parsing
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_CASE_FIELDS = {"ecli", "celex_id"}


def _parse_case_bindings(results: dict) -> list[CaseMetadata]:
    """Parse SPARQL JSON results into CaseMetadata list.

    Deduplicates by ECLI (Edge 1). Skips bindings with missing required fields (Edge 2).
    Returns empty list for zero bindings (REQ-10).
    """
    bindings = results.get("results", {}).get("bindings", [])
    seen_eclis: set[str] = set()
    cases: list[CaseMetadata] = []

    for binding in bindings:
        # Extract values from SPARQL binding format
        values = {k: v.get("value", "") for k, v in binding.items()}

        # Check required fields
        if not all(values.get(f) for f in _REQUIRED_CASE_FIELDS):
            logger.warning(
                "Skipping CELLAR binding with missing required fields: %s", values
            )
            continue

        ecli = values["ecli"]

        # Deduplicate by ECLI (Edge 1: keep first)
        if ecli in seen_eclis:
            continue
        seen_eclis.add(ecli)

        # Parse articles_interpreted (comma-separated string -> list)
        articles_raw = values.get("articles_interpreted", "")
        articles = (
            [a.strip() for a in articles_raw.split(",") if a.strip()]
            if articles_raw
            else []
        )

        cases.append(
            CaseMetadata(
                ecli=ecli,
                case_number=values.get("case_number", ""),
                celex_id=values["celex_id"],
                date=values.get("date", ""),
                court=values.get("court", ""),
                title=values.get("title", ""),
                articles_interpreted=articles,
            )
        )

    return cases


def _parse_document_url(results: dict) -> str | None:
    """Extract and validate first URL from SPARQL results."""
    bindings = results.get("results", {}).get("bindings", [])
    if not bindings:
        return None

    url = bindings[0].get("url", {}).get("value")
    if not url:
        return None

    try:
        validate_eurlex_url(url)
        return url
    except EurLexSecurityError:
        logger.warning("CELLAR returned invalid URL: %s", url)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature Flag Guard (REQ-8)
# ─────────────────────────────────────────────────────────────────────────────


def _feature_guard(default_factory: Callable[[], Any]) -> Callable:
    """Decorator factory: calls default_factory() when case_law is disabled.

    Using a factory callable avoids sharing mutable default values across calls.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            settings = load_settings()
            if not settings.case_law.enabled:
                return default_factory()
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


@_feature_guard(list)
def fetch_cases_for_legislation(celex_id: str) -> list[CaseMetadata]:
    """Query CELLAR for CJEU cases citing the given legislation.

    Args:
        celex_id: Legislation CELEX number (e.g., "32016R0679" for GDPR).

    Returns:
        List of CaseMetadata, deduplicated by ECLI. Empty list if no cases found.

    Raises:
        ValueError: If celex_id format is invalid.
        CellarQueryError: On transport/query failures after retries.
    """
    validated = _validate_legislation_celex(celex_id)
    query = _SPARQL_CASES_FOR_LEGISLATION.format(celex_id=validated)
    results = _execute_sparql(query)
    return _parse_case_bindings(results)


@_feature_guard(lambda: None)
def fetch_case_metadata(ecli: str) -> CaseMetadata | None:
    """Query CELLAR for metadata of a single case by ECLI.

    Args:
        ecli: ECLI identifier (e.g., "ECLI:EU:C:2020:790").

    Returns:
        CaseMetadata or None if not found.

    Raises:
        ValueError: If ECLI format is invalid.
        CellarQueryError: On transport/query failures after retries.
    """
    validated = _validate_ecli(ecli)
    query = _SPARQL_CASE_METADATA.format(ecli=validated)
    results = _execute_sparql(query)
    cases = _parse_case_bindings(results)
    return cases[0] if cases else None


@_feature_guard(lambda: None)
def fetch_case_document_url(celex_id: str) -> str | None:
    """Resolve the EUR-Lex HTML document URL for a case CELEX number.

    Args:
        celex_id: Case law CELEX number (e.g., "62020CJ0790").

    Returns:
        Validated EUR-Lex URL string, or None if not found.

    Raises:
        ValueError: If celex_id format is invalid.
        CellarQueryError: On transport/query failures after retries.
    """
    validated = _validate_case_law_celex(celex_id)
    settings = load_settings()
    query = _SPARQL_CASE_DOCUMENT_URL.format(
        celex_id=validated,
        eurlex_html_base_url=settings.case_law.eurlex_html_base_url,
    )
    results = _execute_sparql(query)
    return _parse_document_url(results)
