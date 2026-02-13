"""EUR-Lex legislation listing service.

Provides functionality to:
- Query EUR-Lex SPARQL endpoint for available EU legislation
- Compare with locally ingested corpora
- Detect outdated versions that need re-ingestion

Security features:
- URL whitelist (only eur-lex.europa.eu and publications.europa.eu)
- CELEX number validation
- HTTPS-only connections
- Rate limiting (1 req/sec with 24h cache)
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

# Security: Only allow requests to official EU domains
ALLOWED_DOMAINS = frozenset([
    "eur-lex.europa.eu",
    "publications.europa.eu",
])

# EUR-Lex SPARQL endpoint
SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

# CELEX number validation pattern
# Format: 5 digits (sector + year) + 1 letter (type) + 1-5 digits (number)
# Examples: 32024R1689, 32024R259, 32024R12345
CELEX_PATTERN = re.compile(r"^\d{5}[A-Z]\d{1,5}$")

# Rate limiting
RATE_LIMIT_SECONDS = 1.0
_last_request_time: float | None = None

# Maximum file size for downloads (50MB)
MAX_DOWNLOAD_SIZE_BYTES = 50 * 1024 * 1024


@dataclass
class LegislationInfo:
    """Metadata about a piece of EU legislation."""

    celex_number: str
    title_da: str
    title_en: str
    last_modified: datetime | None  # Document date (work_date_document)
    in_force: bool
    amended_by: list[str] = field(default_factory=list)

    # Legal dates
    entry_into_force: datetime | None = None  # When the law became binding

    # Local status
    is_ingested: bool = False
    corpus_id: str | None = None  # Short ID like 'gdpr', 'ai-act'
    local_version_date: datetime | None = None
    is_outdated: bool = False

    # Download URL
    html_url: str = ""

    # Additional metadata
    document_type: str = ""  # R=Regulation, L=Directive, D=Decision

    # EuroVoc subject keywords (from EU thesaurus)
    eurovoc_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "celex_number": self.celex_number,
            "title_da": self.title_da,
            "title_en": self.title_en,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "entry_into_force": self.entry_into_force.isoformat() if self.entry_into_force else None,
            "in_force": self.in_force,
            "amended_by": self.amended_by,
            "is_ingested": self.is_ingested,
            "corpus_id": self.corpus_id,
            "local_version_date": self.local_version_date.isoformat() if self.local_version_date else None,
            "is_outdated": self.is_outdated,
            "html_url": self.html_url,
            "document_type": self.document_type,
            "eurovoc_labels": self.eurovoc_labels,
        }


@dataclass
class UpdateStatus:
    """Status of a corpus update check."""

    corpus_id: str
    celex_number: str
    is_outdated: bool
    local_date: datetime | None
    remote_date: datetime | None
    reason: str = ""


class EurLexSecurityError(Exception):
    """Raised when a security check fails."""
    pass


class EurLexValidationError(Exception):
    """Raised when input validation fails."""
    pass


class EurLexNetworkError(Exception):
    """Raised when a network request fails."""
    pass


def validate_eurlex_url(url: str) -> bool:
    """Validate that a URL points to an allowed EUR-Lex domain.

    Args:
        url: The URL to validate

    Returns:
        True if valid

    Raises:
        EurLexSecurityError: If the URL is not allowed
    """
    if not url:
        raise EurLexSecurityError("URL cannot be empty")

    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise EurLexSecurityError(f"Only HTTPS URLs are allowed, got: {parsed.scheme}")

    if parsed.netloc not in ALLOWED_DOMAINS:
        raise EurLexSecurityError(
            f"Domain not allowed: {parsed.netloc}. "
            f"Allowed domains: {', '.join(sorted(ALLOWED_DOMAINS))}"
        )

    return True


def validate_celex(celex: str) -> bool:
    """Validate a CELEX number format.

    Args:
        celex: The CELEX number to validate (e.g., "32024R1689")

    Returns:
        True if valid

    Raises:
        EurLexValidationError: If the format is invalid
    """
    if not celex:
        raise EurLexValidationError("CELEX number cannot be empty")

    celex = celex.strip().upper()

    if not CELEX_PATTERN.match(celex):
        raise EurLexValidationError(
            f"Invalid CELEX format: {celex}. "
            f"Expected format: 5 digits + letter + 4 digits (e.g., 32024R1689)"
        )

    return True


def _rate_limit() -> None:
    """Apply rate limiting to external requests."""
    global _last_request_time

    if _last_request_time is not None:
        elapsed = time.time() - _last_request_time
        if elapsed < RATE_LIMIT_SECONDS:
            time.sleep(RATE_LIMIT_SECONDS - elapsed)

    _last_request_time = time.time()


def _get_cache_date() -> str:
    """Get today's date for cache key (24-hour invalidation)."""
    return datetime.now().strftime("%Y-%m-%d")


# Simple cache: {(query_hash, date): results}
_sparql_cache: dict[tuple[str, str], list[dict]] = {}


def query_sparql(query: str, timeout: int = 120) -> list[dict]:
    """Execute a SPARQL query against EUR-Lex.

    Results are cached for 24 hours to avoid excessive requests.

    Args:
        query: The SPARQL query to execute
        timeout: Request timeout in seconds (default 120 for large queries)

    Returns:
        List of result bindings

    Raises:
        EurLexNetworkError: If the request fails
    """
    # Validate endpoint URL
    validate_eurlex_url(SPARQL_ENDPOINT)

    # Compute cache key
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    cache_date = _get_cache_date()
    cache_key = (query_hash, cache_date)

    # Check cache
    if cache_key in _sparql_cache:
        return _sparql_cache[cache_key]

    # Clean old cache entries (different date)
    old_keys = [k for k in _sparql_cache if k[1] != cache_date]
    for k in old_keys:
        del _sparql_cache[k]

    # Apply rate limiting
    _rate_limit()

    try:
        response = requests.post(
            SPARQL_ENDPOINT,
            data={"query": query},
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "EuLex-RAG-Framework/1.0",
            },
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", {}).get("bindings", [])

        # Cache results
        _sparql_cache[cache_key] = results

        return results

    except requests.RequestException as e:
        raise EurLexNetworkError(f"SPARQL query failed: {e}")


def extract_celex_from_url(url: str) -> str | None:
    """Extract CELEX number from a EUR-Lex URL.

    Args:
        url: A EUR-Lex URL (e.g., "...?uri=CELEX:32024R1689")

    Returns:
        The CELEX number or None if not found
    """
    # Pattern: CELEX:32024R1689 or CELEX%3A32024R1689 in URL
    # Use non-capturing group to match either : or %3A (URL-encoded colon)
    match = re.search(r"CELEX(?::|%3A)(\d{5}[A-Z]\d{4})", url, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def build_html_url(celex: str, language: str = "DA") -> str:
    """Build the EUR-Lex HTML URL for a CELEX number.

    Args:
        celex: The CELEX number
        language: Language code (default: DA for Danish)

    Returns:
        The EUR-Lex HTML URL
    """
    validate_celex(celex)
    return f"https://eur-lex.europa.eu/legal-content/{language}/TXT/HTML/?uri=CELEX:{celex}"


def get_document_type(celex: str) -> str:
    """Get the document type from a CELEX number.

    Args:
        celex: The CELEX number (e.g., "32024R1689")

    Returns:
        Document type: "Regulation", "Directive", "Decision", or "Unknown"
    """
    if not celex or len(celex) < 6:
        return "Unknown"

    type_code = celex[5]
    types = {
        "R": "Regulation",
        "L": "Directive",
        "D": "Decision",
        "Q": "Recommendation",
        "H": "Resolution",
    }
    return types.get(type_code, "Unknown")


from enum import Enum


class DateFilterType(Enum):
    """Type of date to filter legislation by."""
    CREATION = "creation"      # Date when the law was adopted/created
    MODIFICATION = "modification"  # Date when the law was last modified


class DocumentType(Enum):
    """Type of EU legal document."""
    ALL = "all"              # Both regulations and directives
    REGULATION = "regulation"  # EU Regulations (Forordninger) - directly applicable
    DIRECTIVE = "directive"    # EU Directives (Direktiver) - requires national implementation


def _build_sparql_query(
    search_term: str = "",
    year_from: int = 2020,
    year_to: int = 2025,
    date_filter_type: DateFilterType = DateFilterType.CREATION,
    document_type: DocumentType = DocumentType.ALL,
    in_force_only: bool = False,
    include_eurovoc: bool = False,
) -> str:
    """Build SPARQL query for EUR-Lex legislation.

    Queries for regulations and directives with Danish titles.
    Uses the correct CDM ontology structure where expressions belong to works.

    Args:
        search_term: Optional search term for filtering (titles, CELEX, or EuroVoc)
        year_from: Start year (inclusive)
        year_to: End year (inclusive)
        date_filter_type: Whether to filter by creation or modification date
        document_type: Filter by document type (regulation, directive, or all)
        in_force_only: If True, only return legislation currently in force
        include_eurovoc: If True, include EuroVoc labels (slower but enables semantic search)

    Note: Year range is limited to 15 years per query.
    Note: EuroVoc adds ~20-60s to query time. Only enable when searching.
    """
    # EUR-Lex CDM ontology prefixes
    prefixes = """
    PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    """

    # Search filter - if search_term provided, filter on title (DA/EN/DE/FR), CELEX
    # When EuroVoc is enabled, also search on EuroVoc labels
    # Users often search using English terms like "AI Act" even for Danish results
    # Also match with spaces removed so "nis2" matches "NIS 2"
    search_filter = ""
    if search_term:
        safe_term = search_term.replace("\\", "\\\\").replace('"', '\\"')
        # Also create space-stripped version for matching abbreviations like "nis2" -> "NIS 2"
        safe_term_no_spaces = safe_term.replace(" ", "")
        if include_eurovoc:
            # Include EuroVoc in search (slower but more comprehensive)
            # Match both with and without spaces to handle abbreviations like "NIS 2" vs "nis2"
            search_filter = f'''
            FILTER(
                CONTAINS(LCASE(STR(?title_da)), LCASE("{safe_term}")) ||
                CONTAINS(LCASE(STR(?title_en)), LCASE("{safe_term}")) ||
                CONTAINS(LCASE(STR(?title_de)), LCASE("{safe_term}")) ||
                CONTAINS(LCASE(STR(?title_fr)), LCASE("{safe_term}")) ||
                CONTAINS(REPLACE(LCASE(STR(?title_da)), " ", ""), LCASE("{safe_term_no_spaces}")) ||
                CONTAINS(REPLACE(LCASE(STR(?title_en)), " ", ""), LCASE("{safe_term_no_spaces}")) ||
                CONTAINS(UCASE(?celex), UCASE("{safe_term}")) ||
                CONTAINS(LCASE(STR(?eurovoc_da)), LCASE("{safe_term}")) ||
                CONTAINS(LCASE(STR(?eurovoc_en)), LCASE("{safe_term}"))
            )
            '''
        else:
            # Title/CELEX only search (fast)
            search_filter = f'''
            FILTER(
                CONTAINS(LCASE(STR(?title_da)), LCASE("{safe_term}")) ||
                CONTAINS(LCASE(STR(?title_en)), LCASE("{safe_term}")) ||
                CONTAINS(LCASE(STR(?title_de)), LCASE("{safe_term}")) ||
                CONTAINS(LCASE(STR(?title_fr)), LCASE("{safe_term}")) ||
                CONTAINS(REPLACE(LCASE(STR(?title_da)), " ", ""), LCASE("{safe_term_no_spaces}")) ||
                CONTAINS(REPLACE(LCASE(STR(?title_en)), " ", ""), LCASE("{safe_term_no_spaces}")) ||
                CONTAINS(UCASE(?celex), UCASE("{safe_term}"))
            )
            '''

    # Document type filter - use CELEX letter (R=Regulation, L=Directive)
    # Note: We don't filter by rdf:type because most documents only have
    # generic types (cdm:work, cdm:resource_legal), not cdm:regulation/cdm:directive
    if document_type == DocumentType.REGULATION:
        celex_type_letter = "R"
    elif document_type == DocumentType.DIRECTIVE:
        celex_type_letter = "L"
    else:
        celex_type_letter = "[RL]"

    # In-force filter
    # Note: Many documents in EUR-Lex don't have the in-force field set correctly.
    # We use OPTIONAL + FILTER to include documents where:
    # 1. in-force is explicitly true/1, OR
    # 2. in-force field is missing (assume in-force for sector 3 legislation)
    # This avoids filtering out valid legislation like NIS2 that lacks the metadata.
    in_force_filter = ""
    if in_force_only:
        in_force_filter = '''
        OPTIONAL { ?work cdm:resource_legal_in-force ?_inforce . }
        FILTER(!BOUND(?_inforce) || ?_inforce = true || ?_inforce = "1"^^xsd:boolean || STR(?_inforce) = "1")
        '''

    # Build year filter based on CELEX year
    # Note: Both creation and modification use CELEX year because EUR-Lex doesn't
    # have reliable modification dates in the standard metadata. The CELEX year
    # represents when the law was originally adopted.
    year_patterns = [str(y) for y in range(year_from, year_to + 1)]
    year_regex = "|".join(year_patterns)
    celex_filter = f'FILTER(REGEX(?celex, "^3({year_regex}){celex_type_letter}[0-9]+$"))'
    date_filter = ""

    # Build EuroVoc parts conditionally (adds ~20-60s to query time)
    if include_eurovoc:
        eurovoc_select = """
           (GROUP_CONCAT(DISTINCT ?eurovoc_da; separator="||") AS ?eurovoc_labels_da)
           (GROUP_CONCAT(DISTINCT ?eurovoc_en; separator="||") AS ?eurovoc_labels_en)"""
        eurovoc_patterns = """
        # EuroVoc subject keywords - Danish labels
        OPTIONAL {
            ?work cdm:work_is_about_concept_eurovoc ?eurovoc_concept .
            ?eurovoc_concept skos:prefLabel ?eurovoc_da .
            FILTER(lang(?eurovoc_da) = "da")
        }

        # EuroVoc subject keywords - English labels (for search)
        OPTIONAL {
            ?work cdm:work_is_about_concept_eurovoc ?eurovoc_concept_en .
            ?eurovoc_concept_en skos:prefLabel ?eurovoc_en .
            FILTER(lang(?eurovoc_en) = "en")
        }
        """
    else:
        eurovoc_select = ""
        eurovoc_patterns = ""

    query = f"""{prefixes}
    SELECT ?celex ?title_da ?title_en ?date ?entry_into_force ?inforce{eurovoc_select}
    WHERE {{
        # Work must be a resource_legal (ensures we get legislation, not other documents)
        ?work rdf:type cdm:resource_legal .

        ?work cdm:resource_legal_id_celex ?celex .

        # CELEX pattern filter (sector 3 = secondary legislation)
        {celex_filter}

        # Date filter (for modification mode)
        {date_filter}

        # In-force filter
        {in_force_filter}

        # Danish expression (expressions belong to works)
        OPTIONAL {{
            ?expr_da cdm:expression_belongs_to_work ?work .
            ?expr_da cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/DAN> .
            ?expr_da cdm:expression_title ?title_da .
        }}

        # English expression (fallback and for search)
        OPTIONAL {{
            ?expr_en cdm:expression_belongs_to_work ?work .
            ?expr_en cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ENG> .
            ?expr_en cdm:expression_title ?title_en .
        }}

        # German expression (for search - common nicknames like "KI-Verordnung")
        OPTIONAL {{
            ?expr_de cdm:expression_belongs_to_work ?work .
            ?expr_de cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/DEU> .
            ?expr_de cdm:expression_title ?title_de .
        }}

        # French expression (for search)
        OPTIONAL {{
            ?expr_fr cdm:expression_belongs_to_work ?work .
            ?expr_fr cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/FRA> .
            ?expr_fr cdm:expression_title ?title_fr .
        }}
        {eurovoc_patterns}
        # Date of document (optional - some entries may not have it)
        OPTIONAL {{ ?work cdm:work_date_document ?date . }}

        # Entry into force date (when the law becomes legally binding)
        OPTIONAL {{ ?work cdm:resource_legal_date_entry-into-force ?entry_into_force . }}

        # In-force status
        OPTIONAL {{ ?work cdm:resource_legal_in-force ?inforce . }}

        # Must have at least one title
        FILTER(BOUND(?title_da) || BOUND(?title_en))

        {search_filter}
    }}
    GROUP BY ?celex ?title_da ?title_en ?date ?entry_into_force ?inforce
    ORDER BY DESC(?date)
    """
    return query


# Maximum year span allowed per query
MAX_YEAR_SPAN = 15


def list_available_legislation(
    search_term: str = "",
    year_from: int | None = None,
    year_to: int | None = None,
    date_filter_type: DateFilterType = DateFilterType.CREATION,
    document_type: DocumentType = DocumentType.ALL,
    in_force_only: bool = False,
) -> list[LegislationInfo]:
    """List available EU legislation from EUR-Lex via SPARQL.

    Queries the EUR-Lex SPARQL endpoint for regulations and directives,
    preferring Danish titles when available.

    Results are cached for 24 hours to avoid expensive repeated queries.

    Args:
        search_term: Optional search term for filtering titles/CELEX
        year_from: Start year (default: current year - 4)
        year_to: End year (default: current year)
        date_filter_type: Filter by creation date or last modification date
        document_type: Filter by regulation, directive, or all
        in_force_only: If True, only return legislation currently in force

    Returns:
        List of LegislationInfo objects from EUR-Lex

    Raises:
        EurLexNetworkError: If the SPARQL query fails
        ValueError: If year range exceeds MAX_YEAR_SPAN (15 years)
    """
    # Default to last 5 years if not specified
    current_year = datetime.now().year
    if year_to is None:
        year_to = current_year
    if year_from is None:
        year_from = year_to - (MAX_YEAR_SPAN - 1)  # -14 to get 15 years

    # Validate year range
    year_span = year_to - year_from + 1
    if year_span > MAX_YEAR_SPAN:
        raise ValueError(
            f"Year range cannot exceed {MAX_YEAR_SPAN} years. "
            f"Requested: {year_from}-{year_to} ({year_span} years)"
        )
    if year_span < 1:
        raise ValueError(f"Invalid year range: {year_from}-{year_to}")

    # EuroVoc adds significant time to queries (~60s for 15 years)
    # but provides valuable semantic search capability
    # Always include EuroVoc - UI shows progress indicator
    include_eurovoc = True

    query = _build_sparql_query(
        search_term=search_term,
        year_from=year_from,
        year_to=year_to,
        date_filter_type=date_filter_type,
        document_type=document_type,
        in_force_only=in_force_only,
        include_eurovoc=include_eurovoc,
    )

    try:
        results = query_sparql(query)
    except EurLexNetworkError:
        # If SPARQL fails, return empty list (UI will show error)
        raise

    legislation = []
    seen_celex = set()  # Deduplicate

    for binding in results:
        celex = binding.get("celex", {}).get("value", "")
        if not celex or celex in seen_celex:
            continue
        seen_celex.add(celex)

        # Extract values from SPARQL binding
        title_da = binding.get("title_da", {}).get("value", "")
        title_en = binding.get("title_en", {}).get("value", "")
        date_str = binding.get("date", {}).get("value", "")
        entry_into_force_str = binding.get("entry_into_force", {}).get("value", "")
        in_force_str = binding.get("inforce", {}).get("value", "")
        doc_type = binding.get("docType", {}).get("value", "Unknown")

        # Parse EuroVoc labels (prefer Danish, fallback to English)
        eurovoc_da_str = binding.get("eurovoc_labels_da", {}).get("value", "")
        eurovoc_en_str = binding.get("eurovoc_labels_en", {}).get("value", "")

        # Use Danish labels, fall back to English if none
        eurovoc_str = eurovoc_da_str or eurovoc_en_str
        eurovoc_labels = []
        if eurovoc_str:
            # Split by || delimiter and deduplicate while preserving order
            seen = set()
            for label in eurovoc_str.split("||"):
                label = label.strip()
                if label and label not in seen:
                    seen.add(label)
                    eurovoc_labels.append(label)

        # Parse document date
        last_modified = None
        if date_str:
            try:
                last_modified = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except ValueError:
                try:
                    # Try just date format
                    last_modified = datetime.strptime(date_str[:10], "%Y-%m-%d")
                except ValueError:
                    pass

        # Parse entry into force date
        entry_into_force = None
        if entry_into_force_str:
            try:
                entry_into_force = datetime.fromisoformat(entry_into_force_str.replace("Z", "+00:00"))
            except ValueError:
                try:
                    entry_into_force = datetime.strptime(entry_into_force_str[:10], "%Y-%m-%d")
                except ValueError:
                    pass

        # Parse in_force (could be "true"/"false", "1"/"0", or URI)
        in_force = True  # Default to true (unknown = assume in force)
        if in_force_str:
            # EUR-Lex uses "1" for in force, "0" for not in force
            in_force = in_force_str in ("true", "1", "yes", "True")

        info = LegislationInfo(
            celex_number=celex,
            title_da=title_da or title_en,  # Fallback to English if no Danish
            title_en=title_en or title_da,  # Fallback to Danish if no English
            last_modified=last_modified,
            entry_into_force=entry_into_force,
            in_force=in_force,
            html_url=build_html_url(celex),
            document_type=doc_type if doc_type != "Unknown" else get_document_type(celex),
            eurovoc_labels=eurovoc_labels,
        )
        legislation.append(info)

    return legislation


def check_for_updates(
    corpus_id: str,
    local_corpora: dict,
) -> UpdateStatus:
    """Check if a local corpus needs to be updated.

    Args:
        corpus_id: The corpus ID to check
        local_corpora: The local corpora inventory dictionary

    Returns:
        UpdateStatus with outdated status and reason
    """
    corpus_data = local_corpora.get("corpora", {}).get(corpus_id)

    if not corpus_data:
        return UpdateStatus(
            corpus_id=corpus_id,
            celex_number="",
            is_outdated=False,
            local_date=None,
            remote_date=None,
            reason="Corpus not found in local inventory",
        )

    source_url = corpus_data.get("source_url", "")
    celex = extract_celex_from_url(source_url)

    if not celex:
        return UpdateStatus(
            corpus_id=corpus_id,
            celex_number="",
            is_outdated=False,
            local_date=None,
            remote_date=None,
            reason="Could not extract CELEX number from source URL",
        )

    local_date_str = corpus_data.get("ingested_at")
    local_date = None
    if local_date_str:
        try:
            local_date = datetime.fromisoformat(local_date_str.replace("Z", "+00:00"))
        except ValueError:
            pass

    # For now, we can't check remote dates without SPARQL
    # This would require a proper SPARQL query to EUR-Lex
    return UpdateStatus(
        corpus_id=corpus_id,
        celex_number=celex,
        is_outdated=False,  # Would need remote date comparison
        local_date=local_date,
        remote_date=None,
        reason="Version check requires SPARQL query (not implemented yet)",
    )


def get_legislation_by_celex(celex: str) -> LegislationInfo | None:
    """Get metadata for a specific piece of legislation by CELEX number.

    Args:
        celex: The CELEX number (e.g., "32024R1689")

    Returns:
        LegislationInfo or None if not found
    """
    validate_celex(celex)

    # Build basic info from CELEX
    return LegislationInfo(
        celex_number=celex,
        title_da="",  # Would need SPARQL
        title_en="",  # Would need SPARQL
        last_modified=None,
        in_force=True,
        html_url=build_html_url(celex),
        document_type=get_document_type(celex),
    )


def _get_cellar_id_from_celex(celex: str) -> str:
    """Get the Cellar UUID from a CELEX number.

    The CELEX resource returns RDF with a redirect to the Cellar ID.

    Args:
        celex: The CELEX number

    Returns:
        The Cellar UUID

    Raises:
        EurLexNetworkError: If lookup fails
    """
    celex_url = f"https://publications.europa.eu/resource/celex/{celex}"
    validate_eurlex_url(celex_url)

    _rate_limit()

    try:
        # HEAD request to get redirect location
        response = requests.head(
            celex_url,
            headers={
                "User-Agent": "EuLex-RAG-Framework/1.0 (Legal RAG System)",
                "Accept": "*/*",
            },
            timeout=30,
            allow_redirects=True,
        )
        response.raise_for_status()

        # Extract Cellar ID from redirect URL
        # Format: http://publications.europa.eu/resource/cellar/{uuid}/rdf/...
        final_url = response.url
        match = re.search(r"/cellar/([a-f0-9-]{36})", final_url)
        if not match:
            raise EurLexNetworkError(
                f"Could not extract Cellar ID from redirect URL: {final_url}"
            )

        return match.group(1)

    except requests.RequestException as e:
        raise EurLexNetworkError(f"Failed to lookup Cellar ID for {celex}: {e}")


def _download_via_cellar_api(celex: str, output_path: Path) -> Path:
    """Download legislation HTML via EUR-Lex Cellar REST API.

    The Cellar API is the official programmatic access point for EUR-Lex
    documents. This uses a two-step process:
    1. Get Cellar UUID from CELEX number
    2. Fetch HTML using content negotiation from Cellar resource

    Args:
        celex: The CELEX number
        output_path: Directory to save the HTML file

    Returns:
        Path to the downloaded file

    Raises:
        EurLexNetworkError: If download fails
    """
    # Step 1: Get Cellar ID from CELEX
    cellar_id = _get_cellar_id_from_celex(celex)

    # Step 2: Fetch HTML from Cellar resource
    # Note: HTTP (not HTTPS) is required for content negotiation to work
    cellar_url = f"http://publications.europa.eu/resource/cellar/{cellar_id}"

    _rate_limit()

    try:
        # Use content negotiation to request HTML in Danish
        response = requests.get(
            cellar_url,
            headers={
                "User-Agent": "EuLex-RAG-Framework/1.0 (Legal RAG System)",
                # Request HTML/XHTML, prefer Danish
                "Accept": "text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8",
                "Accept-Language": "da, en;q=0.8",
            },
            timeout=120,
            stream=True,
            allow_redirects=True,
        )
        response.raise_for_status()

        # Verify we got HTML/XHTML content
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower() and "xhtml" not in content_type.lower():
            raise EurLexNetworkError(
                f"Unexpected content type from Cellar API: {content_type}"
            )

        # Check content length
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE_BYTES:
            raise EurLexSecurityError(
                f"File too large: {int(content_length)} bytes "
                f"(max: {MAX_DOWNLOAD_SIZE_BYTES} bytes)"
            )

        # Download with size limit
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{celex}.html"

        downloaded = 0
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_SIZE_BYTES:
                    file_path.unlink(missing_ok=True)
                    raise EurLexSecurityError(
                        f"Download exceeded size limit: {MAX_DOWNLOAD_SIZE_BYTES} bytes"
                    )
                f.write(chunk)

        return file_path

    except requests.RequestException as e:
        raise EurLexNetworkError(f"Cellar API download failed for {celex}: {e}")


def _download_via_direct_url(celex: str, output_path: Path) -> Path:
    """Download legislation HTML via direct EUR-Lex URL (fallback method).

    Note: This method may be blocked by AWS WAF bot protection.
    Use _download_via_cellar_api as the primary method.

    Args:
        celex: The CELEX number
        output_path: Directory to save the HTML file

    Returns:
        Path to the downloaded file

    Raises:
        EurLexNetworkError: If download fails
    """
    url = build_html_url(celex)
    validate_eurlex_url(url)

    _rate_limit()

    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "da-DK,da;q=0.9,en;q=0.8",
            },
            timeout=60,
            stream=True,
            allow_redirects=True,
        )

        # Check for WAF challenge (202 status or JavaScript challenge in body)
        if response.status_code == 202:
            raise EurLexNetworkError(
                "EUR-Lex returned WAF challenge (status 202). "
                "Direct download is blocked. Use Cellar API instead."
            )

        response.raise_for_status()

        # Check content length
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE_BYTES:
            raise EurLexSecurityError(
                f"File too large: {int(content_length)} bytes "
                f"(max: {MAX_DOWNLOAD_SIZE_BYTES} bytes)"
            )

        # Download with size limit
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{celex}.html"

        downloaded = 0
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > MAX_DOWNLOAD_SIZE_BYTES:
                    file_path.unlink(missing_ok=True)
                    raise EurLexSecurityError(
                        f"Download exceeded size limit: {MAX_DOWNLOAD_SIZE_BYTES} bytes"
                    )
                f.write(chunk)

        return file_path

    except requests.RequestException as e:
        raise EurLexNetworkError(f"Failed to download {url}: {e}")


def download_legislation_html(celex: str, output_path: Path) -> Path:
    """Download legislation HTML from EUR-Lex.

    Tries multiple download methods in order:
    1. Cellar REST API (official programmatic access)
    2. Direct EUR-Lex URL (may be blocked by WAF)

    Args:
        celex: The CELEX number
        output_path: Directory to save the HTML file

    Returns:
        Path to the downloaded file

    Raises:
        EurLexSecurityError: If URL validation fails
        EurLexNetworkError: If all download methods fail
    """
    validate_celex(celex)

    errors = []

    # Try Cellar API first (official programmatic access)
    try:
        return _download_via_cellar_api(celex, output_path)
    except (EurLexNetworkError, EurLexSecurityError) as e:
        errors.append(f"Cellar API: {e}")

    # Try direct URL as fallback
    try:
        return _download_via_direct_url(celex, output_path)
    except (EurLexNetworkError, EurLexSecurityError) as e:
        errors.append(f"Direct URL: {e}")

    # All methods failed
    raise EurLexNetworkError(
        f"All download methods failed for {celex}:\n" +
        "\n".join(f"  - {err}" for err in errors)
    )


def enrich_corpora_with_status(
    corpora_data: dict,
    available_legislation: list[LegislationInfo],
) -> list[LegislationInfo]:
    """Enrich legislation list with local ingestion status.

    Args:
        corpora_data: The local corpora inventory
        available_legislation: List of available legislation

    Returns:
        Enriched list with is_ingested and is_outdated flags set
    """
    local_corpora = corpora_data.get("corpora", {})

    # Build a map of CELEX -> corpus_id for local corpora
    celex_to_corpus = {}
    for corpus_id, data in local_corpora.items():
        source_url = data.get("source_url", "")
        celex = extract_celex_from_url(source_url)
        if celex:
            celex_to_corpus[celex] = (corpus_id, data)

    # Enrich each legislation entry
    for leg in available_legislation:
        if leg.celex_number in celex_to_corpus:
            corpus_id, data = celex_to_corpus[leg.celex_number]
            leg.is_ingested = True
            leg.corpus_id = corpus_id

            # Use display_name from corpora.json (has proper short name in parentheses)
            display_name = data.get("display_name")
            if display_name:
                leg.title_da = display_name

            # Check for local version date
            ingested_at = data.get("ingested_at")
            if ingested_at:
                try:
                    leg.local_version_date = datetime.fromisoformat(
                        ingested_at.replace("Z", "+00:00")
                    )
                except ValueError:
                    pass

            # Always prefer entry_into_force from corpora.json for ingested laws
            # (SPARQL often returns placeholder dates that are incorrect)
            entry_into_force_str = data.get("entry_into_force")
            if entry_into_force_str:
                try:
                    leg.entry_into_force = datetime.fromisoformat(entry_into_force_str)
                except ValueError:
                    pass

            # Use last_modified from corpora.json if available and SPARQL didn't provide one
            if leg.last_modified is None:
                last_modified_str = data.get("last_modified")
                if last_modified_str:
                    try:
                        leg.last_modified = datetime.fromisoformat(last_modified_str)
                    except ValueError:
                        pass

            # Check if outdated (would need remote date)
            # For now, assume not outdated
            leg.is_outdated = False

    return available_legislation
