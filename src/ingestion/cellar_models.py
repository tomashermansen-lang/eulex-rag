"""Frozen dataclasses and validation constants for CELLAR SPARQL query results.

This module defines the typed return values for the cellar_client module:
- CaseMetadata: metadata for a single CJEU case
- CaseDocumentRef: reference to a case document on EUR-Lex
- CellarQueryError: transport/query failure exception
- ECLI_PATTERN / CASE_LAW_CELEX_PATTERN: validation patterns
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CaseMetadata:
    """Metadata for a single CJEU case returned from CELLAR SPARQL."""

    ecli: str
    case_number: str
    celex_id: str
    date: str  # ISO format (YYYY-MM-DD)
    court: str
    title: str
    articles_interpreted: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseDocumentRef:
    """Reference to a case document on EUR-Lex."""

    celex_id: str
    ecli: str
    html_url: str


class CellarQueryError(Exception):
    """Raised on CELLAR transport or query failures (HTTP errors, timeouts, retries exhausted).

    Separates transport errors from input validation (ValueError).
    """


# ECLI pattern: ECLI:EU:<court>:<year>:<ordinal>[.<sub>]
# Court codes: C = Court of Justice, T = General Court
# Edge 11: allows dots in ordinal (e.g., ECLI:EU:C:2020:790.1)
ECLI_PATTERN = re.compile(r"^ECLI:EU:[CT]:\d{4}:\d+(\.\d+)?$")

# Case law CELEX pattern: 5 digits + 1-2 letter court code + 1-5 digits
# Examples: 62020CJ0790 (CJ), 62020TJ0123 (TJ), 62020C0790 (C)
CASE_LAW_CELEX_PATTERN = re.compile(r"^\d{5}[A-Z]{1,2}\d{1,5}$")
