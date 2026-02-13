from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Dict

class ClaimIntent(str, Enum):
    SCOPE = "SCOPE"
    CLASSIFICATION = "CLASSIFICATION"
    REQUIREMENTS = "REQUIREMENTS"
    ENFORCEMENT = "ENFORCEMENT"
    GENERAL = "GENERAL"


class EvidenceType(str, Enum):
    SCOPE = "SCOPE"
    DEFINITION = "DEFINITION"
    CLASSIFICATION = "CLASSIFICATION"
    FORBIDDEN = "FORBIDDEN"
    ENFORCEMENT = "ENFORCEMENT"
    UNKNOWN = "UNKNOWN"


class UserProfile(str, Enum):
    LEGAL = "LEGAL"
    ENGINEERING = "ENGINEERING"
    ANY = "ANY"


class FocusType(str, Enum):
    CHAPTER = "chapter"
    ARTICLE = "article"
    ANNEX = "annex"
    PREAMBLE = "preamble"
    PREAMBLE_ITEM = "preamble_item"
    SECTION = "section"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FocusSelection:
    type: FocusType
    node_id: str
    title: str | None = None
    heading_path: str | None = None

    # Canonical refs (as stored in chunk metadata)
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    annex: str | None = None
    recital: str | None = None


@dataclass(frozen=True)
class LegalClaimGateResult:
    answer_text: str
    references_structured_all: list[dict[str, Any]]
    allow_reference_fallback: bool


class RAGEngineError(RuntimeError):
    """Raised when the RAG engine encounters a recoverable error."""
