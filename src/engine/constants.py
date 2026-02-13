"""Constants for intent detection, normative claim matching, and citation expansion.

These keyword lists control how the RAG engine classifies user queries and
determines which legal articles are relevant. The lists are organised by intent
category (enforcement, requirements, classification, scope, definitions).

Also used by ingestion/citation_graph.py for article role detection during
law ingestion - this ensures consistent keyword matching across the pipeline.
"""

import re

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Intent detection keywords (substring matching unless noted otherwise)
# ---------------------------------------------------------------------------

# Enforcement intent: complaints, penalties, sanctions, supervisory authorities.
_INTENT_ENFORCEMENT_KEYWORDS_SUBSTR = [
    "klage",
    "klager",
    "bøde",
    "bøder",
    "sanktion",
    "sanktioner",
    "håndhævelse",
    "håndhæv",
    "enforcement",
    "complaint",
    "penalt",
    "fine",
    "sanction",
    "remedy",
    "redress",
    "tilsyn",
    "myndighed",
    "myndigheder",
    "kompetent myndighed",
    "påbud",
    "anke",
    "appel",
    "klagevej",
    "erstatning",
    "kompensation",
]

# Requirements intent (strong): explicit obligation/documentation queries.
_INTENT_REQUIREMENTS_KEYWORDS_STRONG_SUBSTR = [
    "krav",
    "kræves",
    "kræver",
    "forventes",
    "dokumentation",
    "teknisk dokumentation",
    "requirements",
    "hvad skal vi",
    "skal vi",
    "hvordan",
    "must",
    "should",
    "controls",
    "forpligt",
    "forpligtelser",
    "ret til",
    "retten til",
    "indsigtsret",
    "hvordan overholder",
    "hvordan efterlever",
    "overholde",
    "efterleve",
    "implementer",
    "implementere",
    "indføre",
    "etablere",
]

# Classification intent: risk categories, annexes, prohibited AI.
_INTENT_CLASSIFICATION_KEYWORDS_SUBSTR = [
    "klassific",
    "classification",
    "kategori",
    "annex",
    "bilag",
    "risikoklasse",
    "klassificeres",
    "forbudt",
    "prohibited",
    "højrisiko",
    "high-risk",
    "high risk",
]

# Scope/applicability keywords - used for both intent detection AND citation expansion.
# Single source of truth for scope article injection.
_INTENT_SCOPE_KEYWORDS_STRONG_SUBSTR = [
    # Danish
    "omfattet",
    "omfattes",
    "omfatter",
    "anvendelsesområde",
    "finder anvendelse",
    "falder ind under",
    "falder under",
    "falder det under",
    "ind under",
    "gælder for",
    "gælder forordningen",
    # English
    "scope",
    "applicability",
    "applies",
    "shall apply",
    "covered by",
    "subject to",
    "within the scope",
]

# Enforcement keywords matched exactly (not as substring).
# "forbud" must NOT be substring-matched to avoid matching "forbudt" (classification).
_INTENT_ENFORCEMENT_KEYWORDS_EXACT = ["forbud"]

# Requirements intent (weak): indicators that need verb cues to confirm intent.
_INTENT_REQUIREMENTS_KEYWORDS_WEAK_SUBSTR = [
    "foranstaltninger",
    "tekniske og organisatoriske",
    "procedure",
    "politik",
    "kontrol",
    "governance",
    "frister",
    "dokumentere",
    "logning",
    "rapportering",
]

# Verb cues that confirm requirements intent when combined with weak indicators.
_INTENT_REQUIREMENTS_KEYWORDS_VERBS = [
    "krav",
    "kræver",
    "kræves",
    "skal",
    "must",
    "should",
    "hvordan",
    "overhold",
    "efterlev",
    "implement",
]

# Definitions intent: queries about legal terminology.
_INTENT_DEFINITIONS_KEYWORDS_SUBSTR = [
    "definitioner",
    "definitions", 
    "for the purposes of this regulation",
    "i denne forordning forstås",
    "means",
    "begreber",
    "definition",
]

# ---------------------------------------------------------------------------
# Normative claim detection
# ---------------------------------------------------------------------------

_NORMATIVE_SENTENCE_TOKEN_RE = re.compile(
    r"(?i)\b("  # word-boundary tokens
    r"SKAL|B[ØO]R|M[ÅA]\s*IKKE|MAA\s+IKKE|KR[ÆA]VER|KRAEVER|FORPLIGTET|P[ÅA]KR[ÆA]VET|PLIGT|"
    r"MUST|SHALL|SHOULD|REQUIRED|MAY\s*NOT"
    r")\b"
)

# Alias for backwards compatibility.
_NORMATIVE_CLAIM_RE = _NORMATIVE_SENTENCE_TOKEN_RE


def contains_normative_claim(text: str) -> bool:
    """Return True if text contains normative obligation keywords.
    
    Conservative matcher for standalone obligation keywords.
    Used to determine if claims require article-level support.
    """
    if not text:
        return False
    return bool(_NORMATIVE_CLAIM_RE.search(text))
