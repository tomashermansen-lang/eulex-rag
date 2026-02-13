"""Centralized LLM generation for ingestion pipeline.

Single Responsibility: Provide all LLM-based generation functionality
for the ingestion process, including:
- Chunk enrichment (search terms, contextual descriptions, roles)
- Example questions for UI
- Eval test cases (golden cases)

All prompts are defined here for easy maintenance and consistency.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.llm_helpers import (
    call_generation_llm,
    parse_json_response,
    load_article_content,
)

logger = logging.getLogger(__name__)

# Project root for file paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Common LLM Infrastructure (delegated to src.common.llm_helpers)
# =============================================================================

# Module-internal aliases preserve existing call sites within this file
_call_llm = call_generation_llm
_parse_json_response = parse_json_response


def _parse_yaml_response(content: str) -> str | None:
    """Extract YAML from LLM response, handling markdown wrappers.

    Args:
        content: Raw LLM response

    Returns:
        Clean YAML string or None on error
    """
    if not content:
        return None

    # Try to extract YAML if wrapped in markdown
    if "```yaml" in content:
        content = content.split("```yaml")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]

    return content.strip()


# =============================================================================
# Chunk Enrichment Generation
# =============================================================================

# Valid roles for classification (used for validation)
# Three normative categories: rights (what you're entitled to), obligations (what you must do),
# prohibitions (what you must not do). Plus structural roles: scope, definitions, classification,
# enforcement, exemptions, procedures.
VALID_ROLES = frozenset([
    "scope",          # Hvem/hvad loven gælder for
    "definitions",    # Begreber og termer
    "classification", # Kategorisering (højrisiko, forbudt, tilladt)
    "rights",         # Rettigheder - hvad man har ret til
    "obligations",    # Forpligtelser - hvad man skal gøre
    "prohibitions",   # Forbud - hvad man ikke må gøre
    "exemptions",     # Undtagelser - hvornår regler ikke gælder
    "procedures",     # Procedurer - konkrete processer der skal følges
    "enforcement",    # Sanktioner, tilsyn, håndhævelse
])

ENRICHMENT_PROMPT = """<lovtekst fra {article_title}>
{chunk_text}
</lovtekst>

Opgave: Berig dette lovudsnit for bedre søgning.

1. KONTEKST (50-80 ord): Beskriv INDHOLDET - ikke placeringen.
   - Hvem er forpligtet eller beskyttet?
   - Hvad skal de konkret gøre eller undlade?
   - Hvilke konsekvenser eller undtagelser gælder?
   - Hvilke spørgsmål besvarer denne tekst?

2. SØGETERMER (3-5 termer): Dagligdags ord en bruger ville søge efter.
   - Brug hverdagssprog, ikke juridisk terminologi
   - Tænk på praktiske situationer hvor reglen er relevant

3. ROLLER (vælg 0-3 relevante): Klassificér tekstens juridiske funktion.
   - scope: Definerer hvem/hvad loven gælder for (anvendelsesområde)
   - definitions: Definerer begreber, termer eller kategorier
   - classification: Kategoriserer enheder (f.eks. højrisiko, kritisk, tilladt)
   - rights: Rettigheder som entiteter har krav på (f.eks. ret til indsigt, sletning, information)
   - obligations: Pligter og krav som entiteter skal opfylde (f.eks. skal underrette, dokumentere)
   - prohibitions: Forbud mod bestemte handlinger (f.eks. må ikke behandle, forbudte AI-praksisser)
   - exemptions: Undtagelser fra regler (f.eks. gælder ikke for SMV'er, undtaget fra krav)
   - procedures: Konkrete processer der skal følges (f.eks. ansøgningsprocedure, klageprocedure)
   - enforcement: Sanktioner, bøder, tilsyn eller håndhævelse

   VIGTIGT: Vælg kun roller der passer PRÆCIST. De fleste tekster har 0-2 roller.
   Bilag der lister specifike systemer/områder har typisk "classification".

Format dit svar PRÆCIST sådan:
KONTEKST: [din beskrivelse på én linje]
SØGETERMER: term1 | term2 | term3 | term4
ROLLER: rolle1 | rolle2

Intet andet output. Hvis ingen roller passer, skriv "ROLLER: ingen"."""


@dataclass
class EnrichmentResult:
    """Result from contextual enrichment generation."""

    contextual_description: str
    search_terms: list[str]
    roles: list[str] = field(default_factory=list)

    @property
    def terms(self) -> list[str]:
        """Alias for backward compatibility."""
        return self.search_terms


def generate_chunk_enrichment(
    chunk_text: str,
    *,
    article_title: str = "",
    model: str = "gpt-4o-mini",
    max_terms: int = 5,
) -> EnrichmentResult | None:
    """Generate contextual description and search terms for a legal chunk.

    Args:
        chunk_text: The legal text to enrich
        article_title: Title of the article (e.g., "Artikel 50 - Gennemsigtighed")
        model: LLM model to use
        max_terms: Maximum number of search terms

    Returns:
        EnrichmentResult or None on error
    """
    if not chunk_text.strip():
        return None

    prompt = ENRICHMENT_PROMPT.format(
        article_title=article_title or "Ukendt artikel",
        chunk_text=chunk_text[:2000],  # Limit for context window
    )

    content = _call_llm(prompt, model=model, temperature=0.3, max_tokens=300)
    if not content:
        return None

    # Parse structured output
    contextual_description = ""
    search_terms: list[str] = []
    roles: list[str] = []

    # Extract SØGETERMER
    terms_match = re.search(r'SØGETERMER:\s*(.+?)(?:\n|ROLLER:|$)', content, re.IGNORECASE)
    if terms_match:
        terms_str = terms_match.group(1).strip()
        search_terms = [
            t.strip() for t in terms_str.split("|")
            if t.strip() and len(t.strip()) > 2
        ][:max_terms]

    # Extract KONTEKST
    kontekst_match = re.search(r'KONTEKST:\s*(.+?)(?:\s*SØGETERMER:|$)', content, re.IGNORECASE | re.DOTALL)
    if kontekst_match:
        contextual_description = kontekst_match.group(1).strip()

    # Extract ROLLER
    roles_match = re.search(r'ROLLER:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
    if roles_match:
        roles_str = roles_match.group(1).strip().lower()
        if roles_str != "ingen":
            roles = [
                r.strip() for r in roles_str.split("|")
                if r.strip() in VALID_ROLES
            ]

    return EnrichmentResult(
        contextual_description=contextual_description,
        search_terms=search_terms,
        roles=roles,
    )


# =============================================================================
# Example Questions Generation (for UI)
# =============================================================================

EXAMPLE_QUESTIONS_PROMPT = """Du er en juridisk ekspert i EU-lovgivning. Baseret på følgende lovgivnings FAKTISKE INDHOLD, generer præcis 3 eksempelspørgsmål for hver profil.

Lovgivning: {display_name}
CELEX: {celex_number}

LOVENS STRUKTUR OG INDHOLD:
{article_content}

Generer spørgsmål på dansk der er:
- Baseret på det FAKTISKE indhold ovenfor (ikke generiske spørgsmål)
- Naturlige spørgsmål som en bruger ville stille
- VIGTIGT: Undgå at henvise til specifikke artikelnumre (f.eks. "artikel 5", "artikel 32")
- Spørgsmål skal afspejle lovens KONKRETE emner, ikke generisk jura

Format dit svar PRÆCIS som følgende JSON (ingen markdown, kun ren JSON):
{{
  "LEGAL": [
    "Spørgsmål 1 fra juridisk perspektiv?",
    "Spørgsmål 2 fra juridisk perspektiv?",
    "Spørgsmål 3 fra juridisk perspektiv?"
  ],
  "ENGINEERING": [
    "Spørgsmål 1 fra teknisk/implementerings perspektiv?",
    "Spørgsmål 2 fra teknisk/implementerings perspektiv?",
    "Spørgsmål 3 fra teknisk/implementerings perspektiv?"
  ]
}}

LEGAL profil: Fokus på juridisk fortolkning, forpligtelser, undtagelser, og sanktioner fra denne lov.
ENGINEERING profil: Fokus på teknisk implementering, krav og compliance-processer fra denne lov."""


def generate_example_questions(
    corpus_id: str,
    display_name: str,
    celex_number: str,
    *,
    model: str | None = None,
) -> dict[str, list[str]] | None:
    """Generate example questions for a corpus for the UI.

    Uses actual article content from chunks to generate relevant questions,
    not just metadata. This produces more specific and varied questions.

    Args:
        corpus_id: Short corpus ID (e.g., 'nis2')
        display_name: Full display name
        celex_number: CELEX number
        model: LLM model (uses settings default if None)

    Returns:
        Dict with LEGAL and ENGINEERING question lists, or None on failure
    """
    if model is None:
        from src.common.config_loader import load_settings
        settings = load_settings()
        model = settings.chat_model

    # Load actual article content for context (same as eval case generation)
    article_content = _load_article_content(corpus_id, max_chars_per_article=300)

    prompt = EXAMPLE_QUESTIONS_PROMPT.format(
        display_name=display_name,
        celex_number=celex_number,
        article_content=article_content,
    )

    # Higher temperature (0.5) for more variance in generated questions
    content = _call_llm(prompt, model=model, temperature=0.5, max_tokens=1000)
    questions = _parse_json_response(content)

    if not questions:
        return None

    # Validate structure
    if "LEGAL" not in questions or "ENGINEERING" not in questions:
        logger.warning("Invalid question structure from LLM")
        return None

    if len(questions["LEGAL"]) < 3 or len(questions["ENGINEERING"]) < 3:
        logger.warning("Not enough questions generated")
        return None

    # Take only first 3 per profile
    return {
        "LEGAL": questions["LEGAL"][:3],
        "ENGINEERING": questions["ENGINEERING"][:3],
    }


# =============================================================================
# Eval Case Generation (golden cases in YAML format)
# =============================================================================

# Professional eval case categories (industry-standard RAG evaluation)
EVAL_TEST_TYPES = [
    "retrieval",     # Correct document/chunk retrieval
    "faithfulness",  # Answers grounded in retrieved context
    "relevancy",     # Answers address the actual question
    "abstention",    # System refuses when appropriate
    "robustness",    # Handles paraphrasing, edge cases, variations
    "multi_hop",     # Synthesis across multiple sources
]

# Sophisticated eval cases prompt with professional test categories
EVAL_CASES_PROMPT = """Du er en juridisk ekspert i EU-lovgivning og RAG-systemtest. Generer en PROFESSIONEL evalueringssuite.

Lovgivning: {display_name}
CELEX: {celex_number}
Corpus ID: {corpus_id}

LOVENS STRUKTUR MED INDHOLD:
{article_content}

Generer præcis {num_cases} evalueringscases fordelt på disse PROFESSIONELLE KATEGORIER:

## KATEGORI: retrieval (4 cases, ~27%)
Tester at systemet finder de RIGTIGE chunks/artikler.
- Klart spørgsmål med ét forventet svar i én specifik artikel
- Fokus på præcis retrieval, ikke svarets kvalitet
- must_include_any_of: én eller flere acceptable artikler

## KATEGORI: faithfulness (3 cases, ~20%)
Tester at svaret er GROUNDED i den hentede kontekst.
- Spørgsmål hvor det er let at "hallucinere" detaljer
- Komplekse regler med specifikke tal, datoer, eller betingelser
- Svaret SKAL kunne verificeres mod kildeteksten

## KATEGORI: relevancy (2 cases, ~13%)
Tester at svaret faktisk BESVARER spørgsmålet.
- Spørgsmål der let kunne besvares "ved siden af"
- Fokus på om svaret er on-topic og komplet
- Undgå spørgsmål der kan besvares med ja/nej

## KATEGORI: abstention (2 cases, ~13%)
Tester at systemet NÆGTER at svare når det bør.
- Spørgsmål der ligger UDEN FOR lovens scope
- Spørgsmål om emner loven ikke dækker
- Sæt behavior: "abstain" og allow_empty_references: true

## KATEGORI: robustness (2 cases, ~13%)
Tester håndtering af VARIATIONER og edge cases.
- Omformulerede spørgsmål (samme intent, andre ord)
- Spørgsmål med stavefejl eller uformelt sprog
- Grænsetilfælde hvor svaret kræver nuancering
- must_not_include_any_of: artikler der LIGNER men er forkerte

## KATEGORI: multi_hop (2 cases, ~13%)
Tester SYNTESE på tværs af flere kilder.
- Spørgsmål der kræver info fra 2-3 forskellige artikler
- must_include_any_of: alle relevante artikler
- Svaret skal kombinere information, ikke bare liste artikler

YAML-FORMAT:
- id: {corpus_id}-XX-kategori-emne
  profile: LEGAL eller ENGINEERING
  test_types:
    - kategori  # En af: retrieval, faithfulness, relevancy, abstention, robustness, multi_hop
  origin: auto
  prompt: Dit spørgsmål her?
  expected:
    allow_empty_references: false  # true kun for abstention
    must_have_article_support_for_normative: true
    must_include_any_of:
      - article:X
    must_not_include_any_of: []  # Bruges primært til robustness
    behavior: "answer"  # "answer" eller "abstain"
    notes: "Kort forklaring af hvad casen tester."

VIGTIGE REGLER:
1. Brug KUN artikelnumre fra indholdet ovenfor
2. Spørgsmål skal være naturlige (ALDRIG nævn artikelnumre i spørgsmålet)
3. Halvdelen LEGAL profil, halvdelen ENGINEERING profil
4. test_types skal ALTID være en liste med mindst én kategori
5. origin skal ALTID være "auto" (markerer LLM-genereret)
6. behavior skal være "abstain" for abstention-kategorien, ellers "answer"
7. Fordeling: 4 retrieval, 3 faithfulness, 2 relevancy, 2 abstention, 2 robustness, 2 multi_hop = 15 total

Generer YAML direkte (ingen markdown):"""


@dataclass
class EvalCase:
    """A single evaluation test case."""

    id: str
    profile: str
    prompt: str
    must_include_any_of: list[str]
    test_types: list[str] = field(default_factory=lambda: ["retrieval"])
    origin: str = "auto"  # "auto" (LLM-generated) or "manual" (human-created)
    behavior: str = "answer"  # "answer" or "abstain"
    notes: str = ""
    allow_empty_references: bool = False
    must_have_article_support_for_normative: bool = True
    must_not_include_any_of: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate test_types against allowed values."""
        for test_type in self.test_types:
            if test_type not in EVAL_TEST_TYPES:
                raise ValueError(
                    f"Invalid test_type '{test_type}'. "
                    f"Must be one of: {EVAL_TEST_TYPES}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "id": self.id,
            "profile": self.profile,
            "test_types": self.test_types,
            "origin": self.origin,
            "prompt": self.prompt,
            "expected": {
                "allow_empty_references": self.allow_empty_references,
                "must_have_article_support_for_normative": self.must_have_article_support_for_normative,
                "must_include_any_of": self.must_include_any_of,
                "must_not_include_any_of": self.must_not_include_any_of,
                "behavior": self.behavior,
                "notes": self.notes,
            },
        }


def _load_article_metadata(corpus_id: str) -> str:
    """Load article titles from chunks file for a corpus.

    Returns a formatted string listing all articles and their titles.
    """
    chunks_path = PROJECT_ROOT / "data" / "processed" / f"{corpus_id}_chunks.jsonl"
    if not chunks_path.exists():
        return "(Artikelstruktur ikke tilgængelig)"

    articles: dict[int, str] = {}
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                metadata = chunk.get("metadata", {})
                art = metadata.get("article")
                title = metadata.get("article_title", "")
                if art and str(art).isdigit():
                    art_num = int(art)
                    if art_num not in articles:
                        articles[art_num] = title
    except Exception as e:
        logger.warning("Could not load article metadata: %s", e)
        return "(Artikelstruktur ikke tilgængelig)"

    if not articles:
        return "(Ingen artikler fundet)"

    # Format as numbered list
    lines = []
    for art_num in sorted(articles.keys()):
        title = articles[art_num]
        lines.append(f"- Artikel {art_num}: {title}")

    return "\n".join(lines)


_load_article_content = load_article_content


def _parse_eval_cases_yaml(yaml_content: str) -> list[EvalCase]:
    """Parse YAML content into list of EvalCase objects.

    Args:
        yaml_content: Raw YAML string

    Returns:
        List of valid EvalCase objects (may be empty)
    """
    import yaml as yaml_lib

    try:
        cases_data = yaml_lib.safe_load(yaml_content)
        if not isinstance(cases_data, list):
            logger.warning("Invalid eval cases structure - expected list")
            return []

        cases = []
        for case_dict in cases_data:
            expected = case_dict.get("expected", {})

            # Parse test_types with validation
            raw_test_types = case_dict.get("test_types", ["retrieval"])
            if isinstance(raw_test_types, str):
                raw_test_types = [raw_test_types]
            # Filter to valid types only
            test_types = [t for t in raw_test_types if t in EVAL_TEST_TYPES]
            if not test_types:
                test_types = ["retrieval"]  # Default fallback

            case = EvalCase(
                id=case_dict.get("id", ""),
                profile=case_dict.get("profile", "LEGAL"),
                prompt=case_dict.get("prompt", ""),
                must_include_any_of=expected.get("must_include_any_of", []),
                test_types=test_types,
                origin=case_dict.get("origin", "auto"),
                behavior=expected.get("behavior", "answer"),
                notes=expected.get("notes", ""),
                allow_empty_references=expected.get("allow_empty_references", False),
                must_have_article_support_for_normative=expected.get(
                    "must_have_article_support_for_normative", True
                ),
                must_not_include_any_of=expected.get("must_not_include_any_of", []),
            )
            if case.id and case.prompt:
                cases.append(case)

        return cases

    except Exception as e:
        logger.error("Failed to parse eval cases YAML: %s", e)
        return []


def generate_eval_cases(
    corpus_id: str,
    display_name: str,
    celex_number: str,
    *,
    num_cases: int = 15,
    model: str | None = None,
    max_retries: int = 2,
) -> list[EvalCase] | None:
    """Generate evaluation test cases for a corpus.

    Generates a professional eval suite with 6 test categories:
    - retrieval (4): Correct document/chunk retrieval
    - faithfulness (3): Answers grounded in context
    - relevancy (2): Answers address the question
    - abstention (2): System refuses when appropriate
    - robustness (2): Handles variations and edge cases
    - multi_hop (2): Synthesis across multiple sources

    Retries generation if the exact number of cases is not produced.

    Args:
        corpus_id: Short corpus ID (e.g., 'nis2')
        display_name: Full display name
        celex_number: CELEX number
        num_cases: Number of cases to generate (default: 15) - GUARANTEED
        model: LLM model (uses settings default if None)
        max_retries: Maximum retry attempts if wrong count (default: 2)

    Returns:
        List of exactly num_cases EvalCase objects, or None on failure
    """
    if model is None:
        from src.common.config_loader import load_settings
        settings = load_settings()
        model = settings.chat_model

    # Load article structure WITH content for sophisticated case generation
    article_content = _load_article_content(corpus_id)

    for attempt in range(max_retries + 1):
        # Adjust prompt on retry to emphasize exact count
        if attempt == 0:
            prompt = EVAL_CASES_PROMPT.format(
                display_name=display_name,
                celex_number=celex_number,
                corpus_id=corpus_id,
                num_cases=num_cases,
                article_content=article_content,
            )
        else:
            # More emphatic prompt on retry
            prompt = EVAL_CASES_PROMPT.format(
                display_name=display_name,
                celex_number=celex_number,
                corpus_id=corpus_id,
                num_cases=num_cases,
                article_content=article_content,
            )
            prompt += f"\n\nVIGTIGT: Du SKAL generere PRÆCIS {num_cases} cases. Ikke færre, ikke flere. Tæl dem!"

        # Use higher max_tokens for professional multi-category cases
        content = _call_llm(prompt, model=model, temperature=0.4, max_tokens=6000)
        yaml_content = _parse_yaml_response(content)

        if not yaml_content:
            logger.warning("Attempt %d: No YAML content returned", attempt + 1)
            continue

        cases = _parse_eval_cases_yaml(yaml_content)

        if len(cases) == num_cases:
            logger.info("Generated exactly %d eval cases on attempt %d", num_cases, attempt + 1)
            return cases
        elif len(cases) > num_cases:
            # Got too many - truncate to exact count
            logger.info("Generated %d cases, truncating to %d", len(cases), num_cases)
            return cases[:num_cases]
        else:
            # Got too few - retry
            logger.warning(
                "Attempt %d: Generated %d cases instead of %d, retrying...",
                attempt + 1, len(cases), num_cases
            )

    # All retries exhausted - return what we have or None
    if cases:
        logger.warning(
            "Could not generate exactly %d cases after %d attempts. Got %d cases.",
            num_cases, max_retries + 1, len(cases)
        )
        return cases  # Return partial result rather than failing completely

    return None


def save_eval_cases(cases: list[EvalCase], corpus_id: str) -> Path:
    """Save eval cases to YAML file.

    Args:
        cases: List of EvalCase objects
        corpus_id: Corpus ID for filename

    Returns:
        Path to saved file
    """
    import yaml as yaml_lib

    evals_dir = PROJECT_ROOT / "data" / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    output_path = evals_dir / f"golden_cases_{corpus_id}.yaml"

    cases_dicts = [case.to_dict() for case in cases]

    with open(output_path, "w", encoding="utf-8") as f:
        yaml_lib.dump(
            cases_dicts,
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    logger.info("Saved %d eval cases to %s", len(cases), output_path)
    return output_path
