"""Corpus naming utilities.

Single Responsibility: Parse CELEX numbers and generate structured corpus IDs.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


def _load_naming_config() -> dict:
    """Load naming conventions from config file."""
    config_path = Path(__file__).parent.parent.parent / "config" / "naming_conventions.yaml"
    if not config_path.exists():
        # Fallback defaults
        return {
            "celex_type_mapping": {"L": "dir", "R": "reg", "D": "dec"},
            "corpus_id_format": "{known_name}-{type_code}-{year}-{number}",
            "special_type_codes": {
                "implementing_regulation": "cir",
                "delegated_regulation": "dr"
            }
        }

    with open(config_path) as f:
        return yaml.safe_load(f)


# CELEX format: SYYYYTNNN where S=sector (1 digit), YYYY=year, T=type, NNN=number
CELEX_PATTERN = re.compile(r"^(\d)(\d{4})([A-Z])(\d+)$")


def parse_celex(celex_number: str) -> dict:
    """Parse a CELEX number into its components.

    Args:
        celex_number: CELEX identifier (e.g., "32022L2555")

    Returns:
        Dict with keys: sector, year, type_char, number, type_code

    Raises:
        ValueError: If CELEX format is invalid
    """
    celex_number = celex_number.strip().upper()
    match = CELEX_PATTERN.match(celex_number)

    if not match:
        raise ValueError(f"Invalid CELEX format: {celex_number}")

    sector, year, type_char, number = match.groups()

    config = _load_naming_config()
    type_mapping = config.get("celex_type_mapping", {})
    type_code = type_mapping.get(type_char, type_char.lower())

    return {
        "sector": int(sector),
        "year": int(year),
        "type_char": type_char,
        "number": int(number),
        "type_code": type_code,
    }


def generate_corpus_id(
    known_name: str,
    celex_number: str,
    is_implementing: bool = False,
    is_delegated: bool = False,
) -> str:
    """Generate a structured corpus_id from components.

    Args:
        known_name: Publicly known abbreviation (e.g., "NIS2", "GDPR")
        celex_number: CELEX identifier
        is_implementing: True if this is an implementing regulation
        is_delegated: True if this is a delegated regulation

    Returns:
        Structured corpus_id (e.g., "nis2-dir-2022-2555")
    """
    parsed = parse_celex(celex_number)
    config = _load_naming_config()

    # Determine type code
    type_code = parsed["type_code"]
    if is_implementing:
        type_code = config.get("special_type_codes", {}).get("implementing_regulation", "cir")
    elif is_delegated:
        type_code = config.get("special_type_codes", {}).get("delegated_regulation", "dr")

    # Normalize known_name: lowercase, replace spaces with hyphens
    normalized_name = known_name.lower().replace(" ", "-")
    # Remove any characters that aren't lowercase letters, numbers, or hyphens
    normalized_name = re.sub(r"[^a-z0-9-]", "", normalized_name)

    # Format: {known_name}-{type_code}-{year}-{number}
    corpus_id = f"{normalized_name}-{type_code}-{parsed['year']}-{parsed['number']}"

    return corpus_id


def get_type_code_from_celex(celex_number: str) -> str:
    """Get the type code (dir, reg, dec) from a CELEX number.

    Args:
        celex_number: CELEX identifier

    Returns:
        Type code string
    """
    parsed = parse_celex(celex_number)
    return parsed["type_code"]
