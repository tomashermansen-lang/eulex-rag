"""API routes for configuration endpoints.

Single Responsibility: Handle HTTP requests for app configuration.
"""

from __future__ import annotations

from fastapi import APIRouter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import CorporaResponse, CorpusInfo, ExamplesResponse, HealthResponse
import services

router = APIRouter(tags=["config"])


@router.get("/corpora", response_model=CorporaResponse)
async def get_corpora() -> CorporaResponse:
    """Get list of available corpora."""
    corpora_list = services.get_corpora()

    return CorporaResponse(
        corpora=[
            CorpusInfo(
                id=c["id"],
                name=c["name"],
                fullname=c.get("fullname"),
                source_url=c.get("source_url"),
                celex_number=c.get("celex_number"),
                eurovoc_labels=c.get("eurovoc_labels"),
            )
            for c in corpora_list
        ]
    )


@router.get("/examples", response_model=ExamplesResponse)
async def get_examples() -> ExamplesResponse:
    """Get example questions for all corpora and profiles."""
    return ExamplesResponse(examples=services.get_examples())


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")
