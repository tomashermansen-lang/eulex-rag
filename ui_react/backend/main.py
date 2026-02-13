"""FastAPI application entry point.

Single Responsibility: Configure and run the FastAPI application.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes import ask, config, admin, eval, eval_metrics
from routes.eval_cross_law import create_router as create_cross_law_router, CrossLawEvalService

# Application metadata
APP_TITLE = "EuLex Legal Assistant API"
APP_DESCRIPTION = "REST API for the EuLex RAG-based legal assistant"
APP_VERSION = "1.0.0"

# Create FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS configuration
CORS_ORIGINS = [
    "http://localhost:5173",      # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",      # Alternative dev port
    "http://127.0.0.1:3000",
]

# Allow additional origins from environment
extra_origins = os.getenv("CORS_ORIGINS", "")
if extra_origins:
    CORS_ORIGINS.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(ask.router, prefix="/api")
app.include_router(config.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(eval.router, prefix="/api")
app.include_router(eval_metrics.router, prefix="/api/eval/metrics")

# Cross-law eval router (requires service initialization)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = PROJECT_ROOT / "data" / "evals"

# Get valid corpus IDs from settings
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.common.config_loader import load_settings
_settings = load_settings()
_valid_corpus_ids = set(_settings.corpora.keys()) if _settings.corpora else set()

cross_law_service = CrossLawEvalService(EVALS_DIR, _valid_corpus_ids)
app.include_router(create_cross_law_router(cross_law_service), prefix="/api/eval/cross-law")


# Serve static files in production (frontend build)
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="static")


@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/api/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )
