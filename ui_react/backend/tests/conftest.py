"""Pytest fixtures for backend tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient

# Add backend directory to sys.path so relative imports work
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


@dataclass(frozen=True)
class MockAskResult:
    """Mock AskResult for testing."""

    answer: str = "Test answer with [1] citation."
    references: list[str] = None
    references_structured: list[dict[str, Any]] = None
    retrieval_metrics: dict[str, Any] = None

    def __post_init__(self):
        # Use object.__setattr__ for frozen dataclass
        if self.references is None:
            object.__setattr__(self, "references", [])
        if self.references_structured is None:
            object.__setattr__(self, "references_structured", [
                {
                    "idx": 1,
                    "display": "Article 5 - Prohibited AI practices",
                    "chunk_text": "The following AI practices shall be prohibited...",
                    "corpus_id": "ai-act",
                    "article": "5",
                }
            ])
        if self.retrieval_metrics is None:
            object.__setattr__(self, "retrieval_metrics", {
                "best_distance": 0.25,
                "query": "test",
            })


@pytest.fixture
def mock_ask_result():
    """Provide a mock AskResult."""
    return MockAskResult()


@pytest.fixture
def mock_services(monkeypatch, mock_ask_result):
    """Mock the services module.

    Note: We must patch both 'ui_react.backend.services' AND the 'services' module
    that routes import directly (via sys.path manipulation in the route files).
    """
    from ui_react.backend import services

    # Also import the services module as routes see it
    import importlib
    import sys
    backend_path = str(Path(__file__).parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    # Import fresh to ensure we get the same module instance
    if 'services' in sys.modules:
        services_direct = sys.modules['services']
    else:
        services_direct = importlib.import_module('services')

    # Mock get_answer on both module references
    mock_get_answer = lambda **kwargs: mock_ask_result
    monkeypatch.setattr(services, "get_answer", mock_get_answer)
    monkeypatch.setattr(services_direct, "get_answer", mock_get_answer)

    # Mock stream_answer
    def mock_stream(**kwargs):
        yield "Test "
        yield "answer "
        yield "streaming."
        yield mock_ask_result

    monkeypatch.setattr(services, "stream_answer", mock_stream)
    monkeypatch.setattr(services_direct, "stream_answer", mock_stream)

    # Mock get_corpora
    mock_get_corpora = lambda: [
        {"id": "ai-act", "name": "AI ACT", "source_url": "https://example.com/ai-act"},
        {"id": "gdpr", "name": "GDPR", "source_url": "https://example.com/gdpr"},
    ]
    monkeypatch.setattr(services, "get_corpora", mock_get_corpora)
    monkeypatch.setattr(services_direct, "get_corpora", mock_get_corpora)

    # Mock get_examples
    mock_get_examples = lambda: {
        "ai-act": {
            "LEGAL": ["Question 1?", "Question 2?"],
            "ENGINEERING": ["Tech question 1?"],
        }
    }
    monkeypatch.setattr(services, "get_examples", mock_get_examples)
    monkeypatch.setattr(services_direct, "get_examples", mock_get_examples)

    return services


@pytest.fixture
def client(mock_services):
    """Provide a test client with mocked services."""
    from ui_react.backend.main import app

    return TestClient(app)
