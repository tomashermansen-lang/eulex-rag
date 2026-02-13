import builtins
import json
import sys

import pytest

from src import main
from src.common.corpus_registry import normalize_corpus_id
from src.common.config_loader import load_settings


@pytest.fixture(autouse=True)
def reset_argv():
    original = sys.argv[:]
    yield
    sys.argv = original


def test_run_ingest_mode(monkeypatch, tmp_path):
    calls = {}

    class FakeEngine:
        def __init__(self, docs_path, **kwargs):  # noqa: ARG002
            calls["docs_path"] = docs_path

        def ingest_jsonl(self, path):
            calls["ingest"] = path

    monkeypatch.setattr(main, "RAGEngine", FakeEngine)

    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text("{}", encoding="utf-8")

    settings = load_settings()
    law_id = settings.default_corpus
    registry_path = tmp_path / "corpus_registry.json"
    registry_path.write_text(
        json.dumps(
            {
                normalize_corpus_id(law_id): {
                    "display_name": law_id.upper(),
                    "aliases": [law_id],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sys.argv = [
        "prog",
        "--law",
        law_id,
        "--corpus-registry",
        str(registry_path),
        "--ingest",
        str(chunks),
    ]
    main.run()

    assert calls["ingest"] == str(chunks)
    assert calls["docs_path"].endswith("data/sample_docs")


def test_run_load_docs_and_query(monkeypatch, tmp_path):
    calls = {"loaded": False, "answers": []}

    class FakeEngine:
        def __init__(self, docs_path, **kwargs):  # noqa: ARG002
            calls["docs_path"] = docs_path

        def load_documents(self):
            calls["loaded"] = True

        def answer(self, question):
            calls["answers"].append(question)
            return "Svar"

    inputs = iter(["Hvad siger artikel 10?", "quit"])

    monkeypatch.setattr(main, "RAGEngine", FakeEngine)
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    settings = load_settings()
    law_id = settings.default_corpus
    registry = {
        normalize_corpus_id(law_id): {
            "display_name": law_id.upper(),
            "aliases": [law_id],
        }
    }

    registry_path = tmp_path / "corpus_registry.json"
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")

    sys.argv = [
        "prog",
        "--law",
        law_id,
        "--corpus-registry",
        str(registry_path),
        "--load-sample-docs",
    ]
    main.run()

    assert calls["loaded"] is True
    assert calls["answers"] == ["Hvad siger artikel 10?"]
    assert calls["docs_path"].endswith("data/sample_docs")
