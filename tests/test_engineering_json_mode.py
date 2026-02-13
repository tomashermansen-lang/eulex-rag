import json

import pytest
import yaml

from src.engine.rag import RAGEngine
from src.common.config_loader import clear_config_cache, RankingWeights


@pytest.fixture(autouse=True)
def _isolate_config(tmp_path, monkeypatch):
    # Keep JSON-mode tests deterministic by not depending on the repo's default config
    clear_config_cache()
    yield


def _make_engine_with_two_citable_refs() -> RAGEngine:
    engine = RAGEngine.__new__(RAGEngine)
    engine.corpus_id = "ai-act"
    engine.top_k = 3
    engine.enable_toc_rerank = False
    engine.ranking_weights = RankingWeights()

    engine.max_distance = None

    
    
    engine._should_abstain = lambda *a, **k: None  # type: ignore[attr-defined]

    engine.collection = object()
    engine.collection_name = "ai-act_documents"

    def fake_query_with_where(question, k=None, *, where=None):  # noqa: ARG001
        engine._last_retrieved_ids = ["cid-1", "cid-2"]
        engine._last_retrieved_metadatas = [
            {
                "source": "AI Act",
                "article": "12",
                "heading_path": "Record-keeping",
                "location_id": "article:12",
            },
            {
                "source": "AI Act",
                "article": "14",
                "heading_path": "Human oversight",
                "location_id": "article:14",
            },
        ]
        engine._last_distances = [0.1, 0.11]
        return [("doc 1", engine._last_retrieved_metadatas[0]), ("doc 2", engine._last_retrieved_metadatas[1])]

    engine.query_with_where = fake_query_with_where  # type: ignore[attr-defined]
    return engine


def test_json_mode_unknown_field_triggers_repair_then_success(monkeypatch):
    monkeypatch.setenv("ENGINEERING_JSON_MODE", "1")

    engine = _make_engine_with_two_citable_refs()

    calls: list[str] = []

    bad_with_unknown = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet i denne kontekst.", "citations": [1]},
        "obligations": [{"title": "Artikel 12", "text": "Der gælder record-keeping.", "citations": [1]}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": [1]}],
        "open_questions": [],
        "unknown": "nope",
    }

    good = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet i denne kontekst.", "citations": [1]},
        "obligations": [{"title": "Artikel 12", "text": "Der gælder record-keeping.", "citations": [1]}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": [1]}],
        "open_questions": [],
    }

    def fake_call_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(bad_with_unknown, ensure_ascii=False)
        return json.dumps(good, ensure_ascii=False)

    engine._call_llm = fake_call_llm  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(
        engine,
        "Hvilke krav gælder til record-keeping?",
        user_profile="ENGINEERING",
        contract_min_citations=1,
    )

    assert len(calls) == 2
    run = dict(payload.get("run") or {})
    assert run.get("engineering_json_mode") is True
    assert run.get("repair_retry_performed") is True
    assert run.get("enrich_retry_performed") is False
    assert run.get("json_parse_ok") is True
    assert int(run.get("llm_calls_count") or 0) == 2
    assert "[1]" in str(payload.get("answer") or "")


def test_json_mode_enrich_must_not_increase_bullet_counts(monkeypatch):
    monkeypatch.setenv("ENGINEERING_JSON_MODE", "1")

    engine = _make_engine_with_two_citable_refs()

    calls: list[str] = []

    base = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet.", "citations": [1]},
        "obligations": [{"title": "Artikel 12", "text": "Record-keeping gælder.", "citations": [1]}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": [1]}],
        "open_questions": [],
    }

    # Enrich tries to add a new obligation bullet -> should fail-closed deterministically.
    enrich_bad = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet.", "citations": [1, 2]},
        "obligations": [
            {"title": "Artikel 12", "text": "Record-keeping gælder.", "citations": [1]},
            {"title": "Artikel 14", "text": "Human oversight gælder.", "citations": [2]},
        ],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": [1, 2]}],
        "open_questions": [],
    }

    def fake_call_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(base, ensure_ascii=False)
        return json.dumps(enrich_bad, ensure_ascii=False)

    engine._call_llm = fake_call_llm  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(
        engine,
        "Hvilke krav gælder til record-keeping?",
        user_profile="ENGINEERING",
        contract_min_citations=2,
    )

    assert len(calls) == 2
    run = dict(payload.get("run") or {})
    assert run.get("engineering_json_mode") is True
    assert run.get("enrich_retry_performed") is True
    assert str(payload.get("answer") or "").strip() == "MISSING_REF"
    assert run.get("fail_reason") == "schema_fail"


def test_json_mode_retry_budget_max_3_calls_worst_case(monkeypatch):
    monkeypatch.setenv("ENGINEERING_JSON_MODE", "1")

    engine = _make_engine_with_two_citable_refs()

    calls: list[str] = []

    # 1) invalid JSON -> triggers repair
    invalid_json = "{not valid json"

    # 2) repair returns valid schema but insufficient unique citations for min=2 -> triggers enrich
    repaired = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet.", "citations": [1]},
        "obligations": [{"title": "Artikel 12", "text": "Record-keeping gælder.", "citations": [1]}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": [1]}],
        "open_questions": [],
    }

    # 3) enrich returns same bullet counts but adds citations to reach min=2
    enriched = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet.", "citations": [1, 2]},
        "obligations": [{"title": "Artikel 12", "text": "Record-keeping gælder.", "citations": [1, 2]}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": [1, 2]}],
        "open_questions": [],
    }

    def fake_call_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return invalid_json
        if len(calls) == 2:
            return json.dumps(repaired, ensure_ascii=False)
        return json.dumps(enriched, ensure_ascii=False)

    engine._call_llm = fake_call_llm  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(
        engine,
        "Hvilke krav gælder til record-keeping?",
        user_profile="ENGINEERING",
        contract_min_citations=2,
    )

    assert len(calls) == 3
    run = dict(payload.get("run") or {})
    assert int(run.get("llm_calls_count") or 0) == 3
    assert run.get("repair_retry_performed") is True
    assert run.get("enrich_retry_performed") is True
    assert str(payload.get("answer") or "").strip() != "MISSING_REF"
    assert "[1]" in str(payload.get("answer") or "")
    assert "[2]" in str(payload.get("answer") or "")


def test_json_mode_valid_json_with_citations_must_pass_downstream_gates(monkeypatch):
    monkeypatch.setenv("ENGINEERING_JSON_MODE", "1")

    engine = _make_engine_with_two_citable_refs()

    calls: list[str] = []

    good = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet i denne kontekst.", "citations": [1, 2]},
        "obligations": [{"title": "Artikel 12", "text": "Der gælder record-keeping.", "citations": [1]}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": [2]}],
        "open_questions": [],
    }

    def fake_call_llm(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(good, ensure_ascii=False)

    engine._call_llm = fake_call_llm  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(
        engine,
        "Hvilke konkrete krav gælder til record-keeping?",
        user_profile="ENGINEERING",
        contract_min_citations=2,
    )

    ans = str(payload.get("answer") or "").strip()
    assert ans != "MISSING_REF"
    assert "[1]" in ans
    assert "[2]" in ans

    run = dict(payload.get("run") or {})
    assert run.get("engineering_json_mode") is True
    assert int(run.get("llm_calls_count") or 0) == 1
    assert run.get("json_parse_ok") is True


def test_json_mode_schema_only_fallback_then_enrich_still_empty_fails_closed(monkeypatch):
    monkeypatch.setenv("ENGINEERING_JSON_MODE", "1")

    engine = _make_engine_with_two_citable_refs()

    calls: list[str] = []

    # Base JSON is schema-correct but missing required citations -> triggers strict normativ_no_citation.
    base_missing = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet.", "citations": []},
        "obligations": [{"title": "Artikel 12", "text": "Record-keeping gælder.", "citations": []}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": []}],
        "open_questions": [],
    }

    # Enrich fails to add citations (still empty) -> must fail with the hardened reason.
    enrich_still_missing = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet.", "citations": []},
        "obligations": [{"title": "Artikel 12", "text": "Record-keeping gælder.", "citations": []}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": []}],
        "open_questions": [],
    }

    def fake_call_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(base_missing, ensure_ascii=False)
        return json.dumps(enrich_still_missing, ensure_ascii=False)

    engine._call_llm = fake_call_llm  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(
        engine,
        "Hvilke krav gælder til record-keeping?",
        user_profile="ENGINEERING",
        contract_min_citations=2,
    )

    assert len(calls) == 2
    assert str(payload.get("answer") or "").strip() == "MISSING_REF"

    run = dict(payload.get("run") or {})
    assert run.get("engineering_json_mode") is True
    assert run.get("strict_validation_failed_code") == "normativ_no_citation"
    assert run.get("schema_only_fallback_used") is True
    assert run.get("enrich_retry_performed") is True
    assert run.get("enrich_success") is False
    assert run.get("final_fail_reason") == "NORMATIV_NO_CITATION_AFTER_ENRICH"
    assert run.get("fail_reason") == "NORMATIV_NO_CITATION_AFTER_ENRICH"
    assert run.get("fail_reason") != "MIN_CITATIONS_NOT_MET_DOWNSTREAM"


def test_json_mode_schema_only_fallback_then_enrich_success(monkeypatch):
    monkeypatch.setenv("ENGINEERING_JSON_MODE", "1")

    engine = _make_engine_with_two_citable_refs()

    calls: list[str] = []

    base_missing = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet.", "citations": []},
        "obligations": [{"title": "Artikel 12", "text": "Record-keeping gælder.", "citations": []}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": []}],
        "open_questions": [],
    }

    enriched = {
        "classification": {"status": "JA", "text": "Systemet vurderes omfattet.", "citations": [1, 2]},
        "obligations": [{"title": "Artikel 12", "text": "Record-keeping gælder.", "citations": [1, 2]}],
        "system_requirements": [{"level": "SKAL", "text": "Implementere logging.", "citations": [1, 2]}],
        "open_questions": [],
    }

    def fake_call_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(base_missing, ensure_ascii=False)
        return json.dumps(enriched, ensure_ascii=False)

    engine._call_llm = fake_call_llm  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(
        engine,
        "Hvilke krav gælder til record-keeping?",
        user_profile="ENGINEERING",
        contract_min_citations=2,
    )

    assert len(calls) == 2

    ans = str(payload.get("answer") or "").strip()
    assert ans != "MISSING_REF"
    assert "[1]" in ans
    assert "[2]" in ans

    run = dict(payload.get("run") or {})
    assert run.get("engineering_json_mode") is True
    assert run.get("strict_validation_failed_code") == "normativ_no_citation"
    assert run.get("schema_only_fallback_used") is True
    assert run.get("enrich_retry_performed") is True
    assert run.get("enrich_success") is True
    assert run.get("fail_reason") is None


def test_non_json_mode_behavior_unchanged_for_plain_text_llm(monkeypatch):
    monkeypatch.delenv("ENGINEERING_JSON_MODE", raising=False)

    engine = _make_engine_with_two_citable_refs()

    calls: list[str] = []

    def fake_call_llm(prompt: str) -> str:
        calls.append(prompt)
        return "1. Klassifikation og betingelser\n- JA.\n\n2. Relevante juridiske forpligtelser\n- Artikel 12.\n\n3. Konkrete systemkrav\n- SKAL: Implementere logging [1] [2]\n\n4. Åbne spørgsmål / risici\n- (ingen)"

    engine._call_llm = fake_call_llm  # type: ignore[attr-defined]

    payload = RAGEngine.answer_structured(
        engine,
        "Hvilke konkrete krav gælder til record-keeping?",
        user_profile="ENGINEERING",
        contract_min_citations=2,
    )
    assert len(calls) == 1
    ans = str(payload.get("answer") or "").strip()
    assert ans != "MISSING_REF"
    assert "[1]" in ans

