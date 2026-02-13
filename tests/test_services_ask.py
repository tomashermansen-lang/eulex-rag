from __future__ import annotations

from dataclasses import asdict

from src.services.ask import ask


class FakeEngine:
    top_k = 5
    corpus_id = "ai_act"

    def answer_structured(self, question: str):
        assert question
        return {
            "answer": "Svar tekst",
            "references": ["[1] Ref"],
            "retrieval": {
                "distances": [0.12, 0.34],
                "query_collection": "documents",
                "query_where": None,
                "retrieved_ids": ["id1", "id2"],
                "retrieved_metadatas": [{"a": 1}, {"b": 2}],
            },
        }


def test_ask_returns_stable_contract():
    result = ask(question="Test?", law="ai_act", engine=FakeEngine())

    assert result.answer
    assert isinstance(result.references, list)
    assert "best_distance" in result.retrieval_metrics
    assert result.retrieval_metrics["best_distance"] == 0.12
    assert result.retrieval_metrics["used_law"] == "ai_act"
    assert result.retrieval_metrics["used_collection"] == "documents"
    assert result.retrieval_metrics["retrieved_ids"] == ["id1", "id2"]
