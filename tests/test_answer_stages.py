"""Tests for src/engine/answer_stages.py — stage functions extracted from answer_structured().

Step 7.6: Verifies retrieval stage extraction.
Step 7.7: Verifies evidence gating stage extraction.
"""

from dataclasses import fields

from src.engine.answer_stages import (
    RetrievalStageResult,
    EvidenceGateResult,
    PostGenerationResult,
    execute_answer_retrieval_stage,
    execute_evidence_gate_stage,
    execute_post_generation_stage,
)


class TestRetrievalStageResult:
    """Verify RetrievalStageResult dataclass structure."""

    def test_is_dataclass_with_required_fields(self):
        field_names = [f.name for f in fields(RetrievalStageResult)]
        assert "retrieval_result" in field_names
        assert "hits" in field_names
        assert "distances" in field_names
        assert "synthesis_context" in field_names
        assert "use_multi_corpus" in field_names
        assert "ranking_debug_payload" in field_names


class TestExecuteAnswerRetrievalStage:
    """Verify execute_answer_retrieval_stage function."""

    def test_is_callable(self):
        assert callable(execute_answer_retrieval_stage)

    def test_single_corpus_routes_to_modular_retrieval(self, monkeypatch):
        """Single-corpus scope should call ro.modular_retrieval, not cross_law."""
        from types import SimpleNamespace

        from src.engine.rag_config import _RetrievalResult
        from src.engine.planning import UserProfile

        mock_result = _RetrievalResult(
            hits=[("doc1", {"source": "test", "article": "1"})],
            distances=[0.1],
            retrieved_ids=["id1"],
            retrieved_metas=[{"source": "test", "article": "1"}],
            run_meta_updates={"hybrid_rerank": {"scores": [0.9]}},
            selected_chunks=(),
            total_retrieved=1,
            citable_count=1,
        )

        retriever = SimpleNamespace(
            _last_retrieved_ids=[],
            _last_retrieved_metadatas=[],
        )
        pass_tracker = SimpleNamespace(
            record_pass=lambda **kw: None, get_passes=lambda: []
        )

        # Mock ro.modular_retrieval to return a dict (the raw format)
        mock_dict = {
            "hits": mock_result.hits,
            "distances": mock_result.distances,
            "retrieved_ids": mock_result.retrieved_ids,
            "retrieved_metas": mock_result.retrieved_metas,
            "run_meta_updates": mock_result.run_meta_updates,
            "selected_chunks": mock_result.selected_chunks,
            "total_retrieved": mock_result.total_retrieved,
            "citable_count": mock_result.citable_count,
        }

        calls = []

        def mock_modular(**kwargs):
            calls.append("modular")
            return mock_dict

        monkeypatch.setattr(
            "src.engine.answer_stages.ro.modular_retrieval", mock_modular
        )

        run_meta: dict = {}
        result = execute_answer_retrieval_stage(
            question="test question",
            corpus_scope="single",
            target_corpora=None,
            resolved_profile=UserProfile.ENGINEERING,
            where_for_retrieval=None,
            ctx=SimpleNamespace(corpus_id="ai-act", focus=None),
            run_meta=run_meta,
            effective_plan=SimpleNamespace(where=None),
            pass_tracker=pass_tracker,
            corpus_id="ai-act",
            retriever=retriever,
            chroma=None,
            collection=object(),
            available_corpora_fn=lambda: ["ai-act"],
            resolver_fn=lambda: None,
        )

        assert calls == ["modular"]
        assert isinstance(result, RetrievalStageResult)
        assert result.hits == [("doc1", {"source": "test", "article": "1"})]
        assert result.distances == [0.1]
        assert result.use_multi_corpus is False
        assert result.synthesis_context is None
        assert result.ranking_debug_payload == {"scores": [0.9]}
        assert run_meta["corpus_scope"] == "single"
        assert run_meta["laws_searched"] == ["ai-act"]

    def test_multi_corpus_routes_to_cross_law_retrieval(self, monkeypatch):
        """Multi-corpus (scope=all) should call ro.execute_cross_law_retrieval."""
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from src.engine.planning import UserProfile

        retriever = SimpleNamespace(
            _last_retrieved_ids=[],
            _last_retrieved_metadatas=[],
        )
        pass_tracker = SimpleNamespace(
            record_pass=lambda **kw: None, get_passes=lambda: []
        )

        calls = []

        def mock_cross_law(**kwargs):
            calls.append(("cross_law", kwargs.get("corpus_ids")))
            return MagicMock(
                fused_chunks=[],
                per_corpus_hits={},
                total_retrieved=2,
                laws_searched=["ai-act", "gdpr"],
                run_meta_updates={"laws_searched": ["ai-act", "gdpr"]},
            )

        def mock_convert(multi_result, resolved_profile):
            return {
                "hits": [("doc1", {"source": "ai-act"})],
                "distances": [0.2],
                "retrieved_ids": ["id1"],
                "retrieved_metas": [{"source": "ai-act"}],
                "run_meta_updates": {"laws_searched": ["ai-act", "gdpr"]},
                "selected_chunks": (),
                "total_retrieved": 2,
                "citable_count": 1,
            }

        monkeypatch.setattr(
            "src.engine.answer_stages.ro.execute_cross_law_retrieval", mock_cross_law
        )
        monkeypatch.setattr(
            "src.engine.answer_stages.ro.convert_multi_corpus_result", mock_convert
        )

        run_meta: dict = {}
        result = execute_answer_retrieval_stage(
            question="test question",
            corpus_scope="all",
            target_corpora=None,
            resolved_profile=UserProfile.ENGINEERING,
            where_for_retrieval=None,
            ctx=SimpleNamespace(corpus_id="ai-act", focus=None),
            run_meta=run_meta,
            effective_plan=SimpleNamespace(where=None),
            pass_tracker=pass_tracker,
            corpus_id="ai-act",
            retriever=retriever,
            chroma=MagicMock(),
            collection=object(),
            available_corpora_fn=lambda: ["ai-act", "gdpr"],
            resolver_fn=lambda: None,
        )

        assert calls == [("cross_law", ("ai-act", "gdpr"))]
        assert result.use_multi_corpus is True
        assert result.synthesis_context is not None
        assert run_meta["corpus_scope"] == "all"
        assert run_meta["laws_searched"] == ["ai-act", "gdpr"]


class TestEvidenceGateResult:
    """Verify EvidenceGateResult dataclass structure."""

    def test_is_dataclass_with_required_fields(self):
        field_names = [f.name for f in fields(EvidenceGateResult)]
        assert "answer_text" in field_names
        assert "did_abstain" in field_names
        assert "min_citable_required" in field_names
        assert "early_return" in field_names


class TestExecuteEvidenceGateStage:
    """Verify execute_evidence_gate_stage function."""

    def test_is_callable(self):
        assert callable(execute_evidence_gate_stage)

    def test_non_abstain_calls_generation(self, monkeypatch):
        """Normal path: no abstain, sufficient evidence → calls LLM generation."""
        from types import SimpleNamespace
        from src.engine.planning import UserProfile, ClaimIntent

        monkeypatch.setattr(
            "src.engine.answer_stages.execute_generation_stage",
            lambda **kw: "Generated answer [1]",
        )

        run_meta: dict = {}
        result = execute_evidence_gate_stage(
            question="Hvilke krav gælder?",
            hits=[("doc1", {"source": "test"})],
            distances=[0.1],
            focus=None,
            history_context="",
            dry_run=False,
            corpus_scope="single",
            resolved_profile=UserProfile.ENGINEERING,
            effective_plan=SimpleNamespace(where=None, allow_low_evidence_answer=False),
            ctx=SimpleNamespace(corpus_id="test", focus=None),
            plan=SimpleNamespace(intent=ClaimIntent.SCOPE),
            references_structured_all=[{"idx": 1, "display": "Ref 1"}],
            citable_count_total=3,
            context="test context",
            kilder_block="test kilder",
            synthesis_context=None,
            effective_policy=None,
            claim_intent_final=ClaimIntent.SCOPE,
            corpus_debug_on=False,
            contract_min_citations=None,
            run_meta=run_meta,
            payload_context={
                "retrieval_state": {},
                "where_for_retrieval": None,
                "pass_tracker_passes": [],
                "ranking_debug": None,
                "sibling_expansion": {},
            },
            should_abstain_fn=lambda *a, **kw: None,
            llm_fn=lambda prompt: "test",
            resolver_fn=lambda: None,
        )

        assert isinstance(result, EvidenceGateResult)
        assert result.early_return is None
        assert result.did_abstain is False
        assert "Generated answer" in result.answer_text
        assert run_meta["abstain"]["abstained"] is False

    def test_abstain_sets_run_meta_and_returns_reason(self, monkeypatch):
        """Abstain path: should_abstain returns a reason → answer_text = reason."""
        from types import SimpleNamespace
        from src.engine.planning import UserProfile, ClaimIntent

        # Should not be called — generation should be skipped
        monkeypatch.setattr(
            "src.engine.answer_stages.execute_generation_stage",
            lambda **kw: (_ for _ in ()).throw(AssertionError("Should not call LLM")),
        )

        run_meta: dict = {}
        result = execute_evidence_gate_stage(
            question="test",
            hits=[],
            distances=[],
            focus=None,
            history_context="",
            dry_run=False,
            corpus_scope="single",
            resolved_profile=UserProfile.ENGINEERING,
            effective_plan=SimpleNamespace(where=None, allow_low_evidence_answer=False),
            ctx=SimpleNamespace(corpus_id="test", focus=None),
            plan=SimpleNamespace(intent=ClaimIntent.SCOPE),
            references_structured_all=[],
            citable_count_total=0,
            context="",
            kilder_block="",
            synthesis_context=None,
            effective_policy=None,
            claim_intent_final=ClaimIntent.SCOPE,
            corpus_debug_on=False,
            contract_min_citations=None,
            run_meta=run_meta,
            payload_context={
                "retrieval_state": {},
                "where_for_retrieval": None,
                "pass_tracker_passes": [],
                "ranking_debug": None,
                "sibling_expansion": {},
            },
            should_abstain_fn=lambda *a, **kw: "Insufficient evidence",
            llm_fn=lambda prompt: "test",
            resolver_fn=lambda: None,
        )

        assert result.did_abstain is True
        assert result.early_return is None  # Not dry_run, so no early return
        assert "Insufficient evidence" in result.answer_text
        assert run_meta["abstain"]["abstained"] is True

    def test_insufficient_evidence_returns_missing_ref(self, monkeypatch):
        """ENGINEERING with citable < min_required → MISSING_REF."""
        from types import SimpleNamespace
        from src.engine.planning import UserProfile, ClaimIntent

        monkeypatch.setattr(
            "src.engine.answer_stages.execute_generation_stage",
            lambda **kw: (_ for _ in ()).throw(AssertionError("Should not call LLM")),
        )

        run_meta: dict = {}
        result = execute_evidence_gate_stage(
            question="test",
            hits=[("doc1", {"source": "test"})],
            distances=[0.1],
            focus=None,
            history_context="",
            dry_run=False,
            corpus_scope="single",
            resolved_profile=UserProfile.ENGINEERING,
            effective_plan=SimpleNamespace(where=None, allow_low_evidence_answer=False),
            ctx=SimpleNamespace(corpus_id="test", focus=None),
            plan=SimpleNamespace(intent=ClaimIntent.SCOPE),
            references_structured_all=[{"idx": 1, "display": "Ref 1"}],
            citable_count_total=1,  # Below default min of 2
            context="",
            kilder_block="",
            synthesis_context=None,
            effective_policy=None,
            claim_intent_final=ClaimIntent.SCOPE,
            corpus_debug_on=False,
            contract_min_citations=None,
            run_meta=run_meta,
            payload_context={
                "retrieval_state": {},
                "where_for_retrieval": None,
                "pass_tracker_passes": [],
                "ranking_debug": None,
                "sibling_expansion": {},
            },
            should_abstain_fn=lambda *a, **kw: None,
            llm_fn=lambda prompt: "test",
            resolver_fn=lambda: None,
        )

        assert result.answer_text == "MISSING_REF"
        assert result.did_abstain is False
        assert run_meta["final_gate_reason"] == "insufficient_citable_evidence_pre_llm"

    def test_dry_run_abstain_returns_early(self):
        """Abstain + dry_run → early_return payload (not None)."""
        from types import SimpleNamespace
        from src.engine.planning import UserProfile, ClaimIntent

        run_meta: dict = {}
        result = execute_evidence_gate_stage(
            question="test",
            hits=[],
            distances=[],
            focus=None,
            history_context="",
            dry_run=True,
            corpus_scope="single",
            resolved_profile=UserProfile.ENGINEERING,
            effective_plan=SimpleNamespace(where=None, allow_low_evidence_answer=False),
            ctx=SimpleNamespace(corpus_id="test", focus=None),
            plan=SimpleNamespace(intent=ClaimIntent.SCOPE),
            references_structured_all=[],
            citable_count_total=0,
            context="",
            kilder_block="",
            synthesis_context=None,
            effective_policy=None,
            claim_intent_final=ClaimIntent.SCOPE,
            corpus_debug_on=False,
            contract_min_citations=None,
            run_meta=run_meta,
            payload_context={
                "retrieval_state": {},
                "where_for_retrieval": None,
                "pass_tracker_passes": [],
                "ranking_debug": None,
                "sibling_expansion": {},
            },
            should_abstain_fn=lambda *a, **kw: "No evidence found",
            llm_fn=lambda prompt: "test",
            resolver_fn=lambda: None,
        )

        assert result.early_return is not None
        assert result.did_abstain is True
        assert run_meta["dry_run"] is True
        assert run_meta["dry_run_stage"] == "abstained"


class TestPostGenerationResult:
    """Verify PostGenerationResult dataclass structure."""

    def test_is_dataclass_with_required_fields(self):
        field_names = [f.name for f in fields(PostGenerationResult)]
        assert "answer_text" in field_names
        assert "references_structured" in field_names
        assert "reference_lines" in field_names
        assert "used_chunk_ids" in field_names
        assert "did_abstain" in field_names


class TestExecutePostGenerationStage:
    """Verify execute_post_generation_stage function."""

    def test_is_callable(self):
        assert callable(execute_post_generation_stage)

    def test_basic_pipeline_returns_processed_answer(self):
        """Basic flow: answer goes through policy + citation processing."""
        from types import SimpleNamespace
        from src.engine.planning import UserProfile, ClaimIntent

        run_meta: dict = {}
        result = execute_post_generation_stage(
            answer_text="Artikel 12 kræver record-keeping [1]",
            question="Hvilke krav gælder?",
            resolved_profile=UserProfile.LEGAL,
            run_meta=run_meta,
            references_structured_all=[
                {
                    "idx": 1,
                    "display": "AI Act, art. 12",
                    "precise_ref": "AI Act, art. 12",
                    "source": "AI Act",
                },
            ],
            distances=[0.1],
            effective_plan=SimpleNamespace(allow_low_evidence_answer=False),
            effective_policy=None,
            claim_intent_final=ClaimIntent.REQUIREMENTS,
            did_abstain=False,
            bypass_required_support_gate=False,
            min_citable_required=1,
            ctx=SimpleNamespace(
                corpus_id="ai-act", focus=None, user_profile=UserProfile.LEGAL
            ),
            total_retrieved=3,
            citable_count_total=2,
            contract_min_citations=None,
            corpus_debug_on=False,
            max_distance=None,
            corpus_id="ai-act",
            project_root=None,
        )

        assert isinstance(result, PostGenerationResult)
        assert result.answer_text is not None
        assert isinstance(result.references_structured, list)
        assert isinstance(result.reference_lines, list)
