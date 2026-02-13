# tests/test_llm_judge_scorers.py
"""
Tests for LLM-as-judge scorers: FaithfulnessScorer and AnswerRelevancyScorer.

These tests use mocking to avoid actual LLM calls.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from src.eval.scorers import FaithfulnessScorer, AnswerRelevancyScorer, Score
from src.eval.cache import LLMJudgeCache


class TestFaithfulnessScorer:
    """Tests for FaithfulnessScorer."""
    
    def test_fully_grounded_answer(self):
        """Answer with all claims supported by context should score 1.0."""
        scorer = FaithfulnessScorer(use_cache=False)
        
        # Mock the LLM calls
        with patch.object(scorer, '_extract_claims') as mock_extract, \
             patch.object(scorer, '_verify_claims') as mock_verify:
            
            mock_extract.return_value = [
                "The AI Act was adopted in 2024",
                "High-risk AI requires conformity assessment"
            ]
            mock_verify.return_value = [
                {"claim": "The AI Act was adopted in 2024", "supported": True, "explanation": "Found in context"},
                {"claim": "High-risk AI requires conformity assessment", "supported": True, "explanation": "Found in context"},
            ]
            
            result = scorer.score(
                question="When was the AI Act adopted?",
                answer="The AI Act was adopted in 2024. High-risk AI requires conformity assessment.",
                context="The AI Act was formally adopted in 2024. Systems classified as high-risk must undergo conformity assessment.",
            )
            
            assert result.score == 1.0
            assert result.passed is True
            assert "All claims supported" in result.message
    
    def test_hallucinated_claim(self):
        """Answer with unsupported claims should have lower score."""
        scorer = FaithfulnessScorer(threshold=0.8, use_cache=False)
        
        with patch.object(scorer, '_extract_claims') as mock_extract, \
             patch.object(scorer, '_verify_claims') as mock_verify:
            
            mock_extract.return_value = [
                "The AI Act was adopted in 2024",
                "The AI Act requires all AI systems to be registered",  # Hallucination
            ]
            mock_verify.return_value = [
                {"claim": "The AI Act was adopted in 2024", "supported": True, "explanation": "Found"},
                {"claim": "The AI Act requires all AI systems to be registered", "supported": False, "explanation": "Not in context"},
            ]
            
            result = scorer.score(
                question="What does the AI Act require?",
                answer="The AI Act was adopted in 2024. The AI Act requires all AI systems to be registered.",
                context="The AI Act was formally adopted in 2024.",
            )
            
            assert result.score == 0.5  # 1/2 claims supported
            assert result.passed is False  # Below 0.8 threshold
            assert "1/2" in result.message
    
    def test_no_claims_passes(self):
        """Answer with no factual claims should pass."""
        scorer = FaithfulnessScorer(use_cache=False)
        
        with patch.object(scorer, '_extract_claims') as mock_extract:
            mock_extract.return_value = []
            
            result = scorer.score(
                question="Hello?",
                answer="Hello! How can I help you?",
                context="Some context.",
            )
            
            assert result.score == 1.0
            assert result.passed is True
            assert "No factual claims" in result.message
    
    def test_threshold_configurable(self):
        """Threshold should be configurable."""
        scorer_strict = FaithfulnessScorer(threshold=0.9, use_cache=False)
        scorer_lenient = FaithfulnessScorer(threshold=0.5, use_cache=False)
        
        assert scorer_strict.threshold == 0.9
        assert scorer_lenient.threshold == 0.5


class TestAnswerRelevancyScorer:
    """Tests for AnswerRelevancyScorer."""
    
    def test_highly_relevant_answer(self):
        """Answer that fully addresses question should score high."""
        scorer = AnswerRelevancyScorer(use_cache=False)
        
        with patch('src.eval.scorers._call_llm_json') as mock_llm:
            mock_llm.return_value = {
                "score": 9,
                "critique": "The answer directly addresses the question with complete information."
            }
            
            result = scorer.score(
                question="What is a high-risk AI system?",
                answer="A high-risk AI system is defined in Article 6 of the AI Act as...",
            )
            
            assert result.score == 0.9  # 9/10 normalized
            assert result.passed is True
            assert "9/10" in result.message
    
    def test_irrelevant_answer(self):
        """Answer that doesn't address question should score low."""
        scorer = AnswerRelevancyScorer(threshold=0.7, use_cache=False)
        
        with patch('src.eval.scorers._call_llm_json') as mock_llm:
            mock_llm.return_value = {
                "score": 3,
                "critique": "The answer discusses unrelated topics."
            }
            
            result = scorer.score(
                question="What is a high-risk AI system?",
                answer="The weather today is sunny.",
            )
            
            assert result.score == 0.3  # 3/10 normalized
            assert result.passed is False  # Below 0.7 threshold
    
    def test_threshold_configurable(self):
        """Threshold should be configurable."""
        scorer = AnswerRelevancyScorer(threshold=0.6, use_cache=False)
        assert scorer.threshold == 0.6


class TestLLMJudgeCache:
    """Tests for LLM judge caching."""
    
    def test_cache_stores_and_retrieves(self, tmp_path):
        """Cache should store and retrieve results."""
        cache = LLMJudgeCache(cache_dir=tmp_path, enabled=True)
        
        result = {"passed": True, "score": 0.9, "message": "test"}
        cache.set("faithfulness", result, question="q", answer="a", context="c")
        
        retrieved = cache.get("faithfulness", question="q", answer="a", context="c")
        assert retrieved == result
    
    def test_cache_miss_returns_none(self, tmp_path):
        """Cache miss should return None."""
        cache = LLMJudgeCache(cache_dir=tmp_path, enabled=True)
        
        result = cache.get("faithfulness", question="unknown", answer="unknown", context="unknown")
        assert result is None
    
    def test_cache_disabled(self, tmp_path):
        """Disabled cache should always return None."""
        cache = LLMJudgeCache(cache_dir=tmp_path, enabled=False)
        
        result = {"passed": True, "score": 0.9, "message": "test"}
        cache.set("faithfulness", result, question="q", answer="a", context="c")
        
        retrieved = cache.get("faithfulness", question="q", answer="a", context="c")
        assert retrieved is None
    
    def test_cache_clear(self, tmp_path):
        """Cache clear should remove entries."""
        cache = LLMJudgeCache(cache_dir=tmp_path, enabled=True)
        
        cache.set("faithfulness", {"score": 1}, question="q1", answer="a", context="c")
        cache.set("relevancy", {"score": 1}, question="q2", answer="a", context="c")
        
        # Clear only faithfulness
        count = cache.clear("faithfulness")
        assert count == 1
        
        # Faithfulness should be gone
        assert cache.get("faithfulness", question="q1", answer="a", context="c") is None
        # Relevancy should still be there (different scorer name prefix)


class TestEvalRunnerIntegration:
    """Integration tests for eval_runner with new scorers."""
    
    def test_load_golden_cases(self, tmp_path):
        """Should load golden cases from YAML."""
        from src.eval.eval_runner import load_golden_cases
        
        cases_path = tmp_path / "cases.yaml"
        cases_path.write_text("""
- id: test-case-1
  profile: LEGAL
  prompt: What is Article 6?
  expected:
    must_include_any_of:
      - article:6
""")
        
        cases = load_golden_cases(cases_path)
        assert len(cases) == 1
        assert cases[0].id == "test-case-1"
        assert cases[0].profile == "LEGAL"
        assert "article:6" in cases[0].expected.must_include_any_of
    
    def test_build_context_text(self):
        """Should build context text from references."""
        from src.eval.eval_runner import _build_context_text
        
        refs = [
            {"article": "6", "text": "Article 6 content here."},
            {"recital": "42", "text": "Recital 42 says..."},
        ]
        
        context = _build_context_text(refs)
        
        assert "Article 6" in context
        assert "Article 6 content here" in context
        assert "Recital 42" in context


class TestGoldenCaseNewFields:
    """Tests for new GoldenCase fields: test_types and origin."""

    def test_load_golden_case_with_test_types(self, tmp_path):
        """Should parse test_types field from YAML."""
        from src.eval.eval_runner import load_golden_cases

        cases_path = tmp_path / "cases.yaml"
        cases_path.write_text("""
- id: test-case-with-types
  profile: LEGAL
  prompt: What is Article 6?
  test_types:
    - retrieval
    - faithfulness
  expected:
    must_include_any_of:
      - article:6
""")

        cases = load_golden_cases(cases_path)
        assert len(cases) == 1
        assert cases[0].test_types == ("retrieval", "faithfulness")

    def test_load_golden_case_default_test_types(self, tmp_path):
        """Should default to ['retrieval'] when test_types not specified."""
        from src.eval.eval_runner import load_golden_cases

        cases_path = tmp_path / "cases.yaml"
        cases_path.write_text("""
- id: test-case-no-types
  profile: LEGAL
  prompt: What is Article 6?
  expected:
    must_include_any_of:
      - article:6
""")

        cases = load_golden_cases(cases_path)
        assert len(cases) == 1
        assert cases[0].test_types == ("retrieval",)

    def test_load_golden_case_with_origin(self, tmp_path):
        """Should parse origin field from YAML."""
        from src.eval.eval_runner import load_golden_cases

        cases_path = tmp_path / "cases.yaml"
        cases_path.write_text("""
- id: test-case-manual
  profile: ENGINEERING
  prompt: What is required?
  origin: manual
  expected:
    must_include_any_of:
      - article:14
""")

        cases = load_golden_cases(cases_path)
        assert len(cases) == 1
        assert cases[0].origin == "manual"

    def test_load_golden_case_default_origin(self, tmp_path):
        """Should default to 'auto' when origin not specified."""
        from src.eval.eval_runner import load_golden_cases

        cases_path = tmp_path / "cases.yaml"
        cases_path.write_text("""
- id: test-case-auto
  profile: LEGAL
  prompt: What is Article 6?
  expected:
    must_include_any_of:
      - article:6
""")

        cases = load_golden_cases(cases_path)
        assert len(cases) == 1
        assert cases[0].origin == "auto"

    def test_load_golden_case_all_new_fields(self, tmp_path):
        """Should parse all new fields together."""
        from src.eval.eval_runner import load_golden_cases

        cases_path = tmp_path / "cases.yaml"
        cases_path.write_text("""
- id: complete-case
  profile: ENGINEERING
  prompt: Multi-hop question about oversight and logging?
  test_types:
    - retrieval
    - multi_hop
    - faithfulness
  origin: manual
  expected:
    must_include_any_of:
      - article:14
    must_include_all_of:
      - article:12
    behavior: answer
""")

        cases = load_golden_cases(cases_path)
        assert len(cases) == 1
        case = cases[0]
        assert case.test_types == ("retrieval", "multi_hop", "faithfulness")
        assert case.origin == "manual"
        assert case.expected.behavior == "answer"
