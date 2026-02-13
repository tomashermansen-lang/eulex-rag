# src/eval/__init__.py
"""
Evaluation package for the RAG framework.

Architecture follows EVAL = PROD principle:
- eval_runner.py orchestrates test cases
- scorers.py applies scoring functions to production results
- reporters.py formats output for different use cases

Key principle: Eval calls production code (ask.ask), never reimplements.

Usage:
    python -m src.eval.eval_runner --law ai-act
    python -m src.eval.eval_runner --law ai-act --skip-llm
    python -m src.eval.eval_runner --law ai-act --pipeline-analysis
"""
# Lazy imports to avoid circular dependencies
__all__ = [
    "AnchorScorer",
    "ContractScorer", 
    "PipelineBreakdownScorer",
    "ProgressReporter",
    "FailureReporter",
    "run_eval",
]


def __getattr__(name: str):
    if name in ("AnchorScorer", "ContractScorer", "PipelineBreakdownScorer"):
        from .scorers import AnchorScorer, ContractScorer, PipelineBreakdownScorer
        return locals()[name]
    if name in ("ProgressReporter", "FailureReporter"):
        from .reporters import ProgressReporter, FailureReporter
        return locals()[name]
    if name == "run_eval":
        from .eval_runner import run_eval
        return run_eval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
