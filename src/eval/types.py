# src/eval/types.py
"""Shared type definitions for evaluation.

Single Responsibility: Define data structures used across eval modules.
This module has no dependencies on eval_runner or eval_core to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# Type alias for run modes
RunMode = Literal["retrieval_only", "full", "full_with_judge"]


@dataclass(frozen=True)
class ExpectedBehavior:
    """Expected behavior from golden case.

    Attributes:
        behavior: Expected response type:
            - "answer": System should provide a definitive answer (default)
            - "abstain": System should refuse/clarify (e.g., ambiguous questions)
            Abstain cases skip faithfulness scoring per UAEval4RAG best practices.
        min_corpora_cited: Minimum number of distinct corpora that must be cited
            (for cross-law evaluation). None means no requirement.
        required_corpora: Tuple of corpus IDs that must be cited in the answer
            (for cross-law evaluation).
    """
    must_include_any_of: list[str] = field(default_factory=list)
    must_include_any_of_2: list[str] = field(default_factory=list)
    must_include_all_of: list[str] = field(default_factory=list)
    must_not_include_any_of: list[str] = field(default_factory=list)
    contract_check: bool = False
    min_citations: int | None = None
    max_citations: int | None = None
    behavior: str = "answer"  # "answer" | "abstain"
    # Cross-law evaluation fields
    min_corpora_cited: int | None = None
    required_corpora: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoldenCase:
    """A golden test case.

    Attributes:
        id: Unique identifier for the test case
        profile: User profile type (LEGAL or ENGINEERING)
        prompt: The question/prompt to test
        expected: Expected behavior and anchors
        test_types: Evaluation categories this case tests. Valid values:
            - retrieval: Correct document/chunk retrieval
            - faithfulness: Answers grounded in retrieved context
            - relevancy: Answers address the actual question
            - abstention: System refuses when appropriate
            - robustness: Handles paraphrasing, edge cases, variations
            - multi_hop: Synthesis across multiple sources
            - corpus_coverage: Cross-law - all expected corpora cited
            - synthesis_balance: Cross-law - balanced citations across corpora
            - cross_reference_accuracy: Cross-law - inter-law references verified
            - routing_precision: Cross-law - correct laws identified
            - comparison_completeness: Cross-law - all comparison targets covered
        origin: Source of the test case ("auto" for LLM-generated, "manual" for human-created)
        corpus_scope: Corpus scope for cross-law queries:
            - "single": Traditional single-corpus query (default)
            - "explicit": Query specific corpora listed in target_corpora
            - "all": Query all available corpora
        target_corpora: Tuple of corpus IDs to query (for "explicit" scope)
        synthesis_mode: Expected synthesis mode for cross-law queries:
            - "aggregation", "comparison", "unified", "routing", or None
    """
    id: str
    profile: str  # LEGAL|ENGINEERING
    prompt: str
    expected: ExpectedBehavior
    test_types: tuple[str, ...] = ("retrieval",)
    origin: str = "auto"
    # Cross-law evaluation fields (with defaults for backward compatibility)
    corpus_scope: str = "single"  # "single" | "explicit" | "all"
    target_corpora: tuple[str, ...] = ()
    synthesis_mode: str | None = None  # "aggregation" | "comparison" | "unified" | "routing" | None
