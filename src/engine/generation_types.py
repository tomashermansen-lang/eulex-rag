"""Data types for the generation pipeline.

Single Responsibility: Type definitions only. No logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from .types import UserProfile, ClaimIntent

if TYPE_CHECKING:
    from .planning import FocusSelection


class GenerationStrategy(Protocol):
    """Protocol for generation strategies (Dependency Inversion).

    Enables:
    - Adding new strategies without modifying existing code (Open/Closed)
    - Testing with mock strategies
    - Runtime strategy selection
    """

    def __call__(
        self,
        prompt: str,
        llm_fn: Callable[[str], str],
        config: "GenerationConfig",
        allowed_idxs: set[int],
        **kwargs: Any,
    ) -> "StructuredGenerationResult": ...


@dataclass
class GenerationConfig:
    """Profile-specific generation settings for the unified pipeline.

    This configuration drives behavior differences between ENGINEERING and LEGAL
    without requiring separate code paths.
    """

    profile: UserProfile
    require_json_schema: bool = False  # ENGINEERING=True, LEGAL=optional
    min_citations: int | None = None  # ENGINEERING=contract, LEGAL=None
    citation_mode: str = "strict"  # "strict" | "soft" | "none"
    max_repair_attempts: int = 3  # Max LLM repair calls
    max_enrich_attempts: int = 1  # Max LLM enrich calls
    enable_enrich: bool = True  # Allow citation enrichment
    output_format: str = "structured_bullets"  # "structured_bullets" | "prose"
    allow_schema_fallback: bool = True  # Allow schema-only validation fallback
    json_schema_type: str = "engineering"  # "engineering" | "legal"
    soft_json_fallback: bool = False  # Fall back to prose on JSON failure

    @classmethod
    def for_engineering(
        cls,
        *,
        contract_min_citations: int | None = None,
        json_mode_enabled: bool = True,
    ) -> "GenerationConfig":
        """Factory for ENGINEERING profile with strict JSON validation."""
        return cls(
            profile=UserProfile.ENGINEERING,
            require_json_schema=json_mode_enabled,
            min_citations=contract_min_citations,
            citation_mode="strict",
            max_repair_attempts=3,
            max_enrich_attempts=1,
            enable_enrich=True,
            output_format="structured_bullets",
            allow_schema_fallback=True,
            json_schema_type="engineering",
            soft_json_fallback=False,
        )

    @classmethod
    def for_legal(
        cls,
        *,
        contract_min_citations: int | None = None,
        json_mode_enabled: bool = True,
    ) -> "GenerationConfig":
        """Factory for LEGAL profile with optional JSON validation.

        LEGAL uses a different, simpler JSON schema than ENGINEERING:
        - Only 'summary' is required
        - 'key_points', 'legal_basis', 'caveats' are optional
        - Soft validation: falls back to prose on JSON failure (no fail-closed)
        """
        return cls(
            profile=UserProfile.LEGAL,
            require_json_schema=json_mode_enabled,
            min_citations=contract_min_citations,
            citation_mode="soft",
            max_repair_attempts=1,  # One repair attempt for LEGAL
            max_enrich_attempts=0,  # No enrich for LEGAL (simpler schema)
            enable_enrich=False,
            output_format="prose",
            allow_schema_fallback=False,
            json_schema_type="legal",
            soft_json_fallback=True,  # Fall back to prose on failure
        )

    @classmethod
    def for_profile(
        cls,
        profile: UserProfile,
        *,
        contract_min_citations: int | None = None,
        json_mode_enabled: bool = True,
    ) -> "GenerationConfig":
        """Factory that returns appropriate config based on profile."""
        if profile == UserProfile.ENGINEERING:
            return cls.for_engineering(
                contract_min_citations=contract_min_citations,
                json_mode_enabled=json_mode_enabled,
            )
        return cls.for_legal(
            contract_min_citations=contract_min_citations,
            json_mode_enabled=json_mode_enabled,
        )


@dataclass
class StructuredGenerationResult:
    """Unified result from structured generation for all profiles.

    Provides a consistent interface regardless of whether JSON mode or
    prose mode was used.
    """

    answer_text: str  # Final rendered text
    raw_llm_response: str  # Original LLM output
    parsed_json: dict | None = None  # Parsed JSON (if applicable)
    cited_idxs: list[int] = field(default_factory=list)  # Extracted citation indices
    valid_cited_idxs: list[int] = field(
        default_factory=list
    )  # Citations that map to allowed refs
    repair_attempts: int = 0
    enrich_attempts: int = 0
    failed: bool = False
    fail_reason: str | None = None
    debug: dict = field(default_factory=dict)

    @property
    def is_missing_ref(self) -> bool:
        """Check if the answer failed with MISSING_REF."""
        return self.answer_text.strip() == "MISSING_REF"


@dataclass
class GenerationStageResult:
    """Result from the generation stage (Stage 3) of answer_structured.

    Contains everything needed from LLM generation for downstream stages:
    - The answer text
    - The prompt used
    - JSON mode flags and results
    - Debug/telemetry data for run_meta
    """

    answer_text: str
    prompt: str
    engineering_json_mode: bool
    legal_json_mode: bool
    gen_result: StructuredGenerationResult | None = None
    answer_policy: Any = None
    allowed_idxs: set[int] = field(default_factory=set)
    llm_calls_count: int = 0
    debug: dict[str, Any] = field(default_factory=dict)
