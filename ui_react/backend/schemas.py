"""Pydantic schemas for API request/response models.

Single Responsibility: Define data structures for API communication.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Literal


class HistoryMessage(BaseModel):
    """A message in conversation history."""

    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class AskRequest(BaseModel):
    """Request payload for asking a question."""

    question: str = Field(..., min_length=1, description="The question to ask")
    law: str = Field(..., description="Corpus/law identifier (e.g., 'ai-act', 'gdpr')")
    user_profile: str = Field(default="LEGAL", description="User profile: LEGAL or ENGINEERING")
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=10,
        description="Previous conversation messages (max 10)"
    )
    # Cross-law fields
    corpus_scope: Literal["single", "explicit", "all", "discover"] = Field(
        default="single",
        description="Corpus scope: 'single' (default), 'explicit' (target_corpora), 'all', or 'discover' (AI auto-discovery)"
    )
    target_corpora: list[str] = Field(
        default_factory=list,
        description="Target corpus IDs for 'explicit' scope"
    )


class Reference(BaseModel):
    """A source reference from the RAG system."""

    idx: int | str = Field(..., description="Reference index")
    display: str = Field(..., description="Human-readable source description")
    chunk_text: str = Field(default="", description="The actual text content")
    corpus_id: str | None = Field(default=None, description="Corpus identifier")
    article: str | None = Field(default=None, description="Article number if applicable")
    recital: str | None = Field(default=None, description="Recital number if applicable")
    annex: str | None = Field(default=None, description="Annex number if applicable")
    paragraph: str | None = Field(default=None, description="Paragraph number if applicable")
    litra: str | None = Field(default=None, description="Litra if applicable")


class AskResponse(BaseModel):
    """Response payload for a completed answer."""

    answer: str = Field(..., description="The generated answer")
    references: list[Reference] = Field(default_factory=list, description="Source references")
    retrieval_metrics: dict[str, Any] = Field(default_factory=dict, description="Debug metrics")
    response_time_seconds: float = Field(..., description="Time taken to generate answer")
    # Cross-law fields
    synthesis_mode: str | None = Field(
        default=None,
        description="Synthesis mode used: aggregation, comparison, unified, routing, or None for single-law"
    )
    laws_searched: list[str] = Field(
        default_factory=list,
        description="List of corpus IDs that were searched"
    )


class StreamChunk(BaseModel):
    """A chunk of streamed response."""

    type: str = Field(..., description="Event type: 'chunk' or 'result'")
    content: str | None = Field(default=None, description="Text content for chunk events")
    data: AskResponse | None = Field(default=None, description="Full response for result events")


class CorpusInfo(BaseModel):
    """Information about an available corpus."""

    id: str = Field(..., description="Corpus identifier")
    name: str = Field(..., description="Human-readable name")
    fullname: str | None = Field(default=None, description="Full official legal title for citations")
    source_url: str | None = Field(default=None, description="URL to source document")
    celex_number: str | None = Field(default=None, description="CELEX number for EUR-Lex reference")
    eurovoc_labels: list[str] | None = Field(default=None, description="EuroVoc subject keywords from EUR-Lex")


class CorporaResponse(BaseModel):
    """Response with available corpora."""

    corpora: list[CorpusInfo] = Field(default_factory=list, description="Available corpora")


class ExamplesResponse(BaseModel):
    """Response with example questions per corpus and profile."""

    examples: dict[str, dict[str, list[str]]] = Field(
        default_factory=dict,
        description="Example questions: {corpus: {profile: [questions]}}"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Service status")
    version: str = Field(default="1.0.0", description="API version")


# Admin schemas for legislation management

class IngestionQuality(BaseModel):
    """Quality metrics from HTML ingestion."""

    unhandled_patterns: dict[str, int] = Field(
        default_factory=dict,
        description="Unhandled HTML pattern types and counts"
    )
    unhandled_count: int = Field(default=0, description="Total unhandled pattern count")
    unhandled_pct: float = Field(default=0.0, description="Percentage of unhandled patterns")
    structure_coverage_pct: float = Field(default=0.0, description="Percentage of chunks with citable structure")
    chunk_count: int = Field(default=0, description="Total number of chunks")


class LegislationInfo(BaseModel):
    """Information about a piece of EU legislation."""

    celex_number: str = Field(..., description="CELEX number (e.g., 32024R1689)")
    title_da: str = Field(default="", description="Danish title")
    title_en: str = Field(default="", description="English title")
    last_modified: str | None = Field(default=None, description="Document date (ISO format)")
    entry_into_force: str | None = Field(default=None, description="Entry into force date (ISO format)")
    in_force: bool = Field(default=True, description="Whether the legislation is in force")
    amended_by: list[str] = Field(default_factory=list, description="CELEX numbers of amending acts")
    is_ingested: bool = Field(default=False, description="Whether locally ingested")
    corpus_id: str | None = Field(default=None, description="Short corpus ID (e.g., 'gdpr', 'ai-act')")
    local_version_date: str | None = Field(default=None, description="Local ingestion date (ISO format)")
    is_outdated: bool = Field(default=False, description="Whether local version is outdated")
    html_url: str = Field(default="", description="EUR-Lex HTML URL")
    document_type: str = Field(default="", description="Document type (Regulation, Directive, etc.)")
    eurovoc_labels: list[str] = Field(default_factory=list, description="EuroVoc subject keywords (Danish)")
    quality: IngestionQuality | None = Field(default=None, description="Ingestion quality metrics (if ingested)")


class LegislationListResponse(BaseModel):
    """Response with list of available legislation."""

    legislation: list[LegislationInfo] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of legislation items")
    year_from: int = Field(..., description="Start year of the filter range")
    year_to: int = Field(..., description="End year of the filter range")


class UpdateStatus(BaseModel):
    """Status of a corpus update check."""

    corpus_id: str = Field(..., description="Corpus identifier")
    celex_number: str = Field(default="", description="CELEX number")
    is_outdated: bool = Field(default=False, description="Whether local version is outdated")
    local_date: str | None = Field(default=None, description="Local version date (ISO format)")
    remote_date: str | None = Field(default=None, description="Remote version date (ISO format)")
    reason: str = Field(default="", description="Reason for status")


class AddLawRequest(BaseModel):
    """Request to add a new law/corpus."""

    celex_number: str = Field(..., description="CELEX number of the legislation")
    corpus_id: str = Field(..., min_length=1, description="Short ID for the corpus (e.g., 'nis2')")
    display_name: str = Field(..., min_length=1, description="Display name for the UI")
    fullname: str | None = Field(default=None, description="Full official legal title for citations")
    eurovoc_labels: list[str] | None = Field(default=None, description="EuroVoc subject keywords from EUR-Lex")
    generate_eval: bool = Field(default=False, description="Whether to generate eval cases")
    entry_into_force: str | None = Field(default=None, description="Entry into force date (ISO format)")
    last_modified: str | None = Field(default=None, description="Document adoption date (ISO format)")
    eval_run_mode: Literal["retrieval_only", "full", "full_with_judge"] | None = Field(
        default="full", description="Run mode for verification eval"
    )


class IngestionEvent(BaseModel):
    """SSE event for ingestion progress."""

    type: Literal["stage", "progress", "complete", "error"] = Field(..., description="Event type")
    stage: str | None = Field(default=None, description="Current stage name")
    message: str | None = Field(default=None, description="Status message")
    completed: bool | None = Field(default=None, description="Whether stage is completed")
    progress_pct: float | None = Field(default=None, description="Progress percentage (0-100)")
    current: int | None = Field(default=None, description="Current item number")
    total: int | None = Field(default=None, description="Total items")
    corpus_id: str | None = Field(default=None, description="Created corpus ID (for complete event)")
    error: str | None = Field(default=None, description="Error message (for error event)")


class RemoveCorpusResponse(BaseModel):
    """Response after removing a corpus."""

    success: bool = Field(..., description="Whether removal was successful")
    corpus_id: str = Field(..., description="ID of the removed corpus")
    message: str = Field(default="", description="Status message")


# Eval Dashboard schemas

class EvalTestTypeStats(BaseModel):
    """Statistics for a single test type."""

    test_type: str = Field(..., description="Test type name (retrieval, faithfulness, etc.)")
    total: int = Field(default=0, description="Total cases with this test type")
    passed: int = Field(default=0, description="Passed cases")
    failed: int = Field(default=0, description="Failed cases")
    pass_rate: float = Field(default=0.0, description="Pass rate (0.0 - 1.0)")


class EvalLawStats(BaseModel):
    """Eval statistics for a single law/corpus."""

    law: str = Field(..., description="Law/corpus identifier")
    display_name: str = Field(default="", description="Human-readable name")
    total_cases: int = Field(default=0, description="Total test cases")
    passed: int = Field(default=0, description="Passed cases")
    failed: int = Field(default=0, description="Failed cases")
    pass_rate: float = Field(default=0.0, description="Overall pass rate")
    last_run: str | None = Field(default=None, description="Last run timestamp (ISO format)")
    last_run_mode: str | None = Field(default=None, description="Last run mode: retrieval_only, full, full_with_judge")
    by_test_type: list[EvalTestTypeStats] = Field(
        default_factory=list,
        description="Breakdown by test type"
    )


class EvalOverviewResponse(BaseModel):
    """Response with eval overview (matrix view data)."""

    laws: list[EvalLawStats] = Field(default_factory=list, description="Stats per law")
    test_types: list[str] = Field(default_factory=list, description="All test type columns")
    total_cases: int = Field(default=0, description="Total cases across all laws")
    overall_pass_rate: float = Field(default=0.0, description="Overall pass rate")


class EvalRunSummary(BaseModel):
    """Summary of a single eval run."""

    run_id: str = Field(..., description="Run identifier (timestamp-based)")
    law: str = Field(..., description="Law/corpus that was evaluated")
    timestamp: str = Field(..., description="Run timestamp (ISO format)")
    total: int = Field(default=0, description="Total cases")
    passed: int = Field(default=0, description="Passed cases")
    failed: int = Field(default=0, description="Failed cases")
    pass_rate: float = Field(default=0.0, description="Pass rate")
    duration_seconds: float = Field(default=0.0, description="Run duration")
    trigger_source: str = Field(default="cli", description="How run was triggered: cli, api, scheduled")
    run_mode: str = Field(default="full", description="Run mode: retrieval_only, full, full_with_judge")


class EvalRunListResponse(BaseModel):
    """Response with list of historical eval runs."""

    runs: list[EvalRunSummary] = Field(default_factory=list, description="List of runs")
    total: int = Field(default=0, description="Total number of runs")


class EvalCaseResult(BaseModel):
    """Result of a single eval case."""

    case_id: str = Field(..., description="Case identifier")
    profile: str = Field(..., description="User profile (LEGAL/ENGINEERING)")
    prompt: str = Field(default="", description="The test prompt")
    passed: bool = Field(..., description="Whether the case passed")
    test_types: list[str] = Field(default_factory=list, description="Test types for this case")
    origin: str = Field(default="auto", description="Case origin: auto or manual")
    duration_ms: float = Field(default=0.0, description="Case duration in milliseconds")
    scores: dict[str, Any] = Field(default_factory=dict, description="Detailed scores")
    failure_reason: str | None = Field(default=None, description="Reason for failure if any")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    escalated: bool = Field(default=False, description="Whether case was escalated to fallback model")
    escalation_model: str | None = Field(default=None, description="Fallback model used if escalated")


class EvalRunDetailResponse(BaseModel):
    """Detailed response for a single eval run."""

    run_id: str = Field(..., description="Run identifier")
    law: str = Field(..., description="Law/corpus evaluated")
    timestamp: str = Field(..., description="Run timestamp (ISO format)")
    duration_seconds: float = Field(default=0.0, description="Run duration")
    summary: dict[str, Any] = Field(default_factory=dict, description="Run summary stats")
    results: list[EvalCaseResult] = Field(default_factory=list, description="Individual case results")
    stage_stats: dict[str, Any] = Field(default_factory=dict, description="Stats by pipeline stage")
    retry_stats: dict[str, Any] = Field(default_factory=dict, description="Retry statistics")
    escalation_stats: dict[str, Any] = Field(default_factory=dict, description="Model escalation stats")


class TriggerEvalRequest(BaseModel):
    """Request to trigger an eval run."""

    law: str = Field(..., description="Law/corpus to evaluate")
    run_mode: Literal["retrieval_only", "full", "full_with_judge"] = Field(
        default="full",
        description="Run mode: retrieval_only (fast), full (standard), full_with_judge (thorough)"
    )
    case_ids: list[str] | None = Field(default=None, description="Specific case IDs to run (optional)")
    limit: int | None = Field(default=None, description="Limit number of cases (optional)")


class SuggestNamesRequest(BaseModel):
    """Request to suggest corpus names using AI."""

    title: str = Field(..., min_length=1, description="The legislation title (Danish or English)")
    celex_number: str = Field(..., description="CELEX number for context")


class SuggestNamesResponse(BaseModel):
    """Response with AI-suggested names."""

    corpus_id: str = Field(..., description="Suggested structured corpus ID (e.g., 'nis2-dir-2022-2555')")
    display_name: str = Field(..., description="Suggested display name (e.g., 'NIS2-direktivet')")
    fullname: str = Field(default="", description="Full official legal title for citations")


class AnchorListResponse(BaseModel):
    """Response with list of anchors from citation graph."""

    anchors: list[str] = Field(default_factory=list, description="List of anchor references (e.g., 'article:5', 'annex:i')")
    total: int = Field(..., description="Total number of anchors (before limit)")


# Eval Case CRUD schemas

class ExpectedBehaviorSchema(BaseModel):
    """Schema for expected behavior in eval cases."""

    must_include_any_of: list[str] = Field(default_factory=list, description="Anchors where at least one must be present")
    must_include_any_of_2: list[str] = Field(default_factory=list, description="Second set of any-of anchors")
    must_include_all_of: list[str] = Field(default_factory=list, description="Anchors that must all be present")
    must_not_include_any_of: list[str] = Field(default_factory=list, description="Anchors that must not be present")
    contract_check: bool = Field(default=False, description="Whether to check citation count constraints")
    min_citations: int | None = Field(default=None, description="Minimum citation count")
    max_citations: int | None = Field(default=None, description="Maximum citation count")
    behavior: Literal["answer", "abstain"] = Field(default="answer", description="Expected response behavior")
    allow_empty_references: bool = Field(default=False, description="Whether empty references are allowed")
    must_have_article_support_for_normative: bool = Field(default=True, description="Require article support for normative claims")
    notes: str = Field(default="", description="Notes about the expected behavior")


class EvalCaseCreate(BaseModel):
    """Request payload for creating a new eval case."""

    id: str | None = Field(default=None, description="Optional case ID (auto-generated if not provided)")
    profile: Literal["LEGAL", "ENGINEERING"] = Field(..., description="User profile type")
    prompt: str = Field(..., min_length=10, description="The test prompt/question")
    test_types: list[str] = Field(..., min_length=1, description="Test type categories")
    expected: ExpectedBehaviorSchema = Field(default_factory=ExpectedBehaviorSchema, description="Expected behavior")


class EvalCaseUpdate(BaseModel):
    """Request payload for updating an eval case (partial update)."""

    profile: Literal["LEGAL", "ENGINEERING"] | None = Field(default=None, description="User profile type")
    prompt: str | None = Field(default=None, min_length=10, description="The test prompt/question")
    test_types: list[str] | None = Field(default=None, description="Test type categories")
    expected: ExpectedBehaviorSchema | None = Field(default=None, description="Expected behavior")


class EvalCaseResponse(BaseModel):
    """Response payload for an eval case."""

    id: str = Field(..., description="Case identifier")
    profile: str = Field(..., description="User profile type (LEGAL/ENGINEERING)")
    prompt: str = Field(..., description="The test prompt/question")
    test_types: list[str] = Field(..., description="Test type categories")
    origin: Literal["auto", "manual"] = Field(..., description="Case origin")
    expected: ExpectedBehaviorSchema = Field(..., description="Expected behavior")


class EvalCaseListResponse(BaseModel):
    """Response payload for listing eval cases."""

    cases: list[EvalCaseResponse] = Field(default_factory=list, description="List of eval cases")
    total: int = Field(default=0, description="Total number of cases")


# Single case validation (quick test)

class RunSingleCaseRequest(BaseModel):
    """Request to run a single eval case for validation."""

    law: str = Field(..., description="Law/corpus to test against")
    run_mode: Literal["retrieval_only", "full", "full_with_judge"] = Field(
        default="full",
        description="Run mode: retrieval_only (fast), full (standard), full_with_judge (thorough)"
    )
    # Either case_id OR inline definition
    case_id: str | None = Field(default=None, description="Existing case ID to run")
    # Inline definition (used when validating unsaved case)
    prompt: str | None = Field(default=None, min_length=1, description="Test prompt if inline")
    profile: Literal["LEGAL", "ENGINEERING"] | None = Field(default=None, description="User profile if inline")
    test_types: list[str] | None = Field(default=None, description="Test types if inline")
    expected: ExpectedBehaviorSchema | None = Field(default=None, description="Expected behavior if inline")


class SingleCaseResultResponse(BaseModel):
    """Response from running a single eval case."""

    passed: bool = Field(..., description="Whether the case passed overall")
    duration_ms: float = Field(..., description="Execution time in milliseconds")
    scores: dict[str, Any] = Field(default_factory=dict, description="Detailed scorer results")

    # The RAG response (what the system actually returned)
    answer: str = Field(default="", description="Generated answer text")
    references: list[Reference] = Field(default_factory=list, description="Source references")

    # Test definition used (for display)
    test_definition: dict[str, Any] = Field(default_factory=dict, description="Test case definition used")

    # Retrieval metrics (for debugging)
    retrieval_metrics: dict[str, Any] = Field(default_factory=dict, description="Retrieval debug info")

    # Error message if something went wrong
    error: str | None = Field(default=None, description="Error message if test failed to execute")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Dashboard Schemas
# ─────────────────────────────────────────────────────────────────────────────


class TrendInfo(BaseModel):
    """Trend direction and delta for pass rate history."""

    direction: str = Field(..., description="improving | stable | declining | insufficient_data")
    delta_pp: float | None = Field(default=None, description="Percentage-point delta from mean")
    window: int = Field(..., description="How many runs used")
    history: list[float] = Field(default_factory=list, description="Pass rates oldest→newest")


class MetricsSummary(BaseModel):
    """Summary stats for the trust overview."""

    total_cases: int = Field(..., description="Total eval cases across all runs")
    law_count: int = Field(..., description="Number of laws evaluated")
    suite_count: int = Field(..., description="Number of cross-law suites evaluated")
    last_run_timestamp: str | None = Field(default=None, description="ISO timestamp of latest run")
    ingestion_coverage: float | None = Field(default=None, description="Average ingestion coverage %")


class CategorySummary(BaseModel):
    """Pass rate summary for a category (single-law or cross-law)."""

    total: int = Field(..., description="Total cases")
    passed: int = Field(..., description="Passed cases")
    pass_rate: float = Field(..., description="Case-weighted pass rate percentage")
    group_pass_rate: float | None = Field(
        default=None,
        description="Average pass rate across groups (laws or suites), or None if N/A",
    )


class MetricsOverviewResponse(BaseModel):
    """Level 1 trust overview data."""

    unified_pass_rate: float = Field(..., description="Weighted pass rate across all runs")
    health_status: str = Field(..., description="green | yellow | orange | red")
    trend: TrendInfo
    summary: MetricsSummary
    single_law: CategorySummary
    cross_law: CategorySummary
    has_data: bool = Field(..., description="False → empty state")
    sl_run_mode_distribution: dict[str, int] = Field(default_factory=dict, description="SL cases per run_mode")
    cl_run_mode_distribution: dict[str, int] = Field(default_factory=dict, description="CL cases per run_mode")
    sl_case_origin_distribution: dict[str, int] = Field(default_factory=dict, description="SL cases per origin")
    cl_case_origin_distribution: dict[str, int] = Field(default_factory=dict, description="CL cases per origin")
    ingestion_overall_coverage: float = Field(default=0.0, description="Overall ingestion coverage %")
    ingestion_health_status: str = Field(default="green", description="green|yellow|orange|red")
    ingestion_na_count: int = Field(default=0, description="Corpora with 0% coverage (not checked)")


class LawPassRate(BaseModel):
    """Pass rate for a single law."""

    law: str = Field(..., description="Law identifier")
    display_name: str = Field(..., description="Human-readable law name")
    pass_rate: float
    total: int
    passed: int


class SuitePassRate(BaseModel):
    """Pass rate for a cross-law suite."""

    suite_id: str
    name: str
    pass_rate: float
    total: int
    passed: int


class ModePassRate(BaseModel):
    """Pass rate for a synthesis mode."""

    mode: str
    pass_rate: float
    total: int
    passed: int


class DifficultyPassRate(BaseModel):
    """Pass rate for a difficulty level."""

    difficulty: str
    pass_rate: float
    total: int
    passed: int


class ScorerPassRate(BaseModel):
    """Pass rate for a scorer."""

    scorer: str
    label: str = Field(default="", description="Human-readable label")
    pass_rate: float
    total: int
    passed: int
    category: str = Field(default="", description="single_law | cross_law")


class TrendPoint(BaseModel):
    """A single point in a trend line."""

    run_id: str = Field(default="")
    timestamp: str = Field(default="")
    pass_rate: float


class StageBreakdown(BaseModel):
    """Pipeline stage pass rates (single-law)."""

    retrieval: float = Field(default=0.0)
    augmentation: float = Field(default=0.0)
    generation: float = Field(default=0.0)


class MetricsQualityResponse(BaseModel):
    """Level 2 eval quality panel data."""

    per_law: list[LawPassRate] = Field(default_factory=list)
    per_suite: list[SuitePassRate] = Field(default_factory=list)
    per_mode: list[ModePassRate] = Field(default_factory=list)
    per_difficulty: list[DifficultyPassRate] = Field(default_factory=list)
    per_scorer: list[ScorerPassRate] = Field(default_factory=list)
    per_scorer_single_law: list[ScorerPassRate] = Field(default_factory=list)
    trend: list[TrendPoint] = Field(default_factory=list)
    stage_stats: StageBreakdown = Field(default_factory=StageBreakdown)
    escalation_rate: float = Field(default=0.0)
    retry_rate: float = Field(default=0.0)


class Percentiles(BaseModel):
    """Latency percentiles in seconds."""

    p50: float
    p95: float
    p99: float


class ModeDuration(BaseModel):
    """Average duration per run mode."""

    mode: str
    avg_seconds: float
    run_count: int


class CategoryDuration(BaseModel):
    """Average duration per category."""

    category: str
    avg_seconds: float
    run_count: int


class ModeLatency(BaseModel):
    """P50 latency per synthesis mode."""

    mode: str
    p50_seconds: float
    case_count: int


class DifficultyLatency(BaseModel):
    """P50 latency per difficulty level."""

    difficulty: str
    p50_seconds: float
    case_count: int


class HistogramBin(BaseModel):
    """A bin in a duration histogram."""

    range_start: float
    range_end: float
    count: int


class RunModeLatency(BaseModel):
    """Latency stats per evaluation run mode (single-law)."""

    run_mode: str
    p50_seconds: float
    p95_seconds: float
    case_count: int


class LatencyTrendPoint(BaseModel):
    """A data point in a latency trend series."""

    timestamp: str
    median_ms: float
    case_count: int


class RateTrendPoint(BaseModel):
    """A data point in a rate trend series (escalation, retry)."""

    timestamp: str
    rate: float
    count: int
    total: int


class RunModeTrend(BaseModel):
    """Latency trend for a specific run mode."""

    run_mode: str
    points: list[LatencyTrendPoint] = Field(default_factory=list)


class SLPerformance(BaseModel):
    """Single-law performance data."""

    percentiles: Percentiles = Field(default_factory=lambda: Percentiles(p50=0, p95=0, p99=0))
    total_cases: int = Field(default=0)
    escalation_rate: float = Field(default=0.0)
    retry_rate: float = Field(default=0.0)
    histogram_bins: list[HistogramBin] = Field(default_factory=list)
    latency_by_run_mode: list[RunModeLatency] = Field(default_factory=list)
    trend: list[LatencyTrendPoint] = Field(default_factory=list)
    trend_by_run_mode: list[RunModeTrend] = Field(default_factory=list)
    escalation_trend: list[RateTrendPoint] = Field(default_factory=list)
    retry_trend: list[RateTrendPoint] = Field(default_factory=list)


class CLPerformance(BaseModel):
    """Cross-law performance data."""

    percentiles: Percentiles = Field(default_factory=lambda: Percentiles(p50=0, p95=0, p99=0))
    total_cases: int = Field(default=0)
    histogram_bins: list[HistogramBin] = Field(default_factory=list)
    latency_by_synthesis_mode: list[ModeLatency] = Field(default_factory=list)
    latency_by_difficulty: list[DifficultyLatency] = Field(default_factory=list)
    trend: list[LatencyTrendPoint] = Field(default_factory=list)


class MetricsPerformanceResponse(BaseModel):
    """Level 2 processing performance panel data."""

    percentiles: Percentiles
    throughput_per_min: float = Field(default=0.0)
    duration_by_mode: list[ModeDuration] = Field(default_factory=list)
    duration_by_category: list[CategoryDuration] = Field(default_factory=list)
    latency_by_synthesis_mode: list[ModeLatency] = Field(default_factory=list)
    latency_by_difficulty: list[DifficultyLatency] = Field(default_factory=list)
    histogram_bins: list[HistogramBin] = Field(default_factory=list)
    total_cases: int = Field(default=0)
    excluded_zero_duration: int = Field(default=0)
    single_law: SLPerformance = Field(default_factory=SLPerformance)
    cross_law: CLPerformance = Field(default_factory=CLPerformance)


class CorpusHealth(BaseModel):
    """Health data for a single corpus."""

    corpus_id: str
    display_name: str
    coverage: float
    unhandled: int
    chunks: int
    is_ingested: bool = Field(default=True)


class MetricsIngestionResponse(BaseModel):
    """Level 2 ingestion health panel data."""

    overall_coverage: float
    health_status: str
    corpora: list[CorpusHealth] = Field(default_factory=list)


class CaseResultSummary(BaseModel):
    """Summary of a single eval case result (single-law)."""

    case_id: str
    passed: bool
    duration_ms: float
    scores: dict[str, Any] = Field(default_factory=dict)
    score_messages: dict[str, str] = Field(default_factory=dict)
    anchors: list[str] = Field(default_factory=list)
    expected_anchors: list[str] = Field(default_factory=list)
    profile: str = Field(default="")
    prompt: str = Field(default="")


class CrossLawCaseResultSummary(BaseModel):
    """Summary of a cross-law eval case result."""

    case_id: str
    passed: bool
    duration_ms: float
    synthesis_mode: str = Field(default="")
    difficulty: str | None = Field(default=None)
    suite_name: str = Field(default="")
    scores: dict[str, Any] = Field(default_factory=dict)
    score_messages: dict[str, str] = Field(default_factory=dict)
    prompt: str = Field(default="")
    target_corpora: list[str] = Field(default_factory=list)


class AppliedFilters(BaseModel):
    """Active filters in suite drill-down."""

    mode: str | None = Field(default=None)
    difficulty: str | None = Field(default=None)


class MetricsLawDetailResponse(BaseModel):
    """Level 3 per-law drill-down."""

    law: str
    display_name: str
    trend: list[TrendPoint] = Field(default_factory=list)
    scorer_breakdown: list[ScorerPassRate] = Field(default_factory=list)
    latest_results: list[CaseResultSummary] = Field(default_factory=list)


class MetricsSuiteDetailResponse(BaseModel):
    """Level 3 per-suite drill-down."""

    suite_id: str
    name: str
    trend: list[TrendPoint] = Field(default_factory=list)
    scorer_breakdown: list[ScorerPassRate] = Field(default_factory=list)
    latest_results: list[CrossLawCaseResultSummary] = Field(default_factory=list)
    mode_counts: dict[str, int] = Field(default_factory=dict)
    difficulty_counts: dict[str, int] = Field(default_factory=dict)
    applied_filters: AppliedFilters = Field(default_factory=AppliedFilters)


class MetricsScorerDetailResponse(BaseModel):
    """Level 3 per-scorer drill-down."""

    scorer: str
    label: str
    per_law_rates: list[LawPassRate] = Field(default_factory=list)
    trend: list[TrendPoint] = Field(default_factory=list)


class MetricsModeDetailResponse(BaseModel):
    """Level 3 per-mode drill-down."""

    mode: str
    pass_rate: float
    total: int
    passed: int
    applicable_scorers: list[ScorerPassRate] = Field(default_factory=list)
    cases: list[CrossLawCaseResultSummary] = Field(default_factory=list)


class MetricsDifficultyDetailResponse(BaseModel):
    """Level 3 per-difficulty drill-down."""

    difficulty: str
    pass_rate: float
    total: int
    passed: int
    cases: list[CrossLawCaseResultSummary] = Field(default_factory=list)
