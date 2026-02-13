/**
 * TypeScript type definitions for the EuLex UI.
 *
 * Single Responsibility: Define all shared types and interfaces.
 */

/** A source reference from the RAG system */
export interface Reference {
  idx: number | string;
  display: string;
  chunk_text: string;
  corpus_id?: string;
  article?: string;
  recital?: string;
  annex?: string;
  paragraph?: string;
  litra?: string;
}

/** A chat message in the conversation */
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  references?: Reference[];
  responseTime?: number;
  suggestedQuestions?: string[];
  isStreaming?: boolean;
  retrievalMetrics?: Record<string, unknown>;
}

/** Corpus scope options for cross-law queries */
export type CorpusScope = 'single' | 'explicit' | 'all' | 'discover';

/** A corpus matched by AI discovery */
export interface DiscoveryMatch {
  corpus_id: string;
  confidence: number;
  reason: 'alias_match' | 'retrieval_probe' | 'llm_disambiguation';
  display_name?: string;
}

/** Complete discovery result with gating decision */
export interface DiscoveryResult {
  matches: DiscoveryMatch[];
  resolved_scope: string;
  resolved_corpora: string[];
  gate: 'AUTO' | 'SUGGEST' | 'ABSTAIN';
}

/** User settings for the application */
export interface Settings {
  userProfile: 'LEGAL' | 'ENGINEERING';
  debugMode: boolean;
  darkMode: boolean;
  /** Corpus scope for cross-law queries */
  corpusScope: CorpusScope;
  /** Target corpora - used for all modes (single, explicit, all) */
  targetCorpora: string[];
}

/** Information about an available corpus */
export interface CorpusInfo {
  id: string;
  name: string;
  fullname?: string;
  source_url?: string;
  celex_number?: string;
  /** EuroVoc subject keywords from EUR-Lex */
  eurovoc_labels?: string[];
}

/** Response from /api/ask endpoint */
export interface AskResponse {
  answer: string;
  references: Reference[];
  retrieval_metrics: Record<string, unknown>;
  response_time_seconds: number;
}

/** Server-Sent Event chunk types */
export type StreamEventType = 'chunk' | 'result' | 'error';

/** SSE chunk event */
export interface StreamChunkEvent {
  type: 'chunk';
  content: string;
}

/** SSE result event */
export interface StreamResultEvent {
  type: 'result';
  data: AskResponse;
}

/** SSE error event */
export interface StreamErrorEvent {
  type: 'error';
  message: string;
}

/** Union of all SSE event types */
export type StreamEvent = StreamChunkEvent | StreamResultEvent | StreamErrorEvent;

/** A message in conversation history (sent to backend) */
export interface HistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

/** Request payload for asking questions */
export interface AskRequest {
  question: string;
  /** @deprecated Use target_corpora instead. Kept for backend compatibility. */
  law?: string;
  user_profile: string;
  history: HistoryMessage[];
  /** Corpus scope for cross-law queries */
  corpus_scope?: CorpusScope;
  /** Target corpora - primary way to specify which laws to search */
  target_corpora?: string[];
}

// Admin types for legislation management

/** Quality metrics from HTML ingestion */
export interface IngestionQuality {
  unhandled_patterns: Record<string, number>;  // Pattern type -> count
  unhandled_count: number;                     // Total unhandled patterns
  unhandled_pct: number;                       // Percentage of unhandled patterns
  structure_coverage_pct: number;              // Percentage of chunks with citable structure
  chunk_count: number;                         // Total chunks
}

/** Information about a piece of EU legislation */
export interface LegislationInfo {
  celex_number: string;
  title_da: string;
  title_en: string;
  last_modified: string | null;       // Document date
  entry_into_force: string | null;    // When the law became legally binding
  in_force: boolean;
  amended_by: string[];
  is_ingested: boolean;
  corpus_id: string | null;           // Short ID like 'gdpr', 'ai-act'
  local_version_date: string | null;
  is_outdated: boolean;
  html_url: string;
  document_type: string;
  eurovoc_labels: string[];           // EuroVoc subject keywords (Danish)
  quality: IngestionQuality | null;   // Ingestion quality metrics (if ingested)
}

/** Run modes for eval (same as Dashboard) */
export type EvalRunMode = 'retrieval_only' | 'full' | 'full_with_judge';

/** Request to add a new law/corpus */
export interface AddLawRequest {
  celex_number: string;
  corpus_id: string;
  display_name: string;
  fullname?: string | null;           // Full official legal title for citations
  eurovoc_labels?: string[] | null;   // EuroVoc subject keywords from EUR-Lex
  generate_eval: boolean;
  entry_into_force: string | null;    // ISO date
  last_modified: string | null;       // ISO date (adoption date)
  eval_run_mode?: EvalRunMode;        // Run mode for verification eval (optional)
}

/** Ingestion event types */
export type IngestionEventType = 'stage' | 'progress' | 'complete' | 'error';

/** SSE event for ingestion progress */
export interface IngestionStageEvent {
  type: 'stage';
  stage: string;
  message: string;
  completed: boolean;
}

export interface IngestionProgressEvent {
  type: 'progress';
  stage: string;
  progress_pct: number;
  current: number;
  total: number;
}

export interface IngestionCompleteEvent {
  type: 'complete';
  corpus_id: string;
}

export interface IngestionErrorEvent {
  type: 'error';
  error: string;
}

export interface IngestionEvalResultEvent {
  type: 'eval_result';
  case_id: string;
  question: string;
  answer: string;
  expected_articles: string[];
  actual_articles: string[];
  passed: boolean;
  notes: string;
}

export interface IngestionEvalSummaryEvent {
  type: 'eval_summary';
  total: number;
  passed: number;
  failed: number;
}

export interface IngestionPreflightEvent {
  type: 'preflight';
  handled: Record<string, number>;
  unhandled: Record<string, number>;
  warnings: Array<{
    category: string;
    message: string;
    location: string;
    severity: string;
    suggestion: string;
  }>;
}

export type IngestionEvent =
  | IngestionStageEvent
  | IngestionProgressEvent
  | IngestionCompleteEvent
  | IngestionErrorEvent
  | IngestionEvalResultEvent
  | IngestionEvalSummaryEvent
  | IngestionPreflightEvent;

/** Page identifiers for routing */
export type PageId = 'chat' | 'admin';

// Eval Dashboard types

/** Valid test type categories */
export type EvalTestType = 'retrieval' | 'faithfulness' | 'relevancy' | 'abstention' | 'robustness' | 'multi_hop';

/** Statistics for a single test type */
export interface EvalTestTypeStats {
  test_type: EvalTestType;
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
}

/** Eval statistics for a single law/corpus */
export interface EvalLawStats {
  law: string;
  display_name: string;
  total_cases: number;
  passed: number;
  failed: number;
  pass_rate: number;
  last_run: string | null;
  last_run_mode: 'retrieval_only' | 'full' | 'full_with_judge' | null;
  by_test_type: EvalTestTypeStats[];
}

/** Response with eval overview (matrix view data) */
export interface EvalOverviewResponse {
  laws: EvalLawStats[];
  test_types: EvalTestType[];
  total_cases: number;
  overall_pass_rate: number;
}

/** Summary of a single eval run */
export interface EvalRunSummary {
  run_id: string;
  law: string;
  timestamp: string;
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  duration_seconds: number;
  trigger_source: 'cli' | 'api' | 'scheduled';
  run_mode: 'retrieval_only' | 'full' | 'full_with_judge';
}

/** Response with list of historical eval runs */
export interface EvalRunListResponse {
  runs: EvalRunSummary[];
  total: number;
}

/** Result of a single eval case */
export interface EvalCaseResult {
  case_id: string;
  profile: string;
  prompt: string;
  passed: boolean;
  test_types: EvalTestType[];
  origin: 'auto' | 'manual';
  duration_ms: number;
  scores: Record<string, {
    passed: boolean;
    score: number;
    message?: string;
    details?: Record<string, unknown>;
  }>;
  failure_reason: string | null;
  /** Number of retry attempts before pass/fail */
  retry_count?: number;
  /** Whether the case was escalated to a fallback model */
  escalated?: boolean;
  /** The model used for escalation (if escalated) */
  escalation_model?: string | null;
}

/** Detailed response for a single eval run */
export interface EvalRunDetailResponse {
  run_id: string;
  law: string;
  timestamp: string;
  duration_seconds: number;
  summary: Record<string, unknown>;
  results: EvalCaseResult[];
  stage_stats: Record<string, unknown>;
  retry_stats: Record<string, unknown>;
  escalation_stats: Record<string, unknown>;
}

/** Request to trigger an eval run */
export interface TriggerEvalRequest {
  law: string;
  run_mode: 'retrieval_only' | 'full' | 'full_with_judge';
  case_ids?: string[];
  limit?: number;
}

// Eval Case CRUD types

/** Expected behavior schema for eval cases */
export interface ExpectedBehavior {
  must_include_any_of: string[];
  must_include_any_of_2: string[];
  must_include_all_of: string[];
  must_not_include_any_of: string[];
  contract_check: boolean;
  min_citations: number | null;
  max_citations: number | null;
  behavior: 'answer' | 'abstain';
  allow_empty_references: boolean;
  must_have_article_support_for_normative: boolean;
  notes: string;
}

/** An eval case definition */
export interface EvalCase {
  id: string;
  profile: 'LEGAL' | 'ENGINEERING';
  prompt: string;
  test_types: EvalTestType[];
  origin: 'auto' | 'manual';
  expected: ExpectedBehavior;
}

/** Request payload for creating a new eval case */
export interface EvalCaseCreate {
  id?: string;
  profile: 'LEGAL' | 'ENGINEERING';
  prompt: string;
  test_types: string[];
  expected: Partial<ExpectedBehavior>;
}

/** Request payload for updating an eval case */
export interface EvalCaseUpdate {
  profile?: 'LEGAL' | 'ENGINEERING';
  prompt?: string;
  test_types?: string[];
  expected?: Partial<ExpectedBehavior>;
}

/** Response with list of eval cases */
export interface EvalCaseListResponse {
  cases: EvalCase[];
  total: number;
}

/** Response with list of anchors from citation graph */
export interface AnchorListResponse {
  anchors: string[];
  total: number;
}
