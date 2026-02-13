/**
 * API client for communicating with the FastAPI backend.
 *
 * Single Responsibility: Handle all HTTP requests to the backend.
 */

import type {
  AskRequest,
  AskResponse,
  CorpusInfo,
  StreamEvent,
  EvalCase,
  EvalCaseCreate,
  EvalCaseUpdate,
  EvalCaseListResponse,
  AnchorListResponse,
} from '../types';

const API_BASE = '/api';

/**
 * Custom error class for API errors.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public data?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/**
 * Make a JSON API request.
 */
async function fetchJson<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.detail || `Request failed: ${response.statusText}`,
      response.status,
      errorData
    );
  }

  return response.json();
}

/**
 * Get an answer to a question (non-streaming).
 */
export async function askQuestion(request: AskRequest): Promise<AskResponse> {
  return fetchJson<AskResponse>('/ask', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Stream an answer using Server-Sent Events.
 *
 * @param request - The ask request
 * @param onChunk - Callback for each text chunk
 * @param onResult - Callback when final result arrives
 * @param onError - Callback for errors
 * @returns Cleanup function to abort the stream
 */
export function streamAnswer(
  request: AskRequest,
  onChunk: (content: string) => void,
  onResult: (response: AskResponse) => void,
  onError: (error: Error) => void
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const response = await fetch(`${API_BASE}/ask/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new ApiError(`Stream request failed: ${response.statusText}`, response.status);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new ApiError('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          const data = line.slice(6); // Remove "data: " prefix

          if (data === '[DONE]') {
            return;
          }

          try {
            const event: StreamEvent = JSON.parse(data);

            switch (event.type) {
              case 'chunk':
                onChunk(event.content);
                break;
              case 'result':
                onResult(event.data);
                break;
              case 'error':
                onError(new ApiError(event.message));
                break;
            }
          } catch {
            // Ignore JSON parse errors for incomplete messages
          }
        }
      }
    } catch (error) {
      if ((error as Error).name !== 'AbortError') {
        onError(error as Error);
      }
    }
  })();

  return () => controller.abort();
}

/**
 * Get list of available corpora.
 */
export async function getCorpora(): Promise<CorpusInfo[]> {
  const response = await fetchJson<{ corpora: CorpusInfo[] }>('/corpora');
  return response.corpora;
}

/**
 * Get example questions for all corpora.
 */
export async function getExamples(): Promise<Record<string, Record<string, string[]>>> {
  const response = await fetchJson<{ examples: Record<string, Record<string, string[]>> }>('/examples');
  return response.examples;
}

/**
 * Health check.
 */
export async function healthCheck(): Promise<{ status: string; version: string }> {
  return fetchJson('/health');
}

// ============================================================================
// Eval Case CRUD API
// ============================================================================

/**
 * List all eval cases for a law.
 */
export async function listEvalCases(law: string): Promise<EvalCaseListResponse> {
  return fetchJson(`/eval/cases/${encodeURIComponent(law)}`);
}

/**
 * Get a single eval case by ID.
 */
export async function getEvalCase(law: string, caseId: string): Promise<EvalCase> {
  return fetchJson(`/eval/cases/${encodeURIComponent(law)}/${encodeURIComponent(caseId)}`);
}

/**
 * Create a new eval case.
 */
export async function createEvalCase(law: string, data: EvalCaseCreate): Promise<EvalCase> {
  return fetchJson(`/eval/cases/${encodeURIComponent(law)}`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/**
 * Update an existing eval case.
 */
export async function updateEvalCase(law: string, caseId: string, data: EvalCaseUpdate): Promise<EvalCase> {
  return fetchJson(`/eval/cases/${encodeURIComponent(law)}/${encodeURIComponent(caseId)}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

/**
 * Delete an eval case.
 */
export async function deleteEvalCase(law: string, caseId: string): Promise<void> {
  await fetch(`${API_BASE}/eval/cases/${encodeURIComponent(law)}/${encodeURIComponent(caseId)}`, {
    method: 'DELETE',
  });
}

/**
 * Duplicate an eval case.
 */
export async function duplicateEvalCase(law: string, caseId: string): Promise<EvalCase> {
  return fetchJson(`/eval/cases/${encodeURIComponent(law)}/${encodeURIComponent(caseId)}/duplicate`, {
    method: 'POST',
  });
}

// ============================================================================
// Anchor API (for autocomplete in eval case editor)
// ============================================================================

/**
 * List available anchors for a law from citation graph.
 */
export async function listAnchors(law: string, query?: string): Promise<AnchorListResponse> {
  const params = new URLSearchParams();
  if (query) {
    params.set('q', query);
  }
  const queryString = params.toString();
  const url = `/admin/corpus/${encodeURIComponent(law)}/anchors${queryString ? `?${queryString}` : ''}`;
  return fetchJson(url);
}

// ============================================================================
// Single Case Validation (Quick Test)
// ============================================================================

export interface RunSingleCaseRequest {
  law: string;
  run_mode: 'retrieval_only' | 'full' | 'full_with_judge';
  case_id?: string;
  prompt?: string;
  profile?: 'LEGAL' | 'ENGINEERING';
  test_types?: string[];
  expected?: {
    must_include_any_of?: string[];
    must_include_any_of_2?: string[];
    must_include_all_of?: string[];
    must_not_include_any_of?: string[];
    behavior?: 'answer' | 'abstain';
    min_citations?: number | null;
    max_citations?: number | null;
  };
}

export interface SingleCaseResultResponse {
  passed: boolean;
  duration_ms: number;
  scores: Record<string, {
    passed: boolean;
    score: number | null;
    message: string;
  }>;
  answer: string;
  references: Array<{
    idx: number | string;
    display: string;
    chunk_text: string;
    corpus_id?: string;
    article?: string;
    recital?: string;
    annex?: string;
    paragraph?: string;
    litra?: string;
  }>;
  test_definition: Record<string, unknown>;
  retrieval_metrics: Record<string, unknown>;
  error?: string;
}

/**
 * Run a single eval case for quick validation.
 * Results are NOT saved to eval history.
 */
export async function runSingleCase(request: RunSingleCaseRequest): Promise<SingleCaseResultResponse> {
  return fetchJson('/eval/run-single', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}


// ---------------------------------------------------------------------------
// Cross-Law Run-Single (R-ED-15)
// ---------------------------------------------------------------------------

export interface RunSingleCrossLawRequest {
  prompt: string;
  profile: string;
  test_types: string[];
  run_mode: 'retrieval_only' | 'full' | 'full_with_judge';
  expected_behavior: string;
  expected_corpora: string[];
  min_corpora_cited: number;
  must_include_any_of: string[];
  must_include_any_of_2: string[];
  must_include_all_of: string[];
  must_not_include_any_of: string[];
  contract_check: boolean;
  min_citations: number | null;
  max_citations: number | null;
}

export interface CrossLawReference {
  idx: number;
  display: string;
  chunk_text: string;
  corpus_id?: string;
  article?: string;
  recital?: string;
  annex?: string;
  paragraph?: string;
  litra?: string;
}

export interface CrossLawValidationResult {
  passed: boolean;
  duration_ms: number;
  scores: Record<string, { passed: boolean; score: number; message: string }>;
  answer: string;
  references: CrossLawReference[];
  error?: string;
}

/**
 * Run a single cross-law case for in-memory validation.
 * Results are NOT saved — runs with current form values.
 */
export async function runSingleCrossLawCase(
  suiteId: string,
  request: RunSingleCrossLawRequest,
): Promise<CrossLawValidationResult> {
  return fetchJson(`/eval/cross-law/suites/${suiteId}/run-single`, {
    method: 'POST',
    body: JSON.stringify(request),
  });
}
