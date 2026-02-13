/**
 * Tests for MetricsPanel component (C8a-d).
 *
 * Tests: Level 1 trust overview, Level 2 panels, Level 3 drill-down, AI analysis.
 */

import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { MetricsPanel } from '../MetricsPanel';

// ─────────────────────────────────────────────────────────────────────────────
// Mock API responses
// ─────────────────────────────────────────────────────────────────────────────

const MOCK_OVERVIEW = {
  unified_pass_rate: 86.96,
  health_status: 'yellow',
  trend: { direction: 'stable', delta_pp: 0.5, window: 5, history: [86, 87, 87, 86.96] },
  summary: {
    total_cases: 23,
    law_count: 2,
    suite_count: 2,
    last_run_timestamp: '2026-02-10T10:00:00Z',
  },
  single_law: { total: 15, passed: 13, pass_rate: 86.67, group_pass_rate: 85.0 },
  cross_law: { total: 8, passed: 7, pass_rate: 87.5, group_pass_rate: 90.0 },
  has_data: true,
  sl_run_mode_distribution: { retrieval_only: 5, full: 6, full_with_judge: 4 },
  cl_run_mode_distribution: { full_with_judge: 8 },
  sl_case_origin_distribution: { manual: 5, auto: 10 },
  cl_case_origin_distribution: { manual: 3, auto: 5 },
  ingestion_overall_coverage: 98.25,
  ingestion_health_status: 'green',
  ingestion_na_count: 1,
};

const MOCK_QUALITY = {
  per_law: [
    { law: 'ai-act', display_name: 'AI Act', pass_rate: 90.0, total: 10, passed: 9 },
    { law: 'gdpr', display_name: 'GDPR', pass_rate: 80.0, total: 5, passed: 4 },
  ],
  per_suite: [
    { suite_id: 'suite_a', name: 'Test Suite A', pass_rate: 80.0, total: 5, passed: 4 },
  ],
  per_mode: [
    { mode: 'comparison', pass_rate: 100.0, total: 4, passed: 4 },
    { mode: 'discovery', pass_rate: 50.0, total: 2, passed: 1 },
  ],
  per_difficulty: [
    { difficulty: 'easy', pass_rate: 100.0, total: 3, passed: 3 },
    { difficulty: 'hard', pass_rate: 50.0, total: 2, passed: 1 },
  ],
  per_scorer: [
    { scorer: 'corpus_coverage', pass_rate: 87.5, total: 8, passed: 7, category: 'cross_law' },
  ],
  per_scorer_single_law: [
    { scorer: 'retrieval', pass_rate: 100.0, total: 15, passed: 15, category: 'single_law' },
    { scorer: 'faithfulness', pass_rate: 86.67, total: 15, passed: 13, category: 'single_law' },
  ],
  stage_stats: { retrieval: 100.0, augmentation: 93.33, generation: 86.67 },
  escalation_rate: 6.67,
  retry_rate: 13.33,
};

const MOCK_PERFORMANCE = {
  percentiles: { p50: 4.5, p95: 10.2, p99: 12.8 },
  total_cases: 23,
  excluded_zero_duration: 0,
  histogram_bins: [
    { range_start: 0, range_end: 2, count: 3 },
    { range_start: 2, range_end: 4, count: 8 },
    { range_start: 4, range_end: 6, count: 7 },
    { range_start: 6, range_end: 8, count: 3 },
    { range_start: 8, range_end: 10, count: 2 },
  ],
  latency_by_synthesis_mode: [
    { mode: 'comparison', p50_seconds: 4.5, case_count: 4 },
  ],
  latency_by_difficulty: [
    { difficulty: 'easy', p50_seconds: 3.5, case_count: 3 },
  ],
  single_law: {
    percentiles: { p50: 3.2, p95: 8.5, p99: 11.0 },
    total_cases: 15,
    escalation_rate: 6.67,
    retry_rate: 13.33,
    histogram_bins: [
      { range_start: 0, range_end: 3, count: 5 },
      { range_start: 3, range_end: 6, count: 7 },
      { range_start: 6, range_end: 9, count: 3 },
    ],
    latency_by_run_mode: [
      { run_mode: 'retrieval_only', p50_seconds: 1.2, p95_seconds: 2.5, case_count: 5 },
      { run_mode: 'full', p50_seconds: 3.5, p95_seconds: 7.0, case_count: 6 },
      { run_mode: 'full_with_judge', p50_seconds: 8.0, p95_seconds: 12.0, case_count: 4 },
    ],
    trend: [
      { timestamp: '2026-01-26T15:00:00Z', median_ms: 4200, case_count: 10 },
      { timestamp: '2026-01-30T20:00:00Z', median_ms: 3800, case_count: 12 },
      { timestamp: '2026-02-01T10:00:00Z', median_ms: 3200, case_count: 15 },
    ],
    trend_by_run_mode: [
      {
        run_mode: 'retrieval_only',
        points: [
          { timestamp: '2026-01-29T19:49:00Z', median_ms: 1100, case_count: 5 },
          { timestamp: '2026-01-30T20:00:00Z', median_ms: 1200, case_count: 5 },
        ],
      },
      {
        run_mode: 'full',
        points: [
          { timestamp: '2026-01-30T12:00:00Z', median_ms: 3500, case_count: 6 },
          { timestamp: '2026-02-01T10:00:00Z', median_ms: 3200, case_count: 6 },
        ],
      },
      {
        run_mode: 'full_with_judge',
        points: [
          { timestamp: '2026-01-26T15:00:00Z', median_ms: 8200, case_count: 4 },
          { timestamp: '2026-01-30T20:00:00Z', median_ms: 7800, case_count: 4 },
          { timestamp: '2026-02-01T10:00:00Z', median_ms: 7500, case_count: 4 },
        ],
      },
    ],
    escalation_trend: [
      { timestamp: '2026-01-26T00:00:00Z', rate: 5.0, count: 2, total: 40 },
      { timestamp: '2026-01-30T00:00:00Z', rate: 0.0, count: 0, total: 54 },
      { timestamp: '2026-02-01T00:00:00Z', rate: 0.0, count: 0, total: 54 },
    ],
    retry_trend: [
      { timestamp: '2026-01-26T00:00:00Z', rate: 7.5, count: 3, total: 40 },
      { timestamp: '2026-01-30T00:00:00Z', rate: 3.7, count: 2, total: 54 },
      { timestamp: '2026-02-01T00:00:00Z', rate: 5.6, count: 3, total: 54 },
    ],
  },
  cross_law: {
    percentiles: { p50: 6.8, p95: 14.5, p99: 18.2 },
    total_cases: 8,
    histogram_bins: [
      { range_start: 2, range_end: 8, count: 5 },
      { range_start: 8, range_end: 14, count: 3 },
    ],
    latency_by_synthesis_mode: [
      { mode: 'comparison', p50_seconds: 4.5, case_count: 4 },
    ],
    latency_by_difficulty: [
      { difficulty: 'easy', p50_seconds: 3.5, case_count: 3 },
    ],
    trend: [
      { timestamp: '2026-02-11T21:00:00Z', median_ms: 7500, case_count: 8 },
      { timestamp: '2026-02-12T11:00:00Z', median_ms: 6800, case_count: 8 },
    ],
  },
};

const MOCK_INGESTION = {
  overall_coverage: 98.25,
  health_status: 'green',
  corpora: [
    { corpus_id: 'ai-act', display_name: 'AI Act', coverage: 99.5, unhandled: 2, chunks: 929, is_ingested: true },
    { corpus_id: 'gdpr', display_name: 'GDPR', coverage: 97.0, unhandled: 5, chunks: 450, is_ingested: true },
  ],
};

// ─────────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────────

const mockFetch = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  global.fetch = mockFetch;

  // Default: all 4 endpoints succeed
  mockFetch.mockImplementation((url: string) => {
    if (url.includes('/metrics/overview')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
    }
    if (url.includes('/metrics/quality')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
    }
    if (url.includes('/metrics/performance')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
    }
    if (url.includes('/metrics/ingestion')) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
    }
    return Promise.resolve({ ok: false, status: 404 });
  });
});


// ─────────────────────────────────────────────────────────────────────────────
// C8a: Level 1 — Trust Overview
// ─────────────────────────────────────────────────────────────────────────────

describe('MetricsPanel Level 1', () => {
  it('renders health badge with pass rate', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getAllByText('86.96%').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders health status via badge aria-label', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByRole('status', { name: /samlet.*86\.96/i })).toBeInTheDocument();
    });
  });

  it('renders summary row with case/law/suite counts', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getAllByText(/23 cases/).length).toBeGreaterThanOrEqual(1);
      // "2 love" appears in both badge subtitle and summary row
      expect(screen.getAllByText(/2 love/).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/2 suiter/).length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders three status badges: samlet, single-law, cross-law', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      // Three badges with their pass rates (some may appear in scorer lists too)
      expect(screen.getAllByText('86.96%').length).toBeGreaterThanOrEqual(1); // Unified
      expect(screen.getAllByText('86.67%').length).toBeGreaterThanOrEqual(1); // Single-law
      expect(screen.getAllByText('87.5%').length).toBeGreaterThanOrEqual(1);  // Cross-law
      // Labels (appear in both badges, health bubbles, and quality section headers)
      expect(screen.getAllByText('Samlet').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Single-Law').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Cross-Law').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders trend indicator', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      // Trend direction text (may appear in badge + glossary)
      expect(screen.getAllByText(/stabil/i).length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders AI analysis button with description text', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /analys/i })).toBeInTheDocument();
      // Explanation text next to the button
      expect(screen.getByText(/AI-baseret/i)).toBeInTheDocument();
    });
  });

  it('renders AI analysis section always visible (not just when streaming)', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/AI Analyse/)).toBeInTheDocument();
    });
  });

  it('shows empty state when no data', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/metrics/overview')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ ...MOCK_OVERVIEW, has_data: false, unified_pass_rate: 0 }),
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });

    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/ingen eval data/i)).toBeInTheDocument();
    });
  });

  it('shows loading state initially', () => {
    // Never resolve fetch
    mockFetch.mockImplementation(() => new Promise(() => {}));
    render(<MetricsPanel />);
    expect(screen.getByText(/indlæser/i)).toBeInTheDocument();
  });

  it('shows error state on fetch failure', async () => {
    mockFetch.mockRejectedValue(new Error('Network error'));
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/fejl/i)).toBeInTheDocument();
    });
  });
});


// ─────────────────────────────────────────────────────────────────────────────
// C8b: Level 2 — Quality, Performance, Ingestion panels
// ─────────────────────────────────────────────────────────────────────────────

/** Helper: expand a collapsed section by clicking its header. */
async function expandSection(headerPattern: RegExp) {
  const matches = screen.getAllByText(headerPattern);
  // Find the one that's inside a clickable button (section header)
  for (const el of matches) {
    const btn = el.closest('button');
    if (btn) { fireEvent.click(btn); return; }
  }
}

describe('MetricsPanel Level 2', () => {
  it('renders quality panel with section headers', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
  });

  it('renders per-law entries', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);
    await waitFor(() => {
      expect(screen.getAllByText('AI Act').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('GDPR').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders per-mode entries', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);
    await waitFor(() => {
      expect(screen.getAllByText(/comparison/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/discovery/i).length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders performance panel with percentiles', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/performance/i)).toBeInTheDocument();
    });
    await expandSection(/performance/i);
    await waitFor(() => {
      // "Median" appears in badge (Trust Overview) + stat rows + legend
      expect(screen.getAllByText('Median').length).toBeGreaterThanOrEqual(1);
      // "P95" appears in badge (Trust Overview) + stat rows as "P95/case"
      expect(screen.getAllByText('P95').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders ingestion panel with coverage', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getAllByText(/ingestion/i).length).toBeGreaterThanOrEqual(1);
    });
    await expandSection(/ingestion/i);
    await waitFor(() => {
      expect(screen.getAllByText('98.25%').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders pipeline stage stats', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);
    await waitFor(() => {
      expect(screen.getAllByText(/retrieval/i).length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders N/A for zero-coverage ingestion corpora', async () => {
    const ingestionWithZero = {
      overall_coverage: 98.25,
      health_status: 'green',
      corpora: [
        { corpus_id: 'ai-act', display_name: 'AI Act', coverage: 99.5, unhandled: 2, chunks: 929, is_ingested: true },
        { corpus_id: 'dora', display_name: 'DORA', coverage: 0, unhandled: 0, chunks: 0, is_ingested: true },
      ],
    };
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/metrics/overview')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      if (url.includes('/metrics/quality')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      if (url.includes('/metrics/performance')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      if (url.includes('/metrics/ingestion')) return Promise.resolve({ ok: true, json: () => Promise.resolve(ingestionWithZero) });
      return Promise.resolve({ ok: false, status: 404 });
    });
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getAllByText(/ingestion/i).length).toBeGreaterThanOrEqual(1);
    });
    await expandSection(/ingestion/i);
    await waitFor(() => {
      expect(screen.getByText('N/A')).toBeInTheDocument();
    });
  });

  it('renders scorer tooltips on hover', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);
    await waitFor(() => {
      const faithfulnessEl = screen.getByText('Faithfulness');
      expect(faithfulnessEl.closest('[title]')).toBeTruthy();
    });
  });

  it('renders filter input for per-law table', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/søg lov/i)).toBeInTheDocument();
    });
  });

  it('renders filter input for per-suite table', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/søg suite/i)).toBeInTheDocument();
    });
  });

  it('filters per-law entries by text input', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);
    await waitFor(() => {
      expect(screen.getAllByText('AI Act').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('GDPR').length).toBeGreaterThanOrEqual(1);
    });
    // Type filter text
    const filterInput = screen.getByPlaceholderText(/søg lov/i);
    fireEvent.change(filterInput, { target: { value: 'gdpr' } });
    // AI Act should be filtered out, GDPR should remain
    await waitFor(() => {
      expect(screen.queryByText('AI Act')).not.toBeInTheDocument();
      expect(screen.getAllByText('GDPR').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('paginates per-law table when >100 entries', async () => {
    // Generate 150 mock laws
    const manyLaws = Array.from({ length: 150 }, (_, i) => ({
      law: `law-${i}`,
      display_name: `Law ${i}`,
      pass_rate: 80 + (i % 20),
      total: 10,
      passed: 8,
    }));
    const qualityWithManyLaws = {
      ...MOCK_QUALITY,
      per_law: manyLaws,
    };
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/metrics/overview')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      if (url.includes('/metrics/quality')) return Promise.resolve({ ok: true, json: () => Promise.resolve(qualityWithManyLaws) });
      if (url.includes('/metrics/performance')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      if (url.includes('/metrics/ingestion')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      return Promise.resolve({ ok: false, status: 404 });
    });
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);

    // Should show "Law 0" (first page) but NOT "Law 100" (second page)
    await waitFor(() => {
      expect(screen.getByText('Law 0')).toBeInTheDocument();
    });
    expect(screen.queryByText('Law 100')).not.toBeInTheDocument();

    // Should show pagination indicator "1 / 2"
    expect(screen.getByText('1 / 2')).toBeInTheDocument();

    // Click next page
    const nextBtn = screen.getByRole('button', { name: /næste/i });
    fireEvent.click(nextBtn);

    // Should now show "Law 100" and NOT "Law 0"
    await waitFor(() => {
      expect(screen.getByText('Law 100')).toBeInTheDocument();
    });
    expect(screen.queryByText('Law 0')).not.toBeInTheDocument();
  });

  it('renders collapsible sections collapsed by default, expandable on click', async () => {
    render(<MetricsPanel />);
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    // Sections start collapsed — per-law content should NOT be visible
    expect(screen.queryByText('90.0%')).not.toBeInTheDocument();
    // Click to expand
    const qualityHeader = screen.getByText(/eval quality/i);
    const toggleButton = qualityHeader.closest('button');
    expect(toggleButton).toBeTruthy();
    if (toggleButton) fireEvent.click(toggleButton);
    // Per-law entries should now be visible
    await waitFor(() => {
      expect(screen.getAllByText('AI Act').length).toBeGreaterThanOrEqual(1);
    });
    // Click again to collapse
    if (toggleButton) fireEvent.click(toggleButton);
    await waitFor(() => {
      expect(screen.queryByText('90.0%')).not.toBeInTheDocument();
    });
  });
});


// ─────────────────────────────────────────────────────────────────────────────
// C8c: Level 3 — Drill-down
// ─────────────────────────────────────────────────────────────────────────────

const MOCK_LAW_DETAIL = {
  law: 'ai-act',
  display_name: 'AI Act',
  scorer_breakdown: [
    { scorer: 'retrieval', pass_rate: 100.0, total: 10, passed: 10, category: 'single_law' },
    { scorer: 'faithfulness', pass_rate: 90.0, total: 10, passed: 9, category: 'single_law' },
  ],
  latest_results: [
    { case_id: 'ai-0', passed: true, duration_ms: 3000, scores: { retrieval: true, faithfulness: true } },
    { case_id: 'ai-1', passed: false, duration_ms: 3200, scores: { retrieval: true, faithfulness: false } },
  ],
};

const MOCK_MODE_DETAIL = {
  mode: 'comparison',
  pass_rate: 100.0,
  total: 4,
  passed: 4,
  applicable_scorers: [
    { scorer: 'corpus_coverage', pass_rate: 100.0, total: 4, passed: 4, category: 'cross_law' },
  ],
  cases: [
    { case_id: 'c1', passed: true, duration_ms: 5000, synthesis_mode: 'comparison', difficulty: 'easy', scores: {} },
  ],
};

describe('MetricsPanel Level 3', () => {
  it('drills down to law detail on click', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/metrics/overview')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      }
      if (url.includes('/metrics/quality')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      }
      if (url.includes('/metrics/performance')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      }
      if (url.includes('/metrics/ingestion')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      }
      if (url.includes('/metrics/detail/law/ai-act')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_LAW_DETAIL) });
      }
      return Promise.resolve({ ok: false, status: 404 });
    });

    render(<MetricsPanel />);

    // Expand quality section (collapsed by default)
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);

    await waitFor(() => {
      expect(screen.getAllByText('AI Act').length).toBeGreaterThanOrEqual(1);
    });

    // Click the AI Act entry in the per-law table row
    const lawEls = screen.getAllByText('AI Act');
    const lawRow = lawEls.find(el => el.closest('tr.cursor-pointer'));
    if (lawRow?.closest('tr')) {
      fireEvent.click(lawRow.closest('tr')!);
    }

    // Should show law detail
    await waitFor(() => {
      expect(screen.getByText(/ai-0/i)).toBeInTheDocument();
    });
  });

  it('drills down to mode detail on click', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/metrics/overview')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      }
      if (url.includes('/metrics/quality')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      }
      if (url.includes('/metrics/performance')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      }
      if (url.includes('/metrics/ingestion')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      }
      if (url.includes('/metrics/detail/mode/comparison')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_MODE_DETAIL) });
      }
      return Promise.resolve({ ok: false, status: 404 });
    });

    render(<MetricsPanel />);

    // Expand quality section (collapsed by default)
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);

    await waitFor(() => {
      expect(screen.getAllByText(/comparison/i).length).toBeGreaterThanOrEqual(1);
    });

    // Click the comparison entry in the per-mode table row
    const modeEls = screen.getAllByText(/comparison/i);
    const modeRow = modeEls.find(el => el.closest('tr.cursor-pointer'));
    if (modeRow?.closest('tr')) {
      fireEvent.click(modeRow.closest('tr')!);
    }

    // Should show mode detail with case
    await waitFor(() => {
      expect(screen.getByText('c1')).toBeInTheDocument();
    });
  });

  it('renders case filter input in drill-down', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/metrics/overview')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      if (url.includes('/metrics/quality')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      if (url.includes('/metrics/performance')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      if (url.includes('/metrics/ingestion')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      if (url.includes('/metrics/detail/law/ai-act')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_LAW_DETAIL) });
      return Promise.resolve({ ok: false, status: 404 });
    });
    render(<MetricsPanel />);
    await waitFor(() => { expect(screen.getByText(/eval quality/i)).toBeInTheDocument(); });
    await expandSection(/eval quality/i);
    await waitFor(() => { expect(screen.getAllByText('AI Act').length).toBeGreaterThanOrEqual(1); });
    const lawEls = screen.getAllByText('AI Act');
    const lawRow = lawEls.find(el => el.closest('tr.cursor-pointer'));
    if (lawRow?.closest('tr')) fireEvent.click(lawRow.closest('tr')!);
    await waitFor(() => { expect(screen.getByText(/ai-0/i)).toBeInTheDocument(); });
    // Should have a case filter input
    expect(screen.getByPlaceholderText(/søg case/i)).toBeInTheDocument();
  });

  it('filters cases by text input in drill-down', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/metrics/overview')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      if (url.includes('/metrics/quality')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      if (url.includes('/metrics/performance')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      if (url.includes('/metrics/ingestion')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      if (url.includes('/metrics/detail/law/ai-act')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_LAW_DETAIL) });
      return Promise.resolve({ ok: false, status: 404 });
    });
    render(<MetricsPanel />);
    await waitFor(() => { expect(screen.getByText(/eval quality/i)).toBeInTheDocument(); });
    await expandSection(/eval quality/i);
    await waitFor(() => { expect(screen.getAllByText('AI Act').length).toBeGreaterThanOrEqual(1); });
    const lawEls = screen.getAllByText('AI Act');
    const lawRow = lawEls.find(el => el.closest('tr.cursor-pointer'));
    if (lawRow?.closest('tr')) fireEvent.click(lawRow.closest('tr')!);
    await waitFor(() => { expect(screen.getByText(/ai-0/i)).toBeInTheDocument(); });
    // Filter to only show ai-1
    const filterInput = screen.getByPlaceholderText(/søg case/i);
    fireEvent.change(filterInput, { target: { value: 'ai-1' } });
    await waitFor(() => {
      expect(screen.queryByText(/ai-0/i)).not.toBeInTheDocument();
      expect(screen.getByText(/ai-1/i)).toBeInTheDocument();
    });
  });

  it('paginates cases in drill-down with max 20 per page', async () => {
    // Generate 25 cases
    const manyCases = Array.from({ length: 25 }, (_, i) => ({
      case_id: `case-${i}`,
      passed: i % 3 !== 0,
      duration_ms: 2000 + i * 100,
      scores: { retrieval: true },
    }));
    const lawDetailMany = { ...MOCK_LAW_DETAIL, latest_results: manyCases };
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/metrics/overview')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      if (url.includes('/metrics/quality')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      if (url.includes('/metrics/performance')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      if (url.includes('/metrics/ingestion')) return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      if (url.includes('/metrics/detail/law/ai-act')) return Promise.resolve({ ok: true, json: () => Promise.resolve(lawDetailMany) });
      return Promise.resolve({ ok: false, status: 404 });
    });
    render(<MetricsPanel />);
    await waitFor(() => { expect(screen.getByText(/eval quality/i)).toBeInTheDocument(); });
    await expandSection(/eval quality/i);
    await waitFor(() => { expect(screen.getAllByText('AI Act').length).toBeGreaterThanOrEqual(1); });
    const lawEls = screen.getAllByText('AI Act');
    const lawRow = lawEls.find(el => el.closest('tr.cursor-pointer'));
    if (lawRow?.closest('tr')) fireEvent.click(lawRow.closest('tr')!);
    // First page: case-0 visible, case-20 not
    await waitFor(() => { expect(screen.getByText('case-0')).toBeInTheDocument(); });
    expect(screen.queryByText('case-20')).not.toBeInTheDocument();
    // Pagination "1 / 2"
    expect(screen.getByText('1 / 2')).toBeInTheDocument();
    // Click next
    const nextBtn = screen.getByRole('button', { name: /næste/i });
    fireEvent.click(nextBtn);
    await waitFor(() => { expect(screen.getByText('case-20')).toBeInTheDocument(); });
    expect(screen.queryByText('case-0')).not.toBeInTheDocument();
  });

  it('shows back button in drill-down', async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('/metrics/overview')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      }
      if (url.includes('/metrics/quality')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      }
      if (url.includes('/metrics/performance')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      }
      if (url.includes('/metrics/ingestion')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      }
      if (url.includes('/metrics/detail/law/ai-act')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_LAW_DETAIL) });
      }
      return Promise.resolve({ ok: false, status: 404 });
    });

    render(<MetricsPanel />);

    // Expand quality section (collapsed by default)
    await waitFor(() => {
      expect(screen.getByText(/eval quality/i)).toBeInTheDocument();
    });
    await expandSection(/eval quality/i);

    await waitFor(() => {
      expect(screen.getAllByText('AI Act').length).toBeGreaterThanOrEqual(1);
    });

    const lawEls = screen.getAllByText('AI Act');
    const lawRow = lawEls.find(el => el.closest('tr.cursor-pointer'));
    if (lawRow?.closest('tr')) {
      fireEvent.click(lawRow.closest('tr')!);
    }

    await waitFor(() => {
      expect(screen.getByText(/←/)).toBeInTheDocument();
    });
  });
});


// ─────────────────────────────────────────────────────────────────────────────
// C8d: AI Analysis streaming
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Helper: create a mock ReadableStream for SSE events.
 */
function makeSSEStream(events: Array<{ type: string; text?: string; error?: string }>) {
  const lines = events.map((e) => `data: ${JSON.stringify(e)}\n\n`);
  const text = lines.join('');
  const encoder = new TextEncoder();
  const bytes = encoder.encode(text);

  return new ReadableStream({
    start(controller) {
      // Emit all bytes in one chunk
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

describe('MetricsPanel AI Analysis', () => {
  it('shows streaming text after clicking Analysér', async () => {
    const sseStream = makeSSEStream([
      { type: 'start' },
      { type: 'token', text: 'Hello ' },
      { type: 'token', text: 'world' },
      { type: 'complete' },
    ]);

    mockFetch.mockImplementation((url: string, options?: RequestInit) => {
      if (url.includes('/metrics/analyse') && options?.method === 'POST') {
        return Promise.resolve({ ok: true, body: sseStream });
      }
      if (url.includes('/metrics/overview')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      }
      if (url.includes('/metrics/quality')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      }
      if (url.includes('/metrics/performance')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      }
      if (url.includes('/metrics/ingestion')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      }
      return Promise.resolve({ ok: false, status: 404 });
    });

    render(<MetricsPanel />);

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getAllByText('86.96%').length).toBeGreaterThanOrEqual(1);
    });

    // Click AI analysis button
    fireEvent.click(screen.getByRole('button', { name: /analys/i }));

    // Should show streamed text
    await waitFor(() => {
      expect(screen.getByText(/Hello world/)).toBeInTheDocument();
    });
  });

  it('shows error state when analysis fails', async () => {
    const sseStream = makeSSEStream([
      { type: 'start' },
      { type: 'error', error: 'LLM unavailable' },
    ]);

    mockFetch.mockImplementation((url: string, options?: RequestInit) => {
      if (url.includes('/metrics/analyse') && options?.method === 'POST') {
        return Promise.resolve({ ok: true, body: sseStream });
      }
      if (url.includes('/metrics/overview')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      }
      if (url.includes('/metrics/quality')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      }
      if (url.includes('/metrics/performance')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      }
      if (url.includes('/metrics/ingestion')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      }
      return Promise.resolve({ ok: false, status: 404 });
    });

    render(<MetricsPanel />);

    await waitFor(() => {
      expect(screen.getAllByText('86.96%').length).toBeGreaterThanOrEqual(1);
    });

    fireEvent.click(screen.getByRole('button', { name: /analys/i }));

    await waitFor(() => {
      expect(screen.getByText(/fejl/i)).toBeInTheDocument();
    });
  });

  it('disables button while streaming', async () => {
    // Stream that never completes
    const stream = new ReadableStream({
      start() { /* never close */ },
    });

    mockFetch.mockImplementation((url: string, options?: RequestInit) => {
      if (url.includes('/metrics/analyse') && options?.method === 'POST') {
        return Promise.resolve({ ok: true, body: stream });
      }
      if (url.includes('/metrics/overview')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_OVERVIEW) });
      }
      if (url.includes('/metrics/quality')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_QUALITY) });
      }
      if (url.includes('/metrics/performance')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_PERFORMANCE) });
      }
      if (url.includes('/metrics/ingestion')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(MOCK_INGESTION) });
      }
      return Promise.resolve({ ok: false, status: 404 });
    });

    render(<MetricsPanel />);

    await waitFor(() => {
      expect(screen.getAllByText('86.96%').length).toBeGreaterThanOrEqual(1);
    });

    const button = screen.getByRole('button', { name: /analys/i });
    fireEvent.click(button);

    await waitFor(() => {
      expect(button).toBeDisabled();
    });
  });
});
