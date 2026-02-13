/**
 * Metrics Dashboard panel — trust overview, quality/performance/ingestion
 * breakdowns, drill-down, and AI analysis.
 *
 * Single Responsibility: Render the metrics view with 3-level drill-down.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import { Card } from '@tremor/react';
import { SegmentedControl } from '../Common/SegmentedControl';
import {
  formatTimestamp,
  formatLatency,
  getPassRateColorHex,
  MODE_COLORS,
  MODE_DESCRIPTIONS,
  DIFFICULTY_COLORS,
  DIFFICULTY_DESCRIPTIONS,
  CROSS_LAW_SCORER_LABELS,
  EVAL_SCORER_LABELS,
  EVAL_SCORER_DESCRIPTIONS,
  RUN_MODE_LABELS,
} from './evalUtils';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface TrendInfo {
  direction: string;
  delta_pp: number | null;
  window: number;
  history: number[];
}

interface MetricsSummary {
  total_cases: number;
  law_count: number;
  suite_count: number;
  last_run_timestamp: string | null;
}

interface CategorySummary {
  total: number;
  passed: number;
  pass_rate: number;
  group_pass_rate: number | null;
}

interface OverviewData {
  unified_pass_rate: number;
  health_status: string;
  trend: TrendInfo;
  summary: MetricsSummary;
  single_law: CategorySummary;
  cross_law: CategorySummary;
  has_data: boolean;
  sl_run_mode_distribution: Record<string, number>;
  cl_run_mode_distribution: Record<string, number>;
  sl_case_origin_distribution: Record<string, number>;
  cl_case_origin_distribution: Record<string, number>;
  ingestion_overall_coverage: number;
  ingestion_health_status: string;
  ingestion_na_count: number;
}

interface QualityData {
  per_law: Array<{ law: string; display_name: string; pass_rate: number; total: number; passed: number }>;
  per_suite: Array<{ suite_id: string; name: string; pass_rate: number; total: number; passed: number }>;
  per_mode: Array<{ mode: string; pass_rate: number; total: number; passed: number }>;
  per_difficulty: Array<{ difficulty: string; pass_rate: number; total: number; passed: number }>;
  per_scorer: Array<{ scorer: string; pass_rate: number; total: number; passed: number; category: string }>;
  per_scorer_single_law: Array<{ scorer: string; pass_rate: number; total: number; passed: number; category: string }>;
  stage_stats: { retrieval: number; augmentation: number; generation: number };
  escalation_rate: number;
  retry_rate: number;
}

interface HistogramBin {
  range_start: number;
  range_end: number;
  count: number;
}

interface LatencyTrendPoint {
  timestamp: string;
  median_ms: number;
  case_count: number;
}

interface RateTrendPoint {
  timestamp: string;
  rate: number;
  count: number;
  total: number;
}

interface RunModeLatency {
  run_mode: string;
  p50_seconds: number;
  p95_seconds: number;
  case_count: number;
}

interface RunModeTrend {
  run_mode: string;
  points: LatencyTrendPoint[];
}

interface SLPerformance {
  percentiles: { p50: number; p95: number; p99: number };
  total_cases: number;
  escalation_rate: number;
  retry_rate: number;
  histogram_bins: HistogramBin[];
  latency_by_run_mode: RunModeLatency[];
  trend: LatencyTrendPoint[];
  trend_by_run_mode: RunModeTrend[];
  escalation_trend: RateTrendPoint[];
  retry_trend: RateTrendPoint[];
}

interface CLPerformance {
  percentiles: { p50: number; p95: number; p99: number };
  total_cases: number;
  histogram_bins: HistogramBin[];
  latency_by_synthesis_mode: Array<{ mode: string; p50_seconds: number; case_count: number }>;
  latency_by_difficulty: Array<{ difficulty: string; p50_seconds: number; case_count: number }>;
  trend: LatencyTrendPoint[];
}

interface PerformanceData {
  percentiles: { p50: number; p95: number; p99: number };
  total_cases: number;
  excluded_zero_duration: number;
  histogram_bins: HistogramBin[];
  latency_by_synthesis_mode: Array<{ mode: string; p50_seconds: number; case_count: number }>;
  latency_by_difficulty: Array<{ difficulty: string; p50_seconds: number; case_count: number }>;
  single_law: SLPerformance;
  cross_law: CLPerformance;
}

interface IngestionData {
  overall_coverage: number;
  health_status: string;
  corpora: Array<{
    corpus_id: string;
    display_name: string;
    coverage: number;
    unhandled: number;
    chunks: number;
    is_ingested: boolean;
  }>;
}

// Case result types (shared)
interface CaseResult {
  case_id: string;
  passed: boolean;
  duration_ms: number;
  scores: Record<string, boolean>;
  score_messages?: Record<string, string>;
  // Single-law fields
  anchors?: string[];
  expected_anchors?: string[];
  profile?: string;
  // Cross-law fields
  synthesis_mode?: string;
  difficulty?: string | null;
  prompt?: string;
  target_corpora?: string[];
}

// Detail response types (Level 3)
interface LawDetail {
  law: string;
  display_name: string;
  scorer_breakdown: Array<{ scorer: string; pass_rate: number; total: number; passed: number; category: string }>;
  latest_results: CaseResult[];
}

interface SuiteDetail {
  suite_id: string;
  name: string;
  trend: Array<{ run_id: string; timestamp: string; pass_rate: number }>;
  scorer_breakdown: Array<{ scorer: string; pass_rate: number; total: number; passed: number; category: string }>;
  latest_results: CaseResult[];
  mode_counts: Record<string, number>;
  difficulty_counts: Record<string, number>;
  applied_filters: { mode: string | null; difficulty: string | null };
}

interface ModeDetail {
  mode: string;
  pass_rate: number;
  total: number;
  passed: number;
  applicable_scorers: Array<{ scorer: string; pass_rate: number; total: number; passed: number; category: string }>;
  cases: CaseResult[];
}

interface DifficultyDetail {
  difficulty: string;
  pass_rate: number;
  total: number;
  passed: number;
  cases: CaseResult[];
}

interface ScorerDetail {
  scorer: string;
  label: string;
  per_law_rates: Array<{ law: string; display_name: string; pass_rate: number; total: number; passed: number }>;
}

type DetailData = LawDetail | SuiteDetail | ModeDetail | DifficultyDetail | ScorerDetail;
type DrillType = 'law' | 'suite' | 'scorer' | 'mode' | 'difficulty';

const API_BASE = '/api';

// ─────────────────────────────────────────────────────────────────────────────
// Health badge colour mapping
// ─────────────────────────────────────────────────────────────────────────────

// Pill colour classes — matches CrossLawPanel badge pattern (light bg + coloured text)
const MODE_PILL_CLASSES: Record<string, string> = {
  comparison:  'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400',
  discovery:   'bg-violet-100 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400',
  routing:     'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400',
  aggregation: 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400',
};

const DIFFICULTY_PILL_CLASSES: Record<string, string> = {
  easy:   'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400',
  medium: 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400',
  hard:   'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400',
};

const RUN_MODE_CONFIG: Record<string, { label: string; color: string }> = {
  retrieval_only: { label: 'Retrieval', color: '#94a3b8' },
  full: { label: 'Full', color: '#3b82f6' },
  full_with_judge: { label: 'Full + Judge', color: '#22c55e' },
  unknown: { label: 'Ukendt', color: '#d1d5db' },
};

const ORIGIN_CONFIG: Record<string, { label: string; color: string }> = {
  manual: { label: 'Manuel', color: '#3b82f6' },
  auto: { label: 'Auto', color: '#8b5cf6' },
  'auto-generated': { label: 'Auto-gen', color: '#a855f7' },
  unknown: { label: 'Ukendt', color: '#d1d5db' },
};

const HEALTH_COLORS: Record<string, { bg: string; text: string; ring: string }> = {
  green:  { bg: 'bg-green-500',  text: 'text-white', ring: 'ring-green-200 dark:ring-green-800' },
  yellow: { bg: 'bg-yellow-500', text: 'text-white', ring: 'ring-yellow-200 dark:ring-yellow-800' },
  orange: { bg: 'bg-orange-500', text: 'text-white', ring: 'ring-orange-200 dark:ring-orange-800' },
  red:    { bg: 'bg-red-500',    text: 'text-white', ring: 'ring-red-200 dark:ring-red-800' },
};

const TREND_LABELS: Record<string, string> = {
  improving: 'Stigende',
  declining: 'Faldende',
  stable: 'Stabil',
};

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export function MetricsPanel() {
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [quality, setQuality] = useState<QualityData | null>(null);
  const [performance, setPerformance] = useState<PerformanceData | null>(null);
  const [ingestion, setIngestion] = useState<IngestionData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Drill-down state
  const [drillType, setDrillType] = useState<DrillType | null>(null);
  const [drillId, setDrillId] = useState<string | null>(null);
  const [drillLabel, setDrillLabel] = useState<string>('');
  const [detail, setDetail] = useState<DetailData | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [caseFilter, setCaseFilter] = useState('');
  const [caseStatusFilter, setCaseStatusFilter] = useState<'all' | 'failed' | 'passed'>('all');
  const [casePage, setCasePage] = useState(0);

  // AI Analysis state
  const [analysisState, setAnalysisState] = useState<'idle' | 'streaming' | 'complete' | 'error'>('idle');
  const [analysisText, setAnalysisText] = useState('');

  // Filter + pagination state for per-law and per-suite tables
  const [lawFilter, setLawFilter] = useState('');
  const [lawPage, setLawPage] = useState(0);
  const [suiteFilter, setSuiteFilter] = useState('');
  const [suitePage, setSuitePage] = useState(0);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [overviewRes, qualityRes, performanceRes, ingestionRes] = await Promise.all([
        fetch(`${API_BASE}/eval/metrics/overview`),
        fetch(`${API_BASE}/eval/metrics/quality`),
        fetch(`${API_BASE}/eval/metrics/performance`),
        fetch(`${API_BASE}/eval/metrics/ingestion`),
      ]);

      if (!overviewRes.ok) throw new Error(`Fejl ved hentning af oversigt (${overviewRes.status})`);

      const [overviewData, qualityData, performanceData, ingestionData] = await Promise.all([
        overviewRes.json(),
        qualityRes.ok ? qualityRes.json() : null,
        performanceRes.ok ? performanceRes.json() : null,
        ingestionRes.ok ? ingestionRes.json() : null,
      ]);

      setOverview(overviewData);
      setQuality(qualityData);
      setPerformance(performanceData);
      setIngestion(ingestionData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ukendt fejl');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDrillDown = useCallback(async (type: DrillType, id: string, label: string) => {
    setDrillType(type);
    setDrillId(id);
    setDrillLabel(label);
    setDetailLoading(true);
    setDetail(null);
    setCaseFilter('');
    setCaseStatusFilter('all');
    setCasePage(0);
    try {
      const res = await fetch(`${API_BASE}/eval/metrics/detail/${type}/${encodeURIComponent(id)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setDetail(data);
    } catch {
      setDetail(null);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const handleDrillBack = useCallback(() => {
    setDrillType(null);
    setDrillId(null);
    setDrillLabel('');
    setDetail(null);
  }, []);

  const handleStartAnalysis = useCallback(async () => {
    setAnalysisState('streaming');
    setAnalysisText('');
    try {
      const res = await fetch(`${API_BASE}/eval/metrics/analyse`, { method: 'POST' });
      if (!res.ok || !res.body) {
        setAnalysisState('error');
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const stripped = line.replace(/^data: /, '');
          if (!stripped) continue;
          try {
            const event = JSON.parse(stripped);
            if (event.type === 'token' && event.text) {
              setAnalysisText((prev) => prev + event.text);
            } else if (event.type === 'complete') {
              setAnalysisState('complete');
            } else if (event.type === 'error') {
              setAnalysisState('error');
            }
          } catch {
            // Skip unparseable lines
          }
        }
      }
      // If stream ended without explicit complete/error, mark complete
      setAnalysisState((prev) => (prev === 'streaming' ? 'complete' : prev));
    } catch {
      setAnalysisState('error');
    }
  }, []);

  // Filtered + paginated per-law
  const PAGE_SIZE = 100;
  const filteredLaws = useMemo(() => {
    if (!quality?.per_law) return [];
    const q = lawFilter.toLowerCase();
    return q ? quality.per_law.filter((l) => l.display_name.toLowerCase().includes(q)) : quality.per_law;
  }, [quality, lawFilter]);
  const lawPageCount = Math.max(1, Math.ceil(filteredLaws.length / PAGE_SIZE));
  const pagedLaws = useMemo(() => filteredLaws.slice(lawPage * PAGE_SIZE, (lawPage + 1) * PAGE_SIZE), [filteredLaws, lawPage]);

  // Filtered + paginated per-suite
  const filteredSuites = useMemo(() => {
    if (!quality?.per_suite) return [];
    const q = suiteFilter.toLowerCase();
    return q ? quality.per_suite.filter((s) => s.name.toLowerCase().includes(q)) : quality.per_suite;
  }, [quality, suiteFilter]);
  const suitePageCount = Math.max(1, Math.ceil(filteredSuites.length / PAGE_SIZE));
  const pagedSuites = useMemo(() => filteredSuites.slice(suitePage * PAGE_SIZE, (suitePage + 1) * PAGE_SIZE), [filteredSuites, suitePage]);

  // ── Loading state ──
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">Indlæser metrics...</p>
      </div>
    );
  }

  // ── Error state ──
  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 rounded-2xl p-6 text-center">
        <p className="text-sm text-red-600 dark:text-red-400">Fejl: {error}</p>
        <button
          onClick={fetchData}
          className="mt-2 text-sm text-apple-blue hover:underline"
        >
          Prøv igen
        </button>
      </div>
    );
  }

  // ── Empty state ──
  if (overview && !overview.has_data) {
    return (
      <div className="flex flex-col items-center justify-center py-12 gap-3">
        <svg className="w-12 h-12 text-apple-gray-300 dark:text-apple-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
        <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">Ingen eval data fundet</p>
        <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500">Kør en evaluering for at se metrics</p>
      </div>
    );
  }

  if (!overview) return null;

  const healthColors = HEALTH_COLORS[overview.health_status] || HEALTH_COLORS.red;
  const trendLabel = TREND_LABELS[overview.trend.direction] || overview.trend.direction;

  return (
    <div className="flex flex-col gap-4">
      {/* ── Level 1: Trust Overview ── */}
      <Card className="rounded-2xl shadow-sm">
        <div className="flex flex-col gap-3">
          {/* Main row: Eval Health | Case Oprindelse | Run Mode | Ingestion Health */}
          <div className="flex items-start">
            {/* ── Eval Health ── */}
            <DistributionGroup title="Eval Health">
              <StatusBadge
                label="Samlet"
                passRate={overview.unified_pass_rate}
                total={overview.summary.total_cases}
                healthStatus={overview.health_status}
                trend={overview.trend}
                size="lg"
              />
              <StatusBadge
                label="Single-Law"
                passRate={overview.single_law.pass_rate}
                total={overview.single_law.total}
                subtitle={`${overview.summary.law_count} love`}
              />
              <StatusBadge
                label="Cross-Law"
                passRate={overview.cross_law.pass_rate}
                total={overview.cross_law.total}
                subtitle={`${overview.summary.suite_count} suiter`}
              />
            </DistributionGroup>

            <div className="w-px self-stretch bg-apple-gray-200 dark:bg-apple-gray-500 mx-6 shrink-0" />

            {/* ── Case Oprindelse (SL + CL) ── */}
            <DistributionGroup title="Case Oprindelse">
              <DistributionDonut
                label="Single-Law"
                entries={Object.entries(overview.sl_case_origin_distribution || {})}
                configMap={ORIGIN_CONFIG}
                manualPct={getManualPct(overview.sl_case_origin_distribution)}
              />
              <DistributionDonut
                label="Cross-Law"
                entries={Object.entries(overview.cl_case_origin_distribution || {})}
                configMap={ORIGIN_CONFIG}
                manualPct={getManualPct(overview.cl_case_origin_distribution)}
              />
            </DistributionGroup>

            <div className="w-px self-stretch bg-apple-gray-200 dark:bg-apple-gray-500 mx-6 shrink-0" />

            {/* ── Run Mode (SL + CL) ── */}
            <DistributionGroup title="Run Mode">
              <DistributionDonut
                label="Single-Law"
                entries={Object.entries(overview.sl_run_mode_distribution || {})}
                configMap={RUN_MODE_CONFIG}
                warningFn={(entries) => {
                  const total = entries.reduce((s, [, v]) => s + v, 0);
                  const judge = (overview.sl_run_mode_distribution || {})['full_with_judge'] || 0;
                  const pct = total > 0 ? Math.round(judge / total * 100) : 0;
                  if (pct < 30 && total > 0) return { text: `⚠ Få LLM-judge (${pct}%)`, color: pct < 10 ? 'text-red-600 dark:text-red-400' : 'text-orange-600 dark:text-orange-400' };
                  return null;
                }}
              />
              <DistributionDonut
                label="Cross-Law"
                entries={Object.entries(overview.cl_run_mode_distribution || {})}
                configMap={RUN_MODE_CONFIG}
                warningFn={(entries) => {
                  const total = entries.reduce((s, [, v]) => s + v, 0);
                  const judge = (overview.cl_run_mode_distribution || {})['full_with_judge'] || 0;
                  const pct = total > 0 ? Math.round(judge / total * 100) : 0;
                  if (pct < 30 && total > 0) return { text: `⚠ Få LLM-judge (${pct}%)`, color: pct < 10 ? 'text-red-600 dark:text-red-400' : 'text-orange-600 dark:text-orange-400' };
                  return null;
                }}
              />
            </DistributionGroup>

            <div className="w-px self-stretch bg-apple-gray-200 dark:bg-apple-gray-500 mx-6 shrink-0" />

            {/* ── Ingestion Health ── */}
            <DistributionGroup title="Ingestion Health">
              <HealthBubble
                label="HTML-tjek"
                value={`${overview.ingestion_overall_coverage}%`}
                status={overview.ingestion_health_status}
                subtitle={overview.ingestion_na_count > 0
                  ? `${overview.ingestion_na_count} ikke tjekket`
                  : undefined}
              />
              <HealthBubble
                label="Love"
                value={`${overview.summary.law_count}`}
                status="green"
                subtitle="indekseret"
              />
            </DistributionGroup>

            {performance && (
              <>
                <div className="w-px self-stretch bg-apple-gray-200 dark:bg-apple-gray-500 mx-6 shrink-0" />

                {/* ── Performance ── */}
                <DistributionGroup title="Latency">
                  <PerformanceBadge
                    label="Median"
                    value={formatLatency(performance.percentiles.p50 * 1000)}
                    subtitle={`${performance.total_cases} cases`}
                    thresholds={getLatencyColor(performance.percentiles.p50, 'median')}
                    size="lg"
                  />
                  <PerformanceBadge
                    label="P95"
                    value={formatLatency(performance.percentiles.p95 * 1000)}
                    subtitle="95% er hurtigere"
                    thresholds={getLatencyColor(performance.percentiles.p95, 'p95')}
                  />
                </DistributionGroup>

                {performance.single_law && performance.single_law.total_cases > 0 && (
                  <>
                    <div className="w-px self-stretch bg-apple-gray-200 dark:bg-apple-gray-500 mx-6 shrink-0" />

                    <DistributionGroup title="Stabilitet">
                      <PerformanceBadge
                        label="Eskalering"
                        value={`${performance.single_law.escalation_rate.toFixed(1)}%`}
                        subtitle={`${performance.single_law.total_cases} SL cases`}
                        thresholds={getRateColor(performance.single_law.escalation_rate, 'escalation')}
                      />
                      <PerformanceBadge
                        label="Genforsøg"
                        value={`${performance.single_law.retry_rate.toFixed(1)}%`}
                        subtitle={`${performance.single_law.total_cases} SL cases`}
                        thresholds={getRateColor(performance.single_law.retry_rate, 'retry')}
                      />
                    </DistributionGroup>
                  </>
                )}
              </>
            )}
          </div>

          {/* Summary row */}
          <div className="flex items-center gap-4 text-xs text-apple-gray-500 dark:text-apple-gray-400 border-t border-apple-gray-100 dark:border-apple-gray-500 pt-3">
            <span>{overview.summary.total_cases} cases</span>
            <span>·</span>
            <span>{overview.summary.law_count} love</span>
            <span>·</span>
            <span>{overview.summary.suite_count} suiter</span>
            {overview.summary.last_run_timestamp && (
              <>
                <span>·</span>
                <span>Seneste: {formatTimestamp(overview.summary.last_run_timestamp)}</span>
              </>
            )}
          </div>
        </div>
      </Card>

      {/* ── AI Analysis (always visible) ── */}
      <Card className="rounded-2xl shadow-sm">
        <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-500">
          <h3 className="text-base font-semibold text-apple-gray-700 dark:text-white">AI Analyse</h3>
          <div className="flex items-center gap-3 mt-2">
            <button
              onClick={handleStartAnalysis}
              disabled={analysisState === 'streaming'}
              className="px-3 py-1.5 text-sm font-medium rounded-full transition-colors flex items-center gap-1.5 bg-apple-blue/10 text-apple-blue hover:bg-apple-blue/20 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Analysér metrics med AI"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              {analysisState === 'streaming' ? 'Analyserer...' : 'Analysér'}
            </button>
            <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
              AI-baseret gennemgang af eval-resultater. Identificerer mønstre, risici og forbedringsforslag.
            </p>
          </div>
        </div>
        <div className="p-4">
          {analysisState === 'error' && (
            <p className="text-sm text-red-600 dark:text-red-400">Fejl under analyse</p>
          )}
          {analysisText && (() => {
            const glossarySplit = analysisText.split(/## ORDLISTE/i);
            const mainText = glossarySplit[0];
            const glossaryText = glossarySplit[1] || null;
            return (
              <>
                <div className="prose prose-sm dark:prose-invert max-w-none text-apple-gray-700 dark:text-apple-gray-200 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:uppercase [&_h2]:tracking-wider [&_h2]:text-apple-gray-500 [&_h2]:dark:text-apple-gray-400 [&_h2]:mt-5 [&_h2]:mb-2 [&_h2:first-child]:mt-0 [&_ol]:pl-5 [&_ol]:space-y-3 [&_ul]:pl-5 [&_ul]:space-y-1 [&_li]:text-sm [&_p]:text-sm [&_p]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-apple-gray-200 [&_blockquote]:dark:border-apple-gray-600 [&_blockquote]:pl-3 [&_blockquote]:text-xs [&_blockquote]:text-apple-gray-400 [&_strong]:text-apple-gray-800 [&_strong]:dark:text-apple-gray-100">
                  <ReactMarkdown>{mainText}</ReactMarkdown>
                </div>
                {glossaryText && (
                  <div className="mt-4 pt-3 border-t border-apple-gray-100 dark:border-apple-gray-700">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-apple-gray-400 dark:text-apple-gray-500 mb-2">Ordliste</p>
                    <div className="prose prose-sm dark:prose-invert max-w-none text-xs text-apple-gray-400 dark:text-apple-gray-500 [&_p]:text-xs [&_p]:my-0.5 [&_p]:text-apple-gray-400 [&_p]:dark:text-apple-gray-500 [&_strong]:text-apple-gray-500 [&_strong]:dark:text-apple-gray-400">
                      <ReactMarkdown>{glossaryText.trim()}</ReactMarkdown>
                    </div>
                  </div>
                )}
              </>
            );
          })()}
          {analysisState === 'streaming' && !analysisText && (
            <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">Analyserer...</p>
          )}
        </div>
      </Card>

      {/* ── Level 2: Category Panels ── */}

      {/* Eval Quality Panel */}
      {quality && (
        <CollapsibleSection title="Eval Quality">
          <div>
            {/* Two-section layout: Single-Law | Cross-Law */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* ── Single-Law Section ── */}
              <div>
                <p className="text-xs font-medium text-apple-blue uppercase tracking-wider mb-2">
                  Single-Law
                </p>

                {/* Per-law filter */}
                <input
                  type="text"
                  placeholder="Søg lov..."
                  value={lawFilter}
                  onChange={(e) => { setLawFilter(e.target.value); setLawPage(0); }}
                  className="w-full text-xs px-2 py-1 mb-1 rounded-lg border border-apple-gray-100 dark:border-apple-gray-500 bg-apple-gray-50 dark:bg-apple-gray-600 text-apple-gray-700 dark:text-white placeholder-apple-gray-400 focus:outline-none focus:ring-1 focus:ring-apple-blue"
                />

                {/* Per-law pass rates */}
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-apple-gray-500 dark:text-apple-gray-400">
                      <th className="text-left font-medium py-1">Lov</th>
                      <th className="text-right font-medium py-1">Pass rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pagedLaws.map((law) => (
                      <tr
                        key={law.law}
                        className="border-t border-apple-gray-100 dark:border-apple-gray-500 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-600/50 cursor-pointer"
                        tabIndex={0}
                        role="button"
                        onClick={() => handleDrillDown('law', law.law, law.display_name)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleDrillDown('law', law.law, law.display_name); } }}
                      >
                        <td className="py-1.5 text-apple-gray-700 dark:text-white">{law.display_name}</td>
                        <td className="py-1.5 text-right font-medium" style={{ color: getPassRateColorHex(law.pass_rate) }}>
                          {law.pass_rate.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {lawPageCount > 1 && (
                  <div className="flex items-center justify-center gap-1 mt-1 text-xs text-apple-gray-500 dark:text-apple-gray-400">
                    <button
                      onClick={() => setLawPage((p) => Math.max(0, p - 1))}
                      disabled={lawPage === 0}
                      className="px-1.5 py-0.5 rounded hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-30"
                      aria-label="Forrige"
                    >
                      ‹
                    </button>
                    <span>{lawPage + 1} / {lawPageCount}</span>
                    <button
                      onClick={() => setLawPage((p) => Math.min(lawPageCount - 1, p + 1))}
                      disabled={lawPage >= lawPageCount - 1}
                      className="px-1.5 py-0.5 rounded hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-30"
                      aria-label="Næste"
                    >
                      ›
                    </button>
                  </div>
                )}

                {/* Single-law scorers with bars */}
                {quality.per_scorer_single_law.length > 0 && (
                  <div className="mt-4">
                    <p className="text-xs font-medium text-apple-gray-500 dark:text-apple-gray-400 mb-2">Scorers</p>
                    <div className="space-y-1.5">
                      {quality.per_scorer_single_law.map((s) => (
                        <div key={s.scorer} title={EVAL_SCORER_DESCRIPTIONS[s.scorer] || ''}>
                          <div className="flex items-center justify-between text-xs mb-0.5">
                            <span className="text-apple-gray-700 dark:text-white">{EVAL_SCORER_LABELS[s.scorer] || s.scorer}</span>
                            <span className="font-medium" style={{ color: getPassRateColorHex(s.pass_rate) }}>{s.pass_rate.toFixed(1)}%</span>
                          </div>
                          <div className="h-1.5 w-full bg-apple-gray-100 dark:bg-apple-gray-500 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{ width: `${Math.min(s.pass_rate, 100)}%`, backgroundColor: getPassRateColorHex(s.pass_rate) }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Pipeline stages (correctness metrics) */}
                <div className="flex items-center gap-3 flex-wrap text-[10px] text-apple-gray-400 mt-3 pt-2 border-t border-apple-gray-100 dark:border-apple-gray-500">
                  <span>
                    Retrieval <span className="font-medium" style={{ color: getPassRateColorHex(quality.stage_stats.retrieval) }}>{quality.stage_stats.retrieval.toFixed(0)}%</span>
                  </span>
                  <span>
                    Augm. <span className="font-medium" style={{ color: getPassRateColorHex(quality.stage_stats.augmentation) }}>{quality.stage_stats.augmentation.toFixed(0)}%</span>
                  </span>
                  <span>
                    Gen. <span className="font-medium" style={{ color: getPassRateColorHex(quality.stage_stats.generation) }}>{quality.stage_stats.generation.toFixed(0)}%</span>
                  </span>
                </div>
              </div>

              {/* ── Cross-Law Section ── */}
              <div>
                <p className="text-xs font-medium text-apple-blue uppercase tracking-wider mb-2">
                  Cross-Law
                </p>

                {/* Per-suite filter */}
                <input
                  type="text"
                  placeholder="Søg suite..."
                  value={suiteFilter}
                  onChange={(e) => { setSuiteFilter(e.target.value); setSuitePage(0); }}
                  className="w-full text-xs px-2 py-1 mb-1 rounded-lg border border-apple-gray-100 dark:border-apple-gray-500 bg-apple-gray-50 dark:bg-apple-gray-600 text-apple-gray-700 dark:text-white placeholder-apple-gray-400 focus:outline-none focus:ring-1 focus:ring-apple-blue"
                />

                {/* Per-suite pass rates */}
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-apple-gray-500 dark:text-apple-gray-400">
                      <th className="text-left font-medium py-1">Suite</th>
                      <th className="text-right font-medium py-1">Pass rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pagedSuites.map((suite) => (
                      <tr
                        key={suite.suite_id}
                        className="border-t border-apple-gray-100 dark:border-apple-gray-500 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-600/50 cursor-pointer"
                        tabIndex={0}
                        role="button"
                        onClick={() => handleDrillDown('suite', suite.suite_id, suite.name)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleDrillDown('suite', suite.suite_id, suite.name); } }}
                      >
                        <td className="py-1.5 text-apple-gray-700 dark:text-white">{suite.name}</td>
                        <td className="py-1.5 text-right font-medium" style={{ color: getPassRateColorHex(suite.pass_rate) }}>
                          {suite.pass_rate.toFixed(1)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {suitePageCount > 1 && (
                  <div className="flex items-center justify-center gap-1 mt-1 text-xs text-apple-gray-500 dark:text-apple-gray-400">
                    <button
                      onClick={() => setSuitePage((p) => Math.max(0, p - 1))}
                      disabled={suitePage === 0}
                      className="px-1.5 py-0.5 rounded hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-30"
                      aria-label="Forrige"
                    >
                      ‹
                    </button>
                    <span>{suitePage + 1} / {suitePageCount}</span>
                    <button
                      onClick={() => setSuitePage((p) => Math.min(suitePageCount - 1, p + 1))}
                      disabled={suitePage >= suitePageCount - 1}
                      className="px-1.5 py-0.5 rounded hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-30"
                      aria-label="Næste"
                    >
                      ›
                    </button>
                  </div>
                )}

                {/* Per mode + per difficulty side by side */}
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-apple-gray-500 dark:text-apple-gray-400">
                        <th className="text-left font-medium py-1">Tilstand</th>
                        <th className="text-right font-medium py-1">Pass rate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {quality.per_mode.map((m) => (
                        <tr
                          key={m.mode}
                          className="border-t border-apple-gray-100 dark:border-apple-gray-500 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-600/50 cursor-pointer"
                          tabIndex={0}
                          role="button"
                          onClick={() => handleDrillDown('mode', m.mode, m.mode)}
                          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleDrillDown('mode', m.mode, m.mode); } }}
                          title={MODE_DESCRIPTIONS[m.mode] || ''}
                        >
                          <td className="py-1.5 text-apple-gray-700 dark:text-white capitalize">{m.mode}</td>
                          <td className="py-1.5 text-right font-medium" style={{ color: MODE_COLORS[m.mode]?.hex || '#6b7280' }}>
                            {m.pass_rate.toFixed(0)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-apple-gray-500 dark:text-apple-gray-400">
                        <th className="text-left font-medium py-1">Sværhedsgrad</th>
                        <th className="text-right font-medium py-1">Pass rate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {quality.per_difficulty.map((d) => (
                        <tr
                          key={d.difficulty}
                          className="border-t border-apple-gray-100 dark:border-apple-gray-500 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-600/50 cursor-pointer"
                          tabIndex={0}
                          role="button"
                          onClick={() => handleDrillDown('difficulty', d.difficulty, d.difficulty)}
                          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleDrillDown('difficulty', d.difficulty, d.difficulty); } }}
                          title={DIFFICULTY_DESCRIPTIONS[d.difficulty] || ''}
                        >
                          <td className="py-1.5 text-apple-gray-700 dark:text-white capitalize">{d.difficulty}</td>
                          <td className="py-1.5 text-right font-medium" style={{ color: DIFFICULTY_COLORS[d.difficulty]?.hex || '#6b7280' }}>
                            {d.pass_rate.toFixed(0)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Cross-law scorers with bars */}
                <div className="mt-4">
                  <p className="text-xs font-medium text-apple-gray-500 dark:text-apple-gray-400 mb-2">Scorers</p>
                  <div className="space-y-1.5">
                    {quality.per_scorer.map((s) => (
                      <div key={s.scorer} title={EVAL_SCORER_DESCRIPTIONS[s.scorer] || ''}>
                        <div className="flex items-center justify-between text-xs mb-0.5">
                          <span className="text-apple-gray-700 dark:text-white">{CROSS_LAW_SCORER_LABELS[s.scorer] || EVAL_SCORER_LABELS[s.scorer] || s.scorer}</span>
                          <span className="font-medium" style={{ color: getPassRateColorHex(s.pass_rate) }}>{s.pass_rate.toFixed(1)}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-apple-gray-100 dark:bg-apple-gray-500 rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{ width: `${Math.min(s.pass_rate, 100)}%`, backgroundColor: getPassRateColorHex(s.pass_rate) }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </CollapsibleSection>
      )}

      {/* ── Level 3: Drill-down Detail (between Quality and Performance) ── */}
      {drillType && (
        <div className="rounded-2xl bg-white dark:bg-apple-gray-700 shadow-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-500">
            <div className="flex items-center gap-2">
              <button
                onClick={handleDrillBack}
                className="text-apple-blue hover:text-blue-600 text-sm font-medium"
              >
                ←
              </button>
              <h3 className="text-base font-semibold text-apple-gray-700 dark:text-white capitalize">
                {drillLabel}
              </h3>
            </div>
          </div>
          <div className="p-4">
            {detailLoading && (
              <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">Indlæser detaljer...</p>
            )}

            {/* Law detail */}
            {detail && drillType === 'law' && (() => {
              const d = detail as LawDetail;
              return (
                <>
                  {d.scorer_breakdown?.length > 0 && (
                    <div className="mb-4">
                      <p className="text-xs font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider mb-2">Scorer breakdown</p>
                      {d.scorer_breakdown.map((s) => (
                        <div key={s.scorer} className="flex items-center justify-between py-1 px-2 border-t border-apple-gray-100 dark:border-apple-gray-500" title={EVAL_SCORER_DESCRIPTIONS[s.scorer] || ''}>
                          <span className="text-xs text-apple-gray-600 dark:text-apple-gray-300">
                            {EVAL_SCORER_LABELS[s.scorer] || s.scorer}
                          </span>
                          <span className="text-xs font-medium" style={{ color: getPassRateColorHex(s.pass_rate) }}>
                            {s.pass_rate.toFixed(1)}% ({s.passed}/{s.total})
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  {d.latest_results?.length > 0 && (
                    <CaseResultsTable cases={d.latest_results} filter={caseFilter} onFilterChange={(v) => { setCaseFilter(v); setCasePage(0); }} statusFilter={caseStatusFilter} onStatusFilterChange={(v) => { setCaseStatusFilter(v); setCasePage(0); }} page={casePage} onPageChange={setCasePage} />
                  )}
                </>
              );
            })()}

            {/* Mode detail */}
            {detail && drillType === 'mode' && (() => {
              const d = detail as ModeDetail;
              return (
                <>
                  {d.applicable_scorers?.length > 0 && (
                    <div className="mb-4">
                      <p className="text-xs font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider mb-2">Applicable scorers</p>
                      {d.applicable_scorers.map((s) => (
                        <div key={s.scorer} className="flex items-center justify-between py-1 px-2 border-t border-apple-gray-100 dark:border-apple-gray-500" title={EVAL_SCORER_DESCRIPTIONS[s.scorer] || ''}>
                          <span className="text-xs text-apple-gray-600 dark:text-apple-gray-300">
                            {CROSS_LAW_SCORER_LABELS[s.scorer] || EVAL_SCORER_LABELS[s.scorer] || s.scorer}
                          </span>
                          <span className="text-xs font-medium" style={{ color: getPassRateColorHex(s.pass_rate) }}>
                            {s.pass_rate.toFixed(1)}% ({s.passed}/{s.total})
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  {d.cases?.length > 0 && (
                    <CaseResultsTable cases={d.cases} filter={caseFilter} onFilterChange={(v) => { setCaseFilter(v); setCasePage(0); }} statusFilter={caseStatusFilter} onStatusFilterChange={(v) => { setCaseStatusFilter(v); setCasePage(0); }} page={casePage} onPageChange={setCasePage} />
                  )}
                </>
              );
            })()}

            {/* Difficulty detail */}
            {detail && drillType === 'difficulty' && (() => {
              const d = detail as DifficultyDetail;
              return d.cases?.length > 0 ? <CaseResultsTable cases={d.cases} filter={caseFilter} onFilterChange={(v) => { setCaseFilter(v); setCasePage(0); }} statusFilter={caseStatusFilter} onStatusFilterChange={(v) => { setCaseStatusFilter(v); setCasePage(0); }} page={casePage} onPageChange={setCasePage} /> : null;
            })()}

            {/* Suite detail */}
            {detail && drillType === 'suite' && (() => {
              const d = detail as SuiteDetail;
              return (
                <>
                  {d.scorer_breakdown?.length > 0 && (
                    <div className="mb-4">
                      <p className="text-xs font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider mb-2">Scorer breakdown</p>
                      {d.scorer_breakdown.map((s) => (
                        <div key={s.scorer} className="flex items-center justify-between py-1 px-2 border-t border-apple-gray-100 dark:border-apple-gray-500" title={EVAL_SCORER_DESCRIPTIONS[s.scorer] || ''}>
                          <span className="text-xs text-apple-gray-600 dark:text-apple-gray-300">
                            {CROSS_LAW_SCORER_LABELS[s.scorer] || EVAL_SCORER_LABELS[s.scorer] || s.scorer}
                          </span>
                          <span className="text-xs font-medium" style={{ color: getPassRateColorHex(s.pass_rate) }}>
                            {s.pass_rate.toFixed(1)}% ({s.passed}/{s.total})
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  {d.latest_results?.length > 0 && (
                    <CaseResultsTable cases={d.latest_results} filter={caseFilter} onFilterChange={(v) => { setCaseFilter(v); setCasePage(0); }} statusFilter={caseStatusFilter} onStatusFilterChange={(v) => { setCaseStatusFilter(v); setCasePage(0); }} page={casePage} onPageChange={setCasePage} />
                  )}
                </>
              );
            })()}

            {/* Scorer detail */}
            {detail && drillType === 'scorer' && (() => {
              const d = detail as ScorerDetail;
              return d.per_law_rates?.length > 0 ? (
                <div>
                  {d.per_law_rates.map((l) => (
                    <div key={l.law} className="flex items-center justify-between py-1 px-2">
                      <span className="text-xs text-apple-gray-600 dark:text-apple-gray-300">{l.display_name}</span>
                      <span className="text-xs font-medium" style={{ color: getPassRateColorHex(l.pass_rate) }}>
                        {l.pass_rate.toFixed(1)}% ({l.passed}/{l.total})
                      </span>
                    </div>
                  ))}
                </div>
              ) : null;
            })()}
          </div>
        </div>
      )}

      {/* Processing Performance Panel */}
      {performance && (
        <CollapsibleSection title="Processing Performance">
          <div className="space-y-6">

            {/* ── Single-Law Section ───────────────────────────────── */}
            {performance.single_law && performance.single_law.total_cases > 0 && (() => {
              const sl = performance.single_law;
              return (
                <div>
                  <p className="text-xs font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider mb-3">Single-Law</p>

                  {/* Compact inline stat row with vertical dividers */}
                  <div className="flex items-baseline divide-x divide-apple-gray-200 dark:divide-apple-gray-500 mb-4">
                    <div className="pr-4" title="Halvdelen af alle test cases er hurtigere end denne tid (median)">
                      <span className="text-[10px] text-apple-gray-400 block">Median/case</span>
                      <span className="text-sm font-semibold text-apple-gray-700 dark:text-white">{formatLatency(sl.percentiles.p50 * 1000)}</span>
                    </div>
                    <div className="px-4" title="95% af alle test cases er hurtigere end denne tid">
                      <span className="text-[10px] text-apple-gray-400 block">P95/case</span>
                      <span className="text-sm font-semibold text-apple-gray-700 dark:text-white">{formatLatency(sl.percentiles.p95 * 1000)}</span>
                    </div>
                    <div className="px-4" title="Andel af kald der blev eskaleret til en stærkere model">
                      <span className="text-[10px] text-apple-gray-400 block">Eskalering</span>
                      <span className="text-sm font-semibold text-amber-600 dark:text-amber-400">{sl.escalation_rate.toFixed(1)}%</span>
                    </div>
                    <div className="px-4" title="Andel af kald der krævede genforsøg">
                      <span className="text-[10px] text-apple-gray-400 block">Genforsøg</span>
                      <span className="text-sm font-semibold text-orange-600 dark:text-orange-400">{sl.retry_rate.toFixed(1)}%</span>
                    </div>
                    <div className="pl-4" title="Antal evaluerede test cases">
                      <span className="text-[10px] text-apple-gray-400 block">Cases</span>
                      <span className="text-sm font-semibold text-apple-gray-700 dark:text-white">{sl.total_cases}</span>
                    </div>
                  </div>

                  {/* Runtime by run mode — horizontal bar chart (per test case) */}
                  {sl.latency_by_run_mode.length > 0 && (() => {
                    const maxP95 = Math.max(...sl.latency_by_run_mode.map(m => m.p95_seconds));
                    return (
                      <div className="mb-5">
                        <p className="text-[10px] font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider mb-2">Latency pr. test case · tilstand</p>
                        <div className="space-y-2">
                          {sl.latency_by_run_mode.map((m) => {
                            const label = RUN_MODE_LABELS[m.run_mode as keyof typeof RUN_MODE_LABELS]?.label || m.run_mode;
                            const desc = RUN_MODE_LABELS[m.run_mode as keyof typeof RUN_MODE_LABELS]?.description || '';
                            const p50Pct = maxP95 > 0 ? (m.p50_seconds / maxP95) * 100 : 0;
                            const p95Pct = maxP95 > 0 ? (m.p95_seconds / maxP95) * 100 : 0;
                            return (
                              <div key={m.run_mode} title={desc}>
                                <div className="flex items-center justify-between mb-0.5">
                                  <span className="text-xs text-apple-gray-700 dark:text-white">{label}</span>
                                  <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                                    {formatLatency(m.p50_seconds * 1000)} <span className="text-apple-gray-300 dark:text-apple-gray-500">/ {formatLatency(m.p95_seconds * 1000)}</span>
                                    <span className="ml-1 text-[10px] text-apple-gray-400">({m.case_count})</span>
                                  </span>
                                </div>
                                <div className="h-3 bg-apple-gray-100 dark:bg-apple-gray-600 rounded-full overflow-hidden relative">
                                  <div className="absolute inset-y-0 left-0 bg-blue-200 dark:bg-blue-900 rounded-full transition-all" style={{ width: `${Math.max(p95Pct, 2)}%` }} />
                                  <div className="absolute inset-y-0 left-0 bg-apple-blue rounded-full transition-all" style={{ width: `${Math.max(p50Pct, 2)}%` }} />
                                </div>
                              </div>
                            );
                          })}
                          <div className="flex items-center gap-3 text-[9px] text-apple-gray-400">
                            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-apple-blue rounded-full inline-block" />Median</span>
                            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-blue-200 dark:bg-blue-900 rounded-full inline-block" />P95 (95% er hurtigere)</span>
                          </div>
                        </div>
                      </div>
                    );
                  })()}

                  {/* SL Trends — side-by-side with vertical dividers */}
                  {(() => {
                    const RUN_MODE_CHART_COLORS: Record<string, { line: string; dot: string }> = {
                      retrieval_only: { line: '#22c55e', dot: '#86efac' },
                      full: { line: '#3b82f6', dot: '#93c5fd' },
                      full_with_judge: { line: '#8b5cf6', dot: '#c4b5fd' },
                    };
                    // Collect all chart configs into one array
                    const charts: Array<{
                      key: string; label: string; values: number[]; timestamps: string[];
                      formatValue: (v: number) => string; formatDelta: (d: number) => string;
                      lineColor: string; dotColor: string;
                      tooltipExtra?: (i: number) => string;
                      invertDelta?: boolean;
                    }> = [];

                    // Per-run-mode latency trends
                    if (sl.trend_by_run_mode) {
                      for (const mt of sl.trend_by_run_mode) {
                        if (mt.points.length < 2) continue;
                        const colors = RUN_MODE_CHART_COLORS[mt.run_mode] || { line: '#6b7280', dot: '#d1d5db' };
                        const modeLabel = RUN_MODE_LABELS[mt.run_mode as keyof typeof RUN_MODE_LABELS]?.label || mt.run_mode;
                        charts.push({
                          key: `mode-${mt.run_mode}`,
                          label: `${modeLabel} pr. testcase`,
                          values: mt.points.map(p => p.median_ms),
                          timestamps: mt.points.map(p => p.timestamp),
                          formatValue: (v) => formatLatency(v),
                          formatDelta: (d) => formatLatency(Math.abs(d)),
                          lineColor: colors.line,
                          dotColor: colors.dot,
                          tooltipExtra: (i) => `${mt.points[i].case_count} cases`,
                        });
                      }
                    }

                    // Escalation trend
                    if (sl.escalation_trend && sl.escalation_trend.length > 1) {
                      charts.push({
                        key: 'escalation',
                        label: 'Eskalering pr. testcase',
                        values: sl.escalation_trend.map(p => p.rate),
                        timestamps: sl.escalation_trend.map(p => p.timestamp),
                        formatValue: (v) => `${v.toFixed(1)}%`,
                        formatDelta: (d) => `${Math.abs(d).toFixed(1)}pp`,
                        lineColor: '#d97706',
                        dotColor: '#fcd34d',
                        tooltipExtra: (i) => `${sl.escalation_trend[i].count}/${sl.escalation_trend[i].total} cases`,
                      });
                    }

                    // Retry trend
                    if (sl.retry_trend && sl.retry_trend.length > 1) {
                      charts.push({
                        key: 'retry',
                        label: 'Genforsøg pr. testcase',
                        values: sl.retry_trend.map(p => p.rate),
                        timestamps: sl.retry_trend.map(p => p.timestamp),
                        formatValue: (v) => `${v.toFixed(1)}%`,
                        formatDelta: (d) => `${Math.abs(d).toFixed(1)}pp`,
                        lineColor: '#ea580c',
                        dotColor: '#fdba74',
                        tooltipExtra: (i) => `${sl.retry_trend[i].count}/${sl.retry_trend[i].total} cases`,
                      });
                    }

                    if (charts.length === 0) return null;

                    // Shared chart dimensions (taller for side-by-side)
                    const svgW = 240; const svgH = 200;
                    const chartL = 42; const chartR = 235; const chartT = 8; const chartB = 190;
                    const chartW = chartR - chartL; const chartH = chartB - chartT;

                    return (
                      <div className="flex divide-x divide-apple-gray-200 dark:divide-apple-gray-500 mb-2 -mx-2">
                        {charts.map((chart) => {
                          const { values, timestamps } = chart;
                          const minV = Math.min(...values);
                          const maxV = Math.max(...values);
                          const midV = (minV + maxV) / 2;
                          const range = maxV - minV || 1;
                          const latest = values[values.length - 1];
                          const prev = values[values.length - 2];
                          const delta = latest - prev;
                          const deltaColor = delta > 0 ? 'text-red-500' : delta < 0 ? 'text-green-500' : 'text-apple-gray-400';
                          const step = chartW / (values.length - 1);
                          const toX = (i: number) => chartL + i * step;
                          const toY = (v: number) => chartB - ((v - minV) / range) * chartH;
                          const pts = values.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');

                          return (
                            <div key={chart.key} className="flex-1 min-w-0 px-3 first:pl-0 last:pr-0">
                              <div className="flex items-center justify-between mb-1">
                                <p className="text-[10px] font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider truncate">{chart.label}</p>
                                <span className={`text-[10px] font-medium ${deltaColor} ml-1 shrink-0`}>
                                  {delta > 0 ? '↑' : delta < 0 ? '↓' : '→'} {chart.formatDelta(delta)}
                                </span>
                              </div>
                              <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full" style={{ height: 200 }} role="img" aria-label={`${chart.label} trend`}>
                                {/* Grid lines */}
                                <line x1={chartL} y1={chartT} x2={chartR} y2={chartT} stroke="#e5e7eb" strokeWidth="0.5" strokeDasharray="4 3" />
                                <line x1={chartL} y1={(chartT + chartB) / 2} x2={chartR} y2={(chartT + chartB) / 2} stroke="#e5e7eb" strokeWidth="0.5" strokeDasharray="4 3" />
                                <line x1={chartL} y1={chartB} x2={chartR} y2={chartB} stroke="#e5e7eb" strokeWidth="0.5" strokeDasharray="4 3" />
                                {/* Y-axis labels */}
                                <text x={chartL - 4} y={chartT + 4} textAnchor="end" fontSize="9" fill="#9ca3af">{chart.formatValue(maxV)}</text>
                                <text x={chartL - 4} y={(chartT + chartB) / 2 + 3} textAnchor="end" fontSize="9" fill="#9ca3af">{chart.formatValue(midV)}</text>
                                <text x={chartL - 4} y={chartB + 4} textAnchor="end" fontSize="9" fill="#9ca3af">{chart.formatValue(minV)}</text>
                                {/* Line */}
                                <polyline points={pts} fill="none" stroke={chart.lineColor} strokeWidth="2" strokeLinejoin="round" />
                                {/* Data points with tooltips */}
                                {values.map((v, i) => (
                                  <circle key={i} cx={toX(i)} cy={toY(v)} r="3.5" fill={i === values.length - 1 ? chart.lineColor : chart.dotColor} stroke="white" strokeWidth="1" style={{ cursor: 'pointer' }}>
                                    <title>{`${formatTimestamp(timestamps[i])}\n${chart.formatValue(v)}${chart.tooltipExtra ? ` · ${chart.tooltipExtra(i)}` : ''}`}</title>
                                  </circle>
                                ))}
                              </svg>
                              <div className="flex justify-between text-[9px] text-apple-gray-400 mt-0.5">
                                <span>{formatTimestamp(timestamps[0])}</span>
                                <span>{formatTimestamp(timestamps[timestamps.length - 1])}</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()}
                </div>
              );
            })()}

            {/* Divider between SL and CL */}
            {performance.single_law?.total_cases > 0 && performance.cross_law?.total_cases > 0 && (
              <div className="border-t border-apple-gray-100 dark:border-apple-gray-600" />
            )}

            {/* ── Cross-Law Section ────────────────────────────────── */}
            {performance.cross_law && performance.cross_law.total_cases > 0 && (() => {
              const cl = performance.cross_law;
              return (
                <div>
                  <p className="text-xs font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider mb-3">Cross-Law</p>

                  {/* Compact inline stat row with vertical dividers */}
                  <div className="flex items-baseline divide-x divide-apple-gray-200 dark:divide-apple-gray-500 mb-4">
                    <div className="pr-4" title="Halvdelen af alle test cases er hurtigere end denne tid (median)">
                      <span className="text-[10px] text-apple-gray-400 block">Median/case</span>
                      <span className="text-sm font-semibold text-apple-gray-700 dark:text-white">{formatLatency(cl.percentiles.p50 * 1000)}</span>
                    </div>
                    <div className="px-4" title="95% af alle test cases er hurtigere end denne tid">
                      <span className="text-[10px] text-apple-gray-400 block">P95/case</span>
                      <span className="text-sm font-semibold text-apple-gray-700 dark:text-white">{formatLatency(cl.percentiles.p95 * 1000)}</span>
                    </div>
                    <div className="pl-4" title="Antal evaluerede test cases">
                      <span className="text-[10px] text-apple-gray-400 block">Cases</span>
                      <span className="text-sm font-semibold text-apple-gray-700 dark:text-white">{cl.total_cases}</span>
                    </div>
                  </div>

                  {/* Mode/difficulty latency side-by-side with bars */}
                  {(cl.latency_by_synthesis_mode.length > 0 || cl.latency_by_difficulty.length > 0) && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-5">
                      {cl.latency_by_synthesis_mode.length > 0 && (() => {
                        const maxMs = Math.max(...cl.latency_by_synthesis_mode.map(m => m.p50_seconds));
                        return (
                          <div>
                            <p className="text-[10px] font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider mb-2">Latency pr. case · tilstand</p>
                            <div className="space-y-1.5">
                              {cl.latency_by_synthesis_mode.map((m) => {
                                const pct = maxMs > 0 ? (m.p50_seconds / maxMs) * 100 : 0;
                                const color = MODE_COLORS[m.mode]?.hex || '#6b7280';
                                return (
                                  <div key={m.mode} title={MODE_DESCRIPTIONS[m.mode] || ''}>
                                    <div className="flex items-center justify-between mb-0.5">
                                      <span className="text-xs text-apple-gray-700 dark:text-white capitalize">{m.mode}</span>
                                      <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">{formatLatency(m.p50_seconds * 1000)} <span className="text-[10px]">({m.case_count})</span></span>
                                    </div>
                                    <div className="h-2 bg-apple-gray-100 dark:bg-apple-gray-600 rounded-full overflow-hidden">
                                      <div className="h-full rounded-full transition-all" style={{ width: `${Math.max(pct, 3)}%`, backgroundColor: color }} />
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })()}
                      {cl.latency_by_difficulty.length > 0 && (() => {
                        const maxMs = Math.max(...cl.latency_by_difficulty.map(d => d.p50_seconds));
                        return (
                          <div>
                            <p className="text-[10px] font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider mb-2">Latency pr. sværhedsgrad</p>
                            <div className="space-y-1.5">
                              {cl.latency_by_difficulty.map((d) => {
                                const pct = maxMs > 0 ? (d.p50_seconds / maxMs) * 100 : 0;
                                const color = DIFFICULTY_COLORS[d.difficulty]?.hex || '#6b7280';
                                return (
                                  <div key={d.difficulty} title={DIFFICULTY_DESCRIPTIONS[d.difficulty] || ''}>
                                    <div className="flex items-center justify-between mb-0.5">
                                      <span className="text-xs text-apple-gray-700 dark:text-white capitalize">{d.difficulty}</span>
                                      <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">{formatLatency(d.p50_seconds * 1000)} <span className="text-[10px]">({d.case_count})</span></span>
                                    </div>
                                    <div className="h-2 bg-apple-gray-100 dark:bg-apple-gray-600 rounded-full overflow-hidden">
                                      <div className="h-full rounded-full transition-all" style={{ width: `${Math.max(pct, 3)}%`, backgroundColor: color }} />
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        );
                      })()}
                    </div>
                  )}

                  {/* CL Latency trend — same compact format as SL, left-aligned */}
                  {cl.trend.length > 1 && (() => {
                    const values = cl.trend.map(p => p.median_ms);
                    const timestamps = cl.trend.map(p => p.timestamp);
                    const caseCounts = cl.trend.map(p => p.case_count);
                    const minV = Math.min(...values);
                    const maxV = Math.max(...values);
                    const midV = (minV + maxV) / 2;
                    const range = maxV - minV || 1;
                    const latest = values[values.length - 1];
                    const prev = values[values.length - 2];
                    const delta = latest - prev;
                    const deltaColor = delta > 0 ? 'text-red-500' : delta < 0 ? 'text-green-500' : 'text-apple-gray-400';
                    const svgW = 240; const svgH = 200;
                    const chartL = 42; const chartR = 235; const chartT = 8; const chartB = 190;
                    const chartW = chartR - chartL; const chartH = chartB - chartT;
                    const step = chartW / (values.length - 1);
                    const toX = (i: number) => chartL + i * step;
                    const toY = (v: number) => chartB - ((v - minV) / range) * chartH;
                    const pts = values.map((v, i) => `${toX(i)},${toY(v)}`).join(' ');
                    return (
                      <div className="max-w-[20%]">
                        <div className="flex items-center justify-between mb-1">
                          <p className="text-[10px] font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider">Latency pr. testcase</p>
                          <span className={`text-[10px] font-medium ${deltaColor} ml-1 shrink-0`}>
                            {delta > 0 ? '↑' : delta < 0 ? '↓' : '→'} {formatLatency(Math.abs(delta))}
                          </span>
                        </div>
                        <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full" style={{ height: 200 }} role="img" aria-label="Cross-law latency trend">
                          <line x1={chartL} y1={chartT} x2={chartR} y2={chartT} stroke="#e5e7eb" strokeWidth="0.5" strokeDasharray="4 3" />
                          <line x1={chartL} y1={(chartT + chartB) / 2} x2={chartR} y2={(chartT + chartB) / 2} stroke="#e5e7eb" strokeWidth="0.5" strokeDasharray="4 3" />
                          <line x1={chartL} y1={chartB} x2={chartR} y2={chartB} stroke="#e5e7eb" strokeWidth="0.5" strokeDasharray="4 3" />
                          <text x={chartL - 4} y={chartT + 4} textAnchor="end" fontSize="9" fill="#9ca3af">{formatLatency(maxV)}</text>
                          <text x={chartL - 4} y={(chartT + chartB) / 2 + 3} textAnchor="end" fontSize="9" fill="#9ca3af">{formatLatency(midV)}</text>
                          <text x={chartL - 4} y={chartB + 4} textAnchor="end" fontSize="9" fill="#9ca3af">{formatLatency(minV)}</text>
                          <polyline points={pts} fill="none" stroke="#8b5cf6" strokeWidth="2" strokeLinejoin="round" />
                          {values.map((v, i) => (
                            <circle key={i} cx={toX(i)} cy={toY(v)} r="3.5" fill={i === values.length - 1 ? '#8b5cf6' : '#c4b5fd'} stroke="white" strokeWidth="1" style={{ cursor: 'pointer' }}>
                              <title>{`${formatTimestamp(timestamps[i])}\n${formatLatency(v)} · ${caseCounts[i]} cases`}</title>
                            </circle>
                          ))}
                        </svg>
                        <div className="flex justify-between text-[9px] text-apple-gray-400 mt-0.5">
                          <span>{formatTimestamp(timestamps[0])}</span>
                          <span>{formatTimestamp(timestamps[timestamps.length - 1])}</span>
                        </div>
                      </div>
                    );
                  })()}
                </div>
              );
            })()}

          </div>
        </CollapsibleSection>
      )}

      {/* Ingestion Health Panel */}
      {ingestion && (
        <CollapsibleSection title="Ingestion Health">
          <div className="p-4">
            {/* Overall coverage */}
            <div className="flex items-center gap-2 mb-4">
              <span className="text-sm text-apple-gray-600 dark:text-apple-gray-300">Samlet HTML-tjek:</span>
              <span className="text-sm font-semibold" style={{ color: getPassRateColorHex(ingestion.overall_coverage) }}>
                {ingestion.overall_coverage}%
              </span>
            </div>

            {/* Corpus details table */}
            <div className="mt-2">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-apple-gray-500 dark:text-apple-gray-400">
                    <th className="text-left font-medium py-1">Corpus</th>
                    <th className="text-right font-medium py-1">Chunks</th>
                    <th className="text-right font-medium py-1">Ubehandlede</th>
                    <th className="text-right font-medium py-1">HTML-tjek</th>
                  </tr>
                </thead>
                <tbody>
                  {ingestion.corpora.map((c) => (
                    <tr key={c.corpus_id} className="border-t border-apple-gray-100 dark:border-apple-gray-500">
                      <td className="py-1.5 text-apple-gray-700 dark:text-white">{c.display_name}</td>
                      <td className="py-1.5 text-right text-apple-gray-600 dark:text-apple-gray-300">
                        {c.coverage === 0 ? '—' : c.chunks}
                      </td>
                      <td className="py-1.5 text-right text-apple-gray-600 dark:text-apple-gray-300">
                        {c.coverage === 0 ? '—' : c.unhandled}
                      </td>
                      <td className="py-1.5 text-right font-medium">
                        {c.coverage === 0 ? (
                          <span className="text-apple-gray-400">N/A</span>
                        ) : (
                          <span style={{ color: getPassRateColorHex(c.coverage) }}>{c.coverage.toFixed(1)}%</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </CollapsibleSection>
      )}

    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Collapsible section sub-component
// ─────────────────────────────────────────────────────────────────────────────

function CollapsibleSection({ title, children, defaultOpen = false }: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="rounded-2xl bg-white dark:bg-apple-gray-700 shadow-sm overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-4 py-3 cursor-pointer"
      >
        <h3 className="text-sm font-semibold text-apple-gray-400 dark:text-apple-gray-300 uppercase tracking-wider">
          {title}
        </h3>
        <svg
          className={`w-4 h-4 text-apple-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && (
        <div className="px-4 pb-4">
          {children}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Status badge sub-component (Samlet / Single-Law / Cross-Law)
// ─────────────────────────────────────────────────────────────────────────────

function StatusBadge({
  label,
  passRate,
  groupPassRate,
  total,
  healthStatus,
  trend,
  subtitle,
  size = 'sm',
}: {
  label: string;
  passRate: number;
  groupPassRate?: number | null;
  total: number;
  healthStatus?: string;
  trend?: TrendInfo;
  subtitle?: string;
  size?: 'sm' | 'lg';
}) {
  // Derive color from pass rate using health thresholds
  const color = healthStatus
    ? HEALTH_COLORS[healthStatus] || HEALTH_COLORS.red
    : passRate >= 95
      ? HEALTH_COLORS.green
      : passRate >= 80
        ? HEALTH_COLORS.yellow
        : passRate >= 60
          ? HEALTH_COLORS.orange
          : HEALTH_COLORS.red;

  const isLarge = size === 'lg';
  const circleSize = isLarge ? 'w-20 h-20' : 'w-16 h-16';
  const ringSize = isLarge ? 'ring-4' : 'ring-2';
  const textSize = isLarge ? 'text-base font-bold' : 'text-sm font-semibold';

  const trendLabel = trend ? (TREND_LABELS[trend.direction] || trend.direction) : null;

  const fixedWidth = isLarge ? 'w-24' : 'w-20';

  return (
    <div className={`${fixedWidth} flex flex-col items-center gap-1`}>
      <div
        className={`${circleSize} rounded-full flex items-center justify-center ${ringSize} ${color.bg} ${color.text} ${color.ring}`}
        role="status"
        aria-label={`${label}: ${passRate} procent bestået`}
      >
        <span className={textSize}>{passRate}%</span>
      </div>
      <span className="text-xs font-medium text-apple-gray-700 dark:text-white text-center">{label}</span>
      {/* Trend indicator (shown on Samlet badge) */}
      {trend && trend.direction !== 'insufficient_data' && (
        <div className="flex items-center gap-1">
          {trend.direction === 'improving' && <span className="text-green-600 dark:text-green-400 text-xs">↑</span>}
          {trend.direction === 'declining' && <span className="text-red-600 dark:text-red-400 text-xs">↓</span>}
          {trend.direction === 'stable' && <span className="text-apple-gray-400 text-xs">→</span>}
          <span className="text-[10px] text-apple-gray-500 dark:text-apple-gray-400">{trendLabel}</span>
          {trend.delta_pp != null && trend.delta_pp !== 0 && (
            <span className="text-[10px] text-apple-gray-400">
              {trend.delta_pp > 0 ? '+' : ''}{trend.delta_pp.toFixed(1)}pp
            </span>
          )}
        </div>
      )}
      {/* Group pass rate (per-law or per-suite average) */}
      {groupPassRate != null && Math.abs(groupPassRate - passRate) > 0.5 && (
        <span className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500 text-center">
          Ø {groupPassRate}% per {subtitle?.includes('love') ? 'lov' : 'suite'}
        </span>
      )}
      {/* Subtitle (e.g. "12 love", "5 suiter") */}
      {subtitle && !trend && (
        <span className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500 text-center">{subtitle} · {total} cases</span>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Performance badge (latency / rate in colored circle)
// ─────────────────────────────────────────────────────────────────────────────

function PerformanceBadge({
  label,
  value,
  subtitle,
  thresholds,
  size = 'sm',
}: {
  label: string;
  value: string;
  subtitle?: string;
  thresholds: { bg: string; text: string; ring: string };
  size?: 'sm' | 'lg';
}) {
  const isLarge = size === 'lg';
  const circleSize = isLarge ? 'w-20 h-20' : 'w-16 h-16';
  const ringSize = isLarge ? 'ring-4' : 'ring-2';
  const textSize = isLarge ? 'text-sm font-bold' : 'text-xs font-semibold';
  const fixedWidth = isLarge ? 'w-24' : 'w-20';

  return (
    <div className={`${fixedWidth} flex flex-col items-center gap-1`}>
      <div
        className={`${circleSize} rounded-full flex items-center justify-center ${ringSize} ${thresholds.bg} ${thresholds.text} ${thresholds.ring}`}
        role="status"
        aria-label={`${label}: ${value}`}
      >
        <span className={textSize}>{value}</span>
      </div>
      <span className="text-xs font-medium text-apple-gray-700 dark:text-white text-center">{label}</span>
      {subtitle && (
        <span className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500 text-center">{subtitle}</span>
      )}
    </div>
  );
}

/** Color thresholds for latency badges (seconds) */
function getLatencyColor(seconds: number, tier: 'median' | 'p95'): { bg: string; text: string; ring: string } {
  const thresholds = tier === 'median'
    ? [2, 5, 10]   // median: green <2s, yellow <5s, orange <10s, red ≥10s
    : [5, 15, 30];  // p95:    green <5s, yellow <15s, orange <30s, red ≥30s
  if (seconds < thresholds[0]) return HEALTH_COLORS.green;
  if (seconds < thresholds[1]) return HEALTH_COLORS.yellow;
  if (seconds < thresholds[2]) return HEALTH_COLORS.orange;
  return HEALTH_COLORS.red;
}

/** Color thresholds for rate badges (percentage) */
function getRateColor(rate: number, tier: 'escalation' | 'retry'): { bg: string; text: string; ring: string } {
  const thresholds = tier === 'escalation'
    ? [5, 15, 25]   // green <5%, yellow <15%, orange <25%, red ≥25%
    : [5, 10, 20];  // green <5%, yellow <10%, orange <20%, red ≥20%
  if (rate < thresholds[0]) return HEALTH_COLORS.green;
  if (rate < thresholds[1]) return HEALTH_COLORS.yellow;
  if (rate < thresholds[2]) return HEALTH_COLORS.orange;
  return HEALTH_COLORS.red;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mini donut chart (SVG)
// ─────────────────────────────────────────────────────────────────────────────

function MiniDonut({ entries, configMap, size = 48 }: {
  entries: [string, number][];
  configMap: Record<string, { label: string; color: string }>;
  size?: number;
}) {
  const total = entries.reduce((s, [, v]) => s + v, 0);
  if (total === 0) return null;

  const r = size / 2;
  const strokeWidth = size * 0.2;
  const radius = r - strokeWidth / 2;
  const circumference = 2 * Math.PI * radius;

  let offset = 0;
  const segments = entries.map(([key, value]) => {
    const pct = value / total;
    const dash = pct * circumference;
    const seg = { key, dash, gap: circumference - dash, offset, color: configMap[key]?.color || '#d1d5db' };
    offset += dash;
    return seg;
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      {/* Background circle */}
      <circle cx={r} cy={r} r={radius} fill="none" stroke="#e5e7eb" strokeWidth={strokeWidth} className="dark:stroke-gray-600" />
      {/* Segments */}
      {segments.map((seg) => (
        <circle
          key={seg.key}
          cx={r}
          cy={r}
          r={radius}
          fill="none"
          stroke={seg.color}
          strokeWidth={strokeWidth}
          strokeDasharray={`${seg.dash} ${seg.gap}`}
          strokeDashoffset={-seg.offset}
          transform={`rotate(-90 ${r} ${r})`}
        />
      ))}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Distribution group + donut helpers (overview row)
// ─────────────────────────────────────────────────────────────────────────────

function DistributionGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5 shrink-0">
      <p className="text-[10px] font-semibold uppercase tracking-wider text-apple-gray-400 dark:text-apple-gray-500 text-center">{title}</p>
      <div className="flex items-start gap-4">
        {children}
      </div>
    </div>
  );
}

/** Compute manual % from a distribution dict. */
function getManualPct(dist: Record<string, number> | undefined): number {
  if (!dist) return 0;
  const entries = Object.entries(dist);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  if (total === 0) return 0;
  return Math.round((dist['manual'] || 0) / total * 100);
}

/** Health color for manual % thresholds: ≥30% green, ≥15% yellow, ≥5% orange, <5% red. */
const MANUAL_HEALTH_COLOR: Record<string, string> = {
  green:  'text-green-600 dark:text-green-400',
  yellow: 'text-yellow-600 dark:text-yellow-400',
  orange: 'text-orange-600 dark:text-orange-400',
  red:    'text-red-600 dark:text-red-400',
};

function getManualHealthStatus(pct: number): string {
  if (pct >= 30) return 'green';
  if (pct >= 15) return 'yellow';
  if (pct >= 5) return 'orange';
  return 'red';
}

interface WarningResult { text: string; color: string }

function DistributionDonut({ label, entries, configMap, manualPct, warningFn }: {
  label: string;
  entries: [string, number][];
  configMap: Record<string, { label: string; color: string }>;
  /** If provided, shows a health-colored manual % warning. */
  manualPct?: number;
  warningFn?: (entries: [string, number][]) => WarningResult | null;
}) {
  if (entries.length === 0) return null;
  const warning = warningFn?.(entries) ?? null;

  // Manual % warning with health colors
  const manualWarning = manualPct != null ? (() => {
    const status = getManualHealthStatus(manualPct);
    const color = MANUAL_HEALTH_COLOR[status];
    if (manualPct < 30) return { text: `⚠ ${manualPct}% manuelle`, color };
    return { text: `${manualPct}% manuelle`, color };
  })() : null;

  return (
    <div className="w-20 flex flex-col items-center gap-1">
      <MiniDonut entries={entries} configMap={configMap} size={64} />
      <span className="text-xs font-medium text-apple-gray-700 dark:text-white text-center">{label}</span>
      <div className="flex flex-col items-center gap-0">
        {entries.map(([key, count]) => (
          <div key={key} className="flex items-center gap-1 text-[10px]">
            <span className="inline-block w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: configMap[key]?.color || '#d1d5db' }} />
            <span className="text-apple-gray-500 dark:text-apple-gray-400 whitespace-nowrap">{configMap[key]?.label || key}</span>
            <span className="text-apple-gray-700 dark:text-white font-medium">{count}</span>
          </div>
        ))}
      </div>
      {manualWarning && (
        <p className={`text-[10px] ${manualWarning.color} text-center`}>{manualWarning.text}</p>
      )}
      {warning && (
        <p className={`text-[10px] ${warning.color} text-center`}>{warning.text}</p>
      )}
    </div>
  );
}

function HealthBubble({ label, value, status, subtitle }: {
  label: string;
  value: string;
  status: string;
  subtitle?: string;
}) {
  const color = HEALTH_COLORS[status] || HEALTH_COLORS.red;
  return (
    <div className="w-20 flex flex-col items-center gap-1">
      <div className={`w-16 h-16 rounded-full flex items-center justify-center ring-2 ${color.bg} ${color.text} ${color.ring}`}>
        <span className="text-sm font-semibold leading-none">{value}</span>
      </div>
      <span className="text-xs font-medium text-apple-gray-700 dark:text-white text-center">{label}</span>
      {subtitle && (
        <span className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500 text-center">{subtitle}</span>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Case detail panel (expandable row content)
// ─────────────────────────────────────────────────────────────────────────────

function CaseDetailPanel({ caseResult }: { caseResult: CaseResult }) {
  const r = caseResult;
  const scores = r.scores || {};
  const messages = r.score_messages || {};
  const scoreEntries = Object.entries(scores);

  return (
    <div className="flex flex-col gap-2 text-xs">
      {/* Prompt (cross-law) */}
      {r.prompt && (
        <div>
          <span className="font-medium text-apple-gray-500 dark:text-apple-gray-400">Prompt: </span>
          <span className="text-apple-gray-700 dark:text-white">{r.prompt}</span>
        </div>
      )}

      {/* Type badges row */}
      <div className="flex items-center gap-2 flex-wrap">
        {r.synthesis_mode && (
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${MODE_PILL_CLASSES[r.synthesis_mode] || 'bg-apple-gray-100 dark:bg-apple-gray-600 text-apple-gray-600 dark:text-apple-gray-300'}`}
            title={MODE_DESCRIPTIONS[r.synthesis_mode] || ''}
          >
            {r.synthesis_mode.toUpperCase()}
          </span>
        )}
        {r.difficulty && (
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-medium capitalize ${DIFFICULTY_PILL_CLASSES[r.difficulty] || 'bg-apple-gray-100 dark:bg-apple-gray-600 text-apple-gray-600 dark:text-apple-gray-300'}`}
            title={DIFFICULTY_DESCRIPTIONS[r.difficulty] || ''}
          >
            {r.difficulty}
          </span>
        )}
        {r.profile && (
          <span className="px-1.5 py-0.5 rounded-md text-[10px] font-medium bg-apple-gray-200 dark:bg-apple-gray-500 text-apple-gray-600 dark:text-white">
            {r.profile}
          </span>
        )}
      </div>

      {/* Target corpora (cross-law) */}
      {r.target_corpora && r.target_corpora.length > 0 && (
        <div>
          <span className="font-medium text-apple-gray-500 dark:text-apple-gray-400">Target: </span>
          <span className="text-apple-gray-600 dark:text-apple-gray-300">{r.target_corpora.join(', ')}</span>
        </div>
      )}

      {/* Anchors (single-law) */}
      {r.expected_anchors && r.expected_anchors.length > 0 && (
        <div>
          <span className="font-medium text-apple-gray-500 dark:text-apple-gray-400">Forventede anchors: </span>
          <span className="text-apple-gray-600 dark:text-apple-gray-300">{r.expected_anchors.join(', ')}</span>
        </div>
      )}
      {r.anchors && r.anchors.length > 0 && (
        <div>
          <span className="font-medium text-apple-gray-500 dark:text-apple-gray-400">Fundne anchors: </span>
          <span className="text-apple-gray-600 dark:text-apple-gray-300">{r.anchors.join(', ')}</span>
        </div>
      )}

      {/* Per-scorer results */}
      {scoreEntries.length > 0 && (
        <div className="mt-1">
          <p className="font-medium text-apple-gray-500 dark:text-apple-gray-400 mb-1">Scores</p>
          <div className="space-y-0">
            {scoreEntries.map(([scorer, passed]) => (
              <div
                key={scorer}
                className="flex items-center justify-between py-0.5 border-t border-apple-gray-100 dark:border-apple-gray-500"
                title={EVAL_SCORER_DESCRIPTIONS[scorer] || ''}
              >
                <span className="text-apple-gray-600 dark:text-apple-gray-300">
                  {EVAL_SCORER_LABELS[scorer] || CROSS_LAW_SCORER_LABELS[scorer] || scorer}
                </span>
                <div className="flex items-center gap-2">
                  {messages[scorer] && (
                    <span className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500 max-w-[200px] truncate" title={messages[scorer]}>
                      {messages[scorer]}
                    </span>
                  )}
                  <span className={passed ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}>
                    {passed ? '✓' : '✗'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const CASE_PAGE_SIZE = 20;

function CaseResultsTable({ cases, filter, onFilterChange, statusFilter, onStatusFilterChange, page, onPageChange }: {
  cases: CaseResult[];
  filter: string;
  onFilterChange: (v: string) => void;
  statusFilter: 'all' | 'failed' | 'passed';
  onStatusFilterChange: (v: 'all' | 'failed' | 'passed') => void;
  page: number;
  onPageChange: (p: number) => void;
}) {
  const [expandedCase, setExpandedCase] = useState<string | null>(null);
  const q = filter.toLowerCase();
  const filtered = cases.filter((c) => {
    if (statusFilter === 'passed' && !c.passed) return false;
    if (statusFilter === 'failed' && c.passed) return false;
    if (q && !c.case_id.toLowerCase().includes(q)) return false;
    return true;
  });
  const pageCount = Math.max(1, Math.ceil(filtered.length / CASE_PAGE_SIZE));
  const paged = filtered.slice(page * CASE_PAGE_SIZE, (page + 1) * CASE_PAGE_SIZE);

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <SegmentedControl
          options={[
            { value: 'all' as const, label: 'Alle' },
            { value: 'failed' as const, label: 'Fejlet', selectedColor: 'red' },
            { value: 'passed' as const, label: 'Bestået', selectedColor: 'green' },
          ]}
          value={statusFilter}
          onChange={onStatusFilterChange}
          size="sm"
        />
        <input
          type="text"
          placeholder="Søg case..."
          value={filter}
          onChange={(e) => onFilterChange(e.target.value)}
          className="flex-1 text-xs px-2 py-1 rounded-lg border border-apple-gray-100 dark:border-apple-gray-500 bg-apple-gray-50 dark:bg-apple-gray-600 text-apple-gray-700 dark:text-white placeholder-apple-gray-400 focus:outline-none focus:ring-1 focus:ring-apple-blue"
        />
      </div>
      <table className="w-full text-xs table-fixed">
        <thead>
          <tr className="text-apple-gray-500 dark:text-apple-gray-400">
            <th className="text-left font-medium py-1 w-[30%]">Case</th>
            <th className="text-left font-medium py-1">Scores</th>
            <th className="text-right font-medium py-1 w-[50px]">Status</th>
            <th className="text-right font-medium py-1 w-[60px]">Latency</th>
          </tr>
        </thead>
        <tbody>
          {paged.map((r) => {
            const isExpanded = expandedCase === r.case_id;
            const hasExtra = !!(r.prompt || r.anchors?.length || r.expected_anchors?.length || r.target_corpora?.length || (r.score_messages && Object.keys(r.score_messages).length > 0));
            const scoreEntries = r.scores ? Object.entries(r.scores) : [];
            return (
              <React.Fragment key={r.case_id}>
                <tr
                  className="border-t border-apple-gray-100 dark:border-apple-gray-500 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-600/50 cursor-pointer"
                  tabIndex={hasExtra ? 0 : undefined}
                  role={hasExtra ? 'button' : undefined}
                  onClick={() => hasExtra && setExpandedCase(isExpanded ? null : r.case_id)}
                  onKeyDown={hasExtra ? (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setExpandedCase(isExpanded ? null : r.case_id); } } : undefined}
                >
                  <td className="py-1.5 text-apple-gray-700 dark:text-white">
                    <div className="flex items-center gap-1">
                      {hasExtra && (
                        <span className={`inline-block w-3 shrink-0 text-apple-gray-400 transition-transform ${isExpanded ? 'rotate-90' : ''}`}>›</span>
                      )}
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1">
                          <span className="truncate">{r.case_id}</span>
                          {r.synthesis_mode && (
                            <span
                              className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium whitespace-nowrap ${MODE_PILL_CLASSES[r.synthesis_mode] || 'bg-apple-gray-100 dark:bg-apple-gray-600 text-apple-gray-600 dark:text-apple-gray-300'}`}
                              title={MODE_DESCRIPTIONS[r.synthesis_mode] || ''}
                            >
                              {r.synthesis_mode.toUpperCase()}
                            </span>
                          )}
                          {r.difficulty && (
                            <span
                              className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium whitespace-nowrap capitalize ${DIFFICULTY_PILL_CLASSES[r.difficulty] || 'bg-apple-gray-100 dark:bg-apple-gray-600 text-apple-gray-600 dark:text-apple-gray-300'}`}
                              title={DIFFICULTY_DESCRIPTIONS[r.difficulty] || ''}
                            >
                              {r.difficulty}
                            </span>
                          )}
                        </div>
                        {r.prompt && (
                          <p className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500 truncate mt-0.5" title={r.prompt}>
                            {r.prompt}
                          </p>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="py-1.5">
                    <div className="flex items-center gap-0.5 flex-wrap">
                      {scoreEntries.map(([scorer, passed]) => (
                        <span
                          key={scorer}
                          className={`inline-block px-1 py-0 rounded text-[9px] font-medium ${passed ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'}`}
                          title={`${EVAL_SCORER_LABELS[scorer] || CROSS_LAW_SCORER_LABELS[scorer] || scorer}: ${passed ? 'Pass' : 'Fail'}${r.score_messages?.[scorer] ? ` — ${r.score_messages[scorer]}` : ''}`}
                        >
                          {EVAL_SCORER_LABELS[scorer] || CROSS_LAW_SCORER_LABELS[scorer] || scorer}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="py-1.5 text-right">
                    <span className={r.passed ? 'text-green-600' : 'text-red-600'}>
                      {r.passed ? 'Pass' : 'Fail'}
                    </span>
                  </td>
                  <td className="py-1.5 text-right text-apple-gray-600 dark:text-apple-gray-300">
                    {formatLatency(r.duration_ms)}
                  </td>
                </tr>
                {isExpanded && (
                  <tr>
                    <td colSpan={4} className="px-3 py-2 bg-apple-gray-50 dark:bg-apple-gray-600/30">
                      <CaseDetailPanel caseResult={r} />
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
      {pageCount > 1 && (
        <div className="flex items-center justify-center gap-1 mt-1 text-xs text-apple-gray-500 dark:text-apple-gray-400">
          <button
            onClick={() => onPageChange(Math.max(0, page - 1))}
            disabled={page === 0}
            className="px-1.5 py-0.5 rounded hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-30"
            aria-label="Forrige"
          >
            ‹
          </button>
          <span>{page + 1} / {pageCount}</span>
          <button
            onClick={() => onPageChange(Math.min(pageCount - 1, page + 1))}
            disabled={page >= pageCount - 1}
            className="px-1.5 py-0.5 rounded hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-30"
            aria-label="Næste"
          >
            ›
          </button>
        </div>
      )}
    </div>
  );
}
