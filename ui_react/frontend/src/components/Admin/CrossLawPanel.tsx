/**
 * Cross-Law evaluation panel with drill-down navigation.
 *
 * Matches Single-Law EvalDashboard pattern:
 * Level 1: Suite Matrix (overview of all suites)
 * Level 2: Suite Runs (run history for selected suite)
 * Level 3: Run Details (test case results for selected run)
 *
 * Requirements: R-UI-01 through R-UI-06
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { SearchInput } from '../Common/SearchInput';
import { SegmentedControl } from '../Common/SegmentedControl';
import { Tooltip } from '../Common/Tooltip';
import { AnchorListInput } from '../Common/AnchorListInput';
import { LawSelectorPanel } from '../Sidebar/LawSelectorPanel';
import { runSingleCrossLawCase, type CrossLawValidationResult } from '../../services/api';
import type { CorpusInfo } from '../../types';
import {
  EVAL_SCORER_LABELS,
  EVAL_SCORER_DESCRIPTIONS,
  EVAL_TEST_TYPE_LABELS,
  RUN_MODE_LABELS,
  formatPassRate,
  formatDuration,
  formatTimestamp,
  type RunMode,
} from './evalUtils';
import { RunModePopover } from './RunModePopover';
import { ColumnTooltip } from './ColumnTooltip';

const API_BASE = '/api';

// Cross-law test types selectable in case editor (R10.1)
// 4 synthesis modes — each determines its own scorer set automatically.
// No manual test type selection needed.
const SYNTHESIS_MODES = [
  { value: 'comparison', label: 'Comparison', tooltip: 'Sammenlign krav og definitioner mellem to eller flere love' },
  { value: 'aggregation', label: 'Aggregation', tooltip: 'Saml information om et emne på tværs af alle valgte love' },
  { value: 'routing', label: 'Routing', tooltip: 'Identificer hvilke love der gælder for et givent scenarie' },
  { value: 'discovery', label: 'Discovery', tooltip: 'Emnebaseret identifikation af relevante love uden navngivne lovhenvisninger' },
] as const;

// Each mode's mode-specific scorer (shared scorers always apply: retrieval, faithfulness, relevancy, corpus_coverage)
// Discovery has no mode-specific scorer — corpus_coverage handles it via threshold
const MODE_SCORER: Record<string, string> = {
  comparison: 'comparison_completeness',
  routing: 'routing_precision',
  aggregation: 'synthesis_balance',
};

// Shared scorer columns (same order as single-law: Retrieval → Faithfulness → Relevancy → Coverage)
const SHARED_SCORE_TYPES = [
  'anchor_presence',
  'faithfulness',
  'answer_relevancy',
  'corpus_coverage',
] as const;

// Mode-specific scorers — one per synthesis mode (discovery uses corpus_coverage shared scorer)
const MODE_SPECIFIC_SCORERS = ['synthesis_balance', 'routing_precision', 'comparison_completeness'] as const;

// All score types (shared + mode-specific) for type definitions
const CROSS_LAW_ALL_SCORE_TYPES = [...SHARED_SCORE_TYPES, ...MODE_SPECIFIC_SCORERS] as const;

type CrossLawScoreType = typeof CROSS_LAW_ALL_SCORE_TYPES[number];

// Labels imported from ./evalUtils: EVAL_SCORER_LABELS, EVAL_SCORER_DESCRIPTIONS, EVAL_TEST_TYPE_LABELS
// Local aliases for backward compatibility within this file
const CROSS_LAW_TEST_LABELS = EVAL_SCORER_LABELS;
const CROSS_LAW_TEST_DESCRIPTIONS = EVAL_SCORER_DESCRIPTIONS;
const TEST_TYPE_TAG_LABELS = EVAL_TEST_TYPE_LABELS;

// Types
interface CrossLawSuiteStats {
  id: string;
  name: string;
  case_count: number;
  passed: number;
  failed: number;
  pass_rate: number;
  last_run: string | null;
  last_run_mode: string | null;
  scorer_pass_rates: Record<string, number>;
  default_synthesis_mode: string;
  mode_counts: Record<string, number>;
  difficulty_counts: Record<string, number>;
}

interface CrossLawOverview {
  suites: CrossLawSuiteStats[];
  total_cases: number;
  overall_pass_rate: number;
}

interface CrossLawRunSummary {
  run_id: string;
  suite_id: string;
  timestamp: string;
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  duration_seconds: number;
  run_mode: string;
  max_retries?: number;
}

interface CrossLawCaseResult {
  case_id: string;
  prompt: string;
  synthesis_mode: string;
  target_corpora: string[];
  passed: boolean;
  duration_ms: number;
  scores: Record<string, unknown>;
  error: string | null;
  difficulty: string | null;
}

interface CrossLawRunDetail {
  run_id: string;
  suite_id: string;
  timestamp: string;
  duration_seconds: number;
  total: number;
  passed: number;
  failed: number;
  pass_rate: number;
  run_mode: string;
  results: CrossLawCaseResult[];
}

// Test case in a suite (for editing)
interface CrossLawTestCase {
  id: string;
  prompt: string;
  synthesis_mode: string;
  target_corpora: string[];
  expected_articles?: string[];
  origin?: string;
  corpus_scope: string;
  expected_anchors: string[];
  expected_corpora: string[];
  min_corpora_cited: number;
  profile: string;
  disabled: boolean;
  // Extended fields for full eval support
  test_types: string[];
  expected_behavior: string;
  must_include_any_of: string[];
  must_include_any_of_2: string[];
  must_include_all_of: string[];
  must_not_include_any_of: string[];
  contract_check: boolean;
  min_citations: number | null;
  max_citations: number | null;
  notes: string;
  // Quality metadata
  difficulty: string | null;
}

interface CrossLawSuiteDetail {
  id: string;
  name: string;
  description: string;
  target_corpora: string[];
  default_synthesis_mode: string;
  cases: CrossLawTestCase[];
}

// RunMode, RUN_MODE_LABELS imported from ./evalUtils
// RunModePopover imported from ./RunModePopover

// formatPassRate imported from ./evalUtils (returns JSX with 4-tier colour)

// Apple HIG slider: compute fill gradient for WebKit (Firefox uses ::-moz-range-progress)
function sliderFillStyle(value: number, min: number, max: number): React.CSSProperties {
  const pct = ((value - min) / (max - min)) * 100;
  return {
    background: `linear-gradient(to right, var(--slider-fill) ${pct}%, var(--slider-track) ${pct}%)`,
  };
}

// formatTimestamp imported from ./evalUtils (clean Danish format)

export function CrossLawPanel({ onDrillChange }: { onDrillChange?: (drilled: boolean) => void }) {
  // Data state
  const [overview, setOverview] = useState<CrossLawOverview | null>(null);
  const [runs, setRuns] = useState<CrossLawRunSummary[]>([]);
  const [runDetail, setRunDetail] = useState<CrossLawRunDetail | null>(null);

  // Navigation state
  const [selectedSuiteId, setSelectedSuiteId] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  // UI state
  const [isMatrixCollapsed, setIsMatrixCollapsed] = useState(false);
  const [isRunsCollapsed, setIsRunsCollapsed] = useState(false);
  const [expandedCase, setExpandedCase] = useState<string | null>(null);
  const [caseFilter, setCaseFilter] = useState<'all' | 'passed' | 'failed'>('all');
  const [searchQuery, setSearchQuery] = useState('');

  // Sort state for matrix columns
  const [sortField, setSortField] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDirection(d => d === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Run trigger state
  const [triggeringSuite, setTriggeringSuite] = useState<string | null>(null);
  const [triggerMode, setTriggerMode] = useState<RunMode>('retrieval_only');
  const [triggerRetries, setTriggerRetries] = useState<number>(0);
  const [openRunMenu, setOpenRunMenu] = useState<string | null>(null);
  const [runMenuPosition, setRunMenuPosition] = useState({ x: 0, y: 0 });

  // Edit state - for suite metadata modal
  const [editingSuiteId, setEditingSuiteId] = useState<string | null>(null);

  // Test cases view state
  const [viewingCasesSuiteId, setViewingCasesSuiteId] = useState<string | null>(null);
  const [suiteDetail, setSuiteDetail] = useState<CrossLawSuiteDetail | null>(null);
  const [isCasesCollapsed, setIsCasesCollapsed] = useState(false);
  const [expandedTestCase, setExpandedTestCase] = useState<string | null>(null);
  const [caseDeleteConfirm, setCaseDeleteConfirm] = useState<string | null>(null);
  const [isCaseDeleting, setIsCaseDeleting] = useState(false);
  const [casesSearchQuery, setCasesSearchQuery] = useState('');
  const [synthesisFilter, setSynthesisFilter] = useState<'all' | 'comparison' | 'aggregation' | 'routing' | 'discovery'>('all');

  // Loading/error state
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Running eval progress (mirrors AdminPage's RunningEvalState pattern)
  const [runProgress, setRunProgress] = useState<{
    suiteId: string;
    suiteName: string;
    stage: 'running' | 'complete';
    total: number;
    passed: number;
    failed: number;
  } | null>(null);

  // Modal state
  const [showGenerateModal, setShowGenerateModal] = useState(false);
  const [generateForSuiteId, setGenerateForSuiteId] = useState<string | null>(null);

  // Available corpora from API
  const [availableCorpora, setAvailableCorpora] = useState<CorpusInfo[]>([]);


  // Case editor state
  const [editingCaseId, setEditingCaseId] = useState<string | null>(null);
  const [showCaseEditor, setShowCaseEditor] = useState(false);

  // Load overview
  const loadOverview = useCallback(async () => {
    try {
      setIsLoading(true);
      const resp = await fetch(`${API_BASE}/eval/cross-law/overview`);
      if (!resp.ok) throw new Error('Failed to load overview');
      const data: CrossLawOverview = await resp.json();
      setOverview(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadOverview();
  }, [loadOverview]);

  // Load available corpora from API
  useEffect(() => {
    const loadCorpora = async () => {
      try {
        const resp = await fetch(`${API_BASE}/corpora`);
        if (!resp.ok) return;
        const data = await resp.json();
        setAvailableCorpora(data.corpora || []);
      } catch {
        // Fallback: empty — modals will show IDs only
      }
    };
    loadCorpora();
  }, []);

  // Load runs when suite selected
  useEffect(() => {
    if (!selectedSuiteId) {
      setRuns([]);
      return;
    }

    const loadRuns = async () => {
      try {
        const resp = await fetch(`${API_BASE}/eval/cross-law/suites/${selectedSuiteId}/runs`);
        if (!resp.ok) throw new Error('Failed to load runs');
        const data = await resp.json();
        setRuns(data.runs || []);
      } catch (e) {
        console.error('Failed to load runs:', e);
        setRuns([]);
      }
    };

    loadRuns();
  }, [selectedSuiteId]);

  // Load run detail when run selected
  useEffect(() => {
    if (!selectedRunId) {
      setRunDetail(null);
      return;
    }

    const loadRunDetail = async () => {
      try {
        const resp = await fetch(`${API_BASE}/eval/cross-law/runs/${selectedRunId}`);
        if (!resp.ok) throw new Error('Failed to load run');
        const data: CrossLawRunDetail = await resp.json();
        setRunDetail(data);
      } catch (e) {
        console.error('Failed to load run detail:', e);
        setRunDetail(null);
      }
    };

    loadRunDetail();
  }, [selectedRunId]);

  // Signal drill-down state to parent for layout adjustments
  useEffect(() => {
    onDrillChange?.(isMatrixCollapsed);
  }, [isMatrixCollapsed, onDrillChange]);

  // Load suite detail (with cases) when viewing cases
  useEffect(() => {
    if (!viewingCasesSuiteId) {
      setSuiteDetail(null);
      return;
    }

    const loadSuiteDetail = async () => {
      try {
        const resp = await fetch(`${API_BASE}/eval/cross-law/suites/${viewingCasesSuiteId}`);
        if (!resp.ok) throw new Error('Failed to load suite');
        const data: CrossLawSuiteDetail = await resp.json();
        setSuiteDetail(data);
      } catch (e) {
        console.error('Failed to load suite detail:', e);
        setSuiteDetail(null);
      }
    };

    loadSuiteDetail();
  }, [viewingCasesSuiteId]);

  // Handle suite selection (drill down)
  const handleSelectSuite = (suiteId: string) => {
    setSelectedSuiteId(suiteId);
    setSelectedRunId(null);
    setIsMatrixCollapsed(true);
    setIsRunsCollapsed(false);
  };

  // Handle run selection (drill down)
  const handleSelectRun = (runId: string) => {
    setSelectedRunId(runId);
    setIsRunsCollapsed(true);
  };

  // Handle edit test cases (drill down to cases view)
  const handleEditCases = (suiteId: string) => {
    setViewingCasesSuiteId(suiteId);
    setIsMatrixCollapsed(true);
    setIsCasesCollapsed(false);
    // Close runs view if open
    setSelectedSuiteId(null);
    setSelectedRunId(null);
  };

  // Handle back navigation
  const handleBackToMatrix = () => {
    setSelectedSuiteId(null);
    setSelectedRunId(null);
    setViewingCasesSuiteId(null);
    setIsMatrixCollapsed(false);
    setIsRunsCollapsed(false);
    setIsCasesCollapsed(false);
  };

  const handleBackToRuns = () => {
    setSelectedRunId(null);
    setIsRunsCollapsed(false);
  };

  const handleBackFromCases = () => {
    setViewingCasesSuiteId(null);
    setIsMatrixCollapsed(false);
    setIsCasesCollapsed(false);
  };

  // Handle opening case editor for new case
  const handleCreateCase = () => {
    setEditingCaseId(null);
    setShowCaseEditor(true);
  };

  // Handle opening case editor for existing case
  const handleEditCase = (caseId: string) => {
    setEditingCaseId(caseId);
    setShowCaseEditor(true);
  };

  // Handle case saved
  const handleCaseSaved = async () => {
    setShowCaseEditor(false);
    setEditingCaseId(null);
    // Reload suite detail
    if (viewingCasesSuiteId) {
      try {
        const resp = await fetch(`${API_BASE}/eval/cross-law/suites/${viewingCasesSuiteId}`);
        if (resp.ok) {
          const data = await resp.json();
          setSuiteDetail(data);
        }
      } catch (e) {
        console.error('Failed to reload suite detail:', e);
      }
    }
    loadOverview();
  };

  // Handle trigger eval (can be called with explicit suite ID and mode)
  const handleTriggerEval = async (suiteId?: string, mode?: RunMode) => {
    const targetSuiteId = suiteId || selectedSuiteId;
    const targetMode = mode || triggerMode;
    if (!targetSuiteId) return;

    setTriggeringSuite(targetSuiteId);

    // Resolve suite name and case count for progress bar
    const suiteInfo = overview?.suites.find((s) => s.id === targetSuiteId);
    const suiteName = suiteInfo?.name || targetSuiteId;
    const totalCases = suiteInfo?.case_count ?? 0;

    // Show progress bar immediately (before SSE starts)
    setRunProgress({ suiteId: targetSuiteId, suiteName, stage: 'running', total: totalCases, passed: 0, failed: 0 });

    try {
      const resp = await fetch(`${API_BASE}/eval/cross-law/suites/${targetSuiteId}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_mode: targetMode, max_retries: triggerRetries }),
      });

      if (!resp.ok) throw new Error(`Failed to trigger eval: ${resp.status}`);

      // SSE stream handling with progress tracking
      const reader = resp.body?.getReader();
      if (reader) {
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          const chunk = decoder.decode(value, { stream: true });
          buffer += chunk;

          // Process complete SSE lines
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const payload = line.slice(6).trim();
            if (payload === '[DONE]') {
              break;
            }

            try {
              const event = JSON.parse(payload);
              if (event.type === 'start') {
                setRunProgress((prev) => prev ? { ...prev, total: event.total } : prev);
              } else if (event.type === 'case_result') {
                setRunProgress((prev) => prev ? {
                  ...prev,
                  passed: prev.passed + (event.passed ? 1 : 0),
                  failed: prev.failed + (event.passed ? 0 : 1),
                } : prev);
              } else if (event.type === 'complete') {
                setRunProgress((prev) => prev ? { ...prev, stage: 'complete', passed: event.passed, failed: event.failed, total: event.total } : prev);
              }
            } catch (parseErr) {
              console.warn('[cross-law-eval] Parse error:', parseErr, 'payload:', payload.slice(0, 100));
            }
          }
        }
      }

      // Mark complete if SSE didn't send a complete event
      setRunProgress((prev) => prev?.stage !== 'complete' ? (prev ? { ...prev, stage: 'complete' } : prev) : prev);

      // Reload overview after completion
      await loadOverview();

      // If a suite is selected, reload its runs too
      if (selectedSuiteId) {
        const runsResp = await fetch(`${API_BASE}/eval/cross-law/suites/${selectedSuiteId}/runs`);
        if (runsResp.ok) {
          const data = await runsResp.json();
          setRuns(data.runs || []);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to trigger eval');
      setRunProgress(null);
    } finally {
      setTriggeringSuite(null);
    }
  };

  // Get selected suite info
  const selectedSuite = overview?.suites.find((s) => s.id === selectedSuiteId);

  // Filter and sort suites based on search query
  const filteredSuites = useMemo(() => {
    if (!overview?.suites) return [];
    let result = overview.suites;

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter((suite) =>
        suite.name.toLowerCase().includes(query) ||
        suite.id.toLowerCase().includes(query)
      );
    }

    if (sortField) {
      result = [...result].sort((a, b) => {
        let aVal: number | string;
        let bVal: number | string;
        if (sortField === 'name') {
          aVal = a.name.toLowerCase();
          bVal = b.name.toLowerCase();
        } else if (sortField === 'pass_rate') {
          aVal = a.pass_rate ?? -1;
          bVal = b.pass_rate ?? -1;
        } else if (sortField === 'last_run') {
          aVal = a.last_run ?? '';
          bVal = b.last_run ?? '';
        } else if (sortField === 'mode_scorer') {
          // Mode-specific scorer — resolve per suite's synthesis mode
          const aKey = MODE_SCORER[a.default_synthesis_mode];
          const bKey = MODE_SCORER[b.default_synthesis_mode];
          aVal = aKey ? (a.scorer_pass_rates?.[aKey] ?? -1) : -1;
          bVal = bKey ? (b.scorer_pass_rates?.[bKey] ?? -1) : -1;
        } else {
          // Shared scorer column
          aVal = a.scorer_pass_rates?.[sortField] ?? -1;
          bVal = b.scorer_pass_rates?.[sortField] ?? -1;
        }
        if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
        if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
        return 0;
      });
    }

    return result;
  }, [overview?.suites, searchQuery, sortField, sortDirection]);

  // Filter results
  const filteredResults = runDetail?.results.filter((r) => {
    if (caseFilter === 'passed') return r.passed;
    if (caseFilter === 'failed') return !r.passed;
    return true;
  }) || [];

  // Filter test cases in suite detail
  const filteredTestCases = useMemo(() => {
    if (!suiteDetail?.cases) return [];
    let result = [...suiteDetail.cases];

    // Apply synthesis mode filter
    if (synthesisFilter !== 'all') {
      result = result.filter((c) => c.synthesis_mode === synthesisFilter);
    }

    // Apply search
    if (casesSearchQuery.trim()) {
      const query = casesSearchQuery.toLowerCase();
      result = result.filter(
        (c) =>
          c.id.toLowerCase().includes(query) ||
          c.prompt.toLowerCase().includes(query)
      );
    }

    return result;
  }, [suiteDetail?.cases, synthesisFilter, casesSearchQuery]);

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin w-6 h-6 border-2 border-apple-blue border-t-transparent rounded-full" />
      </div>
    );
  }

  // Empty state
  if (!overview || overview.suites.length === 0) {
    return (
      <div className="bg-white dark:bg-apple-gray-700 rounded-2xl border border-apple-gray-200 dark:border-apple-gray-600 p-8 text-center">
        <div className="w-16 h-16 mx-auto mb-4 bg-apple-gray-100 dark:bg-apple-gray-600 rounded-full flex items-center justify-center">
          <svg className="w-8 h-8 text-apple-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-apple-gray-700 dark:text-white mb-2">
          Ingen cross-law eval suites
        </h3>
        <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400 mb-6">
          Opret en ny suite eller importer fra YAML for at komme i gang.
        </p>
        <button
          onClick={() => { setGenerateForSuiteId(null); setShowGenerateModal(true); }}
          className="px-4 py-2 bg-apple-blue hover:bg-apple-blue-hover text-white text-sm font-medium rounded-lg transition-colors"
        >
          Auto-generer suite
        </button>
      </div>
    );
  }

  return (
    <div className={`flex flex-col gap-4 ${isMatrixCollapsed ? 'flex-1 min-h-0' : ''}`}>
      {/* Error display */}
      {error && (
        <div className="px-4 py-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400 flex justify-between">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-500 hover:text-red-700">×</button>
        </div>
      )}

      {/* Level 1: Suite Matrix */}
      <div className="bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500">
        {isMatrixCollapsed ? (
          // Collapsed header - same style as Single-Law
          <button
            onClick={handleBackToMatrix}
            className="w-full px-4 py-3 flex items-center justify-between hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500/50 transition-colors"
          >
            <div className="flex items-center gap-3">
              <svg className="w-4 h-4 text-apple-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <span className="text-sm font-medium text-apple-gray-700 dark:text-white">
                Matrix
              </span>
              <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                {overview.suites.length} {overview.suites.length === 1 ? 'suite' : 'suites'} · {overview.total_cases} cases
              </span>
            </div>
            <svg className="w-4 h-4 text-apple-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        ) : (
          // Expanded matrix
          <>
            {/* Search input */}
            <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-500">
              <div className="flex items-center gap-3">
                <SearchInput
                  value={searchQuery}
                  onChange={setSearchQuery}
                  placeholder="Søg efter suite..."
                  className="flex-1"
                />
                {/* Collapse button - show when suite selected */}
                {selectedSuiteId && (
                  <button
                    onClick={() => setIsMatrixCollapsed(true)}
                    className="p-2 text-apple-gray-400 hover:text-apple-gray-600 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded-lg transition-colors"
                    title="Skjul matrix"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                    </svg>
                  </button>
                )}
              </div>
            </div>

            <div className="overflow-x-auto overflow-y-auto">
              <table className="w-full min-w-[900px]">
                <thead className="sticky top-0 z-20 bg-apple-gray-50 dark:bg-apple-gray-700">
                  <tr className="border-b border-apple-gray-100 dark:border-apple-gray-500">
                    <th
                      className="text-left font-medium text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide px-4 py-3 min-w-[220px] cursor-pointer hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 transition-colors"
                      onClick={() => handleSort('name')}
                    >
                      <div className="flex items-center gap-1">
                        Suite
                        {sortField === 'name' && (
                          <span className="text-apple-blue">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                        )}
                      </div>
                    </th>
                    <th className="text-left font-medium text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide px-2 py-3 whitespace-nowrap cursor-help" title="Syntese-modus og sværhedsgrad">
                      Mode / Sværhed
                    </th>
                    <ColumnTooltip
                      label="Total"
                      description="Samlet bestået rate"
                      sortField={sortField}
                      field="pass_rate"
                      sortDirection={sortDirection}
                      onSort={handleSort}
                    />
                    {SHARED_SCORE_TYPES.map((type) => (
                      <ColumnTooltip
                        key={type}
                        label={CROSS_LAW_TEST_LABELS[type] || type}
                        description={CROSS_LAW_TEST_DESCRIPTIONS[type] || type}
                        sortField={sortField}
                        field={type}
                        sortDirection={sortDirection}
                        onSort={handleSort}
                      />
                    ))}
                    <ColumnTooltip
                      label="Mode"
                      description="Mode-specifik scorer (Comparison/Routing/Balance afhængigt af suite)"
                      sortField={sortField}
                      field="mode_scorer"
                      sortDirection={sortDirection}
                      onSort={handleSort}
                    />
                    <ColumnTooltip
                      label="Sidst kørt"
                      description="Tidspunkt for seneste kørsel"
                      sortField={sortField}
                      field="last_run"
                      sortDirection={sortDirection}
                      onSort={handleSort}
                    />
                    <th className="w-14" />
                  </tr>
                </thead>
              <tbody className="divide-y divide-apple-gray-100 dark:divide-apple-gray-600">
                {filteredSuites.map((suite) => (
                  <tr
                    key={suite.id}
                    className={`hover:bg-apple-gray-50 dark:hover:bg-apple-gray-600/30 transition-colors ${
                      selectedSuiteId === suite.id ? 'bg-apple-blue/5 dark:bg-apple-blue/10' : ''
                    }`}
                  >
                    <td className="px-4 py-3">
                      <button
                        onClick={() => handleSelectSuite(suite.id)}
                        className="text-left"
                      >
                        <div className="text-sm font-medium text-apple-gray-700 dark:text-white">
                          {suite.name}
                        </div>
                        <div className="text-xs text-apple-gray-400 flex items-center gap-1">
                          {suite.case_count} cases
                          {suite.mode_counts && Object.keys(suite.mode_counts).length > 1 && (
                            <span className="text-[9px] text-apple-gray-400">
                              ({Object.entries(suite.mode_counts).map(([m, c]) => `${m[0].toUpperCase()}${c}`).join(' ')})
                            </span>
                          )}
                        </div>
                      </button>
                    </td>
                    <td className="px-2 py-3 whitespace-nowrap">
                      <div className="flex items-center gap-1 flex-wrap">
                        {(() => {
                          const mode = SYNTHESIS_MODES.find(m => m.value === suite.default_synthesis_mode);
                          const modeColor = suite.default_synthesis_mode === 'comparison'
                            ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                            : suite.default_synthesis_mode === 'discovery'
                            ? 'bg-violet-100 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400'
                            : suite.default_synthesis_mode === 'routing'
                            ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                            : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400';
                          return (
                            <span
                              className={`px-2 py-0.5 text-[10px] font-medium rounded-full cursor-help ${modeColor}`}
                              title={mode?.tooltip || suite.default_synthesis_mode}
                            >
                              {mode?.label || suite.default_synthesis_mode}
                            </span>
                          );
                        })()}
                        {/* Difficulty distribution pills */}
                        {suite.difficulty_counts && Object.entries(suite.difficulty_counts).map(([diff, count]) => {
                          const diffColor = diff === 'hard'
                            ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                            : diff === 'easy'
                            ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                            : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400';
                          const diffTooltip = diff === 'hard'
                            ? 'Svær: kræver identifikation af 3+ love eller topic-baseret discovery'
                            : diff === 'easy'
                            ? 'Let: 2 love med få artikelankre'
                            : 'Medium: moderat kompleksitet';
                          return (
                            <span
                              key={diff}
                              className={`px-1.5 py-0.5 text-[10px] font-medium rounded-full cursor-help ${diffColor}`}
                              title={diffTooltip}
                            >
                              {diff === 'hard' ? 'Hård' : diff === 'easy' ? 'Let' : 'Medium'} {count}
                            </span>
                          );
                        })}
                      </div>
                    </td>
                    <td className="px-2 py-3 text-center whitespace-nowrap">
                      <span className="text-sm font-medium">
                        {formatPassRate(suite.pass_rate)}
                      </span>
                      <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 ml-1">
                        {suite.passed}/{suite.passed + suite.failed}
                      </span>
                    </td>
                    {SHARED_SCORE_TYPES.map((type) => {
                      const rate = suite.scorer_pass_rates?.[type];
                      if (rate === undefined || rate === null) {
                        return (
                          <td key={type} className="px-2 py-3 text-center whitespace-nowrap">
                            <span className="text-xs text-apple-gray-300 dark:text-apple-gray-600">—</span>
                          </td>
                        );
                      }
                      return (
                        <td key={type} className="px-2 py-3 text-center whitespace-nowrap">
                          <span className="text-xs font-medium">{formatPassRate(rate)}</span>
                        </td>
                      );
                    })}
                    {/* Dynamic mode-specific scorer column */}
                    {(() => {
                      const modeEntry = MODE_SPECIFIC_SCORERS.map(s => ({ key: s, rate: suite.scorer_pass_rates?.[s] }))
                        .find(e => e.rate !== undefined && e.rate !== null);
                      if (!modeEntry) {
                        // Discovery has no mode-specific scorer — show N/A
                        const isDiscovery = suite.default_synthesis_mode === 'discovery';
                        return (
                          <td className="px-2 py-3 text-center whitespace-nowrap">
                            <span className="text-xs text-apple-gray-300 dark:text-apple-gray-600">
                              {isDiscovery ? 'N/A' : '—'}
                            </span>
                          </td>
                        );
                      }
                      const label = CROSS_LAW_TEST_LABELS[modeEntry.key];
                      return (
                        <td className="px-2 py-3 text-center whitespace-nowrap">
                          <span className="text-xs font-medium">{formatPassRate(modeEntry.rate!)}</span>
                          <span className="text-[10px] text-apple-gray-400 ml-1">{label}</span>
                        </td>
                      );
                    })()}
                    <td className="px-2 py-3 text-center whitespace-nowrap">
                      {suite.last_run ? (
                        <div className="flex items-center justify-center gap-1.5">
                          <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                            {formatTimestamp(suite.last_run)}
                          </span>
                          {suite.last_run_mode && (
                            <span className={`px-1 py-0.5 text-[10px] font-medium rounded ${
                              suite.last_run_mode === 'retrieval_only'
                                ? 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                                : suite.last_run_mode === 'full_with_judge'
                                ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                                : 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                            }`}>
                              {suite.last_run_mode === 'retrieval_only' ? 'Ret.' : suite.last_run_mode === 'full_with_judge' ? 'Full+J' : 'Full'}
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-xs text-apple-gray-300 dark:text-apple-gray-600">—</span>
                      )}
                    </td>
                    <td className="px-3 py-3">
                      <div className="flex items-center gap-1">
                        <button
                          onClick={(e) => { e.stopPropagation(); handleEditCases(suite.id); }}
                          className="p-1.5 text-apple-gray-500 hover:text-apple-gray-700 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded transition-colors"
                          title="Rediger test cases"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                          </svg>
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); setEditingSuiteId(suite.id); }}
                          className="p-1.5 text-apple-gray-500 hover:text-apple-gray-700 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded transition-colors"
                          title="Suite indstillinger"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          </svg>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (openRunMenu === suite.id) {
                              setOpenRunMenu(null);
                            } else {
                              const rect = (e.currentTarget as HTMLButtonElement).getBoundingClientRect();
                              setRunMenuPosition({ x: rect.right, y: rect.bottom + 4 });
                              setOpenRunMenu(suite.id);
                            }
                          }}
                          disabled={triggeringSuite === suite.id}
                          className="p-1.5 text-apple-gray-500 hover:text-apple-gray-700 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded transition-colors disabled:opacity-50"
                          title="Kør eval"
                        >
                          {triggeringSuite === suite.id ? (
                            <div className="w-4 h-4 border-2 border-apple-blue border-t-transparent rounded-full animate-spin" />
                          ) : (
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          )}
                        </button>
                        <RunModePopover
                          isOpen={openRunMenu === suite.id}
                          position={runMenuPosition}
                          onClose={() => setOpenRunMenu(null)}
                          onSelect={(mode) => handleTriggerEval(suite.id, mode)}
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
              <tfoot className="sticky bottom-0 z-10">
                <tr className="border-t-2 border-apple-gray-200 dark:border-apple-gray-400 bg-apple-gray-50 dark:bg-apple-gray-700">
                  <td className="px-4 py-3">
                    <span className="text-sm font-semibold text-apple-gray-700 dark:text-white">
                      Total
                    </span>
                  </td>
                  <td className="px-2 py-3" />
                  <td className="px-2 py-3 text-center">
                    <span className="text-sm font-semibold">
                      {formatPassRate(overview.overall_pass_rate)}
                    </span>
                    <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 ml-1">
                      {filteredSuites.reduce((s, suite) => s + suite.passed, 0)}/{filteredSuites.reduce((s, suite) => s + suite.passed + suite.failed, 0)}
                    </span>
                  </td>
                  {SHARED_SCORE_TYPES.map((type) => {
                    const rates = filteredSuites
                      .map(s => s.scorer_pass_rates?.[type])
                      .filter((r): r is number => r !== undefined && r !== null);
                    if (rates.length === 0) {
                      return <td key={type} className="px-2 py-3 text-center"><span className="text-xs text-apple-gray-300 dark:text-apple-gray-600">—</span></td>;
                    }
                    const avg = rates.reduce((s, r) => s + r, 0) / rates.length;
                    return (
                      <td key={type} className="px-2 py-3 text-center">
                        <span className="text-sm font-semibold">{formatPassRate(avg)}</span>
                        <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 ml-1">
                          ({rates.length})
                        </span>
                      </td>
                    );
                  })}
                  {/* Mode-specific scorer average (excluding discovery which has no mode scorer) */}
                  {(() => {
                    const rates = filteredSuites
                      .map(s => {
                        const key = MODE_SCORER[s.default_synthesis_mode];
                        return key ? s.scorer_pass_rates?.[key] : undefined;
                      })
                      .filter((r): r is number => r !== undefined && r !== null);
                    if (rates.length === 0) {
                      return <td className="px-2 py-3 text-center"><span className="text-xs text-apple-gray-300 dark:text-apple-gray-600">—</span></td>;
                    }
                    const avg = rates.reduce((s, r) => s + r, 0) / rates.length;
                    return (
                      <td className="px-2 py-3 text-center">
                        <span className="text-sm font-semibold">{formatPassRate(avg)}</span>
                        <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 ml-1">
                          ({rates.length})
                        </span>
                      </td>
                    );
                  })()}
                  <td className="px-2 py-3" />
                  <td className="w-14" />
                </tr>
              </tfoot>
            </table>
            <div className="px-4 py-3 border-t border-apple-gray-100 dark:border-apple-gray-600">
              <button
                onClick={() => { setGenerateForSuiteId(null); setShowGenerateModal(true); }}
                className="text-sm text-apple-blue hover:text-apple-blue-hover font-medium"
              >
                + Auto-generer suite
              </button>
            </div>
          </div>
          </>
        )}
      </div>

      {/* Level 2: Suite Runs */}
      <AnimatePresence>
        {selectedSuiteId && selectedSuite && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className={`bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-hidden ${!selectedRunId ? 'flex flex-col flex-1 min-h-0' : ''}`}
          >
            {isRunsCollapsed ? (
              // Collapsed header
              <button
                onClick={handleBackToRuns}
                className="w-full px-4 py-3 flex items-center justify-between bg-apple-gray-50 dark:bg-apple-gray-600/50 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-apple-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  <span className="text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300">
                    Kørselshistorik · {runs.length} kørsler
                  </span>
                </div>
              </button>
            ) : (
              // Expanded runs
              <div>
                {/* Suite header */}
                <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-600 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={handleBackToMatrix}
                      className="p-1 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded"
                    >
                      <svg className="w-4 h-4 text-apple-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                    <div>
                      <div className="text-sm font-semibold text-apple-gray-700 dark:text-white">
                        {selectedSuite.name}
                      </div>
                      <div className="text-xs text-apple-gray-400">
                        {selectedSuite.case_count} test cases
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <select
                      value={triggerMode}
                      onChange={(e) => setTriggerMode(e.target.value as RunMode)}
                      disabled={triggeringSuite === selectedSuiteId}
                      className="px-2 py-1 text-sm rounded-md bg-transparent text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 transition-colors cursor-pointer"
                    >
                      <option value="retrieval_only">Retrieval</option>
                      <option value="full">Full</option>
                      <option value="full_with_judge">Full + Judge</option>
                    </select>
                    <select
                      value={triggerRetries}
                      onChange={(e) => setTriggerRetries(Number(e.target.value))}
                      disabled={triggeringSuite === selectedSuiteId}
                      aria-label="Antal forsøg"
                      title="Forsøg per case (0 = ingen gentagelser)"
                      className="px-2 py-1 text-sm rounded-md bg-transparent text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 transition-colors cursor-pointer"
                    >
                      <option value={0}>0×</option>
                      <option value={1}>1×</option>
                      <option value={3}>3×</option>
                    </select>
                    <button
                      onClick={() => handleTriggerEval()}
                      disabled={triggeringSuite !== null}
                      className="px-3 py-1 text-sm font-medium text-apple-blue hover:bg-apple-blue/10 rounded-md transition-colors disabled:opacity-50"
                    >
                      {triggeringSuite === selectedSuiteId ? (
                        <span className="flex items-center gap-1.5">
                          <div className="w-3 h-3 border-2 border-apple-blue border-t-transparent rounded-full animate-spin" />
                          Kører...
                        </span>
                      ) : (
                        'Kør'
                      )}
                    </button>
                  </div>
                </div>

                {/* Runs list */}
                {runs.length > 0 ? (
                  <div className="flex-1 min-h-0 overflow-y-auto divide-y divide-apple-gray-100 dark:divide-apple-gray-500/50">
                    {runs.map((run, idx) => (
                      <button
                        key={run.run_id}
                        onClick={() => handleSelectRun(run.run_id)}
                        className={`w-full flex items-center px-4 py-3 text-left transition-colors ${
                          selectedRunId === run.run_id
                            ? 'bg-apple-blue/5 dark:bg-apple-blue/10'
                            : 'hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500/30'
                        }`}
                      >
                        {/* Pass rate - fixed width */}
                        <span className="w-14 text-sm font-medium tabular-nums">
                          {formatPassRate(run.pass_rate)}
                        </span>

                        {/* Count - fixed width */}
                        <span className="w-14 text-sm text-apple-gray-600 dark:text-apple-gray-300 tabular-nums">
                          {run.passed}/{run.total}
                        </span>

                        {/* Run mode - fixed width */}
                        <span className="w-24 text-sm text-apple-gray-500 dark:text-apple-gray-400">
                          {RUN_MODE_LABELS[run.run_mode as RunMode]?.label || run.run_mode}
                        </span>

                        {/* Retries badge (cross-law only) */}
                        {(run.max_retries ?? 0) > 0 ? (
                          <span
                            className="w-10 text-xs px-1.5 py-0.5 bg-orange-100 dark:bg-orange-900/30 text-orange-600 dark:text-orange-400 rounded text-center"
                            aria-label={`${run.max_retries} forsøg`}
                          >
                            {run.max_retries}×
                          </span>
                        ) : (
                          <span className="w-10" />
                        )}

                        {/* Timestamp - fixed width */}
                        <span className="w-32 text-sm text-apple-gray-500 dark:text-apple-gray-400">
                          {formatTimestamp(run.timestamp)}
                        </span>

                        {/* Duration - fixed width */}
                        <span className="w-16 text-sm text-apple-gray-400 dark:text-apple-gray-500 tabular-nums">
                          {formatDuration(run.duration_seconds)}
                        </span>

                        {/* Newest badge - fixed width placeholder */}
                        <span className="w-14 text-sm font-medium text-apple-blue">
                          {idx === 0 ? 'Nyeste' : ''}
                        </span>

                        {/* Spacer pushes chevron right */}
                        <div className="flex-1" />

                        {/* Selection indicator + Chevron */}
                        <svg className={`w-4 h-4 flex-shrink-0 transition-colors ${
                          selectedRunId === run.run_id
                            ? 'text-apple-blue'
                            : 'text-apple-gray-300 dark:text-apple-gray-500'
                        }`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="px-4 py-8 text-center">
                    <svg className="w-8 h-8 mx-auto text-apple-gray-300 dark:text-apple-gray-500 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    <p className="text-sm text-apple-gray-400 dark:text-apple-gray-500">
                      Ingen kørsler endnu
                    </p>
                    <p className="text-xs text-apple-gray-300 dark:text-apple-gray-600 mt-1">
                      Klik &quot;Kør&quot; for at starte
                    </p>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Level 3: Run Details */}
      <AnimatePresence>
        {selectedRunId && runDetail && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="flex flex-col flex-1 min-h-0 bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-hidden"
          >
            {/* Header with filter */}
            <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-500 flex-shrink-0">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div>
                    <h3 className="text-sm font-semibold text-apple-gray-700 dark:text-white">
                      Test Cases
                    </h3>
                    <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 mt-1">
                      {runDetail.passed}/{runDetail.total} bestået · {formatDuration(runDetail.duration_seconds)}
                    </p>
                  </div>
                  <SegmentedControl
                    options={[
                      { value: 'all', label: 'Alle' },
                      { value: 'failed', label: 'Fejlet', selectedColor: 'red' },
                      { value: 'passed', label: 'Bestået', selectedColor: 'green' },
                    ]}
                    value={caseFilter}
                    onChange={setCaseFilter}
                    size="sm"
                  />
                </div>
              </div>
            </div>

            {/* Results list */}
            <div className="flex-1 min-h-0 divide-y divide-apple-gray-100 dark:divide-apple-gray-500 overflow-y-auto">
              {filteredResults.map((result) => (
                <div key={result.case_id}>
                  <button
                    onClick={() => setExpandedCase(expandedCase === result.case_id ? null : result.case_id)}
                    className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500/30 transition-colors"
                  >
                    {/* Pass/fail indicator */}
                    <span className={`flex-shrink-0 text-sm ${result.passed ? 'text-green-500' : 'text-red-500'}`}>
                      {result.passed ? '✓' : '✗'}
                    </span>

                    {/* Case info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-apple-gray-700 dark:text-white truncate">
                        {result.prompt}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                          {result.case_id}
                        </span>
                        <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-apple-gray-100 dark:bg-apple-gray-700 text-apple-gray-600 dark:text-apple-gray-300">
                          {result.target_corpora.join(', ')}
                        </span>
                        <span className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-apple-gray-100 dark:bg-apple-gray-700 text-apple-gray-600 dark:text-apple-gray-300">
                          {result.synthesis_mode}
                        </span>
                      </div>
                    </div>

                    {/* Duration */}
                    <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 tabular-nums">
                      {Math.round(result.duration_ms)}ms
                    </span>

                    {/* Expand icon */}
                    <svg
                      className={`w-4 h-4 text-apple-gray-400 transition-transform ${expandedCase === result.case_id ? 'rotate-180' : ''}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {/* Expanded detail */}
                  <AnimatePresence>
                    {expandedCase === result.case_id && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                      >
                        <div className="px-4 pb-4 space-y-3">
                          {/* Scores */}
                          <div className="space-y-2">
                          <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide">
                            Scorer
                          </p>
                          <div className="grid grid-cols-2 gap-2">
                            {Object.entries(result.scores).map(([name, value]) => {
                              const score = value as { passed: boolean; score: number; message?: string };
                              return (
                                <div
                                  key={name}
                                  className={`p-2 rounded-lg cursor-help ${score.passed ? 'bg-green-50 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/20'}`}
                                  title={CROSS_LAW_TEST_DESCRIPTIONS[name as CrossLawScoreType] || name}
                                >
                                  <div className="flex items-center justify-between">
                                    <span className="text-xs font-medium text-apple-gray-700 dark:text-white">
                                      {TEST_TYPE_TAG_LABELS[name] || name}
                                    </span>
                                    <span className={`text-xs ${score.passed ? 'text-green-600' : 'text-red-600'}`}>
                                      {typeof score.score === 'number' ? `${Math.round(score.score * 100)}%` : score.passed ? '✓' : '✗'}
                                    </span>
                                  </div>
                                  {score.message && (
                                    <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 mt-1 truncate">
                                      {score.message}
                                    </p>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                          {result.error && (
                            <div className="mt-2 text-xs text-red-500">
                              Error: {result.error}
                            </div>
                          )}
                          </div>

                          {/* Test Definition inline - matching single-law layout */}
                          {(() => {
                            const caseDef = suiteDetail?.cases.find(c => c.id === result.case_id);
                            if (!caseDef) return null;
                            return (
                              <div className="space-y-2">
                                <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide">
                                  Testdefinition
                                </p>
                                <div className="flex flex-wrap items-center gap-2">
                                  <span
                                    className="px-1.5 py-0.5 text-[10px] rounded bg-apple-gray-200 dark:bg-apple-gray-600 text-apple-gray-600 dark:text-apple-gray-300 cursor-help"
                                    title={caseDef.profile === 'LEGAL' ? 'Målgruppe: Jurister og compliance-ansvarlige' : 'Målgruppe: Udviklere og teknisk personale'}
                                  >
                                    {caseDef.profile}
                                  </span>
                                  <span
                                    className={`px-1.5 py-0.5 text-[10px] rounded cursor-help ${caseDef.origin === 'manual' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'}`}
                                    title={caseDef.origin === 'manual' ? 'Manuelt oprettet test case med verificerede forventninger' : 'Automatisk genereret test case baseret på lovteksten'}
                                  >
                                    {caseDef.origin === 'manual' ? 'Manuel' : 'Auto'}
                                  </span>
                                  <span className="text-apple-gray-300 dark:text-apple-gray-500">|</span>
                                  {caseDef.must_include_any_of && caseDef.must_include_any_of.length > 0 && (
                                    <>
                                      <span className="text-[10px] text-green-600 dark:text-green-400" title="Svaret skal referere til mindst én af disse artikler">≥1:</span>
                                      {caseDef.must_include_any_of.map((item, i) => (
                                        <span key={`any-${i}`} className="px-1.5 py-0.5 text-[10px] rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
                                          {item}
                                        </span>
                                      ))}
                                    </>
                                  )}
                                  {caseDef.must_include_all_of && caseDef.must_include_all_of.length > 0 && (
                                    <>
                                      <span className="text-[10px] text-blue-600 dark:text-blue-400" title="Svaret skal referere til alle disse artikler">Alle:</span>
                                      {caseDef.must_include_all_of.map((item, i) => (
                                        <span key={`all-${i}`} className="px-1.5 py-0.5 text-[10px] rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">
                                          {item}
                                        </span>
                                      ))}
                                    </>
                                  )}
                                  {caseDef.must_not_include_any_of && caseDef.must_not_include_any_of.length > 0 && (
                                    <>
                                      <span className="text-[10px] text-red-600 dark:text-red-400" title="Svaret må ikke referere til disse artikler">Ikke:</span>
                                      {caseDef.must_not_include_any_of.map((item, i) => (
                                        <span key={`not-${i}`} className="px-1.5 py-0.5 text-[10px] rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400">
                                          {item}
                                        </span>
                                      ))}
                                    </>
                                  )}
                                  {caseDef.expected_behavior && (
                                    <span
                                      className={`px-1.5 py-0.5 text-[10px] rounded ${
                                        caseDef.expected_behavior === 'answer'
                                          ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                                          : 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
                                      }`}
                                      title={caseDef.expected_behavior === 'answer' ? 'Systemet skal svare' : 'Systemet skal afstå'}
                                    >
                                      {caseDef.expected_behavior === 'answer' ? 'Skal svare' : 'Skal afstå'}
                                    </span>
                                  )}
                                </div>
                              </div>
                            );
                          })()}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Test Cases Panel */}
      <AnimatePresence>
        {viewingCasesSuiteId && suiteDetail && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="flex flex-col flex-1 min-h-0 bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-hidden"
          >
            {isCasesCollapsed ? (
              // Collapsed header
              <button
                onClick={() => setIsCasesCollapsed(false)}
                className="w-full px-4 py-3 flex items-center justify-between bg-apple-gray-50 dark:bg-apple-gray-600/50 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-apple-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  <span className="text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300">
                    Test Cases · {suiteDetail.cases.length} cases
                  </span>
                </div>
              </button>
            ) : (
              // Expanded cases view
              <div className="flex flex-col">
                {/* Header */}
                <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-500">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <button
                        onClick={handleBackFromCases}
                        className="p-1.5 text-apple-gray-400 hover:text-apple-gray-600 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded-lg transition-colors"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                      <span className="text-sm font-medium text-apple-gray-700 dark:text-white">
                        {suiteDetail.name}
                      </span>
                      <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                        {suiteDetail.cases.length} {suiteDetail.cases.length === 1 ? 'case' : 'cases'}
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <select
                        value={synthesisFilter}
                        onChange={(e) => setSynthesisFilter(e.target.value as typeof synthesisFilter)}
                        className="px-2 py-1 text-sm rounded-md bg-transparent text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 transition-colors cursor-pointer"
                      >
                        <option value="all">Alle</option>
                        <option value="comparison">Comparison</option>
                        <option value="aggregation">Aggregation</option>
                        <option value="routing">Routing</option>
                        <option value="discovery">Discovery</option>
                      </select>
                      <button
                        onClick={() => { setGenerateForSuiteId(viewingCasesSuiteId); setShowGenerateModal(true); }}
                        className="px-3 py-1 text-sm font-medium text-apple-blue hover:bg-apple-blue/10 rounded-md transition-colors"
                      >
                        Generer flere
                      </button>
                      <button
                        onClick={handleCreateCase}
                        className="px-3 py-1 text-sm font-medium text-apple-blue hover:bg-apple-blue/10 rounded-md transition-colors"
                      >
                        Ny
                      </button>
                    </div>
                  </div>

                  {/* Search */}
                  <div className="mt-3">
                    <SearchInput
                      value={casesSearchQuery}
                      onChange={setCasesSearchQuery}
                      placeholder="Søg i cases..."
                      className="w-full"
                    />
                  </div>
                </div>

                {/* Disclaimer banner for auto-generated suites (R7.1) */}
                {suiteDetail.cases.some((c) => c.origin === 'auto-generated') && (
                  <div className="mx-4 mt-3 px-3 py-2 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg flex items-start gap-2">
                    <svg className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                    </svg>
                    <p className="text-xs text-amber-700 dark:text-amber-400">
                      Denne suite indeholder auto-genererede cases. Gennemse venligst alle cases manuelt for nøjagtighed inden evaluering.
                    </p>
                  </div>
                )}

                {/* Cases list */}
                <div className="flex-1 min-h-0 overflow-y-auto">
                  {filteredTestCases.length === 0 ? (
                    <div className="p-8 text-center">
                      <svg className="w-12 h-12 mx-auto text-apple-gray-300 dark:text-apple-gray-500 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                      </svg>
                      <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">
                        {casesSearchQuery || synthesisFilter !== 'all' ? 'Ingen cases matcher filteret' : 'Ingen eval cases endnu'}
                      </p>
                      {!casesSearchQuery && synthesisFilter === 'all' && (
                        <button
                          onClick={handleCreateCase}
                          className="mt-3 text-sm text-apple-blue hover:underline"
                        >
                          Opret den første case
                        </button>
                      )}
                    </div>
                  ) : (
                    <div className="divide-y divide-apple-gray-100 dark:divide-apple-gray-500">
                      {filteredTestCases.map((testCase) => (
                        <div key={testCase.id}>
                          {/* Case row - clickable to expand */}
                          <div
                            className="px-4 py-3 flex items-center gap-3 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500/30 transition-colors cursor-pointer"
                            onClick={() => setExpandedTestCase(expandedTestCase === testCase.id ? null : testCase.id)}
                          >
                            {/* Synthesis mode badge */}
                            <span
                              className={`flex-shrink-0 px-1.5 py-0.5 text-[10px] font-medium rounded ${
                                testCase.synthesis_mode === 'comparison'
                                  ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                                  : testCase.synthesis_mode === 'discovery'
                                  ? 'bg-violet-100 dark:bg-violet-900/30 text-violet-600 dark:text-violet-400'
                                  : testCase.synthesis_mode === 'routing'
                                  ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                                  : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                              }`}
                            >
                              {testCase.synthesis_mode.toUpperCase()}
                            </span>

                            {/* Difficulty badge */}
                            {testCase.difficulty && (
                              <span
                                className={`flex-shrink-0 px-1.5 py-0.5 text-[10px] font-medium rounded ${
                                  testCase.difficulty === 'hard'
                                    ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400'
                                    : testCase.difficulty === 'easy'
                                    ? 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'
                                    : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-600 dark:text-yellow-400'
                                }`}
                              >
                                {testCase.difficulty.toUpperCase()}
                              </span>
                            )}

                            {/* Origin badge — matches single-law purple/blue style */}
                            <span
                              className={`flex-shrink-0 px-1.5 py-0.5 text-[10px] font-medium rounded ${
                                testCase.origin === 'manual'
                                  ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                                  : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                              }`}
                            >
                              {testCase.origin === 'manual' ? 'MANUAL' : 'AUTO'}
                            </span>

                            {/* Disabled indicator */}
                            {testCase.disabled && (
                              <span className="flex-shrink-0 px-1.5 py-0.5 text-[10px] font-medium rounded bg-apple-gray-100 dark:bg-apple-gray-700 text-apple-gray-400 dark:text-apple-gray-500">
                                OFF
                              </span>
                            )}

                            {/* Case info */}
                            <div className="flex-1 min-w-0">
                              <p className={`text-sm truncate ${testCase.disabled ? 'text-apple-gray-400 dark:text-apple-gray-500 line-through' : 'text-apple-gray-700 dark:text-white'}`}>
                                {testCase.prompt}
                              </p>
                              <div className="flex items-center gap-2 mt-1 flex-wrap">
                                <span className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500">
                                  {testCase.id}
                                </span>
                                {/* Test type tags */}
                                {testCase.test_types?.map((tt) => (
                                  <span
                                    key={tt}
                                    className="px-1 py-0.5 text-[10px] rounded bg-apple-gray-100 dark:bg-apple-gray-700 text-apple-gray-600 dark:text-apple-gray-300"
                                  >
                                    {TEST_TYPE_TAG_LABELS[tt] || tt}
                                  </span>
                                ))}
                                {testCase.target_corpora.slice(0, 3).map((corpus) => (
                                  <span
                                    key={corpus}
                                    className="px-1 py-0.5 text-[10px] rounded bg-apple-gray-100 dark:bg-apple-gray-700 text-apple-gray-600 dark:text-apple-gray-300"
                                  >
                                    {corpus}
                                  </span>
                                ))}
                                {testCase.target_corpora.length > 3 && (
                                  <span className="text-[10px] text-apple-gray-400">
                                    +{testCase.target_corpora.length - 3}
                                  </span>
                                )}
                              </div>
                            </div>

                            {/* Expand icon */}
                            <svg
                              className={`w-4 h-4 text-apple-gray-400 transition-transform ${
                                expandedTestCase === testCase.id ? 'rotate-180' : ''
                              }`}
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                            >
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                            </svg>
                          </div>

                          {/* Expanded details */}
                          <AnimatePresence>
                            {expandedTestCase === testCase.id && (
                              <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="overflow-hidden bg-apple-gray-50/50 dark:bg-apple-gray-500/20 border-t border-apple-gray-100 dark:border-apple-gray-600"
                              >
                                <div className="px-4 py-3 pl-6 space-y-3">
                                  {/* Target corpora */}
                                  <div>
                                    <p className="text-[11px] font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                                      Target love
                                    </p>
                                    <div className="flex flex-wrap items-center gap-2">
                                      {testCase.target_corpora.map((corpus) => (
                                        <span
                                          key={corpus}
                                          className="px-2 py-0.5 text-xs font-medium rounded-full bg-apple-gray-200 dark:bg-apple-gray-600 text-apple-gray-600 dark:text-apple-gray-300"
                                        >
                                          {corpus}
                                        </span>
                                      ))}
                                    </div>
                                  </div>

                                  {/* Expected articles */}
                                  {testCase.expected_articles && testCase.expected_articles.length > 0 && (
                                    <div>
                                      <p className="text-[11px] font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                                        Forventede artikler
                                      </p>
                                      <div className="flex flex-wrap items-center gap-2">
                                        {testCase.expected_articles.map((article) => (
                                          <span
                                            key={article}
                                            className="px-2 py-0.5 text-xs font-medium rounded-full bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400"
                                          >
                                            {article}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  )}

                                  {/* Expected behavior — matches single-law pattern */}
                                  <div>
                                    <p className="text-[11px] font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                                      Forventet adfærd
                                    </p>
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                                        testCase.expected_behavior === 'answer'
                                          ? 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400'
                                          : 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400'
                                      }`}>
                                        {testCase.expected_behavior === 'answer' ? 'Skal svare' : 'Skal afstå'}
                                      </span>

                                      {testCase.must_include_any_of && testCase.must_include_any_of.length > 0 && (
                                        <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                                          Kræver én af: {testCase.must_include_any_of.join(', ')}
                                        </span>
                                      )}

                                      {testCase.must_include_any_of_2 && testCase.must_include_any_of_2.length > 0 && (
                                        <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                                          Kræver også én af: {testCase.must_include_any_of_2.join(', ')}
                                        </span>
                                      )}

                                      {testCase.must_include_all_of && testCase.must_include_all_of.length > 0 && (
                                        <span className="text-xs text-blue-500 dark:text-blue-400">
                                          Skal inkludere alle: {testCase.must_include_all_of.join(', ')}
                                        </span>
                                      )}

                                      {testCase.must_not_include_any_of && testCase.must_not_include_any_of.length > 0 && (
                                        <span className="text-xs text-red-500">
                                          Må ikke: {testCase.must_not_include_any_of.join(', ')}
                                        </span>
                                      )}

                                      {testCase.contract_check && (
                                        <span className="text-xs text-purple-500 dark:text-purple-400">
                                          Citeringer: {testCase.min_citations ?? 0}–{testCase.max_citations ?? '∞'}
                                        </span>
                                      )}
                                    </div>
                                  </div>

                                  {testCase.notes && (
                                    <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 italic">
                                      {testCase.notes}
                                    </p>
                                  )}

                                  {/* Actions */}
                                  <div className="flex gap-3 pt-1">
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleEditCase(testCase.id);
                                      }}
                                      className="text-xs font-medium text-apple-blue hover:text-apple-blue/80 transition-colors"
                                    >
                                      Rediger
                                    </button>
                                    <button
                                      onClick={async (e) => {
                                        e.stopPropagation();
                                        try {
                                          const resp = await fetch(`${API_BASE}/eval/cross-law/suites/${viewingCasesSuiteId}/cases/${testCase.id}/duplicate`, {
                                            method: 'POST',
                                          });
                                          if (resp.ok) {
                                            const detailResp = await fetch(`${API_BASE}/eval/cross-law/suites/${viewingCasesSuiteId}`);
                                            if (detailResp.ok) {
                                              const data = await detailResp.json();
                                              setSuiteDetail(data);
                                            }
                                            loadOverview();
                                          }
                                        } catch (err) {
                                          console.error('Failed to duplicate case:', err);
                                        }
                                      }}
                                      className="text-xs font-medium text-apple-gray-500 dark:text-apple-gray-400 hover:text-apple-gray-700 dark:hover:text-apple-gray-200 transition-colors"
                                    >
                                      Duplikér
                                    </button>
                                    {caseDeleteConfirm === testCase.id ? (
                                      <div className="flex items-center gap-2">
                                        <span className="text-xs text-red-500">Slet?</span>
                                        <button
                                          onClick={async (e) => {
                                            e.stopPropagation();
                                            setIsCaseDeleting(true);
                                            try {
                                              const resp = await fetch(`${API_BASE}/eval/cross-law/suites/${viewingCasesSuiteId}/cases/${testCase.id}`, {
                                                method: 'DELETE',
                                              });
                                              if (resp.ok) {
                                                const detailResp = await fetch(`${API_BASE}/eval/cross-law/suites/${viewingCasesSuiteId}`);
                                                if (detailResp.ok) {
                                                  const data = await detailResp.json();
                                                  setSuiteDetail(data);
                                                }
                                                loadOverview();
                                                setExpandedTestCase(null);
                                              }
                                            } catch (err) {
                                              console.error('Failed to delete case:', err);
                                            } finally {
                                              setIsCaseDeleting(false);
                                              setCaseDeleteConfirm(null);
                                            }
                                          }}
                                          disabled={isCaseDeleting}
                                          className="px-2 py-0.5 text-xs font-medium text-white bg-red-500 hover:bg-red-600 rounded transition-colors disabled:opacity-50"
                                        >
                                          {isCaseDeleting ? '...' : 'Ja'}
                                        </button>
                                        <button
                                          onClick={(e) => { e.stopPropagation(); setCaseDeleteConfirm(null); }}
                                          className="px-2 py-0.5 text-xs font-medium text-apple-gray-600 hover:bg-apple-gray-200 dark:hover:bg-apple-gray-600 rounded transition-colors"
                                        >
                                          Nej
                                        </button>
                                      </div>
                                    ) : (
                                      <button
                                        onClick={(e) => { e.stopPropagation(); setCaseDeleteConfirm(testCase.id); }}
                                        className="text-xs font-medium text-red-500 hover:text-red-600 transition-colors"
                                      >
                                        Slet
                                      </button>
                                    )}
                                  </div>
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Edit suite modal */}
      <AnimatePresence>
        {editingSuiteId && (
          <EditSuiteModal
            suiteId={editingSuiteId}
            existingSuite={overview?.suites.find((s) => s.id === editingSuiteId)}
            corporaList={availableCorpora}
            onClose={() => setEditingSuiteId(null)}
            onUpdated={() => {
              setEditingSuiteId(null);
              loadOverview();
            }}
          />
        )}
      </AnimatePresence>

      {/* Case editor modal */}
      <AnimatePresence>
        {showCaseEditor && viewingCasesSuiteId && (
          <CrossLawCaseEditor
            suiteId={viewingCasesSuiteId}
            caseId={editingCaseId}
            existingCase={editingCaseId ? suiteDetail?.cases.find((c) => c.id === editingCaseId) : undefined}
            parentSuite={suiteDetail!}
            onClose={() => { setShowCaseEditor(false); setEditingCaseId(null); }}
            onSaved={handleCaseSaved}
          />
        )}
      </AnimatePresence>

      {/* Generate cases modal */}
      <AnimatePresence>
        {showGenerateModal && (
          <GenerateCasesModal
            existingSuiteId={generateForSuiteId}
            existingSuite={generateForSuiteId ? suiteDetail : null}
            corporaList={availableCorpora}
            onClose={() => { setShowGenerateModal(false); setGenerateForSuiteId(null); }}
            onGenerated={(suiteId) => {
              setShowGenerateModal(false);
              setGenerateForSuiteId(null);
              loadOverview();
              if (suiteId) {
                setViewingCasesSuiteId(suiteId);
              }
            }}
          />
        )}
      </AnimatePresence>

      {/* Running eval progress footer — matches AdminPage single-law style */}
      <AnimatePresence>
        {runProgress && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-6 left-6 right-6 z-50 bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-hidden"
          >
            <div className="px-4 py-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  {runProgress.stage !== 'complete' && (
                    <div className="w-4 h-4 border-2 border-apple-blue border-t-transparent rounded-full animate-spin" />
                  )}
                  {runProgress.stage === 'complete' && (
                    <svg className="w-4 h-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                  <span className="text-sm font-medium text-apple-gray-700 dark:text-white">
                    {runProgress.stage === 'complete' ? 'Eval afsluttet' : 'Kører eval...'}
                  </span>
                  <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                    {runProgress.suiteName}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-green-500 font-medium">{runProgress.passed} ✓</span>
                    <span className="text-red-500 font-medium">{runProgress.failed} ✗</span>
                    <span className="text-apple-gray-400">/ {runProgress.total}</span>
                  </div>
                  {runProgress.stage === 'complete' && (
                    <button
                      onClick={() => setRunProgress(null)}
                      className="px-3 py-1 text-sm font-medium text-apple-blue hover:bg-apple-blue/10 rounded-md transition-colors"
                    >
                      Luk
                    </button>
                  )}
                </div>
              </div>
              {runProgress.total > 0 && (
                <div className="h-1 bg-apple-gray-100 dark:bg-apple-gray-700 rounded-full overflow-hidden">
                  {runProgress.stage === 'complete' ? (
                    <div className="h-full flex">
                      <div
                        className="h-full bg-green-500"
                        style={{ width: `${(runProgress.passed / runProgress.total) * 100}%` }}
                      />
                      <div
                        className="h-full bg-red-500"
                        style={{ width: `${(runProgress.failed / runProgress.total) * 100}%` }}
                      />
                    </div>
                  ) : (
                    <div
                      className="h-full bg-apple-blue transition-all duration-300"
                      style={{ width: `${((runProgress.passed + runProgress.failed) / runProgress.total) * 100}%` }}
                    />
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Spacer when progress bar is visible to prevent content overlap */}
      {runProgress && <div className="h-28" aria-hidden="true" />}
    </div>
  );
}

// Cross-law case editor modal (redesigned to match single-law EvalCaseEditor)
function CrossLawCaseEditor({
  suiteId,
  caseId,
  existingCase,
  parentSuite,
  onClose,
  onSaved,
}: {
  suiteId: string;
  caseId: string | null;
  existingCase?: CrossLawTestCase;
  parentSuite: CrossLawSuiteDetail;
  onClose: () => void;
  onSaved: () => void;
}) {
  const isEditing = !!existingCase;

  // Form state
  const [profile, setProfile] = useState<'LEGAL' | 'ENGINEERING'>(
    (existingCase?.profile as 'LEGAL' | 'ENGINEERING') || 'LEGAL'
  );
  const [prompt, setPrompt] = useState(existingCase?.prompt || '');
  // Test types derived from synthesis mode — no manual selection
  // Discovery has no mode-specific scorer (corpus_coverage handles it via threshold)
  const modeScorer = MODE_SCORER[parentSuite.default_synthesis_mode];
  const derivedTestTypes = ['retrieval', 'faithfulness', 'relevancy', 'corpus_coverage', ...(modeScorer ? [modeScorer] : [])];
  const [disabledState, setDisabledState] = useState(existingCase?.disabled || false);
  const [expectedBehavior, setExpectedBehavior] = useState(existingCase?.expected_behavior || 'answer');
  const [expectedCorpora, setExpectedCorpora] = useState<string[]>(
    existingCase?.expected_corpora || [...parentSuite.target_corpora]
  );
  const [minCorporaCited, setMinCorporaCited] = useState(existingCase?.min_corpora_cited ?? 2);
  const [mustIncludeAnyOf, setMustIncludeAnyOf] = useState<string[]>(existingCase?.must_include_any_of || []);
  const [mustIncludeAnyOf2, setMustIncludeAnyOf2] = useState<string[]>(existingCase?.must_include_any_of_2 || []);
  const [mustIncludeAllOf, setMustIncludeAllOf] = useState<string[]>(existingCase?.must_include_all_of || []);
  const [mustNotIncludeAnyOf, setMustNotIncludeAnyOf] = useState<string[]>(existingCase?.must_not_include_any_of || []);
  const [contractCheck, setContractCheck] = useState(existingCase?.contract_check || false);
  const [minCitations, setMinCitations] = useState<number | null>(existingCase?.min_citations ?? null);
  const [maxCitations, setMaxCitations] = useState<number | null>(existingCase?.max_citations ?? null);
  const [notes, setNotes] = useState(existingCase?.notes || '');

  // UI state
  const [error, setError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Validation state
  type RunMode = 'retrieval_only' | 'full' | 'full_with_judge';
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<CrossLawValidationResult | null>(null);
  const [validationMode, setValidationMode] = useState<RunMode>('full');
  const [showModeDropdown, setShowModeDropdown] = useState(false);
  const validationResultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (validationResult && validationResultRef.current) {
      validationResultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [validationResult]);

  const handleChange = useCallback(() => {
    setHasUnsavedChanges(true);
    setError(null);
  }, []);

  const handleClose = () => {
    if (hasUnsavedChanges) {
      if (!confirm('Du har ugemte ændringer. Er du sikker på at du vil lukke?')) return;
    }
    onClose();
  };

  // Escape key handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') handleClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [hasUnsavedChanges]);

  const toggleExpectedCorpus = (c: string) => {
    handleChange();
    setExpectedCorpora((prev) =>
      prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]
    );
  };

  const validateForm = (): string | null => {
    if (prompt.trim().length < 10) return 'Prompt skal være mindst 10 tegn';
    if (expectedBehavior === 'answer' && expectedCorpora.length === 0)
      return 'Vælg mindst én forventet lov';
    return null;
  };

  const handleSave = async () => {
    const err = validateForm();
    if (err) { setError(err); return; }

    setIsSaving(true);
    setError(null);

    try {
      const payload = {
        prompt: prompt.trim(),
        profile,
        test_types: derivedTestTypes,
        disabled: disabledState,
        expected_behavior: expectedBehavior,
        expected_corpora: expectedCorpora,
        min_corpora_cited: minCorporaCited,
        must_include_any_of: mustIncludeAnyOf,
        must_include_any_of_2: mustIncludeAnyOf2,
        must_include_all_of: mustIncludeAllOf,
        must_not_include_any_of: mustNotIncludeAnyOf,
        contract_check: contractCheck,
        min_citations: minCitations,
        max_citations: maxCitations,
        notes,
        // Suite-level fields (read-only in editor, but required for API)
        synthesis_mode: parentSuite.default_synthesis_mode,
        target_corpora: parentSuite.target_corpora,
      };

      let resp;
      if (isEditing && caseId) {
        resp = await fetch(`${API_BASE}/eval/cross-law/suites/${suiteId}/cases/${caseId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      } else {
        resp = await fetch(`${API_BASE}/eval/cross-law/suites/${suiteId}/cases`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      }

      if (!resp.ok) {
        const data = await resp.json();
        throw new Error(data.detail || 'Kunne ikke gemme');
      }

      setHasUnsavedChanges(false);
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Kunne ikke gemme');
    } finally {
      setIsSaving(false);
    }
  };

  const handleValidate = async () => {
    const err = validateForm();
    if (err) { setError(err); return; }

    setIsValidating(true);
    setError(null);
    setValidationResult(null);

    try {
      const result = await runSingleCrossLawCase(suiteId, {
        prompt: prompt.trim(),
        profile,
        test_types: derivedTestTypes,
        run_mode: validationMode,
        expected_behavior: expectedBehavior,
        expected_corpora: expectedCorpora,
        min_corpora_cited: minCorporaCited,
        must_include_any_of: mustIncludeAnyOf,
        must_include_any_of_2: mustIncludeAnyOf2,
        must_include_all_of: mustIncludeAllOf,
        must_not_include_any_of: mustNotIncludeAnyOf,
        contract_check: contractCheck,
        min_citations: minCitations,
        max_citations: maxCitations,
      });
      setValidationResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Validering fejlede');
    } finally {
      setIsValidating(false);
    }
  };

  const MODE_LABELS: Record<RunMode, { short: string; long: string }> = {
    retrieval_only: { short: 'Ret.', long: 'Kun retrieval (hurtig)' },
    full: { short: 'Full', long: 'Full (med LLM)' },
    full_with_judge: { short: 'Full+J', long: 'Full + LLM Judge' },
  };

  return createPortal(
    <AnimatePresence>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 z-50"
        onClick={handleClose}
      />

      {/* Modal */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.95, y: 20 }}
        className="fixed inset-4 md:inset-y-[5vh] md:inset-x-auto md:left-1/2 md:-translate-x-1/2 md:w-full md:max-w-2xl bg-white dark:bg-apple-gray-600 rounded-2xl shadow-2xl z-50 flex flex-col overflow-hidden"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-apple-gray-100 dark:border-apple-gray-500">
          <div className="flex items-center gap-3">
            <div>
              <h2 className="text-lg font-semibold text-apple-gray-700 dark:text-white">
                {isEditing ? 'Rediger Test Case' : 'Ny Test Case'}
              </h2>
              {isEditing && caseId && (
                <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500 mt-0.5">
                  {caseId}
                </p>
              )}
            </div>
            {isEditing && existingCase?.origin && (
              <span
                className="text-[10px] px-2 py-0.5 rounded-full bg-apple-gray-100 dark:bg-apple-gray-500 text-apple-gray-500 dark:text-apple-gray-300"
                aria-label={`Oprindelse: ${existingCase.origin}`}
              >
                {existingCase.origin}
              </span>
            )}
          </div>
          <button
            onClick={handleClose}
            className="p-2 text-apple-gray-400 hover:text-apple-gray-600 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
          {/* Suite context (read-only) */}
          <div className="bg-apple-gray-50 dark:bg-apple-gray-600/50 rounded-lg p-3" aria-label="Suite kontekst (skrivebeskyttet)">
            <div className="mb-2">
              <span className="text-[10px] uppercase tracking-wide text-apple-gray-400 dark:text-apple-gray-500">
                Synthesis mode
              </span>
              <p className="text-sm font-medium text-apple-gray-700 dark:text-white capitalize">
                {parentSuite.default_synthesis_mode}
              </p>
            </div>
            <div>
              <span className="text-[10px] uppercase tracking-wide text-apple-gray-400 dark:text-apple-gray-500">
                Love
              </span>
              <div className="flex flex-wrap gap-1.5 mt-1">
                {parentSuite.target_corpora.map((c) => (
                  <span
                    key={c}
                    className="px-2 py-0.5 text-xs rounded-full bg-apple-gray-100 dark:bg-apple-gray-500 text-apple-gray-600 dark:text-apple-gray-300"
                  >
                    {c}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Profile radio */}
          <div>
            <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
              Profil *
            </label>
            <div className="flex gap-4" role="radiogroup" aria-label="Profil">
              {(['LEGAL', 'ENGINEERING'] as const).map((p) => (
                <label key={p} className="flex items-center gap-2 text-sm text-apple-gray-700 dark:text-white cursor-pointer">
                  <input
                    type="radio"
                    name="profile"
                    value={p}
                    checked={profile === p}
                    onChange={() => { setProfile(p); handleChange(); }}
                    className="w-4 h-4 text-apple-blue"
                  />
                  {p}
                </label>
              ))}
            </div>
          </div>

          {/* Prompt */}
          <div>
            <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
              Prompt *
            </label>
            <textarea
              value={prompt}
              onChange={(e) => { setPrompt(e.target.value); handleChange(); }}
              className="w-full px-3 py-2 text-sm bg-white dark:bg-apple-gray-700 border border-apple-gray-200 dark:border-apple-gray-500 rounded-lg focus:outline-none focus:ring-2 focus:ring-apple-blue/50 resize-none"
              rows={3}
              placeholder="Skriv testspørgsmålet her..."
              autoFocus
            />
            <p className="text-[10px] text-apple-gray-400 mt-0.5">Min. 10 tegn</p>
          </div>

          {/* Disabled toggle */}
          <label className="flex items-center gap-2 text-sm text-apple-gray-700 dark:text-white cursor-pointer">
            <input
              type="checkbox"
              checked={disabledState}
              onChange={(e) => { setDisabledState(e.target.checked); handleChange(); }}
              className="w-4 h-4 rounded text-apple-blue"
            />
            Deaktiveret
          </label>

          {/* ── Forventet Adfærd divider ── */}
          <div className="border-t border-apple-gray-100 dark:border-apple-gray-500 pt-4">
            <h3 className="text-sm font-semibold text-apple-gray-700 dark:text-white mb-3">
              Forventet Adfærd
            </h3>

            {/* Behavior radio */}
            <div className="mb-4">
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                Forventet respons
              </label>
              <div className="flex gap-4" role="radiogroup" aria-label="Forventet respons">
                {([
                  { value: 'answer', label: 'Svar (answer)' },
                  { value: 'abstain', label: 'Afstå (abstain)' },
                ] as const).map((b) => (
                  <label key={b.value} className="flex items-center gap-2 text-sm text-apple-gray-700 dark:text-white cursor-pointer">
                    <input
                      type="radio"
                      name="behavior"
                      value={b.value}
                      checked={expectedBehavior === b.value}
                      onChange={() => { setExpectedBehavior(b.value); handleChange(); }}
                      className="w-4 h-4 text-apple-blue"
                    />
                    {b.label}
                  </label>
                ))}
              </div>
            </div>

            {/* Expected corpora checkboxes */}
            <div className="mb-4">
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                Forventede love *
              </label>
              <div className="space-y-1">
                {parentSuite.target_corpora.map((c) => (
                  <label key={c} className="flex items-center gap-2 text-sm text-apple-gray-700 dark:text-white cursor-pointer">
                    <input
                      type="checkbox"
                      checked={expectedCorpora.includes(c)}
                      onChange={() => toggleExpectedCorpus(c)}
                      className="w-4 h-4 rounded text-apple-blue"
                    />
                    {c}
                  </label>
                ))}
              </div>
            </div>

            {/* Min corpora cited */}
            <div className="mb-4">
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-1">
                Min. love citeret
              </label>
              <input
                type="number"
                value={minCorporaCited}
                min={1}
                max={parentSuite.target_corpora.length}
                onChange={(e) => { setMinCorporaCited(Number(e.target.value)); handleChange(); }}
                className="w-20 px-2 py-1 text-sm rounded border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700"
              />
              <span className="text-[10px] text-apple-gray-400 ml-2">
                (1–{parentSuite.target_corpora.length})
              </span>
            </div>

            {/* Anchor lists */}
            <AnchorListInput
              label="Skal inkludere mindst én af"
              description="Mindst ét af disse ankre skal findes i svaret"
              values={mustIncludeAnyOf}
              onChange={(v) => { setMustIncludeAnyOf(v); handleChange(); }}
              placeholder="corpus-id:article:5"
            />
            <AnchorListInput
              label="Skal inkludere mindst én af (sæt 2)"
              description="Sekundært sæt — mindst ét af disse skal også findes"
              values={mustIncludeAnyOf2}
              onChange={(v) => { setMustIncludeAnyOf2(v); handleChange(); }}
              placeholder="corpus-id:article:5"
            />
            <AnchorListInput
              label="Skal inkludere alle"
              description="Alle disse ankre skal findes i svaret"
              values={mustIncludeAllOf}
              onChange={(v) => { setMustIncludeAllOf(v); handleChange(); }}
              placeholder="corpus-id:article:5"
            />
            <AnchorListInput
              label="Må ikke inkludere"
              description="Ingen af disse ankre må findes i svaret"
              values={mustNotIncludeAnyOf}
              onChange={(v) => { setMustNotIncludeAnyOf(v); handleChange(); }}
              placeholder="corpus-id:article:5"
            />

            {/* Citation contract */}
            <div className="mb-4 mt-4">
              <label className="flex items-center gap-2 text-sm text-apple-gray-700 dark:text-white cursor-pointer mb-2">
                <input
                  type="checkbox"
                  checked={contractCheck}
                  onChange={(e) => { setContractCheck(e.target.checked); handleChange(); }}
                  className="w-4 h-4 rounded text-apple-blue"
                />
                Aktiver citation constraints
              </label>
              <p className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500 mt-1 ml-6">
                Validerer at svaret har et specifikt antal citationer. Brug min/max felterne til at sætte grænser.
              </p>
              {contractCheck && (
                <div className="flex gap-4 mt-2 ml-6">
                  <div>
                    <label className="block text-[10px] text-apple-gray-500 mb-1">Min</label>
                    <input
                      type="number"
                      value={minCitations ?? ''}
                      min={0}
                      onChange={(e) => { setMinCitations(e.target.value ? Number(e.target.value) : null); handleChange(); }}
                      className="w-20 px-2 py-1 text-sm rounded border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700"
                    />
                  </div>
                  <div>
                    <label className="block text-[10px] text-apple-gray-500 mb-1">Max</label>
                    <input
                      type="number"
                      value={maxCitations ?? ''}
                      min={0}
                      onChange={(e) => { setMaxCitations(e.target.value ? Number(e.target.value) : null); handleChange(); }}
                      className="w-20 px-2 py-1 text-sm rounded border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Notes */}
            <div>
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                Noter
              </label>
              <textarea
                value={notes}
                onChange={(e) => { setNotes(e.target.value); handleChange(); }}
                className="w-full px-3 py-2 text-sm bg-white dark:bg-apple-gray-700 border border-apple-gray-200 dark:border-apple-gray-500 rounded-lg focus:outline-none focus:ring-2 focus:ring-apple-blue/50 resize-none"
                rows={2}
                placeholder="Interne noter om testen..."
              />
            </div>
          </div>

          {/* Validation Result Panel */}
          {validationResult && (
            <div ref={validationResultRef} className="border-t border-apple-gray-100 dark:border-apple-gray-500 pt-4" role="region" aria-live="polite">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-apple-gray-700 dark:text-white flex items-center gap-2">
                  <span className={validationResult.passed ? 'text-green-500' : 'text-red-500'}>
                    {validationResult.passed ? '\u2713' : '\u2717'}
                  </span>
                  Valideringsresultat
                </h3>
                <div className="flex items-center gap-2 text-xs text-apple-gray-400">
                  <span>{Math.round(validationResult.duration_ms)}ms</span>
                  <button
                    onClick={() => setValidationResult(null)}
                    className="p-1 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 rounded"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              </div>

              {validationResult.error && (
                <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg mb-3">
                  <p className="text-sm text-red-600 dark:text-red-400">{validationResult.error}</p>
                </div>
              )}

              {Object.keys(validationResult.scores).length > 0 && (
                <div className="mb-4">
                  <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">Scorer</p>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(validationResult.scores).map(([name, score]) => (
                      <div
                        key={name}
                        className={`p-2 rounded-lg cursor-help ${score.passed ? 'bg-green-50 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/20'}`}
                        title={CROSS_LAW_TEST_DESCRIPTIONS[name as CrossLawScoreType] || name}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-apple-gray-700 dark:text-white">{TEST_TYPE_TAG_LABELS[name] || name}</span>
                          <span className={`text-xs ${score.passed ? 'text-green-600' : 'text-red-600'}`}>
                            {typeof score.score === 'number' ? `${Math.round(score.score * 100)}%` : score.passed ? '\u2713' : '\u2717'}
                          </span>
                        </div>
                        {score.message && (
                          <p className="text-[10px] text-apple-gray-500 dark:text-apple-gray-400 mt-1 line-clamp-2">{score.message}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {validationResult.answer && (
                <div className="mb-4">
                  <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">Genereret svar</p>
                  <div className="p-3 bg-apple-gray-50 dark:bg-apple-gray-700 rounded-lg prose prose-sm dark:prose-invert max-w-none prose-headings:text-sm prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-1 prose-p:my-1 prose-ul:my-1 prose-li:my-0">
                    <ReactMarkdown>{validationResult.answer}</ReactMarkdown>
                  </div>
                </div>
              )}

              {validationResult.references.length > 0 && (
                <div>
                  <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                    Kilder ({validationResult.references.length})
                  </p>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {validationResult.references.map((ref, i) => (
                      <div key={i} className="p-2 bg-apple-gray-50 dark:bg-apple-gray-700 rounded-lg">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-[10px] font-medium text-apple-blue">[{ref.idx}]</span>
                          {ref.corpus_id && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-apple-gray-100 dark:bg-apple-gray-500 text-apple-gray-500 dark:text-apple-gray-300 font-medium">
                              {ref.corpus_id}
                            </span>
                          )}
                          <span className="text-xs text-apple-gray-600 dark:text-apple-gray-300">
                            {ref.display || [ref.article && `Art. ${ref.article}`, ref.annex && `Bilag ${ref.annex}`, ref.recital && `Betr. ${ref.recital}`].filter(Boolean).join(', ') || 'Ukendt'}
                          </span>
                        </div>
                        {ref.chunk_text && (
                          <p className="text-[10px] text-apple-gray-500 dark:text-apple-gray-400 line-clamp-3">{ref.chunk_text}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-apple-gray-100 dark:border-apple-gray-500 bg-apple-gray-50 dark:bg-apple-gray-700">
          {/* Left: Validate split-button */}
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="flex">
                <button
                  type="button"
                  onClick={handleValidate}
                  disabled={isValidating || isSaving}
                  className="px-3 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 bg-white dark:bg-apple-gray-600 border border-apple-gray-200 dark:border-apple-gray-500 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500 rounded-l-lg transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  {isValidating ? (
                    <div className="w-4 h-4 border-2 border-apple-gray-400 border-t-transparent rounded-full animate-spin" />
                  ) : (
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  )}
                  Valider
                </button>
                <button
                  type="button"
                  onClick={() => setShowModeDropdown(!showModeDropdown)}
                  disabled={isValidating || isSaving}
                  className="px-2 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 bg-white dark:bg-apple-gray-600 border border-l-0 border-apple-gray-200 dark:border-apple-gray-500 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500 rounded-r-lg transition-colors disabled:opacity-50"
                >
                  <span className="text-[10px] px-1 py-0.5 bg-apple-gray-100 dark:bg-apple-gray-500 rounded">
                    {MODE_LABELS[validationMode].short}
                  </span>
                  <svg className="w-3 h-3 ml-1 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </div>
              {showModeDropdown && (
                <div className="absolute bottom-full left-0 mb-1 bg-white dark:bg-apple-gray-600 border border-apple-gray-200 dark:border-apple-gray-500 rounded-lg shadow-lg z-10 overflow-hidden">
                  {(Object.keys(MODE_LABELS) as RunMode[]).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      onClick={() => { setValidationMode(mode); setShowModeDropdown(false); }}
                      className={`w-full px-3 py-2 text-left text-xs hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500 transition-colors ${
                        validationMode === mode ? 'bg-apple-blue/10 text-apple-blue' : 'text-apple-gray-600 dark:text-apple-gray-300'
                      }`}
                    >
                      {MODE_LABELS[mode].long}
                    </button>
                  ))}
                </div>
              )}
            </div>
            {error && (
              <p className="text-sm text-red-500">{error}</p>
            )}
            {!error && (
              <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                {isEditing ? '\u00c6ndringer s\u00e6tter origin til "manual"' : 'Nye cases oprettes som "manual"'}
              </p>
            )}
          </div>

          {/* Right: Cancel + Save */}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={handleClose}
              className="px-4 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 rounded-lg transition-colors"
            >
              Annuller
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={isSaving}
              className="px-4 py-2 text-sm font-medium text-white bg-apple-blue hover:bg-apple-blue/90 rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {isSaving && (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              )}
              {isEditing ? 'Gem \u00e6ndringer' : 'Opret case'}
            </button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>,
    document.body
  );
}

// Edit suite modal
function EditSuiteModal({
  suiteId,
  existingSuite,
  corporaList,
  onClose,
  onUpdated,
}: {
  suiteId: string;
  existingSuite?: CrossLawSuiteStats;
  corporaList: CorpusInfo[];
  onClose: () => void;
  onUpdated: () => void;
}) {
  const [name, setName] = useState(existingSuite?.name || '');
  const [description, setDescription] = useState('');
  const [selectedCorpora, setSelectedCorpora] = useState<string[]>([]);
  const [synthesisMode, setSynthesisMode] = useState('comparison');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isDeleting, setIsDeleting] = useState(false);

  // Load full suite details
  useEffect(() => {
    const loadSuiteDetails = async () => {
      try {
        const resp = await fetch(`${API_BASE}/eval/cross-law/suites/${suiteId}`);
        if (!resp.ok) throw new Error('Failed to load suite');
        const data = await resp.json();
        setName(data.name || '');
        setDescription(data.description || '');
        setSelectedCorpora(data.target_corpora || []);
        setSynthesisMode(data.default_synthesis_mode || 'comparison');
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load suite');
      } finally {
        setIsLoading(false);
      }
    };

    loadSuiteDetails();
  }, [suiteId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || selectedCorpora.length < 2) return;

    try {
      const resp = await fetch(`${API_BASE}/eval/cross-law/suites/${suiteId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: name.trim(),
          description: description.trim(),
          target_corpora: selectedCorpora,
          default_synthesis_mode: synthesisMode,
        }),
      });

      if (!resp.ok) {
        const data = await resp.json();
        throw new Error(data.detail || 'Failed to update suite');
      }

      onUpdated();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to update');
    }
  };

  const handleDelete = async () => {
    if (!confirm('Er du sikker på at du vil slette denne suite?')) return;

    setIsDeleting(true);
    try {
      const resp = await fetch(`${API_BASE}/eval/cross-law/suites/${suiteId}`, {
        method: 'DELETE',
      });

      if (!resp.ok) {
        const data = await resp.json();
        throw new Error(data.detail || 'Failed to delete suite');
      }

      onUpdated();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete');
      setIsDeleting(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className="bg-white dark:bg-apple-gray-700 rounded-2xl p-6 w-full max-w-lg shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold text-apple-gray-700 dark:text-white mb-4">
          Rediger eval suite
        </h2>

        {error && (
          <div className="mb-4 px-3 py-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded text-sm text-red-600 dark:text-red-400">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin w-6 h-6 border-2 border-apple-blue border-t-transparent rounded-full" />
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-1">
                Navn
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-white dark:bg-apple-gray-600 border border-apple-gray-200 dark:border-apple-gray-500 rounded-lg focus:outline-none focus:ring-2 focus:ring-apple-blue/50"
                placeholder="f.eks. Incident Notification Tests"
                autoFocus
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-1">
                Beskrivelse (valgfri)
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="w-full px-3 py-2 text-sm bg-white dark:bg-apple-gray-600 border border-apple-gray-200 dark:border-apple-gray-500 rounded-lg focus:outline-none focus:ring-2 focus:ring-apple-blue/50 resize-none"
                rows={2}
                placeholder="Beskrivelse af hvad suiten tester..."
              />
            </div>

            {/* Synthesis mode */}
            <div>
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                Synthesis Mode
              </label>
              <div className="inline-flex rounded-lg bg-apple-gray-100 dark:bg-apple-gray-600 p-0.5" role="radiogroup" aria-label="Synthesis mode">
                {SYNTHESIS_MODES.map((mode) => (
                  <Tooltip key={mode.value} content={mode.tooltip} position="bottom" delay={300}>
                    <button
                      type="button"
                      role="radio"
                      aria-checked={synthesisMode === mode.value}
                      onClick={() => setSynthesisMode(mode.value)}
                      className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150 ${
                        synthesisMode === mode.value
                          ? 'bg-white dark:bg-apple-gray-500 text-apple-gray-700 dark:text-white shadow-sm'
                          : 'text-apple-gray-400 dark:text-apple-gray-300 hover:text-apple-gray-600 dark:hover:text-white'
                      }`}
                    >
                      {mode.label}
                    </button>
                  </Tooltip>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                Target love (mindst 2)
              </label>
              <div className="flex flex-col" style={{ height: '250px' }}>
                <LawSelectorPanel
                  corpusScope="explicit"
                  corpora={corporaList}
                  targetCorpora={selectedCorpora}
                  onTargetCorporaChange={setSelectedCorpora}
                />
              </div>
            </div>

            <div className="flex justify-between pt-2">
              <button
                type="button"
                onClick={handleDelete}
                disabled={isDeleting}
                className="px-4 py-2 text-sm font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors disabled:opacity-50"
              >
                {isDeleting ? 'Sletter...' : 'Slet suite'}
              </button>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 rounded-lg transition-colors"
                >
                  Annuller
                </button>
                <button
                  type="submit"
                  disabled={!name.trim() || selectedCorpora.length < 2}
                  className="px-4 py-2 text-sm font-medium bg-apple-blue hover:bg-apple-blue-dark text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Gem
                </button>
              </div>
            </div>
          </form>
        )}
      </motion.div>
    </motion.div>
  );
}

// Generate cases modal (C5: two-step generation flow)
function GenerateCasesModal({
  existingSuiteId,
  existingSuite,
  corporaList,
  onClose,
  onGenerated,
}: {
  existingSuiteId: string | null;
  existingSuite: CrossLawSuiteDetail | null;
  corporaList: CorpusInfo[];
  onClose: () => void;
  onGenerated: (suiteId: string) => void;
}) {
  // When adding to existing suite: skip Step 1, pre-fill from suite
  const isAddingToSuite = !!(existingSuiteId && existingSuite);
  const [step, setStep] = useState<1 | 2>(isAddingToSuite ? 2 : 1);
  const [selectedCorpora, setSelectedCorpora] = useState<string[]>(
    isAddingToSuite ? existingSuite.target_corpora : []
  );
  const [synthesisMode, setSynthesisMode] = useState<string>(
    isAddingToSuite ? existingSuite.default_synthesis_mode : 'comparison'
  );
  const [suiteName, setSuiteName] = useState('');
  const [suiteDescription, setSuiteDescription] = useState('');
  const [maxCases, setMaxCases] = useState(15);
  const [generationStrategy, setGenerationStrategy] = useState<'standard' | 'inverted'>('standard');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSuggesting, setIsSuggesting] = useState(false);
  const [generationResult, setGenerationResult] = useState<{ suite_id: string; case_count: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Build lookup from corporaList prop
  const corporaById = useMemo(() => {
    const map: Record<string, CorpusInfo> = {};
    for (const c of corporaList) map[c.id] = c;
    return map;
  }, [corporaList]);

  const corpusLabel = (id: string) => corporaById[id]?.name || id;

  const handleAiSuggest = async (type: 'name' | 'description') => {
    if (selectedCorpora.length < 2) return;
    setIsSuggesting(true);
    try {
      const resp = await fetch(`${API_BASE}/eval/cross-law/ai-suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type,
          corpora: selectedCorpora,
          corpora_names: selectedCorpora.map(c => corporaById[c]?.fullname || corpusLabel(c)),
          synthesis_mode: synthesisMode,
        }),
      });
      if (!resp.ok) throw new Error('AI suggest failed');
      const data = await resp.json();
      if (type === 'name') {
        setSuiteName(data.suggestion);
      } else {
        setSuiteDescription(data.suggestion);
      }
    } catch {
      // Silently fail — user can type manually
    } finally {
      setIsSuggesting(false);
    }
  };

  const handleGenerate = async () => {
    if (selectedCorpora.length < 2) return;
    setIsGenerating(true);
    setError(null);

    try {
      const body: Record<string, unknown> = {
        target_corpora: selectedCorpora,
        synthesis_mode: synthesisMode,
        max_cases: maxCases,
        generation_strategy: generationStrategy,
      };

      if (existingSuiteId) {
        body.suite_id = existingSuiteId;
      } else {
        body.suite_name = suiteName || `Cross-Law ${selectedCorpora.map((c) => corpusLabel(c)).join(' & ')}`;
        body.suite_description = suiteDescription;
      }

      const resp = await fetch(`${API_BASE}/eval/cross-law/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || 'Generation failed');
      }

      const data = await resp.json();
      setGenerationResult({ suite_id: data.suite_id, case_count: data.case_count });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  // Success state
  if (generationResult) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="bg-white dark:bg-apple-gray-700 rounded-2xl shadow-xl w-full max-w-sm mx-4 overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Apple-style alert: title, message, actions */}
          <div className="px-6 pt-6 pb-4 text-center">
            <h3 className="text-base font-semibold text-apple-gray-700 dark:text-white mb-1">
              {generationResult.case_count} cases genereret
            </h3>
            <p className="text-sm text-apple-gray-400 dark:text-apple-gray-300">
              Auto-genererede test cases er klar til gennemgang.
            </p>
            <p className="text-xs text-amber-600 dark:text-amber-400 mt-2">
              Cases bør gennemgås manuelt for nøjagtighed.
            </p>
          </div>
          <div className="border-t border-apple-gray-100 dark:border-apple-gray-600 flex">
            <button
              onClick={onClose}
              className="flex-1 py-3 text-sm text-apple-blue hover:bg-apple-gray-50 dark:hover:bg-apple-gray-600 transition-colors border-r border-apple-gray-100 dark:border-apple-gray-600"
            >
              Luk
            </button>
            <button
              onClick={() => onGenerated(generationResult.suite_id)}
              className="flex-1 py-3 text-sm font-semibold text-apple-blue hover:bg-apple-gray-50 dark:hover:bg-apple-gray-600 transition-colors"
            >
              Vis suite
            </button>
          </div>
        </motion.div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.95, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.95, opacity: 0 }}
        className={`bg-white dark:bg-apple-gray-700 rounded-2xl shadow-xl w-full mx-4 overflow-hidden flex flex-col max-h-[85vh] ${isAddingToSuite ? 'max-w-lg' : 'max-w-2xl'}`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-apple-gray-100 dark:border-apple-gray-600 flex-shrink-0">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-apple-gray-700 dark:text-white">
              {isAddingToSuite
                ? `Tilføj cases til ${existingSuite.name}`
                : 'Auto-generer eval suite'}
            </h2>
            {!isAddingToSuite && (
              <div className="flex items-center gap-2 text-xs text-apple-gray-400">
                <span className={step === 1 ? 'text-apple-blue font-medium' : ''}>1. Love</span>
                <span>{'>'}</span>
                <span className={step === 2 ? 'text-apple-blue font-medium' : ''}>2. Konfiguration</span>
              </div>
            )}
          </div>
        </div>

        {/* Step 1: Select laws */}
        {step === 1 && (
          <div className="p-6 flex flex-col flex-1 min-h-0">
            <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400 mb-3">
              Vælg mindst 2 love til cross-law evaluering:
            </p>

            <LawSelectorPanel
              corpusScope="explicit"
              corpora={corporaList}
              targetCorpora={selectedCorpora}
              onTargetCorporaChange={setSelectedCorpora}
            />

            <div className="flex justify-end gap-3 mt-4 pt-4 border-t border-apple-gray-100 dark:border-apple-gray-600 flex-shrink-0">
              <button
                onClick={onClose}
                className="px-4 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 rounded-lg transition-colors"
              >
                Annuller
              </button>
              <button
                onClick={() => setStep(2)}
                disabled={selectedCorpora.length < 2}
                className="px-4 py-2 text-sm font-medium bg-apple-blue hover:bg-apple-blue-dark text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Fortsæt ({selectedCorpora.length} valgt)
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Configuration */}
        {step === 2 && (
          <div className="p-6 overflow-y-auto">
            {/* Synthesis mode — segmented control (Apple HIG) */}
            <div className="mb-4">
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                Syntese-modus
              </label>
              <div className="inline-flex rounded-lg bg-apple-gray-100 dark:bg-apple-gray-600 p-0.5" role="radiogroup" aria-label="Syntese-modus">
                {SYNTHESIS_MODES.map((mode) => (
                  <Tooltip key={mode.value} content={mode.tooltip} position="bottom" delay={300}>
                    <button
                      type="button"
                      role="radio"
                      aria-checked={synthesisMode === mode.value}
                      onClick={() => {
                        if (synthesisMode !== mode.value) {
                          setSynthesisMode(mode.value);
                          setSuiteName('');
                        }
                      }}
                      className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150 ${
                        synthesisMode === mode.value
                          ? 'bg-white dark:bg-apple-gray-500 text-apple-gray-700 dark:text-white shadow-sm'
                          : 'text-apple-gray-400 dark:text-apple-gray-300 hover:text-apple-gray-600 dark:hover:text-white'
                      }`}
                    >
                      {mode.label}
                    </button>
                  </Tooltip>
                ))}
              </div>
            </div>

            {/* Max cases */}
            <div className="mb-4">
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-1">
                Antal cases ({maxCases})
              </label>
              <input
                type="range"
                min={5}
                max={20}
                value={maxCases}
                onChange={(e) => setMaxCases(Number(e.target.value))}
                style={sliderFillStyle(maxCases, 5, 20)}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-apple-gray-400">
                <span>5</span>
                <span>20</span>
              </div>
            </div>

            {/* Generation strategy */}
            <div className="mb-4">
              <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                Genereringsstrategi
              </label>
              <div className="inline-flex rounded-lg bg-apple-gray-100 dark:bg-apple-gray-600 p-0.5" role="radiogroup" aria-label="Genereringsstrategi">
                <Tooltip content="Genererer spørgsmål fra emner — standard tilgang" position="bottom" delay={300}>
                  <button
                    type="button"
                    role="radio"
                    aria-checked={generationStrategy === 'standard'}
                    onClick={() => setGenerationStrategy('standard')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150 ${
                      generationStrategy === 'standard'
                        ? 'bg-white dark:bg-apple-gray-500 text-apple-gray-700 dark:text-white shadow-sm'
                        : 'text-apple-gray-400 dark:text-apple-gray-300 hover:text-apple-gray-600 dark:hover:text-white'
                    }`}
                  >
                    Standard
                  </button>
                </Tooltip>
                <Tooltip content="Starter fra specifikke artikler — giver pålidelige forankringer" position="bottom" delay={300}>
                  <button
                    type="button"
                    role="radio"
                    aria-checked={generationStrategy === 'inverted'}
                    onClick={() => setGenerationStrategy('inverted')}
                    className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all duration-150 ${
                      generationStrategy === 'inverted'
                        ? 'bg-white dark:bg-apple-gray-500 text-apple-gray-700 dark:text-white shadow-sm'
                        : 'text-apple-gray-400 dark:text-apple-gray-300 hover:text-apple-gray-600 dark:hover:text-white'
                    }`}
                  >
                    Inverteret
                  </button>
                </Tooltip>
              </div>
            </div>

            {/* Suite name + AI suggest (only for new suites) */}
            {!existingSuiteId && (
              <>
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300">
                      Suite-navn
                    </label>
                    <button
                      onClick={() => handleAiSuggest('name')}
                      disabled={isSuggesting}
                      className="text-xs text-apple-blue hover:text-apple-blue-dark disabled:opacity-50"
                    >
                      {isSuggesting ? 'Foreslår...' : 'AI forslag'}
                    </button>
                  </div>
                  <input
                    type="text"
                    value={suiteName}
                    onChange={(e) => setSuiteName(e.target.value)}
                    placeholder="Navn på eval suite"
                    className="w-full px-3 py-2 text-sm rounded-lg bg-apple-gray-50 dark:bg-apple-gray-600 border border-apple-gray-200 dark:border-apple-gray-500 text-apple-gray-700 dark:text-white placeholder:text-apple-gray-400"
                  />
                </div>

                <div className="mb-4">
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300">
                      Beskrivelse
                    </label>
                    <button
                      onClick={() => handleAiSuggest('description')}
                      disabled={isSuggesting}
                      className="text-xs text-apple-blue hover:text-apple-blue-dark disabled:opacity-50"
                    >
                      {isSuggesting ? 'Foreslår...' : 'AI forslag'}
                    </button>
                  </div>
                  <textarea
                    value={suiteDescription}
                    onChange={(e) => setSuiteDescription(e.target.value)}
                    placeholder="Kort beskrivelse (valgfri)"
                    rows={4}
                    className="w-full px-3 py-2 text-sm rounded-lg bg-apple-gray-50 dark:bg-apple-gray-600 border border-apple-gray-200 dark:border-apple-gray-500 text-apple-gray-700 dark:text-white placeholder:text-apple-gray-400 resize-none"
                  />
                </div>
              </>
            )}

            {/* Error */}
            {error && (
              <div className="mb-4 px-3 py-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-sm text-red-600 dark:text-red-400">
                {error}
              </div>
            )}

            {/* Selected laws summary */}
            <div className="mb-4 flex flex-wrap gap-1.5">
              {selectedCorpora.map((c) => (
                <span
                  key={c}
                  title={corporaById[c]?.fullname || c}
                  className="px-2 py-0.5 text-xs font-medium bg-apple-blue/10 text-apple-blue rounded-full"
                >
                  {corpusLabel(c)}
                </span>
              ))}
            </div>

            <div className="flex justify-between gap-3">
              {!isAddingToSuite ? (
                <button
                  onClick={() => setStep(1)}
                  className="px-4 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 rounded-lg transition-colors"
                >
                  Tilbage
                </button>
              ) : <div />}
              <div className="flex gap-3">
                <button
                  onClick={onClose}
                  className="px-4 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 rounded-lg transition-colors"
                >
                  Annuller
                </button>
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating || selectedCorpora.length < 2}
                  className="px-4 py-2 text-sm font-medium bg-apple-blue hover:bg-apple-blue-hover text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isGenerating ? (
                    <>
                      <svg className="inline-block w-3.5 h-3.5 mr-1.5 -mt-px animate-spin" viewBox="0 0 16 16" fill="none">
                        <circle cx="8" cy="8" r="6.5" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2.5" />
                        <path d="M14.5 8a6.5 6.5 0 0 0-6.5-6.5" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
                      </svg>
                      Genererer…
                    </>
                  ) : isAddingToSuite ? 'Tilføj cases' : 'Generer cases'}
                </button>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </motion.div>
  );
}
