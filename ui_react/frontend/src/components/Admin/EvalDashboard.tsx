/**
 * Eval Dashboard component for viewing evaluation status across laws.
 *
 * Provides:
 * 1. Matrix view of pass rates (laws × test types)
 * 2. Historical run selector
 * 3. Drill-down into individual cases
 * 4. Run trigger functionality
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import type {
  EvalOverviewResponse,
  EvalRunListResponse,
  EvalRunSummary,
  EvalRunDetailResponse,
  EvalTestType,
} from '../../types';
import { SearchInput } from '../Common/SearchInput';
import { SegmentedControl } from '../Common/SegmentedControl';
import { Tooltip } from '../Common/Tooltip';
import { EvalSuitePanel } from './EvalSuitePanel';
import { EvalTabBar, type EvalMatrixMode } from './EvalTabBar';
import { CrossLawPanel } from './CrossLawPanel';
import { MetricsPanel } from './MetricsPanel';
import { getCorpusFullname } from '../../services/export/pdf/data/corpusMetadata';
import type { RunningEvalState } from './AdminPage';
import {
  formatPassRate,
  formatDuration,
  formatTimestamp,
  EVAL_SCORER_LABELS as SCORER_LABELS,
  EVAL_SCORER_DESCRIPTIONS as SCORER_DESCRIPTIONS,
  EVAL_TEST_TYPE_LABELS as TEST_TYPE_LABELS,
  EVAL_TEST_TYPE_DESCRIPTIONS as TEST_TYPE_DESCRIPTIONS,
  RUN_MODE_LABELS,
  type RunMode,
} from './evalUtils';
import { RunModePopover } from './RunModePopover';
import { ColumnTooltip } from './ColumnTooltip';

const API_BASE = '/api';

// Props for EvalDashboard - running eval state is lifted to AdminPage for persistence
interface EvalDashboardProps {
  runningEval: RunningEvalState;
  onRunningEvalChange: (state: RunningEvalState) => void;
}

// Sort options for the laws table
type SortField = 'name' | 'pass_rate' | 'total_cases' | 'last_run' | EvalTestType;
type SortDirection = 'asc' | 'desc';

// Constants imported from ./evalUtils: TEST_TYPE_LABELS, TEST_TYPE_DESCRIPTIONS,
// SCORER_LABELS, SCORER_DESCRIPTIONS, RUN_MODE_LABELS, RunMode

// RunModePopover imported from ./RunModePopover

// ColumnTooltip imported from ./ColumnTooltip

interface TestDefinition {
  id: string;
  profile: string;
  prompt: string;
  test_types: string[];
  origin: string;
  expected: {
    must_include_any_of?: string[];
    must_include_all_of?: string[];
    must_not_include_any_of?: string[];
    behavior?: string;
  };
}

export function EvalDashboard({ runningEval, onRunningEvalChange }: EvalDashboardProps) {
  // Data state
  const [overview, setOverview] = useState<EvalOverviewResponse | null>(null);
  const [runs, setRuns] = useState<EvalRunSummary[]>([]);
  const [selectedRun, setSelectedRun] = useState<EvalRunDetailResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // UI state
  const [selectedLaw, setSelectedLaw] = useState<string | null>(null);
  const [expandedCase, setExpandedCase] = useState<string | null>(null);
  const [triggerMode, setTriggerMode] = useState<RunMode>('full');
  const [triggeringLaw, setTriggeringLaw] = useState<string | null>(null);
  const [openRunMenu, setOpenRunMenu] = useState<string | null>(null);
  const [runMenuPosition, setRunMenuPosition] = useState({ x: 0, y: 0 });
  const [triggerProgress, setTriggerProgress] = useState<string | null>(null);
  const [showDefinition, setShowDefinition] = useState<string | null>(null);
  const [testDefinition, setTestDefinition] = useState<TestDefinition | null>(null);
  const [loadingDefinition, setLoadingDefinition] = useState(false);

  // Running eval state
  interface RunningCase {
    case_id: string;
    case_num: number;
    total: number;
    prompt: string;
    passed?: boolean;
    duration_ms?: number;
    failure_reason?: string;
    scores?: Record<string, { passed: boolean; score: number; message?: string }>;
  }
  const [runningCases, setRunningCases] = useState<RunningCase[]>([]);
  const [runningStats, setRunningStats] = useState<{ passed: number; failed: number; total: number } | null>(null);
  const [currentStage, setCurrentStage] = useState<string | null>(null);

  // Track if we've done initial restore to prevent race condition
  const hasRestoredRef = useRef(false);
  // Track previous parent state to detect when it's cleared
  const prevParentLawRef = useRef<string | null>(runningEval.law);

  // Restore running state from parent on mount (when coming back to Eval tab)
  // This MUST run before sync effect to prevent clearing restored state
  useEffect(() => {
    if (runningEval.law && !triggeringLaw) {
      setTriggeringLaw(runningEval.law);
      setRunningStats(runningEval.stats);
      setCurrentStage(runningEval.stage);
    }
    hasRestoredRef.current = true;
  // Only run on mount
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Sync parent→child when parent clears (user clicked "Luk" in AdminPage)
  useEffect(() => {
    // Detect when parent was cleared: had a value, now null
    if (prevParentLawRef.current && !runningEval.law) {
      // Clear local state to match parent
      setTriggeringLaw(null);
      setTriggerProgress(null);
      setRunningStats(null);
      setCurrentStage(null);
      setRunningCases([]);
    }
    prevParentLawRef.current = runningEval.law;
  }, [runningEval.law]);

  // Sync local running state to parent (AdminPage) for persistence across tab switches
  // This is the "Apple-style" approach - parent holds state, child just notifies changes
  useEffect(() => {
    // Don't sync until initial restore is complete (prevents clearing restored state)
    if (!hasRestoredRef.current) return;

    if (triggeringLaw) {
      // Don't sync if we would overwrite parent's good data with nulls
      // This prevents race condition where restore hasn't fully propagated yet
      if (!runningStats && runningEval.stats) {
        return; // Parent has data, we don't - wait for our data to arrive
      }
      if (!currentStage && runningEval.stage) {
        return; // Parent has stage, we don't - wait
      }

      // Prefer overview display name, but fall back to existing parent value
      const lawDisplayName = overview?.laws.find(l => l.law === triggeringLaw)?.display_name
        || runningEval.lawDisplayName
        || null;
      onRunningEvalChange({
        law: triggeringLaw,
        lawDisplayName,
        stats: runningStats,
        stage: currentStage as RunningEvalState['stage'],
      });
    }
    // Note: We don't clear parent state here - that's done by AdminPage's clearRunningEval
    // when user clicks "Luk" button. This prevents race conditions on mount.
  }, [triggeringLaw, runningStats, currentStage, overview, onRunningEvalChange, runningEval.lawDisplayName, runningEval.stats, runningEval.stage]);

  // Filter and sort state
  const [searchQuery, setSearchQuery] = useState('');
  const [sortField, setSortField] = useState<SortField>('name');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [caseFilter, setCaseFilter] = useState<'all' | 'failed' | 'passed' | 'escalated'>('all');

  // Panel collapse states - for drill-down navigation
  // States: matrix-expanded (default) → runs-expanded (law selected) → details-expanded (run selected)
  const [isMatrixCollapsed, setIsMatrixCollapsed] = useState(false);
  const [isCrossLawDrilled, setIsCrossLawDrilled] = useState(false);
  const [isRunsCollapsed, setIsRunsCollapsed] = useState(false);

  // Suite editor state
  const [editingSuiteLaw, setEditingSuiteLaw] = useState<string | null>(null);

  // Cross-law vs single-law mode toggle
  const [matrixMode, setMatrixMode] = useState<EvalMatrixMode>('single');

  // Clear single-law state when switching away from single mode
  const handleMatrixModeChange = (mode: EvalMatrixMode) => {
    setMatrixMode(mode);
    if (mode === 'cross' || mode === 'metrics') {
      // Clear single-law panel state to prevent it from appearing in other modes
      setEditingSuiteLaw(null);
      setSelectedLaw(null);
      setSelectedRun(null);
      setIsMatrixCollapsed(false);
      setIsRunsCollapsed(false);
    }
  };

  // Fetch overview data
  const fetchOverview = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/eval/overview`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: EvalOverviewResponse = await response.json();
      setOverview(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Kunne ikke hente eval oversigt');
    }
  }, []);

  // Fetch runs list
  const fetchRuns = useCallback(async (law?: string) => {
    try {
      const params = new URLSearchParams();
      if (law) params.set('law', law);
      params.set('limit', '20');

      const response = await fetch(`${API_BASE}/eval/runs?${params}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: EvalRunListResponse = await response.json();
      setRuns(data.runs);
    } catch (err) {
      console.error('Failed to fetch runs:', err);
    }
  }, []);

  // Fetch run detail
  const fetchRunDetail = useCallback(async (runId: string) => {
    try {
      const response = await fetch(`${API_BASE}/eval/runs/${runId}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: EvalRunDetailResponse = await response.json();
      setSelectedRun(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Kunne ikke hente run detaljer');
    }
  }, []);

  // Check for running evals on backend (reconnect support)
  // Note: Parent (AdminPage) handles state persistence across tab switches
  const checkRunningEvals = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/eval/running`);
      if (!response.ok) return;
      const data = await response.json();
      const running = data.running || {};
      const runningLaws = Object.keys(running);

      if (runningLaws.length > 0) {
        // Backend has a running eval - sync to local state
        const firstRunning = running[runningLaws[0]];
        setTriggeringLaw(runningLaws[0]);
        setTriggerProgress(firstRunning.progress || 'Kører i baggrunden...');
        setRunningStats({
          passed: firstRunning.passed || 0,
          failed: firstRunning.failed || 0,
          total: firstRunning.total || 0,
        });
        setCurrentStage(firstRunning.stage || 'running');
      }
      // If no running eval on backend, don't clear - parent state is authoritative
    } catch {
      // Ignore - not critical
    }
  }, []);

  // Initial load
  useEffect(() => {
    const load = async () => {
      setIsLoading(true);
      await Promise.all([fetchOverview(), fetchRuns(), checkRunningEvals()]);
      setIsLoading(false);
    };
    load();
  }, [fetchOverview, fetchRuns, checkRunningEvals]);

  // Poll for running eval status when we think one is running
  useEffect(() => {
    if (!triggeringLaw) return;
    // Don't poll when showing completion - user will dismiss with "Luk" button
    if (currentStage === 'complete') return;

    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/eval/running`);
        if (!response.ok) return;
        const data = await response.json();
        const running = data.running || {};

        if (Object.keys(running).length === 0) {
          // Backend says no running eval - it must have completed while we were away
          // Check for latest run to get results
          try {
            const runsResponse = await fetch(`${API_BASE}/eval/runs?law=${triggeringLaw}&limit=1`);
            if (runsResponse.ok) {
              const runsData = await runsResponse.json();
              if (runsData.runs && runsData.runs.length > 0) {
                const latestRun = runsData.runs[0];
                // Update stats from completed run and set to complete
                setRunningStats({
                  passed: latestRun.passed,
                  failed: latestRun.total - latestRun.passed,
                  total: latestRun.total,
                });
                setCurrentStage('complete');
                setTriggerProgress(`Færdig: ${latestRun.passed}/${latestRun.total} bestået`);
                // Refresh data
                await Promise.all([fetchOverview(), fetchRuns()]);
                return;
              }
            }
          } catch {
            // Ignore - fall through to clear
          }
          // No completed run found - just clear state
          setCurrentStage(prev => {
            if (prev === 'complete') {
              return prev; // Keep complete state for user to dismiss
            }
            setTriggeringLaw(null);
            setTriggerProgress(null);
            setRunningStats(null);
            return null;
          });
          await Promise.all([fetchOverview(), fetchRuns()]);
        } else {
          // Update progress from first running eval
          const firstRunning = Object.values(running)[0] as {
            progress?: string;
            passed?: number;
            failed?: number;
            total?: number;
            stage?: string;
          };
          setTriggerProgress(firstRunning.progress || 'Kører...');
          if (firstRunning.total) {
            const newPassed = firstRunning.passed || 0;
            const newFailed = firstRunning.failed || 0;
            setRunningStats(prev => {
              if (!prev) {
                return { passed: newPassed, failed: newFailed, total: firstRunning.total! };
              }
              // Only update if polling shows more progress
              const currentProgress = prev.passed + prev.failed;
              const newProgress = newPassed + newFailed;
              if (newProgress >= currentProgress) {
                return { passed: newPassed, failed: newFailed, total: firstRunning.total! };
              }
              return prev;
            });
          }
        }
      } catch {
        // Ignore polling errors
      }
    };

    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [triggeringLaw, currentStage, fetchOverview, fetchRuns]);

  // Load runs when law selection changes
  useEffect(() => {
    // Clear previous run selection to avoid showing stale data
    setSelectedRun(null);
    setExpandedCase(null);
    if (selectedLaw) {
      fetchRuns(selectedLaw);
    }
  }, [selectedLaw, fetchRuns]);

  // Panel state transitions:
  // - Selecting a law → collapse matrix, expand runs
  // - Selecting a run → collapse runs, expand details
  // - Deselecting law → expand matrix, hide runs
  // - Deselecting run → expand runs, hide details
  useEffect(() => {
    if (selectedLaw || editingSuiteLaw) {
      setIsMatrixCollapsed(true);
    } else {
      setIsMatrixCollapsed(false);
      setIsRunsCollapsed(false);
    }
  }, [selectedLaw, editingSuiteLaw]);

  // When a run is selected, collapse the runs list
  useEffect(() => {
    if (selectedRun) {
      setIsRunsCollapsed(true);
    } else {
      setIsRunsCollapsed(false);
    }
  }, [selectedRun]);

  // Close run menu when clicking outside
  useEffect(() => {
    if (!openRunMenu) return;
    const handleClick = () => setOpenRunMenu(null);
    document.addEventListener('click', handleClick);
    return () => document.removeEventListener('click', handleClick);
  }, [openRunMenu]);

  // Fetch test definition
  const fetchTestDefinition = useCallback(async (law: string, caseId: string) => {
    setLoadingDefinition(true);
    try {
      const response = await fetch(`${API_BASE}/eval/definition/${law}/${caseId}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data: TestDefinition = await response.json();
      setTestDefinition(data);
      setShowDefinition(caseId);
    } catch (err) {
      console.error('Failed to fetch test definition:', err);
    } finally {
      setLoadingDefinition(false);
    }
  }, []);

  // Auto-load test definition when a case is expanded
  useEffect(() => {
    if (expandedCase && selectedRun && showDefinition !== expandedCase) {
      fetchTestDefinition(selectedRun.law, expandedCase);
    }
  }, [expandedCase, selectedRun, showDefinition, fetchTestDefinition]);

  // Filter and sort laws
  const filteredAndSortedLaws = useMemo(() => {
    if (!overview) return [];

    let laws = [...overview.laws];

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      laws = laws.filter(
        (law) =>
          law.display_name.toLowerCase().includes(query) ||
          law.law.toLowerCase().includes(query)
      );
    }

    // Sort
    laws.sort((a, b) => {
      let comparison = 0;

      switch (sortField) {
        case 'name':
          comparison = a.display_name.localeCompare(b.display_name, 'da');
          break;
        case 'pass_rate':
          comparison = a.pass_rate - b.pass_rate;
          break;
        case 'total_cases':
          comparison = a.total_cases - b.total_cases;
          break;
        case 'last_run':
          const aDate = a.last_run ? new Date(a.last_run).getTime() : 0;
          const bDate = b.last_run ? new Date(b.last_run).getTime() : 0;
          comparison = aDate - bDate;
          break;
        default:
          // Sort by specific test type pass rate
          const aStats = a.by_test_type.find((s) => s.test_type === sortField);
          const bStats = b.by_test_type.find((s) => s.test_type === sortField);
          comparison = (aStats?.pass_rate || 0) - (bStats?.pass_rate || 0);
      }

      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return laws;
  }, [overview, searchQuery, sortField, sortDirection]);

  // Handle column header click for sorting
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Trigger eval run
  const triggerEval = async (law: string, mode?: RunMode) => {
    const runMode = mode ?? triggerMode;
    setTriggeringLaw(law);
    setTriggerProgress('Starter...');
    setRunningCases([]);
    setRunningStats(null);
    setCurrentStage('loading');

    try {
      const response = await fetch(`${API_BASE}/eval/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          law,
          run_mode: runMode,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      // Read SSE stream
      const reader = response.body?.getReader();
      if (!reader) throw new Error('Ingen stream');

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') {
              // Set to complete state FIRST to prevent polling from clearing state
              // (polling checks currentStage and won't clear if 'complete')
              setCurrentStage('complete');
              // Then refresh data - panel stays visible for user to see results
              await Promise.all([fetchOverview(), fetchRuns(law)]);
              return;
            }

            try {
              const event = JSON.parse(data);
              if (event.type === 'stage') {
                setCurrentStage(event.stage);
                setTriggerProgress(event.message);
              } else if (event.type === 'start') {
                setCurrentStage('running');
                setRunningStats({ passed: 0, failed: 0, total: event.total });
                setTriggerProgress(`Kører ${event.total} test cases...`);
              } else if (event.type === 'case_start') {
                setRunningCases(prev => [...prev, {
                  case_id: event.case_id,
                  case_num: event.case_num,
                  total: event.total,
                  prompt: event.prompt,
                }]);
                setTriggerProgress(`Test ${event.case_num}/${event.total}: ${event.prompt}`);
              } else if (event.type === 'case_result') {
                setRunningCases(prev => prev.map(c =>
                  c.case_id === event.case_id
                    ? { ...c, passed: event.passed, duration_ms: event.duration_ms, failure_reason: event.failure_reason, scores: event.scores }
                    : c
                ));
                setRunningStats({ passed: event.running_passed, failed: event.running_failed, total: event.total });
                const status = event.passed ? '✓' : '✗';
                setTriggerProgress(`${status} ${event.case_num}/${event.total} (${event.running_passed} bestået, ${event.running_failed} fejlet)`);
              } else if (event.type === 'phase') {
                // Only set escalating stage for Phase 2
                if (event.phase === 2) {
                  setCurrentStage('escalating');
                  setTriggerProgress(event.message || `Eskalerer fejlede cases...`);
                }
                // Phase 1 is informational - stay in current stage
              } else if (event.type === 'escalation_result') {
                // Update running stats after each escalation
                setRunningStats(prev => prev ? { ...prev, passed: event.running_passed, failed: event.running_failed } : prev);
                const status = event.passed ? '✓' : '✗';
                setTriggerProgress(`Eskalering: ${status} ${event.case_id}`);
              } else if (event.type === 'complete') {
                setCurrentStage('complete');
                setTriggerProgress(`Færdig: ${event.passed}/${event.total} bestået (${Math.round(event.pass_rate * 100)}%)`);
              } else if (event.type === 'error') {
                throw new Error(event.error);
              }
            } catch (parseErr) {
              // Ignore parse errors for malformed events
              if (parseErr instanceof SyntaxError) continue;
              throw parseErr;
            }
          }
        }
      }
    } catch (err) {
      // Don't clear state on AbortError - this happens when navigating away
      const isAbort = err instanceof DOMException && err.name === 'AbortError';
      if (!isAbort) {
        setError(err instanceof Error ? err.message : 'Eval kørsel fejlede');
        setTriggeringLaw(null);
        setTriggerProgress(null);
        setRunningCases([]);
        setRunningStats(null);
        setCurrentStage(null);
      }
    }
  };

  // formatPassRate, formatDuration, formatTimestamp imported from ./evalUtils

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-apple-blue border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">Henter eval data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 dark:bg-red-900/30 rounded-2xl">
        <p className="text-red-600 dark:text-red-400">{error}</p>
        <button
          onClick={() => {
            setError(null);
            fetchOverview();
          }}
          className="mt-2 text-sm text-red-600 dark:text-red-400 underline"
        >
          Prøv igen
        </button>
      </div>
    );
  }

  return (
    <div className={`flex flex-col gap-6 ${(isMatrixCollapsed && (selectedLaw || editingSuiteLaw)) || isCrossLawDrilled ? 'h-full' : ''} ${triggeringLaw ? 'pb-20' : ''}`}>
      {/* Header: Matrix mode toggle + Summary stats */}
      {overview && (
        <div className="flex items-center justify-between flex-shrink-0">
          <EvalTabBar mode={matrixMode} onModeChange={handleMatrixModeChange} />
          <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">
            {overview.total_cases} test cases · {formatPassRate(overview.overall_pass_rate)} overall
          </p>
        </div>
      )}

      {/* Cross-Law Panel - shown when mode is 'cross' */}
      {matrixMode === 'cross' && (
        <CrossLawPanel onDrillChange={setIsCrossLawDrilled} />
      )}

      {/* Metrics Panel - shown when mode is 'metrics' */}
      {matrixMode === 'metrics' && (
        <MetricsPanel />
      )}

      {/* Matrix view - Laws × Test Types (Single-Law mode) */}
      {matrixMode === 'single' && overview && overview.laws.length > 0 && (
        <div className="bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-hidden">
          {/* Collapsed header - clickable to navigate back */}
          {isMatrixCollapsed ? (
            <button
              onClick={() => {
                // Transition logic: if details open → show runs, if runs open → show matrix
                // Also close suite editor if open
                if (editingSuiteLaw) {
                  setEditingSuiteLaw(null);
                } else if (selectedRun) {
                  // Clear run selection to show runs list
                  setSelectedRun(null);
                  setIsRunsCollapsed(false);
                } else if (selectedLaw) {
                  // Clear law selection to show matrix
                  setSelectedLaw(null);
                  setIsMatrixCollapsed(false);
                } else {
                  setIsMatrixCollapsed(false);
                }
              }}
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
                  {overview.laws.length} {overview.laws.length === 1 ? 'lov' : 'love'} · {formatPassRate(overview.overall_pass_rate)} overall
                </span>
              </div>
              <svg className="w-4 h-4 text-apple-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          ) : (
            <>
              {/* Search input */}
              <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-500">
                <div className="flex items-center gap-3">
                  <SearchInput
                    value={searchQuery}
                    onChange={setSearchQuery}
                    placeholder="Søg efter lovgivning..."
                    className="flex-1"
                  />
                  {/* Collapse button */}
                  {selectedLaw && (
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
            <table className="w-full min-w-[800px]">
              <thead className="sticky top-0 z-20 bg-apple-gray-50 dark:bg-apple-gray-700">
                <tr className="border-b border-apple-gray-100 dark:border-apple-gray-500 bg-apple-gray-50 dark:bg-apple-gray-700">
                  <th
                    className="text-left font-medium text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide px-4 py-3 min-w-[280px] cursor-pointer hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 transition-colors"
                    onClick={() => handleSort('name')}
                  >
                    <div className="flex items-center gap-1">
                      Lov
                      {sortField === 'name' && (
                        <span className="text-apple-blue">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                      )}
                    </div>
                  </th>
                  <th
                    className="text-center font-medium text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide px-2 py-3 w-24 cursor-pointer hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 transition-colors"
                    onClick={() => handleSort('pass_rate')}
                  >
                    <div className="flex items-center justify-center gap-1">
                      Total
                      {sortField === 'pass_rate' && (
                        <span className="text-apple-blue">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                      )}
                    </div>
                  </th>
                  {overview.test_types.map((tt) => (
                    <ColumnTooltip
                      key={tt}
                      label={TEST_TYPE_LABELS[tt]}
                      description={TEST_TYPE_DESCRIPTIONS[tt]}
                      sortField={sortField}
                      field={tt}
                      sortDirection={sortDirection}
                      onSort={handleSort}
                    />
                  ))}
                  <th
                    className="text-center font-medium text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide px-2 py-3 whitespace-nowrap cursor-pointer hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 transition-colors"
                    onClick={() => handleSort('last_run')}
                  >
                    <div className="flex items-center justify-center gap-1">
                      Sidst kørt
                      {sortField === 'last_run' && (
                        <span className="text-apple-blue">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                      )}
                    </div>
                  </th>
                  <th className="w-14"></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-apple-gray-100 dark:divide-apple-gray-500">
                {filteredAndSortedLaws.map((law) => (
                  <tr
                    key={law.law}
                    className={`hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500/50 transition-colors ${
                      selectedLaw === law.law ? 'bg-apple-blue/5 dark:bg-apple-blue/10' : ''
                    }`}
                  >
                    <td className="px-4 py-3">
                      <button
                        onClick={() => setSelectedLaw(selectedLaw === law.law ? null : law.law)}
                        className="text-left"
                      >
                        <Tooltip content={getCorpusFullname(law.law)} maxWidth={450}>
                          <span className="text-sm font-medium text-apple-gray-700 dark:text-white cursor-help">
                            {law.display_name}
                          </span>
                        </Tooltip>
                        <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 ml-2">
                          ({law.law})
                        </span>
                      </button>
                    </td>
                    <td className="px-2 py-3 text-center whitespace-nowrap">
                      <span className="text-sm font-medium">
                        {formatPassRate(law.pass_rate)}
                      </span>
                      <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 ml-1">
                        {law.passed}/{law.total_cases}
                      </span>
                    </td>
                    {overview.test_types.map((tt) => {
                      const stats = law.by_test_type.find((s) => s.test_type === tt);
                      if (!stats || stats.total === 0) {
                        return (
                          <td key={tt} className="px-2 py-3 text-center whitespace-nowrap">
                            <span className="text-xs text-apple-gray-300 dark:text-apple-gray-600">—</span>
                          </td>
                        );
                      }
                      return (
                        <td key={tt} className="px-2 py-3 text-center whitespace-nowrap">
                          <span className="text-sm">{formatPassRate(stats.pass_rate)}</span>
                          <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 ml-1">
                            ({stats.total})
                          </span>
                        </td>
                      );
                    })}
                    <td className="px-2 py-3 text-center whitespace-nowrap">
                      {law.last_run ? (
                        <div className="flex items-center justify-center gap-1.5">
                          <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                            {formatTimestamp(law.last_run)}
                          </span>
                          {law.last_run_mode && (
                            <span className={`px-1 py-0.5 text-[10px] font-medium rounded ${
                              law.last_run_mode === 'retrieval_only'
                                ? 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                                : law.last_run_mode === 'full_with_judge'
                                ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                                : 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                            }`}>
                              {law.last_run_mode === 'retrieval_only' ? 'Ret.' : law.last_run_mode === 'full_with_judge' ? 'Full+J' : 'Full'}
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
                          onClick={(e) => { e.stopPropagation(); setEditingSuiteLaw(law.law); }}
                          className="p-1.5 text-apple-gray-500 hover:text-apple-gray-700 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded transition-colors"
                          title="Rediger eval suite"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                          </svg>
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            if (openRunMenu === law.law) {
                              setOpenRunMenu(null);
                            } else {
                              const rect = (e.currentTarget as HTMLButtonElement).getBoundingClientRect();
                              setRunMenuPosition({ x: rect.right, y: rect.bottom + 4 });
                              setOpenRunMenu(law.law);
                            }
                          }}
                          disabled={triggeringLaw === law.law}
                          className="p-1.5 text-apple-gray-500 hover:text-apple-gray-700 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded transition-colors disabled:opacity-50"
                          title="Kør eval"
                        >
                          {triggeringLaw === law.law ? (
                            <div className="w-4 h-4 border-2 border-apple-blue border-t-transparent rounded-full animate-spin" />
                          ) : (
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                            </svg>
                          )}
                        </button>
                        <RunModePopover
                          isOpen={openRunMenu === law.law}
                          position={runMenuPosition}
                          onClose={() => setOpenRunMenu(null)}
                          onSelect={(mode) => { setTriggerMode(mode); triggerEval(law.law, mode); }}
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
                  <td className="px-3 py-3 text-center">
                    <span className="text-sm font-semibold">
                      {formatPassRate(overview.overall_pass_rate)}
                    </span>
                    <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 ml-1">
                      {overview.laws.reduce((sum, l) => sum + l.passed, 0)}/{overview.total_cases}
                    </span>
                  </td>
                  {overview.test_types.map((tt) => {
                    const totalForType = overview.laws.reduce((sum, l) => {
                      const stats = l.by_test_type.find((s) => s.test_type === tt);
                      return sum + (stats?.total || 0);
                    }, 0);
                    const passedForType = overview.laws.reduce((sum, l) => {
                      const stats = l.by_test_type.find((s) => s.test_type === tt);
                      return sum + (stats?.passed || 0);
                    }, 0);
                    const passRate = totalForType > 0 ? passedForType / totalForType : 0;

                    return (
                      <td key={tt} className="px-3 py-3 text-center">
                        {totalForType > 0 ? (
                          <>
                            <span className="text-sm font-semibold">{formatPassRate(passRate)}</span>
                            <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 ml-1">
                              ({totalForType})
                            </span>
                          </>
                        ) : (
                          <span className="text-xs text-apple-gray-300 dark:text-apple-gray-600">—</span>
                        )}
                      </td>
                    );
                  })}
                  <td className="px-3 py-3"></td>
                  <td className="px-3 py-3"></td>
                </tr>
              </tfoot>
            </table>
          </div>
          </>
          )}
        </div>
      )}

      {/* Eval Suite Editor Panel */}
      <AnimatePresence>
        {editingSuiteLaw && overview && (
          <EvalSuitePanel
            law={editingSuiteLaw}
            displayName={overview.laws.find((l) => l.law === editingSuiteLaw)?.display_name || editingSuiteLaw}
            isOpen={!!editingSuiteLaw}
            onClose={() => setEditingSuiteLaw(null)}
            onCasesChanged={fetchOverview}
          />
        )}
      </AnimatePresence>


      {/* Drill-down: Selected law details (Single-Law mode only) */}
      <AnimatePresence>
        {matrixMode === 'single' && selectedLaw && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={`flex flex-col gap-4 ${isMatrixCollapsed ? 'flex-1 min-h-0' : ''}`}
          >
            {/* Historical runs - collapsible panel */}
            <div className={`bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-hidden ${!isRunsCollapsed ? 'flex-1 min-h-0 flex flex-col' : 'flex-shrink-0'}`}>
              {/* Collapsed state - clickable header to expand */}
              {isRunsCollapsed ? (
                <button
                  onClick={() => {
                    setSelectedRun(null);
                    setIsRunsCollapsed(false);
                  }}
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <svg className="w-4 h-4 text-apple-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-sm font-medium text-apple-gray-700 dark:text-white">
                      Kørselshistorik
                    </span>
                    <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                      {runs.length} {runs.length === 1 ? 'kørsel' : 'kørsler'}
                    </span>
                    {selectedRun && (
                      <span className="text-xs text-apple-blue">
                        {formatTimestamp(selectedRun.timestamp)} valgt
                      </span>
                    )}
                  </div>
                  <svg className="w-4 h-4 text-apple-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              ) : (
                <>
                  {/* Expanded header */}
                  <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-500 flex-shrink-0">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <button
                          onClick={() => setSelectedLaw(null)}
                          className="p-1.5 text-apple-gray-400 hover:text-apple-gray-600 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded-lg transition-colors"
                          title="Luk og vis matrix"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                        <span className="text-sm font-medium text-apple-gray-700 dark:text-white">
                          {overview?.laws.find((l) => l.law === selectedLaw)?.display_name || selectedLaw}
                        </span>
                        <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                          {runs.length} {runs.length === 1 ? 'kørsel' : 'kørsler'}
                        </span>
                      </div>
                      {/* Run controls - Apple style: subtle text buttons */}
                      <div className="flex items-center gap-1">
                        <select
                          value={triggerMode}
                          onChange={(e) => setTriggerMode(e.target.value as typeof triggerMode)}
                          className="px-2 py-1 text-sm rounded-md bg-transparent text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 transition-colors cursor-pointer"
                          disabled={triggeringLaw === selectedLaw}
                        >
                          <option value="retrieval_only">Retrieval</option>
                          <option value="full">Full</option>
                          <option value="full_with_judge">Full + Judge</option>
                        </select>
                        <button
                          onClick={() => triggerEval(selectedLaw!)}
                          disabled={triggeringLaw === selectedLaw}
                          className="px-3 py-1 text-sm font-medium text-apple-blue hover:bg-apple-blue/10 rounded-md transition-colors disabled:opacity-50"
                        >
                          {triggeringLaw === selectedLaw ? (
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
                  </div>

                  {/* Run history list - no max-height, fills available space */}
                  {runs.length > 0 ? (
                    <div className="flex-1 min-h-0 overflow-y-auto divide-y divide-apple-gray-100 dark:divide-apple-gray-500/50">
                      {runs.map((run, idx) => (
                        <button
                          key={run.run_id}
                          onClick={() => fetchRunDetail(run.run_id)}
                          className={`w-full flex items-center px-4 py-3 text-left transition-colors ${
                            selectedRun?.run_id === run.run_id
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
                            {run.run_mode === 'retrieval_only' ? 'Retrieval' : run.run_mode === 'full_with_judge' ? 'Full+Judge' : 'Full'}
                          </span>

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
                            selectedRun?.run_id === run.run_id
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
                        Klik "Kør" for at starte
                      </p>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Run detail - individual cases */}
            {selectedRun && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className={`bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 flex flex-col ${isMatrixCollapsed ? 'flex-1 min-h-0' : ''}`}
              >
                <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-500 flex-shrink-0">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div>
                        <h3 className="text-sm font-semibold text-apple-gray-700 dark:text-white">
                          Test Cases
                        </h3>
                        <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 mt-1">
                          {selectedRun.results.filter((r) => r.passed).length}/{selectedRun.results.length} bestået · {formatDuration(selectedRun.duration_seconds)}
                        </p>
                      </div>
                      {/* Case filter - moved next to summary */}
                      <SegmentedControl
                        options={[
                          { value: 'all', label: 'Alle' },
                          { value: 'failed', label: 'Fejlet', selectedColor: 'red' },
                          { value: 'passed', label: 'Bestået', selectedColor: 'green' },
                          { value: 'escalated', label: 'Eskaleret', selectedColor: 'amber' },
                        ]}
                        value={caseFilter}
                        onChange={setCaseFilter}
                        size="sm"
                      />
                    </div>
                    {/* Escalation stats - only show if there were escalations */}
                    {(() => {
                      const stats = selectedRun.escalation_stats as Record<string, number> | undefined;
                      if (!stats || typeof stats.cases_escalated !== 'number' || stats.cases_escalated === 0) return null;
                      return (
                        <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400">
                          <span title="Antal test cases der blev eskaleret til en stærkere model">
                            {stats.cases_escalated} eskaleret
                          </span>
                          {stats.cases_passed_on_escalation > 0 && (
                            <span className="text-green-600 dark:text-green-400" title="Bestod efter eskalering">
                              ({stats.cases_passed_on_escalation} bestod)
                            </span>
                          )}
                        </div>
                      );
                    })()}
                  </div>
                </div>

                <div className={`divide-y divide-apple-gray-100 dark:divide-apple-gray-500 overflow-y-auto ${isMatrixCollapsed ? 'flex-1' : 'max-h-96'}`}>
                  {selectedRun.results
                    .filter((r) => {
                      if (caseFilter === 'all') return true;
                      if (caseFilter === 'passed') return r.passed;
                      if (caseFilter === 'failed') return !r.passed;
                      if (caseFilter === 'escalated') return r.escalated === true;
                      return true;
                    })
                    .map((result) => (
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
                            {result.test_types.map((tt) => (
                              <span
                                key={tt}
                                title={TEST_TYPE_DESCRIPTIONS[tt]}
                                className="px-1.5 py-0.5 text-[10px] font-medium rounded bg-apple-gray-100 dark:bg-apple-gray-700 text-apple-gray-600 dark:text-apple-gray-300 cursor-help"
                              >
                                {TEST_TYPE_LABELS[tt]}
                              </span>
                            ))}
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
                                  {Object.entries(result.scores).map(([name, score]) => (
                                    <div
                                      key={name}
                                      className={`p-2 rounded-lg cursor-help ${score.passed ? 'bg-green-50 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/20'}`}
                                      title={SCORER_DESCRIPTIONS[name] || name}
                                    >
                                      <div className="flex items-center justify-between">
                                        <span className="text-xs font-medium text-apple-gray-700 dark:text-white">
                                          {SCORER_LABELS[name] || name}
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
                                  ))}
                                  {/* Escalation status - shown as a scorer box */}
                                  <div
                                    className={`p-2 rounded-lg cursor-help ${result.escalated ? 'bg-amber-50 dark:bg-amber-900/20' : 'bg-green-50 dark:bg-green-900/20'}`}
                                    title={SCORER_DESCRIPTIONS['escalation']}
                                  >
                                    <div className="flex items-center justify-between">
                                      <span className="text-xs font-medium text-apple-gray-700 dark:text-white">
                                        Escalation
                                      </span>
                                    </div>
                                    <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 mt-1 truncate">
                                      {result.escalated ? result.escalation_model || 'eskaleret' : 'ingen eskalering'}
                                    </p>
                                  </div>
                                </div>
                              </div>

                              {/* Test Definition inline - compact single row */}
                              <div className="space-y-2">
                                <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide">
                                  Testdefinition
                                </p>
                                {loadingDefinition && showDefinition === result.case_id ? (
                                  <div className="flex items-center gap-2 py-1">
                                    <div className="w-3 h-3 border-2 border-apple-blue border-t-transparent rounded-full animate-spin" />
                                    <span className="text-[10px] text-apple-gray-400">Henter...</span>
                                  </div>
                                ) : testDefinition && showDefinition === result.case_id ? (
                                  <div className="flex flex-wrap items-center gap-2">
                                    {/* Metadata badges */}
                                    <span
                                      className="px-1.5 py-0.5 text-[10px] rounded bg-apple-gray-200 dark:bg-apple-gray-600 text-apple-gray-600 dark:text-apple-gray-300 cursor-help"
                                      title={testDefinition.profile === 'LEGAL' ? 'Målgruppe: Jurister og compliance-ansvarlige' : 'Målgruppe: Udviklere og teknisk personale'}
                                    >
                                      {testDefinition.profile}
                                    </span>
                                    <span
                                      className={`px-1.5 py-0.5 text-[10px] rounded cursor-help ${testDefinition.origin === 'manual' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'}`}
                                      title={testDefinition.origin === 'manual' ? 'Manuelt oprettet test case med verificerede forventninger' : 'Automatisk genereret test case baseret på lovteksten'}
                                    >
                                      {testDefinition.origin === 'manual' ? 'Manuel' : 'Auto'}
                                    </span>
                                    {/* Separator */}
                                    <span className="text-apple-gray-300 dark:text-apple-gray-500">|</span>
                                    {/* Expected articles - inline */}
                                    {testDefinition.expected.must_include_any_of && testDefinition.expected.must_include_any_of.length > 0 && (
                                      <>
                                        <span className="text-[10px] text-green-600 dark:text-green-400" title="Svaret skal referere til mindst én af disse artikler">≥1:</span>
                                        {testDefinition.expected.must_include_any_of.map((item, i) => (
                                          <span key={`any-${i}`} className="px-1.5 py-0.5 text-[10px] rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
                                            {item}
                                          </span>
                                        ))}
                                      </>
                                    )}
                                    {testDefinition.expected.must_include_all_of && testDefinition.expected.must_include_all_of.length > 0 && (
                                      <>
                                        <span className="text-[10px] text-blue-600 dark:text-blue-400" title="Svaret skal referere til alle disse artikler">Alle:</span>
                                        {testDefinition.expected.must_include_all_of.map((item, i) => (
                                          <span key={`all-${i}`} className="px-1.5 py-0.5 text-[10px] rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">
                                            {item}
                                          </span>
                                        ))}
                                      </>
                                    )}
                                    {testDefinition.expected.must_not_include_any_of && testDefinition.expected.must_not_include_any_of.length > 0 && (
                                      <>
                                        <span className="text-[10px] text-red-600 dark:text-red-400" title="Svaret må ikke referere til disse artikler">Ikke:</span>
                                        {testDefinition.expected.must_not_include_any_of.map((item, i) => (
                                          <span key={`not-${i}`} className="px-1.5 py-0.5 text-[10px] rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400">
                                            {item}
                                          </span>
                                        ))}
                                      </>
                                    )}
                                    {testDefinition.expected.behavior && (
                                      <span
                                        className={`px-1.5 py-0.5 text-[10px] rounded ${
                                          testDefinition.expected.behavior === 'answer'
                                            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400'
                                            : 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400'
                                        }`}
                                        title={testDefinition.expected.behavior === 'answer' ? 'Systemet skal svare' : 'Systemet skal afstå'}
                                      >
                                        {testDefinition.expected.behavior === 'answer' ? 'Skal svare' : 'Skal afstå'}
                                      </span>
                                    )}
                                  </div>
                                ) : (
                                  <div className="flex items-center gap-2 py-1">
                                    <div className="w-3 h-3 border-2 border-apple-blue border-t-transparent rounded-full animate-spin" />
                                    <span className="text-[10px] text-apple-gray-400">Henter...</span>
                                  </div>
                                )}
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Empty state */}
      {overview && overview.laws.length === 0 && (
        <div className="text-center py-12">
          <svg className="w-12 h-12 mx-auto text-apple-gray-300 dark:text-apple-gray-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />
          </svg>
          <p className="text-apple-gray-500 dark:text-apple-gray-400">
            Ingen eval cases fundet. Tilføj lovgivning med eval generation for at komme i gang.
          </p>
        </div>
      )}

      {/* Running eval footer is rendered by AdminPage for persistence across tab switches */}
    </div>
  );
}
