/**
 * Admin page component for managing EU legislation.
 *
 * Provides two views:
 * 1. Browse EU legislation (list with search)
 * 2. Add new legislation (with progress tracking)
 */

import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { LegislationInfo, AddLawRequest, IngestionEvent, EvalRunMode, IngestionQuality } from '../../types';
import { EvalDashboard } from './EvalDashboard';
import { SearchInput } from '../Common/SearchInput';
import { SegmentedControl } from '../Common/SegmentedControl';
import { Tooltip } from '../Common/Tooltip';
import { getCorpusFullname } from '../../services/export/pdf/data/corpusMetadata';

type AdminTab = 'legislation' | 'eval';

const API_BASE = '/api';

interface AdminPageProps {
  onNavigateBack: () => void;
  onCorporaRefresh?: () => void;
}

interface EvalResult {
  case_id: string;
  question: string;
  answer: string;
  expected_articles: string[];
  actual_articles: string[];
  passed: boolean;
  notes: string;
}

interface EvalSummary {
  total: number;
  passed: number;
  failed: number;
}

interface PreflightWarning {
  category: string;
  message: string;
  location: string;
  severity: string;
  suggestion: string;
}

interface PreflightResult {
  handled: Record<string, number>;
  unhandled: Record<string, number>;
  warnings: PreflightWarning[];
}

interface IngestionState {
  isRunning: boolean;
  currentStage: string;
  progress: number;
  message: string;
  error: string | null;
  completedStages: string[];
  evalResults: EvalResult[];
  evalSummary: EvalSummary | null;
  preflight: PreflightResult | null;
}

// Running eval state - lifted from EvalDashboard for persistence across tab switches
export interface RunningEvalState {
  law: string | null;
  lawDisplayName: string | null;
  stats: { passed: number; failed: number; total: number } | null;
  stage: 'loading' | 'running' | 'escalating' | 'complete' | null;
}

const STAGE_LABELS: Record<string, string> = {
  download: 'Download HTML',
  chunking: 'Chunking + berigelse',
  indexing: 'Vector Store',
  citation_graph: 'Citation Graph',
  eval_generation: 'LLM-generering',
  eval_run: 'Pipeline-verifikation',
  config_update: 'Konfiguration',
};

// Stages that have meaningful granular progress (show bar)
// Other stages are essentially binary and show pulsing indicator
const STAGES_WITH_PROGRESS = new Set(['chunking', 'eval_generation', 'eval_run']);

// Current year for filter defaults
const CURRENT_YEAR = new Date().getFullYear();
const MAX_YEAR_SPAN = 15;
const ITEMS_PER_PAGE = 200;

// Generate year options (2010 to current year)
const YEAR_OPTIONS = Array.from({ length: CURRENT_YEAR - 2009 }, (_, i) => CURRENT_YEAR - i);

// Sortable columns
type SortField = 'celex_number' | 'title_da' | 'document_type' | 'entry_into_force' | 'last_modified';
type SortDirection = 'asc' | 'desc';

/**
 * Extract the short name (abbreviation) from a title.
 * Titles are in format "Full Name (Short)" - returns "Short".
 * Falls back to corpus_id if no parentheses found.
 */
function getShortName(title: string, corpusId: string | null): string {
  const match = title.match(/\(([^)]+)\)$/);
  return match ? match[1] : corpusId || '';
}

export function AdminPage({ onNavigateBack, onCorporaRefresh }: AdminPageProps) {
  // Tab state
  const [activeTab, setActiveTab] = useState<AdminTab>('legislation');

  // Running eval state - persists across tab switches (Apple-style system status)
  const [runningEval, setRunningEval] = useState<RunningEvalState>({
    law: null,
    lawDisplayName: null,
    stats: null,
    stage: null,
  });

  // Clear running eval state
  const clearRunningEval = useCallback(() => {
    setRunningEval({ law: null, lawDisplayName: null, stats: null, stage: null });
  }, []);

  const [searchTerm, setSearchTerm] = useState('');
  const [legislation, setLegislation] = useState<LegislationInfo[]>([]); // Full dataset from server
  const [isLoading, setIsLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false); // Track if user has initiated a search

  // Filter state - default to 5 years (reasonable default)
  const [yearFrom, setYearFrom] = useState(CURRENT_YEAR - 4);
  const [yearTo, setYearTo] = useState(CURRENT_YEAR);
  const [docType, setDocType] = useState<'regulation' | 'directive' | 'all'>('regulation');
  const [inForceOnly, setInForceOnly] = useState(true);
  const [ingestionFilter, setIngestionFilter] = useState<'all' | 'ingested' | 'not_ingested'>('all');

  // Sorting state
  const [sortField, setSortField] = useState<SortField>('entry_into_force');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);

  // Form state for adding new legislation
  const [selectedLegislation, setSelectedLegislation] = useState<LegislationInfo | null>(null);
  const [corpusId, setCorpusId] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [fullname, setFullname] = useState<string | null>(null);
  const [generateEval, setGenerateEval] = useState(true);
  const [evalRunMode, setEvalRunMode] = useState<EvalRunMode>('full');

  // Ingestion state
  const [ingestion, setIngestion] = useState<IngestionState>({
    isRunning: false,
    currentStage: '',
    progress: 0,
    message: '',
    error: null,
    completedStages: [],
    evalResults: [],
    evalSummary: null,
    preflight: null,
  });

  // Expanded eval result (for showing full answer)
  const [expandedEvalCase, setExpandedEvalCase] = useState<string | null>(null);

  // AI name suggestion state
  const [isSuggestingNames, setIsSuggestingNames] = useState(false);

  // Confirmation dialog state
  const [confirmDialog, setConfirmDialog] = useState<{
    isOpen: boolean;
    corpusId: string;
    displayName: string;
  }>({ isOpen: false, corpusId: '', displayName: '' });

  // Fetch legislation list with progress simulation
  // Note: Search is handled client-side, so we DON'T include searchTerm in dependencies
  const fetchLegislation = useCallback(async () => {
    setIsLoading(true);
    setLoadingProgress(0);
    setError(null);

    // Estimate ~60s for full query, simulate progress
    const yearSpan = yearTo - yearFrom + 1;
    const estimatedSeconds = Math.min(yearSpan * 4, 70); // ~4s per year, max 70s
    let progressInterval: NodeJS.Timeout | null = null;

    try {
      const params = new URLSearchParams();
      // Don't send search term - we filter client-side for instant results
      params.set('year_from', yearFrom.toString());
      params.set('year_to', yearTo.toString());
      params.set('date_filter', 'creation'); // Fixed to creation (CELEX year)
      params.set('doc_type', docType);
      params.set('in_force_only', inForceOnly.toString());

      // Start progress simulation
      const startTime = Date.now();
      progressInterval = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        const progress = Math.min(95, (elapsed / estimatedSeconds) * 100);
        setLoadingProgress(progress);
      }, 200);

      const response = await fetch(`${API_BASE}/admin/legislation?${params}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setLoadingProgress(100);
      setLegislation(data.legislation || []);
      setCurrentPage(1); // Reset to first page on new data
      setHasSearched(true); // Mark that user has searched
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Kunne ikke hente lovgivningsliste');
    } finally {
      if (progressInterval) clearInterval(progressInterval);
      setIsLoading(false);
    }
  }, [yearFrom, yearTo, docType, inForceOnly]); // Note: searchTerm NOT included - filter client-side

  // No auto-fetch - user must click "Søg" button to initiate search

  // Reset pagination when search changes (client-side filtering)
  useEffect(() => {
    setCurrentPage(1);
  }, [searchTerm]);

  // Handle adding new legislation
  const handleAddLegislation = async (leg: LegislationInfo) => {
    setSelectedLegislation(leg);
    // Generate default corpus ID from CELEX
    const defaultId = leg.celex_number.toLowerCase().replace(/[^a-z0-9]/g, '');
    setCorpusId(defaultId.slice(-8)); // Last 8 chars
    setDisplayName(leg.title_da || leg.title_en || leg.celex_number);
    // Use full title as default fullname (AI suggestion may override this)
    setFullname(leg.title_da || leg.title_en || null);
  };

  // Start ingestion
  const startIngestion = async () => {
    if (!selectedLegislation) return;

    setIngestion({
      isRunning: true,
      currentStage: '',
      progress: 0,
      message: 'Starter...',
      error: null,
      completedStages: [],
      evalResults: [],
      evalSummary: null,
      preflight: null,
    });

    try {
      const request: AddLawRequest = {
        celex_number: selectedLegislation.celex_number,
        corpus_id: corpusId,
        display_name: displayName,
        fullname: fullname,
        eurovoc_labels: selectedLegislation.eurovoc_labels || null,
        generate_eval: generateEval,
        entry_into_force: selectedLegislation.entry_into_force,
        last_modified: selectedLegislation.last_modified,
        eval_run_mode: generateEval ? evalRunMode : undefined,
      };

      const response = await fetch(`${API_BASE}/admin/add-law/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
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
              // Stream ended - complete event already handled state update
              return;
            }

            try {
              const event: IngestionEvent = JSON.parse(data);
              handleIngestionEvent(event);
            } catch {
              // Ignore parse errors
            }
          }
        }
      }
    } catch (err) {
      setIngestion((prev) => ({
        ...prev,
        isRunning: false,
        error: err instanceof Error ? err.message : 'Ukendt fejl',
      }));
    }
  };

  // Handle ingestion event
  const handleIngestionEvent = (event: IngestionEvent) => {
    switch (event.type) {
      case 'stage':
        setIngestion((prev) => ({
          ...prev,
          currentStage: event.stage,
          message: event.message,
          completedStages: event.completed
            ? [...prev.completedStages, event.stage]
            : prev.completedStages,
        }));
        break;
      case 'progress':
        setIngestion((prev) => ({
          ...prev,
          currentStage: event.stage,
          progress: event.progress_pct,
        }));
        break;
      case 'complete':
        setIngestion((prev) => ({
          ...prev,
          isRunning: false,
          completedStages: [...prev.completedStages, 'complete'],
        }));
        // Refresh legislation list in background
        fetchLegislation();
        // Refresh corpora data to update corpus registry (for tooltips)
        onCorporaRefresh?.();
        break;
      case 'error':
        setIngestion((prev) => ({
          ...prev,
          isRunning: false,
          error: event.error,
        }));
        break;
      case 'eval_result':
        setIngestion((prev) => ({
          ...prev,
          evalResults: [...prev.evalResults, {
            case_id: event.case_id,
            question: event.question,
            answer: event.answer,
            expected_articles: event.expected_articles,
            actual_articles: event.actual_articles,
            passed: event.passed,
            notes: event.notes,
          }],
        }));
        break;
      case 'eval_summary':
        setIngestion((prev) => ({
          ...prev,
          evalSummary: {
            total: event.total,
            passed: event.passed,
            failed: event.failed,
          },
        }));
        break;
      case 'preflight':
        setIngestion((prev) => ({
          ...prev,
          preflight: {
            handled: event.handled,
            unhandled: event.unhandled,
            warnings: event.warnings,
          },
        }));
        break;
    }
  };

  // Show confirmation dialog for removing a corpus
  const handleRemoveCorpus = (corpusId: string, displayName?: string) => {
    setConfirmDialog({
      isOpen: true,
      corpusId,
      displayName: displayName || corpusId,
    });
  };

  // Actually remove the corpus after confirmation
  const confirmRemoveCorpus = async () => {
    const { corpusId } = confirmDialog;
    setConfirmDialog({ isOpen: false, corpusId: '', displayName: '' });

    try {
      // GUARDRAIL: Include confirm parameter to prevent accidental cross-corpus deletion
      // The backend requires confirm=corpusId to proceed with deletion
      const response = await fetch(`${API_BASE}/admin/corpus/${corpusId}?confirm=${encodeURIComponent(corpusId)}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      // Refresh list
      fetchLegislation();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Kunne ikke fjerne corpus');
    }
  };

  // Close confirmation dialog
  const cancelRemoveCorpus = () => {
    setConfirmDialog({ isOpen: false, corpusId: '', displayName: '' });
  };

  // Cancel ingestion form
  const cancelIngestion = () => {
    setSelectedLegislation(null);
    setExpandedEvalCase(null);
    setIngestion({
      isRunning: false,
      currentStage: '',
      progress: 0,
      message: '',
      error: null,
      completedStages: [],
      evalResults: [],
      evalSummary: null,
      preflight: null,
    });
  };

  // AI name suggestion
  const suggestNames = async () => {
    if (!selectedLegislation) return;

    setIsSuggestingNames(true);
    try {
      const response = await fetch(`${API_BASE}/admin/suggest-names`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: selectedLegislation.title_da || selectedLegislation.title_en || '',
          celex_number: selectedLegislation.celex_number,
        }),
      });

      if (!response.ok) {
        throw new Error('Kunne ikke hente navneforslag');
      }

      const data = await response.json();
      if (data.corpus_id) setCorpusId(data.corpus_id);
      if (data.display_name) setDisplayName(data.display_name);
      if (data.fullname) setFullname(data.fullname);
    } catch (err) {
      // Silently fail - user can still enter manually
      console.error('Name suggestion failed:', err);
    } finally {
      setIsSuggestingNames(false);
    }
  };

  // Sort function
  const sortLegislation = (items: LegislationInfo[]) => {
    return [...items].sort((a, b) => {
      let aVal: string | null = null;
      let bVal: string | null = null;

      switch (sortField) {
        case 'celex_number':
          aVal = a.celex_number;
          bVal = b.celex_number;
          break;
        case 'title_da':
          aVal = a.title_da || a.title_en || '';
          bVal = b.title_da || b.title_en || '';
          break;
        case 'document_type':
          aVal = a.document_type;
          bVal = b.document_type;
          break;
        case 'entry_into_force':
          aVal = a.entry_into_force;
          bVal = b.entry_into_force;
          break;
        case 'last_modified':
          aVal = a.last_modified;
          bVal = b.last_modified;
          break;
      }

      // Handle nulls - push to end
      if (!aVal && !bVal) return 0;
      if (!aVal) return 1;
      if (!bVal) return -1;

      const cmp = aVal.localeCompare(bVal);
      return sortDirection === 'asc' ? cmp : -cmp;
    });
  };

  // Handle column header click
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
    setCurrentPage(1); // Reset to first page when sorting changes
  };

  // Client-side search filtering (instant, no server round-trip)
  // Matches both with and without spaces so "nis2" finds "NIS 2"
  const searchFiltered = searchTerm
    ? legislation.filter((l) => {
        const term = searchTerm.toLowerCase();
        const termNoSpaces = term.replace(/\s+/g, '');
        return (
          l.title_da?.toLowerCase().includes(term) ||
          l.title_en?.toLowerCase().includes(term) ||
          l.title_da?.toLowerCase().replace(/\s+/g, '').includes(termNoSpaces) ||
          l.title_en?.toLowerCase().replace(/\s+/g, '').includes(termNoSpaces) ||
          l.celex_number.toLowerCase().includes(term) ||
          l.eurovoc_labels?.some((label) => label.toLowerCase().includes(term))
        );
      })
    : legislation;

  // Separate ingested legislation for the top section
  const ingestedLegislation = sortLegislation(searchFiltered.filter((l) => l.is_ingested));

  // Filter legislation based on ingestion status filter
  const filteredByIngestion = searchFiltered.filter((l) => {
    if (ingestionFilter === 'all') return true;
    if (ingestionFilter === 'ingested') return l.is_ingested;
    return !l.is_ingested;
  });

  // Sort filtered legislation
  const sortedLegislation = sortLegislation(filteredByIngestion);

  // Paginate
  const totalPages = Math.ceil(sortedLegislation.length / ITEMS_PER_PAGE);
  const paginatedLegislation = sortedLegislation.slice(
    (currentPage - 1) * ITEMS_PER_PAGE,
    currentPage * ITEMS_PER_PAGE
  );

  // Count for stats
  const ingestedCount = legislation.filter((l) => l.is_ingested).length;
  const notIngestedCount = legislation.length - ingestedCount;

  // Sort indicator component
  const SortIndicator = ({ field }: { field: SortField }) => (
    <span className="ml-1 inline-flex">
      {sortField === field ? (
        sortDirection === 'asc' ? '↑' : '↓'
      ) : (
        <span className="text-apple-gray-300 dark:text-apple-gray-500">↕</span>
      )}
    </span>
  );

  // Format date, filtering out invalid placeholder dates from EUR-Lex
  const formatDate = (dateStr: string | null, showNotSet = false): string => {
    if (!dateStr) return showNotSet ? 'Ikke fastsat' : '—';
    const date = new Date(dateStr);
    const year = date.getFullYear();
    // EUR-Lex uses year 1001 as placeholder for unknown dates
    if (year < 1950) return showNotSet ? 'Ikke fastsat' : '—';
    // EUR-Lex uses far future dates as placeholders for undetermined entry_into_force
    // Filter out dates more than 1 year in the future (conservative - only show near-certain dates)
    const maxYear = CURRENT_YEAR + 1;
    if (year > maxYear) return 'Ikke fastsat';
    return date.toLocaleDateString('da-DK');
  };

  // Generate quality tooltip with explanation of consequences
  const getQualityTooltip = (quality: IngestionQuality): string => {
    const pct = 100 - quality.unhandled_pct;
    const patterns = quality.unhandled_count > 0
      ? Object.entries(quality.unhandled_patterns).map(([k, v]) => `  ${k}: ${v}`).join('\n')
      : '';

    if (quality.unhandled_count === 0) {
      return 'Alle HTML-elementer blev genkendt og indekseret.\n\nLoven er fuldt søgbar.';
    }

    if (quality.unhandled_pct <= 2) {
      return `${pct.toFixed(1)}% af HTML-elementer håndteret.\n\n${quality.unhandled_count} elementer sprunget over:\n${patterns}\n\nDisse er typisk billeder eller dekorative elementer.\nLoven fungerer fint til søgning.`;
    }

    if (quality.unhandled_pct <= 5) {
      return `${pct.toFixed(1)}% af HTML-elementer håndteret.\n\n${quality.unhandled_count} elementer sprunget over:\n${patterns}\n\nNogle tabeller eller figurer kan mangle.\nDe fleste spørgsmål kan stadig besvares.`;
    }

    return `${pct.toFixed(1)}% af HTML-elementer håndteret.\n\n${quality.unhandled_count} elementer sprunget over:\n${patterns}\n\nEn del indhold kan mangle i søgeresultater.\nOvervej at forbedre HTML-parseren for denne lov.`;
  };

  return (
    <div className="flex-1 flex flex-col bg-apple-gray-50 dark:bg-apple-gray-700 overflow-hidden">
      {/* Header */}
      <header className="px-6 py-4 bg-white dark:bg-apple-gray-600 border-b border-apple-gray-100 dark:border-apple-gray-500">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <button
              onClick={onNavigateBack}
              className="p-2 rounded-lg hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 transition-colors"
              aria-label="Tilbage"
            >
              <svg className="w-5 h-5 text-apple-gray-500 dark:text-apple-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
            </button>
            <h1 className="text-xl font-semibold text-apple-gray-700 dark:text-white">
              Admin
            </h1>
          </div>
        </div>

        {/* Tabs */}
        <SegmentedControl
          options={[
            { value: 'legislation', label: 'Lovgivning' },
            { value: 'eval', label: 'Eval Dashboard' },
          ]}
          value={activeTab}
          onChange={setActiveTab}
        />
      </header>

      {/* Main content - add bottom padding when eval footer is visible */}
      <div className="flex-1 overflow-auto px-6 pt-6 pb-6">
        {activeTab === 'eval' ? (
          <EvalDashboard
            runningEval={runningEval}
            onRunningEvalChange={setRunningEval}
          />
        ) : (
        <AnimatePresence mode="wait">
          {selectedLegislation ? (
            // Ingestion form/progress view
            <motion.div
              key="ingestion"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-4xl mx-auto"
            >
              <div className="bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 p-6">
                <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500 uppercase tracking-wide mb-1">
                  Tilføjer
                </p>
                <h2 className="text-base font-medium text-apple-gray-700 dark:text-white mb-6">
                  {displayName || corpusId || selectedLegislation.celex_number}
                </h2>

                {!ingestion.isRunning && !ingestion.error && ingestion.completedStages.length === 0 && (
                  <>
                    {/* Form fields */}
                    <div className="space-y-4 mb-6">
                      {/* Header with AI suggest button */}
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300">
                          Navngivning
                        </span>
                        <button
                          type="button"
                          onClick={suggestNames}
                          disabled={isSuggestingNames}
                          className="flex items-center gap-1.5 text-xs text-apple-blue hover:text-apple-blue/80 transition-colors disabled:opacity-50"
                        >
                          {isSuggestingNames ? (
                            <>
                              <div className="w-3 h-3 border-[1.5px] border-apple-blue border-t-transparent rounded-full animate-spin" />
                              <span>Foreslår...</span>
                            </>
                          ) : (
                            <>
                              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                              </svg>
                              <span>Foreslå med AI</span>
                            </>
                          )}
                        </button>
                      </div>

                      <div>
                        <label className="block text-xs text-apple-gray-500 dark:text-apple-gray-400 mb-1">
                          Kort navn (corpus ID)
                        </label>
                        <input
                          type="text"
                          value={corpusId}
                          onChange={(e) => setCorpusId(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, ''))}
                          className="w-full px-4 py-2 rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700 text-apple-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-apple-blue"
                          placeholder="f.eks. nis2"
                        />
                      </div>
                      <div>
                        <label className="block text-xs text-apple-gray-500 dark:text-apple-gray-400 mb-1">
                          Visningsnavn
                        </label>
                        <input
                          type="text"
                          value={displayName}
                          onChange={(e) => setDisplayName(e.target.value)}
                          className="w-full px-4 py-2 rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700 text-apple-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-apple-blue"
                          placeholder="f.eks. NIS2-direktivet"
                        />
                      </div>
                    </div>

                    {/* Eval generation checkbox */}
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={generateEval}
                        onChange={(e) => setGenerateEval(e.target.checked)}
                        className="w-4 h-4 rounded border-apple-gray-300 text-apple-blue focus:ring-apple-blue"
                      />
                      <span className="text-sm text-apple-gray-600 dark:text-apple-gray-300">
                        Generér eval cases (15 test-cases til kvalitetssikring)
                      </span>
                    </label>

                    {/* Run mode selection - only shown when eval is enabled */}
                    {generateEval && (
                      <div className="ml-6 space-y-2">
                        <label className="block text-xs text-apple-gray-500 dark:text-apple-gray-400">
                          Test mode
                        </label>
                        <select
                          value={evalRunMode}
                          onChange={(e) => setEvalRunMode(e.target.value as EvalRunMode)}
                          className="w-full px-3 py-1.5 text-sm rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700 text-apple-gray-700 dark:text-white focus:outline-none focus:ring-2 focus:ring-apple-blue"
                        >
                          <option value="retrieval_only">Retrieval only (hurtig)</option>
                          <option value="full">Full (standard)</option>
                          <option value="full_with_judge">Full + Judge (grundig)</option>
                        </select>
                        <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                          {evalRunMode === 'retrieval_only' && 'Tester kun retrieval - hurtigste option'}
                          {evalRunMode === 'full' && 'Standard test med LLM-generering'}
                          {evalRunMode === 'full_with_judge' && 'Fuld test inkl. LLM-as-judge scoring'}
                        </p>
                      </div>
                    )}

                    {/* Action buttons */}
                    <div className="flex justify-end gap-3">
                      <button
                        onClick={cancelIngestion}
                        className="px-4 py-2 text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded-lg transition-colors"
                      >
                        Annuller
                      </button>
                      <button
                        onClick={startIngestion}
                        disabled={!corpusId || !displayName}
                        className="px-4 py-2 bg-apple-blue text-white rounded-lg hover:bg-apple-blue/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                        Start indlæsning
                      </button>
                    </div>
                  </>
                )}

                {/* Progress display */}
                {(ingestion.isRunning || ingestion.completedStages.length > 0) && (
                  <div className="space-y-4">
                    {Object.entries(STAGE_LABELS).map(([stage, label]) => {
                      const isCompleted = ingestion.completedStages.includes(stage);
                      const isCurrent = ingestion.currentStage === stage;
                      const isPending = !isCompleted && !isCurrent;

                      return (
                        <div key={stage}>
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              {/* Status indicator - minimal */}
                              <span className={`w-4 text-center text-sm ${
                                isCompleted ? 'text-apple-gray-400 dark:text-apple-gray-500' :
                                isCurrent ? 'text-apple-blue' :
                                'text-apple-gray-300 dark:text-apple-gray-600'
                              }`}>
                                {isCompleted ? '✓' : isCurrent ? (
                                  STAGES_WITH_PROGRESS.has(stage) && ingestion.progress > 0 ? '●' : (
                                    <motion.span
                                      animate={{ opacity: [1, 0.4, 1] }}
                                      transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                                    >
                                      ●
                                    </motion.span>
                                  )
                                ) : '○'}
                              </span>

                              {/* Label */}
                              <span className={`flex-1 text-sm ${
                                isCompleted ? 'text-apple-gray-500 dark:text-apple-gray-400' :
                                isCurrent ? 'text-apple-gray-700 dark:text-apple-gray-200' :
                                'text-apple-gray-300 dark:text-apple-gray-600'
                              }`}>
                                {label}
                              </span>

                              {/* Progress percentage for current stage (only for stages with meaningful progress) */}
                              {isCurrent && STAGES_WITH_PROGRESS.has(stage) && ingestion.progress > 0 && (
                                <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 tabular-nums">
                                  {Math.round(ingestion.progress)}%
                                </span>
                              )}
                            </div>

                            {/* Progress bar for current stage - only for stages with meaningful progress */}
                            {isCurrent && STAGES_WITH_PROGRESS.has(stage) && ingestion.progress > 0 && (
                              <div className="ml-6 h-1 bg-apple-gray-100 dark:bg-apple-gray-500 rounded-full overflow-hidden">
                                <motion.div
                                  initial={{ width: 0 }}
                                  animate={{ width: `${ingestion.progress}%` }}
                                  transition={{ duration: 0.3, ease: 'easeOut' }}
                                  className="h-full bg-apple-blue rounded-full"
                                />
                              </div>
                            )}
                          </div>

                          {/* Preflight analysis - shown after download stage */}
                          {stage === 'download' && ingestion.preflight && (
                            <motion.div
                              initial={{ opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ duration: 0.2 }}
                              className="ml-6 mt-1.5 text-xs text-apple-gray-400 dark:text-apple-gray-500"
                            >
                              {Object.keys(ingestion.preflight.unhandled).length > 0 ? (
                                <span>
                                  {Object.values(ingestion.preflight.handled).reduce((a, b) => a + b, 0)} strukturer · {' '}
                                  <span className="text-amber-500 dark:text-amber-400">
                                    {Object.values(ingestion.preflight.unhandled).reduce((a, b) => a + b, 0)} ukendte
                                  </span>
                                </span>
                              ) : (
                                <span>
                                  {Object.values(ingestion.preflight.handled).reduce((a, b) => a + b, 0)} strukturer genkendt
                                </span>
                              )}
                            </motion.div>
                          )}
                        </div>
                      );
                    })}

                    {/* Status message during processing */}
                    {ingestion.isRunning && (
                      <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400 mt-4">
                        {ingestion.message}
                      </p>
                    )}

                    {/* Success state - shown when fully complete */}
                    {!ingestion.isRunning && ingestion.completedStages.includes('complete') && (
                      <div className="mt-8 space-y-6">
                        {/* Success header - minimal Apple style */}
                        <div className="text-center py-2">
                          <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">
                            <span className="font-medium text-apple-gray-700 dark:text-white">{displayName}</span> er nu tilgængeligt
                          </p>
                        </div>

                        {/* Eval results - shown if eval was run */}
                        {ingestion.evalSummary && (
                          <div className="space-y-4">
                            <div className="flex items-center justify-between">
                              <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500 uppercase tracking-wide">
                                Verifikation
                              </span>
                              <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                                {ingestion.evalSummary.passed}/{ingestion.evalSummary.total}
                              </span>
                            </div>

                            {/* Eval case list - minimal */}
                            <div className="divide-y divide-apple-gray-100 dark:divide-apple-gray-600">
                              {(ingestion.evalResults || []).map((result) => (
                                <div key={result.case_id}>
                                  {/* Case header - clickable */}
                                  <button
                                    onClick={() => setExpandedEvalCase(
                                      expandedEvalCase === result.case_id ? null : result.case_id
                                    )}
                                    className="w-full flex items-center gap-3 py-3 text-left"
                                  >
                                    {/* Pass/fail indicator - just a checkmark or dash */}
                                    <span className={`flex-shrink-0 text-sm ${
                                      result.passed
                                        ? 'text-apple-blue'
                                        : 'text-apple-gray-300 dark:text-apple-gray-500'
                                    }`}>
                                      {result.passed ? '✓' : '–'}
                                    </span>

                                    {/* Question preview */}
                                    <span className={`flex-1 text-sm truncate ${
                                      result.passed
                                        ? 'text-apple-gray-700 dark:text-apple-gray-200'
                                        : 'text-apple-gray-400 dark:text-apple-gray-500'
                                    }`}>
                                      {result.question}
                                    </span>

                                    {/* Article refs - just text */}
                                    <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                                      {result.actual_articles.slice(0, 2).map(a => a.replace('article:', '')).join(', ')}
                                      {result.actual_articles.length > 2 && ` +${result.actual_articles.length - 2}`}
                                    </span>

                                    {/* Expand icon */}
                                    <svg
                                      className={`w-4 h-4 text-apple-gray-300 dark:text-apple-gray-500 transition-transform duration-200 ${
                                        expandedEvalCase === result.case_id ? 'rotate-180' : ''
                                      }`}
                                      fill="none"
                                      viewBox="0 0 24 24"
                                      stroke="currentColor"
                                      strokeWidth={1.5}
                                    >
                                      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                                    </svg>
                                  </button>

                                  {/* Expanded answer */}
                                  <AnimatePresence>
                                    {expandedEvalCase === result.case_id && (
                                      <motion.div
                                        initial={{ height: 0, opacity: 0 }}
                                        animate={{ height: 'auto', opacity: 1 }}
                                        exit={{ height: 0, opacity: 0 }}
                                        transition={{ duration: 0.2 }}
                                      >
                                        <div className="pl-7 pb-4 space-y-3">
                                          {/* Answer with markdown formatting */}
                                          <div className="text-sm text-apple-gray-500 dark:text-apple-gray-400 max-h-64 overflow-y-auto leading-relaxed space-y-2">
                                            {(result.answer || '(Intet svar)').split('\n').map((line, idx) => {
                                              // Format ### headers
                                              if (line.startsWith('### ')) {
                                                return (
                                                  <p key={idx} className="font-medium text-apple-gray-700 dark:text-apple-gray-200 pt-2 first:pt-0">
                                                    {line.replace('### ', '')}
                                                  </p>
                                                );
                                              }
                                              // Format ## headers
                                              if (line.startsWith('## ')) {
                                                return (
                                                  <p key={idx} className="font-medium text-apple-gray-700 dark:text-apple-gray-200 pt-2 first:pt-0">
                                                    {line.replace('## ', '')}
                                                  </p>
                                                );
                                              }
                                              // Regular paragraph with bold and citation formatting
                                              if (line.trim()) {
                                                // Process bold (**text**) and citations [n]
                                                const parts = line.split(/(\*\*[^*]+\*\*|\[\d+\])/g);
                                                return (
                                                  <p key={idx}>
                                                    {parts.map((part, i) => {
                                                      // Bold text
                                                      if (part.startsWith('**') && part.endsWith('**')) {
                                                        return (
                                                          <span key={i} className="font-medium text-apple-gray-700 dark:text-apple-gray-200">
                                                            {part.slice(2, -2)}
                                                          </span>
                                                        );
                                                      }
                                                      // Citation reference [n]
                                                      if (/^\[\d+\]$/.test(part)) {
                                                        return (
                                                          <span key={i} className="text-apple-blue text-xs align-super cursor-default" title={`Kilde ${part}`}>
                                                            {part}
                                                          </span>
                                                        );
                                                      }
                                                      return part;
                                                    })}
                                                  </p>
                                                );
                                              }
                                              return null;
                                            })}
                                          </div>

                                          {/* References comparison */}
                                          <div className="text-xs text-apple-gray-400 dark:text-apple-gray-500 pt-2 border-t border-apple-gray-100 dark:border-apple-gray-600">
                                            <span>Forventet: {result.expected_articles.length > 0
                                              ? result.expected_articles.map(a => a.replace('article:', '')).join(', ')
                                              : '—'}</span>
                                            <span className="mx-3">·</span>
                                            <span>Faktisk: {result.actual_articles.length > 0
                                              ? result.actual_articles.map(a => a.replace('article:', '')).join(', ')
                                              : '—'}</span>
                                          </div>
                                        </div>
                                      </motion.div>
                                    )}
                                  </AnimatePresence>
                                </div>
                              ))}
                            </div>

                            {/* Disclaimer */}
                            <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500 pt-4">
                              Automatisk genereret · For permanent eval suite: udvid med juridisk ekspert
                            </p>
                          </div>
                        )}

                        {/* Action button - subtle */}
                        <button
                          onClick={cancelIngestion}
                          className="w-full py-3 text-apple-blue text-sm font-medium hover:text-apple-blue/70 transition-colors"
                        >
                          Færdig
                        </button>
                      </div>
                    )}

                    {/* Done button for partial completion (not fully complete) */}
                    {!ingestion.isRunning && ingestion.completedStages.length > 0 && !ingestion.completedStages.includes('complete') && (
                      <button
                        onClick={cancelIngestion}
                        className="mt-4 px-4 py-2 bg-apple-blue text-white rounded-lg hover:bg-apple-blue/90 transition-colors"
                      >
                        Færdig
                      </button>
                    )}
                  </div>
                )}

                {/* Error display */}
                {ingestion.error && (
                  <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/30 rounded-lg">
                    <p className="text-red-600 dark:text-red-400 text-sm">{ingestion.error}</p>
                    <button
                      onClick={cancelIngestion}
                      className="mt-2 text-sm text-red-600 dark:text-red-400 underline"
                    >
                      Tilbage til listen
                    </button>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            // Browse legislation view
            <motion.div
              key="browse"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="w-full"
            >
              {/* Filters */}
              <div className="mb-6 space-y-4">
                {/* Search input - for filtering locally after fetching from EUR-Lex */}
                <SearchInput
                  value={searchTerm}
                  onChange={setSearchTerm}
                  placeholder={hasSearched ? "Filtrer i resultater..." : "Klik 'Søg' først for at hente data..."}
                  disabled={!hasSearched}
                />

                {/* Filter controls */}
                <div className="flex flex-wrap gap-4 p-4 bg-white dark:bg-apple-gray-600 rounded-xl border border-apple-gray-100 dark:border-apple-gray-500">
                  {/* Year range - filters by adoption year (from CELEX) */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-apple-gray-600 dark:text-apple-gray-300">Vedtaget:</label>
                    <select
                      value={yearFrom}
                      onChange={(e) => {
                        const newFrom = Number(e.target.value);
                        setYearFrom(newFrom);
                        // Ensure range doesn't exceed 5 years
                        if (yearTo - newFrom + 1 > MAX_YEAR_SPAN) {
                          setYearTo(newFrom + MAX_YEAR_SPAN - 1);
                        }
                      }}
                      className="px-2 py-1 rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700 text-apple-gray-700 dark:text-white text-sm"
                    >
                      {YEAR_OPTIONS.filter(y => y <= yearTo).map((year) => (
                        <option key={year} value={year}>{year}</option>
                      ))}
                    </select>
                    <span className="text-apple-gray-400">-</span>
                    <select
                      value={yearTo}
                      onChange={(e) => {
                        const newTo = Number(e.target.value);
                        setYearTo(newTo);
                        // Ensure range doesn't exceed 5 years
                        if (newTo - yearFrom + 1 > MAX_YEAR_SPAN) {
                          setYearFrom(newTo - MAX_YEAR_SPAN + 1);
                        }
                      }}
                      className="px-2 py-1 rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700 text-apple-gray-700 dark:text-white text-sm"
                    >
                      {YEAR_OPTIONS.filter(y => y >= yearFrom).map((year) => (
                        <option key={year} value={year}>{year}</option>
                      ))}
                    </select>
                  </div>

                  {/* Document type */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-apple-gray-600 dark:text-apple-gray-300">Type:</label>
                    <select
                      value={docType}
                      onChange={(e) => setDocType(e.target.value as 'regulation' | 'directive' | 'all')}
                      className="px-2 py-1 rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700 text-apple-gray-700 dark:text-white text-sm"
                    >
                      <option value="regulation">Forordninger</option>
                      <option value="directive">Direktiver</option>
                      <option value="all">Alle</option>
                    </select>
                  </div>

                  {/* In-force toggle */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-apple-gray-600 dark:text-apple-gray-300">Kun gældende:</label>
                    <button
                      onClick={() => setInForceOnly(!inForceOnly)}
                      className={`relative w-10 h-5 rounded-full transition-colors ${
                        inForceOnly
                          ? 'bg-apple-blue'
                          : 'bg-apple-gray-200 dark:bg-apple-gray-500'
                      }`}
                      role="switch"
                      aria-checked={inForceOnly}
                    >
                      <motion.div
                        animate={{ x: inForceOnly ? 20 : 2 }}
                        transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        className="absolute top-0.5 w-4 h-4 bg-white rounded-full shadow"
                      />
                    </button>
                  </div>

                  {/* Ingestion status filter */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-apple-gray-600 dark:text-apple-gray-300">Status:</label>
                    <SegmentedControl
                      options={[
                        { value: 'all', label: 'Alt' },
                        { value: 'ingested', label: 'Indlæst' },
                        { value: 'not_ingested', label: 'Ikke indlæst' },
                      ]}
                      value={ingestionFilter}
                      onChange={(v) => { setIngestionFilter(v); setCurrentPage(1); }}
                      size="sm"
                    />
                  </div>

                  {/* Search button */}
                  <div className="flex items-center ml-auto">
                    <button
                      onClick={fetchLegislation}
                      disabled={isLoading}
                      className="px-4 py-1.5 bg-apple-blue text-white text-sm font-medium rounded-lg hover:bg-apple-blue/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                    >
                      {isLoading ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                          <span>Søger...</span>
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                          </svg>
                          <span>Søg</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Help text */}
                <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500 mt-2">
                  Vælg filtre og klik "Søg" for at hente fra EUR-Lex. Herefter kan du søge lokalt i titler, CELEX-numre og emneord.
                </p>
              </div>

              {/* Error display */}
              {error && (
                <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/30 rounded-xl">
                  <p className="text-red-600 dark:text-red-400">{error}</p>
                </div>
              )}

              {/* Loading state with progress bar */}
              {isLoading && (
                <div className="flex flex-col items-center justify-center py-12 space-y-4">
                  <div className="w-64">
                    <div className="flex justify-between text-xs text-apple-gray-500 dark:text-apple-gray-400 mb-2">
                      <span>Henter fra EUR-Lex...</span>
                      <span>{Math.round(loadingProgress)}%</span>
                    </div>
                    <div className="h-2 bg-apple-gray-200 dark:bg-apple-gray-500 rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${loadingProgress}%` }}
                        className="h-full bg-apple-blue rounded-full"
                        transition={{ duration: 0.2 }}
                      />
                    </div>
                    <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500 mt-2 text-center">
                      Første indlæsning kan tage op til et minut
                    </p>
                  </div>
                </div>
              )}

              {!isLoading && (
                <>
                  {/* Ingested legislation - quick overview at top */}
                  {ingestedLegislation.length > 0 && (
                    <div className="mb-6">
                      <h2 className="text-sm font-semibold text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                        Indlæst ({ingestedLegislation.length})
                      </h2>
                      <div className="bg-white dark:bg-apple-gray-600 rounded-xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-x-auto">
                        <table className="w-full min-w-[1000px]">
                          <thead>
                            <tr className="border-b border-apple-gray-100 dark:border-apple-gray-500 text-xs text-apple-gray-500 dark:text-apple-gray-400 bg-apple-gray-50 dark:bg-apple-gray-700">
                              <th className="text-left font-medium px-3 py-2.5 whitespace-nowrap w-28">CELEX</th>
                              <th className="text-left font-medium px-3 py-2.5">Titel</th>
                              <th className="text-left font-medium px-3 py-2.5 whitespace-nowrap w-20">Type</th>
                              <th className="text-left font-medium px-3 py-2.5 whitespace-nowrap w-24">Vedtaget</th>
                              <th className="text-left font-medium px-3 py-2.5 whitespace-nowrap w-28">Ikrafttræden</th>
                              <th
                                className="text-left font-medium px-3 py-2.5 whitespace-nowrap w-20 cursor-help"
                                title="Andel af HTML-mønstre der blev genkendt under indlæsning.&#10;&#10;100% = Alle mønstre håndteret korrekt&#10;<98% = Nogle elementer (tabeller, figurer) blev muligvis ikke indekseret"
                              >
                                Kvalitet
                              </th>
                              <th className="text-left font-medium px-3 py-2.5 whitespace-nowrap w-20">Status</th>
                              <th className="w-10"></th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-apple-gray-100 dark:divide-apple-gray-500">
                            {ingestedLegislation.map((leg) => (
                              <tr key={leg.celex_number} className="hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500/50 transition-colors">
                                <td className="px-3 py-2 whitespace-nowrap">
                                  <a
                                    href={leg.html_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-xs font-mono text-apple-blue hover:underline"
                                    title="Åbn i EUR-Lex"
                                  >
                                    {leg.celex_number}
                                  </a>
                                </td>
                                <td className="px-3 py-2 max-w-md">
                                  <div className="flex items-center gap-2">
                                    {leg.corpus_id && (
                                      <Tooltip content={getCorpusFullname(leg.corpus_id)} maxWidth={450}>
                                        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-apple-gray-100 text-apple-gray-600 dark:bg-apple-gray-600 dark:text-apple-gray-300 cursor-help">
                                          {getShortName(leg.title_da || leg.title_en || '', leg.corpus_id)}
                                        </span>
                                      </Tooltip>
                                    )}
                                    <Tooltip content={leg.corpus_id ? getCorpusFullname(leg.corpus_id) : (leg.title_da || leg.title_en)} maxWidth={450} className="min-w-0 overflow-hidden">
                                      <p className="text-sm text-apple-gray-700 dark:text-white truncate cursor-help">
                                        {leg.title_da || leg.title_en || '—'}
                                      </p>
                                    </Tooltip>
                                  </div>
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap">
                                  <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                                    {leg.document_type === 'Regulation' ? 'Forordn.' : leg.document_type === 'Directive' ? 'Direktiv' : '—'}
                                  </span>
                                </td>
                                <td className="px-3 py-2 text-xs text-apple-gray-600 dark:text-apple-gray-300 tabular-nums whitespace-nowrap">
                                  {formatDate(leg.last_modified)}
                                </td>
                                <td className="px-3 py-2 text-xs text-apple-gray-600 dark:text-apple-gray-300 tabular-nums whitespace-nowrap">
                                  {formatDate(leg.entry_into_force, true)}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap">
                                  {leg.quality ? (
                                    <span
                                      className={`text-xs tabular-nums ${
                                        leg.quality.unhandled_pct > 5
                                          ? 'text-orange-600 dark:text-orange-400'
                                          : leg.quality.unhandled_pct > 2
                                          ? 'text-yellow-600 dark:text-yellow-400'
                                          : 'text-apple-green'
                                      }`}
                                      title={getQualityTooltip(leg.quality)}
                                    >
                                      {(100 - leg.quality.unhandled_pct).toFixed(1)}%
                                    </span>
                                  ) : (
                                    <span className="text-xs text-apple-gray-400">—</span>
                                  )}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap">
                                  {leg.is_outdated ? (
                                    <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                                      Ny version
                                    </span>
                                  ) : (
                                    <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                                      Aktuel
                                    </span>
                                  )}
                                </td>
                                <td className="px-2 py-2">
                                  {leg.corpus_id && (
                                    <button
                                      onClick={() => handleRemoveCorpus(leg.corpus_id!, getShortName(leg.title_da || leg.title_en || '', leg.corpus_id))}
                                      className="p-1.5 text-apple-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors"
                                      aria-label="Fjern"
                                      title="Fjern corpus"
                                    >
                                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                      </svg>
                                    </button>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Full legislation list with filters */}
                  {sortedLegislation.length > 0 && (
                    <div>
                      <h2 className="text-sm font-semibold text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                        EU-lovgivning ({sortedLegislation.length})
                      </h2>
                      <div className="bg-white dark:bg-apple-gray-600 rounded-xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-x-auto">
                        <table className="w-full min-w-[1100px]">
                          <thead>
                            <tr className="border-b border-apple-gray-100 dark:border-apple-gray-500 text-xs text-apple-gray-500 dark:text-apple-gray-400 bg-apple-gray-50 dark:bg-apple-gray-700">
                              <th
                                className="text-left font-medium px-3 py-2.5 cursor-pointer hover:text-apple-gray-700 dark:hover:text-white transition-colors whitespace-nowrap w-28"
                                onClick={() => handleSort('celex_number')}
                              >
                                CELEX<SortIndicator field="celex_number" />
                              </th>
                              <th
                                className="text-left font-medium px-3 py-2.5 cursor-pointer hover:text-apple-gray-700 dark:hover:text-white transition-colors"
                                onClick={() => handleSort('title_da')}
                              >
                                Titel<SortIndicator field="title_da" />
                              </th>
                              <th className="text-left font-medium px-3 py-2.5 whitespace-nowrap w-48">
                                Emneord
                              </th>
                              <th
                                className="text-left font-medium px-3 py-2.5 cursor-pointer hover:text-apple-gray-700 dark:hover:text-white transition-colors whitespace-nowrap w-20"
                                onClick={() => handleSort('document_type')}
                              >
                                Type<SortIndicator field="document_type" />
                              </th>
                              <th
                                className="text-left font-medium px-3 py-2.5 cursor-pointer hover:text-apple-gray-700 dark:hover:text-white transition-colors whitespace-nowrap w-24"
                                onClick={() => handleSort('last_modified')}
                              >
                                Vedtaget<SortIndicator field="last_modified" />
                              </th>
                              <th
                                className="text-left font-medium px-3 py-2.5 cursor-pointer hover:text-apple-gray-700 dark:hover:text-white transition-colors whitespace-nowrap w-28"
                                onClick={() => handleSort('entry_into_force')}
                              >
                                Ikrafttræden<SortIndicator field="entry_into_force" />
                              </th>
                              <th
                                className="text-left font-medium px-3 py-2.5 whitespace-nowrap w-20 cursor-help"
                                title="Andel af HTML-mønstre der blev genkendt under indlæsning.&#10;&#10;100% = Alle mønstre håndteret korrekt&#10;<98% = Nogle elementer (tabeller, figurer) blev muligvis ikke indekseret"
                              >
                                Kvalitet
                              </th>
                              <th className="w-10"></th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-apple-gray-100 dark:divide-apple-gray-500">
                            {paginatedLegislation.map((leg) => (
                              <tr
                                key={leg.celex_number}
                                className="hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500/50 transition-colors"
                              >
                                <td className="px-3 py-2 whitespace-nowrap">
                                  <a
                                    href={leg.html_url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-xs font-mono text-apple-blue hover:underline"
                                    title="Åbn i EUR-Lex"
                                  >
                                    {leg.celex_number}
                                  </a>
                                </td>
                                <td className="px-3 py-2 max-w-md">
                                  <div className="flex items-center gap-2">
                                    {leg.is_ingested && leg.corpus_id && (
                                      <Tooltip content={getCorpusFullname(leg.corpus_id)} maxWidth={450}>
                                        <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-apple-gray-100 text-apple-gray-600 dark:bg-apple-gray-600 dark:text-apple-gray-300 cursor-help">
                                          {getShortName(leg.title_da || leg.title_en || '', leg.corpus_id)}
                                        </span>
                                      </Tooltip>
                                    )}
                                    <Tooltip content={leg.corpus_id ? getCorpusFullname(leg.corpus_id) : (leg.title_da || leg.title_en)} maxWidth={450} className="min-w-0 overflow-hidden">
                                      <p className="text-sm text-apple-gray-700 dark:text-white truncate cursor-help">
                                        {leg.title_da || leg.title_en || '—'}
                                      </p>
                                    </Tooltip>
                                  </div>
                                </td>
                                <td className="px-3 py-2 max-w-[200px]">
                                  {leg.eurovoc_labels && leg.eurovoc_labels.length > 0 ? (
                                    <div className="flex flex-wrap gap-1" title={leg.eurovoc_labels.join(', ')}>
                                      {leg.eurovoc_labels.slice(0, 3).map((label, idx) => (
                                        <span
                                          key={idx}
                                          className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/50 dark:text-purple-300 truncate max-w-[80px]"
                                        >
                                          {label}
                                        </span>
                                      ))}
                                      {leg.eurovoc_labels.length > 3 && (
                                        <span className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500">
                                          +{leg.eurovoc_labels.length - 3}
                                        </span>
                                      )}
                                    </div>
                                  ) : (
                                    <span className="text-xs text-apple-gray-400">—</span>
                                  )}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap">
                                  <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                                    {leg.document_type === 'Regulation' ? 'Forordn.' : leg.document_type === 'Directive' ? 'Direktiv' : '—'}
                                  </span>
                                </td>
                                <td className="px-3 py-2 text-xs text-apple-gray-600 dark:text-apple-gray-300 tabular-nums whitespace-nowrap">
                                  {formatDate(leg.last_modified)}
                                </td>
                                <td className="px-3 py-2 text-xs text-apple-gray-600 dark:text-apple-gray-300 tabular-nums whitespace-nowrap">
                                  {formatDate(leg.entry_into_force, true)}
                                </td>
                                <td className="px-3 py-2 whitespace-nowrap">
                                  {leg.quality ? (
                                    <span
                                      className={`text-xs tabular-nums ${
                                        leg.quality.unhandled_pct > 5
                                          ? 'text-orange-600 dark:text-orange-400'
                                          : leg.quality.unhandled_pct > 2
                                          ? 'text-yellow-600 dark:text-yellow-400'
                                          : 'text-apple-green'
                                      }`}
                                      title={getQualityTooltip(leg.quality)}
                                    >
                                      {(100 - leg.quality.unhandled_pct).toFixed(1)}%
                                    </span>
                                  ) : leg.is_ingested ? (
                                    <span className="text-xs text-apple-gray-400">—</span>
                                  ) : (
                                    <span className="text-xs text-apple-gray-400">—</span>
                                  )}
                                </td>
                                <td className="px-2 py-2">
                                  {leg.is_ingested ? (
                                    <div className="flex items-center gap-1">
                                      {leg.is_outdated && (
                                        <span className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500" title="Ny version tilgængelig">
                                          ●
                                        </span>
                                      )}
                                      <button
                                        onClick={() => handleRemoveCorpus(leg.corpus_id!, getShortName(leg.title_da || leg.title_en || '', leg.corpus_id))}
                                        className="p-1.5 text-apple-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors"
                                        aria-label="Fjern"
                                        title="Fjern corpus"
                                      >
                                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                        </svg>
                                      </button>
                                    </div>
                                  ) : (
                                    <button
                                      onClick={() => handleAddLegislation(leg)}
                                      className="p-1.5 text-apple-blue hover:bg-apple-blue/10 rounded transition-colors"
                                      aria-label="Tilføj"
                                      title="Tilføj til systemet"
                                    >
                                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                                      </svg>
                                    </button>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      {/* Pagination controls */}
                      {totalPages > 1 && (
                        <div className="flex items-center justify-between mt-4 px-1">
                          <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">
                            Viser {((currentPage - 1) * ITEMS_PER_PAGE) + 1}–{Math.min(currentPage * ITEMS_PER_PAGE, sortedLegislation.length)} af {sortedLegislation.length}
                          </p>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => setCurrentPage(1)}
                              disabled={currentPage === 1}
                              className="px-3 py-1.5 text-sm rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                              ««
                            </button>
                            <button
                              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                              disabled={currentPage === 1}
                              className="px-3 py-1.5 text-sm rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                              «
                            </button>
                            <span className="px-3 py-1.5 text-sm text-apple-gray-600 dark:text-apple-gray-300">
                              Side {currentPage} af {totalPages}
                            </span>
                            <button
                              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                              disabled={currentPage === totalPages}
                              className="px-3 py-1.5 text-sm rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                              »
                            </button>
                            <button
                              onClick={() => setCurrentPage(totalPages)}
                              disabled={currentPage === totalPages}
                              className="px-3 py-1.5 text-sm rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                            >
                              »»
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Empty state for filtered list */}
                  {sortedLegislation.length === 0 && legislation.length > 0 && (
                    <div className="text-center py-12">
                      <svg className="w-12 h-12 mx-auto text-apple-gray-300 dark:text-apple-gray-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                      </svg>
                      <p className="text-apple-gray-500 dark:text-apple-gray-400">
                        {ingestionFilter === 'ingested'
                          ? 'Ingen indlæste love matcher de aktive filtre'
                          : ingestionFilter === 'not_ingested'
                          ? 'Ingen ikke-indlæste love matcher de aktive filtre'
                          : 'Ingen lovgivning matcher de aktive filtre'}
                      </p>
                    </div>
                  )}

                  {/* Empty state - waiting for search */}
                  {!hasSearched && legislation.length === 0 && (
                    <div className="text-center py-16">
                      <svg className="w-16 h-16 mx-auto text-apple-gray-300 dark:text-apple-gray-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      <h3 className="text-lg font-medium text-apple-gray-700 dark:text-white mb-2">
                        Søg i EU-lovgivning
                      </h3>
                      <p className="text-apple-gray-500 dark:text-apple-gray-400 max-w-md mx-auto">
                        Vælg filtre ovenfor (årsinterval, dokumenttype, gældende) og klik "Søg" for at hente lovgivning fra EUR-Lex.
                      </p>
                    </div>
                  )}

                  {/* Empty state for no data after search */}
                  {hasSearched && legislation.length === 0 && (
                    <div className="text-center py-12">
                      <svg className="w-12 h-12 mx-auto text-apple-gray-300 dark:text-apple-gray-500 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      <p className="text-apple-gray-500 dark:text-apple-gray-400">
                        {searchTerm ? 'Ingen lovgivning matcher din søgning' : 'Ingen lovgivning fundet med de valgte filtre'}
                      </p>
                    </div>
                  )}
                </>
              )}
            </motion.div>
          )}
        </AnimatePresence>
        )}

        {/* Spacer to prevent footer overlap - renders inside scrollable area */}
        {runningEval.law && <div className="h-28" aria-hidden="true" />}
      </div>

      {/* Apple-style confirmation dialog */}
      <AnimatePresence>
        {confirmDialog.isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="fixed inset-0 z-50 flex items-center justify-center"
          >
            {/* Backdrop with blur */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/40 backdrop-blur-sm"
              onClick={cancelRemoveCorpus}
            />

            {/* Dialog card */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.15, ease: [0.4, 0, 0.2, 1] }}
              className="relative w-[280px] bg-white/95 dark:bg-apple-gray-600/95 backdrop-blur-xl rounded-2xl shadow-2xl overflow-hidden"
            >
              {/* Content */}
              <div className="px-5 pt-5 pb-4 text-center">
                <h3 className="text-base font-semibold text-apple-gray-700 dark:text-white mb-1">
                  Fjern {confirmDialog.displayName}?
                </h3>
                <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400 leading-snug">
                  Denne handling fjerner alle indekserede data og kan ikke fortrydes.
                </p>
              </div>

              {/* Buttons - Apple style stacked */}
              <div className="border-t border-apple-gray-200/50 dark:border-apple-gray-500/50">
                <button
                  onClick={confirmRemoveCorpus}
                  className="w-full py-3 text-[17px] font-normal text-red-500 hover:bg-apple-gray-100/50 dark:hover:bg-apple-gray-500/50 transition-colors"
                >
                  Fjern
                </button>
              </div>
              <div className="border-t border-apple-gray-200/50 dark:border-apple-gray-500/50">
                <button
                  onClick={cancelRemoveCorpus}
                  className="w-full py-3 text-[17px] font-semibold text-apple-blue hover:bg-apple-gray-100/50 dark:hover:bg-apple-gray-500/50 transition-colors"
                >
                  Annuller
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Running eval footer - Apple-style persistent status (renders at AdminPage level so it persists across tab switches) */}
      <AnimatePresence>
        {runningEval.law && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="fixed bottom-6 left-6 right-6 z-50 bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-hidden"
          >
            <div className="px-4 py-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  {runningEval.stage !== 'complete' && (
                    <div className="w-4 h-4 border-2 border-apple-blue border-t-transparent rounded-full animate-spin" />
                  )}
                  {runningEval.stage === 'complete' && (
                    <svg className="w-4 h-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                  <span className="text-sm font-medium text-apple-gray-700 dark:text-white">
                    {runningEval.stage === 'complete'
                      ? `Eval afsluttet`
                      : runningEval.stage === 'loading'
                      ? `Indlæser...`
                      : runningEval.stage === 'escalating'
                      ? `Eskalerer...`
                      : `Kører eval...`}
                  </span>
                  <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                    {runningEval.lawDisplayName || runningEval.law}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  {runningEval.stats && (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-green-500 font-medium">{runningEval.stats.passed} ✓</span>
                      <span className="text-red-500 font-medium">{runningEval.stats.failed} ✗</span>
                      <span className="text-apple-gray-400">/ {runningEval.stats.total}</span>
                    </div>
                  )}
                  {runningEval.stage === 'complete' && (
                    <button
                      onClick={clearRunningEval}
                      className="px-3 py-1 text-sm font-medium text-apple-blue hover:bg-apple-blue/10 rounded-md transition-colors"
                    >
                      Luk
                    </button>
                  )}
                </div>
              </div>
              {runningEval.stats && runningEval.stats.total > 0 && (
                <div className="h-1 bg-apple-gray-100 dark:bg-apple-gray-700 rounded-full overflow-hidden">
                  {runningEval.stage === 'complete' ? (
                    <div className="h-full flex">
                      <div
                        className="h-full bg-green-500"
                        style={{ width: `${(runningEval.stats.passed / runningEval.stats.total) * 100}%` }}
                      />
                      <div
                        className="h-full bg-red-500"
                        style={{ width: `${(runningEval.stats.failed / runningEval.stats.total) * 100}%` }}
                      />
                    </div>
                  ) : (
                    <div
                      className="h-full bg-apple-blue transition-all duration-300"
                      style={{ width: `${((runningEval.stats.passed + runningEval.stats.failed) / runningEval.stats.total) * 100}%` }}
                    />
                  )}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
