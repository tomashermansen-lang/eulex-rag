/**
 * Panel component for managing eval cases for a law.
 *
 * Shows a list of cases with filtering and provides CRUD operations.
 */

import { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { EvalCase, EvalTestType } from '../../types';
import { listEvalCases, deleteEvalCase, duplicateEvalCase } from '../../services/api';
import { SearchInput } from '../Common/SearchInput';
import { EvalCaseEditor } from './EvalCaseEditor';
import { EVAL_TEST_TYPE_LABELS } from './evalUtils';

interface EvalSuitePanelProps {
  law: string;
  displayName: string;
  isOpen: boolean;
  onClose: () => void;
  onCasesChanged: () => void;
}

type FilterType = 'all' | 'manual' | 'auto' | EvalTestType;

const TEST_TYPE_LABELS = EVAL_TEST_TYPE_LABELS as Record<EvalTestType, string>;

export function EvalSuitePanel({
  law,
  displayName,
  isOpen,
  onClose,
  onCasesChanged,
}: EvalSuitePanelProps) {
  // Data state
  const [cases, setCases] = useState<EvalCase[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // UI state
  const [searchQuery, setSearchQuery] = useState('');
  const [filter, setFilter] = useState<FilterType>('all');
  const [expandedCase, setExpandedCase] = useState<string | null>(null);

  // Editor state
  const [editorOpen, setEditorOpen] = useState(false);
  const [editingCase, setEditingCase] = useState<EvalCase | null>(null);

  // Delete confirmation
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Load cases when panel opens
  useEffect(() => {
    if (isOpen) {
      loadCases();
    }
  }, [isOpen, law]);

  const loadCases = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await listEvalCases(law);
      setCases(response.cases);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Kunne ikke hente cases');
    } finally {
      setIsLoading(false);
    }
  };

  // Filter and search cases
  const filteredCases = useMemo(() => {
    let result = [...cases];

    // Apply filter
    if (filter === 'manual') {
      result = result.filter((c) => c.origin === 'manual');
    } else if (filter === 'auto') {
      result = result.filter((c) => c.origin === 'auto');
    } else if (filter !== 'all') {
      result = result.filter((c) => c.test_types.includes(filter as EvalTestType));
    }

    // Apply search
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (c) =>
          c.id.toLowerCase().includes(query) ||
          c.prompt.toLowerCase().includes(query)
      );
    }

    return result;
  }, [cases, filter, searchQuery]);

  // Handle create new case
  const handleCreate = () => {
    setEditingCase(null);
    setEditorOpen(true);
  };

  // Handle edit case
  const handleEdit = (c: EvalCase) => {
    setEditingCase(c);
    setEditorOpen(true);
  };

  // Handle duplicate
  const handleDuplicate = async (caseId: string) => {
    try {
      const duplicated = await duplicateEvalCase(law, caseId);
      setCases((prev) => [...prev, duplicated]);
      onCasesChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Kunne ikke duplikere');
    }
  };

  // Handle delete
  const handleDelete = async (caseId: string) => {
    setIsDeleting(true);
    try {
      await deleteEvalCase(law, caseId);
      setCases((prev) => prev.filter((c) => c.id !== caseId));
      setDeleteConfirm(null);
      onCasesChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Kunne ikke slette');
    } finally {
      setIsDeleting(false);
    }
  };

  // Handle case saved from editor
  const handleCaseSaved = (savedCase: EvalCase) => {
    if (editingCase) {
      // Update existing
      setCases((prev) =>
        prev.map((c) => (c.id === savedCase.id ? savedCase : c))
      );
    } else {
      // Add new
      setCases((prev) => [...prev, savedCase]);
    }
    setEditorOpen(false);
    setEditingCase(null);
    onCasesChanged();
  };

  if (!isOpen) return null;

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      className="flex flex-col flex-1 min-h-0 bg-white dark:bg-apple-gray-600 rounded-2xl shadow-sm border border-apple-gray-100 dark:border-apple-gray-500 overflow-hidden"
    >
      {/* Header */}
      <div className="px-4 py-3 border-b border-apple-gray-100 dark:border-apple-gray-500">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={onClose}
              className="p-1.5 text-apple-gray-400 hover:text-apple-gray-600 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded-lg transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <span className="text-sm font-medium text-apple-gray-700 dark:text-white">
              {displayName}
            </span>
            <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
              {cases.length} {cases.length === 1 ? 'case' : 'cases'}
            </span>
          </div>
          <div className="flex items-center gap-1">
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value as FilterType)}
              className="px-2 py-1 text-sm rounded-md bg-transparent text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 transition-colors cursor-pointer"
            >
              <option value="all">Alle</option>
              <option value="manual">Manuelle</option>
              <option value="auto">Auto</option>
              <optgroup label="Test Type">
                {Object.entries(TEST_TYPE_LABELS).map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </optgroup>
            </select>
            <button
              onClick={handleCreate}
              className="px-3 py-1 text-sm font-medium text-apple-blue hover:bg-apple-blue/10 rounded-md transition-colors"
            >
              Ny
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="mt-3">
          <SearchInput
            value={searchQuery}
            onChange={setSearchQuery}
            placeholder="Søg i cases..."
            className="w-full"
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-6 h-6 border-2 border-apple-blue border-t-transparent rounded-full animate-spin" />
          </div>
        ) : error ? (
          <div className="p-4 text-center">
            <p className="text-red-500 text-sm">{error}</p>
            <button
              onClick={loadCases}
              className="mt-2 text-sm text-apple-blue hover:underline"
            >
              Prøv igen
            </button>
          </div>
        ) : filteredCases.length === 0 ? (
          <div className="p-8 text-center">
            <svg className="w-12 h-12 mx-auto text-apple-gray-300 dark:text-apple-gray-500 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <p className="text-sm text-apple-gray-500 dark:text-apple-gray-400">
              {searchQuery || filter !== 'all' ? 'Ingen cases matcher filteret' : 'Ingen eval cases endnu'}
            </p>
            {!searchQuery && filter === 'all' && (
              <button
                onClick={handleCreate}
                className="mt-3 text-sm text-apple-blue hover:underline"
              >
                Opret den første case
              </button>
            )}
          </div>
        ) : (
          <div className="divide-y divide-apple-gray-100 dark:divide-apple-gray-500">
            {filteredCases.map((c) => (
              <div key={c.id}>
                {/* Case row */}
                <div
                  className="px-4 py-3 flex items-center gap-3 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500/30 transition-colors cursor-pointer"
                  onClick={() => setExpandedCase(expandedCase === c.id ? null : c.id)}
                >
                  {/* Origin badge */}
                  <span
                    className={`flex-shrink-0 px-1.5 py-0.5 text-[10px] font-medium rounded ${
                      c.origin === 'manual'
                        ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                        : 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400'
                    }`}
                  >
                    {c.origin === 'manual' ? 'MANUAL' : 'AUTO'}
                  </span>

                  {/* Case info */}
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-apple-gray-700 dark:text-white truncate">
                      {c.prompt}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500">
                        {c.id}
                      </span>
                      {c.test_types.slice(0, 3).map((tt) => (
                        <span
                          key={tt}
                          className="px-1 py-0.5 text-[10px] rounded bg-apple-gray-100 dark:bg-apple-gray-700 text-apple-gray-600 dark:text-apple-gray-300"
                        >
                          {TEST_TYPE_LABELS[tt]}
                        </span>
                      ))}
                      {c.test_types.length > 3 && (
                        <span className="text-[10px] text-apple-gray-400">
                          +{c.test_types.length - 3}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Profile badge */}
                  <span className="flex-shrink-0 px-1.5 py-0.5 text-[10px] font-medium rounded bg-apple-gray-200 dark:bg-apple-gray-600 text-apple-gray-600 dark:text-apple-gray-300">
                    {c.profile}
                  </span>

                  {/* Expand icon */}
                  <svg
                    className={`w-4 h-4 text-apple-gray-400 transition-transform ${
                      expandedCase === c.id ? 'rotate-180' : ''
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
                  {expandedCase === c.id && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      className="overflow-hidden bg-apple-gray-50/50 dark:bg-apple-gray-500/20 border-t border-apple-gray-100 dark:border-apple-gray-600"
                    >
                      <div className="px-4 py-3 pl-6 space-y-3">
                        {/* Expected behavior summary */}
                        <div>
                          <p className="text-[11px] font-medium text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                            Forventet adfærd
                          </p>
                          <div className="flex flex-wrap items-center gap-2">
                            <span className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                              c.expected.behavior === 'answer'
                                ? 'bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400'
                                : 'bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400'
                            }`}>
                              {c.expected.behavior === 'answer' ? 'Skal svare' : 'Skal afstå'}
                            </span>

                            {c.expected.must_include_any_of.length > 0 && (
                              <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                                Kræver én af: {c.expected.must_include_any_of.join(', ')}
                              </span>
                            )}

                            {c.expected.must_include_any_of_2 && c.expected.must_include_any_of_2.length > 0 && (
                              <span className="text-xs text-apple-gray-500 dark:text-apple-gray-400">
                                Kræver også én af: {c.expected.must_include_any_of_2.join(', ')}
                              </span>
                            )}

                            {c.expected.must_include_all_of && c.expected.must_include_all_of.length > 0 && (
                              <span className="text-xs text-blue-500 dark:text-blue-400">
                                Skal inkludere alle: {c.expected.must_include_all_of.join(', ')}
                              </span>
                            )}

                            {c.expected.must_not_include_any_of.length > 0 && (
                              <span className="text-xs text-red-500">
                                Må ikke: {c.expected.must_not_include_any_of.join(', ')}
                              </span>
                            )}

                            {c.expected.contract_check && (
                              <span className="text-xs text-purple-500 dark:text-purple-400">
                                Citeringer: {c.expected.min_citations ?? 0}–{c.expected.max_citations ?? '∞'}
                              </span>
                            )}
                          </div>
                        </div>

                        {c.expected.notes && (
                          <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 italic">
                            {c.expected.notes}
                          </p>
                        )}

                        {/* Actions */}
                        <div className="flex gap-3 pt-1">
                          <button
                            onClick={(e) => { e.stopPropagation(); handleEdit(c); }}
                            className="text-xs font-medium text-apple-blue hover:text-apple-blue/80 transition-colors"
                          >
                            Rediger
                          </button>
                          <button
                            onClick={(e) => { e.stopPropagation(); handleDuplicate(c.id); }}
                            className="text-xs font-medium text-apple-gray-500 dark:text-apple-gray-400 hover:text-apple-gray-700 dark:hover:text-apple-gray-200 transition-colors"
                          >
                            Duplikér
                          </button>
                          {deleteConfirm === c.id ? (
                            <div className="flex items-center gap-2">
                              <span className="text-xs text-red-500">Slet?</span>
                              <button
                                onClick={(e) => { e.stopPropagation(); handleDelete(c.id); }}
                                disabled={isDeleting}
                                className="px-2 py-0.5 text-xs font-medium text-white bg-red-500 hover:bg-red-600 rounded transition-colors disabled:opacity-50"
                              >
                                {isDeleting ? '...' : 'Ja'}
                              </button>
                              <button
                                onClick={(e) => { e.stopPropagation(); setDeleteConfirm(null); }}
                                className="px-2 py-0.5 text-xs font-medium text-apple-gray-600 hover:bg-apple-gray-200 dark:hover:bg-apple-gray-600 rounded transition-colors"
                              >
                                Nej
                              </button>
                            </div>
                          ) : (
                            <button
                              onClick={(e) => { e.stopPropagation(); setDeleteConfirm(c.id); }}
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

      {/* Editor modal */}
      <EvalCaseEditor
        law={law}
        existingCase={editingCase}
        isOpen={editorOpen}
        onClose={() => { setEditorOpen(false); setEditingCase(null); }}
        onSaved={handleCaseSaved}
      />
    </motion.div>
  );
}
