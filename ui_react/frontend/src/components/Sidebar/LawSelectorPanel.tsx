/**
 * Law selector panel component.
 *
 * Single Responsibility: Coordinate law selection UI with unified interface for all modes.
 */

import { useState, useMemo } from 'react';
import type { CorpusInfo, CorpusScope } from '../../types';
import { SearchInput } from '../Common/SearchInput';
import { LawListItem, type CheckboxMode } from './LawListItem';
import { EurovocFilterSection } from './EurovocFilterSection';
import { HiddenSelectionSummary } from './HiddenSelectionSummary';

interface LawSelectorPanelProps {
  corpusScope: CorpusScope;
  corpora: CorpusInfo[];
  targetCorpora: string[];
  onTargetCorporaChange: (corpora: string[]) => void;
  disabled?: boolean;
  /** User's current question (for AI label suggestions) */
  question?: string;
}

/**
 * Filter corpora by search query (case-insensitive).
 * Matches against id, name, and fullname.
 */
function filterCorporaByQuery(corpora: CorpusInfo[], query: string): CorpusInfo[] {
  if (!query.trim()) return corpora;

  const lowerQuery = query.toLowerCase();
  return corpora.filter((c) =>
    c.id.toLowerCase().includes(lowerQuery) ||
    (c.name?.toLowerCase().includes(lowerQuery)) ||
    (c.fullname?.toLowerCase().includes(lowerQuery)) ||
    (c.eurovoc_labels?.some((label) => label.toLowerCase().includes(lowerQuery)))
  );
}

/**
 * Filter corpora by selected eurovoc labels.
 * A corpus matches if it has at least one of the selected labels.
 */
function filterCorporaByLabels(corpora: CorpusInfo[], labels: string[]): CorpusInfo[] {
  if (labels.length === 0) return corpora;

  return corpora.filter((c) =>
    c.eurovoc_labels?.some((label) => labels.includes(label))
  );
}

/**
 * Sort corpora alphabetically by ID.
 */
function sortCorpora(corpora: CorpusInfo[]): CorpusInfo[] {
  return [...corpora].sort((a, b) => a.id.localeCompare(b.id));
}

/**
 * Map corpus scope to checkbox mode.
 */
function getCheckboxMode(corpusScope: CorpusScope): CheckboxMode {
  switch (corpusScope) {
    case 'single':
      return 'radio';
    case 'explicit':
      return 'checkbox';
    case 'all':
      return 'disabled';
  }
}

export function LawSelectorPanel({
  corpusScope,
  corpora,
  targetCorpora,
  onTargetCorporaChange,
  disabled = false,
  question = '',
}: LawSelectorPanelProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLabels, setSelectedLabels] = useState<string[]>([]);

  // Sort corpora alphabetically
  const sortedCorpora = useMemo(() => sortCorpora(corpora), [corpora]);

  // Filter by search query
  const queryFiltered = useMemo(
    () => filterCorporaByQuery(sortedCorpora, searchQuery),
    [sortedCorpora, searchQuery]
  );

  // Filter by selected eurovoc labels
  const filteredCorpora = useMemo(
    () => filterCorporaByLabels(queryFiltered, selectedLabels),
    [queryFiltered, selectedLabels]
  );

  // Derive checkbox mode from scope
  const checkboxMode = getCheckboxMode(corpusScope);
  const isAllMode = corpusScope === 'all';
  const isExplicitMode = corpusScope === 'explicit';

  // Handle click based on mode
  const handleClick = (corpusId: string) => {
    if (isAllMode) return; // No interaction in all mode

    if (corpusScope === 'single') {
      // Radio behavior: single selection
      onTargetCorporaChange([corpusId]);
    } else {
      // Toggle for explicit mode
      if (targetCorpora.includes(corpusId)) {
        onTargetCorporaChange(targetCorpora.filter((id) => id !== corpusId));
      } else {
        onTargetCorporaChange([...targetCorpora, corpusId]);
      }
    }
  };

  // Determine if an item is selected
  const isSelected = (corpusId: string): boolean => {
    if (isAllMode) return true; // All are selected in all mode
    return targetCorpora.includes(corpusId);
  };

  // Effective target corpora for all mode
  const effectiveTargetCorpora = isAllMode
    ? filteredCorpora.map((c) => c.id)
    : targetCorpora;

  // Compute hidden selected laws (selected but not visible due to filters)
  const hiddenSelectedLaws = useMemo(() => {
    // Never show hidden laws in all mode
    if (isAllMode) return [];

    const visibleIds = new Set(filteredCorpora.map((c) => c.id));
    return targetCorpora
      .filter((id) => !visibleIds.has(id))
      .map((id) => {
        const corpus = corpora.find((c) => c.id === id);
        return { id, name: corpus?.name ?? id.toUpperCase() };
      });
  }, [targetCorpora, filteredCorpora, corpora, isAllMode]);

  // Handle removing a hidden law
  const handleRemoveHiddenLaw = (id: string) => {
    onTargetCorporaChange(targetCorpora.filter((corpusId) => corpusId !== id));
  };

  // Handle clearing the EuroVoc filter
  const handleClearFilter = () => {
    setSelectedLabels([]);
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Search input */}
      <SearchInput
        value={searchQuery}
        onChange={setSearchQuery}
        placeholder="Søg i love..."
        disabled={disabled}
      />

      {/* EuroVoc filter section with AI suggestions */}
      <div className="mt-2">
        <EurovocFilterSection
          question={question}
          corpora={queryFiltered}
          selectedLabels={selectedLabels}
          onSelectionChange={setSelectedLabels}
          disabled={disabled}
        />
      </div>

      {/* Warning when no law is selected (single or explicit mode) */}
      {!isAllMode && targetCorpora.length === 0 && (
        <div className="mt-3 p-2 text-sm text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 rounded-lg">
          {corpusScope === 'single' ? 'Vælg en lov' : 'Vælg mindst én lov'}
        </div>
      )}

      {/* Hidden selection summary - shows selected laws hidden by filters */}
      <HiddenSelectionSummary
        hiddenLaws={hiddenSelectedLaws}
        onRemove={handleRemoveHiddenLaw}
        onClearFilter={handleClearFilter}
        disabled={disabled}
      />

      {/* Law list - unified flat list for all modes */}
      <div className="mt-3 flex-1 overflow-y-auto scroll-smooth">
        {/* Empty state */}
        {filteredCorpora.length === 0 && (
          <div className="p-4 text-center text-sm text-apple-gray-400">
            Ingen love matcher "{searchQuery}"
          </div>
        )}

        {/* Flat list - styled like the label list */}
        {filteredCorpora.length > 0 && (
          <div className="border border-apple-gray-200 dark:border-apple-gray-600 rounded-lg bg-white dark:bg-apple-gray-800 overflow-hidden">
            {filteredCorpora.map((corpus) => (
              <LawListItem
                key={corpus.id}
                corpus={corpus}
                isSelected={isSelected(corpus.id)}
                checkboxMode={checkboxMode}
                onClick={() => handleClick(corpus.id)}
                disabled={disabled}
                selectedLabels={selectedLabels}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
