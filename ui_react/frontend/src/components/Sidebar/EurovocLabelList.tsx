/**
 * EurovocLabelList component.
 *
 * Single Responsibility: Display searchable, scrollable list of ALL labels.
 */

import { useState, useMemo } from 'react';
import { SearchInput } from '../Common/SearchInput';
import type { LabelWithCount } from './labelUtils';

export interface EurovocLabelListProps {
  labels: LabelWithCount[];
  selectedLabels: string[];
  suggestedLabels: string[];
  onToggle: (label: string) => void;
  disabled?: boolean;
}

/** Maximum height for the scrollable list */
const MAX_LIST_HEIGHT = 200;

/**
 * Searchable, scrollable checkbox list of ALL labels.
 * Supports search filtering, selection state, and suggested highlighting.
 */
export function EurovocLabelList({
  labels,
  selectedLabels,
  suggestedLabels,
  onToggle,
  disabled = false,
}: EurovocLabelListProps) {
  const [searchQuery, setSearchQuery] = useState('');

  // Filter labels by search query
  const filteredLabels = useMemo(() => {
    if (!searchQuery.trim()) return labels;

    const lowerQuery = searchQuery.toLowerCase();
    return labels.filter((l) => l.label.toLowerCase().includes(lowerQuery));
  }, [labels, searchQuery]);

  // Don't render if no labels
  if (labels.length === 0) {
    return null;
  }

  const isFiltering = searchQuery.trim().length > 0;

  return (
    <div className="py-2">
      {/* Header */}
      <div className="mb-2">
        <span
          className="text-xs text-apple-gray-500 dark:text-apple-gray-400"
          title="EuroVoc-klassifikation"
        >
          Emner
        </span>
      </div>

      {/* Search input - full width */}
      <div className="mb-2">
        <SearchInput
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Søg i emner..."
          disabled={disabled}
        />
      </div>

      {/* Scrollable label list */}
      <div
        className="overflow-y-auto border border-apple-gray-200 dark:border-apple-gray-600 rounded-lg bg-white dark:bg-apple-gray-800"
        style={{ maxHeight: MAX_LIST_HEIGHT }}
      >
        {filteredLabels.map(({ label, count }) => {
          const isSelected = selectedLabels.includes(label);
          const isSuggested = suggestedLabels.includes(label);

          return (
            <label
              key={label}
              className={`
                flex items-center gap-2 px-3 py-1.5 text-sm cursor-pointer
                hover:bg-apple-gray-50 dark:hover:bg-apple-gray-700
                ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
              `}
            >
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => onToggle(label)}
                disabled={disabled}
                className="rounded border-apple-gray-300 dark:border-apple-gray-600 text-apple-blue focus:ring-apple-blue"
              />
              {isSuggested && (
                <span className="text-amber-500" title="Foreslået">
                  ★
                </span>
              )}
              <span className="flex-1 text-apple-gray-700 dark:text-apple-gray-300">
                {label}
              </span>
              <span className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                ({count})
              </span>
            </label>
          );
        })}
      </div>

      {/* Match count when filtering */}
      {isFiltering && (
        <div className="mt-1 text-xs text-apple-gray-400 dark:text-apple-gray-500">
          {filteredLabels.length} emner matcher &quot;{searchQuery}&quot;
        </div>
      )}
    </div>
  );
}
