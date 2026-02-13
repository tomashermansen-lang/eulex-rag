/**
 * Section showing selected laws as tokens/pills.
 *
 * Single Responsibility: Display selected laws with overflow handling.
 * Follows Apple HIG token pattern.
 */

import { useState } from 'react';
import { LawToken } from './LawToken';

export interface SelectedLaw {
  id: string;
  name: string;
}

export interface SelectedLawsSectionProps {
  /** List of selected laws */
  selectedLaws: SelectedLaw[];
  /** Callback when a law is removed */
  onRemove: (id: string) => void;
  /** Callback to clear all selections */
  onClearAll: () => void;
  /** Maximum visible tokens before overflow (default: 5) */
  maxVisible?: number;
}

export function SelectedLawsSection({
  selectedLaws,
  onRemove,
  onClearAll,
  maxVisible = 5,
}: SelectedLawsSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const hasOverflow = selectedLaws.length > maxVisible && !isExpanded;
  const visibleLaws = hasOverflow
    ? selectedLaws.slice(0, maxVisible)
    : selectedLaws;
  const overflowCount = selectedLaws.length - maxVisible;

  // Empty state
  if (selectedLaws.length === 0) {
    return (
      <div className="py-4">
        <p className="text-sm text-apple-gray-400 dark:text-apple-gray-500 text-center">
          Ingen love valgt
        </p>
      </div>
    );
  }

  return (
    <div className="py-3">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300">
            Valgte love
          </span>
          <span className="px-1.5 py-0.5 text-xs font-medium bg-apple-blue-500 text-white rounded-full min-w-[20px] text-center">
            {selectedLaws.length}
          </span>
        </div>
        <button
          onClick={onClearAll}
          className="text-xs text-apple-gray-400 hover:text-apple-gray-600 dark:hover:text-apple-gray-300 transition-colors"
          aria-label="Ryd alle"
        >
          Ryd alle
        </button>
      </div>

      {/* Tokens */}
      <div className="flex flex-wrap gap-2">
        {visibleLaws.map((law) => (
          <LawToken
            key={law.id}
            id={law.id}
            name={law.name}
            onRemove={onRemove}
          />
        ))}

        {/* Overflow indicator */}
        {hasOverflow && (
          <button
            onClick={() => setIsExpanded(true)}
            className="inline-flex items-center px-2.5 py-1 bg-apple-gray-200 dark:bg-apple-gray-500 text-apple-gray-600 dark:text-apple-gray-200 rounded-full text-sm hover:bg-apple-gray-300 dark:hover:bg-apple-gray-400 transition-colors"
          >
            +{overflowCount}
          </button>
        )}
      </div>
    </div>
  );
}
