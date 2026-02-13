/**
 * SuggestedLabels component.
 *
 * Single Responsibility: Display AI-suggested labels based on user's question.
 */

import { useMemo } from 'react';
import { suggestLabels, type LabelWithCount } from './labelUtils';

export interface SuggestedLabelsProps {
  question: string;
  allLabels: LabelWithCount[];
  selectedLabels: string[];
  onSelect: (label: string) => void;
  disabled?: boolean;
}

/**
 * Display AI-suggested labels based on user's question.
 * Uses keyword matching (Phase 1) to find relevant labels.
 * Renders nothing when no suggestions or empty question.
 */
export function SuggestedLabels({
  question,
  allLabels,
  selectedLabels,
  onSelect,
  disabled = false,
}: SuggestedLabelsProps) {
  // Get suggested labels based on question keywords
  const suggestions = useMemo(() => {
    const allLabelNames = allLabels.map((l) => l.label);
    const suggested = suggestLabels(question, allLabelNames);

    // Filter out already selected labels
    return suggested.filter((label) => !selectedLabels.includes(label));
  }, [question, allLabels, selectedLabels]);

  // Get labels with counts for display
  const suggestionsWithCount = useMemo(() => {
    return suggestions
      .map((label) => allLabels.find((l) => l.label === label))
      .filter((l): l is LabelWithCount => l !== undefined);
  }, [suggestions, allLabels]);

  // Don't render if no suggestions
  if (suggestionsWithCount.length === 0) {
    return null;
  }

  return (
    <div className="py-2">
      <div className="text-xs text-apple-gray-500 dark:text-apple-gray-400 mb-1.5">
        Foreslået:
      </div>
      <div className="flex flex-wrap gap-1.5">
        {suggestionsWithCount.map(({ label, count }) => (
          <button
            key={label}
            type="button"
            onClick={() => onSelect(label)}
            disabled={disabled}
            className={`
              inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium
              bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300
              hover:bg-amber-200 dark:hover:bg-amber-900/50
              transition-colors
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <span>✨</span>
            <span>{label}</span>
            <span className="text-amber-500 dark:text-amber-400">({count})</span>
          </button>
        ))}
      </div>
    </div>
  );
}
