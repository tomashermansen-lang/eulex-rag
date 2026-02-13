/**
 * HiddenSelectionSummary component.
 *
 * Single Responsibility: Display selected laws hidden by active filters.
 */

import { LawToken } from './LawToken';

interface HiddenLaw {
  id: string;
  name: string;
}

export interface HiddenSelectionSummaryProps {
  /** Laws that are selected but hidden by filters */
  hiddenLaws: HiddenLaw[];
  /** Callback to remove a law from selection */
  onRemove: (id: string) => void;
  /** Callback to clear the EuroVoc filter */
  onClearFilter: () => void;
  /** Whether actions are disabled */
  disabled?: boolean;
}

const MAX_VISIBLE = 5;

/**
 * Displays selected laws hidden by filters, with removal actions.
 * Returns null when no laws are hidden.
 */
export function HiddenSelectionSummary({
  hiddenLaws,
  onRemove,
  onClearFilter,
  disabled = false,
}: HiddenSelectionSummaryProps) {
  // Don't render if no hidden laws
  if (hiddenLaws.length === 0) {
    return null;
  }

  const visibleLaws = hiddenLaws.slice(0, MAX_VISIBLE);
  const overflowCount = hiddenLaws.length - MAX_VISIBLE;

  return (
    <section
      role="status"
      className="bg-amber-50 dark:bg-amber-900/20 rounded-lg p-2 mt-3"
    >
      {/* Warning header */}
      <div className="flex items-center gap-1.5 mb-2">
        <svg
          className="w-4 h-4 text-amber-600 dark:text-amber-400 flex-shrink-0"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
        <span className="text-xs font-semibold text-amber-700 dark:text-amber-400">
          Valgte (skjult af filter)
        </span>
      </div>

      {/* Tokens */}
      <div className="flex flex-wrap gap-1.5 mb-2">
        {visibleLaws.map((law) => (
          <LawToken
            key={law.id}
            id={law.id}
            name={law.name}
            onRemove={onRemove}
          />
        ))}

        {/* Overflow indicator */}
        {overflowCount > 0 && (
          <span className="inline-flex items-center px-2 py-1 bg-amber-200 dark:bg-amber-800 text-amber-700 dark:text-amber-300 rounded-full text-xs font-medium">
            +{overflowCount}
          </span>
        )}
      </div>

      {/* Clear filter button */}
      <button
        type="button"
        onClick={onClearFilter}
        disabled={disabled}
        className={`text-xs text-apple-blue hover:underline ${
          disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
        }`}
      >
        Ryd emnefilter
      </button>
    </section>
  );
}
