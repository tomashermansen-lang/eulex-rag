/**
 * FilterSummaryBar component.
 *
 * Single Responsibility: Display applied filters as removable chips + result count.
 */

export interface FilterSummaryBarProps {
  selectedLabels: string[];
  filteredCount: number;
  totalCount: number;
  onRemove: (label: string) => void;
  onClearAll: () => void;
  disabled?: boolean;
}

/**
 * Display applied filters as removable chips with result count.
 * Renders nothing when no labels are selected.
 */
export function FilterSummaryBar({
  selectedLabels,
  filteredCount,
  totalCount,
  onRemove,
  onClearAll,
  disabled = false,
}: FilterSummaryBarProps) {
  // Don't render if no filters applied
  if (selectedLabels.length === 0) {
    return null;
  }

  // Singular/plural text
  const lawText = totalCount === 1 ? 'lov' : 'love';

  return (
    <div className="flex flex-col gap-2 py-2 px-1">
      {/* Selected label chips */}
      <div className="flex flex-wrap gap-1.5">
        {selectedLabels.map((label) => (
          <span
            key={label}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-apple-blue text-white"
          >
            {label}
            <button
              type="button"
              onClick={() => onRemove(label)}
              disabled={disabled}
              aria-label={`Fjern ${label}`}
              className={`
                ml-0.5 rounded-full p-0.5
                hover:bg-white/20 transition-colors
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 16 16"
                fill="currentColor"
                className="w-3 h-3"
              >
                <path d="M5.28 4.22a.75.75 0 0 0-1.06 1.06L6.94 8l-2.72 2.72a.75.75 0 1 0 1.06 1.06L8 9.06l2.72 2.72a.75.75 0 1 0 1.06-1.06L9.06 8l2.72-2.72a.75.75 0 0 0-1.06-1.06L8 6.94 5.28 4.22Z" />
              </svg>
            </button>
          </span>
        ))}
      </div>

      {/* Count and clear button */}
      <div className="flex items-center justify-between text-xs text-apple-gray-500 dark:text-apple-gray-400">
        <span>
          Viser {filteredCount} af {totalCount} {lawText}
        </span>
        <button
          type="button"
          onClick={onClearAll}
          disabled={disabled}
          className={`
            text-apple-blue hover:underline
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          `}
        >
          Ryd filter
        </button>
      </div>
    </div>
  );
}
