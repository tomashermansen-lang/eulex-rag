/**
 * Reusable search input component.
 *
 * Single Responsibility: Provide a consistent search input across the UI.
 * Based on the EvalDashboard search input style (user preference).
 */

interface SearchInputProps {
  /** Current search value */
  value: string;
  /** Callback when value changes */
  onChange: (value: string) => void;
  /** Placeholder text */
  placeholder?: string;
  /** Whether the input is disabled */
  disabled?: boolean;
  /** Additional className for the container */
  className?: string;
}

/**
 * Search input with magnifying glass icon and clear button.
 */
export function SearchInput({
  value,
  onChange,
  placeholder = 'Søg...',
  disabled = false,
  className = '',
}: SearchInputProps) {
  return (
    <div className={`relative m-[3px] ${className}`}>
      <svg
        className={`absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 ${
          disabled ? 'text-apple-gray-300 dark:text-apple-gray-500' : 'text-apple-gray-400'
        }`}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
        />
      </svg>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        className={`w-full pl-10 pr-10 py-2 text-sm rounded-lg border border-apple-gray-300 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-800 text-apple-gray-700 dark:text-white placeholder-apple-gray-400 focus:outline-none focus:border-transparent focus:shadow-[0_0_0_3px_#007AFF] transition-colors ${
          disabled ? 'opacity-60 cursor-not-allowed' : ''
        }`}
      />
      {value && !disabled && (
        <button
          onClick={() => onChange('')}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-apple-gray-400 hover:text-apple-gray-600 dark:hover:text-apple-gray-300 transition-colors"
          type="button"
          aria-label="Ryd søgning"
        >
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  );
}
