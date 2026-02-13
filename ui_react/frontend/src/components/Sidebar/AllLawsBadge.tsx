/**
 * Summary badge component for "Alle" mode.
 *
 * Single Responsibility: Display a summary indicator showing all laws are selected.
 */

interface AllLawsBadgeProps {
  count: number;
}

export function AllLawsBadge({ count }: AllLawsBadgeProps) {
  return (
    <div
      className="
        flex items-center gap-2
        p-3 rounded-lg
        bg-green-50 dark:bg-green-900/20
        border border-green-200 dark:border-green-800
        text-green-700 dark:text-green-300
      "
    >
      {/* Checkmark icon */}
      <svg
        data-testid="checkmark-icon"
        className="h-5 w-5 flex-shrink-0"
        xmlns="http://www.w3.org/2000/svg"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
      <span className="text-sm font-medium">Alle {count} love valgt</span>
    </div>
  );
}
