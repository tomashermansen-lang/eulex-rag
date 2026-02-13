/**
 * Token/pill component for displaying a selected law.
 *
 * Single Responsibility: Render a removable pill showing a law name.
 * Follows Apple HIG tag/token pattern.
 */

import { MouseEvent } from 'react';

export interface LawTokenProps {
  /** Law name to display */
  name: string;
  /** Optional law ID (passed to onRemove) */
  id?: string;
  /** Callback when remove button is clicked */
  onRemove: (id: string) => void;
}

export function LawToken({ name, id, onRemove }: LawTokenProps) {
  const handleRemove = (e: MouseEvent) => {
    e.stopPropagation();
    onRemove(id ?? name);
  };

  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-1 bg-apple-gray-100 dark:bg-apple-gray-600 text-apple-gray-700 dark:text-apple-gray-200 rounded-full text-sm max-w-full">
      <span className="truncate">{name}</span>
      <button
        onClick={handleRemove}
        aria-label={`Fjern ${name}`}
        className="flex-shrink-0 w-4 h-4 flex items-center justify-center rounded-full hover:bg-apple-gray-200 dark:hover:bg-apple-gray-500 transition-colors"
      >
        <svg
          className="w-3 h-3"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M6 18L18 6M6 6l12 12"
          />
        </svg>
      </button>
    </span>
  );
}
