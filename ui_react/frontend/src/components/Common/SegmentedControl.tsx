/**
 * Reusable segmented control component.
 *
 * Single Responsibility: Provide Apple HIG-compliant segmented controls.
 * Used for tabs, filters, and toggle selections across the UI.
 */

import type { ReactNode } from 'react';

interface SegmentOption<T extends string> {
  /** The value associated with this option */
  value: T;
  /** Display label - can be a string or React element (for icons) */
  label: ReactNode;
  /** Optional: Custom color when selected (for filter states like pass/fail/escalated) */
  selectedColor?: 'default' | 'green' | 'red' | 'amber';
  /** Optional tooltip shown on hover (native title attribute) */
  tooltip?: string;
}

interface SegmentedControlProps<T extends string> {
  /** Available options */
  options: SegmentOption<T>[];
  /** Currently selected value */
  value: T;
  /** Callback when selection changes */
  onChange: (value: T) => void;
  /** Size variant */
  size?: 'sm' | 'md';
  /** Whether segments should have equal width */
  equalWidth?: boolean;
  /** Additional className for the container */
  className?: string;
}

/**
 * Segmented control with Apple-style selection indicator.
 */
export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  size = 'md',
  equalWidth = false,
  className = '',
}: SegmentedControlProps<T>) {
  const sizeClasses = {
    sm: 'px-3 py-1 text-xs',
    md: 'px-4 py-2 text-sm',
  };

  const getSelectedClasses = (option: SegmentOption<T>) => {
    const baseSelected = 'bg-white dark:bg-apple-gray-500 shadow-sm';

    switch (option.selectedColor) {
      case 'green':
        return `${baseSelected} text-green-600 dark:text-green-400`;
      case 'red':
        return `${baseSelected} text-red-600 dark:text-red-400`;
      case 'amber':
        return `${baseSelected} text-amber-600 dark:text-amber-400`;
      default:
        return `${baseSelected} text-apple-gray-700 dark:text-white`;
    }
  };

  return (
    <div
      className={`inline-flex gap-1 p-1 bg-apple-gray-100 dark:bg-apple-gray-700 rounded-lg ${className}`}
      role="tablist"
    >
      {options.map((option) => {
        const isSelected = value === option.value;

        return (
          <button
            key={option.value}
            onClick={() => onChange(option.value)}
            role="tab"
            aria-selected={isSelected}
            title={option.tooltip}
            className={`
              ${sizeClasses[size]}
              ${equalWidth ? 'flex-1' : ''}
              font-medium rounded-md transition-all
              ${isSelected
                ? getSelectedClasses(option)
                : 'text-apple-gray-500 dark:text-apple-gray-400 hover:text-apple-gray-700 dark:hover:text-white'
              }
            `}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
