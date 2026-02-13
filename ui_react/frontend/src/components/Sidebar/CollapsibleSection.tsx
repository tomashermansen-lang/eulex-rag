/**
 * Collapsible section with animated expand/collapse.
 *
 * Single Responsibility: Render a collapsible section with header and content.
 * Follows Apple HIG disclosure pattern.
 */

import { useState, useId, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export interface CollapsibleSectionProps {
  /** Section title */
  title: string;
  /** Optional count badge */
  count?: number;
  /** Whether the section starts open */
  defaultOpen?: boolean;
  /** Callback when section is toggled */
  onToggle?: (isOpen: boolean) => void;
  /** Section content */
  children: ReactNode;
}

export function CollapsibleSection({
  title,
  count,
  defaultOpen = false,
  onToggle,
  children,
}: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const contentId = useId();

  const handleToggle = () => {
    const newState = !isOpen;
    setIsOpen(newState);
    onToggle?.(newState);
  };

  return (
    <div className="border-b border-apple-gray-100 dark:border-apple-gray-500">
      {/* Header button */}
      <button
        onClick={handleToggle}
        aria-expanded={isOpen}
        aria-controls={contentId}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-apple-gray-50 dark:hover:bg-apple-gray-600 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-apple-gray-700 dark:text-apple-gray-200">
            {title}
          </span>
          {count !== undefined && count > 0 && (
            <span className="px-1.5 py-0.5 text-xs font-medium bg-apple-blue-500 text-white rounded-full min-w-[20px] text-center">
              {count}
            </span>
          )}
        </div>

        {/* Chevron indicator */}
        <motion.svg
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="w-4 h-4 text-apple-gray-400"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </motion.svg>
      </button>

      {/* Content */}
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            id={contentId}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-3">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
