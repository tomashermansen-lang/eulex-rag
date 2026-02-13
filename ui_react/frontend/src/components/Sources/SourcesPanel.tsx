/**
 * Sources panel component.
 *
 * Single Responsibility: Display all sources for a message.
 */

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Reference } from '../../types';
import { SourceItem } from './SourceItem';
import { getRefAnchorId } from '../../utils/citations';

interface SourcesPanelProps {
  /** The references to display */
  references: Reference[];
  /** Optional EUR-Lex base URL */
  sourceUrl?: string;
  /** Citation indices that were used in the answer */
  citedIndices?: Set<number>;
  /** Message ID for unique anchor IDs */
  messageId?: string;
}

/**
 * Collapsible panel showing all source references.
 */
export function SourcesPanel({
  references,
  sourceUrl,
  citedIndices,
  messageId,
}: SourcesPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const detailsRef = useRef<HTMLDetailsElement>(null);

  // Get all reference IDs that belong to this panel
  const refIds = references.map((ref) => getRefAnchorId(ref.idx, messageId));

  // Listen for expand events from citation clicks
  useEffect(() => {
    const handleExpand = (e: CustomEvent<{ refId: string }>) => {
      // Only expand if the target reference is in this panel
      if (refIds.includes(e.detail.refId)) {
        setIsExpanded(true);
        if (detailsRef.current) {
          detailsRef.current.open = true;
        }
      }
    };

    window.addEventListener('expandSource' as any, handleExpand);
    return () => window.removeEventListener('expandSource' as any, handleExpand);
  }, [refIds]);

  if (references.length === 0) return null;

  return (
    <details
      ref={detailsRef}
      className="mt-4"
      open={isExpanded}
      onToggle={(e) => setIsExpanded((e.target as HTMLDetailsElement).open)}
    >
      <summary className="cursor-pointer list-none">
        <div className="flex items-center gap-2 text-apple-gray-400 hover:text-apple-gray-600 dark:hover:text-apple-gray-200 transition-colors">
          <motion.svg
            animate={{ rotate: isExpanded ? 90 : 0 }}
            transition={{ duration: 0.2 }}
            className="w-4 h-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </motion.svg>
          <span className="text-sm font-medium">
            Kilder ({references.length})
          </span>
        </div>
      </summary>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="mt-3 space-y-2"
            data-sources-panel
          >
            {references.map((ref) => {
              const isCited = citedIndices?.has(
                typeof ref.idx === 'string' ? parseInt(ref.idx, 10) : ref.idx
              );

              return (
                <SourceItem
                  key={ref.idx}
                  reference={ref}
                  sourceUrl={sourceUrl}
                  defaultExpanded={isCited}
                  messageId={messageId}
                  isCited={isCited}
                />
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </details>
  );
}
