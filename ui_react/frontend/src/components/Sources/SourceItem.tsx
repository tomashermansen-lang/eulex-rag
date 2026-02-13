/**
 * Individual source item component.
 *
 * Single Responsibility: Display a single source reference.
 */

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Reference } from '../../types';
import { getRefAnchorId, buildSourceUrl } from '../../utils/citations';

interface SourceItemProps {
  /** The reference to display */
  reference: Reference;
  /** Optional EUR-Lex base URL */
  sourceUrl?: string;
  /** Whether this source is initially expanded */
  defaultExpanded?: boolean;
  /** Message ID for unique anchor IDs */
  messageId?: string;
  /** Whether this source was cited in the answer */
  isCited?: boolean;
  /** Whether this source is currently selected (blue ring) */
  isSelected?: boolean;
}

/**
 * Display a single source reference with expandable content.
 */
export function SourceItem({
  reference,
  sourceUrl,
  defaultExpanded = false,
  messageId,
  isCited = false,
  isSelected,
}: SourceItemProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);
  const anchorId = getRefAnchorId(reference.idx, messageId);
  const fullUrl = sourceUrl ? buildSourceUrl(sourceUrl, reference) : null;

  // Listen for expand events from citation clicks
  useEffect(() => {
    const handleExpand = (e: CustomEvent<{ refId: string }>) => {
      if (e.detail.refId === anchorId) {
        setIsExpanded(true);
      }
    };

    window.addEventListener('expandSource' as any, handleExpand);
    return () => window.removeEventListener('expandSource' as any, handleExpand);
  }, [anchorId]);

  // Build class list for selection state
  const containerClass = [
    'source-expander',
    'reference-anchor',
    isSelected ? 'source-item-selected' : '',
  ].filter(Boolean).join(' ');

  return (
    <div
      id={anchorId}
      className={containerClass}
      aria-selected={isSelected !== undefined ? isSelected : undefined}
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="source-expander-header w-full text-left group"
        aria-expanded={isExpanded}
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span
            className={`font-medium truncate ${isCited ? 'text-apple-gray-700 dark:text-apple-gray-100' : 'text-apple-gray-500 dark:text-apple-gray-400'}`}
            title={`[${reference.idx}] ${reference.display}`}
          >
            [{reference.idx}] {reference.display}
          </span>
          {isCited && (
            <span className="shrink-0 text-[10px] text-apple-blue">
              citeret
            </span>
          )}
        </div>
        {/* Chevron - hidden by default, visible on hover (Apple style) */}
        <motion.svg
          animate={{ rotate: isExpanded ? 180 : 0 }}
          transition={{ duration: 0.15 }}
          className="w-4 h-4 text-apple-gray-400 opacity-0 group-hover:opacity-100 shrink-0 transition-opacity"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </motion.svg>
      </button>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="source-expander-content space-y-3">
              {fullUrl && (
                <a
                  href={fullUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-apple-gray-400 hover:text-apple-blue transition-colors"
                >
                  <svg
                    className="w-4 h-4"
                    fill="currentColor"
                    viewBox="0 0 16 16"
                  >
                    <path d="M4.715 6.542 3.343 7.914a3 3 0 1 0 4.243 4.243l1.828-1.829A3 3 0 0 0 8.586 5.5L8 6.086a1.002 1.002 0 0 0-.154.199 2 2 0 0 1 .861 3.337L6.88 11.45a2 2 0 1 1-2.83-2.83l.793-.792a4.018 4.018 0 0 1-.128-1.287z" />
                    <path d="M6.586 4.672A3 3 0 0 0 7.414 9.5l.775-.776a2 2 0 0 1-.896-3.346L9.12 3.55a2 2 0 1 1 2.83 2.83l-.793.792c.112.42.155.855.128 1.287l1.372-1.372a3 3 0 1 0-4.243-4.243L6.586 4.672z" />
                  </svg>
                  Åbn i EUR-Lex
                </a>
              )}

              <div>
                <p className="text-[11px] font-medium text-apple-gray-400 uppercase tracking-wide mb-1.5">
                  Kildetekst
                </p>
                <p className="text-apple-gray-600 dark:text-apple-gray-300 whitespace-pre-wrap text-sm leading-relaxed">
                  {reference.chunk_text}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
