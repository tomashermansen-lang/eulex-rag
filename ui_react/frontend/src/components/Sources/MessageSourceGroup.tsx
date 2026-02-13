/**
 * Source group for a single message in the sidepanel.
 *
 * Single Responsibility: Display all sources belonging to one message,
 * with cited sources shown by default and non-cited hidden (progressive disclosure).
 */

import { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Reference } from '../../types';
import { SourceItem } from './SourceItem';
import { useSourcesPanel } from '../../contexts';
import { getRefAnchorId } from '../../utils/citations';

interface MessageSourceGroupProps {
  /** ID of the message these sources belong to */
  messageId: string;
  /** References to display */
  references: Reference[];
  /** Indices of sources that were cited in the answer */
  citedIndices: Set<number>;
  /** Optional EUR-Lex base URL */
  sourceUrl?: string;
}

/**
 * Group of sources for a single message.
 *
 * Apple design: Progressive disclosure - show only cited sources by default,
 * with a toggle to reveal the full context.
 */
export function MessageSourceGroup({
  messageId,
  references,
  citedIndices,
  sourceUrl,
}: MessageSourceGroupProps) {
  const [showNonCited, setShowNonCited] = useState(false);
  const { selectedSourceId } = useSourcesPanel();

  // Separate cited and non-cited sources
  const { citedSources, nonCitedSources } = useMemo(() => {
    const cited: Reference[] = [];
    const nonCited: Reference[] = [];

    references.forEach((ref) => {
      const refIdx = typeof ref.idx === 'string' ? parseInt(ref.idx, 10) : ref.idx;
      if (citedIndices.has(refIdx)) {
        cited.push(ref);
      } else {
        nonCited.push(ref);
      }
    });

    return { citedSources: cited, nonCitedSources: nonCited };
  }, [references, citedIndices]);

  if (references.length === 0) {
    return null;
  }

  return (
    <div
      className="message-source-group"
      data-message-id={messageId}
    >
      {/* Cited sources - always visible */}
      {citedSources.length > 0 && (
        <div className="space-y-2">
          {citedSources.map((ref) => {
            const anchorId = getRefAnchorId(ref.idx, messageId);
            return (
              <SourceItem
                key={ref.idx}
                reference={ref}
                sourceUrl={sourceUrl}
                defaultExpanded={true}
                messageId={messageId}
                isCited={true}
                isSelected={selectedSourceId === anchorId}
              />
            );
          })}
        </div>
      )}

      {/* Non-cited sources toggle - progressive disclosure */}
      {nonCitedSources.length > 0 && (
        <div className="mt-3">
          <button
            onClick={() => setShowNonCited(!showNonCited)}
            className="w-full flex items-center gap-2 py-2 px-3 text-xs text-apple-gray-400 hover:text-apple-gray-600 dark:hover:text-apple-gray-300 hover:bg-apple-gray-100/50 dark:hover:bg-apple-gray-600/50 rounded-lg transition-colors"
          >
            <motion.svg
              animate={{ rotate: showNonCited ? 90 : 0 }}
              transition={{ duration: 0.15 }}
              className="w-3.5 h-3.5"
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
            <span>
              {showNonCited ? 'Skjul' : 'Vis'} {nonCitedSources.length} andre kilder fra søgningen
            </span>
          </button>

          <AnimatePresence>
            {showNonCited && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="space-y-2 pt-2">
                  {nonCitedSources.map((ref) => {
                    const anchorId = getRefAnchorId(ref.idx, messageId);
                    return (
                      <SourceItem
                        key={ref.idx}
                        reference={ref}
                        sourceUrl={sourceUrl}
                        defaultExpanded={false}
                        messageId={messageId}
                        isCited={false}
                        isSelected={selectedSourceId === anchorId}
                      />
                    );
                  })}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Edge case: no cited sources - show toggle immediately */}
      {citedSources.length === 0 && nonCitedSources.length > 0 && (
        <p className="text-xs text-apple-gray-400 mb-2">
          Ingen direkte citerede kilder i dette svar
        </p>
      )}
    </div>
  );
}
