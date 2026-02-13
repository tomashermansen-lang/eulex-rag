/**
 * DiscoveryStatusPanel component.
 *
 * Single Responsibility: Show discovery idle/loading/results states in sidebar.
 * Replaces LawSelectorPanel when corpus_scope === 'discover'.
 *
 * Results are shown as a checkbox list matching LawSelectorPanel layout,
 * enriched with confidence dots and scores. Users select which laws to lock.
 */

import { useState, useEffect } from 'react';
import type { DiscoveryMatch, CorpusInfo } from '../../types';
import { Tooltip } from '../Common/Tooltip';

interface DiscoveryStatusPanelProps {
  discoveries?: DiscoveryMatch[];
  isLoading?: boolean;
  error?: string;
  corpora?: CorpusInfo[];
  /** Callback to lock user-selected corpora as search scope */
  onLock?: (corporaIds: string[]) => void;
}

/** AUTO-tier threshold — items at or above are pre-checked */
const AUTO_THRESHOLD = 0.75;

/** Map confidence to human-readable label with tag color classes and explanatory tooltip */
function getConfidenceLabel(confidence: number): { text: string; colorClass: string; tooltip: string } {
  const pct = `${(confidence * 100).toFixed(0)}%`;
  if (confidence >= 0.75) {
    return {
      text: 'Høj',
      colorClass: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
      tooltip: `Relevans: ${pct} — Høj sandsynlighed for at denne lov er relevant for dit spørgsmål`,
    };
  } else if (confidence >= 0.50) {
    return {
      text: 'Mulig',
      colorClass: 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400',
      tooltip: `Relevans: ${pct} — Mulig relevans, gennemgå om loven passer til dit spørgsmål`,
    };
  }
  return {
    text: 'Lav',
    colorClass: 'bg-apple-gray-100 text-apple-gray-500 dark:bg-apple-gray-700 dark:text-apple-gray-400',
    tooltip: `Relevans: ${pct} — Lav sandsynlighed, medtaget for fuldstændighed`,
  };
}

/**
 * Sidebar panel showing AI discovery state.
 *
 * States:
 * - Idle: prompt text
 * - Loading: shimmer animation
 * - Results: discovered laws with checkboxes, confidence dots, and scores
 * - Error: warning message
 */
export function DiscoveryStatusPanel({
  discoveries,
  isLoading = false,
  error,
  corpora,
  onLock,
}: DiscoveryStatusPanelProps) {
  const [checkedIds, setCheckedIds] = useState<Set<string>>(new Set());

  // Pre-check AUTO-tier items when discoveries change
  useEffect(() => {
    if (!discoveries) {
      setCheckedIds(new Set());
      return;
    }
    const autoTier = discoveries
      .filter((m) => m.confidence >= AUTO_THRESHOLD)
      .map((m) => m.corpus_id);
    setCheckedIds(new Set(autoTier));
  }, [discoveries]);

  const toggleCheck = (corpusId: string) => {
    setCheckedIds((prev) => {
      const next = new Set(prev);
      if (next.has(corpusId)) {
        next.delete(corpusId);
      } else {
        next.add(corpusId);
      }
      return next;
    });
  };

  const handleLock = () => {
    if (!discoveries || !onLock) return;
    // Preserve discovery order for checked items
    const selected = discoveries
      .filter((m) => checkedIds.has(m.corpus_id))
      .map((m) => m.corpus_id);
    onLock(selected);
  };

  // Error state
  if (error) {
    return (
      <div className="p-3 rounded-lg border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-900/20">
        <p className="text-sm text-amber-700 dark:text-amber-300">
          {error}
        </p>
      </div>
    );
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="p-3 rounded-lg border border-apple-gray-200 dark:border-apple-gray-600 bg-white dark:bg-apple-gray-800 space-y-2">
        <div className="h-4 bg-apple-gray-100 dark:bg-apple-gray-700 rounded animate-pulse" />
        <div className="h-4 bg-apple-gray-100 dark:bg-apple-gray-700 rounded animate-pulse w-3/4" />
      </div>
    );
  }

  // Results state
  if (discoveries && discoveries.length > 0) {
    const hasChecked = checkedIds.size > 0;

    return (
      <div className="flex flex-col gap-3">
        <p className="text-sm font-medium text-apple-gray-500 dark:text-apple-gray-300">
          Fundet lovgivning
        </p>

        {/* Law list — matches LawSelectorPanel layout */}
        <div className="border border-apple-gray-200 dark:border-apple-gray-600 rounded-lg bg-white dark:bg-apple-gray-800 overflow-hidden">
          {discoveries.map((match) => {
            const corpus = corpora?.find((c) => c.id === match.corpus_id);
            const displayName = corpus?.name || match.display_name || match.corpus_id;
            const fullname = corpus?.fullname || displayName;
            const shortname = match.corpus_id.toUpperCase();
            const isChecked = checkedIds.has(match.corpus_id);
            const eurovocLabels = corpus?.eurovoc_labels ?? [];
            const eurovocCount = eurovocLabels.length;
            const label = getConfidenceLabel(match.confidence);

            return (
              <Tooltip key={match.corpus_id} content={fullname} delay={400} maxWidth={350} className="w-full">
                <button
                  type="button"
                  onClick={() => toggleCheck(match.corpus_id)}
                  className={`
                    w-full text-left py-1.5 px-3
                    flex items-center gap-2
                    transition-colors cursor-pointer
                    ${isChecked
                      ? 'bg-apple-blue/5 dark:bg-apple-blue/10'
                      : 'hover:bg-apple-gray-50 dark:hover:bg-apple-gray-700'
                    }
                    focus:outline-none focus:ring-2 focus:ring-apple-blue focus:ring-inset
                    border-b border-apple-gray-100 dark:border-apple-gray-700 last:border-b-0
                  `}
                >
                  {/* Checkbox */}
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => toggleCheck(match.corpus_id)}
                    onClick={(e) => e.stopPropagation()}
                    aria-label={`Vælg ${shortname}`}
                    className="h-4 w-4 rounded border-apple-gray-300 dark:border-apple-gray-600 text-apple-blue focus:ring-apple-blue focus:ring-offset-0"
                  />

                  {/* Text content — SHORTNAME · displayname */}
                  <div className="flex-1 min-w-0 flex items-center">
                    <span className="font-medium text-[13px] text-apple-gray-900 dark:text-white flex-shrink-0">
                      {shortname}
                    </span>
                    {displayName !== shortname && (
                      <>
                        <span className="mx-1.5 text-apple-gray-300 dark:text-apple-gray-500">&middot;</span>
                        <span className="text-xs text-apple-gray-400 truncate">
                          {displayName}
                        </span>
                      </>
                    )}
                  </div>

                  {/* EuroVoc badge — fixed width for column alignment */}
                  {eurovocCount > 0 ? (
                    <span
                      className="flex-shrink-0 w-[4.5rem] text-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-apple-gray-100 text-apple-gray-500 dark:bg-apple-gray-700 dark:text-apple-gray-400"
                      title={eurovocLabels.join(', ')}
                    >
                      {eurovocCount} EuroVoc
                    </span>
                  ) : null}

                  {/* Confidence tag — fixed width for column alignment */}
                  <span
                    className={`flex-shrink-0 w-[3.25rem] text-center px-1.5 py-0.5 rounded text-[10px] font-medium ${label.colorClass}`}
                    title={label.tooltip}
                  >
                    {label.text}
                  </span>
                </button>
              </Tooltip>
            );
          })}
        </div>

        {/* Lock button */}
        {onLock && (
          <button
            onClick={handleLock}
            disabled={!hasChecked}
            className="btn-secondary w-full"
            aria-label="Brug fundne love som søgeområde"
          >
            Brug disse love
          </button>
        )}
      </div>
    );
  }

  // Idle state (default)
  return (
    <div className="p-3 rounded-lg bg-apple-gray-50 dark:bg-apple-gray-700">
      <div className="flex items-start gap-2">
        <svg className="w-4 h-4 text-apple-gray-300 dark:text-apple-gray-500 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <p className="text-sm text-apple-gray-400 dark:text-apple-gray-400">
          Relevante love identificeres automatisk n&aring;r du stiller et sp&oslash;rgsm&aring;l i chatten
        </p>
      </div>
    </div>
  );
}
