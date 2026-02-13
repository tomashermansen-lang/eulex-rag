/**
 * AbstentionCard component.
 *
 * Single Responsibility: Show abstention message with candidate pills
 * and action buttons when discovery cannot identify relevant legislation.
 */

import type { DiscoveryMatch } from '../../types';

interface AbstentionCardProps {
  candidates: DiscoveryMatch[];
  onSelectManual: (corpora: string[]) => void;
  onRephrase: () => void;
}

/**
 * Full-width card shown in chat when discovery abstains.
 * Replaces the normal assistant message for ABSTAIN gate.
 *
 * Provides:
 * - Clear abstention headline
 * - Candidate pills with muted scores
 * - "Vælg manuelt" and "Omformuler" action buttons
 */
export function AbstentionCard({
  candidates,
  onSelectManual,
  onRephrase,
}: AbstentionCardProps) {
  const handleSelectManual = () => {
    onSelectManual(candidates.map((c) => c.corpus_id));
  };

  return (
    <div
      className="chat-bubble-assistant"
      role="alert"
    >
      {/* Headline */}
      <p className="text-base font-semibold text-apple-gray-700 dark:text-apple-gray-100">
        Kan ikke identificere relevant lovgivning med sikkerhed.
      </p>

      {/* Candidate pills */}
      {candidates.length > 0 && (
        <div className="mt-3">
          <p className="text-sm text-apple-gray-500 dark:text-apple-gray-300 mb-2">
            Mulige love:
          </p>
          <div className="flex flex-wrap gap-2">
            {candidates.map((c) => (
              <span
                key={c.corpus_id}
                className="inline-flex items-center gap-1.5 px-2 py-1 bg-apple-gray-100 dark:bg-apple-gray-600 rounded-full text-sm text-apple-gray-600 dark:text-apple-gray-200"
              >
                <span>{c.display_name || c.corpus_id}</span>
                <span className="text-xs text-apple-gray-400 tabular-nums">
                  {Math.round(c.confidence * 100)}%
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="mt-4 flex gap-3">
        <button
          onClick={handleSelectManual}
          className="py-2 px-4 text-sm font-medium text-apple-blue hover:text-apple-blue-hover border border-apple-gray-200 dark:border-apple-gray-500 rounded-lg transition-colors"
        >
          Vælg manuelt
        </button>
        <button
          onClick={onRephrase}
          className="py-2 px-4 text-sm font-medium text-apple-gray-500 dark:text-apple-gray-300 hover:text-apple-gray-700 dark:hover:text-apple-gray-100 transition-colors"
        >
          Omformuler ↩
        </button>
      </div>
    </div>
  );
}
