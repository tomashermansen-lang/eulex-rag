/**
 * DiscoveryBanner component.
 *
 * Single Responsibility: Inline banner above assistant message showing
 * which laws were discovered, with gate-appropriate styling.
 */

import type { DiscoveryMatch } from '../../types';

interface DiscoveryBannerProps {
  gate: 'AUTO' | 'SUGGEST' | 'ABSTAIN';
  matches: DiscoveryMatch[];
  /** Only show matches whose corpus_id is in this list. Shows all if omitted. */
  resolvedCorpora?: string[];
  /** Callback to lock discovered laws as search scope. */
  onLock?: () => void;
}

/** Gate-specific visual configuration. */
const GATE_STYLES = {
  AUTO: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-blue-200 dark:border-blue-800',
    text: 'text-blue-700 dark:text-blue-300',
    icon: (
      <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    label: 'Fundet i:',
  },
  SUGGEST: {
    bg: 'bg-amber-50 dark:bg-amber-900/20',
    border: 'border-amber-200 dark:border-amber-800',
    text: 'text-amber-700 dark:text-amber-300',
    icon: (
      <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    ),
    label: 'Muligt relevant:',
  },
} as const;

/**
 * Inline discovery banner shown above assistant messages.
 *
 * - AUTO: Blue info banner with discovered laws
 * - SUGGEST: Amber banner with confidence caveat
 * - ABSTAIN: Renders nothing (AbstentionCard handles this gate)
 */
export function DiscoveryBanner({
  gate,
  matches,
  resolvedCorpora,
  onLock,
}: DiscoveryBannerProps) {
  // ABSTAIN is handled by AbstentionCard, not this component
  if (gate === 'ABSTAIN') return null;

  // Filter to only show matches that were actually used
  const visibleMatches = resolvedCorpora
    ? matches.filter((m) => resolvedCorpora.includes(m.corpus_id))
    : matches;

  // Nothing to show
  if (!visibleMatches || visibleMatches.length === 0) return null;

  const style = GATE_STYLES[gate];

  return (
    <div
      className={`p-3 rounded-lg mb-2 border ${style.bg} ${style.border} ${style.text}`}
      role="status"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-2 flex-1 min-w-0">
          {style.icon}
          <div className="flex-1 min-w-0">
            {/* Primary text with law tokens */}
            <p className="text-sm font-medium">
              <span>{style.label}</span>{' '}
              {visibleMatches.map((match, i) => (
                <span key={match.corpus_id} className="inline-flex items-center gap-1">
                  {i > 0 && <span className="mx-1">&middot;</span>}
                  <span>{match.corpus_id.toUpperCase()}</span>
                </span>
              ))}
            </p>

            {/* Secondary text (SUGGEST only) */}
            {gate === 'SUGGEST' && (
              <p className="text-sm mt-1 opacity-80">
                Svaret kan være ufuldstændigt.
              </p>
            )}
          </div>
        </div>
        {onLock && (
          <button
            onClick={onLock}
            className="text-sm font-medium opacity-80 hover:opacity-100 transition-opacity flex-shrink-0"
          >
            Lås ↗
          </button>
        )}
      </div>
    </div>
  );
}
