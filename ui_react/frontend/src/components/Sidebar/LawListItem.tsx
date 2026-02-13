/**
 * Individual law row component for the law list.
 *
 * Single Responsibility: Render a single law row with checkbox in configurable mode.
 */

import type { CorpusInfo } from '../../types';
import { Tooltip } from '../Common/Tooltip';

/** Checkbox display modes for different corpus scope settings */
export type CheckboxMode = 'radio' | 'checkbox' | 'disabled';

interface LawListItemProps {
  corpus: CorpusInfo;
  isSelected: boolean;
  checkboxMode: CheckboxMode;
  onClick: () => void;
  disabled?: boolean;
  /** Currently selected EuroVoc labels (for showing match count) */
  selectedLabels?: string[];
}

/**
 * Extract shortname from corpus - use uppercase ID.
 */
function getShortname(corpus: CorpusInfo): string {
  return corpus.id.toUpperCase();
}

/**
 * Get display name from corpus - fallback to uppercase ID.
 */
function getDisplayName(corpus: CorpusInfo): string {
  return corpus.name || corpus.id.toUpperCase();
}

export function LawListItem({
  corpus,
  isSelected,
  checkboxMode,
  onClick,
  disabled = false,
  selectedLabels = [],
}: LawListItemProps) {
  const shortname = getShortname(corpus);
  const displayName = getDisplayName(corpus);
  const tooltipContent = corpus.fullname || displayName;

  // Calculate label match counts
  const totalLabels = corpus.eurovoc_labels?.length ?? 0;
  const matchedLabelsList = totalLabels > 0
    ? corpus.eurovoc_labels!.filter((l) => selectedLabels.includes(l))
    : [];
  const matchedLabels = matchedLabelsList.length;

  // Smart tooltip: show matched labels first, then others
  const labelTooltip = (() => {
    if (totalLabels === 0) return 'Ingen emneord';
    if (matchedLabels > 0) {
      const otherLabels = corpus.eurovoc_labels!.filter((l) => !selectedLabels.includes(l));
      return `Matcher: ${matchedLabelsList.join(', ')}\n\nØvrige: ${otherLabels.join(', ')}`;
    }
    return corpus.eurovoc_labels!.join(', ');
  })();

  // In disabled mode, the row itself is disabled and doesn't respond to clicks
  const isDisabledMode = checkboxMode === 'disabled';
  const effectiveDisabled = disabled || isDisabledMode;

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (!effectiveDisabled) {
        onClick();
      }
    }
  };

  const handleCheckboxClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!effectiveDisabled) {
      onClick();
    }
  };

  // Checkbox styling based on mode
  const checkboxClasses = `
    h-4 w-4
    ${checkboxMode === 'radio' ? 'rounded-full' : 'rounded'}
    border-apple-gray-300 dark:border-apple-gray-600
    text-apple-blue
    focus:ring-apple-blue focus:ring-offset-0
    disabled:opacity-50
  `;

  // In disabled mode, checkbox is always checked
  const checkboxChecked = isDisabledMode ? true : isSelected;

  // Build aria-label for accessibility (full text regardless of truncation)
  const ariaLabel = displayName !== shortname
    ? `${shortname} — ${displayName}`
    : shortname;

  return (
    <Tooltip content={tooltipContent} delay={400} maxWidth={350} className="w-full">
      <button
        type="button"
        onClick={effectiveDisabled ? undefined : onClick}
        onKeyDown={handleKeyDown}
        disabled={effectiveDisabled}
        aria-label={ariaLabel}
        className={`
          w-full text-left py-1.5 px-3
          flex items-center gap-2
          transition-colors
          ${isSelected
            ? 'bg-apple-blue/5 dark:bg-apple-blue/10'
            : 'hover:bg-apple-gray-50 dark:hover:bg-apple-gray-700'
          }
          ${effectiveDisabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
          focus:outline-none focus:ring-2 focus:ring-apple-blue focus:ring-inset
          border-b border-apple-gray-100 dark:border-apple-gray-700 last:border-b-0
        `}
      >
        {/* Checkbox - always shown in unified interface */}
        <input
          type="checkbox"
          checked={checkboxChecked}
          onChange={() => {}} // Controlled by parent click
          onClick={handleCheckboxClick}
          disabled={effectiveDisabled}
          aria-label={`Vælg ${shortname}`}
          className={checkboxClasses}
        />

        {/* Text content - compact single-line layout */}
        <div className="flex-1 min-w-0 flex items-center">
          <span className="font-medium text-[13px] text-apple-gray-900 dark:text-white flex-shrink-0">
            {shortname}
          </span>
          {displayName !== shortname && (
            <>
              <span className="mx-1.5 text-apple-gray-300 dark:text-apple-gray-500">·</span>
              <span className="text-xs text-apple-gray-400 truncate">
                {displayName}
              </span>
            </>
          )}
        </div>

        {/* Label count tag */}
        {totalLabels > 0 && (
          <span
            title={labelTooltip}
            className={`
              flex-shrink-0 px-1.5 py-0.5 rounded text-[10px] font-medium
              ${matchedLabels > 0
                ? 'bg-apple-blue/10 text-apple-blue dark:bg-apple-blue/20'
                : 'bg-apple-gray-100 text-apple-gray-500 dark:bg-apple-gray-700 dark:text-apple-gray-400'
              }
            `}
          >
            {matchedLabels > 0 ? `${matchedLabels}/` : ''}{totalLabels} EuroVoc
          </span>
        )}
      </button>
    </Tooltip>
  );
}
