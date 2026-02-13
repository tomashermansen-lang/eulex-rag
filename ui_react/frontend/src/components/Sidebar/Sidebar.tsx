/**
 * Sidebar component.
 *
 * Single Responsibility: Display app settings and controls.
 */

import type { Settings, CorpusInfo, CorpusScope, DiscoveryMatch } from '../../types';
import { SegmentedControl } from '../Common/SegmentedControl';
import { LawSelectorPanel } from './LawSelectorPanel';
import { DiscoveryStatusPanel } from './DiscoveryStatusPanel';

/**
 * Scope option labels with tooltip descriptions.
 */
const SCOPE_OPTIONS: { value: CorpusScope; label: string; tooltip: string }[] = [
  { value: 'discover', label: 'AI', tooltip: 'AI identificerer automatisk relevante love ud fra dit spørgsmål' },
  { value: 'single', label: 'Enkelt', tooltip: 'Søg kun i én bestemt lov' },
  { value: 'explicit', label: 'Udvalgte', tooltip: 'Søg på tværs af love du selv har valgt' },
  { value: 'all', label: 'Alle', tooltip: 'Søg i al tilgængelig lovgivning' },
];

/**
 * Descriptions shown below the scope selector, matching the profile pattern.
 */
const SCOPE_DESCRIPTIONS: Record<CorpusScope, string> = {
  discover: 'AI identificerer automatisk relevante love ud fra dit spørgsmål.',
  single: 'Søg kun i én bestemt lov.',
  explicit: 'Søg på tværs af love du selv har valgt.',
  all: 'Søg i al tilgængelig lovgivning.',
};

interface SidebarProps {
  /** Current settings */
  settings: Settings;
  /** Available corpora */
  corpora: CorpusInfo[];
  /** Callback to update user profile */
  onProfileChange: (profile: 'LEGAL' | 'ENGINEERING') => void;
  /** Callback to update corpus scope */
  onCorpusScopeChange: (scope: CorpusScope) => void;
  /** Callback to update target corpora */
  onTargetCorporaChange: (corpora: string[]) => void;
  /** Callback to clear chat */
  onClearChat: () => void;
  /** Whether there are messages to clear */
  hasMessages: boolean;
  /** Whether selection is disabled (e.g., during streaming) */
  disabled?: boolean;
  /** Discovery matches from latest query (for AI mode) */
  discoveryMatches?: DiscoveryMatch[];
  /** Whether discovery is currently loading */
  discoveryLoading?: boolean;
  /** Callback to lock user-selected discovered laws as search scope */
  onLock?: (corporaIds: string[]) => void;
}

/**
 * Settings sidebar with profile, scope, and law selection controls.
 */
export function Sidebar({
  settings,
  corpora,
  onProfileChange,
  onCorpusScopeChange,
  onTargetCorporaChange,
  onClearChat,
  hasMessages,
  disabled = false,
  discoveryMatches,
  discoveryLoading = false,
  onLock,
}: SidebarProps) {
  return (
    <div className="flex flex-col h-full">
      {/* Settings */}
      <div className="flex-1 p-4 pt-6 flex flex-col overflow-hidden">
        {/* Profile selector (moved to top - most fundamental choice) */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-apple-gray-500 dark:text-apple-gray-300">
            Profil
          </label>
          <SegmentedControl
            options={[
              { value: 'LEGAL', label: 'Juridisk' },
              { value: 'ENGINEERING', label: 'Teknisk' },
            ]}
            value={settings.userProfile}
            onChange={onProfileChange}
            equalWidth
            className="w-full"
          />
          <p className="text-xs text-apple-gray-400 dark:text-apple-gray-400 mt-1 leading-tight">
            {settings.userProfile === 'LEGAL'
              ? 'Fokus på juridisk fortolkning, rettigheder, pligter og retsgrundlag.'
              : 'Fokus på teknisk implementering, krav og compliance-processer.'}
          </p>
        </div>

        {/* Corpus scope selector */}
        <div className="mt-6 space-y-2">
          <label className="text-sm font-medium text-apple-gray-500 dark:text-apple-gray-300">
            Søgeområde
          </label>
          <SegmentedControl
            options={SCOPE_OPTIONS}
            value={settings.corpusScope}
            onChange={onCorpusScopeChange}
            equalWidth
            className="w-full"
          />
          <p className="text-xs text-apple-gray-400 dark:text-apple-gray-400 mt-1 leading-tight">
            {SCOPE_DESCRIPTIONS[settings.corpusScope]}
          </p>
        </div>

        {/* Law selector / Discovery panel (fills remaining space) */}
        <div className="mt-4 flex-1 min-h-0">
          {settings.corpusScope === 'discover' ? (
            <DiscoveryStatusPanel
              discoveries={discoveryMatches}
              isLoading={discoveryLoading}
              corpora={corpora}
              onLock={onLock}
            />
          ) : (
            <LawSelectorPanel
              corpusScope={settings.corpusScope}
              corpora={corpora}
              targetCorpora={settings.targetCorpora}
              onTargetCorporaChange={onTargetCorporaChange}
              disabled={disabled}
            />
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-apple-gray-100 dark:border-apple-gray-500 space-y-3">
        {hasMessages && (
          <button
            onClick={onClearChat}
            className="btn-secondary w-full flex items-center justify-center gap-2"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
            Ryd chat
          </button>
        )}

        <p className="text-xs text-apple-gray-400 text-center">
          EuLex Legal Assistant v1.0
        </p>
      </div>
    </div>
  );
}
