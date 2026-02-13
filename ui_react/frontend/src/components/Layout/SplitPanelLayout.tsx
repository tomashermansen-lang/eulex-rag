/**
 * Split panel layout for chat and sources sidepanel.
 *
 * Single Responsibility: Provide a two-column grid layout with chat on left,
 * sources on right. Responsive: collapses to single column on narrow screens.
 */

import type { ReactNode } from 'react';
import { SourcesPanelProvider } from '../../contexts';

interface SplitPanelLayoutProps {
  /** Content for the chat panel (left side) */
  chatPanel: ReactNode;
  /** Content for the sources panel (right side) */
  sourcesPanel: ReactNode;
  /** Optional additional class name */
  className?: string;
}

/**
 * Two-column layout component for chat and sources.
 *
 * Uses CSS Grid with responsive breakpoint at 1400px.
 * On narrow screens, sources panel is hidden (use BottomSheet instead).
 */
export function SplitPanelLayout({
  chatPanel,
  sourcesPanel,
  className = '',
}: SplitPanelLayoutProps) {
  return (
    <SourcesPanelProvider>
      <div className={`split-panel-container ${className}`.trim()}>
        {/* Chat panel - main content area */}
        <section
          className="chat-panel"
          role="region"
          aria-label="Chat"
        >
          {chatPanel}
        </section>

        {/* Sources sidepanel - complementary content */}
        <aside
          className="sources-sidepanel"
          role="complementary"
          aria-label="Kilder"
        >
          {sourcesPanel}
        </aside>
      </div>
    </SourcesPanelProvider>
  );
}
