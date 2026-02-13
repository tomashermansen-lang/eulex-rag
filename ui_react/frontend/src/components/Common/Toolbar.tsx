/**
 * Application toolbar with title and action buttons.
 *
 * Single Responsibility: Render toolbar layout with title and actions.
 * Follows Apple HIG toolbar patterns.
 */

import type { ReactNode } from 'react';

export interface ToolbarProps {
  appTitle: string;
  /** Trailing action buttons (ExportButton, HamburgerMenu, etc.) */
  actions?: ReactNode;
}

export function Toolbar({ appTitle, actions }: ToolbarProps) {
  return (
    <div
      role="toolbar"
      aria-label="Application toolbar"
      className="h-12 px-4 flex items-center bg-white dark:bg-apple-gray-800 border-b border-apple-gray-200 dark:border-apple-gray-600"
    >
      {/* Leading section - App title */}
      <div data-section="leading" className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-apple-gray-800 dark:text-apple-gray-100">
          {appTitle}
        </h1>
      </div>

      {/* Center section - spacer */}
      <div data-section="center" className="flex-1" />

      {/* Trailing section - Action buttons */}
      <div data-section="trailing" className="flex items-center gap-2">
        {actions}
      </div>
    </div>
  );
}
