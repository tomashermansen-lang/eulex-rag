/**
 * Portal-based popover for selecting eval run mode.
 *
 * Shared between EvalDashboard (single-law) and CrossLawPanel (cross-law).
 */

import { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { RUN_MODE_LABELS, type RunMode } from './evalUtils';

export type { RunMode } from './evalUtils';

export function RunModePopover({
  isOpen,
  position,
  onClose,
  onSelect,
}: {
  isOpen: boolean;
  position: { x: number; y: number };
  onClose: () => void;
  onSelect: (mode: RunMode) => void;
}) {
  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = () => onClose();
    // Small delay to avoid immediate close from the same click
    const timer = setTimeout(() => {
      document.addEventListener('click', handleClickOutside);
    }, 10);
    return () => {
      clearTimeout(timer);
      document.removeEventListener('click', handleClickOutside);
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return createPortal(
    <div
      className="fixed bg-white dark:bg-apple-gray-600 border border-apple-gray-200 dark:border-apple-gray-500 rounded-lg shadow-lg z-[9999] min-w-[180px]"
      style={{
        left: position.x,
        top: position.y,
        transform: 'translateX(-100%)',
      }}
      onClick={(e) => e.stopPropagation()}
    >
      <div className="px-3 py-2 border-b border-apple-gray-100 dark:border-apple-gray-500">
        <span className="text-[10px] font-medium text-apple-gray-400 dark:text-apple-gray-500 uppercase tracking-wide">
          Kør eval
        </span>
      </div>
      {(Object.keys(RUN_MODE_LABELS) as RunMode[]).map((mode) => (
        <button
          key={mode}
          onClick={() => { onSelect(mode); onClose(); }}
          className="w-full px-3 py-2 text-left hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500 transition-colors"
        >
          <div className="text-xs font-medium text-apple-gray-700 dark:text-white">
            {RUN_MODE_LABELS[mode].label}
          </div>
          <div className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500">
            {RUN_MODE_LABELS[mode].description}
          </div>
        </button>
      ))}
    </div>,
    document.body
  );
}
