/**
 * Export button with dropdown for format selection.
 *
 * Single Responsibility: Provide UI for exporting conversations.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ChatMessage } from '../../types';
import { exportAsMarkdown, exportAsPdf } from '../../services/export';

interface ExportButtonProps {
  messages: ChatMessage[];
  disabled?: boolean;
}

export function ExportButton({ messages, disabled = false }: ExportButtonProps) {
  const [showMenu, setShowMenu] = useState(false);
  const [exporting, setExporting] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowMenu(false);
      }
    }

    if (showMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showMenu]);

  const handleExport = useCallback(
    async (format: 'markdown' | 'pdf') => {
      if (exporting || messages.length === 0) return;

      setExporting(true);
      setShowMenu(false);

      try {
        if (format === 'markdown') {
          exportAsMarkdown(messages);
        } else {
          await exportAsPdf(messages);
        }
      } catch (error) {
        console.error('Export failed:', error);
      } finally {
        setExporting(false);
      }
    },
    [messages, exporting]
  );

  const isDisabled = disabled || messages.length === 0 || exporting;

  return (
    <div className="relative" ref={menuRef}>
      <button
        onClick={() => setShowMenu(!showMenu)}
        disabled={isDisabled}
        className={`
          flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg
          transition-colors duration-200
          ${
            isDisabled
              ? 'text-apple-gray-300 dark:text-apple-gray-500 cursor-not-allowed'
              : 'text-apple-gray-500 dark:text-apple-gray-400 hover:text-apple-gray-700 dark:hover:text-white hover:bg-apple-gray-100 dark:hover:bg-apple-gray-700'
          }
        `}
        title={messages.length === 0 ? 'Ingen beskeder at eksportere' : 'Eksporter samtale'}
      >
        {/* Download icon */}
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
          />
        </svg>
        <span>{exporting ? 'Eksporterer...' : 'Eksporter'}</span>
      </button>

      <AnimatePresence>
        {showMenu && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -5 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -5 }}
            transition={{ duration: 0.1 }}
            className="absolute right-0 mt-2 w-40 bg-white dark:bg-apple-gray-800 rounded-xl shadow-lg border border-apple-gray-200 dark:border-apple-gray-600 overflow-hidden z-50"
          >
            <div className="py-1">
              <button
                onClick={() => handleExport('markdown')}
                className="w-full px-4 py-2 text-left text-sm text-apple-gray-700 dark:text-apple-gray-200 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-700 transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Markdown
              </button>
              <button
                onClick={() => handleExport('pdf')}
                className="w-full px-4 py-2 text-left text-sm text-apple-gray-700 dark:text-apple-gray-200 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-700 transition-colors flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
                PDF
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
