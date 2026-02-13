/**
 * Apple-style bottom sheet component.
 *
 * Single Responsibility: Provide a slide-up panel for mobile/narrow screens.
 * Used as responsive fallback when sidepanel doesn't fit.
 */

import { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ReactNode } from 'react';

interface BottomSheetProps {
  /** Whether the sheet is open */
  isOpen: boolean;
  /** Callback when sheet should close */
  onClose: () => void;
  /** Optional title for the sheet header */
  title?: string;
  /** Sheet content */
  children: ReactNode;
}

/**
 * Apple-style bottom sheet with backdrop and drag handle.
 *
 * Slides up from bottom with spring animation.
 * Closes on backdrop tap or handle click.
 */
export function BottomSheet({ isOpen, onClose, title, children }: BottomSheetProps) {
  // Lock body scroll when open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      return () => {
        document.body.style.overflow = '';
      };
    }
  }, [isOpen]);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            data-testid="bottom-sheet-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 bg-black/40 z-40"
            onClick={onClose}
          />

          {/* Sheet */}
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-label={title || 'Bottom sheet'}
            initial={{ y: '100%' }}
            animate={{ y: 0 }}
            exit={{ y: '100%' }}
            transition={{
              type: 'spring',
              stiffness: 300,
              damping: 30,
            }}
            className="bottom-sheet fixed bottom-0 left-0 right-0 z-50 bg-white dark:bg-apple-gray-600 rounded-t-3xl shadow-2xl max-h-[85vh] flex flex-col"
          >
            {/* Handle */}
            <button
              data-testid="bottom-sheet-handle"
              onClick={onClose}
              className="flex justify-center py-3 cursor-pointer"
              aria-label="Luk"
            >
              <div className="w-10 h-1 bg-apple-gray-300 dark:bg-apple-gray-400 rounded-full" />
            </button>

            {/* Header */}
            {title && (
              <div className="px-4 pb-3 border-b border-apple-gray-100 dark:border-apple-gray-500">
                <h2 className="text-lg font-semibold text-apple-gray-700 dark:text-white text-center">
                  {title}
                </h2>
              </div>
            )}

            {/* Content */}
            <div className="flex-1 overflow-y-auto overscroll-contain">
              {children}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
