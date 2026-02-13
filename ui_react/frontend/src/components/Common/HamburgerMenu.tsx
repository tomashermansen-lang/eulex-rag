/**
 * Hamburger menu component for settings.
 *
 * Single Responsibility: Display a dropdown menu with app settings.
 */

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Settings } from '../../types';

interface HamburgerMenuProps {
  /** Current settings */
  settings: Settings;
  /** Callback to toggle debug mode */
  onToggleDebug: () => void;
  /** Callback to toggle dark mode */
  onToggleDarkMode: () => void;
  /** Callback to navigate to admin page */
  onNavigateToAdmin?: () => void;
}

/**
 * Hamburger menu with settings dropdown.
 */
export function HamburgerMenu({
  settings,
  onToggleDebug,
  onToggleDarkMode,
  onNavigateToAdmin,
}: HamburgerMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close menu on click outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="relative" ref={menuRef}>
      {/* Hamburger button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 rounded-lg hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 transition-colors"
        aria-label="Indstillinger"
        aria-expanded={isOpen}
      >
        <svg
          className="w-6 h-6 text-apple-gray-500 dark:text-apple-gray-300"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      </button>

      {/* Dropdown menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 mt-2 w-64 bg-white dark:bg-apple-gray-600 rounded-xl shadow-lg border border-apple-gray-100 dark:border-apple-gray-500 z-50 overflow-hidden"
          >
            <div className="p-4 border-b border-apple-gray-100 dark:border-apple-gray-500">
              <h3 className="text-sm font-semibold text-apple-gray-700 dark:text-white">
                Indstillinger
              </h3>
            </div>

            <div className="p-4 space-y-3">
              {/* Dark mode toggle - icon first, then label, toggle at far right */}
              <div className="flex items-center gap-3">
                {settings.darkMode ? (
                  <svg className="w-5 h-5 text-apple-gray-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 text-apple-gray-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                  </svg>
                )}
                <label className="text-sm text-apple-gray-600 dark:text-apple-gray-300 flex-1">
                  Mørk tilstand
                </label>
                <button
                  onClick={onToggleDarkMode}
                  className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                    settings.darkMode
                      ? 'bg-apple-blue'
                      : 'bg-apple-gray-200 dark:bg-apple-gray-500'
                  }`}
                  role="switch"
                  aria-checked={settings.darkMode}
                >
                  <motion.div
                    animate={{ x: settings.darkMode ? 22 : 2 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    className="absolute top-1 w-4 h-4 bg-white rounded-full shadow"
                  />
                </button>
              </div>

              {/* Debug mode toggle - icon first, then label, toggle at far right */}
              <div className="flex items-center gap-3">
                <svg className="w-5 h-5 text-apple-gray-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
                <label className="text-sm text-apple-gray-600 dark:text-apple-gray-300 flex-1">
                  Fejlsøgning
                </label>
                <button
                  onClick={onToggleDebug}
                  className={`relative w-11 h-6 rounded-full transition-colors flex-shrink-0 ${
                    settings.debugMode
                      ? 'bg-apple-blue'
                      : 'bg-apple-gray-200 dark:bg-apple-gray-500'
                  }`}
                  role="switch"
                  aria-checked={settings.debugMode}
                >
                  <motion.div
                    animate={{ x: settings.debugMode ? 22 : 2 }}
                    transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                    className="absolute top-1 w-4 h-4 bg-white rounded-full shadow"
                  />
                </button>
              </div>

              {/* Divider before admin link */}
              {onNavigateToAdmin && (
                <>
                  <div className="border-t border-apple-gray-100 dark:border-apple-gray-500 my-2" />
                  <button
                    onClick={() => {
                      onNavigateToAdmin();
                      setIsOpen(false);
                    }}
                    className="w-full flex items-center gap-3 py-2 text-left text-sm text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded-lg transition-colors -mx-1 px-1"
                  >
                    <svg className="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                    <span className="flex-1">Administrer lovgivning</span>
                  </button>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
