/**
 * Accessible dropdown component with Apple HIG styling.
 *
 * Single Responsibility: Render dropdown with items, handle selection,
 * and provide full keyboard navigation and ARIA support.
 *
 * Uses:
 * - useDropdownState: open/close state, click-outside
 * - useDropdownKeyboard: keyboard navigation
 */

import { cloneElement, useCallback, useEffect, useRef, ReactElement } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropdownState, useDropdownKeyboard } from '../../hooks';

export interface DropdownItem {
  value: string;
  label: string;
  description?: string;
}

export interface DropdownProps {
  items: DropdownItem[];
  value: string;
  onChange: (value: string) => void;
  trigger: ReactElement;
}

export function Dropdown({ items, value, onChange, trigger }: DropdownProps) {
  const { isOpen, open, close, toggle, triggerRef, menuRef } = useDropdownState();
  const menuElementRef = useRef<HTMLDivElement>(null);

  const handleSelect = useCallback(
    (index: number) => {
      const item = items[index];
      if (item) {
        onChange(item.value);
        close();
      }
    },
    [items, onChange, close]
  );

  const { focusedIndex, handleKeyDown, menuProps, getItemProps } = useDropdownKeyboard({
    itemCount: items.length,
    isOpen,
    onSelect: handleSelect,
    onClose: close,
  });

  // Focus menu when opened
  useEffect(() => {
    if (isOpen && menuElementRef.current) {
      menuElementRef.current.focus();
    }
  }, [isOpen]);

  // Clone trigger and add required props
  const triggerWithProps = cloneElement(trigger, {
    onClick: toggle,
    'aria-haspopup': 'listbox',
    'aria-expanded': isOpen,
    ref: triggerRef,
  });

  return (
    <div className="relative">
      {triggerWithProps}

      <AnimatePresence>
        {isOpen && (
          <motion.div
            ref={(el) => {
              menuElementRef.current = el;
              // Also update menuRef for click-outside detection
              (menuRef as React.MutableRefObject<HTMLElement | null>).current = el;
            }}
            initial={{ opacity: 0, scale: 0.95, y: -5 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -5 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 mt-2 min-w-[200px] bg-white dark:bg-apple-gray-800 rounded-xl shadow-lg border border-apple-gray-200 dark:border-apple-gray-600 overflow-hidden z-50 outline-none"
            {...menuProps}
            onKeyDown={handleKeyDown}
          >
            <div className="py-1">
              {items.map((item, index) => {
                const isSelected = item.value === value;
                const isFocused = index === focusedIndex;
                const itemProps = getItemProps(index);

                return (
                  <div
                    key={item.value}
                    {...itemProps}
                    aria-selected={isSelected}
                    data-focused={isFocused}
                    onClick={() => handleSelect(index)}
                    className={`
                      px-4 py-2 cursor-pointer transition-colors
                      ${isFocused ? 'bg-apple-blue-500/10' : 'hover:bg-apple-gray-50 dark:hover:bg-apple-gray-700'}
                    `}
                  >
                    <div className="flex items-start gap-3">
                      {/* Checkmark column */}
                      <div className="w-5 flex-shrink-0 pt-0.5">
                        {isSelected && (
                          <svg
                            data-checkmark
                            className="w-5 h-5 text-apple-blue-500"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M5 13l4 4L19 7"
                            />
                          </svg>
                        )}
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-apple-gray-800 dark:text-apple-gray-100">
                          {item.label}
                        </div>
                        {item.description && (
                          <div className="text-xs text-apple-gray-500 dark:text-apple-gray-400 mt-0.5">
                            {item.description}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
