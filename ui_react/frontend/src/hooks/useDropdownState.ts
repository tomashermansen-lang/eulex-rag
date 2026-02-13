/**
 * Hook for managing dropdown open/close state with click-outside detection.
 *
 * Responsibilities:
 * - Track open/close state
 * - Provide refs for trigger and menu elements
 * - Handle click-outside to close dropdown
 */

import { useState, useRef, useEffect, useCallback, RefObject } from 'react';

export interface UseDropdownStateReturn {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  triggerRef: RefObject<HTMLElement | null>;
  menuRef: RefObject<HTMLElement | null>;
}

export function useDropdownState(): UseDropdownStateReturn {
  const [isOpen, setIsOpen] = useState(false);
  const triggerRef = useRef<HTMLElement | null>(null);
  const menuRef = useRef<HTMLElement | null>(null);

  const open = useCallback(() => {
    setIsOpen(true);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
  }, []);

  const toggle = useCallback(() => {
    setIsOpen((prev) => !prev);
  }, []);

  // Click-outside detection
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;

      // Don't close if clicking on trigger
      if (triggerRef.current?.contains(target)) {
        return;
      }

      // Don't close if clicking inside menu
      if (menuRef.current?.contains(target)) {
        return;
      }

      // Close for any other click
      setIsOpen(false);
    };

    document.addEventListener('mousedown', handleClickOutside);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [isOpen]);

  return {
    isOpen,
    open,
    close,
    toggle,
    triggerRef,
    menuRef,
  };
}
