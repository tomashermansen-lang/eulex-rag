/**
 * Hook for keyboard navigation in dropdown menus.
 *
 * Responsibilities:
 * - Track focused item index
 * - Handle arrow key navigation with wrapping
 * - Handle selection (Enter/Space)
 * - Handle close (Escape)
 * - Handle jump navigation (Home/End)
 * - Provide accessibility props
 */

import { useState, useCallback, useEffect } from 'react';

export interface UseDropdownKeyboardProps {
  itemCount: number;
  isOpen: boolean;
  onSelect: (index: number) => void;
  onClose: () => void;
}

export interface MenuProps {
  role: 'listbox';
  tabIndex: number;
  onKeyDown: (event: React.KeyboardEvent) => void;
}

export interface ItemProps {
  role: 'option';
  'aria-selected': boolean;
}

export interface UseDropdownKeyboardReturn {
  focusedIndex: number;
  handleKeyDown: (event: React.KeyboardEvent) => void;
  menuProps: MenuProps;
  getItemProps: (index: number) => ItemProps;
}

export function useDropdownKeyboard({
  itemCount,
  isOpen,
  onSelect,
  onClose,
}: UseDropdownKeyboardProps): UseDropdownKeyboardReturn {
  const [focusedIndex, setFocusedIndex] = useState(-1);

  // Reset focused index when dropdown closes
  useEffect(() => {
    if (!isOpen) {
      setFocusedIndex(-1);
    }
  }, [isOpen]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent) => {
      if (!isOpen) return;

      switch (event.key) {
        case 'ArrowDown':
          event.preventDefault();
          setFocusedIndex((prev) => {
            if (prev >= itemCount - 1) return 0;
            return prev + 1;
          });
          break;

        case 'ArrowUp':
          event.preventDefault();
          setFocusedIndex((prev) => {
            if (prev <= 0) return itemCount - 1;
            return prev - 1;
          });
          break;

        case 'Home':
          event.preventDefault();
          setFocusedIndex(0);
          break;

        case 'End':
          event.preventDefault();
          setFocusedIndex(itemCount - 1);
          break;

        case 'Enter':
        case ' ':
          event.preventDefault();
          if (focusedIndex >= 0) {
            onSelect(focusedIndex);
          }
          break;

        case 'Escape':
          event.preventDefault();
          onClose();
          break;
      }
    },
    [isOpen, itemCount, focusedIndex, onSelect, onClose]
  );

  const menuProps: MenuProps = {
    role: 'listbox',
    tabIndex: -1,
    onKeyDown: handleKeyDown,
  };

  const getItemProps = useCallback(
    (index: number): ItemProps => ({
      role: 'option',
      'aria-selected': index === focusedIndex,
    }),
    [focusedIndex]
  );

  return {
    focusedIndex,
    handleKeyDown,
    menuProps,
    getItemProps,
  };
}
