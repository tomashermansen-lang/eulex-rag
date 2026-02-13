/**
 * Tests for useDropdownKeyboard hook.
 *
 * TDD: Test keyboard navigation for dropdown menus.
 */

import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDropdownKeyboard } from '../useDropdownKeyboard';

describe('useDropdownKeyboard', () => {
  const defaultProps = {
    itemCount: 5,
    isOpen: true,
    onSelect: vi.fn(),
    onClose: vi.fn(),
  };

  describe('initial state', () => {
    it('returns focusedIndex as -1 initially', () => {
      const { result } = renderHook(() => useDropdownKeyboard(defaultProps));

      expect(result.current.focusedIndex).toBe(-1);
    });
  });

  describe('arrow navigation', () => {
    it('ArrowDown increments focusedIndex', () => {
      const { result } = renderHook(() => useDropdownKeyboard(defaultProps));

      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(result.current.focusedIndex).toBe(0);

      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(result.current.focusedIndex).toBe(1);
    });

    it('ArrowUp decrements focusedIndex', () => {
      const { result } = renderHook(() => useDropdownKeyboard(defaultProps));

      // First go down twice
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });
      expect(result.current.focusedIndex).toBe(1);

      // Now go up
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowUp',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(result.current.focusedIndex).toBe(0);
    });

    it('ArrowDown wraps at end', () => {
      const { result } = renderHook(() =>
        useDropdownKeyboard({ ...defaultProps, itemCount: 3 })
      );

      // Go to last item
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });
      expect(result.current.focusedIndex).toBe(2);

      // One more should wrap
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(result.current.focusedIndex).toBe(0);
    });

    it('ArrowUp wraps at start', () => {
      const { result } = renderHook(() =>
        useDropdownKeyboard({ ...defaultProps, itemCount: 3 })
      );

      // Start at -1, ArrowUp should go to last item
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowUp',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(result.current.focusedIndex).toBe(2);
    });
  });

  describe('selection', () => {
    it('Enter calls onSelect with focusedIndex', () => {
      const onSelect = vi.fn();
      const { result } = renderHook(() =>
        useDropdownKeyboard({ ...defaultProps, onSelect })
      );

      // Navigate to item 2
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });
      expect(result.current.focusedIndex).toBe(2);

      // Press Enter
      act(() => {
        result.current.handleKeyDown({
          key: 'Enter',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(onSelect).toHaveBeenCalledWith(2);
    });

    it('Space calls onSelect with focusedIndex', () => {
      const onSelect = vi.fn();
      const { result } = renderHook(() =>
        useDropdownKeyboard({ ...defaultProps, onSelect })
      );

      // Navigate to item 1
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      // Press Space
      act(() => {
        result.current.handleKeyDown({
          key: ' ',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(onSelect).toHaveBeenCalledWith(1);
    });

    it('Enter does not call onSelect when focusedIndex is -1', () => {
      const onSelect = vi.fn();
      const { result } = renderHook(() =>
        useDropdownKeyboard({ ...defaultProps, onSelect })
      );

      act(() => {
        result.current.handleKeyDown({
          key: 'Enter',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(onSelect).not.toHaveBeenCalled();
    });
  });

  describe('close', () => {
    it('Escape calls onClose', () => {
      const onClose = vi.fn();
      const { result } = renderHook(() =>
        useDropdownKeyboard({ ...defaultProps, onClose })
      );

      act(() => {
        result.current.handleKeyDown({
          key: 'Escape',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(onClose).toHaveBeenCalled();
    });
  });

  describe('jump navigation', () => {
    it('Home moves to first item', () => {
      const { result } = renderHook(() => useDropdownKeyboard(defaultProps));

      // First navigate somewhere
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });
      expect(result.current.focusedIndex).toBe(2);

      // Press Home
      act(() => {
        result.current.handleKeyDown({
          key: 'Home',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(result.current.focusedIndex).toBe(0);
    });

    it('End moves to last item', () => {
      const { result } = renderHook(() =>
        useDropdownKeyboard({ ...defaultProps, itemCount: 5 })
      );

      act(() => {
        result.current.handleKeyDown({
          key: 'End',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(result.current.focusedIndex).toBe(4);
    });
  });

  describe('accessibility', () => {
    it('returns menuProps with correct role', () => {
      const { result } = renderHook(() => useDropdownKeyboard(defaultProps));

      expect(result.current.menuProps.role).toBe('listbox');
    });

    it('returns menuProps with tabIndex', () => {
      const { result } = renderHook(() => useDropdownKeyboard(defaultProps));

      expect(result.current.menuProps.tabIndex).toBe(-1);
    });

    it('returns getItemProps with correct role', () => {
      const { result } = renderHook(() => useDropdownKeyboard(defaultProps));

      const itemProps = result.current.getItemProps(0);
      expect(itemProps.role).toBe('option');
    });

    it('returns getItemProps with aria-selected based on focusedIndex', () => {
      const { result } = renderHook(() => useDropdownKeyboard(defaultProps));

      // Navigate to item 1
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      expect(result.current.getItemProps(1)['aria-selected']).toBe(true);
      expect(result.current.getItemProps(0)['aria-selected']).toBe(false);
    });
  });

  describe('when closed', () => {
    it('resets focusedIndex when isOpen changes to false', () => {
      const { result, rerender } = renderHook(
        (props) => useDropdownKeyboard(props),
        { initialProps: defaultProps }
      );

      // Navigate to an item
      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });
      expect(result.current.focusedIndex).toBe(1);

      // Close dropdown
      rerender({ ...defaultProps, isOpen: false });

      expect(result.current.focusedIndex).toBe(-1);
    });

    it('does not handle keys when isOpen is false', () => {
      const onSelect = vi.fn();
      const { result } = renderHook(() =>
        useDropdownKeyboard({ ...defaultProps, isOpen: false, onSelect })
      );

      act(() => {
        result.current.handleKeyDown({
          key: 'ArrowDown',
          preventDefault: vi.fn(),
        } as unknown as React.KeyboardEvent);
      });

      // Should still be -1 since dropdown is closed
      expect(result.current.focusedIndex).toBe(-1);
    });
  });
});
