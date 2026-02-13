/**
 * Tests for useDropdownState hook.
 *
 * TDD: Test dropdown open/close state management and click-outside handling.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDropdownState } from '../useDropdownState';

describe('useDropdownState', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('initial state', () => {
    it('returns isOpen as false initially', () => {
      const { result } = renderHook(() => useDropdownState());

      expect(result.current.isOpen).toBe(false);
    });

    it('returns triggerRef', () => {
      const { result } = renderHook(() => useDropdownState());

      expect(result.current.triggerRef).toBeDefined();
      expect(result.current.triggerRef.current).toBeNull();
    });
  });

  describe('state control', () => {
    it('open() sets isOpen to true', () => {
      const { result } = renderHook(() => useDropdownState());

      act(() => {
        result.current.open();
      });

      expect(result.current.isOpen).toBe(true);
    });

    it('close() sets isOpen to false', () => {
      const { result } = renderHook(() => useDropdownState());

      act(() => {
        result.current.open();
      });
      expect(result.current.isOpen).toBe(true);

      act(() => {
        result.current.close();
      });
      expect(result.current.isOpen).toBe(false);
    });

    it('toggle() toggles isOpen state', () => {
      const { result } = renderHook(() => useDropdownState());

      expect(result.current.isOpen).toBe(false);

      act(() => {
        result.current.toggle();
      });
      expect(result.current.isOpen).toBe(true);

      act(() => {
        result.current.toggle();
      });
      expect(result.current.isOpen).toBe(false);
    });
  });

  describe('click outside', () => {
    it('click outside closes dropdown when open', () => {
      const { result } = renderHook(() => useDropdownState());

      // Create a mock trigger element
      const triggerElement = document.createElement('button');
      document.body.appendChild(triggerElement);

      // Set the ref
      Object.defineProperty(result.current.triggerRef, 'current', {
        value: triggerElement,
        writable: true,
      });

      // Open the dropdown
      act(() => {
        result.current.open();
      });
      expect(result.current.isOpen).toBe(true);

      // Click outside (on document body)
      act(() => {
        const event = new MouseEvent('mousedown', { bubbles: true });
        document.body.dispatchEvent(event);
      });

      expect(result.current.isOpen).toBe(false);

      // Cleanup
      document.body.removeChild(triggerElement);
    });

    it('does not close when clicking on trigger', () => {
      const { result } = renderHook(() => useDropdownState());

      // Create a mock trigger element
      const triggerElement = document.createElement('button');
      document.body.appendChild(triggerElement);

      // Set the ref
      Object.defineProperty(result.current.triggerRef, 'current', {
        value: triggerElement,
        writable: true,
      });

      // Open the dropdown
      act(() => {
        result.current.open();
      });
      expect(result.current.isOpen).toBe(true);

      // Click on trigger element
      act(() => {
        const event = new MouseEvent('mousedown', { bubbles: true });
        triggerElement.dispatchEvent(event);
      });

      // Should still be open
      expect(result.current.isOpen).toBe(true);

      // Cleanup
      document.body.removeChild(triggerElement);
    });

    it('does not close when dropdown is already closed', () => {
      const { result } = renderHook(() => useDropdownState());

      expect(result.current.isOpen).toBe(false);

      // Click outside - should not throw or change state
      act(() => {
        const event = new MouseEvent('mousedown', { bubbles: true });
        document.body.dispatchEvent(event);
      });

      expect(result.current.isOpen).toBe(false);
    });
  });

  describe('menuRef for click inside detection', () => {
    it('returns menuRef', () => {
      const { result } = renderHook(() => useDropdownState());

      expect(result.current.menuRef).toBeDefined();
      expect(result.current.menuRef.current).toBeNull();
    });

    it('does not close when clicking inside menu', () => {
      const { result } = renderHook(() => useDropdownState());

      // Create mock elements
      const triggerElement = document.createElement('button');
      const menuElement = document.createElement('div');
      document.body.appendChild(triggerElement);
      document.body.appendChild(menuElement);

      // Set the refs
      Object.defineProperty(result.current.triggerRef, 'current', {
        value: triggerElement,
        writable: true,
      });
      Object.defineProperty(result.current.menuRef, 'current', {
        value: menuElement,
        writable: true,
      });

      // Open the dropdown
      act(() => {
        result.current.open();
      });
      expect(result.current.isOpen).toBe(true);

      // Click inside menu
      act(() => {
        const event = new MouseEvent('mousedown', { bubbles: true });
        menuElement.dispatchEvent(event);
      });

      // Should still be open
      expect(result.current.isOpen).toBe(true);

      // Cleanup
      document.body.removeChild(triggerElement);
      document.body.removeChild(menuElement);
    });
  });
});
