/**
 * Tests for SourcesPanelContext.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { ReactNode } from 'react';
import {
  SourcesPanelProvider,
  useSourcesPanel,
} from '../SourcesPanelContext';

// Wrapper for hooks that need the context
function wrapper({ children }: { children: ReactNode }) {
  return <SourcesPanelProvider>{children}</SourcesPanelProvider>;
}

describe('SourcesPanelContext', () => {
  describe('highlightedSourceId', () => {
    it('is null initially', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      expect(result.current.highlightedSourceId).toBeNull();
    });

    it('can be set via highlightSource', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.highlightSource('ref-msg1-1', 'msg1');
      });

      expect(result.current.highlightedSourceId).toBe('ref-msg1-1');
    });

    it('clears after timeout', async () => {
      vi.useFakeTimers();
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.highlightSource('ref-msg1-1', 'msg1');
      });

      expect(result.current.highlightedSourceId).toBe('ref-msg1-1');

      // Advance timers by 2 seconds (highlight duration)
      act(() => {
        vi.advanceTimersByTime(2000);
      });

      expect(result.current.highlightedSourceId).toBeNull();

      vi.useRealTimers();
    });
  });

  describe('activeMessageId', () => {
    it('is null initially', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      expect(result.current.activeMessageId).toBeNull();
    });

    it('updates when highlightSource is called', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.highlightSource('ref-msg2-3', 'msg2');
      });

      expect(result.current.activeMessageId).toBe('msg2');
    });

    it('can be set directly via setActiveMessageId', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.setActiveMessageId('msg3');
      });

      expect(result.current.activeMessageId).toBe('msg3');
    });
  });

  describe('expandedSourceIds', () => {
    it('is empty initially', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      expect(result.current.expandedSourceIds.size).toBe(0);
    });

    it('expands source when highlightSource is called', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.highlightSource('ref-msg1-1', 'msg1');
      });

      expect(result.current.expandedSourceIds.has('ref-msg1-1')).toBe(true);
    });

    it('toggles source expanded state', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      // First toggle - expand
      act(() => {
        result.current.toggleSourceExpanded('ref-msg1-2');
      });
      expect(result.current.expandedSourceIds.has('ref-msg1-2')).toBe(true);

      // Second toggle - collapse
      act(() => {
        result.current.toggleSourceExpanded('ref-msg1-2');
      });
      expect(result.current.expandedSourceIds.has('ref-msg1-2')).toBe(false);
    });

    it('can expand multiple sources', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.toggleSourceExpanded('ref-msg1-1');
        result.current.toggleSourceExpanded('ref-msg1-2');
        result.current.toggleSourceExpanded('ref-msg2-1');
      });

      expect(result.current.expandedSourceIds.size).toBe(3);
    });
  });

  describe('clearHighlight', () => {
    it('clears highlighted source', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.highlightSource('ref-msg1-1', 'msg1');
      });

      expect(result.current.highlightedSourceId).toBe('ref-msg1-1');

      act(() => {
        result.current.clearHighlight();
      });

      expect(result.current.highlightedSourceId).toBeNull();
    });
  });

  describe('useSourcesPanel outside provider', () => {
    it('throws error when used outside provider', () => {
      // Suppress console.error for this test
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useSourcesPanel());
      }).toThrow('useSourcesPanel must be used within a SourcesPanelProvider');

      consoleSpy.mockRestore();
    });
  });

  describe('selectedSourceId', () => {
    it('is null initially (SPC-01)', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      expect(result.current.selectedSourceId).toBeNull();
    });

    it('can be set via selectSource (SPC-02)', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.selectSource('ref-msg1-2', 'msg1');
      });

      expect(result.current.selectedSourceId).toBe('ref-msg1-2');
    });

    it('selectSource replaces previous selection (SPC-03)', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.selectSource('ref-msg1-1', 'msg1');
      });
      expect(result.current.selectedSourceId).toBe('ref-msg1-1');

      act(() => {
        result.current.selectSource('ref-msg1-2', 'msg1');
      });
      expect(result.current.selectedSourceId).toBe('ref-msg1-2');
    });

    it('selectSource expands the source (SPC-04)', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.selectSource('ref-msg1-3', 'msg1');
      });

      expect(result.current.expandedSourceIds.has('ref-msg1-3')).toBe(true);
    });

    it('selectSource sets activeMessageId (SPC-05)', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.selectSource('ref-msg2-1', 'msg2');
      });

      expect(result.current.activeMessageId).toBe('msg2');
    });
  });

  describe('clearSelection', () => {
    it('clears selectedSourceId to null (SPC-06)', () => {
      const { result } = renderHook(() => useSourcesPanel(), { wrapper });

      act(() => {
        result.current.selectSource('ref-msg1-1', 'msg1');
      });
      expect(result.current.selectedSourceId).toBe('ref-msg1-1');

      act(() => {
        result.current.clearSelection();
      });
      expect(result.current.selectedSourceId).toBeNull();
    });
  });
});
