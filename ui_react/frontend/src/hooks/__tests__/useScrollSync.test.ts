/**
 * Tests for useScrollSync hook.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useScrollSync } from '../useScrollSync';

// Mock IntersectionObserver
const mockObserve = vi.fn();
const mockUnobserve = vi.fn();
const mockDisconnect = vi.fn();

class MockIntersectionObserver {
  callback: IntersectionObserverCallback;

  constructor(callback: IntersectionObserverCallback) {
    this.callback = callback;
  }

  observe = mockObserve;
  unobserve = mockUnobserve;
  disconnect = mockDisconnect;
  root = null;
  rootMargin = '';
  thresholds = [];
  takeRecords = () => [];
}

describe('useScrollSync', () => {
  beforeEach(() => {
    vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);
    mockObserve.mockClear();
    mockUnobserve.mockClear();
    mockDisconnect.mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns activeMessageId as null initially', () => {
    const { result } = renderHook(() => useScrollSync());

    expect(result.current.activeMessageId).toBeNull();
  });

  it('provides setActiveMessageId to update active message', () => {
    const { result } = renderHook(() => useScrollSync());

    act(() => {
      result.current.setActiveMessageId('msg-123');
    });

    expect(result.current.activeMessageId).toBe('msg-123');
  });

  it('provides registerMessageRef to track message elements', () => {
    const { result } = renderHook(() => useScrollSync());
    const mockElement = document.createElement('div');

    act(() => {
      result.current.registerMessageRef('msg-456', mockElement);
    });

    // Should call observe on the IntersectionObserver
    expect(mockObserve).toHaveBeenCalledWith(mockElement);
  });

  it('provides unregisterMessageRef to stop tracking', () => {
    const { result } = renderHook(() => useScrollSync());
    const mockElement = document.createElement('div');

    act(() => {
      result.current.registerMessageRef('msg-789', mockElement);
      result.current.unregisterMessageRef('msg-789');
    });

    expect(mockUnobserve).toHaveBeenCalledWith(mockElement);
  });

  it('provides isPaused state for manual scroll handling', () => {
    const { result } = renderHook(() => useScrollSync());

    expect(result.current.isPaused).toBe(false);
  });

  it('provides pauseSync to temporarily disable syncing', () => {
    const { result } = renderHook(() => useScrollSync());

    act(() => {
      result.current.pauseSync();
    });

    expect(result.current.isPaused).toBe(true);
  });

  it('provides resumeSync to re-enable syncing', () => {
    const { result } = renderHook(() => useScrollSync());

    act(() => {
      result.current.pauseSync();
      result.current.resumeSync();
    });

    expect(result.current.isPaused).toBe(false);
  });

  it('disconnects observer on unmount', () => {
    const { unmount } = renderHook(() => useScrollSync());

    unmount();

    expect(mockDisconnect).toHaveBeenCalled();
  });
});
