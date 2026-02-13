/**
 * Hook for synchronizing scroll between chat and sources panels.
 *
 * Single Responsibility: Track which message is visible and coordinate scroll sync.
 */

import { useState, useCallback, useRef, useEffect } from 'react';

interface ScrollSyncState {
  /** ID of the currently most visible message */
  activeMessageId: string | null;
  /** Whether sync is temporarily paused (e.g., during manual scroll) */
  isPaused: boolean;
}

interface UseScrollSyncReturn extends ScrollSyncState {
  /** Set the active message ID */
  setActiveMessageId: (id: string | null) => void;
  /** Register a message element for intersection observation */
  registerMessageRef: (messageId: string, element: HTMLElement) => void;
  /** Unregister a message element */
  unregisterMessageRef: (messageId: string) => void;
  /** Temporarily pause scroll sync */
  pauseSync: () => void;
  /** Resume scroll sync */
  resumeSync: () => void;
}

/**
 * Hook for managing scroll synchronization between chat and sources panels.
 *
 * Uses IntersectionObserver to track which message is most visible in the viewport.
 * The active message ID can be used to scroll the sources panel to the corresponding sources.
 */
export function useScrollSync(): UseScrollSyncReturn {
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);

  // Track registered elements by message ID
  const elementsRef = useRef<Map<string, HTMLElement>>(new Map());

  // IntersectionObserver instance
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Track intersection ratios to determine most visible
  const visibilityRef = useRef<Map<string, number>>(new Map());

  /**
   * Find the most visible message based on intersection ratios.
   */
  const updateActiveMessage = useCallback(() => {
    if (isPaused) return;

    let maxRatio = 0;
    let mostVisibleId: string | null = null;

    visibilityRef.current.forEach((ratio, id) => {
      if (ratio > maxRatio) {
        maxRatio = ratio;
        mostVisibleId = id;
      }
    });

    if (mostVisibleId && maxRatio > 0.1) {
      setActiveMessageId(mostVisibleId);
    }
  }, [isPaused]);

  /**
   * Initialize IntersectionObserver.
   */
  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          // Find message ID for this element
          const messageId = Array.from(elementsRef.current.entries()).find(
            ([, el]) => el === entry.target
          )?.[0];

          if (messageId) {
            visibilityRef.current.set(messageId, entry.intersectionRatio);
          }
        });

        updateActiveMessage();
      },
      {
        // Observe elements when they enter the middle 60% of viewport
        rootMargin: '-20% 0px -20% 0px',
        threshold: [0, 0.25, 0.5, 0.75, 1],
      }
    );

    return () => {
      observerRef.current?.disconnect();
    };
  }, [updateActiveMessage]);

  /**
   * Register a message element for observation.
   */
  const registerMessageRef = useCallback(
    (messageId: string, element: HTMLElement) => {
      elementsRef.current.set(messageId, element);
      observerRef.current?.observe(element);
    },
    []
  );

  /**
   * Unregister a message element.
   */
  const unregisterMessageRef = useCallback((messageId: string) => {
    const element = elementsRef.current.get(messageId);
    if (element) {
      observerRef.current?.unobserve(element);
      elementsRef.current.delete(messageId);
      visibilityRef.current.delete(messageId);
    }
  }, []);

  /**
   * Pause scroll sync (e.g., when user manually scrolls sources panel).
   */
  const pauseSync = useCallback(() => {
    setIsPaused(true);
  }, []);

  /**
   * Resume scroll sync.
   */
  const resumeSync = useCallback(() => {
    setIsPaused(false);
  }, []);

  return {
    activeMessageId,
    setActiveMessageId,
    isPaused,
    registerMessageRef,
    unregisterMessageRef,
    pauseSync,
    resumeSync,
  };
}
