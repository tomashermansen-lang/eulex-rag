/**
 * Context for coordinating sources panel state across components.
 *
 * Single Responsibility: Manage highlight, expand/collapse, and active message state
 * for the sources sidepanel.
 */

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useRef,
  useEffect,
  type ReactNode,
} from 'react';

/** Duration in ms for highlight animation */
const HIGHLIGHT_DURATION = 2000;

interface SourcesPanelContextValue {
  /** ID of the currently highlighted source (e.g., "ref-msg1-1") */
  highlightedSourceId: string | null;
  /** ID of the active message (for scroll sync) */
  activeMessageId: string | null;
  /** Set of expanded source IDs */
  expandedSourceIds: Set<string>;
  /** Highlight a source and set active message */
  highlightSource: (refId: string, messageId: string) => void;
  /** Clear the highlight */
  clearHighlight: () => void;
  /** Set the active message ID */
  setActiveMessageId: (id: string | null) => void;
  /** Toggle a source's expanded state */
  toggleSourceExpanded: (refId: string) => void;
  /** ID of the currently selected source (persistent, no auto-clear) */
  selectedSourceId: string | null;
  /** Select a source (persistent selection with blue ring) */
  selectSource: (refId: string, messageId: string) => void;
  /** Clear the selection */
  clearSelection: () => void;
}

// Export for optional access (e.g., CitationLink can use context directly without throwing)
export const SourcesPanelContext = createContext<SourcesPanelContextValue | null>(null);

interface SourcesPanelProviderProps {
  children: ReactNode;
}

/**
 * Provider component for sources panel state.
 */
export function SourcesPanelProvider({ children }: SourcesPanelProviderProps) {
  const [highlightedSourceId, setHighlightedSourceId] = useState<string | null>(null);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const [expandedSourceIds, setExpandedSourceIds] = useState<Set<string>>(new Set());
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);

  // Timer ref for clearing highlight
  const highlightTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  /**
   * Clear highlight timer on unmount.
   */
  useEffect(() => {
    return () => {
      if (highlightTimerRef.current) {
        clearTimeout(highlightTimerRef.current);
      }
    };
  }, []);

  /**
   * Highlight a source and expand it.
   */
  const highlightSource = useCallback((refId: string, messageId: string) => {
    // Clear any existing timer
    if (highlightTimerRef.current) {
      clearTimeout(highlightTimerRef.current);
    }

    // Set highlight and active message
    setHighlightedSourceId(refId);
    setActiveMessageId(messageId);

    // Expand the source
    setExpandedSourceIds((prev) => {
      const next = new Set(prev);
      next.add(refId);
      return next;
    });

    // Clear highlight after duration
    highlightTimerRef.current = setTimeout(() => {
      setHighlightedSourceId(null);
    }, HIGHLIGHT_DURATION);
  }, []);

  /**
   * Clear the highlight immediately.
   */
  const clearHighlight = useCallback(() => {
    if (highlightTimerRef.current) {
      clearTimeout(highlightTimerRef.current);
    }
    setHighlightedSourceId(null);
  }, []);

  /**
   * Toggle a source's expanded state.
   */
  const toggleSourceExpanded = useCallback((refId: string) => {
    setExpandedSourceIds((prev) => {
      const next = new Set(prev);
      if (next.has(refId)) {
        next.delete(refId);
      } else {
        next.add(refId);
      }
      return next;
    });
  }, []);

  /**
   * Select a source (persistent selection for blue ring).
   */
  const selectSource = useCallback((refId: string, messageId: string) => {
    setSelectedSourceId(refId);
    setActiveMessageId(messageId);

    // Also expand the source
    setExpandedSourceIds((prev) => {
      const next = new Set(prev);
      next.add(refId);
      return next;
    });
  }, []);

  /**
   * Clear the selection.
   */
  const clearSelection = useCallback(() => {
    setSelectedSourceId(null);
  }, []);

  const value: SourcesPanelContextValue = {
    highlightedSourceId,
    activeMessageId,
    expandedSourceIds,
    highlightSource,
    clearHighlight,
    setActiveMessageId,
    toggleSourceExpanded,
    selectedSourceId,
    selectSource,
    clearSelection,
  };

  return (
    <SourcesPanelContext.Provider value={value}>
      {children}
    </SourcesPanelContext.Provider>
  );
}

/**
 * Hook to access sources panel context.
 *
 * @throws Error if used outside of SourcesPanelProvider
 */
export function useSourcesPanel(): SourcesPanelContextValue {
  const context = useContext(SourcesPanelContext);

  if (context === null) {
    throw new Error('useSourcesPanel must be used within a SourcesPanelProvider');
  }

  return context;
}
