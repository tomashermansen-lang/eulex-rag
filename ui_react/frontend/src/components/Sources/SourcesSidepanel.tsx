/**
 * Sources sidepanel component.
 *
 * Single Responsibility: Display all sources from the conversation
 * in a sidepanel, grouped by message.
 */

import { useEffect, useRef, useCallback } from 'react';
import type { ChatMessage } from '../../types';
import { MessageSourceGroup } from './MessageSourceGroup';
import { parseCitations } from '../../utils/citations';
import { useSourcesPanel } from '../../contexts';

interface SourcesSidepanelProps {
  /** All messages in the conversation */
  messages: ChatMessage[];
  /** Optional EUR-Lex base URL */
  sourceUrl?: string;
}

/**
 * Sidepanel displaying all sources from the conversation.
 *
 * Groups sources by message and highlights/scrolls to sources
 * when citation links are clicked in the chat.
 */
export function SourcesSidepanel({ messages, sourceUrl }: SourcesSidepanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const { highlightSource, clearSelection } = useSourcesPanel();
  const prevMessageCountRef = useRef<number>(0);

  // Filter to only assistant messages with references
  const messagesWithSources = messages.filter(
    (msg) => msg.role === 'assistant' && msg.references && msg.references.length > 0
  );

  // Total source count
  const totalSources = messagesWithSources.reduce(
    (sum, msg) => sum + (msg.references?.length || 0),
    0
  );

  /**
   * Handle expandSource events from citation clicks.
   * Note: Selection ring is now handled by selectSource() in CitationLink.
   * This handler only handles scroll behavior.
   */
  const handleExpandSource = useCallback(
    (e: CustomEvent<{ refId: string; messageId?: string }>) => {
      const { refId, messageId } = e.detail;

      // Expand the source in context (for SourceItem auto-expand)
      if (messageId) {
        highlightSource(refId, messageId);
      }

      // Find and scroll to the source element
      const sourceElement = document.getElementById(refId);
      if (sourceElement) {
        sourceElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    },
    [highlightSource]
  );

  /**
   * Listen for expandSource events.
   */
  useEffect(() => {
    window.addEventListener('expandSource' as any, handleExpandSource);
    return () => window.removeEventListener('expandSource' as any, handleExpandSource);
  }, [handleExpandSource]);

  /**
   * Clear selection when new assistant message arrives.
   */
  useEffect(() => {
    const currentCount = messagesWithSources.length;
    if (currentCount > prevMessageCountRef.current && prevMessageCountRef.current > 0) {
      // New assistant message with sources arrived — clear selection
      clearSelection();
    }
    prevMessageCountRef.current = currentCount;
  }, [messagesWithSources.length, clearSelection]);

  // Empty state
  if (messagesWithSources.length === 0) {
    return (
      <div className="flex items-center justify-center h-full p-8">
        <p className="text-apple-gray-400 text-sm text-center">
          Ingen kilder endnu.
          <br />
          <span className="text-xs">Stil et spørgsmål for at se kilder.</span>
        </p>
      </div>
    );
  }

  return (
    <div ref={panelRef} className="h-full flex flex-col">
      {/* Header - Apple style: clean with subtle badge */}
      <div className="sticky top-0 z-10 px-4 py-3 bg-apple-gray-50/95 dark:bg-apple-gray-700/95 backdrop-blur-sm border-b border-apple-gray-100 dark:border-apple-gray-600">
        <h2 className="text-sm font-semibold text-apple-gray-700 dark:text-apple-gray-100">
          Kilder
          <span className="ml-1.5 text-apple-gray-400 font-normal">
            {totalSources}
          </span>
        </h2>
      </div>

      {/* Source groups */}
      <div className="flex-1 overflow-y-auto">
        {messagesWithSources.map((msg) => (
          <MessageSourceGroup
            key={msg.id}
            messageId={msg.id}
            references={msg.references || []}
            citedIndices={new Set(parseCitations(msg.content))}
            sourceUrl={sourceUrl}
          />
        ))}
      </div>
    </div>
  );
}
