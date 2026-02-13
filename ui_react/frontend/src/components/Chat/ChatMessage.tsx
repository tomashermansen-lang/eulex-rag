/**
 * Chat message component.
 *
 * Single Responsibility: Render a single chat message with appropriate styling.
 */

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ChatMessage as ChatMessageType, DiscoveryMatch } from '../../types';
import { renderWithCitations } from '../Common/CitationLink';
import { SuggestedQuestions } from '../Common/SuggestedQuestions';
import { DiscoveryBanner } from './DiscoveryBanner';
import { AbstentionCard } from './AbstentionCard';

interface ChatMessageProps {
  /** The message to render */
  message: ChatMessageType;
  /** Callback when a suggested question is clicked */
  onSuggestedQuestionClick?: (question: string) => void;
  /** Whether debug mode is enabled */
  debugMode?: boolean;
  /** Retrieval metrics for debug display */
  retrievalMetrics?: Record<string, unknown>;
  /** Callback to switch to manual corpus selection */
  onSelectManual?: (corpora: string[]) => void;
  /** Callback to focus chat input for rephrasing */
  onRephrase?: () => void;
  /** Callback to lock discovered laws as search scope */
  onLock?: () => void;
}

/**
 * Render a single chat message.
 *
 * Note: Sources are now rendered in the sidepanel (SourcesSidepanel),
 * not embedded in each message.
 */
export function ChatMessage({
  message,
  onSuggestedQuestionClick,
  debugMode = false,
  retrievalMetrics,
  onSelectManual,
  onRephrase,
  onLock,
}: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [showDebug, setShowDebug] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);

  // Extract discovery data from retrieval metrics (nested under run.discovery)
  const run = retrievalMetrics?.run as Record<string, unknown> | undefined;
  const discovery = run?.discovery as
    | { gate: 'AUTO' | 'SUGGEST' | 'ABSTAIN'; matches: DiscoveryMatch[]; resolved_corpora?: string[] }
    | undefined;

  const handleCopyAnswer = useCallback(async () => {
    try {
      let textToCopy = message.content;

      // In debug mode, include all metadata
      if (debugMode) {
        const debugData = {
          answer: message.content,
          responseTime: message.responseTime,
          references: message.references,
          retrievalMetrics: retrievalMetrics,
        };
        textToCopy = JSON.stringify(debugData, null, 2);
      }

      await navigator.clipboard.writeText(textToCopy);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, [message.content, message.responseTime, message.references, retrievalMetrics, debugMode]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
      data-message-id={message.id}
    >
      {/* ABSTAIN gate: show AbstentionCard instead of normal message */}
      {!isUser && discovery?.gate === 'ABSTAIN' && onSelectManual && onRephrase ? (
        <AbstentionCard
          candidates={discovery.matches}
          onSelectManual={onSelectManual}
          onRephrase={onRephrase}
        />
      ) : (
      <div className={isUser ? 'chat-bubble-user' : 'chat-bubble-assistant'}>
        {/* Discovery banner (AUTO/SUGGEST only) */}
        {!isUser && discovery && discovery.gate !== 'ABSTAIN' && (
          <DiscoveryBanner
            gate={discovery.gate}
            matches={discovery.matches}
            resolvedCorpora={discovery.resolved_corpora}
            onLock={onLock}
          />
        )}

        {/* Message content */}
        <div className={`prose max-w-none ${isUser ? 'text-white' : 'text-apple-gray-700 dark:text-apple-gray-100'}`}>
          {isUser ? (
            <p className="m-0">{message.content}</p>
          ) : (
            <div>
              {message.content ? (
                renderWithCitations(message.content, message.id)
              ) : message.isStreaming ? (
                <div className="typing-indicator">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              ) : null}
            </div>
          )}
        </div>

        {/* Response time and actions */}
        {!isUser && !message.isStreaming && (
          <div className="mt-2 flex items-center gap-3 flex-wrap">
            {message.responseTime && (
              <p className="text-xs text-apple-gray-400">
                Svar genereret på {message.responseTime.toFixed(1)} sekunder
              </p>
            )}

            {/* Copy button - always visible */}
            <button
              onClick={handleCopyAnswer}
              className="text-xs text-apple-gray-400 hover:text-apple-blue transition-colors flex items-center gap-1"
              title={debugMode ? "Kopiér svar med metadata" : "Kopiér svar"}
            >
              {copySuccess ? (
                <>
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Kopieret
                </>
              ) : (
                <>
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Kopiér{debugMode ? ' alt' : ''}
                </>
              )}
            </button>

            {/* Debug metadata toggle - only in debug mode */}
            {debugMode && retrievalMetrics && Object.keys(retrievalMetrics).length > 0 && (
              <button
                onClick={() => setShowDebug(!showDebug)}
                className="text-xs text-apple-gray-400 hover:text-apple-blue transition-colors flex items-center gap-1"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                {showDebug ? 'Skjul' : 'Vis'} metadata
              </button>
            )}
          </div>
        )}

        {/* Debug metadata panel */}
        <AnimatePresence>
          {debugMode && showDebug && retrievalMetrics && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
              className="mt-3 p-3 bg-apple-gray-50 dark:bg-apple-gray-700 rounded-lg overflow-hidden"
            >
              <p className="text-xs font-semibold text-apple-gray-500 dark:text-apple-gray-300 mb-2">
                Retrieval Metadata
              </p>
              <pre className="text-xs text-apple-gray-600 dark:text-apple-gray-300 overflow-x-auto whitespace-pre-wrap font-mono">
                {JSON.stringify(retrievalMetrics, null, 2)}
              </pre>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Suggested questions */}
        {!isUser &&
          message.suggestedQuestions &&
          message.suggestedQuestions.length > 0 &&
          onSuggestedQuestionClick && (
            <SuggestedQuestions
              questions={message.suggestedQuestions}
              onQuestionClick={onSuggestedQuestionClick}
            />
          )}

      </div>
      )}
    </motion.div>
  );
}
