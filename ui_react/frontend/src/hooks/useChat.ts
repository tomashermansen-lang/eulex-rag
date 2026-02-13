/**
 * Hook for managing chat state and interactions.
 *
 * Single Responsibility: Handle chat messages, streaming, and history.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type { ChatMessage, Settings, AskResponse, HistoryMessage } from '../types';
import { streamAnswer } from '../services/api';
import { extractSuggestedQuestions } from '../utils/suggestions';

/** Maximum number of messages to include in history (5 exchanges = 10 messages) */
const MAX_HISTORY_MESSAGES = 10;

/**
 * Format references as a string for inclusion in history.
 */
function formatReferencesForHistory(references: ChatMessage['references']): string {
  if (!references || references.length === 0) return '';

  const refLines = references.map((ref) => `[${ref.idx}] ${ref.display}`);
  return '\n\nKilder:\n' + refLines.join('\n');
}

/**
 * Build history array from chat messages.
 * Includes full content and references for assistant messages.
 * Excludes streaming messages.
 */
function buildHistoryFromMessages(messages: ChatMessage[]): HistoryMessage[] {
  return messages
    .filter((m) => !m.isStreaming)
    .slice(-MAX_HISTORY_MESSAGES)
    .map((m) => {
      // For assistant messages, include the references
      if (m.role === 'assistant' && m.references) {
        return {
          role: m.role,
          content: m.content + formatReferencesForHistory(m.references),
        };
      }
      return {
        role: m.role,
        content: m.content,
      };
    });
}

/**
 * Generate a unique ID for messages.
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Hook for managing chat functionality.
 *
 * @param settings - Current app settings
 * @returns Chat state and control functions
 */
export function useChat(settings: Settings) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  // Ref to always have access to current messages (avoids stale closure)
  const messagesRef = useRef<ChatMessage[]>([]);

  // Keep ref in sync with state
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  /**
   * Add a user message to the chat.
   */
  const addUserMessage = useCallback((content: string): string => {
    const id = generateId();
    const message: ChatMessage = {
      id,
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, message]);
    return id;
  }, []);

  /**
   * Add or update an assistant message.
   */
  const updateAssistantMessage = useCallback(
    (
      id: string,
      updates: Partial<Omit<ChatMessage, 'id' | 'role' | 'timestamp'>>
    ) => {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === id ? { ...msg, ...updates } : msg
        )
      );
    },
    []
  );

  /**
   * Send a question and stream the response.
   */
  const sendMessage = useCallback(
    async (question: string) => {
      if (!question.trim() || isStreaming) return;

      setError(null);

      // Add user message
      addUserMessage(question);

      // Create placeholder assistant message
      const assistantId = generateId();
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        isStreaming: true,
      };
      setMessages((prev) => [...prev, assistantMessage]);

      setIsStreaming(true);
      let accumulatedContent = '';

      // Build history from current messages using ref (avoids stale closure)
      // This captures all completed messages before this new question
      const history = buildHistoryFromMessages(messagesRef.current);

      // Compute law from targetCorpora for backward compatibility with backend
      // Uses first selected corpus, or empty string if none selected
      const law = settings.targetCorpora[0] || '';

      // Start streaming
      abortRef.current = streamAnswer(
        {
          question,
          law, // Deprecated: kept for backend compatibility
          user_profile: settings.userProfile,
          history,
          corpus_scope: settings.corpusScope,
          target_corpora: settings.targetCorpora,
        },
        // On chunk
        (chunk) => {
          accumulatedContent += chunk;
          updateAssistantMessage(assistantId, {
            content: accumulatedContent,
          });
        },
        // On result
        (response: AskResponse) => {
          const suggestedQuestions = extractSuggestedQuestions(response.answer);

          updateAssistantMessage(assistantId, {
            content: response.answer,
            references: response.references,
            responseTime: response.response_time_seconds,
            suggestedQuestions:
              suggestedQuestions.length > 0 ? suggestedQuestions : undefined,
            retrievalMetrics: response.retrieval_metrics,
            isStreaming: false,
          });
          setIsStreaming(false);
        },
        // On error
        (err) => {
          setError(err.message);
          updateAssistantMessage(assistantId, {
            content: accumulatedContent || 'Der opstod en fejl. Prøv igen.',
            isStreaming: false,
          });
          setIsStreaming(false);
        }
      );
    },
    [settings, isStreaming, addUserMessage, updateAssistantMessage]
  );

  /**
   * Stop the current streaming response.
   */
  const stopStreaming = useCallback(() => {
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
      setIsStreaming(false);
    }
  }, []);

  /**
   * Clear all messages.
   */
  const clearChat = useCallback(() => {
    stopStreaming();
    setMessages([]);
    setError(null);
  }, [stopStreaming]);

  return {
    messages,
    isStreaming,
    error,
    sendMessage,
    stopStreaming,
    clearChat,
    hasMessages: messages.length > 0,
  };
}
