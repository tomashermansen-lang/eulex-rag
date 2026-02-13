/**
 * Chat container component.
 *
 * Single Responsibility: Orchestrate the chat UI layout.
 */

import type { ReactNode } from 'react';
import { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { ChatMessage as ChatMessageType, Settings } from '../../types';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { ExampleQuestions } from '../Common/ExampleQuestions';

interface ChatContainerProps {
  /** Chat messages to display */
  messages: ChatMessageType[];
  /** Whether currently streaming a response */
  isStreaming: boolean;
  /** Callback to send a message */
  onSendMessage: (message: string) => void;
  /** Callback to stop streaming */
  onStopStreaming: () => void;
  /** Example questions to show when chat is empty */
  examples?: string[];
  /** Current settings */
  settings: Settings;
  /** Optional context prompt rendered after message list */
  contextPrompt?: ReactNode;
  /** Callback to lock discovered laws as search scope */
  onLock?: () => void;
}

/**
 * Main chat container with messages and input.
 *
 * Note: Header is now rendered in App.tsx above the split panel layout.
 * Sources are rendered in the SourcesSidepanel.
 */
export function ChatContainer({
  messages,
  isStreaming,
  onSendMessage,
  onStopStreaming,
  examples = [],
  settings,
  contextPrompt,
  onLock,
}: ChatContainerProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const hasMessages = messages.length > 0;

  return (
    <div className="flex flex-col h-full" ref={containerRef}>
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 py-6 space-y-4">
          <AnimatePresence mode="wait">
            {!hasMessages && !isStreaming && examples.length > 0 && (
              <motion.div
                key="examples"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, y: -20 }}
                className="flex items-center justify-center min-h-[50vh]"
              >
                <div className="text-center space-y-6">
                  <ExampleQuestions
                    questions={examples}
                    onQuestionClick={onSendMessage}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Message list */}
          {hasMessages && (
            <div className="space-y-4">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  onSuggestedQuestionClick={onSendMessage}
                  debugMode={settings.debugMode}
                  retrievalMetrics={message.retrievalMetrics}
                  onLock={onLock}
                />
              ))}
            </div>
          )}

          {/* Context prompt (e.g., lock-or-continue after discovery) */}
          {contextPrompt}

          {/* Scroll anchor */}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input area */}
      <ChatInput
        onSend={onSendMessage}
        isStreaming={isStreaming}
        onStopStreaming={onStopStreaming}
        disabled={isStreaming}
      />
    </div>
  );
}
