/**
 * Chat input component.
 *
 * Single Responsibility: Handle user input for sending messages.
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';

interface ChatInputProps {
  /** Callback when user submits a message */
  onSend: (message: string) => void;
  /** Whether input should be disabled */
  disabled?: boolean;
  /** Placeholder text */
  placeholder?: string;
  /** Whether currently streaming a response */
  isStreaming?: boolean;
  /** Callback to stop streaming */
  onStopStreaming?: () => void;
}

/**
 * Chat input field with send button.
 */
export function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Stil dit spørgsmål...',
  isStreaming = false,
  onStopStreaming,
}: ChatInputProps) {
  const [value, setValue] = useState('');
  const [hasOverflow, setHasOverflow] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea and check for overflow
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const newHeight = Math.min(textarea.scrollHeight, 200);
      textarea.style.height = `${newHeight}px`;
      // Check if content overflows (reached max height)
      setHasOverflow(textarea.scrollHeight > 200);
    }
  }, [value]);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      const trimmed = value.trim();
      if (trimmed && !disabled && !isStreaming) {
        onSend(trimmed);
        setValue('');
      }
    },
    [value, disabled, isStreaming, onSend]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  return (
    <form
      onSubmit={handleSubmit}
      className="sticky bottom-0 bg-apple-gray-50 dark:bg-apple-gray-700 border-t border-apple-gray-100 dark:border-apple-gray-600 p-4"
    >
      <div className="max-w-4xl mx-auto">
        <div className="flex items-end gap-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={disabled}
              rows={1}
              className={`input-field resize-none pr-12 min-h-[48px] ${hasOverflow ? 'overflow-y-auto' : 'overflow-y-hidden'}`}
            />
          </div>

          {isStreaming ? (
            <motion.button
              type="button"
              onClick={onStopStreaming}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn-secondary flex items-center gap-2"
            >
              <svg
                className="w-5 h-5"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
              Stop
            </motion.button>
          ) : (
            <motion.button
              type="submit"
              disabled={!value.trim() || disabled}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="btn-primary flex items-center gap-2"
            >
              <svg
                className="w-5 h-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
              Send
            </motion.button>
          )}
        </div>

        <div className="mt-3 text-center space-y-1">
          <p className="text-xs text-apple-gray-400">
            Tryk Enter for at sende, Shift+Enter for ny linje
          </p>
          <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wider">
            AI-genereret indhold · Læringsprojekt · Ikke juridisk rådgivning
          </p>
        </div>
      </div>
    </form>
  );
}
