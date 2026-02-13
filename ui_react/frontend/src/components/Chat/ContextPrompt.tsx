/**
 * ContextPrompt component.
 *
 * Single Responsibility: Inline follow-up prompt offering lock or continue
 * when a user sends a message in discovery mode with existing results.
 */

import { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';

interface ContextPromptProps {
  lawNames: string[];
  onLock: () => void;
  onContinue: () => void;
}

/**
 * Inline prompt with two action buttons.
 *
 * - "Lås til [LAWS]" — lock to discovered laws and send message
 * - "Fortsæt AI-søgning" — send message with discovery scope unchanged
 */
export function ContextPrompt({ lawNames, onLock, onContinue }: ContextPromptProps) {
  const lockRef = useRef<HTMLButtonElement>(null);

  // Auto-focus the lock button on mount
  useEffect(() => {
    lockRef.current?.focus();
  }, []);

  const lawList = lawNames.join(', ');

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      role="status"
      className="mx-4 mb-3 p-3 rounded-lg border border-apple-gray-200 dark:border-apple-gray-600 bg-apple-gray-50 dark:bg-apple-gray-700"
    >
      <p className="text-sm text-apple-gray-500 dark:text-apple-gray-300 mb-2">
        AI fandt <span className="font-medium">{lawList}</span>. Vil du låse søgeområdet?
      </p>
      <div className="flex gap-2">
        <button
          ref={lockRef}
          onClick={onLock}
          className="btn-primary text-sm"
        >
          Lås til {lawList}
        </button>
        <button
          onClick={onContinue}
          className="btn-secondary text-sm"
        >
          Fortsæt AI-søgning
        </button>
      </div>
    </motion.div>
  );
}
