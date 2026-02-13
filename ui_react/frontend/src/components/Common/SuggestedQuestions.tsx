/**
 * Suggested follow-up questions component.
 *
 * Single Responsibility: Display clickable follow-up suggestions.
 */

import { motion } from 'framer-motion';

interface SuggestedQuestionsProps {
  /** Suggested questions to display */
  questions: string[];
  /** Callback when a question is clicked */
  onQuestionClick: (question: string) => void;
}

/**
 * Display suggested follow-up questions as clickable buttons.
 */
export function SuggestedQuestions({
  questions,
  onQuestionClick,
}: SuggestedQuestionsProps) {
  if (questions.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, delay: 0.1 }}
      className="mt-4 space-y-2"
    >
      <p className="text-sm text-apple-gray-400 dark:text-apple-gray-400">
        Klik for at stille spørgsmålet:
      </p>
      <div className="flex flex-wrap gap-2">
        {questions.slice(0, 3).map((question, index) => (
          <motion.button
            key={question}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.15, delay: index * 0.05 }}
            onClick={() => onQuestionClick(question)}
            className="suggestion-chip text-sm"
          >
            {question}
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
