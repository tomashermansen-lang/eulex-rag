/**
 * Example questions component.
 *
 * Single Responsibility: Display example questions before first message.
 */

import { motion } from 'framer-motion';

interface ExampleQuestionsProps {
  /** Example questions to display */
  questions: string[];
  /** Callback when a question is clicked */
  onQuestionClick: (question: string) => void;
}

/**
 * Display example questions as clickable chips.
 */
export function ExampleQuestions({
  questions,
  onQuestionClick,
}: ExampleQuestionsProps) {
  if (questions.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-4"
    >
      <h3 className="text-lg font-medium text-apple-gray-500 dark:text-apple-gray-300 text-center">
        Prøv et eksempel
      </h3>
      <div className="flex flex-col items-center gap-3">
        {questions.map((question, index) => (
          <motion.button
            key={question}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.2, delay: index * 0.05 }}
            onClick={() => onQuestionClick(question)}
            className="suggestion-chip text-center max-w-lg"
          >
            {question}
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
