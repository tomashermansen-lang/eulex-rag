/**
 * Suggested question extraction utilities.
 *
 * Single Responsibility: Extract follow-up suggestions from answers.
 */

/** Markers that indicate the start of a suggestions section */
const SUGGESTION_MARKERS = [
  '**Prøv at spørge**',
  'Du kan prøve at spørge om:',
  'Du kan prøve at spørge:',
  'Prøv at spørge om:',
];

/** Pattern to match quoted questions after various bullet styles */
const QUESTION_PATTERN = /[-·•*]\s*"([^"]+)"/g;

/**
 * Extract suggested questions from an answer text.
 *
 * Handles multiple formats:
 * - **Prøv at spørge** followed by bullet points
 * - "Du kan prøve at spørge om:" followed by bullet points
 *
 * @param answer - The answer text from the RAG system
 * @returns Array of suggested question strings
 *
 * @example
 * extractSuggestedQuestions('... Du kan prøve at spørge om:\n- "Hvad er X?"')
 * // ['Hvad er X?']
 */
export function extractSuggestedQuestions(answer: string): string[] {
  if (!answer) return [];

  // Find which marker is present
  const markerFound = SUGGESTION_MARKERS.find((marker) => answer.includes(marker));
  if (!markerFound) return [];

  // Get text after the marker
  const parts = answer.split(markerFound);
  if (parts.length < 2) return [];

  const section = parts[1];

  // Extract quoted questions
  const questions: string[] = [];
  const matches = section.matchAll(QUESTION_PATTERN);

  for (const match of matches) {
    questions.push(match[1]);
  }

  return questions;
}

/**
 * Check if an answer contains suggested questions.
 *
 * @param answer - The answer text to check
 * @returns True if the answer contains suggestions
 */
export function hasSuggestedQuestions(answer: string): boolean {
  return extractSuggestedQuestions(answer).length > 0;
}
