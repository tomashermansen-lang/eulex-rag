/**
 * Pure utility functions for EuroVoc label handling.
 *
 * Single Responsibility: Label extraction, counting, and suggestion.
 */

import type { CorpusInfo } from '../../types';

export interface LabelWithCount {
  label: string;
  count: number;
}

/** Maximum number of AI suggestions to display */
const MAX_SUGGESTIONS = 5;

/**
 * Suggest relevant labels based on question keywords.
 * Phase 1: Simple keyword matching.
 * Phase 2 (future): Embedding-based semantic matching.
 */
export function suggestLabels(question: string, allLabels: string[]): string[] {
  if (!question.trim()) return [];

  // Extract words longer than 2 characters
  const words = question
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 2);

  if (words.length === 0) return [];

  // Find labels that contain any of the keywords
  return allLabels
    .filter((label) => words.some((word) => label.toLowerCase().includes(word)))
    .slice(0, MAX_SUGGESTIONS);
}

/**
 * Extract ALL unique labels from corpora, sorted by frequency then alphabetically.
 */
export function extractAllSortedLabels(corpora: CorpusInfo[]): LabelWithCount[] {
  const labelCounts = new Map<string, number>();

  for (const corpus of corpora) {
    if (corpus.eurovoc_labels) {
      for (const label of corpus.eurovoc_labels) {
        labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
      }
    }
  }

  return Array.from(labelCounts.entries())
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => {
      // Sort by frequency descending
      if (b.count !== a.count) return b.count - a.count;
      // Then alphabetically
      return a.label.localeCompare(b.label, 'da');
    });
}
