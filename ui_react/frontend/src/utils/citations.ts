/**
 * Citation parsing and transformation utilities.
 *
 * Single Responsibility: Handle citation-related text processing.
 */

/** Regular expression to match bracket citations like [1], [2], etc. */
const CITATION_PATTERN = /\[(\d+)\]/g;

/**
 * Extract all citation indices from text.
 *
 * @param text - The text containing citations
 * @returns Array of unique citation indices (as numbers)
 *
 * @example
 * parseCitations('See [1] and [2].') // [1, 2]
 * parseCitations('No citations here') // []
 */
export function parseCitations(text: string): number[] {
  if (!text) return [];

  const matches = text.matchAll(CITATION_PATTERN);
  const indices = new Set<number>();

  for (const match of matches) {
    indices.add(parseInt(match[1], 10));
  }

  return Array.from(indices).sort((a, b) => a - b);
}

/**
 * Check if text contains any citations.
 *
 * @param text - The text to check
 * @returns True if text contains at least one citation
 */
export function hasCitations(text: string): boolean {
  return CITATION_PATTERN.test(text);
}

/**
 * Generate an anchor ID for a reference.
 *
 * @param idx - The reference index
 * @param messageId - Optional message ID for uniqueness across messages
 * @returns Anchor ID string (e.g., "ref-msg123-1" or "ref-1" if no messageId)
 */
export function getRefAnchorId(idx: number | string, messageId?: string): string {
  if (messageId) {
    return `ref-${messageId}-${idx}`;
  }
  return `ref-${idx}`;
}

/**
 * Build a EUR-Lex URL with anchor for a reference.
 *
 * @param baseUrl - The base EUR-Lex URL
 * @param ref - The reference object
 * @returns Full URL with anchor, or base URL if no anchor needed
 */
export function buildSourceUrl(
  baseUrl: string,
  ref: { recital?: string; article?: string; annex?: string }
): string {
  if (!baseUrl) return '';

  let anchor = '';

  if (ref.recital) {
    anchor = `#rct_${ref.recital}`;
  } else if (ref.article) {
    anchor = `#art_${ref.article}`;
  } else if (ref.annex) {
    anchor = `#anx_${ref.annex}`;
  }

  return `${baseUrl}${anchor}`;
}
