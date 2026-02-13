/**
 * Inline Citation Marker.
 *
 * Inserts citation markers (e.g., [1], [2]) into answer text to enable
 * readers to trace claims to their sources.
 *
 * Single Responsibility: Insert citation markers into text content.
 */

import type { Reference } from '../../../../types';

/**
 * Handles insertion of citation markers into text.
 *
 * Strategy: Append all citation markers at the end of the text.
 * This is simpler and more reliable than trying to match specific
 * claims to specific sources (which would require semantic analysis).
 *
 * Future enhancement: Use LLM to intelligently place markers inline.
 */
export class InlineCitationMarker {
  /**
   * Build a string of citation markers from references.
   *
   * @param references - The references to create markers for
   * @returns Space-separated marker string, e.g., "[1] [2] [3]"
   */
  buildMarkerString(references: Reference[]): string {
    if (references.length === 0) {
      return '';
    }

    return references.map((ref) => `[${ref.idx}]`).join(' ');
  }

  /**
   * Insert citation markers into text.
   *
   * Appends all markers at the end of the text, separated by spaces.
   * This approach is:
   * - Reliable (no semantic analysis needed)
   * - Consistent with legal document conventions
   * - Easy to verify for accuracy
   *
   * @param text - The text content to add markers to
   * @param references - The references to create markers for
   * @returns Text with citation markers appended
   */
  insertMarkers(text: string, references: Reference[]): string {
    const markerString = this.buildMarkerString(references);

    if (!markerString) {
      return text;
    }

    // Add space before markers if text doesn't end with space
    const separator = text.endsWith(' ') || text === '' ? '' : ' ';

    return `${text}${separator}${markerString}`;
  }
}
