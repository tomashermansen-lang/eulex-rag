/**
 * Text Formatter for PDF export.
 *
 * Handles text transformations needed for PDF compatibility:
 * - Danish character handling (preserved for fonts with Nordic support)
 * - Markdown stripping (for plain text rendering)
 *
 * Single Responsibility: Format text for PDF output.
 */

/**
 * Formats text content for PDF rendering.
 */
export class TextFormatter {
  /**
   * Process Danish special characters for PDF compatibility.
   *
   * jsPDF's built-in fonts (times, helvetica, courier) support Latin-1
   * encoding which includes æ, ø, å. We preserve these characters.
   *
   * Note: If you see issues with Danish characters, you may need to
   * embed a custom font with full Unicode support.
   */
  replaceDanishChars(text: string): string {
    // Latin-1 encoding in jsPDF's built-in fonts supports Danish characters
    // No replacement needed - preserve the original characters
    return text;
  }

  /**
   * Strip markdown formatting for plain text output.
   *
   * Removes:
   * - Headers (###)
   * - Bold (**text** or __text__)
   * - Italic (*text* or _text_)
   * - Inline code (`code`)
   * - Excessive whitespace
   */
  stripMarkdown(text: string): string {
    return text
      // Remove headers (### Header -> Header)
      .replace(/^#{1,6}\s+/gm, '')
      // Remove bold (**text** or __text__ -> text)
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/__([^_]+)__/g, '$1')
      // Remove italic (*text* or _text_ -> text)
      .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '$1')
      .replace(/(?<!_)_([^_]+)_(?!_)/g, '$1')
      // Remove inline code (`code` -> code)
      .replace(/`([^`]+)`/g, '$1')
      // Clean up extra whitespace
      .replace(/\n{3,}/g, '\n\n');
  }

  /**
   * Format text for PDF output.
   *
   * Combines all necessary transformations:
   * 1. Strip markdown formatting
   * 2. Replace Danish characters
   */
  formatForPdf(text: string): string {
    if (!text) {
      return '';
    }

    const stripped = this.stripMarkdown(text);
    return this.replaceDanishChars(stripped);
  }
}
