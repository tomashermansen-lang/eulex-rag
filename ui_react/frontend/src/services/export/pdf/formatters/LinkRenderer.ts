/**
 * Link Renderer for PDF export.
 *
 * Renders clickable text with underline styling.
 * Supports both external URLs and internal document links.
 *
 * Single Responsibility: Render underlined link text.
 */

import type { PdfDocument } from '../PdfDocument';

/**
 * Options for link rendering.
 */
export interface LinkOptions {
  /** External URL for the link */
  url?: string;
  /** Target page number for internal link (1-indexed) */
  pageNumber?: number;
  /** Target Y position for internal link (mm from top) */
  top?: number;
}

/** Link color: blue rgb(0, 102, 204) */
const LINK_COLOR = { r: 0, g: 102, b: 204 };

/** Underline offset below text baseline (mm) */
const UNDERLINE_OFFSET = 0.5;

/**
 * Renders text as a clickable link with underline.
 */
export class LinkRenderer {
  /**
   * Render text as a clickable link with underline.
   *
   * @param doc - The PDF document
   * @param text - Text to render
   * @param x - X position in mm
   * @param y - Y position in mm
   * @param options - Link options (url or pageNumber+top)
   * @returns Width of the rendered text in mm
   */
  renderLink(
    doc: PdfDocument,
    text: string,
    x: number,
    y: number,
    options: LinkOptions
  ): number {
    // Set link color
    doc.setTextColor(LINK_COLOR.r, LINK_COLOR.g, LINK_COLOR.b);

    // Render text with link
    doc.textWithLink(text, x, y, options);

    // Get text width for underline
    const width = doc.getTextWidth(text);

    // Draw underline
    this.drawUnderline(doc, x, y, width);

    return width;
  }

  /**
   * Draw underline below text.
   *
   * @param doc - The PDF document
   * @param x - X position of text start
   * @param y - Y position of text baseline
   * @param width - Width of text
   */
  private drawUnderline(doc: PdfDocument, x: number, y: number, width: number): void {
    doc.setDrawColor(LINK_COLOR.r, LINK_COLOR.g, LINK_COLOR.b);
    const underlineY = y + UNDERLINE_OFFSET;
    doc.line(x, underlineY, x + width, underlineY);
  }
}
