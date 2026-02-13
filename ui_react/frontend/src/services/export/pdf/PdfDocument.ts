/**
 * PDF Document Abstraction.
 *
 * Wraps jsPDF to provide:
 * - Type safety
 * - Simplified API
 * - Future flexibility to swap PDF libraries
 *
 * Single Responsibility: Abstract PDF document operations.
 * Dependency Inversion: Code depends on this interface, not jsPDF directly.
 */

import type { jsPDF } from 'jspdf';
import type { TextStyle, PageLayout } from '../types';

/**
 * Wrapper around jsPDF for PDF document creation.
 */
export class PdfDocument {
  private doc: jsPDF;
  private pageLayout: PageLayout;

  constructor(jsPdfInstance: jsPDF, pageLayout: PageLayout) {
    this.doc = jsPdfInstance;
    this.pageLayout = pageLayout;
  }

  // ==================== Text Operations ====================

  /**
   * Draw text at the specified position.
   */
  text(content: string, x: number, y: number): void {
    this.doc.text(content, x, y);
  }

  /**
   * Split text to fit within a maximum width.
   */
  splitTextToSize(text: string, maxWidth: number): string[] {
    return this.doc.splitTextToSize(text, maxWidth);
  }

  // ==================== Styling ====================

  /**
   * Set the font size.
   */
  setFontSize(size: number): void {
    this.doc.setFontSize(size);
  }

  /**
   * Set the font family and weight.
   */
  setFont(family: string, weight: string): void {
    this.doc.setFont(family, weight);
  }

  /**
   * Set the text color (RGB).
   */
  setTextColor(r: number, g: number, b: number): void {
    this.doc.setTextColor(r, g, b);
  }

  /**
   * Set the fill color for shapes (RGB).
   */
  setFillColor(r: number, g: number, b: number): void {
    this.doc.setFillColor(r, g, b);
  }

  /**
   * Set the draw/stroke color (RGB).
   */
  setDrawColor(r: number, g: number, b: number): void {
    this.doc.setDrawColor(r, g, b);
  }

  /**
   * Apply a complete text style.
   */
  applyStyle(style: TextStyle): void {
    this.setFontSize(style.fontSize);
    this.setFont(style.fontFamily, style.fontWeight);
    this.setTextColor(style.color.r, style.color.g, style.color.b);
  }

  // ==================== Drawing ====================

  /**
   * Draw a rectangle.
   *
   * @param style - 'S' for stroke, 'F' for fill, 'DF' for both
   */
  rect(x: number, y: number, width: number, height: number, style: string): void {
    this.doc.rect(x, y, width, height, style);
  }

  /**
   * Draw a line.
   */
  line(x1: number, y1: number, x2: number, y2: number): void {
    this.doc.line(x1, y1, x2, y2);
  }

  // ==================== Page Operations ====================

  /**
   * Add a new page to the document.
   */
  addPage(): void {
    this.doc.addPage();
  }

  /**
   * Get the page height in mm.
   */
  getPageHeight(): number {
    return this.doc.internal.pageSize.getHeight();
  }

  /**
   * Get the page width in mm.
   */
  getPageWidth(): number {
    return this.doc.internal.pageSize.getWidth();
  }

  /**
   * Get the page margin.
   */
  getMargin(): number {
    return this.pageLayout.margin;
  }

  /**
   * Get the maximum content width (page width - 2 * margin).
   */
  getMaxContentWidth(): number {
    return this.getPageWidth() - 2 * this.pageLayout.margin;
  }

  /**
   * Check if we need a new page for the given content height.
   */
  needsNewPage(currentY: number, contentHeight: number): boolean {
    const bottomMargin = this.getPageHeight() - this.pageLayout.margin;
    return currentY + contentHeight > bottomMargin;
  }

  // ==================== Link Operations ====================

  /**
   * Draw text with an attached hyperlink.
   *
   * @param text - The text to display
   * @param x - X position in mm
   * @param y - Y position in mm
   * @param options - Link options (url for external, pageNumber+top for internal)
   */
  textWithLink(
    text: string,
    x: number,
    y: number,
    options: { url?: string; pageNumber?: number; top?: number }
  ): void {
    this.doc.textWithLink(text, x, y, options);
  }

  /**
   * Create a clickable rectangular link area.
   *
   * @param x - X position in mm
   * @param y - Y position in mm
   * @param width - Width in mm
   * @param height - Height in mm
   * @param options - Link options (url for external, pageNumber+top for internal)
   */
  link(
    x: number,
    y: number,
    width: number,
    height: number,
    options: { url?: string; pageNumber?: number; top?: number }
  ): void {
    this.doc.link(x, y, width, height, options);
  }

  /**
   * Get the width of text in current font.
   *
   * @param text - The text to measure
   * @returns Width in mm
   */
  getTextWidth(text: string): number {
    return this.doc.getTextWidth(text);
  }

  /**
   * Get the current page number (1-indexed).
   * Uses getNumberOfPages() since we render sequentially - the current page is always the last one.
   */
  getCurrentPageNumber(): number {
    return this.doc.getNumberOfPages();
  }

  /**
   * Switch to a specific page for adding content or links.
   *
   * @param pageNumber - The page number (1-indexed)
   */
  setPage(pageNumber: number): void {
    this.doc.setPage(pageNumber);
  }

  // ==================== Output ====================

  /**
   * Save the document to a file.
   */
  save(filename: string): void {
    this.doc.save(filename);
  }

  /**
   * Get the underlying jsPDF instance (for advanced operations).
   */
  getJsPdf(): jsPDF {
    return this.doc;
  }
}
