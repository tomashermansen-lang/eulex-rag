/**
 * Shared types for PDF export module.
 *
 * Single Responsibility: Define interfaces for PDF export components.
 */

import type { ChatMessage, Reference } from '../../types';

/** RGB color representation */
export interface RgbColor {
  r: number;
  g: number;
  b: number;
}

/** Text style configuration */
export interface TextStyle {
  fontSize: number;
  fontFamily: 'times' | 'helvetica' | 'courier';
  fontWeight: 'normal' | 'bold';
  color: RgbColor;
  lineHeight: number;
}

/** Page layout configuration */
export interface PageLayout {
  format: 'a4' | 'letter';
  orientation: 'portrait' | 'landscape';
  margin: number;
}

/** Spacing configuration */
export interface SpacingConfig {
  section: number;
  paragraph: number;
  line: number;
  sourceItem: number;
}

/** Complete document styles */
export interface DocumentStyles {
  body: TextStyle;
  heading: TextStyle;
  subheading: TextStyle;
  footnote: TextStyle;
  sourceExcerpt: TextStyle;
  page: PageLayout;
  spacing: SpacingConfig;
}

/** Context passed to section renderers */
export interface RenderContext {
  messages: ChatMessage[];
  currentY: number;
  pageNumber: number;
  pageHeight: number;
  pageWidth: number;
  margin: number;
  maxWidth: number;
}

/** Interface for PDF document abstraction (wraps jsPDF) */
export interface PdfDocumentInterface {
  // Text operations
  text(content: string, x: number, y: number): void;
  splitTextToSize(text: string, maxWidth: number): string[];

  // Styling
  setFontSize(size: number): void;
  setFont(family: string, weight: string): void;
  setTextColor(r: number, g: number, b: number): void;
  setFillColor(r: number, g: number, b: number): void;
  setDrawColor(r: number, g: number, b: number): void;

  // Drawing
  rect(x: number, y: number, width: number, height: number, style: string): void;
  line(x1: number, y1: number, x2: number, y2: number): void;

  // Page operations
  addPage(): void;
  getPageHeight(): number;
  getPageWidth(): number;

  // Output
  save(filename: string): void;
}

/** Interface for PDF section renderers (Open/Closed principle) */
export interface PdfSection {
  /** Render this section to the document */
  render(doc: PdfDocumentInterface, context: RenderContext): number;

  /** Estimate the height this section will require */
  estimateHeight(context: RenderContext): number;
}

/** Interface for citation formatters (Open/Closed principle) */
export interface CitationFormatter {
  /** Format a reference for display in the sources section */
  format(reference: Reference): string;

  /** Format an inline citation marker (e.g., "[1]") */
  formatInline(index: number): string;
}

/** Formatted citation with all parts */
export interface FormattedCitation {
  marker: string;      // e.g., "[1]"
  label: string;       // e.g., "GDPR, Artikel 30, stk. 1"
  excerpt: string;     // Full source text
}
