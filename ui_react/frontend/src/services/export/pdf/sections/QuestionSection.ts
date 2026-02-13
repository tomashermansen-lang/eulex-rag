/**
 * Question Section for PDF export.
 *
 * Renders user questions with a subtle background for visual distinction.
 *
 * Single Responsibility: Render user question blocks.
 */

import type { PdfDocument } from '../PdfDocument';
import { LegalDocumentStyles } from '../styles/LegalDocumentStyles';
import { TextFormatter } from '../formatters/TextFormatter';

/**
 * Renders a user question section.
 */
export class QuestionSection {
  private textFormatter = new TextFormatter();

  /**
   * Render a question and return the new Y position.
   *
   * @param doc - The PDF document
   * @param content - The question text
   * @param startY - Starting Y position
   * @returns New Y position after rendering
   */
  render(doc: PdfDocument, content: string, startY: number): number {
    const margin = doc.getMargin();
    const maxWidth = doc.getMaxContentWidth();
    let y = startY;

    // Check for page break
    const estimatedHeight = this.estimateHeight(doc, content);
    if (doc.needsNewPage(y, estimatedHeight)) {
      doc.addPage();
      y = margin;
    }

    // Format content
    const formattedContent = this.textFormatter.formatForPdf(content);

    // Calculate text dimensions for background
    doc.applyStyle(LegalDocumentStyles.body);
    const lines = doc.splitTextToSize(formattedContent, maxWidth - 10);
    const textHeight = lines.length * LegalDocumentStyles.spacing.line + 10;

    // Draw subtle background
    doc.setFillColor(248, 248, 248);
    doc.rect(margin, y - 2, maxWidth, textHeight, 'F');

    // Label
    doc.applyStyle(LegalDocumentStyles.subheading);
    const label = this.textFormatter.replaceDanishChars('Spørgsmål');
    doc.text(label, margin + 3, y + 5);
    y += 10;

    // Question text
    doc.applyStyle(LegalDocumentStyles.body);
    for (const line of lines) {
      if (doc.needsNewPage(y, LegalDocumentStyles.spacing.line)) {
        doc.addPage();
        y = margin;
      }
      doc.text(line, margin + 3, y);
      y += LegalDocumentStyles.spacing.line;
    }

    y += LegalDocumentStyles.spacing.section;
    return y;
  }

  /**
   * Estimate the height this section will require.
   */
  estimateHeight(doc: PdfDocument, content: string): number {
    const formattedContent = this.textFormatter.formatForPdf(content);
    doc.applyStyle(LegalDocumentStyles.body);
    const lines = doc.splitTextToSize(formattedContent, doc.getMaxContentWidth() - 10);
    return lines.length * LegalDocumentStyles.spacing.line + 20;
  }
}
