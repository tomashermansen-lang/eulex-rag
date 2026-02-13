/**
 * Header Section for PDF export.
 *
 * Renders the document title and export date at the top of the PDF.
 *
 * Single Responsibility: Render document header.
 */

import type { PdfDocument } from '../PdfDocument';
import { LegalDocumentStyles } from '../styles/LegalDocumentStyles';
import { TextFormatter } from '../formatters/TextFormatter';

/**
 * Renders the document header section.
 */
export class HeaderSection {
  private textFormatter = new TextFormatter();

  /**
   * Render the header and return the new Y position.
   *
   * @param doc - The PDF document
   * @param startY - Starting Y position
   * @returns New Y position after rendering
   */
  render(doc: PdfDocument, startY: number): number {
    const margin = doc.getMargin();
    let y = startY;

    // Title
    doc.applyStyle(LegalDocumentStyles.heading);
    doc.setFontSize(18); // Larger for main title
    doc.text('EuLex Analyse', margin, y);
    y += 10;

    // Export date
    doc.applyStyle(LegalDocumentStyles.footnote);
    const dateStr = new Date().toLocaleString('da-DK', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
    const formattedDate = this.textFormatter.replaceDanishChars(dateStr);
    doc.text(`Eksporteret: ${formattedDate}`, margin, y);
    y += 8;

    // Separator line
    doc.setDrawColor(200, 200, 200);
    doc.line(margin, y, doc.getPageWidth() - margin, y);
    y += LegalDocumentStyles.spacing.section;

    return y;
  }

  /**
   * Estimate the height this section will require.
   */
  estimateHeight(): number {
    return 30; // Fixed height for header
  }
}
