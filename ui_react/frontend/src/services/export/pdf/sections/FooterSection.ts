/**
 * Footer Section for PDF export.
 *
 * Renders page footers with page numbers and branding.
 *
 * Single Responsibility: Render document footer.
 */

import type { PdfDocument } from '../PdfDocument';
import { LegalDocumentStyles } from '../styles/LegalDocumentStyles';

/**
 * Renders page footer information.
 */
export class FooterSection {
  /**
   * Render footer on all pages.
   *
   * Note: This should be called after all content is rendered,
   * as it iterates through all pages to add footers.
   *
   * @param doc - The PDF document
   * @param totalPages - Total number of pages
   */
  renderOnAllPages(doc: PdfDocument, totalPages: number): void {
    const jsPdf = doc.getJsPdf();
    const pageHeight = doc.getPageHeight();
    const pageWidth = doc.getPageWidth();
    const margin = doc.getMargin();

    for (let i = 1; i <= totalPages; i++) {
      jsPdf.setPage(i);

      // Footer line
      const footerY = pageHeight - margin + 5;
      doc.setDrawColor(200, 200, 200);
      doc.line(margin, footerY, pageWidth - margin, footerY);

      // Page number (right aligned)
      doc.applyStyle(LegalDocumentStyles.footnote);
      const pageText = `Side ${i} af ${totalPages}`;
      const textWidth = jsPdf.getTextWidth(pageText);
      doc.text(pageText, pageWidth - margin - textWidth, footerY + 5);

      // Branding (left aligned)
      doc.text('EuLex', margin, footerY + 5);

      // AI disclaimer (centered)
      const disclaimerText = 'AI-genereret · Ikke juridisk rådgivning';
      const disclaimerWidth = jsPdf.getTextWidth(disclaimerText);
      const centerX = (pageWidth - disclaimerWidth) / 2;
      doc.text(disclaimerText, centerX, footerY + 5);
    }
  }
}
