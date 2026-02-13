/**
 * Sources Section for PDF export.
 *
 * Renders the citations/sources with professional EU legal formatting including:
 * - Full official law name and EUR-Lex URL
 * - Citation status (cited in answer vs only retrieved)
 * - Proper legal citation format
 *
 * Single Responsibility: Render source citations.
 */

import type { Reference } from '../../../../types';
import type { PdfDocument } from '../PdfDocument';
import { LegalDocumentStyles } from '../styles/LegalDocumentStyles';
import { TextFormatter } from '../formatters/TextFormatter';
import { EuLegalCitationFormatter, type DetailedCitation } from '../formatters/CitationFormatter';
import type { LinkTargetRegistry } from '../LinkTargetRegistry';
import type { LinkRenderer } from '../formatters/LinkRenderer';
import { buildSourceUrl } from '../../../../utils/citations';

/**
 * Renders the sources/citations section with enhanced legal formatting.
 */
export class SourcesSection {
  private textFormatter = new TextFormatter();
  private citationFormatter = new EuLegalCitationFormatter();

  /**
   * Render the sources section and return the new Y position.
   *
   * @param doc - The PDF document
   * @param references - The references to render
   * @param startY - Starting Y position
   * @param answerText - The answer text (to detect which sources were cited)
   * @param linkRegistry - Optional registry for storing source link targets
   * @param linkRenderer - Optional renderer for clickable EUR-Lex links
   * @returns New Y position after rendering
   */
  render(
    doc: PdfDocument,
    references: Reference[],
    startY: number,
    answerText: string = '',
    linkRegistry?: LinkTargetRegistry,
    linkRenderer?: LinkRenderer
  ): number {
    if (!references || references.length === 0) {
      return startY;
    }

    const margin = doc.getMargin();
    const maxWidth = doc.getMaxContentWidth();
    let y = startY;

    // Check if we need a new page for the sources header
    if (doc.needsNewPage(y, 20)) {
      doc.addPage();
      y = margin;
    }

    // Separator line
    doc.setDrawColor(180, 180, 180);
    doc.line(margin, y, margin + maxWidth, y);
    y += 8;

    // Section header
    doc.applyStyle(LegalDocumentStyles.subheading);
    doc.text('Kilder', margin, y);
    y += 12;

    // Create detailed citations
    const citations = references.map((ref) =>
      this.citationFormatter.createDetailedCitation(ref, answerText)
    );

    // Separate cited and uncited sources
    const citedSources = citations.filter((c) => c.wasCited);
    const uncitedSources = citations.filter((c) => !c.wasCited);

    // Render cited sources first
    if (citedSources.length > 0) {
      y = this.renderSourceGroup(doc, citedSources, y, margin, maxWidth, 'Citeret i svaret', references, linkRegistry, linkRenderer);
    }

    // Render uncited sources (retrieved but not cited)
    if (uncitedSources.length > 0) {
      if (citedSources.length > 0) {
        y += 6; // Extra spacing between groups
      }
      y = this.renderSourceGroup(doc, uncitedSources, y, margin, maxWidth, 'Hentet (ikke citeret)', references, linkRegistry, linkRenderer);
    }

    return y;
  }

  /**
   * Render a group of sources with a sub-header.
   */
  private renderSourceGroup(
    doc: PdfDocument,
    citations: DetailedCitation[],
    startY: number,
    margin: number,
    maxWidth: number,
    groupLabel: string,
    references: Reference[],
    linkRegistry?: LinkTargetRegistry,
    linkRenderer?: LinkRenderer
  ): number {
    let y = startY;

    // Group label (smaller, gray)
    doc.setFontSize(9);
    doc.setFont('helvetica', 'bold');
    doc.setTextColor(100, 100, 100);
    doc.text(groupLabel, margin, y);
    doc.setTextColor(0, 0, 0);
    y += 6;

    // Render each source
    for (const citation of citations) {
      const sourceHeight = this.estimateSingleSourceHeight(doc, citation);
      if (doc.needsNewPage(y, sourceHeight)) {
        doc.addPage();
        y = margin;
      }

      // Find matching reference for URL building
      const ref = references.find(r => `[${r.idx}]` === citation.marker);
      y = this.renderSingleSource(doc, citation, y, margin, maxWidth, ref, linkRegistry, linkRenderer);
    }

    return y;
  }

  /**
   * Render a single source citation with full metadata.
   */
  private renderSingleSource(
    doc: PdfDocument,
    citation: DetailedCitation,
    startY: number,
    margin: number,
    maxWidth: number,
    reference?: Reference,
    linkRegistry?: LinkTargetRegistry,
    linkRenderer?: LinkRenderer
  ): number {
    let y = startY;

    // Register source position for internal links
    if (linkRegistry) {
      const sourceId = `source-${citation.marker}`;
      linkRegistry.register(sourceId, doc.getCurrentPageNumber(), y);
    }

    // Line 1: Marker + Short citation (bold)
    // e.g., "[1] GDPR, Artikel 30, stk. 1"
    doc.applyStyle(LegalDocumentStyles.footnote);
    doc.setFont('helvetica', 'bold');
    const headerText = `${citation.marker} ${citation.shortCitation}`;
    const formattedHeader = this.textFormatter.replaceDanishChars(headerText);
    doc.text(formattedHeader, margin, y);
    y += 4.5;

    // Line 2: Full law name (normal, slightly smaller)
    // e.g., "Persondataforordningen (GDPR)"
    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    const formattedLawName = this.textFormatter.replaceDanishChars(citation.fullLawName);
    doc.text(formattedLawName, margin + 5, y);
    y += 4;

    // Line 3: EUR-Lex URL (if available)
    if (citation.eurLexUrl) {
      // Build anchored URL
      const anchoredUrl = reference
        ? buildSourceUrl(citation.eurLexUrl, {
            article: reference.article,
            recital: reference.recital,
            annex: reference.annex,
          })
        : citation.eurLexUrl;

      if (linkRenderer) {
        // Render as clickable link with underline
        doc.setFontSize(8);
        linkRenderer.renderLink(doc, 'Åbn i EUR-Lex', margin + 5, y, { url: anchoredUrl });
        doc.setTextColor(0, 0, 0);
      } else {
        // Fallback: render as plain text
        doc.setFontSize(8);
        doc.setTextColor(0, 102, 204); // Blue for link
        doc.text(citation.eurLexUrl, margin + 5, y);
        doc.setTextColor(0, 0, 0);
      }
      y += 4;
    }

    // Source excerpt (indented, gray)
    if (citation.excerpt) {
      doc.applyStyle(LegalDocumentStyles.sourceExcerpt);
      const formattedExcerpt = this.textFormatter.replaceDanishChars(citation.excerpt);

      // Indent source text and wrap
      const excerptMaxWidth = maxWidth - 10;
      const excerptLines = doc.splitTextToSize(formattedExcerpt, excerptMaxWidth);

      for (const line of excerptLines) {
        if (doc.needsNewPage(y, 4)) {
          doc.addPage();
          y = margin;
        }
        doc.text(line, margin + 5, y);
        y += 4;
      }

      // Reset to black
      doc.setTextColor(0, 0, 0);
    }

    y += LegalDocumentStyles.spacing.sourceItem;
    return y;
  }

  /**
   * Estimate the height for a single source.
   */
  private estimateSingleSourceHeight(doc: PdfDocument, citation: DetailedCitation): number {
    const maxWidth = doc.getMaxContentWidth();

    // Header + law name + URL
    let height = 4.5 + 4 + (citation.eurLexUrl ? 4 : 0);

    // Excerpt lines
    if (citation.excerpt) {
      doc.applyStyle(LegalDocumentStyles.sourceExcerpt);
      const excerptLines = doc.splitTextToSize(citation.excerpt, maxWidth - 10);
      height += excerptLines.length * 4;
    }

    height += LegalDocumentStyles.spacing.sourceItem;
    return height;
  }

  /**
   * Estimate the total height for all sources.
   */
  estimateHeight(doc: PdfDocument, references: Reference[], answerText: string = ''): number {
    if (!references || references.length === 0) {
      return 0;
    }

    const citations = references.map((ref) =>
      this.citationFormatter.createDetailedCitation(ref, answerText)
    );

    let totalHeight = 20; // Header + separator
    totalHeight += 6; // Group label

    for (const citation of citations) {
      totalHeight += this.estimateSingleSourceHeight(doc, citation);
    }

    return totalHeight;
  }
}
