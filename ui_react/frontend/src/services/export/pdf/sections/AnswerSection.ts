/**
 * Answer Section for PDF export.
 *
 * Renders assistant answers with proper formatting for:
 * - Subheadings (bold text or markdown headers)
 * - Body paragraphs
 * - Inline citations (already present in the text)
 *
 * Single Responsibility: Render assistant answer blocks with structure.
 */

import type { Reference } from '../../../../types';
import type { PdfDocument } from '../PdfDocument';
import { LegalDocumentStyles } from '../styles/LegalDocumentStyles';
import { TextFormatter } from '../formatters/TextFormatter';
import type { CitationLinkCollector } from '../CitationLinkCollector';

/** Citation pattern to find markers like [1], [2], etc. */
const CITATION_PATTERN = /\[(\d+)\]/g;

/** A parsed block of content */
interface ContentBlock {
  type: 'heading' | 'paragraph';
  text: string;
}

/**
 * Renders an assistant answer section with proper structure.
 */
export class AnswerSection {
  private textFormatter = new TextFormatter();

  /**
   * Parse content into structured blocks.
   * Detects:
   * - Markdown headers (## Header or **Header**)
   * - Bold-only lines as subheadings
   * - Regular paragraphs
   */
  private parseContent(content: string): ContentBlock[] {
    const blocks: ContentBlock[] = [];
    const lines = content.split('\n');

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      if (!line) {
        continue; // Skip empty lines
      }

      // Check for markdown header (## Header)
      const headerMatch = line.match(/^#{1,3}\s+(.+)$/);
      if (headerMatch) {
        blocks.push({ type: 'heading', text: headerMatch[1] });
        continue;
      }

      // Check for bold-only line (**Header** or __Header__)
      // These are common subheading patterns in LLM responses
      const boldMatch = line.match(/^\*\*([^*]+)\*\*$/) || line.match(/^__([^_]+)__$/);
      if (boldMatch) {
        blocks.push({ type: 'heading', text: boldMatch[1] });
        continue;
      }

      // Check for line that starts with bold followed by colon or newline
      // e.g., "**Retsgrundlag:**" or "**Konklusion**"
      const boldStartMatch = line.match(/^\*\*([^*]+)\*\*:?\s*$/);
      if (boldStartMatch && line.length < 60) {
        blocks.push({ type: 'heading', text: boldStartMatch[1].replace(/:$/, '') });
        continue;
      }

      // Regular paragraph - may contain inline bold, citations, etc.
      blocks.push({ type: 'paragraph', text: line });
    }

    return blocks;
  }

  /**
   * Clean markdown formatting from text while preserving citations.
   * Removes bold/italic markers but keeps [1], [2] etc.
   */
  private cleanMarkdown(text: string): string {
    return text
      .replace(/\*\*([^*]+)\*\*/g, '$1') // Remove bold
      .replace(/__([^_]+)__/g, '$1')      // Remove underscore bold
      .replace(/\*([^*]+)\*/g, '$1')      // Remove italic
      .replace(/_([^_]+)_/g, '$1')        // Remove underscore italic
      .replace(/`([^`]+)`/g, '$1');       // Remove inline code
  }

  /**
   * Render an answer with proper structure and return the new Y position.
   *
   * @param doc - The PDF document
   * @param content - The answer text (with inline citations already present)
   * @param references - References (used only for validation, not for inserting markers)
   * @param startY - Starting Y position
   * @param citationCollector - Optional collector for tracking citation positions
   * @returns New Y position after rendering
   */
  render(
    doc: PdfDocument,
    content: string,
    references: Reference[] | undefined,
    startY: number,
    citationCollector?: CitationLinkCollector
  ): number {
    const margin = doc.getMargin();
    const maxWidth = doc.getMaxContentWidth();
    let y = startY;

    // Section label
    doc.applyStyle(LegalDocumentStyles.subheading);
    doc.text('Svar', margin, y);
    y += 10;

    // Parse content into structured blocks
    const blocks = this.parseContent(content);

    for (const block of blocks) {
      if (block.type === 'heading') {
        // Render as subheading
        if (doc.needsNewPage(y, 15)) {
          doc.addPage();
          y = margin;
        }

        y += 4; // Extra space before heading
        doc.applyStyle(LegalDocumentStyles.subheading);
        const headingText = this.textFormatter.replaceDanishChars(block.text);
        doc.text(headingText, margin, y);
        y += 8;
      } else {
        // Render as body paragraph
        const cleanedText = this.cleanMarkdown(block.text);
        const formattedText = this.textFormatter.replaceDanishChars(cleanedText);

        doc.applyStyle(LegalDocumentStyles.body);
        const lines = doc.splitTextToSize(formattedText, maxWidth);

        for (const line of lines) {
          if (doc.needsNewPage(y, LegalDocumentStyles.spacing.line)) {
            doc.addPage();
            y = margin;
          }
          doc.text(line, margin, y);

          // Track citation positions in this line
          if (citationCollector) {
            this.recordCitationPositions(doc, line, margin, y, citationCollector);
          }

          y += LegalDocumentStyles.spacing.line;
        }

        y += 2; // Small spacing between paragraphs
      }
    }

    y += LegalDocumentStyles.spacing.paragraph;
    return y;
  }

  /**
   * Find and record positions of citation markers in a line of text.
   *
   * @param doc - The PDF document (for text width measurement)
   * @param line - The rendered line of text
   * @param lineX - X position where the line starts
   * @param lineY - Y position of the line
   * @param collector - The citation collector to record to
   */
  private recordCitationPositions(
    doc: PdfDocument,
    line: string,
    lineX: number,
    lineY: number,
    collector: CitationLinkCollector
  ): void {
    // Find all citations in the line
    const matches = line.matchAll(CITATION_PATTERN);

    for (const match of matches) {
      const marker = match[0]; // e.g., "[1]"
      const index = match.index!;

      // Calculate x position by measuring text before the citation
      const textBefore = line.substring(0, index);
      const xOffset = doc.getTextWidth(textBefore);

      // Get citation width
      const citationWidth = doc.getTextWidth(marker);

      // Record the position
      // Height is approximate based on font size (body style is ~11pt ≈ 4mm)
      collector.record(
        marker,
        doc.getCurrentPageNumber(),
        lineX + xOffset,
        lineY,
        citationWidth,
        4 // Approximate height in mm
      );
    }
  }

  /**
   * Estimate the height this section will require.
   */
  estimateHeight(doc: PdfDocument, content: string): number {
    const blocks = this.parseContent(content);
    let totalHeight = 15; // Label

    for (const block of blocks) {
      if (block.type === 'heading') {
        totalHeight += 15;
      } else {
        const cleanedText = this.cleanMarkdown(block.text);
        doc.applyStyle(LegalDocumentStyles.body);
        const lines = doc.splitTextToSize(cleanedText, doc.getMaxContentWidth());
        totalHeight += lines.length * LegalDocumentStyles.spacing.line + 2;
      }
    }

    return totalHeight;
  }
}
