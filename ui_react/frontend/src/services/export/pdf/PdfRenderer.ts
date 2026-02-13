/**
 * PDF Renderer - Orchestrates PDF document generation.
 *
 * Coordinates all section renderers to produce a complete PDF document.
 *
 * Single Responsibility: Orchestrate PDF rendering.
 * Open/Closed: Add new sections without modifying this class.
 * Dependency Inversion: Depends on section abstractions, not implementations.
 */

import type { ChatMessage } from '../../../types';
import type { PageLayout } from '../types';
import { PdfDocument } from './PdfDocument';
import { LegalDocumentStyles } from './styles/LegalDocumentStyles';
import { HeaderSection } from './sections/HeaderSection';
import { QuestionSection } from './sections/QuestionSection';
import { AnswerSection } from './sections/AnswerSection';
import { SourcesSection } from './sections/SourcesSection';
import { FooterSection } from './sections/FooterSection';
import { LinkTargetRegistry } from './LinkTargetRegistry';
import { LinkRenderer } from './formatters/LinkRenderer';
import { CitationLinkCollector } from './CitationLinkCollector';

/**
 * Renders chat messages to a professional legal PDF document.
 */
export class PdfRenderer {
  private headerSection = new HeaderSection();
  private questionSection = new QuestionSection();
  private answerSection = new AnswerSection();
  private sourcesSection = new SourcesSection();
  private footerSection = new FooterSection();
  private linkRegistry = new LinkTargetRegistry();
  private linkRenderer = new LinkRenderer();
  private citationCollector = new CitationLinkCollector();

  /**
   * Render messages to a PDF document.
   *
   * @param jsPdfInstance - The jsPDF instance to render to
   * @param messages - Chat messages to render
   * @returns The PdfDocument wrapper for saving
   */
  render(jsPdfInstance: InstanceType<typeof import('jspdf').jsPDF>, messages: ChatMessage[]): PdfDocument {
    const pageLayout: PageLayout = {
      format: LegalDocumentStyles.page.format,
      orientation: LegalDocumentStyles.page.orientation,
      margin: LegalDocumentStyles.page.margin,
    };

    const doc = new PdfDocument(jsPdfInstance, pageLayout);
    const margin = doc.getMargin();

    // Clear registries from previous renders
    this.linkRegistry.clear();
    this.citationCollector.clear();

    // Start rendering
    let y = margin;

    // Header
    y = this.headerSection.render(doc, y);

    // Messages
    for (const msg of messages) {
      if (msg.role === 'user') {
        y = this.renderUserMessage(doc, msg, y);
      } else {
        y = this.renderAssistantMessage(doc, msg, y);
      }
    }

    // Create internal links from citations to sources (deferred linking)
    this.createInternalLinks(doc);

    // Footer on all pages
    const totalPages = doc.getJsPdf().getNumberOfPages();
    this.footerSection.renderOnAllPages(doc, totalPages);

    return doc;
  }

  /**
   * Render a user message (question).
   */
  private renderUserMessage(doc: PdfDocument, msg: ChatMessage, startY: number): number {
    return this.questionSection.render(doc, msg.content, startY);
  }

  /**
   * Render an assistant message (answer with sources).
   */
  private renderAssistantMessage(doc: PdfDocument, msg: ChatMessage, startY: number): number {
    let y = startY;

    // Answer content with inline citations
    // Pass citationCollector to track citation positions for internal linking
    y = this.answerSection.render(doc, msg.content, msg.references, y, this.citationCollector);

    // Sources section with clickable EUR-Lex links
    // Pass linkRegistry for position tracking and linkRenderer for clickable links
    if (msg.references && msg.references.length > 0) {
      y = this.sourcesSection.render(
        doc,
        msg.references,
        y,
        msg.content,
        this.linkRegistry,
        this.linkRenderer
      );
    }

    // Add spacing between Q&A pairs
    y += LegalDocumentStyles.spacing.section;

    return y;
  }

  /**
   * Create internal links from citations to their source positions.
   * Called after all content is rendered (deferred linking).
   *
   * @param doc - The PDF document
   */
  private createInternalLinks(doc: PdfDocument): void {
    const citationPositions = this.citationCollector.getAll();

    for (const citation of citationPositions) {
      // Look up the source position for this citation marker
      // Citation marker is "[1]", source ID is "source-[1]"
      const sourceId = `source-${citation.marker}`;
      const target = this.linkRegistry.get(sourceId);

      if (target) {
        // Switch to the page where the citation is rendered
        doc.setPage(citation.pageNumber);

        // Create a clickable link area that jumps to the source
        // The link area covers the citation text
        doc.link(
          citation.x,
          citation.y - citation.height, // y is baseline, adjust to top
          citation.width,
          citation.height + 1, // Add a bit of padding
          {
            pageNumber: target.pageNumber,
            top: target.top,
          }
        );
      }
    }
  }
}
