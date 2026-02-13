/**
 * Export service for conversation downloads.
 *
 * Single Responsibility: Handle conversation export to various formats.
 *
 * This module maintains backward compatibility with the original export.ts
 * while using the new modular PDF export system internally.
 */

import type { ChatMessage } from '../../types';
import { PdfRenderer } from './pdf/PdfRenderer';
import { LegalDocumentStyles } from './pdf/styles/LegalDocumentStyles';

// ==================== Utility Functions ====================

/**
 * Format date for export filename.
 */
function formatDateForFilename(): string {
  const now = new Date();
  return now.toISOString().slice(0, 10);
}

/**
 * Format timestamp for display.
 */
function formatTimestamp(date: Date): string {
  return date.toLocaleString('da-DK', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

// ==================== Markdown Export ====================

/**
 * Format messages as Markdown string.
 */
export function formatAsMarkdown(messages: ChatMessage[]): string {
  const lines: string[] = [
    '# EuLex Samtale',
    '',
    `Eksporteret: ${formatTimestamp(new Date())}`,
    '',
    '---',
    '',
  ];

  for (const msg of messages) {
    if (msg.role === 'user') {
      lines.push('## Spørgsmål');
      lines.push('');
      lines.push(msg.content);
      lines.push('');
    } else {
      lines.push('## Svar');
      if (msg.responseTime) {
        lines.push(`*Svartid: ${msg.responseTime.toFixed(1)}s*`);
      }
      lines.push('');
      lines.push(msg.content);
      lines.push('');

      // Add references with source text if present
      if (msg.references && msg.references.length > 0) {
        lines.push('### Kilder');
        lines.push('');
        for (const ref of msg.references) {
          lines.push(`**[${ref.idx}] ${ref.display}**`);
          if (ref.chunk_text) {
            lines.push('');
            lines.push(`> ${ref.chunk_text.substring(0, 500)}${ref.chunk_text.length > 500 ? '...' : ''}`);
          }
          lines.push('');
        }
      }
    }

    lines.push('---');
    lines.push('');
  }

  return lines.join('\n');
}

// ==================== Download Utility ====================

/**
 * Download content as a file.
 */
export function downloadBlob(
  content: string | Blob,
  filename: string,
  mimeType: string
): void {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = filename;

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

// ==================== Public Export Functions ====================

/**
 * Export conversation as Markdown file.
 */
export function exportAsMarkdown(messages: ChatMessage[]): void {
  const markdown = formatAsMarkdown(messages);
  const filename = `eulex-samtale-${formatDateForFilename()}.md`;
  downloadBlob(markdown, filename, 'text/markdown');
}

/**
 * Export conversation as PDF file.
 *
 * Uses the new professional legal document formatting:
 * - 12pt Times font for body text (legal standard)
 * - Inline citation markers [1], [2], etc.
 * - EU legal citation format (OSCOLA/EUR-Lex style)
 * - Full source excerpts (not truncated)
 * - Professional visual hierarchy
 *
 * Uses dynamic import to avoid loading jspdf until needed.
 */
export async function exportAsPdf(messages: ChatMessage[]): Promise<void> {
  const { jsPDF } = await import('jspdf');

  const doc = new jsPDF({
    orientation: LegalDocumentStyles.page.orientation,
    unit: 'mm',
    format: LegalDocumentStyles.page.format,
  });

  const renderer = new PdfRenderer();
  const pdfDoc = renderer.render(doc, messages);

  const filename = `eulex-analyse-${formatDateForFilename()}.pdf`;
  pdfDoc.save(filename);
}

// ==================== Re-exports for direct access ====================

export { LegalDocumentStyles } from './pdf/styles/LegalDocumentStyles';
export { EuLegalCitationFormatter } from './pdf/formatters/CitationFormatter';
export { setCorpusRegistry } from './pdf/data/corpusMetadata';
