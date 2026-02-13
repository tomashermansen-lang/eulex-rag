/**
 * Export service for conversation downloads.
 *
 * Re-exports from the new modular export system for backward compatibility.
 * See ./export/ for the implementation.
 *
 * Single Responsibility: Public API for conversation export.
 */

export {
  formatAsMarkdown,
  downloadBlob,
  exportAsMarkdown,
  exportAsPdf,
  LegalDocumentStyles,
  EuLegalCitationFormatter,
  setCorpusRegistry,
} from './export/index';
