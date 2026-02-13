/**
 * EU Legal Citation Formatter.
 *
 * Formats references according to EU legal citation standards (OSCOLA/EUR-Lex).
 * Includes full official names, CELEX numbers, and EUR-Lex URLs.
 *
 * Single Responsibility: Format legal citations for display.
 * Open/Closed: Extend by adding new corpus mappings without modifying format logic.
 */

import type { Reference } from '../../../../types';
import type { CitationFormatter } from '../../types';
import { getCorpusMetadata, getCorpusDisplayName as getDisplayName, getCorpusFullname, type CorpusMetadata } from '../data/corpusMetadata';

/** Detailed citation information for PDF rendering */
export interface DetailedCitation {
  /** Reference marker, e.g., "[1]" */
  marker: string;
  /** Short citation, e.g., "GDPR, Artikel 30, stk. 1" */
  shortCitation: string;
  /** Full official law name, e.g., "Persondataforordningen (GDPR)" */
  fullLawName: string;
  /** Specific provision, e.g., "Artikel 30, stk. 1" */
  provision: string;
  /** EUR-Lex URL for the source law */
  eurLexUrl: string | null;
  /** CELEX number, e.g., "32016R0679" */
  celexNumber: string | null;
  /** Whether this reference was cited in the answer text */
  wasCited: boolean;
  /** The source excerpt text */
  excerpt: string;
}

/**
 * Formats citations according to EU legal standards with full metadata.
 *
 * Format examples:
 * - Short: "GDPR, Artikel 30, stk. 1"
 * - Full: "Persondataforordningen (GDPR), Artikel 30, stk. 1"
 * - With URL: EUR-Lex link included
 */
export class EuLegalCitationFormatter implements CitationFormatter {
  /**
   * Get the display name for a corpus (e.g., "Persondataforordningen (GDPR)").
   */
  getCorpusDisplayName(corpusId: string): string {
    return getDisplayName(corpusId);
  }

  /**
   * Get the full official legal title for a corpus.
   */
  getCorpusFullLegalName(corpusId: string): string {
    return getCorpusFullname(corpusId);
  }

  /**
   * Get full metadata for a corpus.
   */
  getCorpusMetadata(corpusId: string): CorpusMetadata | undefined {
    return getCorpusMetadata(corpusId);
  }

  /**
   * Format the provision part of a reference (article, recital, or annex).
   */
  formatProvision(reference: Reference): string {
    const { article, paragraph, litra, recital, annex } = reference;

    if (recital) {
      return `Betragtning ${recital}`;
    }

    if (annex) {
      return `Bilag ${annex}`;
    }

    if (article) {
      const parts = [`Artikel ${article}`];
      if (paragraph) {
        parts.push(`stk. ${paragraph}`);
      }
      if (litra) {
        parts.push(`litra ${litra}`);
      }
      return parts.join(', ');
    }

    return '';
  }

  /**
   * Format a reference for display in the sources section (short form).
   *
   * Priority order:
   * 1. Article + paragraph + litra (most specific)
   * 2. Recital
   * 3. Annex
   * 4. Fallback to original display
   */
  format(reference: Reference): string {
    const { corpus_id, display } = reference;

    if (!corpus_id) {
      return display;
    }

    const corpusName = this.getCorpusDisplayName(corpus_id);
    const provision = this.formatProvision(reference);

    if (!provision) {
      return display;
    }

    return `${corpusName}, ${provision}`;
  }

  /**
   * Format an inline citation marker.
   *
   * @param index - The 1-based citation index
   * @returns Bracketed marker string, e.g., "[1]"
   */
  formatInline(index: number): string {
    return `[${index}]`;
  }

  /**
   * Check if a reference was cited in the answer text.
   *
   * Looks for the reference marker (e.g., "[1]") in the text.
   */
  wasCitedInText(reference: Reference, answerText: string): boolean {
    const marker = `[${reference.idx}]`;
    return answerText.includes(marker);
  }

  /**
   * Create detailed citation information for PDF rendering.
   *
   * @param reference - The reference to format
   * @param answerText - The answer text to check for citations
   * @returns Detailed citation with all metadata
   */
  createDetailedCitation(reference: Reference, answerText: string): DetailedCitation {
    const { corpus_id, chunk_text, idx } = reference;

    const metadata = corpus_id ? this.getCorpusMetadata(corpus_id) : undefined;
    const provision = this.formatProvision(reference);

    return {
      marker: `[${idx}]`,
      shortCitation: this.format(reference),
      fullLawName: metadata?.displayName ?? reference.display,
      provision: provision || reference.display,
      eurLexUrl: metadata?.eurLexUrl ?? null,
      celexNumber: metadata?.celexNumber ?? null,
      wasCited: this.wasCitedInText(reference, answerText),
      excerpt: chunk_text,
    };
  }
}
