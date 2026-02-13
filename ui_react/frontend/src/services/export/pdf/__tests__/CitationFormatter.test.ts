/**
 * Tests for EuLegalCitationFormatter.
 *
 * TDD: These tests define EU legal citation formatting requirements.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { EuLegalCitationFormatter } from '../formatters/CitationFormatter';
import { setCorpusRegistry } from '../data/corpusMetadata';
import type { Reference, CorpusInfo } from '../../../../types';

// Test data simulating API response
const testCorpora: CorpusInfo[] = [
  {
    id: 'gdpr',
    name: 'Persondataforordningen (GDPR)',
    source_url: 'https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32016R0679',
    celex_number: '32016R0679',
  },
  {
    id: 'ai-act',
    name: 'Forordning om kunstig intelligens (AI Act)',
    source_url: 'https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32024R1689',
    celex_number: '32024R1689',
  },
  {
    id: 'data-act',
    name: 'Forordning om harmoniserede regler for dataadgang (Data Act)',
    source_url: 'https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32023R2854',
    celex_number: '32023R2854',
  },
  {
    id: 'dora',
    name: 'Forordning om digital operationel modstandsdygtighed (DORA)',
    source_url: 'https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32022R2554',
    celex_number: '32022R2554',
  },
  {
    id: 'nis2',
    name: 'Direktiv om cybersikkerhed (NIS2)',
    source_url: 'https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32022L2555',
    celex_number: '32022L2555',
  },
];

describe('EuLegalCitationFormatter', () => {
  const formatter = new EuLegalCitationFormatter();

  // Populate the registry before each test
  beforeEach(() => {
    setCorpusRegistry(testCorpora);
  });

  describe('format()', () => {
    it('formats GDPR reference with article and paragraph', () => {
      const ref: Reference = {
        idx: 1,
        display: 'Artikel 30, stk. 1',
        chunk_text: 'Test text',
        corpus_id: 'gdpr',
        article: '30',
        paragraph: '1',
      };
      expect(formatter.format(ref)).toBe('Persondataforordningen (GDPR), Artikel 30, stk. 1');
    });

    it('formats AI Act reference correctly', () => {
      const ref: Reference = {
        idx: 2,
        display: 'Artikel 6',
        chunk_text: 'Test text',
        corpus_id: 'ai-act',
        article: '6',
      };
      expect(formatter.format(ref)).toBe('Forordning om kunstig intelligens (AI Act), Artikel 6');
    });

    it('formats Data Act reference with paragraph', () => {
      const ref: Reference = {
        idx: 3,
        display: 'Artikel 3, stk. 3',
        chunk_text: 'Test text',
        corpus_id: 'data-act',
        article: '3',
        paragraph: '3',
      };
      expect(formatter.format(ref)).toBe('Forordning om harmoniserede regler for dataadgang (Data Act), Artikel 3, stk. 3');
    });

    it('formats DORA reference', () => {
      const ref: Reference = {
        idx: 4,
        display: 'Artikel 5',
        chunk_text: 'Test text',
        corpus_id: 'dora',
        article: '5',
      };
      expect(formatter.format(ref)).toBe('Forordning om digital operationel modstandsdygtighed (DORA), Artikel 5');
    });

    it('formats NIS2 reference', () => {
      const ref: Reference = {
        idx: 5,
        display: 'Artikel 21',
        chunk_text: 'Test text',
        corpus_id: 'nis2',
        article: '21',
      };
      expect(formatter.format(ref)).toBe('Direktiv om cybersikkerhed (NIS2), Artikel 21');
    });

    it('handles litra (a, b, c subsections)', () => {
      const ref: Reference = {
        idx: 6,
        display: 'Artikel 5, stk. 1, litra a',
        chunk_text: 'Test text',
        corpus_id: 'gdpr',
        article: '5',
        paragraph: '1',
        litra: 'a',
      };
      expect(formatter.format(ref)).toBe('Persondataforordningen (GDPR), Artikel 5, stk. 1, litra a');
    });

    it('handles recitals (betragtninger)', () => {
      const ref: Reference = {
        idx: 7,
        display: 'Betragtning 26',
        chunk_text: 'Test text',
        corpus_id: 'gdpr',
        recital: '26',
      };
      expect(formatter.format(ref)).toBe('Persondataforordningen (GDPR), Betragtning 26');
    });

    it('handles annexes (bilag)', () => {
      const ref: Reference = {
        idx: 8,
        display: 'Bilag I',
        chunk_text: 'Test text',
        corpus_id: 'ai-act',
        annex: 'I',
      };
      expect(formatter.format(ref)).toBe('Forordning om kunstig intelligens (AI Act), Bilag I');
    });

    it('falls back to display when corpus_id is missing', () => {
      const ref: Reference = {
        idx: 9,
        display: 'Some Unknown Reference',
        chunk_text: 'Test text',
      };
      expect(formatter.format(ref)).toBe('Some Unknown Reference');
    });

    it('falls back to display for unknown corpus', () => {
      const ref: Reference = {
        idx: 10,
        display: 'Artikel 99',
        chunk_text: 'Test text',
        corpus_id: 'unknown-law',
        article: '99',
      };
      expect(formatter.format(ref)).toBe('unknown-law, Artikel 99');
    });
  });

  describe('formatInline()', () => {
    it('formats index as bracketed superscript marker', () => {
      expect(formatter.formatInline(1)).toBe('[1]');
    });

    it('handles double-digit indices', () => {
      expect(formatter.formatInline(12)).toBe('[12]');
    });

    it('handles zero index', () => {
      expect(formatter.formatInline(0)).toBe('[0]');
    });
  });

  describe('getCorpusDisplayName()', () => {
    it('maps gdpr to full display name', () => {
      expect(formatter.getCorpusDisplayName('gdpr')).toBe('Persondataforordningen (GDPR)');
    });

    it('maps ai-act to full display name', () => {
      expect(formatter.getCorpusDisplayName('ai-act')).toBe('Forordning om kunstig intelligens (AI Act)');
    });

    it('maps data-act to full display name', () => {
      expect(formatter.getCorpusDisplayName('data-act')).toBe('Forordning om harmoniserede regler for dataadgang (Data Act)');
    });

    it('maps dora to full display name', () => {
      expect(formatter.getCorpusDisplayName('dora')).toBe('Forordning om digital operationel modstandsdygtighed (DORA)');
    });

    it('maps nis2 to full display name', () => {
      expect(formatter.getCorpusDisplayName('nis2')).toBe('Direktiv om cybersikkerhed (NIS2)');
    });

    it('returns corpus_id for unknown corpus', () => {
      expect(formatter.getCorpusDisplayName('some-law')).toBe('some-law');
    });
  });

  describe('wasCitedInText()', () => {
    it('returns true when marker is in text', () => {
      const ref: Reference = { idx: 1, display: 'Test', chunk_text: '' };
      const text = 'This is some text [1] with a citation.';
      expect(formatter.wasCitedInText(ref, text)).toBe(true);
    });

    it('returns false when marker is not in text', () => {
      const ref: Reference = { idx: 2, display: 'Test', chunk_text: '' };
      const text = 'This is some text [1] with a different citation.';
      expect(formatter.wasCitedInText(ref, text)).toBe(false);
    });

    it('handles string indices', () => {
      const ref: Reference = { idx: '1a', display: 'Test', chunk_text: '' };
      const text = 'Text with [1a] marker.';
      expect(formatter.wasCitedInText(ref, text)).toBe(true);
    });
  });

  describe('createDetailedCitation()', () => {
    it('creates detailed citation with full metadata', () => {
      const ref: Reference = {
        idx: 1,
        display: 'Artikel 30, stk. 1',
        chunk_text: 'Source excerpt text',
        corpus_id: 'gdpr',
        article: '30',
        paragraph: '1',
      };
      const answerText = 'The answer text [1] cites this source.';

      const citation = formatter.createDetailedCitation(ref, answerText);

      expect(citation.marker).toBe('[1]');
      expect(citation.shortCitation).toBe('Persondataforordningen (GDPR), Artikel 30, stk. 1');
      expect(citation.fullLawName).toBe('Persondataforordningen (GDPR)');
      expect(citation.provision).toBe('Artikel 30, stk. 1');
      expect(citation.eurLexUrl).toContain('eur-lex.europa.eu');
      expect(citation.celexNumber).toBe('32016R0679');
      expect(citation.wasCited).toBe(true);
      expect(citation.excerpt).toBe('Source excerpt text');
    });

    it('marks uncited references correctly', () => {
      const ref: Reference = {
        idx: 2,
        display: 'Artikel 5',
        chunk_text: 'Text',
        corpus_id: 'gdpr',
        article: '5',
      };
      const answerText = 'The answer only cites [1].';

      const citation = formatter.createDetailedCitation(ref, answerText);
      expect(citation.wasCited).toBe(false);
    });

    it('handles unknown corpus gracefully', () => {
      const ref: Reference = {
        idx: 1,
        display: 'Unknown Reference',
        chunk_text: 'Text',
        corpus_id: 'unknown-law',
      };

      const citation = formatter.createDetailedCitation(ref, '');

      expect(citation.fullLawName).toBe('Unknown Reference');
      expect(citation.eurLexUrl).toBeNull();
      expect(citation.celexNumber).toBeNull();
    });
  });
});
