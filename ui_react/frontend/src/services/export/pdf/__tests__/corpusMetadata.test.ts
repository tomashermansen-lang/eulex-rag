/**
 * Tests for corpus metadata registry.
 *
 * TDD: These tests verify corpus metadata mapping for legal citations.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  getCorpusMetadata,
  getCorpusId,
  getCorpusDisplayName,
  getCorpusFullname,
  setCorpusRegistry,
  CORPUS_METADATA,
} from '../data/corpusMetadata';
import type { CorpusInfo } from '../../../../types';

// Test data simulating API response
const testCorpora: CorpusInfo[] = [
  {
    id: 'gdpr',
    name: 'Persondataforordningen (GDPR)',
    fullname: 'Europa-Parlamentets og Rådets forordning (EU) 2016/679 af 27. april 2016 om beskyttelse af fysiske personer i forbindelse med behandling af personoplysninger',
    source_url: 'https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32016R0679',
    celex_number: '32016R0679',
  },
  {
    id: 'ai-act',
    name: 'Forordning om kunstig intelligens (AI-ACT)',
    fullname: 'Europa-Parlamentets og Rådets forordning (EU) 2024/1689 af 13. juni 2024 om harmoniserede regler for kunstig intelligens',
    source_url: 'https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32024R1689',
    celex_number: '32024R1689',
  },
  {
    id: 'dora',
    name: 'Forordning om digital operationel modstandsdygtighed (DORA)',
    source_url: 'https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32022R2554',
    celex_number: '32022R2554',
  },
];

describe('corpusMetadata', () => {
  // Populate the registry before each test
  beforeEach(() => {
    setCorpusRegistry(testCorpora);
  });

  describe('setCorpusRegistry()', () => {
    it('populates the registry from API data', () => {
      const metadata = getCorpusMetadata('gdpr');
      expect(metadata).toBeDefined();
      expect(metadata?.celexNumber).toBe('32016R0679');
    });

    it('stores fullname from API data', () => {
      const metadata = getCorpusMetadata('gdpr');
      expect(metadata?.fullname).toBe('Europa-Parlamentets og Rådets forordning (EU) 2016/679 af 27. april 2016 om beskyttelse af fysiske personer i forbindelse med behandling af personoplysninger');
    });

    it('uses source_url from corpus data (HTML format for anchor support)', () => {
      const metadata = getCorpusMetadata('gdpr');
      expect(metadata?.eurLexUrl).toBe('https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32016R0679');
    });

    it('handles corpus without celex_number', () => {
      setCorpusRegistry([
        { id: 'test', name: 'Test Law (TEST)' },
      ]);
      const metadata = getCorpusMetadata('test');
      expect(metadata).toBeDefined();
      expect(metadata?.celexNumber).toBeNull();
      expect(metadata?.eurLexUrl).toBeNull();
    });

    it('handles corpus without fullname', () => {
      const metadata = getCorpusMetadata('dora');
      expect(metadata?.fullname).toBeNull();
    });
  });

  describe('CORPUS_METADATA proxy', () => {
    it('provides access to registered corpora', () => {
      expect(CORPUS_METADATA['gdpr']).toBeDefined();
      expect(CORPUS_METADATA['gdpr'].celexNumber).toBe('32016R0679');
      expect(CORPUS_METADATA['gdpr'].corpusId).toBe('gdpr');
    });

    it('returns undefined for unknown corpus', () => {
      expect(CORPUS_METADATA['unknown']).toBeUndefined();
    });
  });

  describe('getCorpusMetadata()', () => {
    it('returns metadata for known corpus', () => {
      const metadata = getCorpusMetadata('gdpr');
      expect(metadata).toBeDefined();
      expect(metadata?.displayName).toBe('Persondataforordningen (GDPR)');
    });

    it('returns undefined for unknown corpus', () => {
      const metadata = getCorpusMetadata('unknown-law');
      expect(metadata).toBeUndefined();
    });

    it('normalizes underscores to hyphens', () => {
      const metadata = getCorpusMetadata('ai_act');
      expect(metadata).toBeDefined();
      expect(metadata?.corpusId).toBe('ai-act');
    });

    it('is case-insensitive', () => {
      const metadata = getCorpusMetadata('GDPR');
      expect(metadata).toBeDefined();
    });
  });

  describe('getCorpusId()', () => {
    it('returns corpus ID for GDPR', () => {
      expect(getCorpusId('gdpr')).toBe('gdpr');
    });

    it('returns corpus ID for AI Act', () => {
      expect(getCorpusId('ai-act')).toBe('ai-act');
    });

    it('returns input for unknown corpus', () => {
      expect(getCorpusId('unknown')).toBe('unknown');
    });
  });

  describe('getCorpusDisplayName()', () => {
    it('returns full name for GDPR', () => {
      expect(getCorpusDisplayName('gdpr')).toBe('Persondataforordningen (GDPR)');
    });

    it('returns full name for DORA', () => {
      expect(getCorpusDisplayName('dora')).toBe('Forordning om digital operationel modstandsdygtighed (DORA)');
    });

    it('returns corpus ID for unknown corpus', () => {
      expect(getCorpusDisplayName('unknown')).toBe('unknown');
    });
  });

  describe('getCorpusFullname()', () => {
    it('returns full legal title for GDPR', () => {
      expect(getCorpusFullname('gdpr')).toBe('Europa-Parlamentets og Rådets forordning (EU) 2016/679 af 27. april 2016 om beskyttelse af fysiske personer i forbindelse med behandling af personoplysninger');
    });

    it('returns display name when fullname is not available', () => {
      expect(getCorpusFullname('dora')).toBe('Forordning om digital operationel modstandsdygtighed (DORA)');
    });

    it('returns corpus ID for unknown corpus', () => {
      expect(getCorpusFullname('unknown')).toBe('unknown');
    });
  });
});
