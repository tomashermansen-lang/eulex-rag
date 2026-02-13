/**
 * Tests for labelUtils - pure functions for EuroVoc label handling.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect } from 'vitest';
import { suggestLabels, extractAllSortedLabels, type LabelWithCount } from '../labelUtils';
import type { CorpusInfo } from '../../../types';

describe('suggestLabels', () => {
  const allLabels = [
    'kunstig intelligens',
    'databeskyttelse',
    'personoplysninger',
    'informationssikkerhed',
    'innovation',
    'finansiel teknologi',
    'cybersikkerhed',
  ];

  describe('keyword matching', () => {
    it('returns empty array for empty question', () => {
      expect(suggestLabels('', allLabels)).toEqual([]);
    });

    it('returns empty array for whitespace-only question', () => {
      expect(suggestLabels('   ', allLabels)).toEqual([]);
    });

    it('matches labels containing question keywords', () => {
      const result = suggestLabels('kunstig intelligens regler', allLabels);
      expect(result).toContain('kunstig intelligens');
    });

    it('matches partial keywords (case-insensitive)', () => {
      const result = suggestLabels('INNOVATION i virksomheder', allLabels);
      expect(result).toContain('innovation');
    });

    it('ignores short words (<=2 chars)', () => {
      const result = suggestLabels('AI og ML', allLabels);
      // 'AI', 'og', 'ML' are all <=2 chars, should return empty
      expect(result).toEqual([]);
    });

    it('matches multiple labels from same question', () => {
      const result = suggestLabels('databeskyttelse og informationssikkerhed', allLabels);
      expect(result).toContain('databeskyttelse');
      expect(result).toContain('informationssikkerhed');
    });

    it('limits results to max 5 suggestions', () => {
      // Question that could match many labels
      const manyLabels = Array.from({ length: 10 }, (_, i) => `term${i}`);
      const question = manyLabels.join(' ');
      const result = suggestLabels(question, manyLabels);
      expect(result.length).toBeLessThanOrEqual(5);
    });

    it('returns empty array when no labels match', () => {
      const result = suggestLabels('noget helt andet', allLabels);
      expect(result).toEqual([]);
    });
  });
});

describe('extractAllSortedLabels', () => {
  const mockCorpora: CorpusInfo[] = [
    { id: 'gdpr', name: 'GDPR', eurovoc_labels: ['databeskyttelse', 'personoplysninger'] },
    { id: 'ai-act', name: 'AI Act', eurovoc_labels: ['kunstig intelligens', 'innovation'] },
    { id: 'nis2', name: 'NIS2', eurovoc_labels: ['cybersikkerhed', 'informationssikkerhed'] },
    { id: 'dora', name: 'DORA', eurovoc_labels: ['finansiel teknologi', 'informationssikkerhed'] },
  ];

  describe('label extraction', () => {
    it('extracts all unique labels from corpora', () => {
      const result = extractAllSortedLabels(mockCorpora);
      const labels = result.map((r) => r.label);

      expect(labels).toContain('databeskyttelse');
      expect(labels).toContain('kunstig intelligens');
      expect(labels).toContain('informationssikkerhed');
      // informationssikkerhed appears twice but should only be in result once
      expect(labels.filter((l) => l === 'informationssikkerhed')).toHaveLength(1);
    });

    it('returns empty array when no corpora provided', () => {
      expect(extractAllSortedLabels([])).toEqual([]);
    });

    it('returns empty array when corpora have no eurovoc_labels', () => {
      const corporaWithoutLabels: CorpusInfo[] = [{ id: 'test', name: 'Test' }];
      expect(extractAllSortedLabels(corporaWithoutLabels)).toEqual([]);
    });

    it('handles corpora with undefined eurovoc_labels', () => {
      const mixedCorpora: CorpusInfo[] = [
        { id: 'a', name: 'A', eurovoc_labels: ['label1'] },
        { id: 'b', name: 'B' }, // no eurovoc_labels
      ];
      const result = extractAllSortedLabels(mixedCorpora);
      expect(result).toHaveLength(1);
      expect(result[0].label).toBe('label1');
    });
  });

  describe('counting', () => {
    it('counts label occurrences across corpora', () => {
      const result = extractAllSortedLabels(mockCorpora);
      const infoSec = result.find((r) => r.label === 'informationssikkerhed');

      // informationssikkerhed appears in NIS2 and DORA
      expect(infoSec?.count).toBe(2);
    });

    it('counts unique labels as 1', () => {
      const result = extractAllSortedLabels(mockCorpora);
      const dataprotection = result.find((r) => r.label === 'databeskyttelse');

      // databeskyttelse only appears in GDPR
      expect(dataprotection?.count).toBe(1);
    });
  });

  describe('sorting', () => {
    it('sorts by frequency descending', () => {
      const result = extractAllSortedLabels(mockCorpora);

      // informationssikkerhed (count: 2) should come before databeskyttelse (count: 1)
      const infoSecIndex = result.findIndex((r) => r.label === 'informationssikkerhed');
      const dataIndex = result.findIndex((r) => r.label === 'databeskyttelse');

      expect(infoSecIndex).toBeLessThan(dataIndex);
    });

    it('sorts alphabetically when frequency is equal', () => {
      const corporaEqualFreq: CorpusInfo[] = [
        { id: 'a', name: 'A', eurovoc_labels: ['beta', 'alpha'] },
      ];
      const result = extractAllSortedLabels(corporaEqualFreq);

      expect(result[0].label).toBe('alpha');
      expect(result[1].label).toBe('beta');
    });
  });

  describe('return type', () => {
    it('returns LabelWithCount objects', () => {
      const result = extractAllSortedLabels(mockCorpora);

      expect(result.length).toBeGreaterThan(0);
      expect(result[0]).toHaveProperty('label');
      expect(result[0]).toHaveProperty('count');
      expect(typeof result[0].label).toBe('string');
      expect(typeof result[0].count).toBe('number');
    });
  });
});
