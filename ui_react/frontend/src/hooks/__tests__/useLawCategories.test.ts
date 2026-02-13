/**
 * Tests for useLawCategories hook.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useLawCategories } from '../useLawCategories';
import type { CorpusInfo } from '../../types';

describe('useLawCategories', () => {
  const mockCorpora: CorpusInfo[] = [
    { id: 'gdpr', name: 'GDPR', eurovoc_labels: ['databeskyttelse'] },
    { id: 'ai-act', name: 'AI Act', eurovoc_labels: ['kunstig-intelligens'] },
    { id: 'nis2', name: 'NIS2', eurovoc_labels: ['cybersikkerhed'] },
    { id: 'dora', name: 'DORA', eurovoc_labels: ['finansiel-regulering'] },
    { id: 'cyberrobusthed', name: 'Cyberrobusthed', eurovoc_labels: ['cybersikkerhed'] },
    { id: 'no-category', name: 'No Category Law' }, // No eurovoc_labels
  ];

  describe('groupByCategory', () => {
    it('groups laws by their category', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      expect(result.current.categories).toContain('cybersikkerhed');
      expect(result.current.categories).toContain('databeskyttelse');
      expect(result.current.categories).toContain('kunstig-intelligens');
      expect(result.current.categories).toContain('finansiel-regulering');
    });

    it('returns laws grouped by category', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      const cyberLaws = result.current.lawsByCategory['cybersikkerhed'];
      expect(cyberLaws).toHaveLength(2);
      expect(cyberLaws.map((l) => l.id)).toContain('nis2');
      expect(cyberLaws.map((l) => l.id)).toContain('cyberrobusthed');
    });

    it('puts laws without category in uncategorized group', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      const uncategorized = result.current.lawsByCategory['uncategorized'];
      expect(uncategorized).toHaveLength(1);
      expect(uncategorized[0].id).toBe('no-category');
    });

    it('returns empty categories for empty corpora list', () => {
      const { result } = renderHook(() => useLawCategories([]));

      expect(result.current.categories).toHaveLength(0);
      expect(Object.keys(result.current.lawsByCategory)).toHaveLength(0);
    });
  });

  describe('category metadata', () => {
    it('provides category display names', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      expect(result.current.getCategoryDisplayName('cybersikkerhed')).toBe('Cybersikkerhed');
      expect(result.current.getCategoryDisplayName('databeskyttelse')).toBe('Databeskyttelse');
      expect(result.current.getCategoryDisplayName('kunstig-intelligens')).toBe('Kunstig intelligens');
      expect(result.current.getCategoryDisplayName('finansiel-regulering')).toBe('Finansiel regulering');
    });

    it('returns formatted name for unknown categories', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      expect(result.current.getCategoryDisplayName('some-new-category')).toBe('Some new category');
    });

    it('provides count per category', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      expect(result.current.getCategoryCount('cybersikkerhed')).toBe(2);
      expect(result.current.getCategoryCount('databeskyttelse')).toBe(1);
      expect(result.current.getCategoryCount('uncategorized')).toBe(1);
    });
  });

  describe('sorting', () => {
    it('sorts categories alphabetically by display name', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      const categories = result.current.categories;
      // Uncategorized should be last
      expect(categories[categories.length - 1]).toBe('uncategorized');

      // Others should be alphabetical (by display name)
      const displayNames = categories
        .filter((c) => c !== 'uncategorized')
        .map((c) => result.current.getCategoryDisplayName(c));

      const sorted = [...displayNames].sort((a, b) => a.localeCompare(b, 'da'));
      expect(displayNames).toEqual(sorted);
    });

    it('sorts laws within each category alphabetically by name', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      const cyberLaws = result.current.lawsByCategory['cybersikkerhed'];
      const names = cyberLaws.map((l) => l.name);
      const sorted = [...names].sort((a, b) => a.localeCompare(b, 'da'));
      expect(names).toEqual(sorted);
    });
  });

  describe('filtering', () => {
    it('filters laws by search term', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      const filtered = result.current.filterLaws('gdpr');
      expect(filtered).toHaveLength(1);
      expect(filtered[0].id).toBe('gdpr');
    });

    it('filters case-insensitively', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      const filtered = result.current.filterLaws('GDPR');
      expect(filtered).toHaveLength(1);
      expect(filtered[0].id).toBe('gdpr');
    });

    it('returns all laws when search term is empty', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      const filtered = result.current.filterLaws('');
      expect(filtered).toHaveLength(mockCorpora.length);
    });

    it('filters by partial match on name', () => {
      const { result } = renderHook(() => useLawCategories(mockCorpora));

      const filtered = result.current.filterLaws('cyber');
      expect(filtered).toHaveLength(1);
      expect(filtered[0].id).toBe('cyberrobusthed');
    });
  });

  describe('memoization', () => {
    it('returns stable references when corpora do not change', () => {
      const { result, rerender } = renderHook(() => useLawCategories(mockCorpora));

      const firstCategories = result.current.categories;
      const firstLawsByCategory = result.current.lawsByCategory;

      rerender();

      expect(result.current.categories).toBe(firstCategories);
      expect(result.current.lawsByCategory).toBe(firstLawsByCategory);
    });
  });
});
