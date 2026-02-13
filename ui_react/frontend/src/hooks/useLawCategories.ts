/**
 * Hook for grouping and filtering laws by category.
 *
 * Single Responsibility: Provide category-based organization of laws.
 */

import { useMemo, useCallback } from 'react';
import type { CorpusInfo } from '../types';

/** Category display name mapping (Danish) */
const CATEGORY_DISPLAY_NAMES: Record<string, string> = {
  cybersikkerhed: 'Cybersikkerhed',
  databeskyttelse: 'Databeskyttelse',
  'kunstig-intelligens': 'Kunstig intelligens',
  dataokonomi: 'Dataøkonomi',
  'finansiel-regulering': 'Finansiel regulering',
  forsvar: 'Forsvar',
  innovation: 'Innovation',
  uncategorized: 'Øvrige',
};

export interface UseLawCategoriesReturn {
  /** List of unique categories, sorted alphabetically with 'uncategorized' last */
  categories: string[];
  /** Laws grouped by category */
  lawsByCategory: Record<string, CorpusInfo[]>;
  /** Get display name for a category */
  getCategoryDisplayName: (category: string) => string;
  /** Get count of laws in a category */
  getCategoryCount: (category: string) => number;
  /** Filter laws by search term */
  filterLaws: (searchTerm: string) => CorpusInfo[];
}

/**
 * Format a category ID to a display name.
 * Replaces hyphens with spaces and capitalizes first letter.
 */
function formatCategoryName(category: string): string {
  if (CATEGORY_DISPLAY_NAMES[category]) {
    return CATEGORY_DISPLAY_NAMES[category];
  }

  // Format: some-category -> Some category
  return category
    .replace(/-/g, ' ')
    .replace(/^\w/, (c) => c.toUpperCase());
}

/**
 * Hook for organizing laws by category.
 *
 * @param corpora - List of available corpora
 * @returns Category organization utilities
 */
export function useLawCategories(corpora: CorpusInfo[]): UseLawCategoriesReturn {
  // Group laws by category (uses first eurovoc_label as primary category)
  const lawsByCategory = useMemo(() => {
    const grouped: Record<string, CorpusInfo[]> = {};

    for (const corpus of corpora) {
      // Use first eurovoc label as primary category, fallback to uncategorized
      const category = (corpus.eurovoc_labels && corpus.eurovoc_labels[0]) || 'uncategorized';
      if (!grouped[category]) {
        grouped[category] = [];
      }
      grouped[category].push(corpus);
    }

    // Sort laws within each category alphabetically by name
    for (const category of Object.keys(grouped)) {
      grouped[category].sort((a, b) => a.name.localeCompare(b.name, 'da'));
    }

    return grouped;
  }, [corpora]);

  // Get sorted list of categories (alphabetically by display name, uncategorized last)
  const categories = useMemo(() => {
    const cats = Object.keys(lawsByCategory);

    return cats.sort((a, b) => {
      // Uncategorized always last
      if (a === 'uncategorized') return 1;
      if (b === 'uncategorized') return -1;

      // Sort by display name
      const nameA = formatCategoryName(a);
      const nameB = formatCategoryName(b);
      return nameA.localeCompare(nameB, 'da');
    });
  }, [lawsByCategory]);

  // Get display name for a category
  const getCategoryDisplayName = useCallback((category: string): string => {
    return formatCategoryName(category);
  }, []);

  // Get count of laws in a category
  const getCategoryCount = useCallback(
    (category: string): number => {
      return lawsByCategory[category]?.length || 0;
    },
    [lawsByCategory]
  );

  // Filter laws by search term
  const filterLaws = useCallback(
    (searchTerm: string): CorpusInfo[] => {
      if (!searchTerm.trim()) {
        return corpora;
      }

      const term = searchTerm.toLowerCase();
      return corpora.filter((corpus) =>
        corpus.name.toLowerCase().includes(term) ||
        corpus.id.toLowerCase().includes(term)
      );
    },
    [corpora]
  );

  return {
    categories,
    lawsByCategory,
    getCategoryDisplayName,
    getCategoryCount,
    filterLaws,
  };
}
