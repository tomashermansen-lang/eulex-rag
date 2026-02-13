/**
 * Category-grouped law list component.
 *
 * Single Responsibility: Render laws grouped by category with collapsible sections.
 */

import { useMemo } from 'react';
import { CollapsibleSection } from './CollapsibleSection';
import { LawListItem, type CheckboxMode } from './LawListItem';
import { useLawCategories } from '../../hooks';
import type { CorpusInfo } from '../../types';

export interface CategoryLawListProps {
  /** List of available corpora */
  corpora: CorpusInfo[];
  /** Currently selected corpus IDs */
  targetCorpora: string[];
  /** Checkbox display mode */
  checkboxMode: CheckboxMode;
  /** Callback when a law is clicked */
  onLawClick: (corpusId: string) => void;
  /** Whether the list is disabled */
  disabled?: boolean;
  /** Set of expanded category IDs */
  expandedCategories?: Set<string>;
  /** Callback when a category is toggled */
  onCategoryToggle?: (categoryId: string) => void;
}

/**
 * Renders laws grouped by category with collapsible sections.
 */
export function CategoryLawList({
  corpora,
  targetCorpora,
  checkboxMode,
  onLawClick,
  disabled = false,
  expandedCategories = new Set(),
  onCategoryToggle,
}: CategoryLawListProps) {
  const {
    categories,
    lawsByCategory,
    getCategoryDisplayName,
    getCategoryCount,
  } = useLawCategories(corpora);

  // Compute which categories should be expanded (either explicitly or because they contain a selected law)
  const categoriesWithSelectedLaws = useMemo(() => {
    const result = new Set<string>();
    for (const corpusId of targetCorpora) {
      const corpus = corpora.find((c) => c.id === corpusId);
      if (corpus) {
        // Use first eurovoc_label as primary category
        const category = (corpus.eurovoc_labels && corpus.eurovoc_labels[0]) || 'uncategorized';
        result.add(category);
      }
    }
    return result;
  }, [corpora, targetCorpora]);

  // Check if a law is selected
  const isSelected = (corpusId: string): boolean => {
    return targetCorpora.includes(corpusId);
  };

  // Determine if a category should be open
  const isCategoryOpen = (category: string): boolean => {
    return expandedCategories.has(category) || categoriesWithSelectedLaws.has(category);
  };

  // Handle category toggle
  const handleCategoryToggle = (category: string) => {
    onCategoryToggle?.(category);
  };

  if (corpora.length === 0) {
    return <div />;
  }

  return (
    <>
      {categories.map((category) => {
        const laws = lawsByCategory[category] || [];
        const displayName = getCategoryDisplayName(category);
        const count = getCategoryCount(category);

        return (
          <CollapsibleSection
            key={category}
            title={displayName}
            count={count}
            defaultOpen={isCategoryOpen(category)}
            onToggle={() => handleCategoryToggle(category)}
          >
            {laws.map((corpus) => (
              <LawListItem
                key={corpus.id}
                corpus={corpus}
                isSelected={isSelected(corpus.id)}
                checkboxMode={checkboxMode}
                onClick={() => onLawClick(corpus.id)}
                disabled={disabled}
              />
            ))}
          </CollapsibleSection>
        );
      })}
    </>
  );
}
