/**
 * EurovocFilterSection component.
 *
 * Single Responsibility: Orchestrate all three layers (suggestions, list, summary).
 */

import { useMemo } from 'react';
import type { CorpusInfo } from '../../types';
import { extractAllSortedLabels, suggestLabels } from './labelUtils';
import { SuggestedLabels } from './SuggestedLabels';
import { EurovocLabelList } from './EurovocLabelList';
import { FilterSummaryBar } from './FilterSummaryBar';

export interface EurovocFilterSectionProps {
  question: string;
  corpora: CorpusInfo[];
  selectedLabels: string[];
  onSelectionChange: (labels: string[]) => void;
  disabled?: boolean;
}

/**
 * Orchestrates three-layer EuroVoc filter UI:
 * 1. SuggestedLabels - AI-powered suggestions based on question
 * 2. EurovocLabelList - Full searchable list of all labels
 * 3. FilterSummaryBar - Applied filters with counts
 */
export function EurovocFilterSection({
  question,
  corpora,
  selectedLabels,
  onSelectionChange,
  disabled = false,
}: EurovocFilterSectionProps) {
  // Extract all labels with counts from corpora
  const allLabels = useMemo(() => extractAllSortedLabels(corpora), [corpora]);

  // Get suggested labels based on question
  const suggestedLabelNames = useMemo(() => {
    const allLabelNames = allLabels.map((l) => l.label);
    return suggestLabels(question, allLabelNames);
  }, [question, allLabels]);

  // Calculate filtered count
  const filteredCount = useMemo(() => {
    if (selectedLabels.length === 0) return corpora.length;

    return corpora.filter((c) =>
      c.eurovoc_labels?.some((label) => selectedLabels.includes(label))
    ).length;
  }, [corpora, selectedLabels]);

  // Don't render if no labels available
  if (allLabels.length === 0) {
    return null;
  }

  // Handle adding a label to selection
  const handleSelect = (label: string) => {
    if (!selectedLabels.includes(label)) {
      onSelectionChange([...selectedLabels, label]);
    }
  };

  // Handle toggling a label (add or remove)
  const handleToggle = (label: string) => {
    if (selectedLabels.includes(label)) {
      onSelectionChange(selectedLabels.filter((l) => l !== label));
    } else {
      onSelectionChange([...selectedLabels, label]);
    }
  };

  // Handle removing a single label
  const handleRemove = (label: string) => {
    onSelectionChange(selectedLabels.filter((l) => l !== label));
  };

  // Handle clearing all selections
  const handleClearAll = () => {
    onSelectionChange([]);
  };

  return (
    <div className="space-y-1">
      {/* Layer 1: AI-powered suggestions (conditional) */}
      <SuggestedLabels
        question={question}
        allLabels={allLabels}
        selectedLabels={selectedLabels}
        onSelect={handleSelect}
        disabled={disabled}
      />

      {/* Layer 2: Full searchable label list (always visible) */}
      <EurovocLabelList
        labels={allLabels}
        selectedLabels={selectedLabels}
        suggestedLabels={suggestedLabelNames}
        onToggle={handleToggle}
        disabled={disabled}
      />

      {/* Layer 3: Applied filter summary (conditional) */}
      <FilterSummaryBar
        selectedLabels={selectedLabels}
        filteredCount={filteredCount}
        totalCount={corpora.length}
        onRemove={handleRemove}
        onClearAll={handleClearAll}
        disabled={disabled}
      />
    </div>
  );
}
