/**
 * Tests for LawSelectorPanel component.
 *
 * TDD: Tests for unified law selector interface.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { LawSelectorPanel } from '../LawSelectorPanel';
import type { CorpusInfo } from '../../../types';
import type { CheckboxMode } from '../LawListItem';

// Mock child components
vi.mock('../../Common/SearchInput', () => ({
  SearchInput: ({ value, onChange, placeholder, disabled }: {
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
    disabled?: boolean;
  }) => (
    <input
      data-testid="law-search-input"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
    />
  ),
}));

vi.mock('../LawListItem', () => ({
  LawListItem: ({ corpus, isSelected, checkboxMode, onClick, disabled }: {
    corpus: CorpusInfo;
    isSelected: boolean;
    checkboxMode: CheckboxMode;
    onClick: () => void;
    disabled?: boolean;
  }) => (
    <button
      data-testid={`law-item-${corpus.id}`}
      data-selected={isSelected}
      data-checkbox-mode={checkboxMode}
      onClick={onClick}
      disabled={disabled}
    >
      {corpus.id}
    </button>
  ),
}));

vi.mock('../HiddenSelectionSummary', () => ({
  HiddenSelectionSummary: ({ hiddenLaws, onRemove, onClearFilter, disabled }: {
    hiddenLaws: Array<{ id: string; name: string }>;
    onRemove: (id: string) => void;
    onClearFilter: () => void;
    disabled?: boolean;
  }) => hiddenLaws.length > 0 ? (
    <div data-testid="hidden-selection-summary">
      {hiddenLaws.map(law => (
        <button
          key={law.id}
          data-testid={`hidden-law-${law.id}`}
          onClick={() => onRemove(law.id)}
        >
          {law.name}
        </button>
      ))}
      <button data-testid="clear-filter-btn" onClick={onClearFilter} disabled={disabled}>
        Ryd emnefilter
      </button>
    </div>
  ) : null,
}));

describe('LawSelectorPanel', () => {
  const mockCorpora: CorpusInfo[] = [
    { id: 'ai-act', name: 'AI-forordningen (AI Act)' },
    { id: 'gdpr', name: 'Persondataforordningen (GDPR)' },
    { id: 'nis2', name: 'NIS2-direktivet' },
  ];

  const defaultProps = {
    corpusScope: 'single' as const,
    corpora: mockCorpora,
    targetCorpora: [],
    onTargetCorporaChange: vi.fn(),
    disabled: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('single mode (corpusScope="single")', () => {
    it('renders search input', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="single" />);

      expect(screen.getByTestId('law-search-input')).toBeInTheDocument();
    });

    it('renders law list with radio checkboxMode', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="single" />);

      const aiActItem = screen.getByTestId('law-item-ai-act');
      expect(aiActItem).toHaveAttribute('data-checkbox-mode', 'radio');
    });

    it('selecting a law updates targetCorpora with single item', () => {
      const onTargetCorporaChange = vi.fn();
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpusScope="single"
          onTargetCorporaChange={onTargetCorporaChange}
        />
      );

      fireEvent.click(screen.getByTestId('law-item-gdpr'));

      expect(onTargetCorporaChange).toHaveBeenCalledWith(['gdpr']);
    });

    it('selecting different law replaces previous', () => {
      const onTargetCorporaChange = vi.fn();
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpusScope="single"
          targetCorpora={['ai-act']}
          onTargetCorporaChange={onTargetCorporaChange}
        />
      );

      fireEvent.click(screen.getByTestId('law-item-gdpr'));

      expect(onTargetCorporaChange).toHaveBeenCalledWith(['gdpr']);
    });

    it('selected law is marked as selected', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpusScope="single"
          targetCorpora={['ai-act']}
        />
      );

      expect(screen.getByTestId('law-item-ai-act')).toHaveAttribute('data-selected', 'true');
      expect(screen.getByTestId('law-item-gdpr')).toHaveAttribute('data-selected', 'false');
    });

    it('search filters laws', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="single" />);

      fireEvent.change(screen.getByTestId('law-search-input'), { target: { value: 'gdpr' } });

      expect(screen.getByTestId('law-item-gdpr')).toBeInTheDocument();
      expect(screen.queryByTestId('law-item-ai-act')).not.toBeInTheDocument();
    });

    it('shows empty state when search has no results', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="single" />);

      fireEvent.change(screen.getByTestId('law-search-input'), { target: { value: 'nonexistent' } });

      expect(screen.getByText(/ingen love matcher/i)).toBeInTheDocument();
    });

    it('shows warning when no law selected', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpusScope="single"
          targetCorpora={[]}
        />
      );

      expect(screen.getByText(/vælg en lov/i)).toBeInTheDocument();
    });
  });

  describe('explicit mode (corpusScope="explicit")', () => {
    it('renders search input', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="explicit" />);

      expect(screen.getByTestId('law-search-input')).toBeInTheDocument();
    });

    it('renders law list with checkbox checkboxMode', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="explicit" />);

      const aiActItem = screen.getByTestId('law-item-ai-act');
      expect(aiActItem).toHaveAttribute('data-checkbox-mode', 'checkbox');
    });

    it('checking a law adds to targetCorpora', () => {
      const onTargetCorporaChange = vi.fn();
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpusScope="explicit"
          targetCorpora={['ai-act']}
          onTargetCorporaChange={onTargetCorporaChange}
        />
      );

      fireEvent.click(screen.getByTestId('law-item-gdpr'));

      expect(onTargetCorporaChange).toHaveBeenCalledWith(['ai-act', 'gdpr']);
    });

    it('unchecking a law removes from targetCorpora', () => {
      const onTargetCorporaChange = vi.fn();
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpusScope="explicit"
          targetCorpora={['ai-act', 'gdpr']}
          onTargetCorporaChange={onTargetCorporaChange}
        />
      );

      fireEvent.click(screen.getByTestId('law-item-gdpr'));

      expect(onTargetCorporaChange).toHaveBeenCalledWith(['ai-act']);
    });

    it('renders flat list without section headers', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpusScope="explicit"
          targetCorpora={['ai-act']}
        />
      );

      // All laws should be visible in a flat list
      expect(screen.getByTestId('law-item-ai-act')).toBeInTheDocument();
      expect(screen.getByTestId('law-item-gdpr')).toBeInTheDocument();
      expect(screen.getByTestId('law-item-nis2')).toBeInTheDocument();
    });

    it('shows warning when no laws selected', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpusScope="explicit"
          targetCorpora={[]}
        />
      );

      expect(screen.getByText(/vælg mindst én lov/i)).toBeInTheDocument();
    });
  });

  describe('all mode (corpusScope="all")', () => {
    it('renders search input', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="all" />);

      expect(screen.getByTestId('law-search-input')).toBeInTheDocument();
    });

    it('renders all laws with disabled checkboxMode', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="all" />);

      const aiActItem = screen.getByTestId('law-item-ai-act');
      expect(aiActItem).toHaveAttribute('data-checkbox-mode', 'disabled');
    });

    it('all laws are marked as selected', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="all" />);

      expect(screen.getByTestId('law-item-ai-act')).toHaveAttribute('data-selected', 'true');
      expect(screen.getByTestId('law-item-gdpr')).toHaveAttribute('data-selected', 'true');
      expect(screen.getByTestId('law-item-nis2')).toHaveAttribute('data-selected', 'true');
    });

    it('clicking a law does not change selection', () => {
      const onTargetCorporaChange = vi.fn();
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpusScope="all"
          onTargetCorporaChange={onTargetCorporaChange}
        />
      );

      fireEvent.click(screen.getByTestId('law-item-gdpr'));

      expect(onTargetCorporaChange).not.toHaveBeenCalled();
    });

    it('search still filters the list', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="all" />);

      fireEvent.change(screen.getByTestId('law-search-input'), { target: { value: 'gdpr' } });

      expect(screen.getByTestId('law-item-gdpr')).toBeInTheDocument();
      expect(screen.queryByTestId('law-item-ai-act')).not.toBeInTheDocument();
    });
  });

  describe('search matches eurovoc labels', () => {
    const corporaWithLabels: CorpusInfo[] = [
      { id: 'ai-act', name: 'AI-forordningen', eurovoc_labels: ['kunstig intelligens', 'teknologi'] },
      { id: 'gdpr', name: 'GDPR', eurovoc_labels: ['databeskyttelse', 'personoplysninger'] },
      { id: 'nis2', name: 'NIS2', eurovoc_labels: ['cybersikkerhed'] },
    ];

    it('search by eurovoc label finds matching law', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpora={corporaWithLabels}
          corpusScope="single"
        />
      );

      // First search input is the main law search (second is EuroVoc label search)
      const searchInputs = screen.getAllByTestId('law-search-input');
      fireEvent.change(searchInputs[0], {
        target: { value: 'databeskyttelse' },
      });

      // GDPR has 'databeskyttelse' as eurovoc label — should be found
      expect(screen.getByTestId('law-item-gdpr')).toBeInTheDocument();
      // Others should be filtered out
      expect(screen.queryByTestId('law-item-ai-act')).not.toBeInTheDocument();
      expect(screen.queryByTestId('law-item-nis2')).not.toBeInTheDocument();
    });

    it('search by partial eurovoc label finds matching law', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpora={corporaWithLabels}
          corpusScope="single"
        />
      );

      const searchInputs = screen.getAllByTestId('law-search-input');
      fireEvent.change(searchInputs[0], {
        target: { value: 'cyber' },
      });

      expect(screen.getByTestId('law-item-nis2')).toBeInTheDocument();
      expect(screen.queryByTestId('law-item-ai-act')).not.toBeInTheDocument();
      expect(screen.queryByTestId('law-item-gdpr')).not.toBeInTheDocument();
    });
  });

  describe('general behavior', () => {
    it('laws are sorted alphabetically by ID', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="single" />);

      const buttons = screen.getAllByRole('button');
      const ids = buttons.map(b => b.textContent);

      // Should be: ai-act, gdpr, nis2 (alphabetical)
      expect(ids).toEqual(['ai-act', 'gdpr', 'nis2']);
    });

    it('is disabled when disabled=true', () => {
      render(<LawSelectorPanel {...defaultProps} corpusScope="single" disabled={true} />);

      expect(screen.getByTestId('law-search-input')).toBeDisabled();
      expect(screen.getByTestId('law-item-ai-act')).toBeDisabled();
    });
  });

  describe('hidden selection summary', () => {
    const corporaWithLabels: CorpusInfo[] = [
      { id: 'ai-act', name: 'AI-forordningen', eurovoc_labels: ['kunstig intelligens'] },
      { id: 'gdpr', name: 'GDPR', eurovoc_labels: ['databeskyttelse'] },
      { id: 'nis2', name: 'NIS2', eurovoc_labels: ['cybersikkerhed'] },
    ];

    it('shows HiddenSelectionSummary when selection hidden by EuroVoc filter', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpora={corporaWithLabels}
          corpusScope="explicit"
          targetCorpora={['gdpr']}
        />
      );

      // Click on "kunstig intelligens" checkbox to filter
      const aiCheckbox = screen.getByRole('checkbox', { name: /kunstig intelligens/i });
      fireEvent.click(aiCheckbox);

      // GDPR should be hidden (doesn't have 'kunstig intelligens')
      expect(screen.getByTestId('hidden-selection-summary')).toBeInTheDocument();
      expect(screen.getByTestId('hidden-law-gdpr')).toBeInTheDocument();
    });

    it('does not show HiddenSelectionSummary when selection visible', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpora={corporaWithLabels}
          corpusScope="explicit"
          targetCorpora={['ai-act']}
        />
      );

      // No filter applied, so AI-ACT is visible
      expect(screen.queryByTestId('hidden-selection-summary')).not.toBeInTheDocument();
    });

    it('does not show HiddenSelectionSummary in all mode', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpora={corporaWithLabels}
          corpusScope="all"
          targetCorpora={['gdpr']}
        />
      );

      // Click on filter
      const aiCheckbox = screen.getByRole('checkbox', { name: /kunstig intelligens/i });
      fireEvent.click(aiCheckbox);

      // Even with EuroVoc filter, all mode should not show hidden summary
      expect(screen.queryByTestId('hidden-selection-summary')).not.toBeInTheDocument();
    });

    it('removes law from targetCorpora when onRemove called', () => {
      const onTargetCorporaChange = vi.fn();

      render(
        <LawSelectorPanel
          {...defaultProps}
          corpora={corporaWithLabels}
          corpusScope="explicit"
          targetCorpora={['gdpr', 'nis2']}
          onTargetCorporaChange={onTargetCorporaChange}
        />
      );

      // Apply filter to hide GDPR and NIS2
      const aiCheckbox = screen.getByRole('checkbox', { name: /kunstig intelligens/i });
      fireEvent.click(aiCheckbox);

      // Remove GDPR from hidden summary
      fireEvent.click(screen.getByTestId('hidden-law-gdpr'));

      expect(onTargetCorporaChange).toHaveBeenCalledWith(['nis2']);
    });

    it('clears EuroVoc filter when onClearFilter called', () => {
      render(
        <LawSelectorPanel
          {...defaultProps}
          corpora={corporaWithLabels}
          corpusScope="explicit"
          targetCorpora={['gdpr']}
        />
      );

      // Apply filter
      const aiCheckbox = screen.getByRole('checkbox', { name: /kunstig intelligens/i });
      fireEvent.click(aiCheckbox);

      // Hidden summary should appear
      expect(screen.getByTestId('hidden-selection-summary')).toBeInTheDocument();

      // Clear filter
      fireEvent.click(screen.getByTestId('clear-filter-btn'));

      // Hidden summary should disappear (GDPR now visible)
      expect(screen.queryByTestId('hidden-selection-summary')).not.toBeInTheDocument();
    });
  });
});
