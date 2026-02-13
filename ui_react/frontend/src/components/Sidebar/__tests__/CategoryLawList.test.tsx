/**
 * Tests for CategoryLawList component.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CategoryLawList } from '../CategoryLawList';
import type { CorpusInfo } from '../../../types';
import type { CheckboxMode } from '../LawListItem';

// Mock CollapsibleSection
vi.mock('../CollapsibleSection', () => ({
  CollapsibleSection: ({
    title,
    count,
    defaultOpen,
    children,
  }: {
    title: string;
    count?: number;
    defaultOpen?: boolean;
    children: React.ReactNode;
  }) => (
    <div data-testid={`category-${title}`} data-count={count} data-default-open={defaultOpen}>
      <span>{title}</span>
      {count !== undefined && <span data-testid={`count-${title}`}>{count}</span>}
      <div data-testid={`content-${title}`}>{children}</div>
    </div>
  ),
}));

// Mock LawListItem
vi.mock('../LawListItem', () => ({
  LawListItem: ({
    corpus,
    isSelected,
    checkboxMode,
    onClick,
    disabled,
  }: {
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
      {corpus.name}
    </button>
  ),
}));

describe('CategoryLawList', () => {
  const mockCorpora: CorpusInfo[] = [
    { id: 'gdpr', name: 'GDPR', eurovoc_labels: ['databeskyttelse'] },
    { id: 'ai-act', name: 'AI Act', eurovoc_labels: ['kunstig-intelligens'] },
    { id: 'nis2', name: 'NIS2', eurovoc_labels: ['cybersikkerhed'] },
    { id: 'cyberrobusthed', name: 'Cyberrobusthed', eurovoc_labels: ['cybersikkerhed'] },
    { id: 'no-category', name: 'No Category Law' },
  ];

  const defaultProps = {
    corpora: mockCorpora,
    targetCorpora: [] as string[],
    checkboxMode: 'radio' as CheckboxMode,
    onLawClick: vi.fn(),
    disabled: false,
    expandedCategories: new Set<string>(),
    onCategoryToggle: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('category grouping', () => {
    it('renders collapsible sections for each category', () => {
      render(<CategoryLawList {...defaultProps} />);

      expect(screen.getByTestId('category-Cybersikkerhed')).toBeInTheDocument();
      expect(screen.getByTestId('category-Databeskyttelse')).toBeInTheDocument();
      expect(screen.getByTestId('category-Kunstig intelligens')).toBeInTheDocument();
    });

    it('renders uncategorized section for laws without category', () => {
      render(<CategoryLawList {...defaultProps} />);

      expect(screen.getByTestId('category-Øvrige')).toBeInTheDocument();
    });

    it('shows law count for each category', () => {
      render(<CategoryLawList {...defaultProps} />);

      // Cybersikkerhed has 2 laws (nis2, cyberrobusthed)
      expect(screen.getByTestId('count-Cybersikkerhed')).toHaveTextContent('2');
      // Databeskyttelse has 1 law
      expect(screen.getByTestId('count-Databeskyttelse')).toHaveTextContent('1');
    });

    it('groups laws correctly by category', () => {
      render(<CategoryLawList {...defaultProps} />);

      // Cybersecurity category should contain both nis2 and cyberrobusthed
      const cyberContent = screen.getByTestId('content-Cybersikkerhed');
      expect(cyberContent).toContainElement(screen.getByTestId('law-item-nis2'));
      expect(cyberContent).toContainElement(screen.getByTestId('law-item-cyberrobusthed'));
    });
  });

  describe('law selection', () => {
    it('calls onLawClick when a law is clicked', () => {
      const onLawClick = vi.fn();
      render(<CategoryLawList {...defaultProps} onLawClick={onLawClick} />);

      fireEvent.click(screen.getByTestId('law-item-gdpr'));

      expect(onLawClick).toHaveBeenCalledWith('gdpr');
    });

    it('marks selected laws correctly', () => {
      render(
        <CategoryLawList
          {...defaultProps}
          targetCorpora={['gdpr', 'nis2']}
        />
      );

      expect(screen.getByTestId('law-item-gdpr')).toHaveAttribute('data-selected', 'true');
      expect(screen.getByTestId('law-item-nis2')).toHaveAttribute('data-selected', 'true');
      expect(screen.getByTestId('law-item-ai-act')).toHaveAttribute('data-selected', 'false');
    });

    it('passes checkboxMode to LawListItem', () => {
      render(<CategoryLawList {...defaultProps} checkboxMode="checkbox" />);

      expect(screen.getByTestId('law-item-gdpr')).toHaveAttribute('data-checkbox-mode', 'checkbox');
    });
  });

  describe('disabled state', () => {
    it('disables all law items when disabled', () => {
      render(<CategoryLawList {...defaultProps} disabled={true} />);

      expect(screen.getByTestId('law-item-gdpr')).toBeDisabled();
      expect(screen.getByTestId('law-item-ai-act')).toBeDisabled();
    });
  });

  describe('empty state', () => {
    it('renders nothing when corpora is empty', () => {
      const { container } = render(<CategoryLawList {...defaultProps} corpora={[]} />);

      expect(container.firstChild).toBeEmptyDOMElement();
    });
  });

  describe('alphabetical sorting', () => {
    it('sorts laws within each category alphabetically', () => {
      render(<CategoryLawList {...defaultProps} />);

      const cyberContent = screen.getByTestId('content-Cybersikkerhed');
      const buttons = cyberContent.querySelectorAll('button');
      const names = Array.from(buttons).map((b) => b.textContent);

      // Cyberrobusthed comes before NIS2 alphabetically
      expect(names).toEqual(['Cyberrobusthed', 'NIS2']);
    });
  });

  describe('category expansion', () => {
    it('expands categories in expandedCategories set', () => {
      render(
        <CategoryLawList
          {...defaultProps}
          expandedCategories={new Set(['cybersikkerhed'])}
        />
      );

      // Only Cybersikkerhed should be open
      expect(screen.getByTestId('category-Cybersikkerhed')).toHaveAttribute('data-default-open', 'true');
      expect(screen.getByTestId('category-Databeskyttelse')).toHaveAttribute('data-default-open', 'false');
    });

    it('auto-expands category containing selected law', () => {
      render(
        <CategoryLawList
          {...defaultProps}
          targetCorpora={['gdpr']}
        />
      );

      // Databeskyttelse category should be auto-expanded because gdpr is selected
      expect(screen.getByTestId('category-Databeskyttelse')).toHaveAttribute('data-default-open', 'true');
    });
  });
});
