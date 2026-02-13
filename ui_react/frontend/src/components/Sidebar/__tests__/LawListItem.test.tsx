/**
 * Tests for LawListItem component.
 *
 * TDD: Tests for unified checkbox interface.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { LawListItem } from '../LawListItem';
import type { CorpusInfo } from '../../../types';

// Mock Tooltip
vi.mock('../../Common/Tooltip', () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe('LawListItem', () => {
  const mockCorpus: CorpusInfo = {
    id: 'ai-act',
    name: 'AI-forordningen (AI Act)',
    fullname: 'EUROPA-PARLAMENTETS OG RÅDETS FORORDNING (EU) 2024/1689',
  };

  const defaultProps = {
    corpus: mockCorpus,
    isSelected: false,
    checkboxMode: 'checkbox' as const,
    onClick: vi.fn(),
  };

  describe('rendering', () => {
    it('renders shortname extracted from corpus name', () => {
      render(<LawListItem {...defaultProps} />);

      // Shortname is extracted as uppercase ID
      expect(screen.getByText('AI-ACT')).toBeInTheDocument();
    });

    it('renders display name', () => {
      render(<LawListItem {...defaultProps} />);

      expect(screen.getByText('AI-forordningen (AI Act)')).toBeInTheDocument();
    });

    it('uses uppercase ID as fallback when name is missing (M1)', () => {
      const corpusWithoutName: CorpusInfo = { id: 'gdpr' };
      render(<LawListItem {...defaultProps} corpus={corpusWithoutName} />);

      // Only shortname shows when displayName === shortname (compact layout)
      const gdprElements = screen.getAllByText('GDPR');
      expect(gdprElements.length).toBe(1); // Shortname only, no redundant display name
    });
  });

  describe('checkboxMode behavior', () => {
    describe('checkbox mode', () => {
      it('renders checkbox input', () => {
        render(<LawListItem {...defaultProps} checkboxMode="checkbox" />);

        expect(screen.getByRole('checkbox')).toBeInTheDocument();
      });

      it('checkbox has rounded class (square corners)', () => {
        render(<LawListItem {...defaultProps} checkboxMode="checkbox" />);

        expect(screen.getByRole('checkbox')).toHaveClass('rounded');
        expect(screen.getByRole('checkbox')).not.toHaveClass('rounded-full');
      });

      it('checkbox is checked when isSelected=true', () => {
        render(<LawListItem {...defaultProps} checkboxMode="checkbox" isSelected={true} />);

        expect(screen.getByRole('checkbox')).toBeChecked();
      });

      it('checkbox is unchecked when isSelected=false', () => {
        render(<LawListItem {...defaultProps} checkboxMode="checkbox" isSelected={false} />);

        expect(screen.getByRole('checkbox')).not.toBeChecked();
      });

      it('calls onClick when clicked', () => {
        const onClick = vi.fn();
        render(<LawListItem {...defaultProps} checkboxMode="checkbox" onClick={onClick} />);

        fireEvent.click(screen.getByRole('button'));

        expect(onClick).toHaveBeenCalledTimes(1);
      });
    });

    describe('radio mode', () => {
      it('renders checkbox input with radio styling', () => {
        render(<LawListItem {...defaultProps} checkboxMode="radio" />);

        expect(screen.getByRole('checkbox')).toBeInTheDocument();
      });

      it('checkbox has rounded-full class (circular)', () => {
        render(<LawListItem {...defaultProps} checkboxMode="radio" />);

        expect(screen.getByRole('checkbox')).toHaveClass('rounded-full');
      });

      it('calls onClick when clicked', () => {
        const onClick = vi.fn();
        render(<LawListItem {...defaultProps} checkboxMode="radio" onClick={onClick} />);

        fireEvent.click(screen.getByRole('button'));

        expect(onClick).toHaveBeenCalledTimes(1);
      });
    });

    describe('disabled mode', () => {
      it('renders checkbox input', () => {
        render(<LawListItem {...defaultProps} checkboxMode="disabled" />);

        expect(screen.getByRole('checkbox')).toBeInTheDocument();
      });

      it('checkbox is always checked', () => {
        render(<LawListItem {...defaultProps} checkboxMode="disabled" isSelected={false} />);

        // In disabled mode, checkbox is always checked regardless of isSelected
        expect(screen.getByRole('checkbox')).toBeChecked();
      });

      it('checkbox is disabled', () => {
        render(<LawListItem {...defaultProps} checkboxMode="disabled" />);

        expect(screen.getByRole('checkbox')).toBeDisabled();
      });

      it('row has reduced opacity', () => {
        render(<LawListItem {...defaultProps} checkboxMode="disabled" />);

        expect(screen.getByRole('button')).toHaveClass('opacity-50');
      });

      it('does not call onClick when clicked', () => {
        const onClick = vi.fn();
        render(<LawListItem {...defaultProps} checkboxMode="disabled" onClick={onClick} />);

        fireEvent.click(screen.getByRole('button'));

        expect(onClick).not.toHaveBeenCalled();
      });

      it('row has cursor-not-allowed', () => {
        render(<LawListItem {...defaultProps} checkboxMode="disabled" />);

        expect(screen.getByRole('button')).toHaveClass('cursor-not-allowed');
      });
    });
  });

  describe('selection state', () => {
    it('has selected styling when isSelected=true', () => {
      render(<LawListItem {...defaultProps} isSelected={true} />);

      const item = screen.getByRole('button');
      expect(item).toHaveClass('bg-apple-blue/5');
    });

    it('does not have selected styling when isSelected=false', () => {
      render(<LawListItem {...defaultProps} isSelected={false} />);

      const item = screen.getByRole('button');
      expect(item).not.toHaveClass('bg-apple-blue/5');
    });
  });

  describe('click behavior', () => {
    it('calls onClick when row is clicked', () => {
      const onClick = vi.fn();
      render(<LawListItem {...defaultProps} onClick={onClick} />);

      fireEvent.click(screen.getByRole('button'));

      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it('calls onClick when checkbox is clicked', () => {
      const onClick = vi.fn();
      render(<LawListItem {...defaultProps} checkboxMode="checkbox" onClick={onClick} />);

      fireEvent.click(screen.getByRole('checkbox'));

      expect(onClick).toHaveBeenCalledTimes(1);
    });
  });

  describe('disabled state (prop)', () => {
    it('is disabled when disabled=true', () => {
      render(<LawListItem {...defaultProps} disabled={true} />);

      expect(screen.getByRole('button')).toBeDisabled();
    });

    it('does not call onClick when disabled', () => {
      const onClick = vi.fn();
      render(<LawListItem {...defaultProps} onClick={onClick} disabled={true} />);

      fireEvent.click(screen.getByRole('button'));

      expect(onClick).not.toHaveBeenCalled();
    });

    it('checkbox is disabled when disabled=true', () => {
      render(<LawListItem {...defaultProps} checkboxMode="checkbox" disabled={true} />);

      expect(screen.getByRole('checkbox')).toBeDisabled();
    });
  });

  describe('keyboard support', () => {
    it('supports Enter key to select', () => {
      const onClick = vi.fn();
      render(<LawListItem {...defaultProps} onClick={onClick} />);

      fireEvent.keyDown(screen.getByRole('button'), { key: 'Enter' });

      expect(onClick).toHaveBeenCalledTimes(1);
    });

    it('supports Space key to toggle', () => {
      const onClick = vi.fn();
      render(<LawListItem {...defaultProps} checkboxMode="checkbox" onClick={onClick} />);

      fireEvent.keyDown(screen.getByRole('button'), { key: ' ' });

      expect(onClick).toHaveBeenCalledTimes(1);
    });
  });

  describe('EuroVoc label tag', () => {
    const corpusWithLabels: CorpusInfo = {
      id: 'gdpr',
      name: 'GDPR',
      eurovoc_labels: ['databeskyttelse', 'personoplysninger', 'informationssikkerhed'],
    };

    it('shows total labels with EuroVoc suffix', () => {
      render(
        <LawListItem {...defaultProps} corpus={corpusWithLabels} />
      );

      // Should show "3 EuroVoc" (not "3 emneord")
      expect(screen.getByText(/3 EuroVoc/)).toBeInTheDocument();
    });

    it('shows matched/total format when labels are selected', () => {
      render(
        <LawListItem
          {...defaultProps}
          corpus={corpusWithLabels}
          selectedLabels={['databeskyttelse', 'informationssikkerhed']}
        />
      );

      // Should show "2/3 EuroVoc" (2 matched out of 3 total)
      expect(screen.getByText(/2\/3 EuroVoc/)).toBeInTheDocument();
    });

    it('does not show tag when corpus has no labels', () => {
      render(<LawListItem {...defaultProps} />);

      // No tag should be shown
      expect(screen.queryByText(/EuroVoc/)).not.toBeInTheDocument();
    });

    it('has blue styling when labels match', () => {
      render(
        <LawListItem
          {...defaultProps}
          corpus={corpusWithLabels}
          selectedLabels={['databeskyttelse']}
        />
      );

      const tag = screen.getByText(/1\/3 EuroVoc/);
      expect(tag).toHaveClass('text-apple-blue');
    });

    it('has gray styling when no labels match', () => {
      render(
        <LawListItem
          {...defaultProps}
          corpus={corpusWithLabels}
          selectedLabels={[]}
        />
      );

      const tag = screen.getByText(/3 EuroVoc/);
      expect(tag).toHaveClass('text-apple-gray-500');
    });
  });

  describe('smart tooltip', () => {
    const corpusWithLabels: CorpusInfo = {
      id: 'gdpr',
      name: 'GDPR',
      eurovoc_labels: ['databeskyttelse', 'personoplysninger', 'informationssikkerhed'],
    };

    it('shows all labels in tooltip when no labels selected', () => {
      render(
        <LawListItem {...defaultProps} corpus={corpusWithLabels} />
      );

      const tag = screen.getByText(/3 EuroVoc/);
      // Tooltip should show all labels joined
      expect(tag).toHaveAttribute('title', 'databeskyttelse, personoplysninger, informationssikkerhed');
    });

    it('shows "Matcher:" and "Øvrige:" sections when labels match', () => {
      render(
        <LawListItem
          {...defaultProps}
          corpus={corpusWithLabels}
          selectedLabels={['databeskyttelse', 'informationssikkerhed']}
        />
      );

      const tag = screen.getByText(/2\/3 EuroVoc/);
      const tooltip = tag.getAttribute('title');

      // Should have "Matcher:" section with matched labels first
      expect(tooltip).toContain('Matcher:');
      expect(tooltip).toContain('databeskyttelse');
      expect(tooltip).toContain('informationssikkerhed');

      // Should have "Øvrige:" section with remaining labels
      expect(tooltip).toContain('Øvrige:');
      expect(tooltip).toContain('personoplysninger');
    });

    it('shows "Ingen emneord" when corpus has no labels', () => {
      const corpusWithoutLabels: CorpusInfo = { id: 'test', name: 'Test' };
      render(
        <LawListItem {...defaultProps} corpus={corpusWithoutLabels} />
      );

      // No tag shown when no labels
      expect(screen.queryByText(/EuroVoc/)).not.toBeInTheDocument();
    });
  });

  describe('compact layout (T1-T5)', () => {
    it('T1: renders separator between shortname and display name', () => {
      render(<LawListItem {...defaultProps} />);

      // Middle dot separator should be present when displayName differs from shortname
      expect(screen.getByText('·')).toBeInTheDocument();
    });

    it('T2: does not render separator when displayName equals shortname', () => {
      const corpusWithoutName: CorpusInfo = { id: 'gdpr' };
      render(<LawListItem {...defaultProps} corpus={corpusWithoutName} />);

      // No separator when names are equal (fallback case)
      expect(screen.queryByText('·')).not.toBeInTheDocument();
    });

    it('T3: has accessible aria-label with full text', () => {
      render(<LawListItem {...defaultProps} />);

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-label', 'AI-ACT — AI-forordningen (AI Act)');
    });

    it('T4: has aria-label with only shortname when names equal', () => {
      const corpusWithoutName: CorpusInfo = { id: 'gdpr' };
      render(<LawListItem {...defaultProps} corpus={corpusWithoutName} />);

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-label', 'GDPR');
    });

    it('T5: has compact padding class', () => {
      render(<LawListItem {...defaultProps} />);

      const button = screen.getByRole('button');
      expect(button).toHaveClass('py-1.5');
    });
  });
});
