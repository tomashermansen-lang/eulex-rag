/**
 * Tests for SuggestedLabels component.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SuggestedLabels } from '../SuggestedLabels';
import type { LabelWithCount } from '../labelUtils';

describe('SuggestedLabels', () => {
  const mockLabels: LabelWithCount[] = [
    { label: 'kunstig intelligens', count: 5 },
    { label: 'databeskyttelse', count: 3 },
    { label: 'innovation', count: 2 },
    { label: 'personoplysninger', count: 2 },
  ];

  const defaultProps = {
    question: 'Hvad er reglerne for kunstig intelligens?',
    allLabels: mockLabels,
    selectedLabels: [] as string[],
    onSelect: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders nothing when question is empty', () => {
      const { container } = render(
        <SuggestedLabels {...defaultProps} question="" />
      );

      expect(container.firstChild).toBeNull();
    });

    it('renders nothing when no suggestions match', () => {
      const { container } = render(
        <SuggestedLabels {...defaultProps} question="noget helt andet" />
      );

      expect(container.firstChild).toBeNull();
    });

    it('renders suggested labels with sparkle indicator', () => {
      render(<SuggestedLabels {...defaultProps} />);

      // Should match "kunstig intelligens" from question
      expect(screen.getByText(/kunstig intelligens/)).toBeInTheDocument();
      // Should have sparkle emoji
      expect(screen.getByText(/✨/)).toBeInTheDocument();
    });

    it('renders section header', () => {
      render(<SuggestedLabels {...defaultProps} />);

      expect(screen.getByText('Foreslået:')).toBeInTheDocument();
    });

    it('shows count in suggestion chips', () => {
      render(<SuggestedLabels {...defaultProps} />);

      // Should show count in parentheses
      expect(screen.getByText(/\(5\)/)).toBeInTheDocument();
    });

    it('does not render already selected labels as suggestions', () => {
      render(
        <SuggestedLabels
          {...defaultProps}
          selectedLabels={['kunstig intelligens']}
        />
      );

      // kunstig intelligens is selected, so should not appear in suggestions
      // If no other suggestions, section should not render
      const kunstigElements = screen.queryAllByText(/kunstig intelligens/);
      expect(kunstigElements).toHaveLength(0);
    });
  });

  describe('interactions', () => {
    it('calls onSelect when suggestion chip is clicked', () => {
      const onSelect = vi.fn();
      render(<SuggestedLabels {...defaultProps} onSelect={onSelect} />);

      fireEvent.click(screen.getByText(/kunstig intelligens/));

      expect(onSelect).toHaveBeenCalledWith('kunstig intelligens');
    });
  });

  describe('disabled state', () => {
    it('disables suggestion chips when disabled prop is true', () => {
      render(<SuggestedLabels {...defaultProps} disabled={true} />);

      const chip = screen.getByText(/kunstig intelligens/).closest('button');
      expect(chip).toBeDisabled();
    });
  });
});
