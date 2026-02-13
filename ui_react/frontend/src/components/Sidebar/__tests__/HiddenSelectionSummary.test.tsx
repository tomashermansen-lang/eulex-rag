/**
 * Tests for HiddenSelectionSummary component.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { HiddenSelectionSummary } from '../HiddenSelectionSummary';

// Mock LawToken to simplify testing
vi.mock('../LawToken', () => ({
  LawToken: ({ name, id, onRemove }: { name: string; id?: string; onRemove: (id: string) => void }) => (
    <span data-testid={`law-token-${id ?? name}`}>
      {name}
      <button onClick={() => onRemove(id ?? name)} aria-label={`Fjern ${name}`}>×</button>
    </span>
  ),
}));

describe('HiddenSelectionSummary', () => {
  const defaultProps = {
    hiddenLaws: [{ id: 'gdpr', name: 'GDPR' }],
    onRemove: vi.fn(),
    onClearFilter: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('returns null when hiddenLaws is empty', () => {
      const { container } = render(
        <HiddenSelectionSummary {...defaultProps} hiddenLaws={[]} />
      );

      expect(container.firstChild).toBeNull();
    });

    it('renders warning header when hiddenLaws > 0', () => {
      render(<HiddenSelectionSummary {...defaultProps} />);

      expect(screen.getByText(/Valgte.*skjult af filter/i)).toBeInTheDocument();
    });

    it('renders LawToken for each hidden law', () => {
      const hiddenLaws = [
        { id: 'gdpr', name: 'GDPR' },
        { id: 'nis2', name: 'NIS2' },
      ];
      render(<HiddenSelectionSummary {...defaultProps} hiddenLaws={hiddenLaws} />);

      expect(screen.getByTestId('law-token-gdpr')).toBeInTheDocument();
      expect(screen.getByTestId('law-token-nis2')).toBeInTheDocument();
    });

    it('shows clear filter button', () => {
      render(<HiddenSelectionSummary {...defaultProps} />);

      expect(screen.getByRole('button', { name: /ryd emnefilter/i })).toBeInTheDocument();
    });

    it('has amber warning styling', () => {
      const { container } = render(<HiddenSelectionSummary {...defaultProps} />);

      const section = container.firstChild as HTMLElement;
      expect(section).toHaveClass('bg-amber-50');
    });
  });

  describe('interactions', () => {
    it('calls onRemove when token × clicked', () => {
      const onRemove = vi.fn();
      render(<HiddenSelectionSummary {...defaultProps} onRemove={onRemove} />);

      fireEvent.click(screen.getByLabelText(/fjern gdpr/i));

      expect(onRemove).toHaveBeenCalledWith('gdpr');
    });

    it('calls onClearFilter when button clicked', () => {
      const onClearFilter = vi.fn();
      render(<HiddenSelectionSummary {...defaultProps} onClearFilter={onClearFilter} />);

      fireEvent.click(screen.getByRole('button', { name: /ryd emnefilter/i }));

      expect(onClearFilter).toHaveBeenCalledTimes(1);
    });
  });

  describe('overflow', () => {
    it('shows overflow indicator when > 5 laws', () => {
      const hiddenLaws = [
        { id: '1', name: 'Law 1' },
        { id: '2', name: 'Law 2' },
        { id: '3', name: 'Law 3' },
        { id: '4', name: 'Law 4' },
        { id: '5', name: 'Law 5' },
        { id: '6', name: 'Law 6' },
        { id: '7', name: 'Law 7' },
      ];
      render(<HiddenSelectionSummary {...defaultProps} hiddenLaws={hiddenLaws} />);

      // Should show 5 tokens + overflow indicator
      expect(screen.getByText('+2')).toBeInTheDocument();
      // 6th and 7th should not be rendered as tokens
      expect(screen.queryByTestId('law-token-6')).not.toBeInTheDocument();
      expect(screen.queryByTestId('law-token-7')).not.toBeInTheDocument();
    });

    it('shows all tokens when exactly 5 laws', () => {
      const hiddenLaws = [
        { id: '1', name: 'Law 1' },
        { id: '2', name: 'Law 2' },
        { id: '3', name: 'Law 3' },
        { id: '4', name: 'Law 4' },
        { id: '5', name: 'Law 5' },
      ];
      render(<HiddenSelectionSummary {...defaultProps} hiddenLaws={hiddenLaws} />);

      // All 5 should be visible, no overflow
      expect(screen.queryByText(/\+\d/)).not.toBeInTheDocument();
    });
  });

  describe('disabled state', () => {
    it('disables clear filter button when disabled=true', () => {
      render(<HiddenSelectionSummary {...defaultProps} disabled={true} />);

      expect(screen.getByRole('button', { name: /ryd emnefilter/i })).toBeDisabled();
    });

    it('does not call onClearFilter when disabled', () => {
      const onClearFilter = vi.fn();
      render(
        <HiddenSelectionSummary {...defaultProps} onClearFilter={onClearFilter} disabled={true} />
      );

      fireEvent.click(screen.getByRole('button', { name: /ryd emnefilter/i }));

      expect(onClearFilter).not.toHaveBeenCalled();
    });
  });

  describe('accessibility', () => {
    it('has role="status" for screen reader announcement', () => {
      render(<HiddenSelectionSummary {...defaultProps} />);

      expect(screen.getByRole('status')).toBeInTheDocument();
    });

    it('warning icon has aria-hidden', () => {
      const { container } = render(<HiddenSelectionSummary {...defaultProps} />);

      const svg = container.querySelector('svg');
      expect(svg).toHaveAttribute('aria-hidden', 'true');
    });
  });
});
