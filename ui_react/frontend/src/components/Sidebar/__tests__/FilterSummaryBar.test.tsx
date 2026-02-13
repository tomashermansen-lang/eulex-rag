/**
 * Tests for FilterSummaryBar component.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { FilterSummaryBar } from '../FilterSummaryBar';

describe('FilterSummaryBar', () => {
  const defaultProps = {
    selectedLabels: ['databeskyttelse', 'informationssikkerhed'],
    filteredCount: 5,
    totalCount: 12,
    onRemove: vi.fn(),
    onClearAll: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders filter count text', () => {
      render(<FilterSummaryBar {...defaultProps} />);

      expect(screen.getByText(/Viser 5 af 12 love/)).toBeInTheDocument();
    });

    it('renders selected labels as removable chips', () => {
      render(<FilterSummaryBar {...defaultProps} />);

      expect(screen.getByText('databeskyttelse')).toBeInTheDocument();
      expect(screen.getByText('informationssikkerhed')).toBeInTheDocument();
    });

    it('renders nothing when no labels selected', () => {
      const { container } = render(
        <FilterSummaryBar {...defaultProps} selectedLabels={[]} />
      );

      expect(container.firstChild).toBeNull();
    });

    it('renders clear all button', () => {
      render(<FilterSummaryBar {...defaultProps} />);

      expect(screen.getByText('Ryd filter')).toBeInTheDocument();
    });
  });

  describe('interactions', () => {
    it('calls onRemove when chip X is clicked', () => {
      const onRemove = vi.fn();
      render(<FilterSummaryBar {...defaultProps} onRemove={onRemove} />);

      // Find the remove button for databeskyttelse
      const removeButtons = screen.getAllByLabelText(/Fjern/);
      fireEvent.click(removeButtons[0]);

      expect(onRemove).toHaveBeenCalledWith('databeskyttelse');
    });

    it('calls onClearAll when clear button is clicked', () => {
      const onClearAll = vi.fn();
      render(<FilterSummaryBar {...defaultProps} onClearAll={onClearAll} />);

      fireEvent.click(screen.getByText('Ryd filter'));

      expect(onClearAll).toHaveBeenCalled();
    });
  });

  describe('disabled state', () => {
    it('disables all buttons when disabled prop is true', () => {
      render(<FilterSummaryBar {...defaultProps} disabled={true} />);

      const clearButton = screen.getByText('Ryd filter');
      expect(clearButton).toBeDisabled();

      const removeButtons = screen.getAllByLabelText(/Fjern/);
      removeButtons.forEach((button) => {
        expect(button).toBeDisabled();
      });
    });
  });

  describe('singular/plural text', () => {
    it('uses singular form for 1 law', () => {
      render(<FilterSummaryBar {...defaultProps} filteredCount={1} totalCount={1} />);

      expect(screen.getByText(/Viser 1 af 1 lov/)).toBeInTheDocument();
    });

    it('uses plural form for multiple laws', () => {
      render(<FilterSummaryBar {...defaultProps} filteredCount={5} totalCount={12} />);

      expect(screen.getByText(/Viser 5 af 12 love/)).toBeInTheDocument();
    });
  });
});
