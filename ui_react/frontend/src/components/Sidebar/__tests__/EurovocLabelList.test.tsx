/**
 * Tests for EurovocLabelList component.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { EurovocLabelList } from '../EurovocLabelList';
import type { LabelWithCount } from '../labelUtils';

describe('EurovocLabelList', () => {
  const mockLabels: LabelWithCount[] = [
    { label: 'informationssikkerhed', count: 6 },
    { label: 'databeskyttelse', count: 3 },
    { label: 'kunstig intelligens', count: 2 },
    { label: 'innovation', count: 2 },
    { label: 'cybersikkerhed', count: 1 },
  ];

  const defaultProps = {
    labels: mockLabels,
    selectedLabels: [] as string[],
    suggestedLabels: [] as string[],
    onToggle: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders all labels with counts', () => {
      render(<EurovocLabelList {...defaultProps} />);

      expect(screen.getByText(/informationssikkerhed/)).toBeInTheDocument();
      expect(screen.getByText(/databeskyttelse/)).toBeInTheDocument();
      expect(screen.getByText(/\(6\)/)).toBeInTheDocument();
      expect(screen.getByText(/\(3\)/)).toBeInTheDocument();
    });

    it('renders search input', () => {
      render(<EurovocLabelList {...defaultProps} />);

      expect(screen.getByPlaceholderText(/Søg i emner/)).toBeInTheDocument();
    });

    it('renders section header with plain language', () => {
      render(<EurovocLabelList {...defaultProps} />);

      // "Emner" is plain language (like Westlaw's "Topics")
      expect(screen.getByText('Emner')).toBeInTheDocument();
    });

    it('header has EuroVoc tooltip', () => {
      render(<EurovocLabelList {...defaultProps} />);

      const header = screen.getByText('Emner');
      expect(header).toHaveAttribute('title', 'EuroVoc-klassifikation');
    });

    it('renders nothing when no labels provided', () => {
      const { container } = render(
        <EurovocLabelList {...defaultProps} labels={[]} />
      );

      expect(container.firstChild).toBeNull();
    });

    it('highlights suggested labels with star', () => {
      render(
        <EurovocLabelList
          {...defaultProps}
          suggestedLabels={['kunstig intelligens']}
        />
      );

      // Should have star indicator for suggested label
      const kunstigRow = screen.getByText(/kunstig intelligens/).closest('label');
      expect(kunstigRow?.textContent).toContain('★');
    });

    it('shows selected labels as checked', () => {
      render(
        <EurovocLabelList
          {...defaultProps}
          selectedLabels={['databeskyttelse', 'innovation']}
        />
      );

      const checkboxes = screen.getAllByRole('checkbox');
      const dataCheckbox = checkboxes.find((cb) =>
        cb.closest('label')?.textContent?.includes('databeskyttelse')
      );
      const innovationCheckbox = checkboxes.find((cb) =>
        cb.closest('label')?.textContent?.includes('innovation')
      );

      expect(dataCheckbox).toBeChecked();
      expect(innovationCheckbox).toBeChecked();
    });
  });

  describe('search filtering', () => {
    it('filters labels by search query', () => {
      render(<EurovocLabelList {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText(/Søg i emner/);
      fireEvent.change(searchInput, { target: { value: 'data' } });

      // Should show databeskyttelse (contains "data")
      expect(screen.getByText(/databeskyttelse/)).toBeInTheDocument();

      // Should not show labels that don't match
      expect(screen.queryByText(/informationssikkerhed/)).not.toBeInTheDocument();
    });

    it('shows match count when filtering', () => {
      render(<EurovocLabelList {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText(/Søg i emner/);
      fireEvent.change(searchInput, { target: { value: 'sikkerhed' } });

      // informationssikkerhed and cybersikkerhed match
      expect(screen.getByText(/2 emner matcher/)).toBeInTheDocument();
    });

    it('is case-insensitive', () => {
      render(<EurovocLabelList {...defaultProps} />);

      const searchInput = screen.getByPlaceholderText(/Søg i emner/);
      fireEvent.change(searchInput, { target: { value: 'INNOVATION' } });

      expect(screen.getByText(/innovation/)).toBeInTheDocument();
    });
  });

  describe('interactions', () => {
    it('calls onToggle when checkbox is clicked', () => {
      const onToggle = vi.fn();
      render(<EurovocLabelList {...defaultProps} onToggle={onToggle} />);

      const checkbox = screen.getAllByRole('checkbox')[0];
      fireEvent.click(checkbox);

      expect(onToggle).toHaveBeenCalledWith('informationssikkerhed');
    });

    it('calls onToggle when label row is clicked', () => {
      const onToggle = vi.fn();
      render(<EurovocLabelList {...defaultProps} onToggle={onToggle} />);

      fireEvent.click(screen.getByText(/databeskyttelse/));

      expect(onToggle).toHaveBeenCalledWith('databeskyttelse');
    });
  });

  describe('disabled state', () => {
    it('disables all checkboxes when disabled prop is true', () => {
      render(<EurovocLabelList {...defaultProps} disabled={true} />);

      const checkboxes = screen.getAllByRole('checkbox');
      checkboxes.forEach((checkbox) => {
        expect(checkbox).toBeDisabled();
      });
    });

    it('disables search input when disabled', () => {
      render(<EurovocLabelList {...defaultProps} disabled={true} />);

      expect(screen.getByPlaceholderText(/Søg i emner/)).toBeDisabled();
    });
  });
});
