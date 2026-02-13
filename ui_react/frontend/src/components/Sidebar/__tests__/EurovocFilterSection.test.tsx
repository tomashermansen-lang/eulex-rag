/**
 * Tests for EurovocFilterSection component.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { EurovocFilterSection } from '../EurovocFilterSection';
import type { CorpusInfo } from '../../../types';

describe('EurovocFilterSection', () => {
  const mockCorpora: CorpusInfo[] = [
    { id: 'gdpr', name: 'GDPR', eurovoc_labels: ['databeskyttelse', 'personoplysninger'] },
    { id: 'ai-act', name: 'AI Act', eurovoc_labels: ['kunstig intelligens', 'innovation'] },
    { id: 'nis2', name: 'NIS2', eurovoc_labels: ['cybersikkerhed', 'informationssikkerhed'] },
    { id: 'dora', name: 'DORA', eurovoc_labels: ['finansiel teknologi', 'informationssikkerhed'] },
  ];

  const defaultProps = {
    question: '',
    corpora: mockCorpora,
    selectedLabels: [] as string[],
    onSelectionChange: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('integration - three layers', () => {
    it('renders label list (always visible)', () => {
      render(<EurovocFilterSection {...defaultProps} />);

      // Label list should always be visible with plain language header
      expect(screen.getByText('Emner')).toBeInTheDocument();
      expect(screen.getByText(/informationssikkerhed/)).toBeInTheDocument();
    });

    it('shows suggestions when question has matching keywords', () => {
      render(
        <EurovocFilterSection {...defaultProps} question="kunstig intelligens" />
      );

      // Should show suggestions section
      expect(screen.getByText('Foreslået:')).toBeInTheDocument();
      // Should have sparkle chip
      expect(screen.getByText(/✨/)).toBeInTheDocument();
    });

    it('hides suggestions when question has no matching keywords', () => {
      render(
        <EurovocFilterSection {...defaultProps} question="noget helt andet" />
      );

      // Should not show suggestions section
      expect(screen.queryByText('Foreslået:')).not.toBeInTheDocument();
    });

    it('shows filter summary when labels are selected', () => {
      render(
        <EurovocFilterSection
          {...defaultProps}
          selectedLabels={['databeskyttelse']}
        />
      );

      // Should show summary bar
      expect(screen.getByText(/Viser/)).toBeInTheDocument();
      expect(screen.getByText('Ryd filter')).toBeInTheDocument();
    });

    it('hides filter summary when no labels selected', () => {
      render(<EurovocFilterSection {...defaultProps} />);

      // Should not show summary bar
      expect(screen.queryByText('Ryd filter')).not.toBeInTheDocument();
    });
  });

  describe('selection flow', () => {
    it('calls onSelectionChange when label is selected from list', () => {
      const onSelectionChange = vi.fn();
      render(
        <EurovocFilterSection
          {...defaultProps}
          onSelectionChange={onSelectionChange}
        />
      );

      // Click a checkbox in the list
      const checkbox = screen.getAllByRole('checkbox')[0];
      fireEvent.click(checkbox);

      expect(onSelectionChange).toHaveBeenCalled();
    });

    it('calls onSelectionChange when suggestion chip is clicked', () => {
      const onSelectionChange = vi.fn();
      render(
        <EurovocFilterSection
          {...defaultProps}
          question="kunstig intelligens"
          onSelectionChange={onSelectionChange}
        />
      );

      // Click the suggestion chip
      fireEvent.click(screen.getByText(/✨/).closest('button')!);

      expect(onSelectionChange).toHaveBeenCalled();
    });

    it('calls onSelectionChange when filter is removed from summary', () => {
      const onSelectionChange = vi.fn();
      render(
        <EurovocFilterSection
          {...defaultProps}
          selectedLabels={['databeskyttelse']}
          onSelectionChange={onSelectionChange}
        />
      );

      // Click remove button on chip
      fireEvent.click(screen.getByLabelText(/Fjern/));

      expect(onSelectionChange).toHaveBeenCalledWith([]);
    });

    it('calls onSelectionChange with empty array when clear all is clicked', () => {
      const onSelectionChange = vi.fn();
      render(
        <EurovocFilterSection
          {...defaultProps}
          selectedLabels={['databeskyttelse', 'innovation']}
          onSelectionChange={onSelectionChange}
        />
      );

      fireEvent.click(screen.getByText('Ryd filter'));

      expect(onSelectionChange).toHaveBeenCalledWith([]);
    });
  });

  describe('filtered count', () => {
    it('shows correct filtered count in summary', () => {
      render(
        <EurovocFilterSection
          {...defaultProps}
          selectedLabels={['informationssikkerhed']}
        />
      );

      // informationssikkerhed appears in NIS2 and DORA = 2 laws
      expect(screen.getByText(/Viser 2 af 4 love/)).toBeInTheDocument();
    });
  });

  describe('empty state', () => {
    it('renders nothing when no corpora provided', () => {
      const { container } = render(
        <EurovocFilterSection {...defaultProps} corpora={[]} />
      );

      expect(container.firstChild).toBeNull();
    });

    it('renders nothing when corpora have no eurovoc_labels', () => {
      const corporaWithoutLabels: CorpusInfo[] = [
        { id: 'test', name: 'Test' },
      ];
      const { container } = render(
        <EurovocFilterSection {...defaultProps} corpora={corporaWithoutLabels} />
      );

      expect(container.firstChild).toBeNull();
    });
  });

  describe('disabled state', () => {
    it('disables all interactive elements when disabled', () => {
      render(
        <EurovocFilterSection
          {...defaultProps}
          question="kunstig intelligens"
          selectedLabels={['databeskyttelse']}
          disabled={true}
        />
      );

      // Search input should be disabled
      expect(screen.getByPlaceholderText(/Søg i emner/)).toBeDisabled();

      // Checkboxes should be disabled
      const checkboxes = screen.getAllByRole('checkbox');
      checkboxes.forEach((cb) => {
        expect(cb).toBeDisabled();
      });
    });
  });
});
