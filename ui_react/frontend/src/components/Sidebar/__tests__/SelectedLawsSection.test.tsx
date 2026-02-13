/**
 * Tests for SelectedLawsSection component.
 *
 * TDD: Test selected laws display with tokens and overflow.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SelectedLawsSection } from '../SelectedLawsSection';

const mockLaws = [
  { id: 'gdpr', name: 'GDPR' },
  { id: 'ai-act', name: 'AI Act' },
  { id: 'dsa', name: 'DSA' },
  { id: 'dma', name: 'DMA' },
  { id: 'nis2', name: 'NIS2' },
];

describe('SelectedLawsSection', () => {
  describe('empty state', () => {
    it('renders empty state when no laws selected', () => {
      render(
        <SelectedLawsSection
          selectedLaws={[]}
          onRemove={vi.fn()}
          onClearAll={vi.fn()}
        />
      );

      expect(screen.getByText(/ingen love valgt/i)).toBeInTheDocument();
    });
  });

  describe('rendering', () => {
    it('renders tokens for selected laws', () => {
      render(
        <SelectedLawsSection
          selectedLaws={mockLaws.slice(0, 2)}
          onRemove={vi.fn()}
          onClearAll={vi.fn()}
        />
      );

      expect(screen.getByText('GDPR')).toBeInTheDocument();
      expect(screen.getByText('AI Act')).toBeInTheDocument();
    });

    it('renders header with count', () => {
      render(
        <SelectedLawsSection
          selectedLaws={mockLaws.slice(0, 3)}
          onRemove={vi.fn()}
          onClearAll={vi.fn()}
        />
      );

      expect(screen.getByText(/valgte love/i)).toBeInTheDocument();
      expect(screen.getByText('3')).toBeInTheDocument();
    });
  });

  describe('overflow', () => {
    it('shows overflow indicator when more than maxVisible', () => {
      render(
        <SelectedLawsSection
          selectedLaws={mockLaws}
          onRemove={vi.fn()}
          onClearAll={vi.fn()}
          maxVisible={3}
        />
      );

      // Should show +2 indicator
      expect(screen.getByText('+2')).toBeInTheDocument();
    });

    it('does not show overflow when laws fit', () => {
      render(
        <SelectedLawsSection
          selectedLaws={mockLaws.slice(0, 2)}
          onRemove={vi.fn()}
          onClearAll={vi.fn()}
          maxVisible={3}
        />
      );

      expect(screen.queryByText(/\+\d/)).not.toBeInTheDocument();
    });

    it('clicking +N expands to show all', async () => {
      const user = userEvent.setup();
      render(
        <SelectedLawsSection
          selectedLaws={mockLaws}
          onRemove={vi.fn()}
          onClearAll={vi.fn()}
          maxVisible={3}
        />
      );

      // Initially only 3 visible plus overflow indicator
      expect(screen.getByText('GDPR')).toBeInTheDocument();
      expect(screen.queryByText('NIS2')).not.toBeInTheDocument();
      expect(screen.getByText('+2')).toBeInTheDocument();

      // Click to expand
      await user.click(screen.getByText('+2'));

      // Now all should be visible
      expect(screen.getByText('GDPR')).toBeInTheDocument();
      expect(screen.getByText('NIS2')).toBeInTheDocument();
      expect(screen.queryByText('+2')).not.toBeInTheDocument();
    });
  });

  describe('interaction', () => {
    it('removing token calls onRemove', async () => {
      const onRemove = vi.fn();
      const user = userEvent.setup();
      render(
        <SelectedLawsSection
          selectedLaws={mockLaws.slice(0, 2)}
          onRemove={onRemove}
          onClearAll={vi.fn()}
        />
      );

      // Click remove on GDPR token
      const removeButtons = screen.getAllByRole('button', { name: /fjern/i });
      await user.click(removeButtons[0]);

      expect(onRemove).toHaveBeenCalledWith('gdpr');
    });

    it('clear all button calls onClearAll', async () => {
      const onClearAll = vi.fn();
      const user = userEvent.setup();
      render(
        <SelectedLawsSection
          selectedLaws={mockLaws.slice(0, 2)}
          onRemove={vi.fn()}
          onClearAll={onClearAll}
        />
      );

      await user.click(screen.getByRole('button', { name: /ryd alle/i }));

      expect(onClearAll).toHaveBeenCalled();
    });

    it('does not show clear all when no laws selected', () => {
      render(
        <SelectedLawsSection
          selectedLaws={[]}
          onRemove={vi.fn()}
          onClearAll={vi.fn()}
        />
      );

      expect(screen.queryByRole('button', { name: /ryd alle/i })).not.toBeInTheDocument();
    });
  });
});
