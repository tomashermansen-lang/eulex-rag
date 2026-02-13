/**
 * Tests for AbstentionCard component.
 *
 * TDD: Tests written first per design spec (DESIGN_ai-law-discovery.md, C5).
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AbstentionCard } from '../AbstentionCard';
import type { DiscoveryMatch } from '../../../types';

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
      <div {...props}>{children}</div>
    ),
  },
}));

const candidates: DiscoveryMatch[] = [
  { corpus_id: 'ai-act', confidence: 0.38, reason: 'retrieval_probe', display_name: 'AI-Act' },
  { corpus_id: 'gdpr', confidence: 0.31, reason: 'retrieval_probe', display_name: 'GDPR' },
  { corpus_id: 'nis2', confidence: 0.24, reason: 'retrieval_probe', display_name: 'NIS2' },
];

describe('AbstentionCard', () => {
  const defaultProps = {
    candidates,
    onSelectManual: vi.fn(),
    onRephrase: vi.fn(),
  };

  describe('rendering', () => {
    it('renders abstention headline', () => {
      render(<AbstentionCard {...defaultProps} />);

      expect(
        screen.getByText(/Kan ikke identificere relevant lovgivning/)
      ).toBeInTheDocument();
    });

    it('renders "Mulige love:" subheading', () => {
      render(<AbstentionCard {...defaultProps} />);

      expect(screen.getByText(/Mulige love:/)).toBeInTheDocument();
    });

    it('renders candidate pills with names and scores', () => {
      render(<AbstentionCard {...defaultProps} />);

      expect(screen.getByText('AI-Act')).toBeInTheDocument();
      expect(screen.getByText('GDPR')).toBeInTheDocument();
      expect(screen.getByText('NIS2')).toBeInTheDocument();
      expect(screen.getByText('38%')).toBeInTheDocument();
      expect(screen.getByText('31%')).toBeInTheDocument();
      expect(screen.getByText('24%')).toBeInTheDocument();
    });

    it('renders "Vælg manuelt" action button', () => {
      render(<AbstentionCard {...defaultProps} />);

      expect(
        screen.getByRole('button', { name: /Vælg manuelt/ })
      ).toBeInTheDocument();
    });

    it('renders "Omformuler" action button', () => {
      render(<AbstentionCard {...defaultProps} />);

      expect(
        screen.getByRole('button', { name: /Omformuler/ })
      ).toBeInTheDocument();
    });
  });

  describe('interactions', () => {
    it('calls onSelectManual with top candidate corpora when "Vælg manuelt" clicked', async () => {
      const onSelectManual = vi.fn();
      const user = userEvent.setup();

      render(
        <AbstentionCard
          {...defaultProps}
          onSelectManual={onSelectManual}
        />
      );

      await user.click(screen.getByRole('button', { name: /Vælg manuelt/ }));

      expect(onSelectManual).toHaveBeenCalledWith(
        ['ai-act', 'gdpr', 'nis2']
      );
    });

    it('calls onRephrase when "Omformuler" clicked', async () => {
      const onRephrase = vi.fn();
      const user = userEvent.setup();

      render(
        <AbstentionCard
          {...defaultProps}
          onRephrase={onRephrase}
        />
      );

      await user.click(screen.getByRole('button', { name: /Omformuler/ }));

      expect(onRephrase).toHaveBeenCalled();
    });
  });

  describe('empty candidates', () => {
    it('renders without candidate pills when no candidates', () => {
      render(
        <AbstentionCard
          candidates={[]}
          onSelectManual={defaultProps.onSelectManual}
          onRephrase={defaultProps.onRephrase}
        />
      );

      expect(
        screen.getByText(/Kan ikke identificere relevant lovgivning/)
      ).toBeInTheDocument();
      expect(screen.queryByText(/Mulige love:/)).not.toBeInTheDocument();
    });
  });

  describe('fallback display name', () => {
    it('uses corpus_id when display_name is not provided', () => {
      const nakedCandidates: DiscoveryMatch[] = [
        { corpus_id: 'data-act', confidence: 0.35, reason: 'retrieval_probe' },
      ];

      render(
        <AbstentionCard
          candidates={nakedCandidates}
          onSelectManual={defaultProps.onSelectManual}
          onRephrase={defaultProps.onRephrase}
        />
      );

      expect(screen.getByText('data-act')).toBeInTheDocument();
    });
  });

  describe('accessibility', () => {
    it('has alert role for screen readers', () => {
      render(<AbstentionCard {...defaultProps} />);

      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('action buttons meet touch target size (via padding)', () => {
      render(<AbstentionCard {...defaultProps} />);

      const manualButton = screen.getByRole('button', { name: /Vælg manuelt/ });
      // Button should have py-3 px-4 classes for 44x44pt minimum
      expect(manualButton.className).toContain('py-');
      expect(manualButton.className).toContain('px-');
    });
  });
});
