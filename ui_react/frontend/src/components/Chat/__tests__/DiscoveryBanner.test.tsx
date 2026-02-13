/**
 * Tests for DiscoveryBanner component.
 *
 * TDD: Tests written first per design spec (DESIGN_ai-law-discovery.md, C4).
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DiscoveryBanner } from '../DiscoveryBanner';
import type { DiscoveryMatch } from '../../../types';

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
      <div {...props}>{children}</div>
    ),
  },
}));

const autoMatch: DiscoveryMatch = {
  corpus_id: 'ai-act',
  confidence: 0.92,
  reason: 'retrieval_probe',
  display_name: 'AI-Act',
};

const suggestMatch: DiscoveryMatch = {
  corpus_id: 'gdpr',
  confidence: 0.68,
  reason: 'retrieval_probe',
  display_name: 'GDPR',
};

const lowMatch: DiscoveryMatch = {
  corpus_id: 'nis2',
  confidence: 0.38,
  reason: 'retrieval_probe',
  display_name: 'NIS2',
};

describe('DiscoveryBanner', () => {
  describe('AUTO gate', () => {
    it('renders with blue styling for AUTO gate', () => {
      const { container } = render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch]}
        />
      );

      const banner = container.firstChild as HTMLElement;
      expect(banner.className).toContain('bg-blue-50');
      expect(banner.className).toContain('border-blue-200');
    });

    it('shows "Fundet i:" text for AUTO gate', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch]}
        />
      );

      expect(screen.getByText(/Fundet i:/)).toBeInTheDocument();
    });

    it('displays law SHORTNAME (uppercase corpus_id)', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch]}
        />
      );

      expect(screen.getByText('AI-ACT')).toBeInTheDocument();
    });

    it('renders multiple laws with SHORTNAME format', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch, { ...suggestMatch, confidence: 0.82 }]}
        />
      );

      expect(screen.getByText('AI-ACT')).toBeInTheDocument();
      expect(screen.getByText('GDPR')).toBeInTheDocument();
    });

    it('does not show secondary text for AUTO gate', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch]}
        />
      );

      expect(screen.queryByText(/ufuldstændigt/)).not.toBeInTheDocument();
    });

    it('does not show CTA buttons for AUTO gate', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch]}
        />
      );

      expect(screen.queryByText('Vælg manuelt')).not.toBeInTheDocument();
      expect(screen.queryByText(/Omformuler/)).not.toBeInTheDocument();
    });
  });

  describe('SUGGEST gate', () => {
    it('renders with amber styling for SUGGEST gate', () => {
      const { container } = render(
        <DiscoveryBanner
          gate="SUGGEST"
          matches={[suggestMatch]}
        />
      );

      const banner = container.firstChild as HTMLElement;
      expect(banner.className).toContain('bg-amber-50');
      expect(banner.className).toContain('border-amber-200');
    });

    it('shows "Muligt relevant:" text for SUGGEST gate', () => {
      render(
        <DiscoveryBanner
          gate="SUGGEST"
          matches={[suggestMatch]}
        />
      );

      expect(screen.getByText(/Muligt relevant:/)).toBeInTheDocument();
    });

    it('shows caveat text for SUGGEST gate', () => {
      render(
        <DiscoveryBanner
          gate="SUGGEST"
          matches={[suggestMatch]}
        />
      );

      expect(screen.getByText(/Svaret kan være ufuldstændigt/)).toBeInTheDocument();
    });
  });

  describe('ABSTAIN gate', () => {
    it('renders nothing for ABSTAIN gate (AbstentionCard handles it)', () => {
      const { container } = render(
        <DiscoveryBanner
          gate="ABSTAIN"
          matches={[lowMatch]}
        />
      );

      expect(container.firstChild).toBeNull();
    });
  });

  describe('empty matches', () => {
    it('renders nothing when matches array is empty', () => {
      const { container } = render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[]}
        />
      );

      expect(container.firstChild).toBeNull();
    });
  });

  describe('resolvedCorpora filtering', () => {
    it('only shows matches in resolvedCorpora when provided', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch, suggestMatch, lowMatch]}
          resolvedCorpora={['ai-act', 'gdpr']}
        />
      );

      expect(screen.getByText('AI-ACT')).toBeInTheDocument();
      expect(screen.getByText('GDPR')).toBeInTheDocument();
      expect(screen.queryByText('NIS2')).not.toBeInTheDocument();
    });

    it('shows all matches when resolvedCorpora is not provided', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch, suggestMatch]}
        />
      );

      expect(screen.getByText('AI-ACT')).toBeInTheDocument();
      expect(screen.getByText('GDPR')).toBeInTheDocument();
    });
  });

  describe('fallback display name', () => {
    it('uses uppercase corpus_id as SHORTNAME', () => {
      const matchWithoutName: DiscoveryMatch = {
        corpus_id: 'data-act',
        confidence: 0.85,
        reason: 'alias_match',
      };

      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[matchWithoutName]}
        />
      );

      expect(screen.getByText('DATA-ACT')).toBeInTheDocument();
    });
  });

  describe('lock button (R1.1)', () => {
    it('renders lock button when onLock provided', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch]}
          onLock={vi.fn()}
        />
      );

      expect(screen.getByRole('button', { name: /Lås/i })).toBeInTheDocument();
    });

    it('does not render lock button when onLock absent', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch]}
        />
      );

      expect(screen.queryByRole('button', { name: /Lås/i })).not.toBeInTheDocument();
    });

    it('calls onLock when lock button is clicked', () => {
      const onLock = vi.fn();
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch]}
          onLock={onLock}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /Lås/i }));
      expect(onLock).toHaveBeenCalledOnce();
    });
  });

  describe('accessibility', () => {
    it('has appropriate role for screen readers', () => {
      render(
        <DiscoveryBanner
          gate="AUTO"
          matches={[autoMatch]}
        />
      );

      expect(screen.getByRole('status')).toBeInTheDocument();
    });
  });
});
