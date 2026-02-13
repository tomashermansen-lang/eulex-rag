/**
 * Tests for DiscoveryStatusPanel component.
 *
 * TDD: Tests for checkbox-based discovery list with enriched display,
 * pre-checking AUTO-tier items, and lock with user-selected corpora.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DiscoveryStatusPanel } from '../DiscoveryStatusPanel';
import type { DiscoveryMatch, CorpusInfo } from '../../../types';

// Mock Tooltip — renders children + stores content as data attribute
vi.mock('../../Common/Tooltip', () => ({
  Tooltip: ({
    content,
    children,
  }: {
    content: string;
    children: React.ReactNode;
  }) => (
    <span data-tooltip={content}>{children}</span>
  ),
}));

const mockMatches: DiscoveryMatch[] = [
  { corpus_id: 'ai-act', confidence: 0.82, reason: 'retrieval_probe', display_name: 'AI-Act' },
  { corpus_id: 'nis2', confidence: 0.77, reason: 'retrieval_probe', display_name: 'NIS2-direktivet' },
];

const mockCorpora: CorpusInfo[] = [
  {
    id: 'ai-act',
    name: 'Forordning om kunstig intelligens (AI Act)',
    fullname: 'Europa-Parlamentets og Rådets forordning (EU) 2024/1689 om kunstig intelligens',
    eurovoc_labels: ['kunstig intelligens', 'ny teknologi', 'digital omstilling'],
  },
  {
    id: 'nis2',
    name: 'NIS2-direktivet',
    fullname: 'Europa-Parlamentets direktiv om netværks- og informationssikkerhed',
    eurovoc_labels: ['cybersikkerhed', 'informationssikkerhed'],
  },
  {
    id: 'gdpr',
    name: 'Persondataforordningen (GDPR)',
    fullname: 'General Data Protection Regulation',
    eurovoc_labels: ['databeskyttelse', 'personoplysninger'],
  },
];

describe('DiscoveryStatusPanel', () => {
  describe('enriched display format', () => {
    it('shows SHORTNAME (uppercase corpus_id) for each match', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
        />
      );

      expect(screen.getByText('AI-ACT')).toBeInTheDocument();
      expect(screen.getByText('NIS2')).toBeInTheDocument();
    });

    it('shows displayname after SHORTNAME separator', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
        />
      );

      expect(screen.getByText('Forordning om kunstig intelligens (AI Act)')).toBeInTheDocument();
    });

    it('renders fullname as Tooltip content', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
        />
      );

      const aiActTooltip = screen.getByText('AI-ACT').closest('[data-tooltip]');
      expect(aiActTooltip).toHaveAttribute(
        'data-tooltip',
        'Europa-Parlamentets og Rådets forordning (EU) 2024/1689 om kunstig intelligens'
      );
    });

    it('shows confidence text labels instead of raw percentages', () => {
      const mixedMatches: DiscoveryMatch[] = [
        { corpus_id: 'ai-act', confidence: 0.82, reason: 'retrieval_probe', display_name: 'AI-Act' },
        { corpus_id: 'gdpr', confidence: 0.60, reason: 'retrieval_probe', display_name: 'GDPR' },
        { corpus_id: 'dora', confidence: 0.40, reason: 'retrieval_probe', display_name: 'DORA' },
      ];

      render(
        <DiscoveryStatusPanel
          discoveries={mixedMatches}
          corpora={mockCorpora}
        />
      );

      // Text labels instead of percentages
      expect(screen.getByText('Høj')).toBeInTheDocument();
      expect(screen.getByText('Mulig')).toBeInTheDocument();
      expect(screen.getByText('Lav')).toBeInTheDocument();
    });

    it('shows explanatory tooltip on hover with percentage and description', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
        />
      );

      const labels = screen.getAllByText('Høj');
      expect(labels[0].closest('[title]')).toHaveAttribute(
        'title',
        'Relevans: 82% — Høj sandsynlighed for at denne lov er relevant for dit spørgsmål'
      );
      expect(labels[1].closest('[title]')).toHaveAttribute(
        'title',
        'Relevans: 77% — Høj sandsynlighed for at denne lov er relevant for dit spørgsmål'
      );
    });

    it('shows tier-appropriate tooltip for Mulig and Lav', () => {
      const mixedMatches: DiscoveryMatch[] = [
        { corpus_id: 'gdpr', confidence: 0.60, reason: 'retrieval_probe', display_name: 'GDPR' },
        { corpus_id: 'dora', confidence: 0.40, reason: 'retrieval_probe', display_name: 'DORA' },
      ];

      render(
        <DiscoveryStatusPanel
          discoveries={mixedMatches}
          corpora={mockCorpora}
        />
      );

      const mulig = screen.getByText('Mulig');
      expect(mulig.closest('[title]')).toHaveAttribute(
        'title',
        'Relevans: 60% — Mulig relevans, gennemgå om loven passer til dit spørgsmål'
      );

      const lav = screen.getByText('Lav');
      expect(lav.closest('[title]')).toHaveAttribute(
        'title',
        'Relevans: 40% — Lav sandsynlighed, medtaget for fuldstændighed'
      );
    });

    it('renders EuroVoc badge with label count', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
        />
      );

      expect(screen.getByText('3 EuroVoc')).toBeInTheDocument();
      expect(screen.getByText('2 EuroVoc')).toBeInTheDocument();
    });

    it('shows EuroVoc label names on hover', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
        />
      );

      const aiActBadge = screen.getByText('3 EuroVoc');
      expect(aiActBadge).toHaveAttribute('title', 'kunstig intelligens, ny teknologi, digital omstilling');

      const nis2Badge = screen.getByText('2 EuroVoc');
      expect(nis2Badge).toHaveAttribute('title', 'cybersikkerhed, informationssikkerhed');
    });

    it('falls back to display_name when no corpora match', () => {
      const unmatchedMatch: DiscoveryMatch[] = [
        { corpus_id: 'unknown-law', confidence: 0.70, reason: 'retrieval_probe', display_name: 'Unknown Law' },
      ];

      render(
        <DiscoveryStatusPanel
          discoveries={unmatchedMatch}
          corpora={mockCorpora}
        />
      );

      expect(screen.getByText('UNKNOWN-LAW')).toBeInTheDocument();
      expect(screen.getByText('Unknown Law')).toBeInTheDocument();
    });
  });

  describe('checkboxes', () => {
    it('renders a checkbox for each discovered law', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
        />
      );

      const checkboxes = screen.getAllByRole('checkbox');
      expect(checkboxes).toHaveLength(2);
    });

    it('pre-checks AUTO-tier items (confidence >= 0.75)', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
        />
      );

      const checkboxes = screen.getAllByRole('checkbox');
      // Both mockMatches have confidence >= 0.75 (0.82 and 0.77)
      expect(checkboxes[0]).toBeChecked();
      expect(checkboxes[1]).toBeChecked();
    });

    it('does not pre-check SUGGEST-tier items (confidence < 0.75)', () => {
      const mixedMatches: DiscoveryMatch[] = [
        { corpus_id: 'ai-act', confidence: 0.82, reason: 'retrieval_probe', display_name: 'AI-Act' },
        { corpus_id: 'gdpr', confidence: 0.60, reason: 'retrieval_probe', display_name: 'GDPR' },
      ];

      render(
        <DiscoveryStatusPanel
          discoveries={mixedMatches}
          corpora={mockCorpora}
        />
      );

      const checkboxes = screen.getAllByRole('checkbox');
      expect(checkboxes[0]).toBeChecked();       // 0.82 >= 0.75
      expect(checkboxes[1]).not.toBeChecked();   // 0.60 < 0.75
    });

    it('toggles checkbox on click', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
          onLock={vi.fn()}
        />
      );

      const checkboxes = screen.getAllByRole('checkbox');
      // Initially checked (AUTO-tier)
      expect(checkboxes[0]).toBeChecked();
      // Click to uncheck
      fireEvent.click(checkboxes[0]);
      expect(checkboxes[0]).not.toBeChecked();
      // Click to re-check
      fireEvent.click(checkboxes[0]);
      expect(checkboxes[0]).toBeChecked();
    });
  });

  describe('lock button', () => {
    it('renders lock button when onLock provided', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
          onLock={vi.fn()}
        />
      );

      expect(screen.getByRole('button', { name: /Brug fundne love som søgeområde/ })).toBeInTheDocument();
    });

    it('calls onLock with checked corpus IDs', () => {
      const onLock = vi.fn();
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
          onLock={onLock}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: /Brug fundne love som søgeområde/ }));
      // Both items are AUTO-tier (>= 0.75), so both pre-checked
      expect(onLock).toHaveBeenCalledWith(['ai-act', 'nis2']);
    });

    it('calls onLock with only checked items after unchecking one', () => {
      const onLock = vi.fn();
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
          onLock={onLock}
        />
      );

      // Uncheck nis2 (second checkbox)
      const checkboxes = screen.getAllByRole('checkbox');
      fireEvent.click(checkboxes[1]);

      // Now click lock
      fireEvent.click(screen.getByRole('button', { name: /Brug fundne love som søgeområde/ }));
      expect(onLock).toHaveBeenCalledWith(['ai-act']);
    });

    it('disables button when no items are checked', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
          onLock={vi.fn()}
        />
      );

      // Uncheck both
      const checkboxes = screen.getAllByRole('checkbox');
      fireEvent.click(checkboxes[0]);
      fireEvent.click(checkboxes[1]);

      expect(screen.getByRole('button', { name: /Brug fundne love som søgeområde/ })).toBeDisabled();
    });

    it('does not render lock button when onLock is not provided', () => {
      render(
        <DiscoveryStatusPanel
          discoveries={mockMatches}
          corpora={mockCorpora}
        />
      );

      expect(screen.queryByRole('button', { name: /Brug fundne love som søgeområde/ })).not.toBeInTheDocument();
    });
  });

  describe('idle state', () => {
    it('shows idle message when no discoveries', () => {
      render(<DiscoveryStatusPanel />);

      expect(screen.getByText(/Relevante love identificeres automatisk/)).toBeInTheDocument();
    });
  });

  describe('loading state', () => {
    it('shows shimmer when loading', () => {
      const { container } = render(<DiscoveryStatusPanel isLoading />);

      expect(container.querySelectorAll('.animate-pulse')).toHaveLength(2);
    });
  });
});
