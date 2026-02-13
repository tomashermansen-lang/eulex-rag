/**
 * Tests for SourcesPanel component.
 *
 * TDD: Tests written to cover panel expansion and source display.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { SourcesPanel } from '../SourcesPanel';
import type { Reference } from '../../../types';

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
      <div {...props}>{children}</div>
    ),
    svg: ({ children, ...props }: React.SVGAttributes<SVGSVGElement>) => (
      <svg {...props}>{children}</svg>
    ),
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock SourceItem
vi.mock('../SourceItem', () => ({
  SourceItem: ({
    reference,
    isCited,
  }: {
    reference: Reference;
    isCited?: boolean;
  }) => (
    <div data-testid={`source-item-${reference.idx}`} data-cited={isCited}>
      {reference.display}
    </div>
  ),
}));

// Mock getRefAnchorId
vi.mock('../../../utils/citations', () => ({
  getRefAnchorId: (idx: number | string, messageId?: string) =>
    `ref-${messageId || 'default'}-${idx}`,
}));

describe('SourcesPanel', () => {
  const mockReferences: Reference[] = [
    { idx: 1, display: 'Article 1', chunk_text: 'Text 1' },
    { idx: 2, display: 'Article 2', chunk_text: 'Text 2' },
    { idx: 3, display: 'Article 3', chunk_text: 'Text 3' },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('empty state', () => {
    it('renders nothing when references array is empty', () => {
      const { container } = render(<SourcesPanel references={[]} />);

      expect(container.innerHTML).toBe('');
    });
  });

  describe('rendering', () => {
    it('renders sources count', () => {
      render(<SourcesPanel references={mockReferences} />);

      expect(screen.getByText('Kilder (3)')).toBeInTheDocument();
    });

    it('renders as collapsed by default', () => {
      render(<SourcesPanel references={mockReferences} />);

      // Source items should not be visible initially
      expect(screen.queryByTestId('source-item-1')).not.toBeInTheDocument();
    });

    it('renders details element', () => {
      const { container } = render(<SourcesPanel references={mockReferences} />);

      const details = container.querySelector('details');
      expect(details).toBeInTheDocument();
      expect(details).not.toHaveAttribute('open');
    });
  });

  describe('expansion', () => {
    it('shows sources when expanded via toggle event', () => {
      const { container } = render(<SourcesPanel references={mockReferences} />);

      // Simulate details toggle event
      const details = container.querySelector('details')!;
      Object.defineProperty(details, 'open', { value: true, writable: true });
      fireEvent(details, new Event('toggle', { bubbles: true }));

      // Sources should now be visible
      expect(screen.getByTestId('source-item-1')).toBeInTheDocument();
      expect(screen.getByTestId('source-item-2')).toBeInTheDocument();
      expect(screen.getByTestId('source-item-3')).toBeInTheDocument();
    });

    it('renders source displays when expanded', () => {
      const { container } = render(<SourcesPanel references={mockReferences} />);

      // Simulate details toggle event
      const details = container.querySelector('details')!;
      Object.defineProperty(details, 'open', { value: true, writable: true });
      fireEvent(details, new Event('toggle', { bubbles: true }));

      expect(screen.getByText('Article 1')).toBeInTheDocument();
      expect(screen.getByText('Article 2')).toBeInTheDocument();
      expect(screen.getByText('Article 3')).toBeInTheDocument();
    });

    it('hides sources when collapsed after being open', () => {
      const { container } = render(<SourcesPanel references={mockReferences} />);
      const details = container.querySelector('details')!;

      // Expand
      Object.defineProperty(details, 'open', { value: true, writable: true });
      fireEvent(details, new Event('toggle', { bubbles: true }));
      expect(screen.getByTestId('source-item-1')).toBeInTheDocument();

      // Collapse
      Object.defineProperty(details, 'open', { value: false, writable: true });
      fireEvent(details, new Event('toggle', { bubbles: true }));
      expect(screen.queryByTestId('source-item-1')).not.toBeInTheDocument();
    });
  });

  describe('cited indices', () => {
    it('passes isCited=true for cited references', () => {
      const citedIndices = new Set([1, 3]);

      const { container } = render(
        <SourcesPanel references={mockReferences} citedIndices={citedIndices} />
      );

      // Expand via toggle event
      const details = container.querySelector('details')!;
      Object.defineProperty(details, 'open', { value: true, writable: true });
      fireEvent(details, new Event('toggle', { bubbles: true }));

      expect(screen.getByTestId('source-item-1')).toHaveAttribute(
        'data-cited',
        'true'
      );
      expect(screen.getByTestId('source-item-2')).toHaveAttribute(
        'data-cited',
        'false'
      );
      expect(screen.getByTestId('source-item-3')).toHaveAttribute(
        'data-cited',
        'true'
      );
    });

    it('handles string indices correctly', () => {
      const refsWithStringIdx: Reference[] = [
        { idx: '5', display: 'Article 5', chunk_text: 'Text' },
      ];
      const citedIndices = new Set([5]);

      const { container } = render(
        <SourcesPanel references={refsWithStringIdx} citedIndices={citedIndices} />
      );

      // Expand via toggle event
      const details = container.querySelector('details')!;
      Object.defineProperty(details, 'open', { value: true, writable: true });
      fireEvent(details, new Event('toggle', { bubbles: true }));

      expect(screen.getByTestId('source-item-5')).toHaveAttribute(
        'data-cited',
        'true'
      );
    });
  });

  describe('expand event', () => {
    it('expands panel when expandSource event is dispatched for matching ref', async () => {
      render(<SourcesPanel references={mockReferences} messageId="msg-1" />);

      // Panel should be collapsed initially
      expect(screen.queryByTestId('source-item-1')).not.toBeInTheDocument();

      // Dispatch expand event
      act(() => {
        window.dispatchEvent(
          new CustomEvent('expandSource', {
            detail: { refId: 'ref-msg-1-1' },
          })
        );
      });

      // Panel should now be expanded
      await waitFor(() => {
        expect(screen.getByTestId('source-item-1')).toBeInTheDocument();
      });
    });

    it('does not expand panel when expandSource event is for different ref', () => {
      render(<SourcesPanel references={mockReferences} messageId="msg-1" />);

      // Dispatch expand event for different message
      act(() => {
        window.dispatchEvent(
          new CustomEvent('expandSource', {
            detail: { refId: 'ref-msg-2-1' },
          })
        );
      });

      // Panel should still be collapsed
      expect(screen.queryByTestId('source-item-1')).not.toBeInTheDocument();
    });
  });

  describe('single reference', () => {
    it('shows correct count for single reference', () => {
      render(
        <SourcesPanel references={[mockReferences[0]]} />
      );

      expect(screen.getByText('Kilder (1)')).toBeInTheDocument();
    });
  });
});
