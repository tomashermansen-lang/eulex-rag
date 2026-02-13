/**
 * Tests for SourceItem component.
 *
 * TDD: Tests written to cover source item expand/collapse and display.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SourceItem } from '../SourceItem';
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

describe('SourceItem', () => {
  const mockReference: Reference = {
    idx: 1,
    display: 'Article 5 - Principles',
    chunk_text: 'Personal data shall be processed lawfully, fairly and in a transparent manner.',
    article: 'Article 5',
    score: 0.95,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders reference display with index (SI-01)', () => {
      render(<SourceItem reference={mockReference} />);

      expect(screen.getByText('[1] Article 5 - Principles')).toBeInTheDocument();
    });

    it('renders with correct anchor id', () => {
      const { container } = render(
        <SourceItem reference={mockReference} messageId="msg123" />
      );

      // Format is ref-{messageId}-{idx}
      const element = container.querySelector('#ref-msg123-1');
      expect(element).toBeInTheDocument();
    });

    it('renders with anchor id without messageId', () => {
      const { container } = render(<SourceItem reference={mockReference} />);

      const element = container.querySelector('#ref-1');
      expect(element).toBeInTheDocument();
    });
  });

  describe('cited badge', () => {
    it('shows citeret badge when isCited is true (SI-02)', () => {
      render(<SourceItem reference={mockReference} isCited={true} />);

      expect(screen.getByText('citeret')).toBeInTheDocument();
    });

    it('does not show citeret badge when isCited is false', () => {
      render(<SourceItem reference={mockReference} isCited={false} />);

      expect(screen.queryByText('citeret')).not.toBeInTheDocument();
    });

    it('does not show citeret badge by default', () => {
      render(<SourceItem reference={mockReference} />);

      expect(screen.queryByText('citeret')).not.toBeInTheDocument();
    });
  });

  describe('expand/collapse', () => {
    it('is collapsed by default', () => {
      render(<SourceItem reference={mockReference} />);

      expect(screen.queryByText('Kildetekst')).not.toBeInTheDocument();
    });

    it('expands content when clicked (SI-03)', () => {
      render(<SourceItem reference={mockReference} />);

      const button = screen.getByRole('button');
      fireEvent.click(button);

      expect(screen.getByText('Kildetekst')).toBeInTheDocument();
    });

    it('collapses content when clicked again (SI-04)', () => {
      render(<SourceItem reference={mockReference} />);

      const button = screen.getByRole('button');

      // Expand
      fireEvent.click(button);
      expect(screen.getByText('Kildetekst')).toBeInTheDocument();

      // Collapse
      fireEvent.click(button);
      expect(screen.queryByText('Kildetekst')).not.toBeInTheDocument();
    });

    it('starts expanded when defaultExpanded is true', () => {
      render(<SourceItem reference={mockReference} defaultExpanded={true} />);

      expect(screen.getByText('Kildetekst')).toBeInTheDocument();
    });

    it('shows chunk_text when expanded (SI-07)', () => {
      render(<SourceItem reference={mockReference} defaultExpanded={true} />);

      expect(screen.getByText(mockReference.chunk_text)).toBeInTheDocument();
    });

    it('has correct aria-expanded attribute', () => {
      render(<SourceItem reference={mockReference} />);

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-expanded', 'false');

      fireEvent.click(button);
      expect(button).toHaveAttribute('aria-expanded', 'true');
    });
  });

  describe('EUR-Lex link', () => {
    it('shows EUR-Lex link when sourceUrl provided (SI-05)', () => {
      render(
        <SourceItem
          reference={mockReference}
          sourceUrl="https://eur-lex.europa.eu/legal-content/DA/TXT/?uri=CELEX:32016R0679"
          defaultExpanded={true}
        />
      );

      expect(screen.getByText('Åbn i EUR-Lex')).toBeInTheDocument();
    });

    it('does not show EUR-Lex link when sourceUrl not provided', () => {
      render(<SourceItem reference={mockReference} defaultExpanded={true} />);

      expect(screen.queryByText('Åbn i EUR-Lex')).not.toBeInTheDocument();
    });

    it('link opens in new tab', () => {
      render(
        <SourceItem
          reference={mockReference}
          sourceUrl="https://eur-lex.europa.eu/test"
          defaultExpanded={true}
        />
      );

      const link = screen.getByText('Åbn i EUR-Lex');
      expect(link).toHaveAttribute('target', '_blank');
      expect(link).toHaveAttribute('rel', 'noopener noreferrer');
    });
  });

  describe('custom expand event', () => {
    it('listens for expandSource events (SI-06)', () => {
      // Test that the component adds event listener on mount
      const addEventListenerSpy = vi.spyOn(window, 'addEventListener');

      render(<SourceItem reference={mockReference} messageId="msg-1" />);

      expect(addEventListenerSpy).toHaveBeenCalledWith(
        'expandSource',
        expect.any(Function)
      );

      addEventListenerSpy.mockRestore();
    });

    it('removes event listener on unmount', () => {
      const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');

      const { unmount } = render(
        <SourceItem reference={mockReference} messageId="msg-1" />
      );

      unmount();

      expect(removeEventListenerSpy).toHaveBeenCalledWith(
        'expandSource',
        expect.any(Function)
      );

      removeEventListenerSpy.mockRestore();
    });
  });

  describe('title attribute', () => {
    it('has title with full reference info', () => {
      render(<SourceItem reference={mockReference} />);

      const span = screen.getByTitle('[1] Article 5 - Principles');
      expect(span).toBeInTheDocument();
    });
  });

  describe('selection state', () => {
    it('applies source-item-selected class when isSelected is true (SI-08)', () => {
      const { container } = render(
        <SourceItem reference={mockReference} isSelected={true} />
      );

      const element = container.querySelector('.source-item-selected');
      expect(element).toBeInTheDocument();
    });

    it('does not apply source-item-selected class when isSelected is false', () => {
      const { container } = render(
        <SourceItem reference={mockReference} isSelected={false} />
      );

      const element = container.querySelector('.source-item-selected');
      expect(element).not.toBeInTheDocument();
    });

    it('does not apply source-item-selected class by default', () => {
      const { container } = render(<SourceItem reference={mockReference} />);

      const element = container.querySelector('.source-item-selected');
      expect(element).not.toBeInTheDocument();
    });

    it('sets aria-selected="true" when isSelected is true (SI-09)', () => {
      const { container } = render(
        <SourceItem reference={mockReference} isSelected={true} />
      );

      const element = container.querySelector('[aria-selected="true"]');
      expect(element).toBeInTheDocument();
    });

    it('sets aria-selected="false" when isSelected is false (SI-10)', () => {
      const { container } = render(
        <SourceItem reference={mockReference} isSelected={false} />
      );

      const element = container.querySelector('[aria-selected="false"]');
      expect(element).toBeInTheDocument();
    });
  });
});
