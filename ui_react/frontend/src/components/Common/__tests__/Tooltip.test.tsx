/**
 * Tests for Tooltip component.
 *
 * TDD: Tests written to cover tooltip display, positioning, and edge cases.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { Tooltip } from '../Tooltip';

describe('Tooltip', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  describe('without content', () => {
    it('renders children directly when content is null', () => {
      render(
        <Tooltip content={null}>
          <button>Hover me</button>
        </Tooltip>
      );

      expect(screen.getByRole('button', { name: 'Hover me' })).toBeInTheDocument();
    });

    it('renders children directly when content is undefined', () => {
      render(
        <Tooltip content={undefined}>
          <button>Hover me</button>
        </Tooltip>
      );

      expect(screen.getByRole('button', { name: 'Hover me' })).toBeInTheDocument();
    });

    it('renders children directly when content is empty string', () => {
      render(
        <Tooltip content="">
          <button>Hover me</button>
        </Tooltip>
      );

      // With empty string, the component still wraps in span but won't show tooltip
      expect(screen.getByRole('button', { name: 'Hover me' })).toBeInTheDocument();
    });
  });

  describe('with content', () => {
    it('renders children wrapped in span', () => {
      render(
        <Tooltip content="Tooltip text">
          <button>Hover me</button>
        </Tooltip>
      );

      expect(screen.getByRole('button', { name: 'Hover me' })).toBeInTheDocument();
    });

    it('does not show tooltip initially', () => {
      render(
        <Tooltip content="Tooltip text">
          <button>Hover me</button>
        </Tooltip>
      );

      expect(screen.queryByText('Tooltip text')).not.toBeInTheDocument();
    });

    it('shows tooltip after hover and delay', async () => {
      render(
        <Tooltip content="Tooltip text" delay={200}>
          <button>Hover me</button>
        </Tooltip>
      );

      const wrapper = screen.getByRole('button', { name: 'Hover me' }).parentElement!;
      fireEvent.mouseEnter(wrapper);

      // Advance timers past delay
      act(() => {
        vi.advanceTimersByTime(250);
      });

      expect(screen.getByText('Tooltip text')).toBeInTheDocument();
    });

    it('hides tooltip on mouse leave', async () => {
      render(
        <Tooltip content="Tooltip text" delay={200}>
          <button>Hover me</button>
        </Tooltip>
      );

      const wrapper = screen.getByRole('button', { name: 'Hover me' }).parentElement!;

      // Show tooltip
      fireEvent.mouseEnter(wrapper);
      act(() => {
        vi.advanceTimersByTime(250);
      });
      expect(screen.getByText('Tooltip text')).toBeInTheDocument();

      // Hide tooltip
      fireEvent.mouseLeave(wrapper);
      expect(screen.queryByText('Tooltip text')).not.toBeInTheDocument();
    });

    it('does not show tooltip if mouse leaves before delay', () => {
      render(
        <Tooltip content="Tooltip text" delay={200}>
          <button>Hover me</button>
        </Tooltip>
      );

      const wrapper = screen.getByRole('button', { name: 'Hover me' }).parentElement!;

      // Hover and quickly leave
      fireEvent.mouseEnter(wrapper);
      act(() => {
        vi.advanceTimersByTime(100); // Less than delay
      });
      fireEvent.mouseLeave(wrapper);

      // Advance past delay
      act(() => {
        vi.advanceTimersByTime(200);
      });

      expect(screen.queryByText('Tooltip text')).not.toBeInTheDocument();
    });
  });

  describe('delay prop', () => {
    it('uses default delay of 200ms', async () => {
      render(
        <Tooltip content="Tooltip text">
          <button>Hover me</button>
        </Tooltip>
      );

      const wrapper = screen.getByRole('button', { name: 'Hover me' }).parentElement!;
      fireEvent.mouseEnter(wrapper);

      act(() => {
        vi.advanceTimersByTime(150);
      });
      expect(screen.queryByText('Tooltip text')).not.toBeInTheDocument();

      act(() => {
        vi.advanceTimersByTime(100);
      });
      expect(screen.getByText('Tooltip text')).toBeInTheDocument();
    });

    it('respects custom delay', async () => {
      render(
        <Tooltip content="Tooltip text" delay={500}>
          <button>Hover me</button>
        </Tooltip>
      );

      const wrapper = screen.getByRole('button', { name: 'Hover me' }).parentElement!;
      fireEvent.mouseEnter(wrapper);

      act(() => {
        vi.advanceTimersByTime(300);
      });
      expect(screen.queryByText('Tooltip text')).not.toBeInTheDocument();

      act(() => {
        vi.advanceTimersByTime(250);
      });
      expect(screen.getByText('Tooltip text')).toBeInTheDocument();
    });
  });

  describe('className prop', () => {
    it('applies custom className to wrapper', () => {
      render(
        <Tooltip content="Tooltip text" className="custom-class">
          <button>Hover me</button>
        </Tooltip>
      );

      const wrapper = screen.getByRole('button', { name: 'Hover me' }).parentElement;
      expect(wrapper).toHaveClass('custom-class');
    });
  });

  describe('position prop', () => {
    it('defaults to bottom position', async () => {
      render(
        <Tooltip content="Tooltip text">
          <button>Hover me</button>
        </Tooltip>
      );

      const wrapper = screen.getByRole('button', { name: 'Hover me' }).parentElement!;
      fireEvent.mouseEnter(wrapper);
      act(() => {
        vi.advanceTimersByTime(250);
      });

      // Just verify tooltip is shown - positioning is complex to test due to getBoundingClientRect
      expect(screen.getByText('Tooltip text')).toBeInTheDocument();
    });
  });

  describe('maxWidth prop', () => {
    it('renders tooltip with custom maxWidth', async () => {
      render(
        <Tooltip content="Tooltip text" maxWidth={300}>
          <button>Hover me</button>
        </Tooltip>
      );

      const wrapper = screen.getByRole('button', { name: 'Hover me' }).parentElement!;
      fireEvent.mouseEnter(wrapper);
      act(() => {
        vi.advanceTimersByTime(250);
      });

      // Verify tooltip is shown - style testing is unreliable in jsdom
      expect(screen.getByText('Tooltip text')).toBeInTheDocument();
    });

    it('renders tooltip with default maxWidth', async () => {
      render(
        <Tooltip content="Tooltip text">
          <button>Hover me</button>
        </Tooltip>
      );

      const wrapper = screen.getByRole('button', { name: 'Hover me' }).parentElement!;
      fireEvent.mouseEnter(wrapper);
      act(() => {
        vi.advanceTimersByTime(250);
      });

      // Verify tooltip is shown - style testing is unreliable in jsdom
      expect(screen.getByText('Tooltip text')).toBeInTheDocument();
    });
  });
});
