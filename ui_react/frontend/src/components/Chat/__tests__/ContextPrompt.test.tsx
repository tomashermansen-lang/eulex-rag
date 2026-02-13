/**
 * Tests for ContextPrompt component.
 *
 * Verifies: R1.5 (inline follow-up prompt) and R1.6 (lock/continue actions).
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ContextPrompt } from '../ContextPrompt';

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: Record<string, unknown>) => {
      const { initial, animate, transition, ...rest } = props;
      return <div {...rest}>{children as React.ReactNode}</div>;
    },
  },
}));

describe('ContextPrompt', () => {
  const defaultProps = {
    lawNames: ['AI-ACT', 'NIS2'],
    onLock: vi.fn(),
    onContinue: vi.fn(),
  };

  it('renders law names in the prompt text', () => {
    render(<ContextPrompt {...defaultProps} />);

    // Law names appear in both prompt text and lock button
    expect(screen.getByText(/AI fandt/)).toBeInTheDocument();
    expect(screen.getAllByText(/AI-ACT/)).toHaveLength(2); // prompt + button
  });

  it('renders lock button', () => {
    render(<ContextPrompt {...defaultProps} />);

    expect(screen.getByRole('button', { name: /Lås til/i })).toBeInTheDocument();
  });

  it('renders continue button', () => {
    render(<ContextPrompt {...defaultProps} />);

    expect(screen.getByRole('button', { name: /Fortsæt AI-søgning/i })).toBeInTheDocument();
  });

  it('calls onLock when lock button is clicked', () => {
    const onLock = vi.fn();
    render(<ContextPrompt {...defaultProps} onLock={onLock} />);

    fireEvent.click(screen.getByRole('button', { name: /Lås til/i }));
    expect(onLock).toHaveBeenCalledOnce();
  });

  it('calls onContinue when continue button is clicked', () => {
    const onContinue = vi.fn();
    render(<ContextPrompt {...defaultProps} onContinue={onContinue} />);

    fireEvent.click(screen.getByRole('button', { name: /Fortsæt AI-søgning/i }));
    expect(onContinue).toHaveBeenCalledOnce();
  });

  it('displays single law name correctly', () => {
    render(<ContextPrompt {...defaultProps} lawNames={['GDPR']} />);

    // Appears in both prompt text and lock button
    expect(screen.getAllByText(/GDPR/)).toHaveLength(2);
    expect(screen.getByText(/AI fandt/)).toBeInTheDocument();
  });

  it('has status role for accessibility', () => {
    render(<ContextPrompt {...defaultProps} />);

    expect(screen.getByRole('status')).toBeInTheDocument();
  });
});
