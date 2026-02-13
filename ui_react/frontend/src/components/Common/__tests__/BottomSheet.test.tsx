/**
 * Tests for BottomSheet component.
 *
 * Apple-style slide-up panel for mobile/narrow screens.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BottomSheet } from '../BottomSheet';

describe('BottomSheet', () => {
  it('renders nothing when not open', () => {
    const { container } = render(
      <BottomSheet isOpen={false} onClose={() => {}}>
        <div>Content</div>
      </BottomSheet>
    );

    expect(container.firstChild).toBeNull();
  });

  it('renders children when open', () => {
    render(
      <BottomSheet isOpen={true} onClose={() => {}}>
        <div>Sheet Content</div>
      </BottomSheet>
    );

    expect(screen.getByText('Sheet Content')).toBeInTheDocument();
  });

  it('renders backdrop when open', () => {
    render(
      <BottomSheet isOpen={true} onClose={() => {}}>
        <div>Content</div>
      </BottomSheet>
    );

    expect(screen.getByTestId('bottom-sheet-backdrop')).toBeInTheDocument();
  });

  it('calls onClose when backdrop is clicked', () => {
    const onClose = vi.fn();
    render(
      <BottomSheet isOpen={true} onClose={onClose}>
        <div>Content</div>
      </BottomSheet>
    );

    fireEvent.click(screen.getByTestId('bottom-sheet-backdrop'));
    expect(onClose).toHaveBeenCalled();
  });

  it('renders close button with handle', () => {
    render(
      <BottomSheet isOpen={true} onClose={() => {}}>
        <div>Content</div>
      </BottomSheet>
    );

    // Apple-style drag handle
    expect(screen.getByTestId('bottom-sheet-handle')).toBeInTheDocument();
  });

  it('calls onClose when handle is clicked', () => {
    const onClose = vi.fn();
    render(
      <BottomSheet isOpen={true} onClose={onClose}>
        <div>Content</div>
      </BottomSheet>
    );

    fireEvent.click(screen.getByTestId('bottom-sheet-handle'));
    expect(onClose).toHaveBeenCalled();
  });

  it('renders with title when provided', () => {
    render(
      <BottomSheet isOpen={true} onClose={() => {}} title="Kilder">
        <div>Content</div>
      </BottomSheet>
    );

    expect(screen.getByText('Kilder')).toBeInTheDocument();
  });

  it('applies bottom-sheet class to container', () => {
    render(
      <BottomSheet isOpen={true} onClose={() => {}}>
        <div>Content</div>
      </BottomSheet>
    );

    expect(screen.getByRole('dialog')).toHaveClass('bottom-sheet');
  });
});
