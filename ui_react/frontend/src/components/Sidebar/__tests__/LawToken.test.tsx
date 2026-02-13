/**
 * Tests for LawToken component.
 *
 * TDD: Test token/pill rendering for selected laws.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { LawToken } from '../LawToken';

describe('LawToken', () => {
  describe('rendering', () => {
    it('renders law name', () => {
      render(<LawToken name="GDPR" onRemove={vi.fn()} />);

      expect(screen.getByText('GDPR')).toBeInTheDocument();
    });

    it('renders remove button', () => {
      render(<LawToken name="GDPR" onRemove={vi.fn()} />);

      expect(screen.getByRole('button', { name: /fjern/i })).toBeInTheDocument();
    });

    it('truncates long names', () => {
      render(
        <LawToken
          name="Very Long Law Name That Should Be Truncated"
          onRemove={vi.fn()}
        />
      );

      // The component should have text-overflow handling
      const token = screen.getByText('Very Long Law Name That Should Be Truncated');
      expect(token).toHaveClass('truncate');
    });
  });

  describe('interaction', () => {
    it('clicking remove button calls onRemove', async () => {
      const onRemove = vi.fn();
      const user = userEvent.setup();
      render(<LawToken name="GDPR" id="gdpr" onRemove={onRemove} />);

      await user.click(screen.getByRole('button', { name: /fjern/i }));

      expect(onRemove).toHaveBeenCalledWith('gdpr');
    });

    it('remove button stops propagation', async () => {
      const onRemove = vi.fn();
      const onContainerClick = vi.fn();
      const user = userEvent.setup();

      render(
        <div onClick={onContainerClick}>
          <LawToken name="GDPR" id="gdpr" onRemove={onRemove} />
        </div>
      );

      await user.click(screen.getByRole('button', { name: /fjern/i }));

      expect(onRemove).toHaveBeenCalled();
      expect(onContainerClick).not.toHaveBeenCalled();
    });
  });

  describe('accessibility', () => {
    it('has accessible remove label', () => {
      render(<LawToken name="AI Act" onRemove={vi.fn()} />);

      const removeButton = screen.getByRole('button');
      expect(removeButton).toHaveAttribute('aria-label', 'Fjern AI Act');
    });
  });

  describe('styling', () => {
    it('applies pill/token styling', () => {
      const { container } = render(<LawToken name="GDPR" onRemove={vi.fn()} />);

      // Check for pill-like styling (rounded, inline-flex)
      const token = container.firstChild as HTMLElement;
      expect(token).toHaveClass('rounded-full');
      expect(token).toHaveClass('inline-flex');
    });
  });
});
