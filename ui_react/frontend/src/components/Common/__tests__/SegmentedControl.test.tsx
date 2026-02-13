/**
 * Tests for SegmentedControl component.
 *
 * TDD: Tests written to cover selection, styling, and accessibility.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SegmentedControl } from '../SegmentedControl';

describe('SegmentedControl', () => {
  const mockOnChange = vi.fn();

  const defaultOptions = [
    { value: 'option1', label: 'Option 1' },
    { value: 'option2', label: 'Option 2' },
    { value: 'option3', label: 'Option 3' },
  ] as const;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders all options', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
        />
      );

      expect(screen.getByText('Option 1')).toBeInTheDocument();
      expect(screen.getByText('Option 2')).toBeInTheDocument();
      expect(screen.getByText('Option 3')).toBeInTheDocument();
    });

    it('renders buttons for each option', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
        />
      );

      const buttons = screen.getAllByRole('tab');
      expect(buttons).toHaveLength(3);
    });

    it('renders with tablist role', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
        />
      );

      expect(screen.getByRole('tablist')).toBeInTheDocument();
    });
  });

  describe('selection', () => {
    it('marks selected option with aria-selected', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option2"
          onChange={mockOnChange}
        />
      );

      const option1 = screen.getByRole('tab', { name: 'Option 1' });
      const option2 = screen.getByRole('tab', { name: 'Option 2' });
      const option3 = screen.getByRole('tab', { name: 'Option 3' });

      expect(option1).toHaveAttribute('aria-selected', 'false');
      expect(option2).toHaveAttribute('aria-selected', 'true');
      expect(option3).toHaveAttribute('aria-selected', 'false');
    });

    it('calls onChange when option clicked', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
        />
      );

      fireEvent.click(screen.getByText('Option 2'));

      expect(mockOnChange).toHaveBeenCalledWith('option2');
      expect(mockOnChange).toHaveBeenCalledTimes(1);
    });

    it('calls onChange when already selected option clicked', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
        />
      );

      fireEvent.click(screen.getByText('Option 1'));

      expect(mockOnChange).toHaveBeenCalledWith('option1');
    });
  });

  describe('sizes', () => {
    it('uses medium size by default', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
        />
      );

      const button = screen.getByRole('tab', { name: 'Option 1' });
      expect(button).toHaveClass('px-4', 'py-2', 'text-sm');
    });

    it('applies small size classes', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
          size="sm"
        />
      );

      const button = screen.getByRole('tab', { name: 'Option 1' });
      expect(button).toHaveClass('px-3', 'py-1', 'text-xs');
    });
  });

  describe('custom colors', () => {
    it('applies green color for selected option with green selectedColor', () => {
      const coloredOptions = [
        { value: 'pass', label: 'Pass', selectedColor: 'green' as const },
        { value: 'fail', label: 'Fail', selectedColor: 'red' as const },
      ];

      render(
        <SegmentedControl
          options={coloredOptions}
          value="pass"
          onChange={mockOnChange}
        />
      );

      const passButton = screen.getByRole('tab', { name: 'Pass' });
      expect(passButton).toHaveClass('text-green-600');
    });

    it('applies red color for selected option with red selectedColor', () => {
      const coloredOptions = [
        { value: 'pass', label: 'Pass', selectedColor: 'green' as const },
        { value: 'fail', label: 'Fail', selectedColor: 'red' as const },
      ];

      render(
        <SegmentedControl
          options={coloredOptions}
          value="fail"
          onChange={mockOnChange}
        />
      );

      const failButton = screen.getByRole('tab', { name: 'Fail' });
      expect(failButton).toHaveClass('text-red-600');
    });

    it('applies amber color for selected option with amber selectedColor', () => {
      const coloredOptions = [
        { value: 'escalated', label: 'Escalated', selectedColor: 'amber' as const },
      ];

      render(
        <SegmentedControl
          options={coloredOptions}
          value="escalated"
          onChange={mockOnChange}
        />
      );

      const button = screen.getByRole('tab', { name: 'Escalated' });
      expect(button).toHaveClass('text-amber-600');
    });
  });

  describe('equal width', () => {
    it('does not apply flex-1 by default', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
        />
      );

      const button = screen.getByRole('tab', { name: 'Option 1' });
      expect(button).not.toHaveClass('flex-1');
    });

    it('applies flex-1 when equalWidth is true', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
          equalWidth={true}
        />
      );

      const button = screen.getByRole('tab', { name: 'Option 1' });
      expect(button).toHaveClass('flex-1');
    });
  });

  describe('custom className', () => {
    it('applies additional className to container', () => {
      render(
        <SegmentedControl
          options={defaultOptions as unknown as { value: string; label: string }[]}
          value="option1"
          onChange={mockOnChange}
          className="custom-class"
        />
      );

      const container = screen.getByRole('tablist');
      expect(container).toHaveClass('custom-class');
    });
  });
});
