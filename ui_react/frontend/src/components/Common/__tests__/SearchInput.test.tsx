/**
 * Tests for SearchInput component.
 *
 * TDD: Tests written to cover input, clear, and disabled states.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SearchInput } from '../SearchInput';

describe('SearchInput', () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders an input element', () => {
      render(<SearchInput value="" onChange={mockOnChange} />);

      expect(screen.getByRole('textbox')).toBeInTheDocument();
    });

    it('uses default placeholder', () => {
      render(<SearchInput value="" onChange={mockOnChange} />);

      expect(screen.getByPlaceholderText('Søg...')).toBeInTheDocument();
    });

    it('uses custom placeholder', () => {
      render(
        <SearchInput value="" onChange={mockOnChange} placeholder="Find noget..." />
      );

      expect(screen.getByPlaceholderText('Find noget...')).toBeInTheDocument();
    });

    it('displays current value', () => {
      render(<SearchInput value="test query" onChange={mockOnChange} />);

      expect(screen.getByRole('textbox')).toHaveValue('test query');
    });

    it('renders search icon', () => {
      const { container } = render(<SearchInput value="" onChange={mockOnChange} />);

      const svg = container.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });
  });

  describe('input handling', () => {
    it('calls onChange when typing', () => {
      render(<SearchInput value="" onChange={mockOnChange} />);

      fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hello' } });

      expect(mockOnChange).toHaveBeenCalledWith('hello');
    });

    it('calls onChange on each keystroke', () => {
      render(<SearchInput value="hel" onChange={mockOnChange} />);

      fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hell' } });

      expect(mockOnChange).toHaveBeenCalledWith('hell');
    });
  });

  describe('clear button', () => {
    it('shows clear button when value is not empty', () => {
      render(<SearchInput value="something" onChange={mockOnChange} />);

      expect(screen.getByRole('button', { name: 'Ryd søgning' })).toBeInTheDocument();
    });

    it('does not show clear button when value is empty', () => {
      render(<SearchInput value="" onChange={mockOnChange} />);

      expect(screen.queryByRole('button', { name: 'Ryd søgning' })).not.toBeInTheDocument();
    });

    it('clears input when clear button clicked', () => {
      render(<SearchInput value="something" onChange={mockOnChange} />);

      fireEvent.click(screen.getByRole('button', { name: 'Ryd søgning' }));

      expect(mockOnChange).toHaveBeenCalledWith('');
    });

    it('does not show clear button when disabled even with value', () => {
      render(<SearchInput value="something" onChange={mockOnChange} disabled={true} />);

      expect(screen.queryByRole('button', { name: 'Ryd søgning' })).not.toBeInTheDocument();
    });
  });

  describe('disabled state', () => {
    it('disables input when disabled prop is true', () => {
      render(<SearchInput value="" onChange={mockOnChange} disabled={true} />);

      expect(screen.getByRole('textbox')).toBeDisabled();
    });

    it('is not disabled by default', () => {
      render(<SearchInput value="" onChange={mockOnChange} />);

      expect(screen.getByRole('textbox')).not.toBeDisabled();
    });

    it('applies disabled styling', () => {
      render(<SearchInput value="" onChange={mockOnChange} disabled={true} />);

      const input = screen.getByRole('textbox');
      expect(input).toHaveClass('opacity-60', 'cursor-not-allowed');
    });
  });

  describe('custom className', () => {
    it('applies additional className to container', () => {
      const { container } = render(
        <SearchInput value="" onChange={mockOnChange} className="custom-class" />
      );

      expect(container.firstChild).toHaveClass('custom-class');
    });
  });

  describe('styling', () => {
    it('uses box-shadow on focus (matches source selection ring)', () => {
      render(<SearchInput value="" onChange={mockOnChange} />);

      const input = screen.getByRole('textbox');
      // box-shadow matches source-item-selected exactly: 0 0 0 3px #007AFF
      expect(input.className).toContain('focus:shadow-[0_0_0_3px_#007AFF]');
      expect(input.className).toContain('focus:border-transparent');
    });

    it('uses bordered style with apple-gray-300 border', () => {
      render(<SearchInput value="" onChange={mockOnChange} />);

      const input = screen.getByRole('textbox');
      expect(input.className).toContain('border-apple-gray-300');
    });

    it('uses white background in light mode', () => {
      render(<SearchInput value="" onChange={mockOnChange} />);

      const input = screen.getByRole('textbox');
      expect(input.className).toContain('bg-white');
    });
  });
});
