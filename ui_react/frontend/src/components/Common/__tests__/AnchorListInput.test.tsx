/**
 * Tests for AnchorListInput component.
 *
 * Reusable component for managing anchor lists (article:5, annex:iii, etc.)
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AnchorListInput } from '../AnchorListInput';

// Mock the API module
vi.mock('../../../services/api', () => ({
  listAnchors: vi.fn(),
}));

describe('AnchorListInput', () => {
  const defaultProps = {
    label: 'Test Label',
    values: [],
    onChange: vi.fn(),
  };

  it('renders label', () => {
    render(<AnchorListInput {...defaultProps} />);
    expect(screen.getByText('Test Label')).toBeInTheDocument();
  });

  it('renders description when provided', () => {
    render(
      <AnchorListInput
        {...defaultProps}
        description="Test description"
      />
    );
    expect(screen.getByText('Test description')).toBeInTheDocument();
  });

  it('renders existing values as tags', () => {
    render(
      <AnchorListInput
        {...defaultProps}
        values={['article:5', 'annex:iii']}
      />
    );
    expect(screen.getByText('article:5')).toBeInTheDocument();
    expect(screen.getByText('annex:iii')).toBeInTheDocument();
  });

  it('adds valid anchor when clicking add button', () => {
    const onChange = vi.fn();
    render(
      <AnchorListInput
        {...defaultProps}
        onChange={onChange}
      />
    );

    const input = screen.getByPlaceholderText('article:5');
    fireEvent.change(input, { target: { value: 'article:10' } });
    fireEvent.click(screen.getByText('Tilføj'));

    expect(onChange).toHaveBeenCalledWith(['article:10']);
  });

  it('adds valid anchor when pressing Enter', () => {
    const onChange = vi.fn();
    render(
      <AnchorListInput
        {...defaultProps}
        onChange={onChange}
      />
    );

    const input = screen.getByPlaceholderText('article:5');
    fireEvent.change(input, { target: { value: 'section:2' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(onChange).toHaveBeenCalledWith(['section:2']);
  });

  it('shows error for invalid anchor format', () => {
    render(<AnchorListInput {...defaultProps} />);

    const input = screen.getByPlaceholderText('article:5');
    fireEvent.change(input, { target: { value: 'invalid-format' } });
    fireEvent.click(screen.getByText('Tilføj'));

    expect(screen.getByText(/Format:/)).toBeInTheDocument();
  });

  it('shows error for duplicate anchor', () => {
    render(
      <AnchorListInput
        {...defaultProps}
        values={['article:5']}
      />
    );

    const input = screen.getByPlaceholderText('article:5');
    fireEvent.change(input, { target: { value: 'article:5' } });
    fireEvent.click(screen.getByText('Tilføj'));

    expect(screen.getByText('Allerede tilføjet')).toBeInTheDocument();
  });

  it('removes anchor when clicking remove button', () => {
    const onChange = vi.fn();
    render(
      <AnchorListInput
        {...defaultProps}
        values={['article:5', 'annex:iii']}
        onChange={onChange}
      />
    );

    // Find the remove button for the first tag
    const removeButtons = screen.getAllByRole('button', { name: '' });
    // First button is the remove button for 'article:5'
    fireEvent.click(removeButtons[0]);

    expect(onChange).toHaveBeenCalledWith(['annex:iii']);
  });

  it('clears input after successful add', () => {
    render(<AnchorListInput {...defaultProps} />);

    const input = screen.getByPlaceholderText('article:5') as HTMLInputElement;
    fireEvent.change(input, { target: { value: 'article:10' } });
    fireEvent.click(screen.getByText('Tilføj'));

    expect(input.value).toBe('');
  });

  it('uses custom placeholder when provided', () => {
    render(
      <AnchorListInput
        {...defaultProps}
        placeholder="recital:1"
      />
    );
    expect(screen.getByPlaceholderText('recital:1')).toBeInTheDocument();
  });

  it('normalizes anchor to lowercase', () => {
    const onChange = vi.fn();
    render(
      <AnchorListInput
        {...defaultProps}
        onChange={onChange}
      />
    );

    const input = screen.getByPlaceholderText('article:5');
    fireEvent.change(input, { target: { value: 'ARTICLE:10' } });
    fireEvent.click(screen.getByText('Tilføj'));

    expect(onChange).toHaveBeenCalledWith(['article:10']);
  });

  it('validates various anchor types', () => {
    const onChange = vi.fn();
    const validAnchors = [
      'article:5',
      'section:2a',
      'annex:iii',
      'recital:10',
      'paragraph:3',
      'chapter:iv',
    ];

    validAnchors.forEach((anchor) => {
      onChange.mockClear();
      const { unmount } = render(
        <AnchorListInput
          {...defaultProps}
          onChange={onChange}
        />
      );

      const input = screen.getByPlaceholderText('article:5');
      fireEvent.change(input, { target: { value: anchor } });
      fireEvent.click(screen.getByText('Tilføj'));

      expect(onChange).toHaveBeenCalledWith([anchor.toLowerCase()]);
      unmount();
    });
  });

  it('does not add empty value', () => {
    const onChange = vi.fn();
    render(
      <AnchorListInput
        {...defaultProps}
        onChange={onChange}
      />
    );

    fireEvent.click(screen.getByText('Tilføj'));

    expect(onChange).not.toHaveBeenCalled();
  });

  it('shows error message when trying to add empty value', () => {
    render(<AnchorListInput {...defaultProps} />);

    fireEvent.click(screen.getByText('Tilføj'));

    expect(screen.getByText(/Indtast en anchor/i)).toBeInTheDocument();
  });

  it('fetches suggestions when law prop is provided and user types', async () => {
    const { listAnchors } = await import('../../../services/api');
    vi.mocked(listAnchors).mockResolvedValue({ anchors: ['article:1', 'article:2'], total: 2 });

    render(
      <AnchorListInput
        {...defaultProps}
        law="ai-act"
      />
    );

    const input = screen.getByPlaceholderText('article:5');
    fireEvent.change(input, { target: { value: 'art' } });

    // Wait for debounce
    await waitFor(() => {
      expect(listAnchors).toHaveBeenCalledWith('ai-act', 'art');
    }, { timeout: 500 });
  });

  it('shows suggestions dropdown when results available', async () => {
    const { listAnchors } = await import('../../../services/api');
    vi.mocked(listAnchors).mockResolvedValue({ anchors: ['article:5', 'article:6'], total: 2 });

    render(
      <AnchorListInput
        {...defaultProps}
        law="ai-act"
      />
    );

    const input = screen.getByPlaceholderText('article:5');
    fireEvent.change(input, { target: { value: 'art' } });

    await waitFor(() => {
      expect(screen.getByText('article:5')).toBeInTheDocument();
      expect(screen.getByText('article:6')).toBeInTheDocument();
    });
  });

  it('adds anchor when suggestion clicked', async () => {
    const onChange = vi.fn();
    const { listAnchors } = await import('../../../services/api');
    vi.mocked(listAnchors).mockResolvedValue({ anchors: ['article:5'], total: 1 });

    render(
      <AnchorListInput
        {...defaultProps}
        law="ai-act"
        onChange={onChange}
      />
    );

    const input = screen.getByPlaceholderText('article:5');
    fireEvent.change(input, { target: { value: 'art' } });

    await waitFor(() => {
      expect(screen.getByText('article:5')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('article:5'));

    expect(onChange).toHaveBeenCalledWith(['article:5']);
  });
});
