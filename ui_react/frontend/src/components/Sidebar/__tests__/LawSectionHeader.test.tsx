/**
 * Tests for LawSectionHeader component.
 *
 * TDD: Tests written BEFORE implementation.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LawSectionHeader } from '../LawSectionHeader';

describe('LawSectionHeader', () => {
  describe('rendering', () => {
    it('renders label text', () => {
      render(<LawSectionHeader label="Valgte" count={2} />);

      expect(screen.getByText('Valgte')).toBeInTheDocument();
    });

    it('renders count in parentheses', () => {
      render(<LawSectionHeader label="Valgte" count={3} />);

      expect(screen.getByText('(3)')).toBeInTheDocument();
    });

    it('renders with different label and count', () => {
      render(<LawSectionHeader label="Tilgængelige" count={5} />);

      expect(screen.getByText('Tilgængelige')).toBeInTheDocument();
      expect(screen.getByText('(5)')).toBeInTheDocument();
    });

    it('has uppercase styling class', () => {
      render(<LawSectionHeader label="Valgte" count={2} />);

      const header = screen.getByText('Valgte').closest('div');
      expect(header).toHaveClass('uppercase');
    });
  });
});
