/**
 * Tests for AllLawsBadge component.
 *
 * TDD: Tests written BEFORE implementation.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AllLawsBadge } from '../AllLawsBadge';

describe('AllLawsBadge', () => {
  describe('rendering', () => {
    it('renders "Alle X love valgt" with correct count', () => {
      render(<AllLawsBadge count={5} />);

      expect(screen.getByText(/alle 5 love valgt/i)).toBeInTheDocument();
    });

    it('renders with different count', () => {
      render(<AllLawsBadge count={10} />);

      expect(screen.getByText(/alle 10 love valgt/i)).toBeInTheDocument();
    });

    it('renders checkmark icon', () => {
      render(<AllLawsBadge count={5} />);

      expect(screen.getByTestId('checkmark-icon')).toBeInTheDocument();
    });

    it('handles singular form for count=1', () => {
      render(<AllLawsBadge count={1} />);

      expect(screen.getByText(/alle 1 love valgt/i)).toBeInTheDocument();
    });
  });
});
