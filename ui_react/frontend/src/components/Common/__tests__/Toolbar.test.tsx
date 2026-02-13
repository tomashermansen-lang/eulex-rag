/**
 * Tests for Toolbar component.
 *
 * TDD: Test toolbar rendering and layout.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Toolbar } from '../Toolbar';

describe('Toolbar', () => {
  const defaultProps = {
    appTitle: 'EuLex',
  };

  describe('rendering', () => {
    it('renders app title', () => {
      render(<Toolbar {...defaultProps} />);

      expect(screen.getByText('EuLex')).toBeInTheDocument();
    });

    it('renders actions when provided', () => {
      render(
        <Toolbar
          {...defaultProps}
          actions={<button>Custom Action</button>}
        />
      );

      expect(screen.getByRole('button', { name: /custom action/i })).toBeInTheDocument();
    });

    it('renders multiple actions', () => {
      render(
        <Toolbar
          {...defaultProps}
          actions={
            <>
              <button>Export</button>
              <button>Menu</button>
            </>
          }
        />
      );

      expect(screen.getByRole('button', { name: /export/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /menu/i })).toBeInTheDocument();
    });

    it('renders empty trailing section when no actions', () => {
      const { container } = render(<Toolbar {...defaultProps} />);

      const trailing = container.querySelector('[data-section="trailing"]');
      expect(trailing).toBeInTheDocument();
      expect(trailing?.children.length).toBe(0);
    });
  });

  describe('layout', () => {
    it('has correct structure with leading, center, and trailing sections', () => {
      const { container } = render(<Toolbar {...defaultProps} />);

      // Check for toolbar role
      expect(screen.getByRole('toolbar')).toBeInTheDocument();

      // Check sections exist
      expect(container.querySelector('[data-section="leading"]')).toBeInTheDocument();
      expect(container.querySelector('[data-section="center"]')).toBeInTheDocument();
      expect(container.querySelector('[data-section="trailing"]')).toBeInTheDocument();
    });

    it('has accessible label', () => {
      render(<Toolbar {...defaultProps} />);

      expect(screen.getByRole('toolbar')).toHaveAttribute('aria-label', 'Application toolbar');
    });
  });
});
