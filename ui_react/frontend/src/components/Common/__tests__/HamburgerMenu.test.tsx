/**
 * Tests for HamburgerMenu component.
 *
 * TDD: Tests written to cover menu interactions and settings toggles.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { HamburgerMenu } from '../HamburgerMenu';
import type { Settings } from '../../../types';

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
      <div {...props}>{children}</div>
    ),
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe('HamburgerMenu', () => {
  const defaultSettings: Settings = {
    law: 'ai-act',
    userProfile: 'LEGAL',
    debugMode: false,
    darkMode: false,
  };

  const mockOnToggleDebug = vi.fn();
  const mockOnToggleDarkMode = vi.fn();
  const mockOnNavigateToAdmin = vi.fn();

  const defaultProps = {
    settings: defaultSettings,
    onToggleDebug: mockOnToggleDebug,
    onToggleDarkMode: mockOnToggleDarkMode,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders hamburger button', () => {
      render(<HamburgerMenu {...defaultProps} />);

      expect(screen.getByRole('button', { name: 'Indstillinger' })).toBeInTheDocument();
    });

    it('button has correct aria-expanded when closed', () => {
      render(<HamburgerMenu {...defaultProps} />);

      expect(screen.getByRole('button', { name: 'Indstillinger' })).toHaveAttribute(
        'aria-expanded',
        'false'
      );
    });

    it('does not show menu initially', () => {
      render(<HamburgerMenu {...defaultProps} />);

      expect(screen.queryByText('Mørk tilstand')).not.toBeInTheDocument();
    });
  });

  describe('menu toggle', () => {
    it('shows menu when button clicked', () => {
      render(<HamburgerMenu {...defaultProps} />);

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      expect(screen.getByText('Indstillinger')).toBeInTheDocument();
      expect(screen.getByText('Mørk tilstand')).toBeInTheDocument();
      expect(screen.getByText('Fejlsøgning')).toBeInTheDocument();
    });

    it('updates aria-expanded when open', () => {
      render(<HamburgerMenu {...defaultProps} />);

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      expect(screen.getByRole('button', { name: 'Indstillinger' })).toHaveAttribute(
        'aria-expanded',
        'true'
      );
    });

    it('closes menu when button clicked again', () => {
      render(<HamburgerMenu {...defaultProps} />);

      const button = screen.getByRole('button', { name: 'Indstillinger' });

      fireEvent.click(button);
      expect(screen.getByText('Mørk tilstand')).toBeInTheDocument();

      fireEvent.click(button);
      expect(screen.queryByText('Mørk tilstand')).not.toBeInTheDocument();
    });

    it('closes menu when clicking outside', () => {
      render(
        <div>
          <div data-testid="outside">Outside</div>
          <HamburgerMenu {...defaultProps} />
        </div>
      );

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));
      expect(screen.getByText('Mørk tilstand')).toBeInTheDocument();

      fireEvent.mouseDown(screen.getByTestId('outside'));
      expect(screen.queryByText('Mørk tilstand')).not.toBeInTheDocument();
    });
  });

  describe('dark mode toggle', () => {
    it('calls onToggleDarkMode when toggle clicked', () => {
      render(<HamburgerMenu {...defaultProps} />);

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      const switches = screen.getAllByRole('switch');
      fireEvent.click(switches[0]); // Dark mode is first switch

      expect(mockOnToggleDarkMode).toHaveBeenCalled();
    });

    it('shows dark mode as off when darkMode is false', () => {
      render(<HamburgerMenu {...defaultProps} />);

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      const switches = screen.getAllByRole('switch');
      expect(switches[0]).toHaveAttribute('aria-checked', 'false');
    });

    it('shows dark mode as on when darkMode is true', () => {
      render(
        <HamburgerMenu
          {...defaultProps}
          settings={{ ...defaultSettings, darkMode: true }}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      const switches = screen.getAllByRole('switch');
      expect(switches[0]).toHaveAttribute('aria-checked', 'true');
    });
  });

  describe('debug mode toggle', () => {
    it('calls onToggleDebug when toggle clicked', () => {
      render(<HamburgerMenu {...defaultProps} />);

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      const switches = screen.getAllByRole('switch');
      fireEvent.click(switches[1]); // Debug toggle is second

      expect(mockOnToggleDebug).toHaveBeenCalled();
    });

    it('shows debug mode as off when debugMode is false', () => {
      render(<HamburgerMenu {...defaultProps} />);

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      const switches = screen.getAllByRole('switch');
      expect(switches[1]).toHaveAttribute('aria-checked', 'false');
    });

    it('shows debug mode as on when debugMode is true', () => {
      render(
        <HamburgerMenu
          {...defaultProps}
          settings={{ ...defaultSettings, debugMode: true }}
        />
      );

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      const switches = screen.getAllByRole('switch');
      expect(switches[1]).toHaveAttribute('aria-checked', 'true');
    });
  });

  describe('admin navigation', () => {
    it('does not show admin link when onNavigateToAdmin not provided', () => {
      render(<HamburgerMenu {...defaultProps} />);

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      expect(screen.queryByText('Administrer lovgivning')).not.toBeInTheDocument();
    });

    it('shows admin link when onNavigateToAdmin provided', () => {
      render(
        <HamburgerMenu {...defaultProps} onNavigateToAdmin={mockOnNavigateToAdmin} />
      );

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));

      expect(screen.getByText('Administrer lovgivning')).toBeInTheDocument();
    });

    it('calls onNavigateToAdmin and closes menu when clicked', () => {
      render(
        <HamburgerMenu {...defaultProps} onNavigateToAdmin={mockOnNavigateToAdmin} />
      );

      fireEvent.click(screen.getByRole('button', { name: 'Indstillinger' }));
      fireEvent.click(screen.getByText('Administrer lovgivning'));

      expect(mockOnNavigateToAdmin).toHaveBeenCalled();
      expect(screen.queryByText('Mørk tilstand')).not.toBeInTheDocument();
    });
  });
});
