/**
 * Tests for CollapsibleSection component.
 *
 * TDD: Test collapsible section rendering, interaction, and accessibility.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CollapsibleSection } from '../CollapsibleSection';

describe('CollapsibleSection', () => {
  describe('rendering', () => {
    it('renders title', () => {
      render(
        <CollapsibleSection title="Test Section">
          <p>Content</p>
        </CollapsibleSection>
      );

      expect(screen.getByText('Test Section')).toBeInTheDocument();
    });

    it('renders count badge when provided', () => {
      render(
        <CollapsibleSection title="Test Section" count={5}>
          <p>Content</p>
        </CollapsibleSection>
      );

      expect(screen.getByText('5')).toBeInTheDocument();
    });

    it('does not render count badge when count is 0', () => {
      render(
        <CollapsibleSection title="Test Section" count={0}>
          <p>Content</p>
        </CollapsibleSection>
      );

      // Count badge should not be present
      expect(screen.queryByText('0')).not.toBeInTheDocument();
    });
  });

  describe('default state', () => {
    it('starts collapsed by default', () => {
      render(
        <CollapsibleSection title="Test Section">
          <p>Content</p>
        </CollapsibleSection>
      );

      // Content should not be visible when collapsed
      expect(screen.queryByText('Content')).not.toBeInTheDocument();
    });

    it('starts open when defaultOpen=true', () => {
      render(
        <CollapsibleSection title="Test Section" defaultOpen>
          <p>Content</p>
        </CollapsibleSection>
      );

      expect(screen.getByText('Content')).toBeInTheDocument();
    });
  });

  describe('interaction', () => {
    it('clicking header toggles content visibility', async () => {
      const user = userEvent.setup();
      render(
        <CollapsibleSection title="Test Section">
          <p>Content</p>
        </CollapsibleSection>
      );

      // Initially collapsed
      expect(screen.queryByText('Content')).not.toBeInTheDocument();

      // Click to expand
      await user.click(screen.getByRole('button'));
      expect(screen.getByText('Content')).toBeInTheDocument();

      // Click to collapse
      await user.click(screen.getByRole('button'));
      await waitFor(() => {
        expect(screen.queryByText('Content')).not.toBeInTheDocument();
      });
    });

    it('calls onToggle callback when toggled', async () => {
      const onToggle = vi.fn();
      const user = userEvent.setup();
      render(
        <CollapsibleSection title="Test Section" onToggle={onToggle}>
          <p>Content</p>
        </CollapsibleSection>
      );

      await user.click(screen.getByRole('button'));
      expect(onToggle).toHaveBeenCalledWith(true);

      await user.click(screen.getByRole('button'));
      expect(onToggle).toHaveBeenCalledWith(false);
    });
  });

  describe('accessibility', () => {
    it('has aria-expanded attribute', () => {
      render(
        <CollapsibleSection title="Test Section">
          <p>Content</p>
        </CollapsibleSection>
      );

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-expanded', 'false');
    });

    it('aria-expanded is true when open', async () => {
      const user = userEvent.setup();
      render(
        <CollapsibleSection title="Test Section">
          <p>Content</p>
        </CollapsibleSection>
      );

      await user.click(screen.getByRole('button'));

      const button = screen.getByRole('button');
      expect(button).toHaveAttribute('aria-expanded', 'true');
    });

    it('has aria-controls linking to content region', () => {
      render(
        <CollapsibleSection title="Test Section" defaultOpen>
          <p>Content</p>
        </CollapsibleSection>
      );

      const button = screen.getByRole('button');
      const controlsId = button.getAttribute('aria-controls');
      expect(controlsId).toBeTruthy();

      // Find the region with that ID
      const region = document.getElementById(controlsId!);
      expect(region).toBeInTheDocument();
    });
  });

  describe('chevron indicator', () => {
    it('shows chevron icon', () => {
      render(
        <CollapsibleSection title="Test Section">
          <p>Content</p>
        </CollapsibleSection>
      );

      // Check for SVG chevron
      const button = screen.getByRole('button');
      const svg = button.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });
  });
});
