/**
 * Tests for Dropdown component.
 *
 * TDD: Test dropdown rendering, interaction, and accessibility.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Dropdown } from '../Dropdown';

const mockItems = [
  { value: 'option1', label: 'Option 1', description: 'First option' },
  { value: 'option2', label: 'Option 2', description: 'Second option' },
  { value: 'option3', label: 'Option 3' },
];

describe('Dropdown', () => {
  describe('rendering', () => {
    it('renders trigger element', () => {
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      expect(screen.getByRole('button', { name: 'Select' })).toBeInTheDocument();
    });

    it('does not render menu when closed', () => {
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
    });

    it('renders items when open', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));

      expect(screen.getByRole('listbox')).toBeInTheDocument();
      expect(screen.getByText('Option 1')).toBeInTheDocument();
      expect(screen.getByText('Option 2')).toBeInTheDocument();
      expect(screen.getByText('Option 3')).toBeInTheDocument();
    });

    it('renders item descriptions when provided', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));

      expect(screen.getByText('First option')).toBeInTheDocument();
      expect(screen.getByText('Second option')).toBeInTheDocument();
    });
  });

  describe('interaction', () => {
    it('clicking trigger opens menu', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      expect(screen.queryByRole('listbox')).not.toBeInTheDocument();

      await user.click(screen.getByRole('button', { name: 'Select' }));

      expect(screen.getByRole('listbox')).toBeInTheDocument();
    });

    it('clicking item calls onChange with value', async () => {
      const onChange = vi.fn();
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={onChange}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));
      await user.click(screen.getByText('Option 2'));

      expect(onChange).toHaveBeenCalledWith('option2');
    });

    it('clicking item closes menu', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));
      expect(screen.getByRole('listbox')).toBeInTheDocument();

      await user.click(screen.getByText('Option 2'));

      // Wait for exit animation to complete
      await waitFor(() => {
        expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
      });
    });
  });

  describe('selection', () => {
    it('selected item shows checkmark', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option2"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));

      // Find the selected item and check for checkmark
      const options = screen.getAllByRole('option');
      const selectedOption = options.find((opt) =>
        opt.textContent?.includes('Option 2')
      );
      expect(selectedOption).toHaveAttribute('aria-selected', 'true');
      expect(selectedOption?.querySelector('[data-checkmark]')).toBeInTheDocument();
    });

    it('non-selected items do not show checkmark', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option2"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));

      const options = screen.getAllByRole('option');
      const nonSelectedOption = options.find((opt) =>
        opt.textContent?.includes('Option 1')
      );
      expect(nonSelectedOption).toHaveAttribute('aria-selected', 'false');
      expect(
        nonSelectedOption?.querySelector('[data-checkmark]')
      ).not.toBeInTheDocument();
    });
  });

  describe('keyboard', () => {
    it('ESC closes menu', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));
      expect(screen.getByRole('listbox')).toBeInTheDocument();

      await user.keyboard('{Escape}');

      // Wait for exit animation to complete
      await waitFor(() => {
        expect(screen.queryByRole('listbox')).not.toBeInTheDocument();
      });
    });

    it('Enter selects focused item', async () => {
      const onChange = vi.fn();
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={onChange}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));

      // Navigate down twice and select
      await user.keyboard('{ArrowDown}{ArrowDown}{Enter}');

      expect(onChange).toHaveBeenCalledWith('option2');
    });

    it('ArrowDown navigates through items', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));

      // Navigate down
      await user.keyboard('{ArrowDown}');

      const options = screen.getAllByRole('option');
      // First item should have visual focus indicator
      expect(options[0]).toHaveAttribute('data-focused', 'true');
    });
  });

  describe('accessibility', () => {
    it('has correct ARIA attributes on trigger', async () => {
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      const trigger = screen.getByRole('button', { name: 'Select' });
      expect(trigger).toHaveAttribute('aria-haspopup', 'listbox');
      expect(trigger).toHaveAttribute('aria-expanded', 'false');
    });

    it('aria-expanded is true when open', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));

      const trigger = screen.getByRole('button', { name: 'Select' });
      expect(trigger).toHaveAttribute('aria-expanded', 'true');
    });

    it('menu has role listbox', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));

      expect(screen.getByRole('listbox')).toBeInTheDocument();
    });

    it('items have role option', async () => {
      const user = userEvent.setup();
      render(
        <Dropdown
          items={mockItems}
          value="option1"
          onChange={vi.fn()}
          trigger={<button>Select</button>}
        />
      );

      await user.click(screen.getByRole('button', { name: 'Select' }));

      const options = screen.getAllByRole('option');
      expect(options).toHaveLength(3);
    });
  });
});
