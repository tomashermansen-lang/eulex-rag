/**
 * Tests for ChatInput component.
 *
 * TDD: Tests written to cover input handling and submission.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatInput } from '../ChatInput';

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
  motion: {
    button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
      <button {...props}>{children}</button>
    ),
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
      <div {...props}>{children}</div>
    ),
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe('ChatInput', () => {
  const mockOnSend = vi.fn();
  const mockOnStopStreaming = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders textarea with placeholder', () => {
      render(<ChatInput onSend={mockOnSend} />);

      expect(screen.getByPlaceholderText('Stil dit spørgsmål...')).toBeInTheDocument();
    });

    it('renders custom placeholder when provided', () => {
      render(<ChatInput onSend={mockOnSend} placeholder="Ask a question..." />);

      expect(screen.getByPlaceholderText('Ask a question...')).toBeInTheDocument();
    });

    it('renders Send button when not streaming', () => {
      render(<ChatInput onSend={mockOnSend} />);

      expect(screen.getByText('Send')).toBeInTheDocument();
    });

    it('renders Stop button when streaming', () => {
      render(
        <ChatInput
          onSend={mockOnSend}
          isStreaming={true}
          onStopStreaming={mockOnStopStreaming}
        />
      );

      expect(screen.getByText('Stop')).toBeInTheDocument();
      expect(screen.queryByText('Send')).not.toBeInTheDocument();
    });

    it('renders helper text', () => {
      render(<ChatInput onSend={mockOnSend} />);

      expect(screen.getByText(/Tryk Enter for at sende/)).toBeInTheDocument();
    });

    it('renders AI disclaimer with visible styling (CI-09)', () => {
      render(<ChatInput onSend={mockOnSend} />);

      const disclaimer = screen.getByText(/Ikke juridisk rådgivning/);
      expect(disclaimer).toBeInTheDocument();
      // Must NOT use nearly-invisible 10px text
      expect(disclaimer.className).not.toContain('text-[10px]');
      // Must use at minimum text-xs (12px) for readability
      expect(disclaimer.className).toContain('text-xs');
    });
  });

  describe('input handling', () => {
    it('updates value when typing', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      await user.type(textarea, 'Hello');

      expect(textarea).toHaveValue('Hello');
    });

    it('is disabled when disabled prop is true', () => {
      render(<ChatInput onSend={mockOnSend} disabled={true} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      expect(textarea).toBeDisabled();
    });
  });

  describe('submission', () => {
    it('calls onSend when form is submitted', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      await user.type(textarea, 'Test message');

      const sendButton = screen.getByText('Send');
      await user.click(sendButton);

      expect(mockOnSend).toHaveBeenCalledWith('Test message');
    });

    it('clears input after submission', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      await user.type(textarea, 'Test message');
      await user.click(screen.getByText('Send'));

      expect(textarea).toHaveValue('');
    });

    it('trims whitespace from message', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      await user.type(textarea, '  Test message  ');
      await user.click(screen.getByText('Send'));

      expect(mockOnSend).toHaveBeenCalledWith('Test message');
    });

    it('does not send empty messages', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      await user.type(textarea, '   ');
      await user.click(screen.getByText('Send'));

      expect(mockOnSend).not.toHaveBeenCalled();
    });

    it('does not send when disabled', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} disabled={true} />);

      // Can't type in disabled textarea, so we skip that
      // Just verify the button behavior
      const sendButton = screen.getByText('Send');
      expect(sendButton).toBeDisabled();
    });

    it('does not send when streaming', async () => {
      render(
        <ChatInput
          onSend={mockOnSend}
          isStreaming={true}
          onStopStreaming={mockOnStopStreaming}
        />
      );

      // Should show Stop button, not Send
      expect(screen.queryByText('Send')).not.toBeInTheDocument();
    });
  });

  describe('keyboard shortcuts', () => {
    it('submits on Enter key', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      await user.type(textarea, 'Test message');
      await user.keyboard('{Enter}');

      expect(mockOnSend).toHaveBeenCalledWith('Test message');
    });

    it('does not submit on Shift+Enter (allows newline)', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      await user.type(textarea, 'Line 1');
      await user.keyboard('{Shift>}{Enter}{/Shift}');
      await user.type(textarea, 'Line 2');

      expect(mockOnSend).not.toHaveBeenCalled();
      // Note: The actual newline behavior depends on the textarea, not our code
    });
  });

  describe('stop streaming', () => {
    it('calls onStopStreaming when Stop button clicked', async () => {
      const user = userEvent.setup();
      render(
        <ChatInput
          onSend={mockOnSend}
          isStreaming={true}
          onStopStreaming={mockOnStopStreaming}
        />
      );

      const stopButton = screen.getByText('Stop');
      await user.click(stopButton);

      expect(mockOnStopStreaming).toHaveBeenCalled();
    });
  });

  describe('Send button state', () => {
    it('is disabled when input is empty', () => {
      render(<ChatInput onSend={mockOnSend} />);

      const sendButton = screen.getByText('Send');
      expect(sendButton).toBeDisabled();
    });

    it('is enabled when input has content', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      await user.type(textarea, 'Test');

      const sendButton = screen.getByText('Send');
      expect(sendButton).not.toBeDisabled();
    });

    it('is disabled when only whitespace', async () => {
      const user = userEvent.setup();
      render(<ChatInput onSend={mockOnSend} />);

      const textarea = screen.getByPlaceholderText('Stil dit spørgsmål...');
      await user.type(textarea, '   ');

      const sendButton = screen.getByText('Send');
      expect(sendButton).toBeDisabled();
    });
  });
});
