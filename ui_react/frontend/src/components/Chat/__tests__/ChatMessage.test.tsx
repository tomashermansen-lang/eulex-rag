/**
 * Tests for ChatMessage component.
 *
 * TDD: Tests written to cover message rendering and interactions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { ChatMessage } from '../ChatMessage';
import type { ChatMessage as ChatMessageType } from '../../../types';

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
      <div {...props}>{children}</div>
    ),
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock CitationLink
vi.mock('../../Common/CitationLink', () => ({
  renderWithCitations: (content: string, _messageId: string) => (
    <div data-testid="citation-content">{content}</div>
  ),
}));

// Mock SuggestedQuestions
vi.mock('../../Common/SuggestedQuestions', () => ({
  SuggestedQuestions: ({
    questions,
    onQuestionClick,
  }: {
    questions: string[];
    onQuestionClick: (q: string) => void;
  }) => (
    <div data-testid="suggested-questions">
      {questions.map((q, i) => (
        <button key={i} onClick={() => onQuestionClick(q)}>
          {q}
        </button>
      ))}
    </div>
  ),
}));

// Import the global clipboard mock from test setup
import { clipboardMock } from '../../../test/setup';

describe('ChatMessage', () => {
  const mockOnSuggestedQuestionClick = vi.fn();

  beforeEach(() => {
    // Clear only local mocks, not the global clipboard mock
    mockOnSuggestedQuestionClick.mockClear();
    // Reset clipboard mock calls but keep implementation
    clipboardMock.writeText.mockClear();
    clipboardMock.writeText.mockResolvedValue(undefined);
  });

  describe('user messages', () => {
    const userMessage: ChatMessageType = {
      id: 'user-1',
      role: 'user',
      content: 'What is GDPR?',
      timestamp: new Date(),
    };

    it('renders user message content', () => {
      render(<ChatMessage message={userMessage} />);

      expect(screen.getByText('What is GDPR?')).toBeInTheDocument();
    });

    it('applies user styling (right-aligned)', () => {
      render(<ChatMessage message={userMessage} />);

      const container = screen.getByText('What is GDPR?').closest('[data-message-id]');
      expect(container).toHaveClass('justify-end');
    });

    it('does not show response time for user messages', () => {
      render(<ChatMessage message={userMessage} />);

      expect(screen.queryByText(/sekunder/)).not.toBeInTheDocument();
    });
  });

  describe('assistant messages', () => {
    const assistantMessage: ChatMessageType = {
      id: 'assistant-1',
      role: 'assistant',
      content: 'GDPR is a regulation about data protection.',
      timestamp: new Date(),
      responseTime: 1.5,
      references: [
        { idx: 1, display: 'Article 1', chunk_text: 'Test' },
      ],
    };

    it('renders assistant message content', () => {
      render(<ChatMessage message={assistantMessage} />);

      expect(screen.getByTestId('citation-content')).toHaveTextContent(
        'GDPR is a regulation about data protection.'
      );
    });

    it('applies assistant styling (left-aligned)', () => {
      render(<ChatMessage message={assistantMessage} />);

      const container = screen.getByTestId('citation-content').closest('[data-message-id]');
      expect(container).toHaveClass('justify-start');
    });

    it('shows response time', () => {
      render(<ChatMessage message={assistantMessage} />);

      expect(screen.getByText(/1\.5 sekunder/)).toBeInTheDocument();
    });

    it('shows copy button', () => {
      render(<ChatMessage message={assistantMessage} />);

      expect(screen.getByTitle(/Kopiér svar/)).toBeInTheDocument();
    });
  });

  describe('streaming state', () => {
    const streamingMessage: ChatMessageType = {
      id: 'assistant-streaming',
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
    };

    it('shows typing indicator when streaming with no content', () => {
      render(<ChatMessage message={streamingMessage} />);

      expect(document.querySelector('.typing-indicator')).toBeInTheDocument();
    });

    it('shows content when streaming with content', () => {
      const messageWithContent: ChatMessageType = {
        ...streamingMessage,
        content: 'Partial response...',
      };

      render(<ChatMessage message={messageWithContent} />);

      expect(screen.getByTestId('citation-content')).toHaveTextContent('Partial response...');
    });

    it('does not show response time while streaming', () => {
      render(<ChatMessage message={streamingMessage} />);

      expect(screen.queryByText(/sekunder/)).not.toBeInTheDocument();
    });

    it('does not show copy button while streaming', () => {
      render(<ChatMessage message={streamingMessage} />);

      expect(screen.queryByTitle(/Kopiér/)).not.toBeInTheDocument();
    });
  });

  describe('copy functionality', () => {
    const message: ChatMessageType = {
      id: 'assistant-1',
      role: 'assistant',
      content: 'Test content to copy',
      timestamp: new Date(),
      responseTime: 1.0,
    };

    it('copies content to clipboard when copy button clicked', async () => {
      // Create a fresh spy for this test
      const writeTextSpy = vi.spyOn(navigator.clipboard, 'writeText');

      render(<ChatMessage message={message} />);

      const copyButton = screen.getByTitle(/Kopiér svar/);
      fireEvent.click(copyButton);

      await waitFor(() => {
        expect(writeTextSpy).toHaveBeenCalledWith('Test content to copy');
      });
    });

    it('shows success message after copying', async () => {
      const user = userEvent.setup();
      render(<ChatMessage message={message} />);

      const copyButton = screen.getByTitle(/Kopiér svar/);
      await user.click(copyButton);

      await waitFor(() => {
        expect(screen.getByText('Kopieret')).toBeInTheDocument();
      });
    });

    it('copies JSON in debug mode', async () => {
      // Create a fresh spy for this test
      const writeTextSpy = vi.spyOn(navigator.clipboard, 'writeText');

      render(
        <ChatMessage
          message={message}
          debugMode={true}
          retrievalMetrics={{ chunks_retrieved: 5 }}
        />
      );

      const copyButton = screen.getByTitle(/Kopiér svar med metadata/);
      fireEvent.click(copyButton);

      await waitFor(() => {
        expect(writeTextSpy).toHaveBeenCalledWith(
          expect.stringContaining('"answer"')
        );
      });
    });
  });

  describe('debug mode', () => {
    const message: ChatMessageType = {
      id: 'assistant-1',
      role: 'assistant',
      content: 'Test answer',
      timestamp: new Date(),
      responseTime: 1.0,
    };

    const metrics = { chunks_retrieved: 5, reranking_time: 0.2 };

    it('shows metadata toggle button in debug mode', () => {
      render(
        <ChatMessage
          message={message}
          debugMode={true}
          retrievalMetrics={metrics}
        />
      );

      expect(screen.getByText('Vis metadata')).toBeInTheDocument();
    });

    it('does not show metadata toggle when debug mode is off', () => {
      render(
        <ChatMessage
          message={message}
          debugMode={false}
          retrievalMetrics={metrics}
        />
      );

      expect(screen.queryByText('Vis metadata')).not.toBeInTheDocument();
    });

    it('toggles metadata panel visibility', async () => {
      const user = userEvent.setup();
      render(
        <ChatMessage
          message={message}
          debugMode={true}
          retrievalMetrics={metrics}
        />
      );

      // Initially hidden
      expect(screen.queryByText('Retrieval Metadata')).not.toBeInTheDocument();

      // Show metadata
      await user.click(screen.getByText('Vis metadata'));
      expect(screen.getByText('Retrieval Metadata')).toBeInTheDocument();

      // Hide metadata
      await user.click(screen.getByText('Skjul metadata'));
      // AnimatePresence would normally handle this, but our mock immediately removes it
    });

    it('displays metrics as JSON', async () => {
      const user = userEvent.setup();
      render(
        <ChatMessage
          message={message}
          debugMode={true}
          retrievalMetrics={metrics}
        />
      );

      await user.click(screen.getByText('Vis metadata'));

      expect(screen.getByText(/chunks_retrieved/)).toBeInTheDocument();
    });
  });

  describe('suggested questions', () => {
    const messageWithSuggestions: ChatMessageType = {
      id: 'assistant-1',
      role: 'assistant',
      content: 'Here is the answer.',
      timestamp: new Date(),
      responseTime: 1.0,
      suggestedQuestions: ['Follow-up 1?', 'Follow-up 2?'],
    };

    it('renders suggested questions when provided', () => {
      render(
        <ChatMessage
          message={messageWithSuggestions}
          onSuggestedQuestionClick={mockOnSuggestedQuestionClick}
        />
      );

      expect(screen.getByTestId('suggested-questions')).toBeInTheDocument();
      expect(screen.getByText('Follow-up 1?')).toBeInTheDocument();
      expect(screen.getByText('Follow-up 2?')).toBeInTheDocument();
    });

    it('calls callback when suggested question is clicked', async () => {
      const user = userEvent.setup();
      render(
        <ChatMessage
          message={messageWithSuggestions}
          onSuggestedQuestionClick={mockOnSuggestedQuestionClick}
        />
      );

      await user.click(screen.getByText('Follow-up 1?'));

      expect(mockOnSuggestedQuestionClick).toHaveBeenCalledWith('Follow-up 1?');
    });

    it('does not render suggested questions for user messages', () => {
      const userMessage: ChatMessageType = {
        id: 'user-1',
        role: 'user',
        content: 'Question',
        timestamp: new Date(),
        suggestedQuestions: ['Should not appear'],
      };

      render(
        <ChatMessage
          message={userMessage}
          onSuggestedQuestionClick={mockOnSuggestedQuestionClick}
        />
      );

      expect(screen.queryByTestId('suggested-questions')).not.toBeInTheDocument();
    });

    it('does not render when no callback provided', () => {
      render(<ChatMessage message={messageWithSuggestions} />);

      expect(screen.queryByTestId('suggested-questions')).not.toBeInTheDocument();
    });
  });
});
