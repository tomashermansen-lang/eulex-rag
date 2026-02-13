/**
 * Tests for SuggestedQuestions component.
 *
 * TDD: Tests written to cover suggested question display and interactions.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SuggestedQuestions } from '../SuggestedQuestions';

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
      <div {...props}>{children}</div>
    ),
    button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
      <button {...props}>{children}</button>
    ),
  },
}));

describe('SuggestedQuestions', () => {
  const mockOnQuestionClick = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders nothing when questions array is empty', () => {
    const { container } = render(
      <SuggestedQuestions questions={[]} onQuestionClick={mockOnQuestionClick} />
    );

    expect(container.innerHTML).toBe('');
  });

  it('renders instruction text', () => {
    render(
      <SuggestedQuestions
        questions={['Follow-up 1?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    expect(screen.getByText('Klik for at stille spørgsmålet:')).toBeInTheDocument();
  });

  it('renders questions as buttons', () => {
    render(
      <SuggestedQuestions
        questions={['Follow-up 1?', 'Follow-up 2?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    expect(screen.getByRole('button', { name: 'Follow-up 1?' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Follow-up 2?' })).toBeInTheDocument();
  });

  it('limits display to 3 questions maximum', () => {
    const questions = [
      'Question 1?',
      'Question 2?',
      'Question 3?',
      'Question 4?',
      'Question 5?',
    ];

    render(
      <SuggestedQuestions
        questions={questions}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    expect(screen.getByText('Question 1?')).toBeInTheDocument();
    expect(screen.getByText('Question 2?')).toBeInTheDocument();
    expect(screen.getByText('Question 3?')).toBeInTheDocument();
    expect(screen.queryByText('Question 4?')).not.toBeInTheDocument();
    expect(screen.queryByText('Question 5?')).not.toBeInTheDocument();
  });

  it('calls onQuestionClick when a question is clicked', () => {
    render(
      <SuggestedQuestions
        questions={['Follow-up 1?', 'Follow-up 2?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    fireEvent.click(screen.getByText('Follow-up 2?'));

    expect(mockOnQuestionClick).toHaveBeenCalledWith('Follow-up 2?');
    expect(mockOnQuestionClick).toHaveBeenCalledTimes(1);
  });

  it('renders buttons with suggestion-chip class', () => {
    render(
      <SuggestedQuestions
        questions={['Follow-up 1?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    const button = screen.getByRole('button');
    expect(button).toHaveClass('suggestion-chip');
  });

  it('renders with small text styling', () => {
    render(
      <SuggestedQuestions
        questions={['Follow-up 1?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    const button = screen.getByRole('button');
    expect(button).toHaveClass('text-sm');
  });

  it('handles exactly 3 questions without truncation', () => {
    const questions = ['Question 1?', 'Question 2?', 'Question 3?'];

    render(
      <SuggestedQuestions
        questions={questions}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    const buttons = screen.getAllByRole('button');
    expect(buttons).toHaveLength(3);
  });

  it('handles single question', () => {
    render(
      <SuggestedQuestions
        questions={['Only question?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    expect(screen.getByText('Only question?')).toBeInTheDocument();
    expect(screen.getAllByRole('button')).toHaveLength(1);
  });
});
