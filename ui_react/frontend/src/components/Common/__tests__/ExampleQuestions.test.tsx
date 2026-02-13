/**
 * Tests for ExampleQuestions component.
 *
 * TDD: Tests written to cover question display and click handling.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExampleQuestions } from '../ExampleQuestions';

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

describe('ExampleQuestions', () => {
  const mockOnQuestionClick = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders nothing when questions array is empty', () => {
    const { container } = render(
      <ExampleQuestions questions={[]} onQuestionClick={mockOnQuestionClick} />
    );

    expect(container.innerHTML).toBe('');
  });

  it('renders header text', () => {
    render(
      <ExampleQuestions
        questions={['Question 1?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    expect(screen.getByText('Prøv et eksempel')).toBeInTheDocument();
  });

  it('renders all questions', () => {
    const questions = ['Question 1?', 'Question 2?', 'Question 3?'];

    render(
      <ExampleQuestions
        questions={questions}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    questions.forEach((q) => {
      expect(screen.getByText(q)).toBeInTheDocument();
    });
  });

  it('renders questions as buttons', () => {
    render(
      <ExampleQuestions
        questions={['Question 1?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    expect(screen.getByRole('button', { name: 'Question 1?' })).toBeInTheDocument();
  });

  it('calls onQuestionClick when a question is clicked', () => {
    render(
      <ExampleQuestions
        questions={['Question 1?', 'Question 2?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    fireEvent.click(screen.getByText('Question 2?'));

    expect(mockOnQuestionClick).toHaveBeenCalledWith('Question 2?');
    expect(mockOnQuestionClick).toHaveBeenCalledTimes(1);
  });

  it('renders buttons with suggestion-chip class', () => {
    render(
      <ExampleQuestions
        questions={['Question 1?']}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    const button = screen.getByRole('button');
    expect(button).toHaveClass('suggestion-chip');
  });

  it('handles many questions', () => {
    const questions = Array.from({ length: 10 }, (_, i) => `Question ${i + 1}?`);

    render(
      <ExampleQuestions
        questions={questions}
        onQuestionClick={mockOnQuestionClick}
      />
    );

    // All questions should be rendered
    questions.forEach((q) => {
      expect(screen.getByText(q)).toBeInTheDocument();
    });
  });
});
