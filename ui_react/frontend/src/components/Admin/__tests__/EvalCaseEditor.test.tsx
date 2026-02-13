/**
 * Tests for EvalCaseEditor component.
 *
 * Modal for creating and editing eval cases.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { EvalCaseEditor } from '../EvalCaseEditor';
import type { EvalCase, ExpectedBehavior } from '../../../types';

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
      <div {...props}>{children}</div>
    ),
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock createPortal to render in the same DOM tree
vi.mock('react-dom', async () => {
  const actual = await vi.importActual('react-dom');
  return {
    ...actual,
    createPortal: (children: React.ReactNode) => children,
  };
});

// Mock API calls
vi.mock('../../../services/api', () => ({
  createEvalCase: vi.fn(),
  updateEvalCase: vi.fn(),
}));

import { createEvalCase, updateEvalCase } from '../../../services/api';

describe('EvalCaseEditor', () => {
  const defaultExpected: ExpectedBehavior = {
    must_include_any_of: [],
    must_include_any_of_2: [],
    must_include_all_of: [],
    must_not_include_any_of: [],
    contract_check: false,
    min_citations: null,
    max_citations: null,
    behavior: 'answer',
    allow_empty_references: false,
    must_have_article_support_for_normative: true,
    notes: '',
  };

  const mockCase: EvalCase = {
    id: 'test-law-01-test',
    profile: 'LEGAL',
    prompt: 'What are the requirements for testing?',
    test_types: ['retrieval'],
    origin: 'auto',
    expected: defaultExpected,
  };

  const defaultProps = {
    law: 'test-law',
    isOpen: true,
    onClose: vi.fn(),
    onSaved: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders nothing when not open', () => {
    const { container } = render(
      <EvalCaseEditor {...defaultProps} isOpen={false} />
    );
    expect(container.innerHTML).toBe('');
  });

  it('renders modal when open', () => {
    render(<EvalCaseEditor {...defaultProps} />);
    expect(screen.getByText('Ny Test Case')).toBeInTheDocument();
  });

  it('shows "Ny Test Case" title for create mode', () => {
    render(<EvalCaseEditor {...defaultProps} />);
    expect(screen.getByText('Ny Test Case')).toBeInTheDocument();
  });

  it('shows "Rediger Test Case" title for edit mode', () => {
    render(
      <EvalCaseEditor {...defaultProps} existingCase={mockCase} />
    );
    expect(screen.getByText('Rediger Test Case')).toBeInTheDocument();
  });

  it('shows case ID when editing', () => {
    render(
      <EvalCaseEditor {...defaultProps} existingCase={mockCase} />
    );
    expect(screen.getByText(mockCase.id)).toBeInTheDocument();
  });

  it('populates form fields when editing', () => {
    render(
      <EvalCaseEditor {...defaultProps} existingCase={mockCase} />
    );

    // Check prompt is populated
    const textarea = screen.getByPlaceholderText(/Skriv testspørgsmålet/);
    expect(textarea).toHaveValue(mockCase.prompt);
  });

  it('shows validation error for short prompt', async () => {
    render(<EvalCaseEditor {...defaultProps} />);

    const textarea = screen.getByPlaceholderText(/Skriv testspørgsmålet/);
    fireEvent.change(textarea, { target: { value: 'Short' } });

    const saveButton = screen.getByText('Opret case');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(screen.getByText(/mindst 10 tegn/)).toBeInTheDocument();
    });
  });

  it('shows validation error when no test types selected', async () => {
    render(<EvalCaseEditor {...defaultProps} />);

    const textarea = screen.getByPlaceholderText(/Skriv testspørgsmålet/);
    fireEvent.change(textarea, { target: { value: 'This is a long enough prompt for testing' } });

    // Deselect the default retrieval test type
    const retrievalButton = screen.getByText('Retrieval');
    fireEvent.click(retrievalButton);

    const saveButton = screen.getByText('Opret case');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(screen.getByText(/mindst én test type/)).toBeInTheDocument();
    });
  });

  it('toggles test types when clicked', () => {
    render(<EvalCaseEditor {...defaultProps} />);

    const faithfulnessButton = screen.getByText('Faithfulness');

    // Initially not selected (check background class)
    expect(faithfulnessButton).toHaveClass('bg-apple-gray-100');

    // Click to select
    fireEvent.click(faithfulnessButton);
    expect(faithfulnessButton).toHaveClass('bg-apple-blue');

    // Click again to deselect
    fireEvent.click(faithfulnessButton);
    expect(faithfulnessButton).toHaveClass('bg-apple-gray-100');
  });

  it('switches profile when radio button clicked', () => {
    render(<EvalCaseEditor {...defaultProps} />);

    const engineeringRadio = screen.getByLabelText('ENGINEERING');
    fireEvent.click(engineeringRadio);

    expect(engineeringRadio).toBeChecked();
  });

  it('calls createEvalCase API when creating new case', async () => {
    const mockCreated: EvalCase = {
      ...mockCase,
      id: 'test-law-01-new',
      origin: 'manual',
    };
    (createEvalCase as ReturnType<typeof vi.fn>).mockResolvedValue(mockCreated);

    const onSaved = vi.fn();
    render(<EvalCaseEditor {...defaultProps} onSaved={onSaved} />);

    const textarea = screen.getByPlaceholderText(/Skriv testspørgsmålet/);
    fireEvent.change(textarea, { target: { value: 'What are the compliance requirements?' } });

    const saveButton = screen.getByText('Opret case');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(createEvalCase).toHaveBeenCalledWith('test-law', expect.objectContaining({
        profile: 'LEGAL',
        prompt: 'What are the compliance requirements?',
        test_types: ['retrieval'],
      }));
    });

    await waitFor(() => {
      expect(onSaved).toHaveBeenCalledWith(mockCreated);
    });
  });

  it('calls updateEvalCase API when editing', async () => {
    const mockUpdated: EvalCase = {
      ...mockCase,
      prompt: 'Updated question?',
      origin: 'manual',
    };
    (updateEvalCase as ReturnType<typeof vi.fn>).mockResolvedValue(mockUpdated);

    const onSaved = vi.fn();
    render(
      <EvalCaseEditor {...defaultProps} existingCase={mockCase} onSaved={onSaved} />
    );

    const textarea = screen.getByPlaceholderText(/Skriv testspørgsmålet/);
    fireEvent.change(textarea, { target: { value: 'Updated question?' } });

    const saveButton = screen.getByText('Gem ændringer');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(updateEvalCase).toHaveBeenCalledWith(
        'test-law',
        mockCase.id,
        expect.objectContaining({
          prompt: 'Updated question?',
        })
      );
    });

    await waitFor(() => {
      expect(onSaved).toHaveBeenCalledWith(mockUpdated);
    });
  });

  it('shows API error when save fails', async () => {
    (createEvalCase as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('Network error'));

    render(<EvalCaseEditor {...defaultProps} />);

    const textarea = screen.getByPlaceholderText(/Skriv testspørgsmålet/);
    fireEvent.change(textarea, { target: { value: 'What are the requirements for testing?' } });

    const saveButton = screen.getByText('Opret case');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });

  it('calls onClose when cancel button clicked', () => {
    const onClose = vi.fn();
    render(<EvalCaseEditor {...defaultProps} onClose={onClose} />);

    const cancelButton = screen.getByText('Annuller');
    fireEvent.click(cancelButton);

    expect(onClose).toHaveBeenCalled();
  });

  it('calls onClose when close button clicked', () => {
    const onClose = vi.fn();
    render(<EvalCaseEditor {...defaultProps} onClose={onClose} />);

    // Close button is the X in the header
    const closeButtons = screen.getAllByRole('button');
    const closeButton = closeButtons.find((btn) =>
      btn.querySelector('svg path[d*="18L18 6"]')
    );

    if (closeButton) {
      fireEvent.click(closeButton);
      expect(onClose).toHaveBeenCalled();
    }
  });

  it('shows behavior radio buttons', () => {
    render(<EvalCaseEditor {...defaultProps} />);

    expect(screen.getByLabelText(/Svar/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Afstå/)).toBeInTheDocument();
  });

  it('switches behavior when radio button clicked', () => {
    render(<EvalCaseEditor {...defaultProps} />);

    const abstainRadio = screen.getByLabelText(/Afstå/);
    fireEvent.click(abstainRadio);

    expect(abstainRadio).toBeChecked();
  });

  it('shows info text about origin change when editing', () => {
    render(
      <EvalCaseEditor {...defaultProps} existingCase={mockCase} />
    );

    expect(screen.getByText(/sætter origin til "manual"/)).toBeInTheDocument();
  });

  it('shows info text about manual origin when creating', () => {
    render(<EvalCaseEditor {...defaultProps} />);

    expect(screen.getByText(/oprettes som "manual"/)).toBeInTheDocument();
  });

  it('shows tooltip with description on test type buttons', () => {
    render(<EvalCaseEditor {...defaultProps} />);

    const retrievalButton = screen.getByText('Retrieval');
    expect(retrievalButton).toHaveAttribute('title');
    expect(retrievalButton.getAttribute('title')).toContain('hentes');

    const faithfulnessButton = screen.getByText('Faithfulness');
    expect(faithfulnessButton).toHaveAttribute('title');
    expect(faithfulnessButton.getAttribute('title')).toContain('trofast');
  });

  it('shows citation constraints description text', () => {
    render(<EvalCaseEditor {...defaultProps} />);

    expect(screen.getByText(/Validerer at svaret har/i)).toBeInTheDocument();
  });
});
