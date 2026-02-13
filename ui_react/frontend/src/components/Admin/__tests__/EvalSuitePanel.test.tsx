/**
 * Tests for EvalSuitePanel component.
 *
 * Panel for managing eval cases for a law.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { EvalSuitePanel } from '../EvalSuitePanel';
import type { EvalCase, ExpectedBehavior, EvalCaseListResponse } from '../../../types';

// Mock framer-motion
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
      <div {...props}>{children}</div>
    ),
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

// Mock EvalCaseEditor to simplify testing
vi.mock('../EvalCaseEditor', () => ({
  EvalCaseEditor: ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) =>
    isOpen ? <div data-testid="eval-case-editor"><button onClick={onClose}>Close Editor</button></div> : null,
}));

// Mock API
vi.mock('../../../services/api', () => ({
  listEvalCases: vi.fn(),
  deleteEvalCase: vi.fn(),
  duplicateEvalCase: vi.fn(),
}));

import { listEvalCases, deleteEvalCase, duplicateEvalCase } from '../../../services/api';

describe('EvalSuitePanel', () => {
  const defaultExpected: ExpectedBehavior = {
    must_include_any_of: ['article:5'],
    must_include_any_of_2: [],
    must_include_all_of: [],
    must_not_include_any_of: [],
    contract_check: false,
    min_citations: null,
    max_citations: null,
    behavior: 'answer',
    allow_empty_references: false,
    must_have_article_support_for_normative: true,
    notes: 'Test note',
  };

  const mockCases: EvalCase[] = [
    {
      id: 'test-law-01-first',
      profile: 'LEGAL',
      prompt: 'What are the first requirements?',
      test_types: ['retrieval'],
      origin: 'auto',
      expected: defaultExpected,
    },
    {
      id: 'test-law-02-second',
      profile: 'ENGINEERING',
      prompt: 'How does the second feature work?',
      test_types: ['retrieval', 'faithfulness'],
      origin: 'manual',
      expected: { ...defaultExpected, behavior: 'abstain' },
    },
    {
      id: 'test-law-03-third',
      profile: 'LEGAL',
      prompt: 'What about the third scenario?',
      test_types: ['relevancy'],
      origin: 'auto',
      expected: defaultExpected,
    },
  ];

  const defaultProps = {
    law: 'test-law',
    displayName: 'Test Law',
    isOpen: true,
    onClose: vi.fn(),
    onCasesChanged: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (listEvalCases as ReturnType<typeof vi.fn>).mockResolvedValue({
      cases: mockCases,
      total: mockCases.length,
    } as EvalCaseListResponse);
  });

  it('renders nothing when not open', () => {
    const { container } = render(
      <EvalSuitePanel {...defaultProps} isOpen={false} />
    );
    expect(container.innerHTML).toBe('');
  });

  it('shows loading spinner initially', async () => {
    // Make the API slow
    (listEvalCases as ReturnType<typeof vi.fn>).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    render(<EvalSuitePanel {...defaultProps} />);

    // Look for the spinner
    expect(document.querySelector('.animate-spin')).toBeInTheDocument();
  });

  it('shows error state when API fails', async () => {
    (listEvalCases as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Failed to load')
    );

    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('Failed to load')).toBeInTheDocument();
    });
  });

  it('shows retry button on error', async () => {
    (listEvalCases as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Failed to load')
    );

    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('Prøv igen')).toBeInTheDocument();
    });
  });

  it('shows empty state when no cases', async () => {
    (listEvalCases as ReturnType<typeof vi.fn>).mockResolvedValue({
      cases: [],
      total: 0,
    });

    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText(/Ingen eval cases endnu/)).toBeInTheDocument();
    });
  });

  it('renders case list after loading', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
      expect(screen.getByText('How does the second feature work?')).toBeInTheDocument();
      expect(screen.getByText('What about the third scenario?')).toBeInTheDocument();
    });
  });

  it('shows header with display name', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    expect(screen.getByText('Test Law')).toBeInTheDocument();
  });

  it('shows case count', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('3 cases')).toBeInTheDocument();
    });
  });

  it('shows origin badges', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      const autoBadges = screen.getAllByText('AUTO');
      const manualBadges = screen.getAllByText('MANUAL');

      expect(autoBadges).toHaveLength(2);
      expect(manualBadges).toHaveLength(1);
    });
  });

  it('filters by manual origin', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    const filterSelect = screen.getByRole('combobox');
    fireEvent.change(filterSelect, { target: { value: 'manual' } });

    await waitFor(() => {
      expect(screen.queryByText('What are the first requirements?')).not.toBeInTheDocument();
      expect(screen.getByText('How does the second feature work?')).toBeInTheDocument();
    });
  });

  it('filters by auto origin', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('How does the second feature work?')).toBeInTheDocument();
    });

    const filterSelect = screen.getByRole('combobox');
    fireEvent.change(filterSelect, { target: { value: 'auto' } });

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
      expect(screen.queryByText('How does the second feature work?')).not.toBeInTheDocument();
    });
  });

  it('filters by test type', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    const filterSelect = screen.getByRole('combobox');
    fireEvent.change(filterSelect, { target: { value: 'relevancy' } });

    await waitFor(() => {
      expect(screen.queryByText('What are the first requirements?')).not.toBeInTheDocument();
      expect(screen.getByText('What about the third scenario?')).toBeInTheDocument();
    });
  });

  it('searches cases by prompt', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText(/Søg i cases/);
    fireEvent.change(searchInput, { target: { value: 'second' } });

    await waitFor(() => {
      expect(screen.queryByText('What are the first requirements?')).not.toBeInTheDocument();
      expect(screen.getByText('How does the second feature work?')).toBeInTheDocument();
    });
  });

  it('searches cases by ID', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText(/Søg i cases/);
    fireEvent.change(searchInput, { target: { value: '03-third' } });

    await waitFor(() => {
      expect(screen.queryByText('What are the first requirements?')).not.toBeInTheDocument();
      expect(screen.getByText('What about the third scenario?')).toBeInTheDocument();
    });
  });

  it('shows empty state when filter matches nothing', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText(/Søg i cases/);
    fireEvent.change(searchInput, { target: { value: 'nonexistent' } });

    await waitFor(() => {
      expect(screen.getByText(/Ingen cases matcher filteret/)).toBeInTheDocument();
    });
  });

  it('opens editor when "Ny" button clicked', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    const newCaseButton = screen.getByText('Ny');
    fireEvent.click(newCaseButton);

    expect(screen.getByTestId('eval-case-editor')).toBeInTheDocument();
  });

  it('expands case details on click', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    // Click the first case row
    const firstCase = screen.getByText('What are the first requirements?').closest('div[class*="cursor-pointer"]');
    if (firstCase) {
      fireEvent.click(firstCase);

      await waitFor(() => {
        expect(screen.getByText('Forventet adfærd')).toBeInTheDocument();
        expect(screen.getByText('Skal svare')).toBeInTheDocument();
      });
    }
  });

  it('shows edit button in expanded case', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    // Click to expand
    const firstCase = screen.getByText('What are the first requirements?').closest('div[class*="cursor-pointer"]');
    if (firstCase) {
      fireEvent.click(firstCase);

      await waitFor(() => {
        expect(screen.getByText('Rediger')).toBeInTheDocument();
      });
    }
  });

  it('shows duplicate button in expanded case', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    // Click to expand
    const firstCase = screen.getByText('What are the first requirements?').closest('div[class*="cursor-pointer"]');
    if (firstCase) {
      fireEvent.click(firstCase);

      await waitFor(() => {
        expect(screen.getByText('Duplikér')).toBeInTheDocument();
      });
    }
  });

  it('duplicates case when duplicate button clicked', async () => {
    const duplicatedCase: EvalCase = {
      ...mockCases[0],
      id: 'test-law-04-copy',
      origin: 'manual',
    };
    (duplicateEvalCase as ReturnType<typeof vi.fn>).mockResolvedValue(duplicatedCase);

    const onCasesChanged = vi.fn();
    render(<EvalSuitePanel {...defaultProps} onCasesChanged={onCasesChanged} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    // Click to expand
    const firstCase = screen.getByText('What are the first requirements?').closest('div[class*="cursor-pointer"]');
    if (firstCase) {
      fireEvent.click(firstCase);
    }

    await waitFor(() => {
      expect(screen.getByText('Duplikér')).toBeInTheDocument();
    });

    const duplicateButton = screen.getByText('Duplikér');
    fireEvent.click(duplicateButton);

    await waitFor(() => {
      expect(duplicateEvalCase).toHaveBeenCalledWith('test-law', 'test-law-01-first');
      expect(onCasesChanged).toHaveBeenCalled();
    });
  });

  it('shows delete confirmation when delete clicked', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    // Expand first case
    const firstCase = screen.getByText('What are the first requirements?').closest('div[class*="cursor-pointer"]');
    if (firstCase) {
      fireEvent.click(firstCase);
    }

    await waitFor(() => {
      expect(screen.getByText('Slet')).toBeInTheDocument();
    });

    const deleteButton = screen.getByText('Slet');
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(screen.getByText('Slet?')).toBeInTheDocument();
      expect(screen.getByText('Ja')).toBeInTheDocument();
      expect(screen.getByText('Nej')).toBeInTheDocument();
    });
  });

  it('deletes case when confirmed', async () => {
    (deleteEvalCase as ReturnType<typeof vi.fn>).mockResolvedValue(undefined);

    const onCasesChanged = vi.fn();
    render(<EvalSuitePanel {...defaultProps} onCasesChanged={onCasesChanged} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    // Expand first case
    const firstCase = screen.getByText('What are the first requirements?').closest('div[class*="cursor-pointer"]');
    if (firstCase) {
      fireEvent.click(firstCase);
    }

    await waitFor(() => {
      expect(screen.getByText('Slet')).toBeInTheDocument();
    });

    // Click delete
    fireEvent.click(screen.getByText('Slet'));

    await waitFor(() => {
      expect(screen.getByText('Ja')).toBeInTheDocument();
    });

    // Confirm delete
    fireEvent.click(screen.getByText('Ja'));

    await waitFor(() => {
      expect(deleteEvalCase).toHaveBeenCalledWith('test-law', 'test-law-01-first');
      expect(onCasesChanged).toHaveBeenCalled();
    });
  });

  it('cancels delete when "Nej" clicked', async () => {
    render(<EvalSuitePanel {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText('What are the first requirements?')).toBeInTheDocument();
    });

    // Expand first case
    const firstCase = screen.getByText('What are the first requirements?').closest('div[class*="cursor-pointer"]');
    if (firstCase) {
      fireEvent.click(firstCase);
    }

    await waitFor(() => {
      expect(screen.getByText('Slet')).toBeInTheDocument();
    });

    // Click delete
    fireEvent.click(screen.getByText('Slet'));

    await waitFor(() => {
      expect(screen.getByText('Nej')).toBeInTheDocument();
    });

    // Cancel
    fireEvent.click(screen.getByText('Nej'));

    await waitFor(() => {
      // Should be back to regular delete button
      expect(screen.queryByText('Slet?')).not.toBeInTheDocument();
      expect(screen.getByText('Slet')).toBeInTheDocument();
    });
  });

  it('calls onClose when close button clicked', async () => {
    const onClose = vi.fn();
    render(<EvalSuitePanel {...defaultProps} onClose={onClose} />);

    // Find the close button (X in header)
    const closeButtons = screen.getAllByRole('button');
    const closeButton = closeButtons.find((btn) =>
      btn.querySelector('svg path[d*="18L18 6"]')
    );

    if (closeButton) {
      fireEvent.click(closeButton);
      expect(onClose).toHaveBeenCalled();
    }
  });
});
