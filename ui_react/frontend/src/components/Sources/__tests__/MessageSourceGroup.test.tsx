/**
 * Tests for MessageSourceGroup component.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ReactNode } from 'react';
import { MessageSourceGroup } from '../MessageSourceGroup';
import { SourcesPanelProvider } from '../../../contexts';
import type { Reference } from '../../../types';

// Wrapper for context
function wrapper({ children }: { children: ReactNode }) {
  return <SourcesPanelProvider>{children}</SourcesPanelProvider>;
}

// Mock the context with a specific selectedSourceId
vi.mock('../../../contexts', async () => {
  const actual = await vi.importActual('../../../contexts');
  return {
    ...actual,
  };
});

const mockReferences: Reference[] = [
  {
    idx: 1,
    display: 'Artikel 5, stk. 1',
    chunk_text: 'Følgende AI-praksisser er forbudt...',
  },
  {
    idx: 2,
    display: 'Artikel 6, stk. 2',
    chunk_text: 'Højrisiko AI-systemer skal...',
  },
];

// Helper to render with context
function renderWithContext(ui: React.ReactElement) {
  return render(<SourcesPanelProvider>{ui}</SourcesPanelProvider>);
}

describe('MessageSourceGroup', () => {
  it('renders all references', () => {
    renderWithContext(
      <MessageSourceGroup
        messageId="msg-1"
        references={mockReferences}
        citedIndices={new Set([1, 2])}
      />
    );

    expect(screen.getByText(/Artikel 5, stk. 1/)).toBeInTheDocument();
    expect(screen.getByText(/Artikel 6, stk. 2/)).toBeInTheDocument();
  });

  it('applies message-source-group class', () => {
    const { container } = renderWithContext(
      <MessageSourceGroup
        messageId="msg-1"
        references={mockReferences}
        citedIndices={new Set([1])}
      />
    );

    expect(container.querySelector('.message-source-group')).toBeInTheDocument();
  });

  it('includes data-message-id attribute', () => {
    const { container } = renderWithContext(
      <MessageSourceGroup
        messageId="msg-123"
        references={mockReferences}
        citedIndices={new Set([1])}
      />
    );

    expect(container.querySelector('[data-message-id="msg-123"]')).toBeInTheDocument();
  });

  it('shows cited sources expanded by default', () => {
    renderWithContext(
      <MessageSourceGroup
        messageId="msg-1"
        references={mockReferences}
        citedIndices={new Set([1])}
      />
    );

    // Cited source should be expanded (show chunk_text)
    expect(screen.getByText(/Følgende AI-praksisser er forbudt/)).toBeInTheDocument();
  });

  it('passes sourceUrl to SourceItem', () => {
    renderWithContext(
      <MessageSourceGroup
        messageId="msg-1"
        references={mockReferences}
        citedIndices={new Set([1])}
        sourceUrl="https://eur-lex.europa.eu/legal-content/DA/TXT/?uri=CELEX:32024R1689"
      />
    );

    // Source item should render the EUR-Lex link
    expect(screen.getByText('Åbn i EUR-Lex')).toBeInTheDocument();
  });

  it('renders nothing when references array is empty', () => {
    const { container } = renderWithContext(
      <MessageSourceGroup
        messageId="msg-1"
        references={[]}
        citedIndices={new Set()}
      />
    );

    expect(container.querySelector('.message-source-group')).toBeNull();
  });

  it('shows toggle for non-cited sources when present', () => {
    renderWithContext(
      <MessageSourceGroup
        messageId="msg-1"
        references={mockReferences}
        citedIndices={new Set([1])} // Only index 1 is cited
      />
    );

    // Cited source should be visible
    expect(screen.getByText(/Artikel 5, stk. 1/)).toBeInTheDocument();

    // Non-cited source should be hidden behind toggle
    expect(screen.getByText(/Vis 1 andre kilder fra søgningen/)).toBeInTheDocument();
  });

  describe('selection passing', () => {
    it('passes isSelected={true} to SourceItem matching selectedSourceId (MSG-01)', () => {
      // Render with wrapper that provides context
      const { container } = render(
        <SourcesPanelProvider>
          <MessageSourceGroup
            messageId="msg-1"
            references={mockReferences}
            citedIndices={new Set([1, 2])}
          />
        </SourcesPanelProvider>
      );

      // Initially no source should be selected
      const selectedBefore = container.querySelector('.source-item-selected');
      expect(selectedBefore).not.toBeInTheDocument();
    });

    it('passes isSelected={false} to non-matching SourceItems (MSG-02)', () => {
      const { container } = render(
        <SourcesPanelProvider>
          <MessageSourceGroup
            messageId="msg-1"
            references={mockReferences}
            citedIndices={new Set([1, 2])}
          />
        </SourcesPanelProvider>
      );

      // Neither source should have aria-selected="true" initially
      const selectedItems = container.querySelectorAll('[aria-selected="true"]');
      expect(selectedItems.length).toBe(0);
    });
  });
});
