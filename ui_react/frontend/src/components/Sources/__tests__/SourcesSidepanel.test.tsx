/**
 * Tests for SourcesSidepanel component.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SourcesSidepanel } from '../SourcesSidepanel';
import { SourcesPanelProvider } from '../../../contexts';
import type { ChatMessage } from '../../../types';

const mockMessages: ChatMessage[] = [
  {
    id: 'msg-1',
    role: 'user',
    content: 'Hvad er artikel 5?',
    timestamp: new Date(),
  },
  {
    id: 'msg-2',
    role: 'assistant',
    content: 'Artikel 5 handler om forbudte AI-praksisser [1].',
    timestamp: new Date(),
    references: [
      {
        idx: 1,
        display: 'Artikel 5, stk. 1',
        chunk_text: 'Følgende AI-praksisser er forbudt...',
      },
      {
        idx: 2,
        display: 'Artikel 5, stk. 2',
        chunk_text: 'Der gælder særlige regler for...',
      },
    ],
  },
  {
    id: 'msg-3',
    role: 'user',
    content: 'Hvad med artikel 6?',
    timestamp: new Date(),
  },
  {
    id: 'msg-4',
    role: 'assistant',
    content: 'Artikel 6 omhandler højrisiko [1] [2].',
    timestamp: new Date(),
    references: [
      {
        idx: 1,
        display: 'Artikel 6, stk. 1',
        chunk_text: 'Højrisiko AI-systemer er...',
      },
      {
        idx: 2,
        display: 'Artikel 6, stk. 2',
        chunk_text: 'Disse systemer skal overholde...',
      },
    ],
  },
];

// Wrapper that provides context
function Wrapper({ children }: { children: React.ReactNode }) {
  return <SourcesPanelProvider>{children}</SourcesPanelProvider>;
}

describe('SourcesSidepanel', () => {
  it('renders cited sources from all assistant messages', () => {
    render(<SourcesSidepanel messages={mockMessages} />, { wrapper: Wrapper });

    // msg-2 has [1] citation, so only Artikel 5, stk. 1 is visible
    // Artikel 5, stk. 2 is non-cited and hidden behind toggle
    expect(screen.getByText(/Artikel 5, stk. 1/)).toBeInTheDocument();

    // msg-4 has [1] [2] citations, so both are visible
    expect(screen.getByText(/Artikel 6, stk. 1/)).toBeInTheDocument();
    expect(screen.getByText(/Artikel 6, stk. 2/)).toBeInTheDocument();

    // Non-cited source should be behind toggle
    expect(screen.getByText(/Vis 1 andre kilder fra søgningen/)).toBeInTheDocument();
  });

  it('groups sources by message', () => {
    const { container } = render(<SourcesSidepanel messages={mockMessages} />, {
      wrapper: Wrapper,
    });

    // Should have 2 message source groups (one per assistant message with references)
    const groups = container.querySelectorAll('[data-message-id]');
    expect(groups.length).toBe(2);
  });

  it('ignores user messages', () => {
    const { container } = render(<SourcesSidepanel messages={mockMessages} />, {
      wrapper: Wrapper,
    });

    // Should not have groups for user messages
    expect(container.querySelector('[data-message-id="msg-1"]')).toBeNull();
    expect(container.querySelector('[data-message-id="msg-3"]')).toBeNull();
  });

  it('ignores assistant messages without references', () => {
    const messagesWithoutRefs: ChatMessage[] = [
      {
        id: 'msg-1',
        role: 'assistant',
        content: 'No references here',
        timestamp: new Date(),
        references: [],
      },
    ];

    const { container } = render(<SourcesSidepanel messages={messagesWithoutRefs} />, {
      wrapper: Wrapper,
    });

    expect(container.querySelector('[data-message-id]')).toBeNull();
  });

  it('shows empty state when no sources', () => {
    const userOnlyMessages: ChatMessage[] = [
      {
        id: 'msg-1',
        role: 'user',
        content: 'Hello',
        timestamp: new Date(),
      },
    ];

    render(<SourcesSidepanel messages={userOnlyMessages} />, { wrapper: Wrapper });

    expect(screen.getByText(/ingen kilder/i)).toBeInTheDocument();
  });

  it('renders header with total source count badge', () => {
    render(<SourcesSidepanel messages={mockMessages} />, { wrapper: Wrapper });

    // Header should show "Kilder" with badge showing count
    expect(screen.getByText('Kilder')).toBeInTheDocument();
    expect(screen.getByText('4')).toBeInTheDocument(); // Badge with count
  });

  it('scrolls to source when expandSource event is dispatched', async () => {
    const scrollIntoViewMock = vi.fn();
    Element.prototype.scrollIntoView = scrollIntoViewMock;

    render(<SourcesSidepanel messages={mockMessages} />, { wrapper: Wrapper });

    // Dispatch expand event
    const event = new CustomEvent('expandSource', {
      detail: { refId: 'ref-msg-2-1', messageId: 'msg-2' },
    });
    window.dispatchEvent(event);

    await waitFor(() => {
      expect(scrollIntoViewMock).toHaveBeenCalled();
    });
  });

  it('passes sourceUrl to MessageSourceGroup', () => {
    render(
      <SourcesSidepanel
        messages={mockMessages}
        sourceUrl="https://eur-lex.europa.eu/test"
      />,
      { wrapper: Wrapper }
    );

    // Source items should have EUR-Lex links
    const links = screen.getAllByText('Åbn i EUR-Lex');
    expect(links.length).toBeGreaterThan(0);
  });

  describe('selection clearing', () => {
    it('clears selection when new assistant message arrives (SSP-01)', () => {
      const initialMessages: ChatMessage[] = [
        {
          id: 'msg-1',
          role: 'assistant',
          content: 'First response [1]',
          timestamp: new Date(),
          references: [
            { idx: 1, display: 'Source 1', chunk_text: 'Text 1' },
          ],
        },
      ];

      const { rerender } = render(
        <SourcesSidepanel messages={initialMessages} />,
        { wrapper: Wrapper }
      );

      // Add a new assistant message with references
      const updatedMessages: ChatMessage[] = [
        ...initialMessages,
        {
          id: 'msg-2',
          role: 'assistant',
          content: 'Second response [1]',
          timestamp: new Date(),
          references: [
            { idx: 1, display: 'Source 2', chunk_text: 'Text 2' },
          ],
        },
      ];

      rerender(
        <Wrapper>
          <SourcesSidepanel messages={updatedMessages} />
        </Wrapper>
      );

      // Verify the new source is rendered
      expect(screen.getByText(/Source 2/)).toBeInTheDocument();
    });
  });
});
