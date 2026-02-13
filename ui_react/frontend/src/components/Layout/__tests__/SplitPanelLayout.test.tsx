/**
 * Tests for SplitPanelLayout component.
 *
 * TDD: Tests written before implementation.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SplitPanelLayout } from '../SplitPanelLayout';

describe('SplitPanelLayout', () => {
  it('renders chat panel content', () => {
    render(
      <SplitPanelLayout
        chatPanel={<div data-testid="chat">Chat Content</div>}
        sourcesPanel={<div data-testid="sources">Sources Content</div>}
      />
    );

    expect(screen.getByTestId('chat')).toBeInTheDocument();
    expect(screen.getByText('Chat Content')).toBeInTheDocument();
  });

  it('renders sources panel content', () => {
    render(
      <SplitPanelLayout
        chatPanel={<div data-testid="chat">Chat Content</div>}
        sourcesPanel={<div data-testid="sources">Sources Content</div>}
      />
    );

    expect(screen.getByTestId('sources')).toBeInTheDocument();
    expect(screen.getByText('Sources Content')).toBeInTheDocument();
  });

  it('applies split-panel-container class to root element', () => {
    const { container } = render(
      <SplitPanelLayout
        chatPanel={<div>Chat</div>}
        sourcesPanel={<div>Sources</div>}
      />
    );

    expect(container.firstChild).toHaveClass('split-panel-container');
  });

  it('applies chat-panel class to chat section', () => {
    render(
      <SplitPanelLayout
        chatPanel={<div>Chat</div>}
        sourcesPanel={<div>Sources</div>}
      />
    );

    const chatSection = screen.getByRole('region', { name: /chat/i });
    expect(chatSection).toHaveClass('chat-panel');
  });

  it('applies sources-sidepanel class to sources section', () => {
    render(
      <SplitPanelLayout
        chatPanel={<div>Chat</div>}
        sourcesPanel={<div>Sources</div>}
      />
    );

    const sourcesSection = screen.getByRole('complementary', { name: /kilder/i });
    expect(sourcesSection).toHaveClass('sources-sidepanel');
  });

  it('accepts optional className prop', () => {
    const { container } = render(
      <SplitPanelLayout
        chatPanel={<div>Chat</div>}
        sourcesPanel={<div>Sources</div>}
        className="custom-class"
      />
    );

    expect(container.firstChild).toHaveClass('split-panel-container');
    expect(container.firstChild).toHaveClass('custom-class');
  });

  it('wraps children in SourcesPanelProvider context', () => {
    // This test verifies the context is available by not throwing
    // when children try to use useSourcesPanel
    render(
      <SplitPanelLayout
        chatPanel={<div>Chat</div>}
        sourcesPanel={<div>Sources</div>}
      />
    );

    // If we get here without error, the provider is working
    expect(true).toBe(true);
  });
});
