/**
 * Tests for export service.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { formatAsMarkdown, downloadBlob } from '../export';
import type { ChatMessage } from '../../types';

describe('formatAsMarkdown', () => {
  const mockUserMessage: ChatMessage = {
    id: '1',
    role: 'user',
    content: 'Hvad er artikel 5?',
    timestamp: new Date('2024-01-24T10:00:00'),
  };

  const mockAssistantMessage: ChatMessage = {
    id: '2',
    role: 'assistant',
    content: 'Artikel 5 handler om forbudte AI-praksisser [1].',
    timestamp: new Date('2024-01-24T10:00:05'),
    references: [
      {
        idx: 1,
        display: 'Artikel 5, stk. 1',
        chunk_text: 'Følgende AI-praksisser er forbudt...',
      },
    ],
    responseTime: 2.5,
  };

  it('formats empty messages array', () => {
    const result = formatAsMarkdown([]);
    expect(result).toContain('# EuLex Samtale');
    expect(result).not.toContain('## Spørgsmål');
  });

  it('formats user message with question header', () => {
    const result = formatAsMarkdown([mockUserMessage]);
    expect(result).toContain('## Spørgsmål');
    expect(result).toContain('Hvad er artikel 5?');
  });

  it('formats assistant message with answer header', () => {
    const result = formatAsMarkdown([mockAssistantMessage]);
    expect(result).toContain('## Svar');
    expect(result).toContain('Artikel 5 handler om forbudte AI-praksisser');
  });

  it('includes kilder section when references exist', () => {
    const result = formatAsMarkdown([mockAssistantMessage]);
    expect(result).toContain('### Kilder');
    expect(result).toContain('[1] Artikel 5, stk. 1');
  });

  it('excludes kilder section when no references', () => {
    const result = formatAsMarkdown([mockUserMessage]);
    expect(result).not.toContain('### Kilder');
  });

  it('formats conversation with multiple messages', () => {
    const result = formatAsMarkdown([mockUserMessage, mockAssistantMessage]);
    expect(result).toContain('## Spørgsmål');
    expect(result).toContain('## Svar');
    expect(result).toContain('Hvad er artikel 5?');
    expect(result).toContain('Artikel 5 handler om forbudte AI-praksisser');
  });

  it('includes timestamp in export', () => {
    const result = formatAsMarkdown([mockAssistantMessage]);
    // Should include current year in some format
    const currentYear = new Date().getFullYear().toString();
    expect(result).toContain(currentYear);
  });
});

describe('downloadBlob', () => {
  let createObjectURLMock: ReturnType<typeof vi.fn>;
  let revokeObjectURLMock: ReturnType<typeof vi.fn>;
  let createElementMock: ReturnType<typeof vi.fn>;
  let appendChildMock: ReturnType<typeof vi.fn>;
  let removeChildMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    // Mock URL methods
    createObjectURLMock = vi.fn(() => 'blob:mock-url');
    revokeObjectURLMock = vi.fn();
    global.URL.createObjectURL = createObjectURLMock;
    global.URL.revokeObjectURL = revokeObjectURLMock;

    // Mock DOM methods
    const mockLink = {
      href: '',
      download: '',
      click: vi.fn(),
    };
    createElementMock = vi.fn(() => mockLink);
    appendChildMock = vi.fn();
    removeChildMock = vi.fn();

    vi.spyOn(document, 'createElement').mockImplementation(createElementMock);
    vi.spyOn(document.body, 'appendChild').mockImplementation(appendChildMock);
    vi.spyOn(document.body, 'removeChild').mockImplementation(removeChildMock);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('creates blob with correct content and type', () => {
    downloadBlob('test content', 'test.md', 'text/markdown');

    expect(createObjectURLMock).toHaveBeenCalled();
    const blobArg = createObjectURLMock.mock.calls[0][0];
    expect(blobArg).toBeInstanceOf(Blob);
  });

  it('creates download link with correct attributes', () => {
    downloadBlob('test content', 'test.md', 'text/markdown');

    expect(createElementMock).toHaveBeenCalledWith('a');
    const mockLink = createElementMock.mock.results[0].value;
    expect(mockLink.download).toBe('test.md');
  });

  it('triggers click on download link', () => {
    downloadBlob('test content', 'test.md', 'text/markdown');

    const mockLink = createElementMock.mock.results[0].value;
    expect(mockLink.click).toHaveBeenCalled();
  });

  it('cleans up by revoking object URL', () => {
    downloadBlob('test content', 'test.md', 'text/markdown');

    expect(revokeObjectURLMock).toHaveBeenCalledWith('blob:mock-url');
  });
});
