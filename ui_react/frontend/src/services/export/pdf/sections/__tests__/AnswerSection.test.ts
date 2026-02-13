/**
 * Tests for AnswerSection PDF export.
 *
 * TDD: Tests verify answer content parsing and rendering.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AnswerSection } from '../AnswerSection';
import type { PdfDocument } from '../../PdfDocument';

// Mock PdfDocument
function createMockDoc(): PdfDocument {
  return {
    getMargin: vi.fn().mockReturnValue(20),
    getMaxContentWidth: vi.fn().mockReturnValue(170),
    applyStyle: vi.fn(),
    text: vi.fn(),
    needsNewPage: vi.fn().mockReturnValue(false),
    addPage: vi.fn(),
    splitTextToSize: vi.fn((text: string) => [text]), // Return text as single line
    getLineHeight: vi.fn().mockReturnValue(6),
  } as unknown as PdfDocument;
}

describe('AnswerSection', () => {
  let section: AnswerSection;
  let mockDoc: PdfDocument;

  beforeEach(() => {
    section = new AnswerSection();
    mockDoc = createMockDoc();
  });

  describe('render', () => {
    it('renders section label "Svar" (AS-01)', () => {
      section.render(mockDoc, 'Some answer text', [], 50);

      expect(mockDoc.text).toHaveBeenCalledWith('Svar', 20, 50);
    });

    it('returns updated Y position (AS-02)', () => {
      const result = section.render(mockDoc, 'Simple paragraph', [], 50);

      expect(result).toBeGreaterThan(50);
    });

    it('parses markdown headers (AS-03)', () => {
      const content = '## Heading\nParagraph text';
      section.render(mockDoc, content, [], 50);

      // Should render heading and paragraph
      expect(mockDoc.text).toHaveBeenCalledWith('Svar', 20, 50);
      expect(mockDoc.applyStyle).toHaveBeenCalled();
    });

    it('parses bold-only lines as headings (AS-04)', () => {
      const content = '**Bold Heading**\nRegular text';
      section.render(mockDoc, content, [], 50);

      expect(mockDoc.text).toHaveBeenCalledWith('Svar', 20, 50);
      // Bold heading should be rendered
      expect(mockDoc.applyStyle).toHaveBeenCalled();
    });

    it('cleans markdown formatting from paragraphs (AS-05)', () => {
      const content = 'Text with **bold** and *italic* formatting';
      section.render(mockDoc, content, [], 50);

      // mockDoc.splitTextToSize should receive cleaned text
      expect(mockDoc.splitTextToSize).toHaveBeenCalled();
    });

    it('handles empty content gracefully (AS-06)', () => {
      const result = section.render(mockDoc, '', [], 50);

      // Should still render label and return valid Y
      expect(mockDoc.text).toHaveBeenCalledWith('Svar', 20, 50);
      expect(result).toBeGreaterThan(50);
    });

    it('adds new page when needed (AS-07)', () => {
      vi.mocked(mockDoc.needsNewPage).mockReturnValue(true);

      section.render(mockDoc, 'Long content\nAnother line', [], 50);

      expect(mockDoc.addPage).toHaveBeenCalled();
    });

    it('preserves inline citations (AS-08)', () => {
      const content = 'According to [1] and [2], the law applies.';
      section.render(mockDoc, content, [], 50);

      // Citations should be preserved in output
      expect(mockDoc.splitTextToSize).toHaveBeenCalledWith(
        expect.stringContaining('[1]'),
        expect.any(Number)
      );
    });

    it('handles multiple headings and paragraphs (AS-09)', () => {
      const content = `## Section 1
First paragraph.

## Section 2
Second paragraph.`;

      section.render(mockDoc, content, [], 50);

      // Multiple texts should be rendered
      expect(mockDoc.text).toHaveBeenCalledTimes(5); // Svar + 2 headings + 2 paragraphs
    });

    it('handles bold headings with colon (AS-10)', () => {
      const content = '**Retsgrundlag:**\nSome legal basis text.';
      section.render(mockDoc, content, [], 50);

      expect(mockDoc.applyStyle).toHaveBeenCalled();
    });
  });

  describe('estimateHeight', () => {
    it('returns positive height for content (AS-11)', () => {
      const height = section.estimateHeight(mockDoc, 'Some text');

      expect(height).toBeGreaterThan(0);
    });

    it('accounts for headings (AS-12)', () => {
      const withHeading = '## Heading\nParagraph';
      const withoutHeading = 'Paragraph';

      const heightWithHeading = section.estimateHeight(mockDoc, withHeading);
      const heightWithoutHeading = section.estimateHeight(mockDoc, withoutHeading);

      expect(heightWithHeading).toBeGreaterThan(heightWithoutHeading);
    });

    it('accounts for multiple paragraphs (AS-13)', () => {
      const oneParagraph = 'Single paragraph.';
      const twoParagraphs = 'First paragraph.\n\nSecond paragraph.';

      const heightOne = section.estimateHeight(mockDoc, oneParagraph);
      const heightTwo = section.estimateHeight(mockDoc, twoParagraphs);

      expect(heightTwo).toBeGreaterThan(heightOne);
    });

    it('handles empty content (AS-14)', () => {
      const height = section.estimateHeight(mockDoc, '');

      // Should return at least label height
      expect(height).toBeGreaterThan(0);
    });
  });

  describe('content parsing', () => {
    it('skips empty lines between paragraphs (AS-15)', () => {
      const content = 'First paragraph.\n\n\n\nSecond paragraph.';
      section.render(mockDoc, content, [], 50);

      // Should render Svar + 2 paragraphs (not empty lines)
      expect(mockDoc.text).toHaveBeenCalledTimes(3);
    });

    it('handles mixed heading formats (AS-16)', () => {
      const content = `# H1 Header
## H2 Header
### H3 Header
Regular text`;

      section.render(mockDoc, content, [], 50);

      expect(mockDoc.text).toHaveBeenCalled();
    });

    it('handles underscore bold format (AS-17)', () => {
      const content = '__Underscore Heading__\nSome text.';
      section.render(mockDoc, content, [], 50);

      expect(mockDoc.applyStyle).toHaveBeenCalled();
    });

    it('removes inline code formatting (AS-18)', () => {
      const content = 'Text with `inline code` here.';
      section.render(mockDoc, content, [], 50);

      // Should clean inline code markers
      expect(mockDoc.splitTextToSize).toHaveBeenCalledWith(
        expect.not.stringContaining('`'),
        expect.any(Number)
      );
    });
  });

  describe('citation position tracking', () => {
    let mockCitationCollector: { record: ReturnType<typeof vi.fn> };

    beforeEach(() => {
      mockCitationCollector = {
        record: vi.fn(),
      };
      // Add getTextWidth to mock doc for position calculation
      (mockDoc as unknown as Record<string, unknown>).getTextWidth = vi.fn().mockReturnValue(10);
      (mockDoc as unknown as Record<string, unknown>).getCurrentPageNumber = vi.fn().mockReturnValue(1);
    });

    it('records citation positions when collector provided (AS-19)', () => {
      const content = 'According to [1], the data must be processed.';
      section.render(
        mockDoc,
        content,
        [],
        50,
        mockCitationCollector as unknown as import('../../CitationLinkCollector').CitationLinkCollector
      );

      expect(mockCitationCollector.record).toHaveBeenCalledWith(
        '[1]',
        expect.any(Number),
        expect.any(Number),
        expect.any(Number),
        expect.any(Number),
        expect.any(Number)
      );
    });

    it('records multiple citations in same line (AS-20)', () => {
      const content = 'See [1] and [2] for details.';
      section.render(
        mockDoc,
        content,
        [],
        50,
        mockCitationCollector as unknown as import('../../CitationLinkCollector').CitationLinkCollector
      );

      expect(mockCitationCollector.record).toHaveBeenCalledTimes(2);
    });

    it('does not record when no collector provided (AS-21)', () => {
      const content = 'According to [1], the law applies.';
      // No collector passed - should not throw
      expect(() => section.render(mockDoc, content, [], 50)).not.toThrow();
    });

    it('tracks citations across multiple paragraphs (AS-22)', () => {
      const content = 'First [1] paragraph.\n\nSecond [2] paragraph.';
      section.render(
        mockDoc,
        content,
        [],
        50,
        mockCitationCollector as unknown as import('../../CitationLinkCollector').CitationLinkCollector
      );

      expect(mockCitationCollector.record).toHaveBeenCalledTimes(2);
    });
  });
});
