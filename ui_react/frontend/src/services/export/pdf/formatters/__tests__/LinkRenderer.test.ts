/**
 * Tests for LinkRenderer.
 *
 * TDD: These tests define link rendering with underline styling for PDF export.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { LinkRenderer } from '../LinkRenderer';
import type { PdfDocument } from '../../PdfDocument';

// Mock PdfDocument
function createMockDoc(): PdfDocument {
  return {
    textWithLink: vi.fn(),
    setTextColor: vi.fn(),
    setDrawColor: vi.fn(),
    getTextWidth: vi.fn().mockReturnValue(15),
    line: vi.fn(),
  } as unknown as PdfDocument;
}

describe('LinkRenderer', () => {
  let renderer: LinkRenderer;
  let mockDoc: PdfDocument;

  beforeEach(() => {
    renderer = new LinkRenderer();
    mockDoc = createMockDoc();
  });

  describe('renderLink()', () => {
    it('renders external link with URL (LR-01)', () => {
      renderer.renderLink(mockDoc, 'Click here', 10, 20, { url: 'https://example.com' });

      expect(mockDoc.textWithLink).toHaveBeenCalledWith(
        'Click here',
        10,
        20,
        { url: 'https://example.com' }
      );
    });

    it('renders internal link with pageNumber and top (LR-02)', () => {
      renderer.renderLink(mockDoc, 'Go to source', 10, 20, { pageNumber: 2, top: 50 });

      expect(mockDoc.textWithLink).toHaveBeenCalledWith(
        'Go to source',
        10,
        20,
        { pageNumber: 2, top: 50 }
      );
    });

    it('sets blue link color rgb(0, 102, 204) (LR-03)', () => {
      renderer.renderLink(mockDoc, 'Link', 10, 20, { url: 'https://example.com' });

      expect(mockDoc.setTextColor).toHaveBeenCalledWith(0, 102, 204);
    });

    it('draws underline via line() (LR-04)', () => {
      renderer.renderLink(mockDoc, 'Link text', 10, 20, { url: 'https://example.com' });

      expect(mockDoc.setDrawColor).toHaveBeenCalledWith(0, 102, 204);
      expect(mockDoc.line).toHaveBeenCalled();
    });

    it('returns text width for layout (LR-05)', () => {
      const width = renderer.renderLink(mockDoc, 'Link', 10, 20, { url: 'https://example.com' });

      expect(width).toBe(15);
    });

    it('positions underline below text (LR-06)', () => {
      renderer.renderLink(mockDoc, 'Link', 10, 20, { url: 'https://example.com' });

      // Line should be drawn from x to x+width, at y + small offset
      expect(mockDoc.line).toHaveBeenCalledWith(
        10,           // x start
        expect.any(Number), // y + offset (below text)
        25,           // x + width (10 + 15)
        expect.any(Number)  // same y
      );
    });
  });
});
