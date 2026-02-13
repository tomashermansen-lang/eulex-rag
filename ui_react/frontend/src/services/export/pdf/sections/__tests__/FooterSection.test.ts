/**
 * Tests for FooterSection PDF export.
 *
 * TDD: Tests verify footer rendering on all pages.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { FooterSection } from '../FooterSection';
import type { PdfDocument } from '../../PdfDocument';

// Mock jsPDF instance
function createMockJsPdf() {
  return {
    setPage: vi.fn(),
    getTextWidth: vi.fn().mockReturnValue(50),
  };
}

// Mock PdfDocument
function createMockDoc() {
  const mockJsPdf = createMockJsPdf();
  return {
    getJsPdf: vi.fn().mockReturnValue(mockJsPdf),
    getPageHeight: vi.fn().mockReturnValue(297),
    getPageWidth: vi.fn().mockReturnValue(210),
    getMargin: vi.fn().mockReturnValue(20),
    setDrawColor: vi.fn(),
    line: vi.fn(),
    applyStyle: vi.fn(),
    text: vi.fn(),
    mockJsPdf,
  } as unknown as PdfDocument & { mockJsPdf: ReturnType<typeof createMockJsPdf> };
}

describe('FooterSection', () => {
  let section: FooterSection;
  let mockDoc: ReturnType<typeof createMockDoc>;

  beforeEach(() => {
    section = new FooterSection();
    mockDoc = createMockDoc();
  });

  describe('renderOnAllPages', () => {
    it('draws footer line on each page (FS-01)', () => {
      section.renderOnAllPages(mockDoc, 3);

      expect(mockDoc.line).toHaveBeenCalledTimes(3);
    });

    it('renders page numbers (FS-02)', () => {
      section.renderOnAllPages(mockDoc, 2);

      expect(mockDoc.text).toHaveBeenCalledWith(
        'Side 1 af 2',
        expect.any(Number),
        expect.any(Number)
      );
      expect(mockDoc.text).toHaveBeenCalledWith(
        'Side 2 af 2',
        expect.any(Number),
        expect.any(Number)
      );
    });

    it('renders EuLex branding (FS-03)', () => {
      section.renderOnAllPages(mockDoc, 1);

      expect(mockDoc.text).toHaveBeenCalledWith(
        'EuLex',
        20, // margin
        expect.any(Number)
      );
    });

    it('sets page for each iteration (FS-04)', () => {
      section.renderOnAllPages(mockDoc, 3);

      expect(mockDoc.mockJsPdf.setPage).toHaveBeenCalledWith(1);
      expect(mockDoc.mockJsPdf.setPage).toHaveBeenCalledWith(2);
      expect(mockDoc.mockJsPdf.setPage).toHaveBeenCalledWith(3);
    });

    it('draws separator line with correct color (FS-05)', () => {
      section.renderOnAllPages(mockDoc, 1);

      expect(mockDoc.setDrawColor).toHaveBeenCalledWith(200, 200, 200);
    });

    it('applies footnote style for text (FS-06)', () => {
      section.renderOnAllPages(mockDoc, 1);

      expect(mockDoc.applyStyle).toHaveBeenCalled();
    });

    it('handles single page document (FS-07)', () => {
      section.renderOnAllPages(mockDoc, 1);

      expect(mockDoc.text).toHaveBeenCalledWith(
        'Side 1 af 1',
        expect.any(Number),
        expect.any(Number)
      );
    });

    it('calculates text width for alignment (FS-08)', () => {
      section.renderOnAllPages(mockDoc, 1);

      expect(mockDoc.mockJsPdf.getTextWidth).toHaveBeenCalled();
    });

    it('renders AI disclaimer on every page (FS-09)', () => {
      section.renderOnAllPages(mockDoc, 2);

      const disclaimerText = 'AI-genereret · Ikke juridisk rådgivning';
      // Should appear once per page
      const disclaimerCalls = (mockDoc.text as ReturnType<typeof vi.fn>).mock.calls.filter(
        (call: unknown[]) => call[0] === disclaimerText
      );
      expect(disclaimerCalls).toHaveLength(2);
    });

    it('renders AI disclaimer centered between branding and page number (FS-10)', () => {
      section.renderOnAllPages(mockDoc, 1);

      const disclaimerText = 'AI-genereret · Ikke juridisk rådgivning';
      expect(mockDoc.text).toHaveBeenCalledWith(
        disclaimerText,
        expect.any(Number),
        expect.any(Number)
      );
    });
  });
});
