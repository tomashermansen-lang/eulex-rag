/**
 * Tests for PdfDocument link extensions.
 *
 * TDD: These tests define hyperlink functionality for PDF export.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PdfDocument } from '../PdfDocument';
import type { jsPDF } from 'jspdf';

// Mock jsPDF instance
function createMockJsPdf() {
  return {
    text: vi.fn(),
    textWithLink: vi.fn(),
    link: vi.fn(),
    getTextWidth: vi.fn().mockReturnValue(20),
    setPage: vi.fn(),
    internal: {
      pageSize: {
        getHeight: vi.fn().mockReturnValue(297),
        getWidth: vi.fn().mockReturnValue(210),
      },
      getNumberOfPages: vi.fn().mockReturnValue(1),
      getCurrentPageInfo: vi.fn().mockReturnValue({ pageNumber: 1 }),
    },
    getNumberOfPages: vi.fn().mockReturnValue(1),
    setFontSize: vi.fn(),
    setFont: vi.fn(),
    setTextColor: vi.fn(),
    setFillColor: vi.fn(),
    setDrawColor: vi.fn(),
    splitTextToSize: vi.fn((text: string) => [text]),
    rect: vi.fn(),
    line: vi.fn(),
    addPage: vi.fn(),
    save: vi.fn(),
  } as unknown as jsPDF;
}

describe('PdfDocument Link Extensions', () => {
  let doc: PdfDocument;
  let mockJsPdf: ReturnType<typeof createMockJsPdf>;

  beforeEach(() => {
    mockJsPdf = createMockJsPdf();
    doc = new PdfDocument(mockJsPdf, { format: 'a4', orientation: 'portrait', margin: 25 });
  });

  describe('textWithLink()', () => {
    it('delegates to jsPDF with external URL (PD-01)', () => {
      doc.textWithLink('Click here', 10, 20, { url: 'https://example.com' });

      expect(mockJsPdf.textWithLink).toHaveBeenCalledWith(
        'Click here',
        10,
        20,
        { url: 'https://example.com' }
      );
    });

    it('delegates to jsPDF with internal link (PD-02)', () => {
      doc.textWithLink('Go to page 2', 10, 20, { pageNumber: 2, top: 50 });

      expect(mockJsPdf.textWithLink).toHaveBeenCalledWith(
        'Go to page 2',
        10,
        20,
        { pageNumber: 2, top: 50 }
      );
    });
  });

  describe('link()', () => {
    it('creates rectangular link area (PD-03)', () => {
      doc.link(10, 20, 30, 5, { pageNumber: 2, top: 100 });

      expect(mockJsPdf.link).toHaveBeenCalledWith(
        10,
        20,
        30,
        5,
        { pageNumber: 2, top: 100 }
      );
    });

    it('creates link with external URL (PD-03b)', () => {
      doc.link(10, 20, 30, 5, { url: 'https://example.com' });

      expect(mockJsPdf.link).toHaveBeenCalledWith(
        10,
        20,
        30,
        5,
        { url: 'https://example.com' }
      );
    });
  });

  describe('getTextWidth()', () => {
    it('returns text width from jsPDF (PD-04)', () => {
      const width = doc.getTextWidth('Test text');

      expect(mockJsPdf.getTextWidth).toHaveBeenCalledWith('Test text');
      expect(width).toBe(20);
    });
  });

  describe('getCurrentPageNumber()', () => {
    it('returns current page number (PD-05)', () => {
      const pageNum = doc.getCurrentPageNumber();

      expect(pageNum).toBe(1);
    });
  });

  describe('setPage()', () => {
    it('changes active page (PD-06)', () => {
      doc.setPage(3);

      expect(mockJsPdf.setPage).toHaveBeenCalledWith(3);
    });
  });
});
