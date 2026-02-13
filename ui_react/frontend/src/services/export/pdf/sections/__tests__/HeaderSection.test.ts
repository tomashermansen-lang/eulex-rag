/**
 * Tests for HeaderSection PDF export.
 *
 * TDD: Tests verify header rendering with title and date.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HeaderSection } from '../HeaderSection';
import type { PdfDocument } from '../../PdfDocument';

// Mock PdfDocument
function createMockDoc(): PdfDocument {
  return {
    getMargin: vi.fn().mockReturnValue(20),
    getPageWidth: vi.fn().mockReturnValue(210),
    applyStyle: vi.fn(),
    setFontSize: vi.fn(),
    text: vi.fn(),
    setDrawColor: vi.fn(),
    line: vi.fn(),
  } as unknown as PdfDocument;
}

describe('HeaderSection', () => {
  let section: HeaderSection;
  let mockDoc: PdfDocument;

  beforeEach(() => {
    section = new HeaderSection();
    mockDoc = createMockDoc();
  });

  describe('render', () => {
    it('renders title "EuLex Analyse" (HS-01)', () => {
      section.render(mockDoc, 50);

      expect(mockDoc.text).toHaveBeenCalledWith('EuLex Analyse', 20, 50);
    });

    it('sets title font size to 18 (HS-02)', () => {
      section.render(mockDoc, 50);

      expect(mockDoc.setFontSize).toHaveBeenCalledWith(18);
    });

    it('renders export date (HS-03)', () => {
      section.render(mockDoc, 50);

      // Second text call should be the date
      expect(mockDoc.text).toHaveBeenCalledWith(
        expect.stringContaining('Eksporteret:'),
        20,
        expect.any(Number)
      );
    });

    it('draws separator line (HS-04)', () => {
      section.render(mockDoc, 50);

      expect(mockDoc.setDrawColor).toHaveBeenCalledWith(200, 200, 200);
      expect(mockDoc.line).toHaveBeenCalled();
    });

    it('returns updated Y position (HS-05)', () => {
      const result = section.render(mockDoc, 50);

      expect(result).toBeGreaterThan(50);
    });

    it('applies heading style for title (HS-06)', () => {
      section.render(mockDoc, 50);

      expect(mockDoc.applyStyle).toHaveBeenCalled();
    });

    it('applies footnote style for date (HS-07)', () => {
      section.render(mockDoc, 50);

      // applyStyle should be called multiple times
      expect(mockDoc.applyStyle).toHaveBeenCalledTimes(2);
    });
  });

  describe('estimateHeight', () => {
    it('returns fixed height of 30 (HS-08)', () => {
      const height = section.estimateHeight();

      expect(height).toBe(30);
    });
  });
});
