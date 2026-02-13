/**
 * Tests for QuestionSection PDF export.
 *
 * TDD: Tests verify question content rendering with background.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { QuestionSection } from '../QuestionSection';
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
    splitTextToSize: vi.fn((text: string) => [text]),
    setFillColor: vi.fn(),
    rect: vi.fn(),
  } as unknown as PdfDocument;
}

describe('QuestionSection', () => {
  let section: QuestionSection;
  let mockDoc: PdfDocument;

  beforeEach(() => {
    section = new QuestionSection();
    mockDoc = createMockDoc();
  });

  describe('render', () => {
    it('renders label "Sporgsmaal" (QS-01)', () => {
      section.render(mockDoc, 'Test question', 50);

      // Should render the question label (Danish chars replaced)
      expect(mockDoc.text).toHaveBeenCalledWith(
        expect.stringMatching(/Sp.*rgsm.*l/),
        expect.any(Number),
        expect.any(Number)
      );
    });

    it('returns updated Y position (QS-02)', () => {
      const result = section.render(mockDoc, 'Test question', 50);

      expect(result).toBeGreaterThan(50);
    });

    it('draws subtle background (QS-03)', () => {
      section.render(mockDoc, 'Test question', 50);

      expect(mockDoc.setFillColor).toHaveBeenCalledWith(248, 248, 248);
      expect(mockDoc.rect).toHaveBeenCalled();
    });

    it('renders question text (QS-04)', () => {
      section.render(mockDoc, 'What is GDPR?', 50);

      expect(mockDoc.splitTextToSize).toHaveBeenCalled();
      expect(mockDoc.text).toHaveBeenCalled();
    });

    it('adds new page when needed (QS-05)', () => {
      vi.mocked(mockDoc.needsNewPage).mockReturnValue(true);

      section.render(mockDoc, 'Long question text', 50);

      expect(mockDoc.addPage).toHaveBeenCalled();
    });

    it('handles empty content (QS-06)', () => {
      const result = section.render(mockDoc, '', 50);

      expect(result).toBeGreaterThan(50);
    });

    it('applies subheading style for label (QS-07)', () => {
      section.render(mockDoc, 'Test', 50);

      expect(mockDoc.applyStyle).toHaveBeenCalled();
    });

    it('applies body style for text (QS-08)', () => {
      section.render(mockDoc, 'Test', 50);

      // applyStyle called multiple times for different elements
      expect(mockDoc.applyStyle).toHaveBeenCalled();
    });
  });

  describe('estimateHeight', () => {
    it('returns positive height (QS-09)', () => {
      const height = section.estimateHeight(mockDoc, 'Some text');

      expect(height).toBeGreaterThan(0);
    });

    it('accounts for multiple lines (QS-10)', () => {
      vi.mocked(mockDoc.splitTextToSize).mockReturnValueOnce(['Line 1']);
      const height1 = section.estimateHeight(mockDoc, 'Short');

      vi.mocked(mockDoc.splitTextToSize).mockReturnValueOnce(['Line 1', 'Line 2', 'Line 3']);
      const height2 = section.estimateHeight(mockDoc, 'Longer text that wraps');

      expect(height2).toBeGreaterThan(height1);
    });
  });
});
