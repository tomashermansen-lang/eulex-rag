/**
 * Tests for SourcesSection PDF export.
 *
 * TDD: Tests verify sources/citations rendering with proper grouping.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { SourcesSection } from '../SourcesSection';
import type { PdfDocument } from '../../PdfDocument';
import type { Reference } from '../../../../../types';
import { setCorpusRegistry } from '../../data/corpusMetadata';
import type { CorpusInfo } from '../../../../../types';

// Test corpus data
const testCorpora: CorpusInfo[] = [
  {
    id: 'gdpr',
    name: 'Persondataforordningen (GDPR)',
    source_url: 'https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:32016R0679',
    celex_number: '32016R0679',
  },
];

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
    setDrawColor: vi.fn(),
    line: vi.fn(),
    setFontSize: vi.fn(),
    setFont: vi.fn(),
    setTextColor: vi.fn(),
  } as unknown as PdfDocument;
}

describe('SourcesSection', () => {
  let section: SourcesSection;
  let mockDoc: PdfDocument;

  const mockReferences: Reference[] = [
    {
      idx: 1,
      display: 'Article 5 - Principles',
      chunk_text: 'Personal data shall be processed lawfully.',
      article: 'Article 5',
      corpus_id: 'gdpr',
      score: 0.95,
    },
    {
      idx: 2,
      display: 'Article 6 - Lawfulness',
      chunk_text: 'Processing shall be lawful.',
      article: 'Article 6',
      corpus_id: 'gdpr',
      score: 0.90,
    },
  ];

  beforeEach(() => {
    section = new SourcesSection();
    mockDoc = createMockDoc();
    setCorpusRegistry(testCorpora);
  });

  describe('render', () => {
    it('returns unchanged Y for empty references (SS-01)', () => {
      const result = section.render(mockDoc, [], 50);

      expect(result).toBe(50);
    });

    it('returns unchanged Y for null references (SS-02)', () => {
      const result = section.render(mockDoc, null as unknown as Reference[], 50);

      expect(result).toBe(50);
    });

    it('renders section header "Kilder" (SS-03)', () => {
      section.render(mockDoc, mockReferences, 50);

      expect(mockDoc.text).toHaveBeenCalledWith('Kilder', 20, expect.any(Number));
    });

    it('draws separator line (SS-04)', () => {
      section.render(mockDoc, mockReferences, 50);

      expect(mockDoc.setDrawColor).toHaveBeenCalledWith(180, 180, 180);
      expect(mockDoc.line).toHaveBeenCalled();
    });

    it('returns updated Y position (SS-05)', () => {
      const result = section.render(mockDoc, mockReferences, 50);

      expect(result).toBeGreaterThan(50);
    });

    it('renders citation markers (SS-06)', () => {
      section.render(mockDoc, mockReferences, 50);

      expect(mockDoc.text).toHaveBeenCalledWith(
        expect.stringContaining('[1]'),
        expect.any(Number),
        expect.any(Number)
      );
    });

    it('separates cited and uncited sources (SS-07)', () => {
      const answerText = 'According to [1], the data must be processed lawfully.';
      section.render(mockDoc, mockReferences, 50, answerText);

      // Should render group labels
      expect(mockDoc.text).toHaveBeenCalledWith(
        'Citeret i svaret',
        expect.any(Number),
        expect.any(Number)
      );
    });

    it('renders EUR-Lex URLs in blue (SS-08)', () => {
      section.render(mockDoc, mockReferences, 50);

      expect(mockDoc.setTextColor).toHaveBeenCalledWith(0, 102, 204);
    });

    it('adds new page when needed (SS-09)', () => {
      vi.mocked(mockDoc.needsNewPage).mockReturnValue(true);

      section.render(mockDoc, mockReferences, 50);

      expect(mockDoc.addPage).toHaveBeenCalled();
    });

    it('renders source excerpts (SS-10)', () => {
      section.render(mockDoc, mockReferences, 50);

      expect(mockDoc.splitTextToSize).toHaveBeenCalled();
    });
  });

  describe('estimateHeight', () => {
    it('returns 0 for empty references (SS-11)', () => {
      const height = section.estimateHeight(mockDoc, []);

      expect(height).toBe(0);
    });

    it('returns 0 for null references (SS-12)', () => {
      const height = section.estimateHeight(mockDoc, null as unknown as Reference[]);

      expect(height).toBe(0);
    });

    it('returns positive height for references (SS-13)', () => {
      const height = section.estimateHeight(mockDoc, mockReferences);

      expect(height).toBeGreaterThan(0);
    });

    it('increases with more references (SS-14)', () => {
      const heightOne = section.estimateHeight(mockDoc, [mockReferences[0]]);
      const heightTwo = section.estimateHeight(mockDoc, mockReferences);

      expect(heightTwo).toBeGreaterThan(heightOne);
    });
  });

  describe('link functionality', () => {
    let mockLinkRegistry: { has: ReturnType<typeof vi.fn>; get: ReturnType<typeof vi.fn>; register: ReturnType<typeof vi.fn> };
    let mockLinkRenderer: { renderLink: ReturnType<typeof vi.fn> };

    beforeEach(() => {
      mockLinkRegistry = {
        has: vi.fn().mockReturnValue(true),
        get: vi.fn().mockReturnValue({ pageNumber: 1, top: 70 }),
        register: vi.fn(),
      };
      mockLinkRenderer = {
        renderLink: vi.fn().mockReturnValue(50),
      };

      // Add link methods to mock doc
      (mockDoc as unknown as Record<string, unknown>).textWithLink = vi.fn();
      (mockDoc as unknown as Record<string, unknown>).getCurrentPageNumber = vi.fn().mockReturnValue(1);
      (mockDoc as unknown as Record<string, unknown>).getTextWidth = vi.fn().mockReturnValue(50);
    });

    it('registers source position in registry (SS-15)', () => {
      section.render(mockDoc, mockReferences, 50, '', mockLinkRegistry as unknown as import('../../LinkTargetRegistry').LinkTargetRegistry);

      expect(mockLinkRegistry.register).toHaveBeenCalledWith('source-[1]', 1, expect.any(Number));
      expect(mockLinkRegistry.register).toHaveBeenCalledWith('source-[2]', 1, expect.any(Number));
    });

    it('renders EUR-Lex link as clickable with LinkRenderer (SS-16)', () => {
      section.render(
        mockDoc,
        mockReferences,
        50,
        '',
        mockLinkRegistry as unknown as import('../../LinkTargetRegistry').LinkTargetRegistry,
        mockLinkRenderer as unknown as import('../../formatters/LinkRenderer').LinkRenderer
      );

      expect(mockLinkRenderer.renderLink).toHaveBeenCalled();
    });

    it('EUR-Lex link includes article anchor (SS-17)', () => {
      section.render(
        mockDoc,
        mockReferences,
        50,
        '',
        mockLinkRegistry as unknown as import('../../LinkTargetRegistry').LinkTargetRegistry,
        mockLinkRenderer as unknown as import('../../formatters/LinkRenderer').LinkRenderer
      );

      // Check that renderLink was called with URL containing article anchor
      const calls = mockLinkRenderer.renderLink.mock.calls;
      const urlCall = calls.find((call: unknown[]) => {
        const options = call[4] as { url?: string };
        return options?.url?.includes('#art_');
      });
      expect(urlCall).toBeDefined();
    });

    it('omits EUR-Lex link when no URL available (SS-18)', () => {
      const refsWithoutUrl: Reference[] = [
        {
          idx: 1,
          display: 'Custom Source',
          chunk_text: 'Some text',
          corpus_id: 'unknown-corpus',
        },
      ];

      section.render(
        mockDoc,
        refsWithoutUrl,
        50,
        '',
        mockLinkRegistry as unknown as import('../../LinkTargetRegistry').LinkTargetRegistry,
        mockLinkRenderer as unknown as import('../../formatters/LinkRenderer').LinkRenderer
      );

      // LinkRenderer should not be called for sources without URL
      expect(mockLinkRenderer.renderLink).not.toHaveBeenCalled();
    });
  });
});
