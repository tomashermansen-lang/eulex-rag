/**
 * Typography and styling constants for professional legal PDF documents.
 *
 * Based on:
 * - Typography for Lawyers (Matthew Butterick)
 * - WCAG 2.1 accessibility guidelines
 * - Legal document formatting standards
 *
 * Single Responsibility: Define visual styling for legal documents.
 */

import type { DocumentStyles, RgbColor } from '../../types';

/**
 * Calculate WCAG contrast ratio between two colors.
 * Formula: (L1 + 0.05) / (L2 + 0.05) where L1 is lighter.
 */
export function calculateContrastRatio(foreground: RgbColor, background: RgbColor): number {
  const getLuminance = (color: RgbColor): number => {
    const [r, g, b] = [color.r, color.g, color.b].map((c) => {
      const sRGB = c / 255;
      return sRGB <= 0.03928
        ? sRGB / 12.92
        : Math.pow((sRGB + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };

  const l1 = getLuminance(foreground);
  const l2 = getLuminance(background);

  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);

  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Professional legal document styles.
 *
 * Typography choices:
 * - Times: Classic legal serif font, excellent readability
 * - 12pt body: Standard legal minimum, meets accessibility
 * - 10pt footnotes: Traditional, meets WCAG minimum
 * - 1.5 line height: Optimal readability for dense text
 */
export const LegalDocumentStyles: DocumentStyles = {
  body: {
    fontSize: 12,
    fontFamily: 'helvetica',
    fontWeight: 'normal',
    color: { r: 0, g: 0, b: 0 },
    lineHeight: 1.5,
  },

  heading: {
    fontSize: 14,
    fontFamily: 'helvetica',
    fontWeight: 'bold',
    color: { r: 0, g: 0, b: 0 },
    lineHeight: 1.3,
  },

  subheading: {
    fontSize: 13,  // Slightly larger than body (12pt) for visual hierarchy
    fontFamily: 'helvetica',
    fontWeight: 'bold',
    color: { r: 0, g: 0, b: 0 },
    lineHeight: 1.4,
  },

  footnote: {
    fontSize: 10,
    fontFamily: 'helvetica',
    fontWeight: 'normal',
    color: { r: 0, g: 0, b: 0 },
    lineHeight: 1.4,
  },

  sourceExcerpt: {
    fontSize: 9,
    fontFamily: 'helvetica',
    fontWeight: 'normal',
    // Gray that meets WCAG AA (4.5:1 contrast on white)
    // RGB(80,80,80) has contrast ratio ~5.9:1
    color: { r: 80, g: 80, b: 80 },
    lineHeight: 1.3,
  },

  page: {
    format: 'a4',
    orientation: 'portrait',
    margin: 25, // 25mm ~ 1 inch, standard legal margin
  },

  spacing: {
    section: 12,      // Between major sections
    paragraph: 6,     // Between paragraphs
    line: 5,          // Base line spacing in mm
    sourceItem: 8,    // Between source citations
  },
};
