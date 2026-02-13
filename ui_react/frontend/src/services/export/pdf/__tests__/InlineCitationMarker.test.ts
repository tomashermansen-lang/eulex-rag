/**
 * Tests for InlineCitationMarker.
 *
 * TDD: These tests define how citation markers should be inserted into text.
 */

import { describe, it, expect } from 'vitest';
import { InlineCitationMarker } from '../formatters/InlineCitationMarker';
import type { Reference } from '../../../../types';

describe('InlineCitationMarker', () => {
  const marker = new InlineCitationMarker();

  const createRef = (idx: number): Reference => ({
    idx,
    display: `Artikel ${idx}`,
    chunk_text: 'Test chunk text',
  });

  describe('insertMarkers()', () => {
    it('appends single marker at end of text', () => {
      const text = 'Data controllers must keep records.';
      const refs = [createRef(1)];

      const result = marker.insertMarkers(text, refs);
      expect(result).toBe('Data controllers must keep records. [1]');
    });

    it('appends multiple markers at end', () => {
      const text = 'This regulation applies broadly.';
      const refs = [createRef(1), createRef(2), createRef(3)];

      const result = marker.insertMarkers(text, refs);
      expect(result).toBe('This regulation applies broadly. [1] [2] [3]');
    });

    it('preserves text when no references', () => {
      const text = 'Plain text without citations.';
      const result = marker.insertMarkers(text, []);
      expect(result).toBe('Plain text without citations.');
    });

    it('handles empty text', () => {
      const refs = [createRef(1)];
      const result = marker.insertMarkers('', refs);
      expect(result).toBe('[1]');
    });

    it('handles string idx values', () => {
      const ref: Reference = {
        idx: '1a',
        display: 'Artikel 5',
        chunk_text: 'Test',
      };
      const result = marker.insertMarkers('Text here.', [ref]);
      expect(result).toBe('Text here. [1a]');
    });

    it('preserves existing markdown formatting', () => {
      const text = '**Bold text** and *italic text*.';
      const refs = [createRef(1)];

      const result = marker.insertMarkers(text, refs);
      expect(result).toBe('**Bold text** and *italic text*. [1]');
    });

    it('handles text ending without punctuation', () => {
      const text = 'This is the text';
      const refs = [createRef(1)];

      const result = marker.insertMarkers(text, refs);
      expect(result).toBe('This is the text [1]');
    });

    it('handles multi-paragraph text', () => {
      const text = 'First paragraph.\n\nSecond paragraph.';
      const refs = [createRef(1), createRef(2)];

      const result = marker.insertMarkers(text, refs);
      expect(result).toBe('First paragraph.\n\nSecond paragraph. [1] [2]');
    });
  });

  describe('buildMarkerString()', () => {
    it('builds single marker string', () => {
      const refs = [createRef(1)];
      const result = marker.buildMarkerString(refs);
      expect(result).toBe('[1]');
    });

    it('builds multiple markers with spaces', () => {
      const refs = [createRef(1), createRef(2)];
      const result = marker.buildMarkerString(refs);
      expect(result).toBe('[1] [2]');
    });

    it('returns empty string for no references', () => {
      const result = marker.buildMarkerString([]);
      expect(result).toBe('');
    });

    it('handles mixed numeric and string indices', () => {
      const refs: Reference[] = [
        { idx: 1, display: 'A', chunk_text: '' },
        { idx: '2a', display: 'B', chunk_text: '' },
      ];
      const result = marker.buildMarkerString(refs);
      expect(result).toBe('[1] [2a]');
    });
  });
});
