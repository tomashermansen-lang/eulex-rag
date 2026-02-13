/**
 * Tests for TextFormatter.
 *
 * TDD: These tests define text formatting requirements for PDF output.
 */

import { describe, it, expect } from 'vitest';
import { TextFormatter } from '../formatters/TextFormatter';

describe('TextFormatter', () => {
  const formatter = new TextFormatter();

  describe('replaceDanishChars()', () => {
    it('preserves lowercase æ', () => {
      expect(formatter.replaceDanishChars('æble')).toBe('æble');
    });

    it('preserves lowercase ø', () => {
      expect(formatter.replaceDanishChars('rød')).toBe('rød');
    });

    it('preserves lowercase å', () => {
      expect(formatter.replaceDanishChars('gå')).toBe('gå');
    });

    it('preserves uppercase Æ', () => {
      expect(formatter.replaceDanishChars('Æble')).toBe('Æble');
    });

    it('preserves uppercase Ø', () => {
      expect(formatter.replaceDanishChars('Øst')).toBe('Øst');
    });

    it('preserves uppercase Å', () => {
      expect(formatter.replaceDanishChars('Århus')).toBe('Århus');
    });

    it('preserves multiple Danish characters', () => {
      expect(formatter.replaceDanishChars('Spørgsmål')).toBe('Spørgsmål');
    });

    it('preserves non-Danish characters', () => {
      expect(formatter.replaceDanishChars('Hello World')).toBe('Hello World');
    });
  });

  describe('stripMarkdown()', () => {
    it('removes headers', () => {
      expect(formatter.stripMarkdown('### Header')).toBe('Header');
    });

    it('removes bold markers', () => {
      expect(formatter.stripMarkdown('**bold text**')).toBe('bold text');
    });

    it('removes underscore bold markers', () => {
      expect(formatter.stripMarkdown('__bold text__')).toBe('bold text');
    });

    it('removes italic markers', () => {
      expect(formatter.stripMarkdown('*italic text*')).toBe('italic text');
    });

    it('removes underscore italic markers', () => {
      expect(formatter.stripMarkdown('_italic text_')).toBe('italic text');
    });

    it('removes inline code markers', () => {
      expect(formatter.stripMarkdown('`code`')).toBe('code');
    });

    it('handles nested formatting', () => {
      expect(formatter.stripMarkdown('**bold and *italic***')).toContain('bold');
    });

    it('cleans up excessive newlines', () => {
      expect(formatter.stripMarkdown('A\n\n\n\nB')).toBe('A\n\nB');
    });

    it('preserves regular text', () => {
      expect(formatter.stripMarkdown('Regular text.')).toBe('Regular text.');
    });
  });

  describe('formatForPdf()', () => {
    it('strips markdown while preserving Danish characters', () => {
      const input = '### Spørgsmål\n\n**Test** text';
      const result = formatter.formatForPdf(input);

      expect(result).toContain('Spørgsmål');
      expect(result).not.toContain('###');
      expect(result).not.toContain('**');
    });

    it('handles empty string', () => {
      expect(formatter.formatForPdf('')).toBe('');
    });
  });
});
