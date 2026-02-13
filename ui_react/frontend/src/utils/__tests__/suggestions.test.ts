/**
 * Tests for suggestion extraction utilities.
 *
 * TDD: Tests written to cover question extraction from answers.
 */

import { describe, it, expect } from 'vitest';
import { extractSuggestedQuestions, hasSuggestedQuestions } from '../suggestions';

describe('extractSuggestedQuestions', () => {
  describe('empty or invalid input', () => {
    it('returns empty array for empty string', () => {
      expect(extractSuggestedQuestions('')).toEqual([]);
    });

    it('returns empty array for undefined-like input', () => {
      expect(extractSuggestedQuestions(undefined as unknown as string)).toEqual([]);
    });

    it('returns empty array when no markers present', () => {
      const answer = 'This is a regular answer without any suggestion markers.';
      expect(extractSuggestedQuestions(answer)).toEqual([]);
    });
  });

  describe('marker: **Prøv at spørge**', () => {
    it('extracts questions with hyphen bullet', () => {
      const answer = `
        Here is the answer.

        **Prøv at spørge**
        - "Hvad er GDPR?"
        - "Hvordan virker det?"
      `;
      expect(extractSuggestedQuestions(answer)).toEqual([
        'Hvad er GDPR?',
        'Hvordan virker det?',
      ]);
    });

    it('extracts questions with dot bullet', () => {
      const answer = `
        Answer text.

        **Prøv at spørge**
        · "Første spørgsmål?"
        · "Andet spørgsmål?"
      `;
      expect(extractSuggestedQuestions(answer)).toEqual([
        'Første spørgsmål?',
        'Andet spørgsmål?',
      ]);
    });

    it('extracts questions with bullet point', () => {
      const answer = `
        Answer.

        **Prøv at spørge**
        • "Spørgsmål med bullet?"
      `;
      expect(extractSuggestedQuestions(answer)).toEqual(['Spørgsmål med bullet?']);
    });

    it('extracts questions with asterisk bullet', () => {
      const answer = `
        Answer.

        **Prøv at spørge**
        * "Asterisk spørgsmål?"
      `;
      expect(extractSuggestedQuestions(answer)).toEqual(['Asterisk spørgsmål?']);
    });
  });

  describe('marker: Du kan prøve at spørge om:', () => {
    it('extracts questions', () => {
      const answer = `
        The answer is here.

        Du kan prøve at spørge om:
        - "Hvad betyder artikel 1?"
        - "Hvad er konsekvenserne?"
      `;
      expect(extractSuggestedQuestions(answer)).toEqual([
        'Hvad betyder artikel 1?',
        'Hvad er konsekvenserne?',
      ]);
    });
  });

  describe('marker: Du kan prøve at spørge:', () => {
    it('extracts questions', () => {
      const answer = `
        Main answer.

        Du kan prøve at spørge:
        - "Et nyt spørgsmål?"
      `;
      expect(extractSuggestedQuestions(answer)).toEqual(['Et nyt spørgsmål?']);
    });
  });

  describe('marker: Prøv at spørge om:', () => {
    it('extracts questions', () => {
      const answer = `
        Info here.

        Prøv at spørge om:
        - "Spørgsmål 1?"
        - "Spørgsmål 2?"
      `;
      expect(extractSuggestedQuestions(answer)).toEqual([
        'Spørgsmål 1?',
        'Spørgsmål 2?',
      ]);
    });
  });

  describe('edge cases', () => {
    it('returns empty array when marker is at end with no content', () => {
      const answer = 'Answer text. **Prøv at spørge**';
      expect(extractSuggestedQuestions(answer)).toEqual([]);
    });

    it('handles marker with no quoted questions', () => {
      const answer = `
        Answer.

        **Prøv at spørge**
        No actual quoted questions here.
      `;
      expect(extractSuggestedQuestions(answer)).toEqual([]);
    });

    it('ignores questions before the marker', () => {
      const answer = `
        - "This should be ignored?"

        **Prøv at spørge**
        - "This should be included?"
      `;
      expect(extractSuggestedQuestions(answer)).toEqual(['This should be included?']);
    });

    it('handles questions with nested quotes', () => {
      const answer = `
        **Prøv at spørge**
        - "Hvad betyder 'consent'?"
      `;
      // The regex captures up to the first closing quote
      expect(extractSuggestedQuestions(answer)).toEqual(["Hvad betyder 'consent'?"]);
    });
  });
});

describe('hasSuggestedQuestions', () => {
  it('returns true when questions exist', () => {
    const answer = `
      Answer.

      **Prøv at spørge**
      - "Spørgsmål?"
    `;
    expect(hasSuggestedQuestions(answer)).toBe(true);
  });

  it('returns false for empty string', () => {
    expect(hasSuggestedQuestions('')).toBe(false);
  });

  it('returns false when no markers present', () => {
    expect(hasSuggestedQuestions('Regular answer without suggestions.')).toBe(false);
  });

  it('returns false when marker present but no questions', () => {
    const answer = `
      Answer.

      **Prøv at spørge**
      No questions here.
    `;
    expect(hasSuggestedQuestions(answer)).toBe(false);
  });
});
