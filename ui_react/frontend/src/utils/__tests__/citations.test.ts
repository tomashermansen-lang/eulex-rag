/**
 * Tests for citation parsing and transformation utilities.
 *
 * TDD: Tests written to verify citation-related text processing.
 */

import { describe, it, expect } from 'vitest';
import {
  parseCitations,
  hasCitations,
  getRefAnchorId,
  buildSourceUrl,
} from '../citations';

describe('parseCitations', () => {
  it('extracts single citation (CIT-01)', () => {
    expect(parseCitations('See [1].')).toEqual([1]);
  });

  it('extracts multiple citations (CIT-02)', () => {
    expect(parseCitations('See [1] and [2].')).toEqual([1, 2]);
  });

  it('returns unique citations sorted (CIT-03)', () => {
    expect(parseCitations('[3] then [1] and [3] again.')).toEqual([1, 3]);
  });

  it('returns empty array for no citations (CIT-04)', () => {
    expect(parseCitations('No citations here.')).toEqual([]);
  });

  it('returns empty array for empty string', () => {
    expect(parseCitations('')).toEqual([]);
  });

  it('returns empty array for null/undefined', () => {
    expect(parseCitations(null as unknown as string)).toEqual([]);
    expect(parseCitations(undefined as unknown as string)).toEqual([]);
  });

  it('handles multi-digit citations', () => {
    expect(parseCitations('[12] and [123]')).toEqual([12, 123]);
  });

  it('ignores non-numeric brackets', () => {
    expect(parseCitations('[a] and [1]')).toEqual([1]);
  });

  it('handles citations in markdown text', () => {
    const text = 'According to **Article 5** [1], the principles [2] apply.';
    expect(parseCitations(text)).toEqual([1, 2]);
  });
});

describe('hasCitations', () => {
  it('returns true when text has citations (CIT-05)', () => {
    expect(hasCitations('See [1].')).toBe(true);
  });

  it('returns false when text has no citations (CIT-06)', () => {
    expect(hasCitations('No citations here.')).toBe(false);
  });

  it('returns false for empty string', () => {
    expect(hasCitations('')).toBe(false);
  });

  it('returns true for multiple citations', () => {
    expect(hasCitations('[1] and [2]')).toBe(true);
  });

  it('handles brackets without numbers', () => {
    expect(hasCitations('[a] and [b]')).toBe(false);
  });
});

describe('getRefAnchorId', () => {
  it('generates anchor with messageId (CIT-07)', () => {
    expect(getRefAnchorId(1, 'msg123')).toBe('ref-msg123-1');
  });

  it('generates anchor without messageId (CIT-08)', () => {
    expect(getRefAnchorId(1)).toBe('ref-1');
  });

  it('handles string idx', () => {
    expect(getRefAnchorId('5', 'msg456')).toBe('ref-msg456-5');
  });

  it('handles empty messageId', () => {
    expect(getRefAnchorId(3, '')).toBe('ref-3');
  });

  it('handles undefined messageId', () => {
    expect(getRefAnchorId(2, undefined)).toBe('ref-2');
  });
});

describe('buildSourceUrl', () => {
  const baseUrl = 'https://eur-lex.europa.eu/legal-content/DA/TXT/?uri=CELEX:32016R0679';

  it('adds article anchor (CIT-09)', () => {
    const result = buildSourceUrl(baseUrl, { article: '5' });
    expect(result).toBe(`${baseUrl}#art_5`);
  });

  it('adds recital anchor (CIT-10)', () => {
    const result = buildSourceUrl(baseUrl, { recital: '26' });
    expect(result).toBe(`${baseUrl}#rct_26`);
  });

  it('adds annex anchor (CIT-11)', () => {
    const result = buildSourceUrl(baseUrl, { annex: 'I' });
    expect(result).toBe(`${baseUrl}#anx_I`);
  });

  it('prioritizes recital over article (CIT-12)', () => {
    const result = buildSourceUrl(baseUrl, { recital: '10', article: '5' });
    expect(result).toBe(`${baseUrl}#rct_10`);
  });

  it('prioritizes article over annex', () => {
    const result = buildSourceUrl(baseUrl, { article: '5', annex: 'I' });
    expect(result).toBe(`${baseUrl}#art_5`);
  });

  it('returns base URL when no anchor fields (CIT-13)', () => {
    const result = buildSourceUrl(baseUrl, {});
    expect(result).toBe(baseUrl);
  });

  it('returns empty string for empty baseUrl', () => {
    expect(buildSourceUrl('', { article: '5' })).toBe('');
  });

  it('handles complex article identifiers', () => {
    const result = buildSourceUrl(baseUrl, { article: '5(1)(a)' });
    expect(result).toBe(`${baseUrl}#art_5(1)(a)`);
  });
});
