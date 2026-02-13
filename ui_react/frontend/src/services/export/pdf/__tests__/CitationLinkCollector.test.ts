/**
 * Tests for CitationLinkCollector.
 *
 * TDD: These tests define citation position tracking for deferred internal links.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { CitationLinkCollector } from '../CitationLinkCollector';

describe('CitationLinkCollector', () => {
  let collector: CitationLinkCollector;

  beforeEach(() => {
    collector = new CitationLinkCollector();
  });

  describe('record()', () => {
    it('stores citation position (CLC-01)', () => {
      collector.record('[1]', 1, 20, 50, 10, 4);

      const positions = collector.getAll();
      expect(positions).toHaveLength(1);
      expect(positions[0]).toEqual({
        marker: '[1]',
        pageNumber: 1,
        x: 20,
        y: 50,
        width: 10,
        height: 4,
      });
    });

    it('stores multiple citations (CLC-02)', () => {
      collector.record('[1]', 1, 20, 50, 10, 4);
      collector.record('[2]', 1, 30, 50, 10, 4);
      collector.record('[1]', 2, 20, 100, 10, 4);

      expect(collector.getAll()).toHaveLength(3);
    });
  });

  describe('getAll()', () => {
    it('returns empty array when no citations recorded (CLC-03)', () => {
      expect(collector.getAll()).toEqual([]);
    });

    it('returns all recorded citations (CLC-04)', () => {
      collector.record('[1]', 1, 20, 50, 10, 4);
      collector.record('[2]', 1, 40, 50, 10, 4);

      const all = collector.getAll();
      expect(all).toHaveLength(2);
      expect(all[0].marker).toBe('[1]');
      expect(all[1].marker).toBe('[2]');
    });
  });

  describe('clear()', () => {
    it('removes all recorded citations (CLC-05)', () => {
      collector.record('[1]', 1, 20, 50, 10, 4);
      collector.record('[2]', 1, 30, 50, 10, 4);

      collector.clear();

      expect(collector.getAll()).toEqual([]);
    });
  });

  describe('getByMarker()', () => {
    it('returns all positions for a specific marker (CLC-06)', () => {
      collector.record('[1]', 1, 20, 50, 10, 4);
      collector.record('[2]', 1, 30, 50, 10, 4);
      collector.record('[1]', 2, 20, 100, 10, 4);

      const positions = collector.getByMarker('[1]');
      expect(positions).toHaveLength(2);
      expect(positions[0].pageNumber).toBe(1);
      expect(positions[1].pageNumber).toBe(2);
    });

    it('returns empty array for unknown marker (CLC-07)', () => {
      collector.record('[1]', 1, 20, 50, 10, 4);

      expect(collector.getByMarker('[99]')).toEqual([]);
    });
  });
});
