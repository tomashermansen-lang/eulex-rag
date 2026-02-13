/**
 * Tests for LinkTargetRegistry.
 *
 * TDD: These tests define link target position storage for PDF internal links.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { LinkTargetRegistry } from '../LinkTargetRegistry';

describe('LinkTargetRegistry', () => {
  let registry: LinkTargetRegistry;

  beforeEach(() => {
    registry = new LinkTargetRegistry();
  });

  describe('register()', () => {
    it('stores position retrievable via get() (LTR-01)', () => {
      registry.register('source-[1]', 1, 100);

      const target = registry.get('source-[1]');
      expect(target).toEqual({ pageNumber: 1, top: 100 });
    });

    it('stores multiple positions independently (LTR-01b)', () => {
      registry.register('source-[1]', 1, 100);
      registry.register('source-[2]', 2, 50);

      expect(registry.get('source-[1]')).toEqual({ pageNumber: 1, top: 100 });
      expect(registry.get('source-[2]')).toEqual({ pageNumber: 2, top: 50 });
    });

    it('overwrites existing position for same id (LTR-01c)', () => {
      registry.register('source-[1]', 1, 100);
      registry.register('source-[1]', 2, 200);

      expect(registry.get('source-[1]')).toEqual({ pageNumber: 2, top: 200 });
    });
  });

  describe('get()', () => {
    it('returns undefined for unknown id (LTR-02)', () => {
      const target = registry.get('unknown');

      expect(target).toBeUndefined();
    });
  });

  describe('has()', () => {
    it('returns true for registered id (LTR-03a)', () => {
      registry.register('source-[1]', 1, 100);

      expect(registry.has('source-[1]')).toBe(true);
    });

    it('returns false for unknown id (LTR-03b)', () => {
      expect(registry.has('unknown')).toBe(false);
    });
  });

  describe('clear()', () => {
    it('removes all entries (LTR-04)', () => {
      registry.register('source-[1]', 1, 100);
      registry.register('source-[2]', 2, 50);

      registry.clear();

      expect(registry.has('source-[1]')).toBe(false);
      expect(registry.has('source-[2]')).toBe(false);
    });
  });
});
