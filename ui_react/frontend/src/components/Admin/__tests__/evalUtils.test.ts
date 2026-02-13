/**
 * Tests for shared eval utility functions and constants.
 *
 * Tests: TC1 (formatPassRate), TC2 (getPassRateColor), TC3 (formatDuration),
 * TC4 (formatTimestamp), TC5 (label constants).
 */

import { describe, it, expect } from 'vitest';
import {
  getPassRateColor,
  getPassRateColorHex,
  getPassRateTremorColor,
  formatDuration,
  formatLatency,
  formatTimestamp,
  EVAL_SCORER_LABELS,
  EVAL_SCORER_DESCRIPTIONS,
  EVAL_TEST_TYPE_LABELS,
  EVAL_TEST_TYPE_DESCRIPTIONS,
  RUN_MODE_LABELS,
  MODE_COLORS,
  DIFFICULTY_COLORS,
  CROSS_LAW_SCORER_LABELS,
} from '../evalUtils';

describe('getPassRateColor', () => {
  it('returns green for ≥95%', () => {
    expect(getPassRateColor(0.96)).toBe('text-green-500');
    expect(getPassRateColor(0.95)).toBe('text-green-500');
    expect(getPassRateColor(1.0)).toBe('text-green-500');
  });

  it('returns yellow for ≥80% and <95%', () => {
    expect(getPassRateColor(0.87)).toBe('text-yellow-500');
    expect(getPassRateColor(0.80)).toBe('text-yellow-500');
    expect(getPassRateColor(0.94)).toBe('text-yellow-500');
  });

  it('returns orange for ≥60% and <80%', () => {
    expect(getPassRateColor(0.65)).toBe('text-orange-500');
    expect(getPassRateColor(0.60)).toBe('text-orange-500');
    expect(getPassRateColor(0.79)).toBe('text-orange-500');
  });

  it('returns red for <60%', () => {
    expect(getPassRateColor(0.45)).toBe('text-red-500');
    expect(getPassRateColor(0.0)).toBe('text-red-500');
    expect(getPassRateColor(0.59)).toBe('text-red-500');
  });
});

describe('formatDuration', () => {
  it('formats seconds-only durations', () => {
    expect(formatDuration(45)).toBe('45s');
  });

  it('formats minutes and seconds', () => {
    expect(formatDuration(90)).toBe('1m 30s');
  });

  it('handles zero seconds', () => {
    expect(formatDuration(0)).toBe('0s');
  });

  it('handles long durations', () => {
    expect(formatDuration(630)).toBe('10m 30s');
  });

  it('rounds fractional seconds', () => {
    expect(formatDuration(45.7)).toBe('46s');
  });
});

describe('formatTimestamp', () => {
  it('formats to clean Danish style with day, month, hours:minutes', () => {
    // Use a fixed timezone-independent check
    const result = formatTimestamp('2025-01-30T08:38:00Z');
    // Result should contain the day number
    expect(result).toMatch(/\d+/);
    // Should contain a month abbreviation (jan, feb, etc.)
    expect(result).toMatch(/[a-z]{3}/);
    // Should contain time in HH:MM format
    expect(result).toMatch(/\d{2}:\d{2}/);
  });
});

describe('EVAL_SCORER_LABELS', () => {
  it('includes single-law scorer keys', () => {
    expect(EVAL_SCORER_LABELS['anchor_presence']).toBe('Retrieval');
    expect(EVAL_SCORER_LABELS['faithfulness']).toBe('Faithfulness');
    expect(EVAL_SCORER_LABELS['answer_relevancy']).toBe('Relevancy');
  });

  it('includes cross-law-only scorer keys', () => {
    expect(EVAL_SCORER_LABELS['corpus_coverage']).toBe('Coverage');
    expect(EVAL_SCORER_LABELS['synthesis_balance']).toBe('Balance');
    expect(EVAL_SCORER_LABELS['routing_precision']).toBe('Routing');
    expect(EVAL_SCORER_LABELS['comparison_completeness']).toBe('Comparison');
  });

  it('returns undefined for unknown keys (caller uses ?? fallback)', () => {
    expect(EVAL_SCORER_LABELS['unknown_key']).toBeUndefined();
  });
});

describe('EVAL_SCORER_DESCRIPTIONS', () => {
  it('includes descriptions for all scorer keys', () => {
    expect(EVAL_SCORER_DESCRIPTIONS['anchor_presence']).toBeDefined();
    expect(EVAL_SCORER_DESCRIPTIONS['corpus_coverage']).toBeDefined();
  });
});

describe('EVAL_TEST_TYPE_LABELS', () => {
  it('includes single-law test type keys', () => {
    expect(EVAL_TEST_TYPE_LABELS['retrieval']).toBe('Retrieval');
    expect(EVAL_TEST_TYPE_LABELS['faithfulness']).toBe('Faithfulness');
    expect(EVAL_TEST_TYPE_LABELS['relevancy']).toBe('Relevancy');
    expect(EVAL_TEST_TYPE_LABELS['abstention']).toBe('Abstention');
    expect(EVAL_TEST_TYPE_LABELS['robustness']).toBe('Robustness');
    expect(EVAL_TEST_TYPE_LABELS['multi_hop']).toBe('Multi-hop');
  });
});

describe('EVAL_TEST_TYPE_DESCRIPTIONS', () => {
  it('includes descriptions for all test types', () => {
    expect(EVAL_TEST_TYPE_DESCRIPTIONS['retrieval']).toBeDefined();
    expect(EVAL_TEST_TYPE_DESCRIPTIONS['faithfulness']).toBeDefined();
  });
});

describe('RUN_MODE_LABELS', () => {
  it('has all three run modes', () => {
    expect(RUN_MODE_LABELS['retrieval_only']).toBeDefined();
    expect(RUN_MODE_LABELS['full']).toBeDefined();
    expect(RUN_MODE_LABELS['full_with_judge']).toBeDefined();
  });

  it('each mode has label and description', () => {
    for (const mode of Object.values(RUN_MODE_LABELS)) {
      expect(mode.label).toBeDefined();
      expect(mode.description).toBeDefined();
    }
  });
});


// ─────────────────────────────────────────────────────────────────────────────
// C7: New utility functions and constants for metrics dashboard
// ─────────────────────────────────────────────────────────────────────────────

describe('getPassRateColorHex', () => {
  it('returns green hex for ≥95%', () => {
    expect(getPassRateColorHex(96)).toBe('#22c55e');
    expect(getPassRateColorHex(95)).toBe('#22c55e');
    expect(getPassRateColorHex(100)).toBe('#22c55e');
  });

  it('returns yellow hex for ≥80% and <95%', () => {
    expect(getPassRateColorHex(87)).toBe('#eab308');
    expect(getPassRateColorHex(80)).toBe('#eab308');
    expect(getPassRateColorHex(94)).toBe('#eab308');
  });

  it('returns orange hex for ≥60% and <80%', () => {
    expect(getPassRateColorHex(65)).toBe('#f97316');
    expect(getPassRateColorHex(60)).toBe('#f97316');
    expect(getPassRateColorHex(79)).toBe('#f97316');
  });

  it('returns red hex for <60%', () => {
    expect(getPassRateColorHex(45)).toBe('#ef4444');
    expect(getPassRateColorHex(0)).toBe('#ef4444');
    expect(getPassRateColorHex(59)).toBe('#ef4444');
  });
});

describe('getPassRateTremorColor', () => {
  it('returns emerald for ≥95%', () => {
    expect(getPassRateTremorColor(96)).toBe('emerald');
    expect(getPassRateTremorColor(95)).toBe('emerald');
  });

  it('returns yellow for ≥80% and <95%', () => {
    expect(getPassRateTremorColor(87)).toBe('yellow');
    expect(getPassRateTremorColor(80)).toBe('yellow');
  });

  it('returns orange for ≥60% and <80%', () => {
    expect(getPassRateTremorColor(65)).toBe('orange');
    expect(getPassRateTremorColor(60)).toBe('orange');
  });

  it('returns red for <60%', () => {
    expect(getPassRateTremorColor(45)).toBe('red');
    expect(getPassRateTremorColor(0)).toBe('red');
  });
});

describe('formatLatency', () => {
  it('formats sub-second as milliseconds', () => {
    expect(formatLatency(450)).toBe('450ms');
    expect(formatLatency(999)).toBe('999ms');
  });

  it('formats seconds with one decimal', () => {
    expect(formatLatency(1000)).toBe('1.0s');
    expect(formatLatency(3500)).toBe('3.5s');
    expect(formatLatency(12800)).toBe('12.8s');
  });

  it('formats minutes and seconds for ≥60s', () => {
    expect(formatLatency(60000)).toBe('1m 0s');
    expect(formatLatency(90000)).toBe('1m 30s');
    expect(formatLatency(150000)).toBe('2m 30s');
  });

  it('handles zero', () => {
    expect(formatLatency(0)).toBe('0ms');
  });
});

describe('MODE_COLORS', () => {
  it('has entries for all synthesis modes', () => {
    expect(MODE_COLORS['comparison']).toBeDefined();
    expect(MODE_COLORS['discovery']).toBeDefined();
    expect(MODE_COLORS['routing']).toBeDefined();
    expect(MODE_COLORS['aggregation']).toBeDefined();
  });

  it('each entry has hex and tremor fields', () => {
    for (const entry of Object.values(MODE_COLORS)) {
      expect(entry.hex).toMatch(/^#[0-9a-f]{6}$/);
      expect(typeof entry.tremor).toBe('string');
    }
  });
});

describe('DIFFICULTY_COLORS', () => {
  it('has entries for all difficulty levels', () => {
    expect(DIFFICULTY_COLORS['easy']).toBeDefined();
    expect(DIFFICULTY_COLORS['medium']).toBeDefined();
    expect(DIFFICULTY_COLORS['hard']).toBeDefined();
  });

  it('each entry has hex and tremor fields', () => {
    for (const entry of Object.values(DIFFICULTY_COLORS)) {
      expect(entry.hex).toMatch(/^#[0-9a-f]{6}$/);
      expect(typeof entry.tremor).toBe('string');
    }
  });
});

describe('CROSS_LAW_SCORER_LABELS', () => {
  it('has labels for cross-law scorers', () => {
    expect(CROSS_LAW_SCORER_LABELS['corpus_coverage']).toBe('Corpus Coverage');
    expect(CROSS_LAW_SCORER_LABELS['comparison_completeness']).toBe('Comparison Completeness');
    expect(CROSS_LAW_SCORER_LABELS['routing_precision']).toBe('Routing Precision');
  });
});
