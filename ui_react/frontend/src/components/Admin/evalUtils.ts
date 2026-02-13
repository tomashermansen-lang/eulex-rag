/**
 * Shared eval formatting utilities and label constants.
 *
 * Single source of truth for functions and constants used by both
 * EvalDashboard (single-law) and CrossLawPanel (cross-law).
 */

import type { ReactElement } from 'react';
import { createElement } from 'react';

// --- Pass-rate colour thresholds (4-tier scale) ---

export function getPassRateColor(rate: number): string {
  const pct = Math.round(rate * 100);
  if (pct >= 95) return 'text-green-500';
  if (pct >= 80) return 'text-yellow-500';
  if (pct >= 60) return 'text-orange-500';
  return 'text-red-500';
}

/** Hex colour for Recharts <Cell> fill (rate is already a percentage 0-100). */
export function getPassRateColorHex(rate: number): string {
  if (rate >= 95) return '#22c55e';
  if (rate >= 80) return '#eab308';
  if (rate >= 60) return '#f97316';
  return '#ef4444';
}

/** Tremor colour name for BarList/SparkAreaChart color prop (rate is 0-100). */
export function getPassRateTremorColor(rate: number): string {
  if (rate >= 95) return 'emerald';
  if (rate >= 80) return 'yellow';
  if (rate >= 60) return 'orange';
  return 'red';
}

export function formatPassRate(rate: number): ReactElement {
  const pct = Math.round(rate * 100);
  const color = getPassRateColor(rate);
  return createElement('span', { className: color }, `${pct}%`);
}

// --- Duration formatting ---

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

// --- Timestamp formatting (clean Danish: "30 jan 08:38") ---

export function formatTimestamp(ts: string): string {
  const date = new Date(ts);
  const day = date.getDate();
  const month = date.toLocaleString('da-DK', { month: 'short' }).replace('.', '');
  const hours = date.getHours().toString().padStart(2, '0');
  const mins = date.getMinutes().toString().padStart(2, '0');
  return `${day} ${month} ${hours}:${mins}`;
}

// --- Scorer labels (merged single-law + cross-law) ---

export const EVAL_SCORER_LABELS: Record<string, string> = {
  // Single-law scorers
  anchor_presence: 'Retrieval',
  faithfulness: 'Faithfulness',
  answer_relevancy: 'Relevancy',
  contract_compliance: 'Contract',
  pipeline_breakdown: 'Pipeline',
  escalation: 'Escalation',
  abstention: 'Abstention',
  // Cross-law-only scorers
  corpus_coverage: 'Coverage',
  synthesis_balance: 'Balance',
  routing_precision: 'Routing',
  comparison_completeness: 'Comparison',
};

export const EVAL_SCORER_DESCRIPTIONS: Record<string, string> = {
  // Single-law
  anchor_presence: 'Finder systemet de rigtige dokumenter og artikler?',
  faithfulness: 'Er svaret forankret i kildematerialet?',
  answer_relevancy: 'Besvarer svaret det stillede spørgsmål?',
  contract_compliance: 'Overholder svaret kontraktkrav (min/max citater)?',
  pipeline_breakdown: 'Er hele pipeline-flowet gennemført korrekt?',
  escalation: 'Blev spørgsmålet eskaleret til en stærkere model?',
  abstention: 'Afstår systemet korrekt når det bør?',
  // Cross-law
  corpus_coverage: 'Dækker svaret alle forventede love?',
  synthesis_balance: 'Er citater balanceret mellem love?',
  routing_precision: 'Identificeres de rigtige love?',
  comparison_completeness: 'Er alle sammenlignede love dækket?',
};

// --- Test type labels (merged single-law + cross-law tag labels) ---

export const EVAL_TEST_TYPE_LABELS: Record<string, string> = {
  // Single-law test types
  retrieval: 'Retrieval',
  faithfulness: 'Faithfulness',
  relevancy: 'Relevancy',
  abstention: 'Abstention',
  robustness: 'Robustness',
  multi_hop: 'Multi-hop',
  // Cross-law scorer keys (used as tag labels)
  anchor_presence: 'Retrieval',
  answer_relevancy: 'Relevancy',
  corpus_coverage: 'Coverage',
  synthesis_balance: 'Balance',
  routing_precision: 'Routing',
  comparison_completeness: 'Comparison',
  discovery: 'Discovery',
};

export const EVAL_TEST_TYPE_DESCRIPTIONS: Record<string, string> = {
  retrieval: 'Finder systemet de rigtige dokumenter?',
  faithfulness: 'Er svaret forankret i kildematerialet?',
  relevancy: 'Besvarer svaret det stillede spørgsmål?',
  abstention: 'Afstår systemet når det bør?',
  robustness: 'Håndterer systemet variationer?',
  multi_hop: 'Kan systemet syntetisere fra flere kilder?',
};

// --- Run mode labels ---

export type RunMode = 'retrieval_only' | 'full' | 'full_with_judge';

export const RUN_MODE_LABELS: Record<RunMode, { label: string; description: string }> = {
  retrieval_only: { label: 'Kun retrieval', description: 'Hurtig test af dokumentsøgning' },
  full: { label: 'Full', description: 'Komplet test med LLM-svar' },
  full_with_judge: { label: 'Full + Judge', description: 'Komplet test med LLM-evaluering' },
};

// --- Latency formatting (milliseconds input) ---

/** Format milliseconds as human-readable latency: <1s→"XXXms", ≥1s→"X.Xs", ≥60s→"Xm Xs". */
export function formatLatency(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60000);
  const secs = Math.round((ms % 60000) / 1000);
  return `${mins}m ${secs}s`;
}

// --- Synthesis mode colour palette (matching CrossLawPanel badges) ---

export const MODE_COLORS: Record<string, { hex: string; tremor: string }> = {
  comparison:  { hex: '#3b82f6', tremor: 'blue' },
  discovery:   { hex: '#8b5cf6', tremor: 'violet' },
  routing:     { hex: '#22c55e', tremor: 'emerald' },
  aggregation: { hex: '#a855f7', tremor: 'purple' },
};

export const MODE_DESCRIPTIONS: Record<string, string> = {
  comparison: 'Sammenligner bestemmelser på tværs af love',
  discovery: 'Finder relevante bestemmelser i ukendt lovgivning',
  routing: 'Identificerer hvilke love der er relevante for spørgsmålet',
  aggregation: 'Samler information fra flere love til ét svar',
};

// --- Difficulty colour palette (matching CrossLawPanel badges) ---

export const DIFFICULTY_COLORS: Record<string, { hex: string; tremor: string }> = {
  easy:   { hex: '#22c55e', tremor: 'emerald' },
  medium: { hex: '#eab308', tremor: 'yellow' },
  hard:   { hex: '#ef4444', tremor: 'red' },
};

export const DIFFICULTY_DESCRIPTIONS: Record<string, string> = {
  easy: 'Simple spørgsmål med direkte svar',
  medium: 'Moderate spørgsmål der kræver sammenhæng',
  hard: 'Komplekse spørgsmål der kræver dyb analyse',
};

// --- Cross-law scorer display labels (verbose names for metrics dashboard) ---

export const CROSS_LAW_SCORER_LABELS: Record<string, string> = {
  corpus_coverage: 'Corpus Coverage',
  synthesis_balance: 'Synthesis Balance',
  cross_reference_accuracy: 'Cross-Reference',
  comparison_completeness: 'Comparison Completeness',
  routing_precision: 'Routing Precision',
};
