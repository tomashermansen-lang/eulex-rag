/**
 * Corpus Metadata Registry for PDF export.
 *
 * Provides corpus metadata for legal citations. Data is populated dynamically
 * from the API to stay in sync with corpora.json (single source of truth).
 *
 * Single Responsibility: Manage corpus metadata for PDF citations.
 */

import type { CorpusInfo } from '../../../../types';

export interface CorpusMetadata {
  /** Corpus identifier (e.g., "gdpr-reg-2016-679", "ai-act") */
  corpusId: string;
  /** Display name in Danish for UI */
  displayName: string;
  /** Full official legal title for citations */
  fullname: string | null;
  /** CELEX number for EUR-Lex reference */
  celexNumber: string | null;
  /** Direct URL to EUR-Lex document */
  eurLexUrl: string | null;
}

/**
 * Registry for corpus metadata.
 * Populated from API data at runtime.
 */
const corpusRegistry = new Map<string, CorpusMetadata>();

/**
 * Build EUR-Lex URL from CELEX number.
 * Uses HTML format which supports anchor links for articles/recitals.
 */
function buildEurLexUrl(celexNumber: string | null | undefined): string | null {
  if (!celexNumber) return null;
  return `https://eur-lex.europa.eu/legal-content/DA/TXT/HTML/?uri=CELEX:${celexNumber}`;
}

/**
 * Populate the corpus registry from API data.
 * Call this when corpora data is loaded from the API.
 *
 * @param corpora - Array of corpus info from /api/corpora
 */
export function setCorpusRegistry(corpora: CorpusInfo[]): void {
  corpusRegistry.clear();

  for (const corpus of corpora) {
    const metadata: CorpusMetadata = {
      corpusId: corpus.id,
      displayName: corpus.name,
      fullname: corpus.fullname ?? null,
      celexNumber: corpus.celex_number ?? null,
      // Use source_url from corpus data (same as web UI) for anchor support
      eurLexUrl: corpus.source_url ?? buildEurLexUrl(corpus.celex_number),
    };
    corpusRegistry.set(corpus.id.toLowerCase(), metadata);
  }
}

/**
 * Get metadata for a corpus by ID.
 *
 * @param corpusId - The corpus identifier (e.g., "gdpr", "ai-act")
 * @returns Corpus metadata or undefined if not found
 */
export function getCorpusMetadata(corpusId: string): CorpusMetadata | undefined {
  // Normalize corpus_id (handle underscores vs hyphens)
  const normalizedId = corpusId.toLowerCase().replace(/_/g, '-');
  return corpusRegistry.get(normalizedId);
}

/**
 * Get the corpus ID for inline use.
 *
 * @param corpusId - The corpus identifier
 * @returns Corpus ID or the input if not found
 */
export function getCorpusId(corpusId: string): string {
  const metadata = getCorpusMetadata(corpusId);
  return metadata?.corpusId ?? corpusId;
}

/**
 * Get the full official legal title for a corpus.
 *
 * @param corpusId - The corpus identifier
 * @returns Full legal title or displayName if fullname not available
 */
export function getCorpusFullname(corpusId: string): string {
  const metadata = getCorpusMetadata(corpusId);
  return metadata?.fullname ?? metadata?.displayName ?? corpusId;
}

/**
 * Get the full display name for a corpus.
 *
 * @param corpusId - The corpus identifier
 * @returns Full display name or the corpusId if not found
 */
export function getCorpusDisplayName(corpusId: string): string {
  const metadata = getCorpusMetadata(corpusId);
  return metadata?.displayName ?? corpusId;
}

/**
 * For backward compatibility and tests.
 * Returns all registered corpora as a record.
 */
export function getCorpusMetadataRecord(): Record<string, CorpusMetadata> {
  const record: Record<string, CorpusMetadata> = {};
  for (const [id, metadata] of corpusRegistry) {
    record[id] = metadata;
  }
  return record;
}

/**
 * CORPUS_METADATA proxy for backward compatibility.
 * Provides access to registered corpora as if they were a static object.
 */
export const CORPUS_METADATA: Record<string, CorpusMetadata> = new Proxy(
  {} as Record<string, CorpusMetadata>,
  {
    get(_, prop: string) {
      return corpusRegistry.get(prop.toLowerCase());
    },
    ownKeys() {
      return Array.from(corpusRegistry.keys());
    },
    getOwnPropertyDescriptor(_, prop: string) {
      const key = prop.toLowerCase();
      if (corpusRegistry.has(key)) {
        return {
          enumerable: true,
          configurable: true,
          value: corpusRegistry.get(key),
        };
      }
      return undefined;
    },
  }
);
