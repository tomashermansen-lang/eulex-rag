/**
 * Citation Link Collector for PDF export.
 *
 * Tracks positions of citation markers during answer rendering,
 * enabling deferred creation of internal links after sources are rendered.
 *
 * Single Responsibility: Collect citation positions for deferred linking.
 */

/**
 * Position data for a rendered citation marker.
 */
export interface CitationPosition {
  /** The citation marker text, e.g., "[1]" */
  marker: string;
  /** Page number where citation is rendered (1-indexed) */
  pageNumber: number;
  /** X position in mm */
  x: number;
  /** Y position in mm (baseline) */
  y: number;
  /** Width of the citation text in mm */
  width: number;
  /** Height of the citation text in mm */
  height: number;
}

/**
 * Collects citation positions during rendering for deferred link creation.
 */
export class CitationLinkCollector {
  private positions: CitationPosition[] = [];

  /**
   * Record a citation position.
   *
   * @param marker - The citation marker, e.g., "[1]"
   * @param pageNumber - Current page number (1-indexed)
   * @param x - X position in mm
   * @param y - Y position in mm
   * @param width - Width of text in mm
   * @param height - Height of text in mm
   */
  record(
    marker: string,
    pageNumber: number,
    x: number,
    y: number,
    width: number,
    height: number
  ): void {
    this.positions.push({ marker, pageNumber, x, y, width, height });
  }

  /**
   * Get all recorded citation positions.
   */
  getAll(): CitationPosition[] {
    return [...this.positions];
  }

  /**
   * Get all positions for a specific marker.
   *
   * @param marker - The citation marker to find
   * @returns Array of positions (same citation may appear multiple times)
   */
  getByMarker(marker: string): CitationPosition[] {
    return this.positions.filter((p) => p.marker === marker);
  }

  /**
   * Clear all recorded positions.
   */
  clear(): void {
    this.positions = [];
  }
}
