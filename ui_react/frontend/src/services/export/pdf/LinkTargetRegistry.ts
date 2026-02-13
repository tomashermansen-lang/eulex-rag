/**
 * Registry for PDF internal link targets.
 *
 * Stores page number and Y position for each link target (e.g., source citations).
 * Used to create internal document links from citations to their source entries.
 *
 * Single Responsibility: Store and retrieve link target positions.
 */

/**
 * A link target position within the PDF document.
 */
export interface LinkTarget {
  /** The page number (1-indexed) */
  pageNumber: number;
  /** The Y position from top of page in mm */
  top: number;
}

/**
 * Registry for storing and retrieving link target positions.
 */
export class LinkTargetRegistry {
  private targets = new Map<string, LinkTarget>();

  /**
   * Register a link target position.
   *
   * @param id - Unique identifier (e.g., "source-[1]")
   * @param pageNumber - The page number (1-indexed)
   * @param top - The Y position from top of page in mm
   */
  register(id: string, pageNumber: number, top: number): void {
    this.targets.set(id, { pageNumber, top });
  }

  /**
   * Get a link target position by id.
   *
   * @param id - The target identifier
   * @returns The target position, or undefined if not found
   */
  get(id: string): LinkTarget | undefined {
    return this.targets.get(id);
  }

  /**
   * Check if a link target is registered.
   *
   * @param id - The target identifier
   * @returns True if registered
   */
  has(id: string): boolean {
    return this.targets.has(id);
  }

  /**
   * Clear all registered targets.
   */
  clear(): void {
    this.targets.clear();
  }
}
