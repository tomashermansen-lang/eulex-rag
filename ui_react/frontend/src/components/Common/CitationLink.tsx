/**
 * Clickable citation link component with markdown rendering.
 *
 * Single Responsibility: Render markdown text with clickable citations.
 */

import { useCallback, useContext } from 'react';
import ReactMarkdown from 'react-markdown';
import { getRefAnchorId } from '../../utils/citations';
import { SourcesPanelContext } from '../../contexts';

interface CitationLinkProps {
  /** The citation index */
  idx: number | string;
  /** Message ID for unique anchor targeting */
  messageId?: string;
  /** Optional callback when citation is clicked */
  onCitationClick?: (idx: number | string) => void;
}

/**
 * A clickable citation that scrolls to the referenced source.
 */
export function CitationLink({ idx, messageId, onCitationClick }: CitationLinkProps) {
  // Use context directly (optional - may be null if not in provider)
  const sourcesPanel = useContext(SourcesPanelContext);

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();

      const refId = getRefAnchorId(idx, messageId);

      // Set selection via context if available (for persistent blue ring)
      if (sourcesPanel && messageId) {
        sourcesPanel.selectSource(refId, messageId);
      }

      // Dispatch custom event to expand sources panel and source item
      // Include messageId for sidepanel scroll coordination
      window.dispatchEvent(
        new CustomEvent('expandSource', { detail: { refId, messageId } })
      );

      // Scroll to the reference after a short delay to allow expansion
      setTimeout(() => {
        const refElement = document.getElementById(refId);
        if (refElement) {
          refElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
          // Add highlight animation
          refElement.classList.add('citation-highlight');
          setTimeout(() => {
            refElement.classList.remove('citation-highlight');
          }, 2000);
        }
      }, 200);

      onCitationClick?.(idx);
    },
    [idx, messageId, onCitationClick, sourcesPanel]
  );

  return (
    <button
      onClick={handleClick}
      className="citation-link"
      aria-label={`Gå til kilde ${idx}`}
    >
      [{idx}]
    </button>
  );
}

/**
 * Replace citation patterns [1], [2] with placeholder markers for post-processing.
 */
function preprocessCitations(text: string): { processed: string; citations: Map<string, number> } {
  const citations = new Map<string, number>();
  let counter = 0;

  // Replace citations with placeholders to protect them from markdown parsing
  const processed = text.replace(/\[(\d+)\]/g, (_, num) => {
    const placeholder = `%%CITE_${counter}%%`;
    citations.set(placeholder, parseInt(num, 10));
    counter++;
    return placeholder;
  });

  return { processed, citations };
}

/**
 * Render markdown text with clickable citations.
 */
export function renderWithCitations(
  text: string,
  messageId?: string,
  onCitationClick?: (idx: number | string) => void
): React.ReactNode {
  // Preprocess to protect citations from markdown parsing
  const { processed, citations } = preprocessCitations(text);

  // Custom component to handle text nodes and replace citation placeholders
  const components = {
    // Override text rendering to inject citation links
    p: ({ children, ...props }: React.HTMLAttributes<HTMLParagraphElement> & { children?: React.ReactNode }) => {
      const processChildren = (child: React.ReactNode): React.ReactNode => {
        if (typeof child === 'string') {
          const parts: React.ReactNode[] = [];
          let remaining = child;
          let key = 0;

          for (const [placeholder, idx] of citations) {
            const splitIndex = remaining.indexOf(placeholder);
            if (splitIndex !== -1) {
              if (splitIndex > 0) {
                parts.push(remaining.slice(0, splitIndex));
              }
              parts.push(
                <CitationLink
                  key={`cite-${idx}-${key++}`}
                  idx={idx}
                  messageId={messageId}
                  onCitationClick={onCitationClick}
                />
              );
              remaining = remaining.slice(splitIndex + placeholder.length);
            }
          }

          if (remaining) {
            parts.push(remaining);
          }

          return parts.length > 0 ? parts : child;
        }
        return child;
      };

      const processedChildren = Array.isArray(children)
        ? children.map(processChildren)
        : processChildren(children);

      return <p {...props}>{processedChildren}</p>;
    },
    // Style h3 headers (legal section headers)
    h3: ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement> & { children?: React.ReactNode }) => (
      <h3 {...props} className="font-semibold text-apple-gray-700 dark:text-apple-gray-100 text-base mt-5 mb-2 first:mt-0">
        {children}
      </h3>
    ),
    // Also handle list items
    li: ({ children, ...props }: React.HTMLAttributes<HTMLLIElement> & { children?: React.ReactNode }) => {
      const processChildren = (child: React.ReactNode): React.ReactNode => {
        if (typeof child === 'string') {
          const parts: React.ReactNode[] = [];
          let remaining = child;
          let key = 0;

          for (const [placeholder, idx] of citations) {
            const splitIndex = remaining.indexOf(placeholder);
            if (splitIndex !== -1) {
              if (splitIndex > 0) {
                parts.push(remaining.slice(0, splitIndex));
              }
              parts.push(
                <CitationLink
                  key={`cite-${idx}-${key++}`}
                  idx={idx}
                  messageId={messageId}
                  onCitationClick={onCitationClick}
                />
              );
              remaining = remaining.slice(splitIndex + placeholder.length);
            }
          }

          if (remaining) {
            parts.push(remaining);
          }

          return parts.length > 0 ? parts : child;
        }
        return child;
      };

      const processedChildren = Array.isArray(children)
        ? children.map(processChildren)
        : processChildren(children);

      return <li {...props}>{processedChildren}</li>;
    },
  };

  return <ReactMarkdown components={components}>{processed}</ReactMarkdown>;
}
