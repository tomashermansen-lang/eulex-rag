/**
 * Reusable Tooltip component.
 *
 * Single Responsibility: Show tooltip on hover with any content.
 * Uses portal for proper z-index handling across overflow containers.
 */

import { useState, useRef, useCallback, ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface TooltipProps {
  /** The content to show in the tooltip */
  content: string | null | undefined;
  /** The children to wrap (the trigger element) */
  children: ReactNode;
  /** Optional className for the wrapper span */
  className?: string;
  /** Tooltip position relative to trigger */
  position?: 'top' | 'bottom';
  /** Maximum width of tooltip in pixels */
  maxWidth?: number;
  /** Delay before showing tooltip in ms */
  delay?: number;
}

interface TooltipPosition {
  x: number;
  y: number;
  arrowOffset: number; // Offset from center for arrow (in pixels)
}

/**
 * A tooltip that shows additional information on hover.
 * Only renders tooltip if content is provided and non-empty.
 */
export function Tooltip({
  content,
  children,
  className = '',
  position = 'bottom',
  maxWidth = 400,
  delay = 200,
}: TooltipProps) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPos, setTooltipPos] = useState<TooltipPosition>({
    x: 0,
    y: 0,
    arrowOffset: 0,
  });
  const triggerRef = useRef<HTMLSpanElement>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleMouseEnter = useCallback(() => {
    if (!content) return;

    timeoutRef.current = setTimeout(() => {
      if (triggerRef.current) {
        const rect = triggerRef.current.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const padding = 16; // Minimum distance from viewport edge

        // Calculate initial x position (centered on trigger)
        const triggerCenterX = rect.left + rect.width / 2;

        // Estimate tooltip width (use maxWidth as upper bound)
        const estimatedWidth = Math.min(maxWidth, viewportWidth - padding * 2);
        const halfWidth = estimatedWidth / 2;

        // Calculate where tooltip left/right edges would be if centered
        const tooltipLeft = triggerCenterX - halfWidth;
        const tooltipRight = triggerCenterX + halfWidth;

        // Adjust x if tooltip would overflow viewport
        let x = triggerCenterX;
        let arrowOffset = 0;

        if (tooltipLeft < padding) {
          // Overflows left - shift tooltip right
          x = padding + halfWidth;
          arrowOffset = triggerCenterX - x; // Negative offset (arrow moves left)
        } else if (tooltipRight > viewportWidth - padding) {
          // Overflows right - shift tooltip left
          x = viewportWidth - padding - halfWidth;
          arrowOffset = triggerCenterX - x; // Positive offset (arrow moves right)
        }

        const y = position === 'bottom' ? rect.bottom + 8 : rect.top - 8;
        setTooltipPos({ x, y, arrowOffset });
      }
      setShowTooltip(true);
    }, delay);
  }, [content, position, delay, maxWidth]);

  const handleMouseLeave = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setShowTooltip(false);
  }, []);

  // Don't wrap if no content - just return children
  if (!content) {
    return <>{children}</>;
  }

  return (
    <span
      ref={triggerRef}
      className={`inline-block ${className}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      {showTooltip &&
        createPortal(
          <div
            className="fixed px-3 py-2 bg-gray-800 dark:bg-gray-700 text-white text-sm rounded-lg z-[9999] shadow-lg pointer-events-none"
            style={{
              left: tooltipPos.x,
              top: tooltipPos.y,
              transform:
                position === 'bottom'
                  ? 'translateX(-50%)'
                  : 'translateX(-50%) translateY(-100%)',
              maxWidth: `${maxWidth}px`,
            }}
          >
            {content}
            {/* Arrow pointing to trigger - position adjusted based on tooltip shift */}
            <div
              className={`absolute border-4 border-transparent ${
                position === 'bottom'
                  ? 'border-b-gray-800 dark:border-b-gray-700'
                  : 'border-t-gray-800 dark:border-t-gray-700'
              }`}
              style={{
                ...(position === 'bottom'
                  ? { bottom: '100%' }
                  : { top: '100%' }),
                left: `calc(50% + ${tooltipPos.arrowOffset}px)`,
                transform: 'translateX(-50%)',
              }}
            />
          </div>,
          document.body
        )}
    </span>
  );
}
