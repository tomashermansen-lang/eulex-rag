/**
 * Sortable column header with hover tooltip portal.
 *
 * Shared between EvalDashboard (single-law) and CrossLawPanel (cross-law).
 * Uses a generic type parameter for sort field to support both panels' field types.
 */

import { useState, useRef } from 'react';
import { createPortal } from 'react-dom';

interface ColumnTooltipProps<T extends string = string> {
  label: string;
  description: string;
  sortField: T | null;
  field: T;
  sortDirection: 'asc' | 'desc';
  onSort: (field: T) => void;
}

export function ColumnTooltip<T extends string = string>({
  label,
  description,
  sortField: currentSort,
  field,
  sortDirection: dir,
  onSort,
}: ColumnTooltipProps<T>) {
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const thRef = useRef<HTMLTableCellElement>(null);

  const handleMouseEnter = () => {
    if (thRef.current) {
      const rect = thRef.current.getBoundingClientRect();
      setTooltipPos({
        x: rect.left + rect.width / 2,
        y: rect.bottom + 4,
      });
    }
    setShowTooltip(true);
  };

  return (
    <th
      ref={thRef}
      className="text-center font-medium text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide px-2 py-3 whitespace-nowrap cursor-pointer hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 transition-colors"
      onClick={() => onSort(field)}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setShowTooltip(false)}
    >
      <div className="flex items-center justify-center gap-1">
        {label}
        {currentSort === field && (
          <span className="text-apple-blue">{dir === 'asc' ? '↑' : '↓'}</span>
        )}
      </div>
      {showTooltip && createPortal(
        <div
          className="fixed px-3 py-2 bg-gray-800 text-white text-xs rounded-lg whitespace-nowrap z-[9999] shadow-lg pointer-events-none"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
            transform: 'translateX(-50%)',
          }}
        >
          {description}
          <div
            className="absolute border-4 border-transparent border-b-gray-800"
            style={{
              bottom: '100%',
              left: '50%',
              transform: 'translateX(-50%)',
            }}
          />
        </div>,
        document.body
      )}
    </th>
  );
}
