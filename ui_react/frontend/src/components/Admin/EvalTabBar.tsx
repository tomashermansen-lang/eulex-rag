/**
 * Tab bar for switching between Single-Law, Cross-Law, and Metrics eval views.
 *
 * Single Responsibility: Mode switching UI.
 */

import { SegmentedControl } from '../Common/SegmentedControl';

export type EvalMatrixMode = 'single' | 'cross' | 'metrics';

interface EvalTabBarProps {
  mode: EvalMatrixMode;
  onModeChange: (mode: EvalMatrixMode) => void;
}

// Chart icon component (bar chart)
function ChartIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  );
}

// Pie chart icon for metrics tab
function PieChartIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
    </svg>
  );
}

export function EvalTabBar({ mode, onModeChange }: EvalTabBarProps) {
  return (
    <div className="flex items-center gap-2">
      <SegmentedControl
        value={mode}
        onChange={onModeChange}
        options={[
          {
            value: 'single',
            label: (
              <span className="flex items-center gap-1.5">
                <ChartIcon className="w-3.5 h-3.5" />
                Single-Law Test Matrix
              </span>
            )
          },
          {
            value: 'cross',
            label: (
              <span className="flex items-center gap-1.5">
                <ChartIcon className="w-3.5 h-3.5" />
                Cross-Law Test Matrix
              </span>
            )
          },
          {
            value: 'metrics',
            label: (
              <span className="flex items-center gap-1.5">
                <PieChartIcon className="w-3.5 h-3.5" />
                Eval Metrics
              </span>
            )
          },
        ]}
        size="sm"
      />
    </div>
  );
}
