/**
 * Tests for EvalTabBar component (renamed from CrossLawTabBar).
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { EvalTabBar } from '../EvalTabBar';

describe('EvalTabBar', () => {
  it('renders all three mode options', () => {
    const onModeChange = vi.fn();
    render(<EvalTabBar mode="single" onModeChange={onModeChange} />);

    expect(screen.getByText('Single-Law Test Matrix')).toBeInTheDocument();
    expect(screen.getByText('Cross-Law Test Matrix')).toBeInTheDocument();
    expect(screen.getByText('Eval Metrics')).toBeInTheDocument();
  });

  it('calls onModeChange when switching to cross mode', () => {
    const onModeChange = vi.fn();
    render(<EvalTabBar mode="single" onModeChange={onModeChange} />);

    fireEvent.click(screen.getByText('Cross-Law Test Matrix'));
    expect(onModeChange).toHaveBeenCalledWith('cross');
  });

  it('calls onModeChange when switching to single mode', () => {
    const onModeChange = vi.fn();
    render(<EvalTabBar mode="cross" onModeChange={onModeChange} />);

    fireEvent.click(screen.getByText('Single-Law Test Matrix'));
    expect(onModeChange).toHaveBeenCalledWith('single');
  });

  it('calls onModeChange when switching to metrics mode', () => {
    const onModeChange = vi.fn();
    render(<EvalTabBar mode="single" onModeChange={onModeChange} />);

    fireEvent.click(screen.getByText('Eval Metrics'));
    expect(onModeChange).toHaveBeenCalledWith('metrics');
  });

  it('shows icons in all labels', () => {
    const onModeChange = vi.fn();
    render(<EvalTabBar mode="single" onModeChange={onModeChange} />);

    const svgs = document.querySelectorAll('svg');
    expect(svgs.length).toBe(3); // One icon per tab
  });
});
