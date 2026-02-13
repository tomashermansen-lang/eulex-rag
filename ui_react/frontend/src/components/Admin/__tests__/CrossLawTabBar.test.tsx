/**
 * Tests for CrossLawTabBar component.
 */

import { render, screen, fireEvent } from '@testing-library/react';
import { CrossLawTabBar } from '../CrossLawTabBar';
import type { EvalMatrixMode } from '../CrossLawTabBar';

describe('CrossLawTabBar', () => {
  it('renders with single mode selected', () => {
    const onModeChange = vi.fn();
    render(<CrossLawTabBar mode="single" onModeChange={onModeChange} />);

    expect(screen.getByText('Single-Law Test Matrix')).toBeInTheDocument();
    expect(screen.getByText('Cross-Law Test Matrix')).toBeInTheDocument();
  });

  it('calls onModeChange when switching to cross mode', () => {
    const onModeChange = vi.fn();
    render(<CrossLawTabBar mode="single" onModeChange={onModeChange} />);

    fireEvent.click(screen.getByText('Cross-Law Test Matrix'));
    expect(onModeChange).toHaveBeenCalledWith('cross');
  });

  it('calls onModeChange when switching to single mode', () => {
    const onModeChange = vi.fn();
    render(<CrossLawTabBar mode="cross" onModeChange={onModeChange} />);

    fireEvent.click(screen.getByText('Single-Law Test Matrix'));
    expect(onModeChange).toHaveBeenCalledWith('single');
  });

  it('shows chart icons in labels', () => {
    const onModeChange = vi.fn();
    render(<CrossLawTabBar mode="single" onModeChange={onModeChange} />);

    // Check that SVG chart icons are rendered
    const svgs = document.querySelectorAll('svg');
    expect(svgs.length).toBe(2); // One icon per tab
  });
});
