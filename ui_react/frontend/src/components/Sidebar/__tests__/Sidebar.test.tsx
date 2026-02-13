/**
 * Tests for Sidebar component.
 *
 * TDD: Tests updated to match redesigned sidebar structure.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Sidebar } from '../Sidebar';
import type { Settings, CorpusInfo } from '../../../types';

// Mock Tooltip — renders children + stores content as data attribute
vi.mock('../../Common/Tooltip', () => ({
  Tooltip: ({
    content,
    children,
  }: {
    content: string;
    children: React.ReactNode;
  }) => (
    <span data-tooltip={content}>{children}</span>
  ),
}));

// Mock SegmentedControl
vi.mock('../../Common/SegmentedControl', () => ({
  SegmentedControl: ({
    options,
    value,
    onChange,
  }: {
    options: { value: string; label: React.ReactNode; tooltip?: string }[];
    value: string;
    onChange: (value: string) => void;
  }) => (
    <div data-testid="segmented-control">
      {options.map((opt) => (
        <button
          key={opt.value}
          data-selected={value === opt.value}
          onClick={() => onChange(opt.value)}
          title={opt.tooltip}
        >
          {opt.label}
        </button>
      ))}
    </div>
  ),
}));

// Mock LawSelectorPanel
vi.mock('../LawSelectorPanel', () => ({
  LawSelectorPanel: ({
    corpusScope,
    corpora,
    targetCorpora,
    onTargetCorporaChange,
    disabled,
  }: {
    corpusScope: string;
    corpora: CorpusInfo[];
    targetCorpora: string[];
    onTargetCorporaChange: (corpora: string[]) => void;
    disabled?: boolean;
  }) => (
    <div
      data-testid="law-selector-panel"
      data-corpus-scope={corpusScope}
      data-corpora-count={corpora.length}
      data-target-corpora={targetCorpora.join(',')}
      data-disabled={disabled}
    >
      <button onClick={() => onTargetCorporaChange(['gdpr'])}>
        Select GDPR
      </button>
    </div>
  ),
}));

describe('Sidebar', () => {
  const defaultSettings: Settings = {
    userProfile: 'LEGAL',
    debugMode: false,
    darkMode: false,
    corpusScope: 'single',
    targetCorpora: ['ai-act'],
  };

  const mockCorpora: CorpusInfo[] = [
    { id: 'gdpr', name: 'Persondataforordningen (GDPR)', fullname: 'General Data Protection Regulation' },
    { id: 'ai-act', name: 'AI-forordningen (AI Act)', fullname: 'Artificial Intelligence Act' },
    { id: 'nis2', name: 'NIS2-direktivet (NIS2)', fullname: 'NIS2 Directive' },
  ];

  const mockOnProfileChange = vi.fn();
  const mockOnCorpusScopeChange = vi.fn();
  const mockOnTargetCorporaChange = vi.fn();
  const mockOnClearChat = vi.fn();

  const defaultProps = {
    settings: defaultSettings,
    corpora: mockCorpora,
    onProfileChange: mockOnProfileChange,
    onCorpusScopeChange: mockOnCorpusScopeChange,
    onTargetCorporaChange: mockOnTargetCorporaChange,
    onClearChat: mockOnClearChat,
    hasMessages: true,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders sidebar with profil label', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByText('Profil')).toBeInTheDocument();
    });

    it('renders sidebar with søgeområde label', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByText('Søgeområde')).toBeInTheDocument();
    });

    it('renders version info', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByText('EuLex Legal Assistant v1.0')).toBeInTheDocument();
    });

    it('does not render lovgivning label (removed)', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.queryByText('Lovgivning')).not.toBeInTheDocument();
    });
  });

  describe('profile selector', () => {
    it('renders segmented controls (profile and scope)', () => {
      render(<Sidebar {...defaultProps} />);

      // There are 2 segmented controls: profile and scope
      expect(screen.getAllByTestId('segmented-control')).toHaveLength(2);
    });

    it('renders profile options', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByText('Juridisk')).toBeInTheDocument();
      expect(screen.getByText('Teknisk')).toBeInTheDocument();
    });

    it('calls onProfileChange when profile selected', () => {
      render(<Sidebar {...defaultProps} />);

      fireEvent.click(screen.getByText('Teknisk'));

      expect(mockOnProfileChange).toHaveBeenCalledWith('ENGINEERING');
    });

    it('shows description for LEGAL profile', () => {
      render(<Sidebar {...defaultProps} />);

      expect(
        screen.getByText(/Fokus på juridisk fortolkning/)
      ).toBeInTheDocument();
    });

    it('shows description for ENGINEERING profile', () => {
      render(
        <Sidebar
          {...defaultProps}
          settings={{ ...defaultSettings, userProfile: 'ENGINEERING' }}
        />
      );

      expect(
        screen.getByText(/Fokus på teknisk implementering/)
      ).toBeInTheDocument();
    });
  });

  describe('corpus scope selector', () => {
    it('renders scope options', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByText('Enkelt')).toBeInTheDocument();
      expect(screen.getByText('Udvalgte')).toBeInTheDocument();
      expect(screen.getByText('Alle')).toBeInTheDocument();
    });

    it('calls onCorpusScopeChange when scope selected', () => {
      render(<Sidebar {...defaultProps} />);

      fireEvent.click(screen.getByText('Udvalgte'));

      expect(mockOnCorpusScopeChange).toHaveBeenCalledWith('explicit');
    });

    it('shows tooltip for Enkelt scope segment', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByText('Enkelt')).toHaveAttribute('title', 'Søg kun i én bestemt lov');
    });

    it('shows tooltip for Udvalgte scope segment', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByText('Udvalgte')).toHaveAttribute('title', 'Søg på tværs af love du selv har valgt');
    });

    it('shows tooltip for Alle scope segment', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByText('Alle')).toHaveAttribute('title', 'Søg i al tilgængelig lovgivning');
    });

    it('shows description for single scope', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByText(/Søg kun i én bestemt lov/)).toBeInTheDocument();
    });

    it('shows description for discover scope', () => {
      render(
        <Sidebar
          {...defaultProps}
          settings={{ ...defaultSettings, corpusScope: 'discover' }}
        />
      );

      expect(screen.getByText(/AI identificerer automatisk relevante love/)).toBeInTheDocument();
    });
  });

  describe('law selector panel', () => {
    it('renders LawSelectorPanel', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByTestId('law-selector-panel')).toBeInTheDocument();
    });

    it('passes corpusScope to LawSelectorPanel', () => {
      render(
        <Sidebar
          {...defaultProps}
          settings={{ ...defaultSettings, corpusScope: 'explicit' }}
        />
      );

      expect(screen.getByTestId('law-selector-panel')).toHaveAttribute(
        'data-corpus-scope',
        'explicit'
      );
    });

    it('passes corpora to LawSelectorPanel', () => {
      render(<Sidebar {...defaultProps} />);

      expect(screen.getByTestId('law-selector-panel')).toHaveAttribute(
        'data-corpora-count',
        '3'
      );
    });

    it('passes targetCorpora to LawSelectorPanel', () => {
      render(
        <Sidebar
          {...defaultProps}
          settings={{ ...defaultSettings, targetCorpora: ['ai-act', 'gdpr'] }}
        />
      );

      expect(screen.getByTestId('law-selector-panel')).toHaveAttribute(
        'data-target-corpora',
        'ai-act,gdpr'
      );
    });

    it('calls onTargetCorporaChange when law selected', () => {
      render(<Sidebar {...defaultProps} />);

      fireEvent.click(screen.getByText('Select GDPR'));

      expect(mockOnTargetCorporaChange).toHaveBeenCalledWith(['gdpr']);
    });

    it('passes disabled prop to LawSelectorPanel', () => {
      render(<Sidebar {...defaultProps} disabled={true} />);

      expect(screen.getByTestId('law-selector-panel')).toHaveAttribute(
        'data-disabled',
        'true'
      );
    });
  });

  describe('clear chat button', () => {
    it('shows clear button when hasMessages is true', () => {
      render(<Sidebar {...defaultProps} hasMessages={true} />);

      expect(screen.getByRole('button', { name: /Ryd chat/ })).toBeInTheDocument();
    });

    it('hides clear button when hasMessages is false', () => {
      render(<Sidebar {...defaultProps} hasMessages={false} />);

      expect(screen.queryByRole('button', { name: /Ryd chat/ })).not.toBeInTheDocument();
    });

    it('calls onClearChat when clear button clicked', () => {
      render(<Sidebar {...defaultProps} hasMessages={true} />);

      fireEvent.click(screen.getByRole('button', { name: /Ryd chat/ }));

      expect(mockOnClearChat).toHaveBeenCalled();
    });
  });
});
