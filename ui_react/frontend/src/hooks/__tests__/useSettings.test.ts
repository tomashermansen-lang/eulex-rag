/**
 * Tests for useSettings hook.
 *
 * TDD: Tests updated to match redesigned settings without `law` field.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useSettings } from '../useSettings';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] ?? null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

// Mock matchMedia for dark mode detection
const matchMediaMock = vi.fn().mockImplementation((query: string) => ({
  matches: query === '(prefers-color-scheme: dark)' ? false : false,
  media: query,
  onchange: null,
  addListener: vi.fn(),
  removeListener: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  dispatchEvent: vi.fn(),
}));

describe('useSettings', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', localStorageMock);
    vi.stubGlobal('matchMedia', matchMediaMock);
    localStorageMock.clear();
    localStorageMock.getItem.mockClear();
    localStorageMock.setItem.mockClear();
    // Reset document classList
    document.documentElement.classList.remove('dark');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe('initial state', () => {
    it('returns default settings when localStorage is empty', () => {
      const { result } = renderHook(() => useSettings());

      expect(result.current.settings).toEqual({
        userProfile: 'LEGAL',
        debugMode: false,
        darkMode: false,
        corpusScope: 'single',
        targetCorpora: [],
      });
    });

    it('loads settings from localStorage if available', () => {
      localStorageMock.getItem.mockReturnValueOnce(
        JSON.stringify({ targetCorpora: ['gdpr'], userProfile: 'ENGINEERING' })
      );

      const { result } = renderHook(() => useSettings());

      expect(result.current.settings.targetCorpora).toEqual(['gdpr']);
      expect(result.current.settings.userProfile).toBe('ENGINEERING');
    });

    it('merges stored settings with defaults', () => {
      localStorageMock.getItem.mockReturnValueOnce(
        JSON.stringify({ targetCorpora: ['nis2'] })
      );

      const { result } = renderHook(() => useSettings());

      expect(result.current.settings.targetCorpora).toEqual(['nis2']);
      expect(result.current.settings.userProfile).toBe('LEGAL'); // default
      expect(result.current.settings.debugMode).toBe(false); // default
    });

    it('handles invalid JSON in localStorage gracefully', () => {
      localStorageMock.getItem.mockReturnValueOnce('invalid-json');

      const { result } = renderHook(() => useSettings());

      // Should fall back to defaults
      expect(result.current.settings.targetCorpora).toEqual([]);
    });

    it('migrates legacy law field to targetCorpora', () => {
      // Legacy settings with `law` field
      localStorageMock.getItem.mockReturnValueOnce(
        JSON.stringify({ law: 'ai-act', userProfile: 'LEGAL' })
      );

      const { result } = renderHook(() => useSettings());

      // Should migrate law to targetCorpora
      expect(result.current.settings.targetCorpora).toEqual(['ai-act']);
      // law field should not be in settings
      expect((result.current.settings as Record<string, unknown>).law).toBeUndefined();
    });

    it('does not migrate law if targetCorpora already has values', () => {
      localStorageMock.getItem.mockReturnValueOnce(
        JSON.stringify({ law: 'ai-act', targetCorpora: ['gdpr', 'nis2'] })
      );

      const { result } = renderHook(() => useSettings());

      // Should keep existing targetCorpora, not overwrite with law
      expect(result.current.settings.targetCorpora).toEqual(['gdpr', 'nis2']);
    });
  });

  describe('setSettings', () => {
    it('updates multiple settings at once', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.setSettings({ targetCorpora: ['dora'], debugMode: true });
      });

      expect(result.current.settings.targetCorpora).toEqual(['dora']);
      expect(result.current.settings.debugMode).toBe(true);
    });

    it('persists settings to localStorage', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.setSettings({ targetCorpora: ['data-act'] });
      });

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        'eulex-settings',
        expect.stringContaining('"targetCorpora":["data-act"]')
      );
    });
  });

  describe('setTargetCorpora', () => {
    it('updates the target corpora setting', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.setTargetCorpora(['gdpr', 'ai-act']);
      });

      expect(result.current.settings.targetCorpora).toEqual(['gdpr', 'ai-act']);
    });

    it('can set empty array', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.setTargetCorpora(['gdpr']);
      });

      act(() => {
        result.current.setTargetCorpora([]);
      });

      expect(result.current.settings.targetCorpora).toEqual([]);
    });
  });

  describe('setCorpusScope', () => {
    it('updates corpus scope to explicit', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.setCorpusScope('explicit');
      });

      expect(result.current.settings.corpusScope).toBe('explicit');
    });

    it('updates corpus scope to all', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.setCorpusScope('all');
      });

      expect(result.current.settings.corpusScope).toBe('all');
    });

    it('updates corpus scope to single', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.setCorpusScope('all');
      });

      act(() => {
        result.current.setCorpusScope('single');
      });

      expect(result.current.settings.corpusScope).toBe('single');
    });
  });

  describe('setUserProfile', () => {
    it('updates the user profile to ENGINEERING', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.setUserProfile('ENGINEERING');
      });

      expect(result.current.settings.userProfile).toBe('ENGINEERING');
    });

    it('updates the user profile to LEGAL', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.setUserProfile('ENGINEERING');
      });

      act(() => {
        result.current.setUserProfile('LEGAL');
      });

      expect(result.current.settings.userProfile).toBe('LEGAL');
    });
  });

  describe('toggleDebugMode', () => {
    it('toggles debug mode on', () => {
      const { result } = renderHook(() => useSettings());

      expect(result.current.settings.debugMode).toBe(false);

      act(() => {
        result.current.toggleDebugMode();
      });

      expect(result.current.settings.debugMode).toBe(true);
    });

    it('toggles debug mode off', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.toggleDebugMode(); // on
      });

      act(() => {
        result.current.toggleDebugMode(); // off
      });

      expect(result.current.settings.debugMode).toBe(false);
    });
  });

  describe('toggleDarkMode', () => {
    it('toggles dark mode on and adds class to document', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.toggleDarkMode();
      });

      expect(result.current.settings.darkMode).toBe(true);
      expect(document.documentElement.classList.contains('dark')).toBe(true);
    });

    it('toggles dark mode off and removes class from document', () => {
      const { result } = renderHook(() => useSettings());

      act(() => {
        result.current.toggleDarkMode(); // on
      });

      act(() => {
        result.current.toggleDarkMode(); // off
      });

      expect(result.current.settings.darkMode).toBe(false);
      expect(document.documentElement.classList.contains('dark')).toBe(false);
    });
  });

  describe('resetSettings', () => {
    it('resets all settings to defaults', () => {
      const { result } = renderHook(() => useSettings());

      // Change some settings
      act(() => {
        result.current.setSettings({
          targetCorpora: ['nis2'],
          userProfile: 'ENGINEERING',
          debugMode: true,
          darkMode: true,
        });
      });

      // Reset
      act(() => {
        result.current.resetSettings();
      });

      expect(result.current.settings).toEqual({
        userProfile: 'LEGAL',
        debugMode: false,
        darkMode: false,
        corpusScope: 'single',
        targetCorpora: [],
      });
    });
  });
});
