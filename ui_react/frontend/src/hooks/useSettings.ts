/**
 * Hook for managing user settings.
 *
 * Single Responsibility: Handle settings state and persistence.
 */

import { useState, useCallback, useEffect } from 'react';
import type { Settings, CorpusScope } from '../types';

const STORAGE_KEY = 'eulex-settings';

const DEFAULT_SETTINGS: Settings = {
  userProfile: 'LEGAL',
  debugMode: false,
  darkMode: window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false,
  corpusScope: 'single',
  targetCorpora: [],
};

/** Legacy settings interface for migration */
interface LegacySettings extends Partial<Settings> {
  law?: string;
}

/**
 * Load settings from localStorage with migration from legacy format.
 * Migrates `law` field to `targetCorpora` if present.
 */
function loadSettings(): Settings {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed: LegacySettings = JSON.parse(stored);

      // Migration: if legacy `law` field exists and targetCorpora is empty,
      // migrate to targetCorpora
      if (parsed.law && (!parsed.targetCorpora || parsed.targetCorpora.length === 0)) {
        parsed.targetCorpora = [parsed.law];
      }

      // Remove legacy field before merging
      const { law: _legacyLaw, ...migratedSettings } = parsed;

      return { ...DEFAULT_SETTINGS, ...migratedSettings };
    }
  } catch {
    // Ignore localStorage errors
  }
  return DEFAULT_SETTINGS;
}

/**
 * Save settings to localStorage.
 */
function saveSettings(settings: Settings): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Ignore localStorage errors
  }
}

/**
 * Hook for managing application settings.
 *
 * @returns Settings state and update functions
 */
export function useSettings() {
  const [settings, setSettingsState] = useState<Settings>(loadSettings);

  // Persist settings on change
  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  // Apply dark mode class to document
  useEffect(() => {
    if (settings.darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [settings.darkMode]);

  const setSettings = useCallback((updates: Partial<Settings>) => {
    setSettingsState((prev) => ({ ...prev, ...updates }));
  }, []);

  const setUserProfile = useCallback((userProfile: 'LEGAL' | 'ENGINEERING') => {
    setSettings({ userProfile });
  }, [setSettings]);

  const toggleDebugMode = useCallback(() => {
    setSettings({ debugMode: !settings.debugMode });
  }, [settings.debugMode, setSettings]);

  const toggleDarkMode = useCallback(() => {
    setSettings({ darkMode: !settings.darkMode });
  }, [settings.darkMode, setSettings]);

  const setCorpusScope = useCallback((corpusScope: CorpusScope) => {
    // Clear target corpora when switching to discover — AI picks its own
    if (corpusScope === 'discover') {
      setSettings({ corpusScope, targetCorpora: [] });
    } else {
      setSettings({ corpusScope });
    }
  }, [setSettings]);

  const setTargetCorpora = useCallback((targetCorpora: string[]) => {
    setSettings({ targetCorpora });
  }, [setSettings]);

  const resetSettings = useCallback(() => {
    setSettingsState(DEFAULT_SETTINGS);
  }, []);

  return {
    settings,
    setSettings,
    setUserProfile,
    toggleDebugMode,
    toggleDarkMode,
    setCorpusScope,
    setTargetCorpora,
    resetSettings,
  };
}
