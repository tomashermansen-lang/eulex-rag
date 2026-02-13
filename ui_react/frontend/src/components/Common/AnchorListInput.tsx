/**
 * Reusable component for managing a list of anchor references.
 *
 * Used in the eval case editor for managing must_include_any_of, etc.
 * Supports autocomplete when a law prop is provided.
 */

import { useState, useEffect, useRef } from 'react';
import { listAnchors } from '../../services/api';

interface AnchorListInputProps {
  label: string;
  description?: string;
  values: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  /** When provided, enables autocomplete from citation graph */
  law?: string;
}

const ANCHOR_PATTERN = /^(article|section|annex|recital|paragraph|chapter):[a-z0-9]+$/i;

// Syntax help with Danish translations
const ANCHOR_SYNTAX_HELP = [
  { en: 'article', da: 'artikel', example: 'article:5' },
  { en: 'annex', da: 'bilag', example: 'annex:iii' },
  { en: 'recital', da: 'betragtning', example: 'recital:10' },
  { en: 'chapter', da: 'kapitel', example: 'chapter:iv' },
  { en: 'section', da: 'afsnit', example: 'section:2' },
  { en: 'paragraph', da: 'stk.', example: 'paragraph:3' },
];

export function AnchorListInput({
  label,
  description,
  values,
  onChange,
  placeholder = 'article:5',
  law,
}: AnchorListInputProps) {
  const [inputValue, setInputValue] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [showSyntaxHelp, setShowSyntaxHelp] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Fetch suggestions when input changes (debounced)
  useEffect(() => {
    if (!law || inputValue.length < 2) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    const timer = setTimeout(async () => {
      try {
        const response = await listAnchors(law, inputValue);
        // Filter out already added values
        const filtered = response.anchors.filter(a => !values.includes(a));
        setSuggestions(filtered);
        setShowSuggestions(filtered.length > 0);
      } catch {
        // Silently fail - autocomplete is optional
        setSuggestions([]);
      }
    }, 200);

    return () => clearTimeout(timer);
  }, [inputValue, law, values]);

  // Close suggestions on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleAdd = () => {
    const trimmed = inputValue.trim().toLowerCase();

    if (!trimmed) {
      setError('Indtast en anchor reference');
      return;
    }

    if (!ANCHOR_PATTERN.test(trimmed)) {
      setError('Format: type:værdi (fx article:5, annex:iii)');
      return;
    }

    if (values.includes(trimmed)) {
      setError('Allerede tilføjet');
      return;
    }

    onChange([...values, trimmed]);
    setInputValue('');
    setError(null);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleAdd();
    }
  };

  const handleRemove = (index: number) => {
    onChange(values.filter((_, i) => i !== index));
  };

  const handleSelectSuggestion = (anchor: string) => {
    onChange([...values, anchor]);
    setInputValue('');
    setSuggestions([]);
    setShowSuggestions(false);
    setError(null);
  };

  return (
    <div className="space-y-2">
      <div>
        <div className="flex items-center gap-1.5">
          <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300">
            {label}
          </label>
          <button
            type="button"
            onClick={() => setShowSyntaxHelp(!showSyntaxHelp)}
            className="text-apple-gray-400 hover:text-apple-blue transition-colors"
            title="Vis syntaks hjælp"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>
        </div>
        {description && (
          <p className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500 mt-0.5">
            {description}
          </p>
        )}
        {/* Syntax help */}
        {showSyntaxHelp && (
          <div className="mt-2 p-2 bg-apple-gray-50 dark:bg-apple-gray-700 rounded-lg text-[10px]">
            <p className="text-apple-gray-500 dark:text-apple-gray-400 mb-1.5">
              Format: <code className="px-1 py-0.5 bg-apple-gray-200 dark:bg-apple-gray-600 rounded">type:værdi</code>
            </p>
            <div className="grid grid-cols-3 gap-x-3 gap-y-1">
              {ANCHOR_SYNTAX_HELP.map(({ en, da, example }) => (
                <div key={en} className="flex items-center gap-1">
                  <span className="text-apple-gray-400 dark:text-apple-gray-500">{da}:</span>
                  <code className="text-apple-blue">{example}</code>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Current values */}
      {values.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {values.map((value, index) => (
            <span
              key={`${value}-${index}`}
              className="inline-flex items-center gap-1 px-2 py-1 text-xs rounded-md bg-apple-gray-100 dark:bg-apple-gray-700 text-apple-gray-700 dark:text-apple-gray-300"
            >
              {value}
              <button
                type="button"
                onClick={() => handleRemove(index)}
                className="text-apple-gray-400 hover:text-red-500 transition-colors"
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </span>
          ))}
        </div>
      )}

      {/* Add new input */}
      <div className="relative" ref={containerRef}>
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value);
              setError(null);
            }}
            onKeyDown={handleKeyDown}
            onFocus={() => suggestions.length > 0 && setShowSuggestions(true)}
            placeholder={placeholder}
            className="flex-1 px-2 py-1.5 text-xs rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700 text-apple-gray-700 dark:text-white placeholder-apple-gray-400 dark:placeholder-apple-gray-500 focus:outline-none focus:ring-2 focus:ring-apple-blue/50"
          />
          <button
            type="button"
            onClick={handleAdd}
            className="px-3 py-1.5 text-xs font-medium text-apple-blue hover:bg-apple-blue/10 rounded-lg transition-colors"
          >
            Tilføj
          </button>
        </div>

        {/* Suggestions dropdown */}
        {showSuggestions && suggestions.length > 0 && (
          <div className="absolute z-10 w-full mt-1 bg-white dark:bg-apple-gray-700 border border-apple-gray-200 dark:border-apple-gray-600 rounded-lg shadow-lg max-h-40 overflow-y-auto">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => handleSelectSuggestion(suggestion)}
                className="w-full px-3 py-2 text-left text-xs text-apple-gray-700 dark:text-apple-gray-200 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 transition-colors"
              >
                {suggestion}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <p className="text-[10px] text-red-500">{error}</p>
      )}
    </div>
  );
}
