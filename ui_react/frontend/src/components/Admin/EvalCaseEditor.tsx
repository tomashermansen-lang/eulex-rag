/**
 * Modal component for creating and editing eval cases.
 *
 * Features:
 * - Create new cases with auto-generated ID
 * - Edit existing cases (changes origin to 'manual')
 * - Anchor list management for all four anchor types
 * - Test type checkboxes
 * - Profile and behavior selection
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import type { EvalCase, EvalCaseCreate, EvalCaseUpdate, EvalTestType, ExpectedBehavior } from '../../types';
import { AnchorListInput } from '../Common/AnchorListInput';
import { createEvalCase, updateEvalCase, runSingleCase, type SingleCaseResultResponse } from '../../services/api';

interface EvalCaseEditorProps {
  law: string;
  existingCase?: EvalCase | null;
  isOpen: boolean;
  onClose: () => void;
  onSaved: (savedCase: EvalCase) => void;
}

const TEST_TYPES: { value: EvalTestType; label: string; description: string }[] = [
  { value: 'retrieval', label: 'Retrieval', description: 'Tester om de rigtige dokumenter/chunks hentes fra indekset' },
  { value: 'faithfulness', label: 'Faithfulness', description: 'Tester om svaret er trofast mod de citerede kilder (ingen hallucination)' },
  { value: 'relevancy', label: 'Relevancy', description: 'Tester om svaret er relevant for spørgsmålet' },
  { value: 'abstention', label: 'Abstention', description: 'Tester om systemet korrekt afstår fra at svare når det bør' },
  { value: 'robustness', label: 'Robustness', description: 'Tester om systemet giver konsistente svar på omformulerede spørgsmål' },
  { value: 'multi_hop', label: 'Multi-hop', description: 'Tester om systemet kan kombinere information fra flere kilder' },
];

const DEFAULT_EXPECTED: ExpectedBehavior = {
  must_include_any_of: [],
  must_include_any_of_2: [],
  must_include_all_of: [],
  must_not_include_any_of: [],
  contract_check: false,
  min_citations: null,
  max_citations: null,
  behavior: 'answer',
  allow_empty_references: false,
  must_have_article_support_for_normative: true,
  notes: '',
};

export function EvalCaseEditor({
  law,
  existingCase,
  isOpen,
  onClose,
  onSaved,
}: EvalCaseEditorProps) {
  const isEditing = !!existingCase;

  // Form state
  const [profile, setProfile] = useState<'LEGAL' | 'ENGINEERING'>('LEGAL');
  const [prompt, setPrompt] = useState('');
  const [testTypes, setTestTypes] = useState<EvalTestType[]>(['retrieval']);
  const [expected, setExpected] = useState<ExpectedBehavior>(DEFAULT_EXPECTED);

  // UI state
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Validation state
  type RunMode = 'retrieval_only' | 'full' | 'full_with_judge';
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<SingleCaseResultResponse | null>(null);
  const [validationMode, setValidationMode] = useState<RunMode>('full');
  const [showModeDropdown, setShowModeDropdown] = useState(false);
  const validationResultRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to validation result when it appears
  useEffect(() => {
    if (validationResult && validationResultRef.current) {
      validationResultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [validationResult]);

  // Initialize form when editing
  useEffect(() => {
    if (existingCase) {
      setProfile(existingCase.profile);
      setPrompt(existingCase.prompt);
      setTestTypes(existingCase.test_types);
      setExpected(existingCase.expected);
    } else {
      // Reset for new case
      setProfile('LEGAL');
      setPrompt('');
      setTestTypes(['retrieval']);
      setExpected(DEFAULT_EXPECTED);
    }
    setHasUnsavedChanges(false);
    setError(null);
    setValidationResult(null);
  }, [existingCase, isOpen]);

  // Track changes
  const handleChange = useCallback(() => {
    setHasUnsavedChanges(true);
    setError(null);
  }, []);

  // Handle close with unsaved changes warning
  const handleClose = () => {
    if (hasUnsavedChanges) {
      if (!confirm('Du har ugemte ændringer. Er du sikker på at du vil lukke?')) {
        return;
      }
    }
    onClose();
  };

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        handleClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, hasUnsavedChanges]);

  // Toggle test type
  const toggleTestType = (tt: EvalTestType) => {
    setTestTypes((prev) =>
      prev.includes(tt)
        ? prev.filter((t) => t !== tt)
        : [...prev, tt]
    );
    handleChange();
  };

  // Update expected field
  const updateExpected = <K extends keyof ExpectedBehavior>(
    field: K,
    value: ExpectedBehavior[K]
  ) => {
    setExpected((prev) => ({ ...prev, [field]: value }));
    handleChange();
  };

  // Validate form
  const validateForm = (): string | null => {
    if (prompt.trim().length < 10) {
      return 'Prompt skal være mindst 10 tegn';
    }
    if (testTypes.length === 0) {
      return 'Vælg mindst én test type';
    }
    return null;
  };

  // Save case
  const handleSave = async () => {
    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsSaving(true);
    setError(null);

    try {
      let savedCase: EvalCase;

      if (isEditing && existingCase) {
        // Update existing
        const updateData: EvalCaseUpdate = {
          profile,
          prompt,
          test_types: testTypes,
          expected,
        };
        savedCase = await updateEvalCase(law, existingCase.id, updateData);
      } else {
        // Create new
        const createData: EvalCaseCreate = {
          profile,
          prompt,
          test_types: testTypes,
          expected,
        };
        savedCase = await createEvalCase(law, createData);
      }

      setHasUnsavedChanges(false);
      onSaved(savedCase);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Kunne ikke gemme');
    } finally {
      setIsSaving(false);
    }
  };

  // Run validation
  const handleValidate = async () => {
    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsValidating(true);
    setError(null);
    setValidationResult(null);

    try {
      console.log('[Valider] Sending request...', { law, validationMode, prompt: prompt.slice(0, 50) });
      const result = await runSingleCase({
        law,
        run_mode: validationMode,
        prompt,
        profile,
        test_types: testTypes,
        expected: {
          must_include_any_of: expected.must_include_any_of,
          must_include_any_of_2: expected.must_include_any_of_2,
          must_include_all_of: expected.must_include_all_of,
          must_not_include_any_of: expected.must_not_include_any_of,
          behavior: expected.behavior,
          min_citations: expected.min_citations,
          max_citations: expected.max_citations,
        },
      });
      console.log('[Valider] Got result:', result.passed, result.error);
      setValidationResult(result);
    } catch (err) {
      console.error('[Valider] Request failed:', err);
      setError(err instanceof Error ? err.message : 'Validering fejlede');
    } finally {
      setIsValidating(false);
    }
  };

  // Mode labels
  const MODE_LABELS: Record<RunMode, { short: string; long: string }> = {
    retrieval_only: { short: 'Ret.', long: 'Kun retrieval (hurtig)' },
    full: { short: 'Full', long: 'Full (med LLM)' },
    full_with_judge: { short: 'Full+J', long: 'Full + LLM Judge' },
  };

  if (!isOpen) return null;

  return createPortal(
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-50"
            onClick={handleClose}
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="fixed inset-4 md:inset-y-[5vh] md:inset-x-auto md:left-1/2 md:-translate-x-1/2 md:w-full md:max-w-2xl bg-white dark:bg-apple-gray-600 rounded-2xl shadow-2xl z-50 flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-apple-gray-100 dark:border-apple-gray-500">
              <div>
                <h2 className="text-lg font-semibold text-apple-gray-700 dark:text-white">
                  {isEditing ? 'Rediger Test Case' : 'Ny Test Case'}
                </h2>
                {isEditing && existingCase && (
                  <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500 mt-0.5">
                    {existingCase.id}
                  </p>
                )}
              </div>
              <button
                onClick={handleClose}
                className="p-2 text-apple-gray-400 hover:text-apple-gray-600 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-500 rounded-lg transition-colors"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Body - scrollable */}
            <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6">
              {/* Profile */}
              <div>
                <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                  Profil *
                </label>
                <div className="flex gap-4" role="radiogroup" aria-label="Profil">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="profile"
                      value="LEGAL"
                      checked={profile === 'LEGAL'}
                      onChange={() => { setProfile('LEGAL'); handleChange(); }}
                      className="w-4 h-4 text-apple-blue"
                    />
                    <span className="text-sm text-apple-gray-700 dark:text-white">LEGAL</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      name="profile"
                      value="ENGINEERING"
                      checked={profile === 'ENGINEERING'}
                      onChange={() => { setProfile('ENGINEERING'); handleChange(); }}
                      className="w-4 h-4 text-apple-blue"
                    />
                    <span className="text-sm text-apple-gray-700 dark:text-white">ENGINEERING</span>
                  </label>
                </div>
              </div>

              {/* Prompt */}
              <div>
                <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                  Prompt *
                </label>
                <textarea
                  value={prompt}
                  onChange={(e) => { setPrompt(e.target.value); handleChange(); }}
                  placeholder="Skriv testspørgsmålet her..."
                  rows={3}
                  autoFocus
                  className="w-full px-3 py-2 text-sm rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700 text-apple-gray-700 dark:text-white placeholder-apple-gray-400 dark:placeholder-apple-gray-500 focus:outline-none focus:ring-2 focus:ring-apple-blue/50 resize-none"
                />
                <p className="text-[10px] text-apple-gray-400 mt-1">
                  Min. 10 tegn
                </p>
              </div>

              {/* Test Types */}
              <div>
                <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                  Test Types *
                </label>
                <div className="flex flex-wrap gap-2">
                  {TEST_TYPES.map(({ value, label, description }) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => toggleTestType(value)}
                      title={description}
                      className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                        testTypes.includes(value)
                          ? 'bg-apple-blue text-white'
                          : 'bg-apple-gray-100 dark:bg-apple-gray-700 text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-200 dark:hover:bg-apple-gray-600'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Expected Behavior Section */}
              <div className="border-t border-apple-gray-100 dark:border-apple-gray-500 pt-4">
                <h3 className="text-sm font-semibold text-apple-gray-700 dark:text-white mb-4">
                  Forventet Adfærd
                </h3>

                {/* Behavior */}
                <div className="mb-4">
                  <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                    Forventet respons
                  </label>
                  <div className="flex gap-4" role="radiogroup" aria-label="Forventet respons">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="behavior"
                        value="answer"
                        checked={expected.behavior === 'answer'}
                        onChange={() => updateExpected('behavior', 'answer')}
                        className="w-4 h-4 text-apple-blue"
                      />
                      <span className="text-sm text-apple-gray-700 dark:text-white">Svar (answer)</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        name="behavior"
                        value="abstain"
                        checked={expected.behavior === 'abstain'}
                        onChange={() => updateExpected('behavior', 'abstain')}
                        className="w-4 h-4 text-apple-blue"
                      />
                      <span className="text-sm text-apple-gray-700 dark:text-white">Afstå (abstain)</span>
                    </label>
                  </div>
                </div>

                {/* Anchor Lists */}
                <div className="space-y-4">
                  <AnchorListInput
                    label="Skal inkludere mindst én af"
                    description="Svaret skal referere til mindst én af disse"
                    values={expected.must_include_any_of}
                    onChange={(values) => updateExpected('must_include_any_of', values)}
                    law={law}
                  />

                  <AnchorListInput
                    label="Skal inkludere mindst én af (sæt 2)"
                    description="Sekundært sæt af any-of anchors"
                    values={expected.must_include_any_of_2}
                    onChange={(values) => updateExpected('must_include_any_of_2', values)}
                    law={law}
                  />

                  <AnchorListInput
                    label="Skal inkludere alle"
                    description="Svaret skal referere til alle disse"
                    values={expected.must_include_all_of}
                    onChange={(values) => updateExpected('must_include_all_of', values)}
                    law={law}
                  />

                  <AnchorListInput
                    label="Må ikke inkludere"
                    description="Svaret må ikke referere til disse"
                    values={expected.must_not_include_any_of}
                    onChange={(values) => updateExpected('must_not_include_any_of', values)}
                    law={law}
                  />
                </div>

                {/* Citation constraints */}
                <div className="mt-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={expected.contract_check}
                      onChange={(e) => updateExpected('contract_check', e.target.checked)}
                      className="w-4 h-4 rounded text-apple-blue"
                    />
                    <span className="text-sm text-apple-gray-700 dark:text-white">
                      Aktiver citation constraints
                    </span>
                  </label>
                  <p className="text-[10px] text-apple-gray-400 dark:text-apple-gray-500 mt-1 ml-6">
                    Validerer at svaret har et specifikt antal citationer. Brug min/max felterne til at sætte grænser.
                  </p>

                  {expected.contract_check && (
                    <div className="flex gap-4 mt-2 ml-6">
                      <div>
                        <label className="block text-[10px] text-apple-gray-500 mb-1">Min</label>
                        <input
                          type="number"
                          min="0"
                          value={expected.min_citations ?? ''}
                          onChange={(e) => updateExpected('min_citations', e.target.value ? parseInt(e.target.value) : null)}
                          className="w-20 px-2 py-1 text-sm rounded border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] text-apple-gray-500 mb-1">Max</label>
                        <input
                          type="number"
                          min="0"
                          value={expected.max_citations ?? ''}
                          onChange={(e) => updateExpected('max_citations', e.target.value ? parseInt(e.target.value) : null)}
                          className="w-20 px-2 py-1 text-sm rounded border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700"
                        />
                      </div>
                    </div>
                  )}
                </div>

                {/* Notes */}
                <div className="mt-4">
                  <label className="block text-xs font-medium text-apple-gray-600 dark:text-apple-gray-300 mb-2">
                    Noter
                  </label>
                  <textarea
                    value={expected.notes}
                    onChange={(e) => updateExpected('notes', e.target.value)}
                    placeholder="Interne noter om testen..."
                    rows={2}
                    className="w-full px-3 py-2 text-sm rounded-lg border border-apple-gray-200 dark:border-apple-gray-500 bg-white dark:bg-apple-gray-700 text-apple-gray-700 dark:text-white placeholder-apple-gray-400 dark:placeholder-apple-gray-500 focus:outline-none focus:ring-2 focus:ring-apple-blue/50 resize-none"
                  />
                </div>
              </div>

              {/* Validation Result Panel */}
              {validationResult && (
                <div ref={validationResultRef} className="border-t border-apple-gray-100 dark:border-apple-gray-500 pt-4" role="region" aria-live="polite" aria-label="Valideringsresultat">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-apple-gray-700 dark:text-white flex items-center gap-2">
                      <span className={validationResult.passed ? 'text-green-500' : 'text-red-500'}>
                        {validationResult.passed ? '✓' : '✗'}
                      </span>
                      Valideringsresultat
                    </h3>
                    <div className="flex items-center gap-2 text-xs text-apple-gray-400">
                      <span>{Math.round(validationResult.duration_ms)}ms</span>
                      <button
                        onClick={() => setValidationResult(null)}
                        className="p-1 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 rounded"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  </div>

                  {/* Error message */}
                  {validationResult.error && (
                    <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-lg mb-3">
                      <p className="text-sm text-red-600 dark:text-red-400">{validationResult.error}</p>
                    </div>
                  )}

                  {/* Scores grid */}
                  {Object.keys(validationResult.scores).length > 0 && (
                    <div className="mb-4">
                      <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                        Scorer
                      </p>
                      <div className="grid grid-cols-2 gap-2">
                        {Object.entries(validationResult.scores).map(([name, score]) => (
                          <div
                            key={name}
                            className={`p-2 rounded-lg cursor-help ${score.passed ? 'bg-green-50 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/20'}`}
                            title={name}
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-medium text-apple-gray-700 dark:text-white">
                                {name}
                              </span>
                              <span className={`text-xs ${score.passed ? 'text-green-600' : 'text-red-600'}`}>
                                {typeof score.score === 'number' ? `${Math.round(score.score * 100)}%` : score.passed ? '✓' : '✗'}
                              </span>
                            </div>
                            {score.message && (
                              <p className="text-[10px] text-apple-gray-500 dark:text-apple-gray-400 mt-1 line-clamp-2">
                                {score.message}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Generated answer */}
                  {validationResult.answer && (
                    <div className="mb-4">
                      <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                        Genereret svar
                      </p>
                      <div className="p-3 bg-apple-gray-50 dark:bg-apple-gray-700 rounded-lg prose prose-sm dark:prose-invert max-w-none prose-headings:text-sm prose-headings:font-semibold prose-headings:mt-3 prose-headings:mb-1 prose-p:my-1 prose-ul:my-1 prose-li:my-0">
                        <ReactMarkdown>
                          {validationResult.answer}
                        </ReactMarkdown>
                      </div>
                    </div>
                  )}

                  {/* References */}
                  {validationResult.references.length > 0 && (
                    <div>
                      <p className="text-xs text-apple-gray-500 dark:text-apple-gray-400 uppercase tracking-wide mb-2">
                        Kilder ({validationResult.references.length})
                      </p>
                      <div className="space-y-2 max-h-48 overflow-y-auto">
                        {validationResult.references.map((ref, i) => (
                          <div
                            key={i}
                            className="p-2 bg-apple-gray-50 dark:bg-apple-gray-700 rounded-lg"
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-[10px] font-medium text-apple-blue">
                                [{ref.idx}]
                              </span>
                              <span className="text-xs text-apple-gray-600 dark:text-apple-gray-300">
                                {ref.display || [ref.article && `Art. ${ref.article}`, ref.annex && `Bilag ${ref.annex}`, ref.recital && `Betr. ${ref.recital}`].filter(Boolean).join(', ') || 'Ukendt'}
                              </span>
                            </div>
                            {ref.chunk_text && (
                              <p className="text-[10px] text-apple-gray-500 dark:text-apple-gray-400 line-clamp-3">
                                {ref.chunk_text}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between px-6 py-4 border-t border-apple-gray-100 dark:border-apple-gray-500 bg-apple-gray-50 dark:bg-apple-gray-700">
              {/* Left side - Validate button with mode dropdown */}
              <div className="flex items-center gap-3">
                <div className="relative">
                  <div className="flex">
                    <button
                      type="button"
                      onClick={handleValidate}
                      disabled={isValidating || isSaving}
                      className="px-3 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 bg-white dark:bg-apple-gray-600 border border-apple-gray-200 dark:border-apple-gray-500 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500 rounded-l-lg transition-colors disabled:opacity-50 flex items-center gap-2"
                    >
                      {isValidating ? (
                        <div className="w-4 h-4 border-2 border-apple-gray-400 border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      )}
                      Valider
                    </button>
                    <button
                      type="button"
                      onClick={() => setShowModeDropdown(!showModeDropdown)}
                      disabled={isValidating || isSaving}
                      className="px-2 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 bg-white dark:bg-apple-gray-600 border border-l-0 border-apple-gray-200 dark:border-apple-gray-500 hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500 rounded-r-lg transition-colors disabled:opacity-50"
                    >
                      <span className="text-[10px] px-1 py-0.5 bg-apple-gray-100 dark:bg-apple-gray-500 rounded">
                        {MODE_LABELS[validationMode].short}
                      </span>
                      <svg className="w-3 h-3 ml-1 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>
                  </div>
                  {/* Mode dropdown */}
                  {showModeDropdown && (
                    <div className="absolute bottom-full left-0 mb-1 bg-white dark:bg-apple-gray-600 border border-apple-gray-200 dark:border-apple-gray-500 rounded-lg shadow-lg z-10 overflow-hidden">
                      {(Object.keys(MODE_LABELS) as RunMode[]).map((mode) => (
                        <button
                          key={mode}
                          type="button"
                          onClick={() => { setValidationMode(mode); setShowModeDropdown(false); }}
                          className={`w-full px-3 py-2 text-left text-xs hover:bg-apple-gray-50 dark:hover:bg-apple-gray-500 transition-colors ${
                            validationMode === mode ? 'bg-apple-blue/10 text-apple-blue' : 'text-apple-gray-600 dark:text-apple-gray-300'
                          }`}
                        >
                          {MODE_LABELS[mode].long}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                {error && (
                  <p className="text-sm text-red-500">{error}</p>
                )}
                {!error && (
                  <p className="text-xs text-apple-gray-400 dark:text-apple-gray-500">
                    {isEditing ? 'Ændringer sætter origin til "manual"' : 'Nye cases oprettes som "manual"'}
                  </p>
                )}
              </div>

              {/* Right side - Cancel and Save */}
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={handleClose}
                  className="px-4 py-2 text-sm font-medium text-apple-gray-600 dark:text-apple-gray-300 hover:bg-apple-gray-100 dark:hover:bg-apple-gray-600 rounded-lg transition-colors"
                >
                  Annuller
                </button>
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={isSaving}
                  className="px-4 py-2 text-sm font-medium text-white bg-apple-blue hover:bg-apple-blue/90 rounded-lg transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  {isSaving && (
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  )}
                  {isEditing ? 'Gem ændringer' : 'Opret case'}
                </button>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>,
    document.body
  );
}
