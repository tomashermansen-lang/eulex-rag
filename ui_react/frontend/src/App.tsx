/**
 * Main application component.
 *
 * Orchestrates the overall app layout and state management.
 */

import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { ChatContainer } from './components/Chat';
import { ContextPrompt } from './components/Chat/ContextPrompt';
import { Sidebar } from './components/Sidebar';
import { AdminPage } from './components/Admin';
import { SourcesSidepanel } from './components/Sources';
import { BottomSheet, ExportButton, HamburgerMenu, Toolbar } from './components/Common';
import { SourcesPanelProvider } from './contexts';
import { useChat, useSettings } from './hooks';
import { getCorpora, getExamples } from './services/api';
import { setCorpusRegistry } from './services/export';
import type { CorpusInfo, DiscoveryMatch, PageId } from './types';

function App() {
  const {
    settings,
    setUserProfile,
    toggleDebugMode,
    toggleDarkMode,
    setCorpusScope,
    setTargetCorpora,
  } = useSettings();

  const {
    messages,
    isStreaming,
    sendMessage,
    stopStreaming,
    clearChat,
    hasMessages,
  } = useChat(settings);

  const [corpora, setCorpora] = useState<CorpusInfo[]>([]);
  const [examples, setExamples] = useState<Record<string, Record<string, string[]>>>({});
  const [sourceUrl, setSourceUrl] = useState<string | undefined>();
  const [currentPage, setCurrentPage] = useState<PageId>('chat');
  const [isSourcesSheetOpen, setIsSourcesSheetOpen] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const isLockingRef = useRef(false);

  // Count total sources for the FAB badge
  const totalSources = messages.reduce(
    (sum, msg) => sum + (msg.references?.length || 0),
    0
  );

  // Load corpora and examples on mount
  useEffect(() => {
    async function loadConfig() {
      try {
        const [corporaData, examplesData] = await Promise.all([
          getCorpora(),
          getExamples(),
        ]);
        setCorpora(corporaData);
        setExamples(examplesData);
        // Populate the corpus registry for PDF export
        setCorpusRegistry(corporaData);
      } catch (error) {
        console.error('Failed to load config:', error);
      }
    }
    loadConfig();
  }, []);

  // Compute primary law from targetCorpora for source URL and examples
  const primaryLaw = settings.targetCorpora[0] || '';

  // Update source URL when primary law changes
  useEffect(() => {
    const corpus = corpora.find((c) => c.id === primaryLaw);
    setSourceUrl(corpus?.source_url);
  }, [primaryLaw, corpora]);

  // Auto-clear chat when corpus scope or target corpora changes (with 300ms debounce)
  const isInitialMount = useRef(true);
  const prevCorpusScope = useRef(settings.corpusScope);
  const prevTargetCorpora = useRef(settings.targetCorpora);

  useEffect(() => {
    // Skip on initial mount
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    // Skip auto-clear when locking (preserves chat history)
    if (isLockingRef.current) {
      isLockingRef.current = false;
      prevCorpusScope.current = settings.corpusScope;
      prevTargetCorpora.current = settings.targetCorpora;
      return;
    }

    // Check if corpus scope or target corpora actually changed
    const scopeChanged = prevCorpusScope.current !== settings.corpusScope;
    const corporaChanged =
      prevTargetCorpora.current.length !== settings.targetCorpora.length ||
      prevTargetCorpora.current.some((c, i) => c !== settings.targetCorpora[i]);

    if (!scopeChanged && !corporaChanged) {
      return;
    }

    // Update refs for next comparison
    prevCorpusScope.current = settings.corpusScope;
    prevTargetCorpora.current = settings.targetCorpora;

    // Only clear if there are messages to clear
    if (!hasMessages) {
      return;
    }

    // Debounce the clear action
    const timeoutId = setTimeout(() => {
      clearChat();
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [settings.corpusScope, settings.targetCorpora, hasMessages, clearChat]);

  // Extract full discovery result from the latest completed assistant message
  const discoveryResult = useMemo((): {
    matches: DiscoveryMatch[];
    gate: 'AUTO' | 'SUGGEST' | 'ABSTAIN';
    resolved_corpora?: string[];
  } | undefined => {
    if (settings.corpusScope !== 'discover') return undefined;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i];
      if (msg.role === 'assistant' && !msg.isStreaming && msg.retrievalMetrics) {
        const run = msg.retrievalMetrics.run as Record<string, unknown> | undefined;
        const disc = run?.discovery as
          | { matches: DiscoveryMatch[]; gate: 'AUTO' | 'SUGGEST' | 'ABSTAIN'; resolved_corpora?: string[] }
          | undefined;
        if (disc?.matches) return disc;
      }
    }
    return undefined;
  }, [messages, settings.corpusScope]);

  // Get examples for current law and profile
  const currentExamples = examples[primaryLaw]?.[settings.userProfile] || [];

  // Lock discovered laws as explicit search scope (preserves chat)
  const lockToCorpora = useCallback(
    (corporaIds: string[]) => {
      isLockingRef.current = true;
      setCorpusScope('explicit');
      setTargetCorpora(corporaIds);
    },
    [setCorpusScope, setTargetCorpora]
  );

  // Effect-based dispatch: send pending message after scope change is committed
  useEffect(() => {
    if (pendingMessage !== null && settings.corpusScope === 'explicit') {
      sendMessage(pendingMessage);
      setPendingMessage(null);
    }
  }, [pendingMessage, settings.corpusScope, sendMessage]);

  // Handle sending message — intercept in discovery mode with existing results
  const handleSendMessage = useCallback(
    (message: string) => {
      if (
        settings.corpusScope === 'discover' &&
        discoveryResult &&
        discoveryResult.gate !== 'ABSTAIN'
      ) {
        // Show context prompt instead of sending immediately
        setPendingMessage(message);
        return;
      }
      sendMessage(message);
    },
    [sendMessage, settings.corpusScope, discoveryResult]
  );

  // Context prompt callbacks
  const handleLockAndSend = useCallback(() => {
    const resolved = discoveryResult?.resolved_corpora ?? discoveryResult?.matches.map((m) => m.corpus_id) ?? [];
    lockToCorpora(resolved);
    // pendingMessage stays set — useEffect will send after scope propagates
  }, [discoveryResult, lockToCorpora]);

  const handleContinueAndSend = useCallback(() => {
    if (pendingMessage !== null) {
      sendMessage(pendingMessage);
      setPendingMessage(null);
    }
  }, [pendingMessage, sendMessage]);

  // Navigate to admin page
  const navigateToAdmin = useCallback(() => setCurrentPage('admin'), []);
  const navigateToChat = useCallback(() => setCurrentPage('chat'), []);

  // Refresh corpora data and update registry
  const refreshCorpora = useCallback(() => {
    Promise.all([getCorpora(), getExamples()])
      .then(([corporaData, examplesData]) => {
        setCorpora(corporaData);
        setExamples(examplesData);
        // Update the corpus registry for PDF export
        setCorpusRegistry(corporaData);
      })
      .catch(console.error);
  }, []);

  // Refresh corpora and examples when returning from admin (might have added new ones)
  const handleNavigateBackFromAdmin = useCallback(() => {
    navigateToChat();
    refreshCorpora();
  }, [navigateToChat, refreshCorpora]);

  return (
    <div className="flex flex-col h-screen bg-apple-gray-50 dark:bg-apple-gray-700">
      {currentPage === 'admin' ? (
        <AdminPage onNavigateBack={handleNavigateBackFromAdmin} onCorporaRefresh={refreshCorpora} />
      ) : (
        <SourcesPanelProvider>
          {/* Toolbar - spans full width */}
          <Toolbar
            appTitle="EuLex Legal Assistant"
            actions={
              <>
                <ExportButton messages={messages} />
                <HamburgerMenu
                  settings={settings}
                  onToggleDebug={toggleDebugMode}
                  onToggleDarkMode={toggleDarkMode}
                  onNavigateToAdmin={navigateToAdmin}
                />
              </>
            }
          />

          {/* Three panel layout - automatic 1/3 each */}
          <div className="three-panel-layout flex-1">
            {/* Sidebar panel */}
            <aside className="sidebar-panel" role="region" aria-label="Indstillinger">
              <Sidebar
                settings={settings}
                corpora={corpora}
                onProfileChange={setUserProfile}
                onCorpusScopeChange={setCorpusScope}
                onTargetCorporaChange={setTargetCorpora}
                onClearChat={clearChat}
                hasMessages={hasMessages}
                disabled={isStreaming}
                discoveryMatches={discoveryResult?.matches}
                discoveryLoading={settings.corpusScope === 'discover' && isStreaming}
                onLock={discoveryResult ? (ids: string[]) => lockToCorpora(ids) : undefined}
              />
            </aside>

            {/* Chat panel */}
            <main className="chat-panel" role="main" aria-label="Chat">
              <ChatContainer
                messages={messages}
                isStreaming={isStreaming}
                onSendMessage={handleSendMessage}
                onStopStreaming={stopStreaming}
                examples={currentExamples}
                settings={settings}
                onLock={discoveryResult ? () => lockToCorpora(
                  discoveryResult.resolved_corpora ?? discoveryResult.matches.map((m) => m.corpus_id)
                ) : undefined}
                contextPrompt={pendingMessage !== null && discoveryResult ? (
                  <ContextPrompt
                    lawNames={(discoveryResult.resolved_corpora ?? discoveryResult.matches.map((m) => m.corpus_id)).map((id) => id.toUpperCase())}
                    onLock={handleLockAndSend}
                    onContinue={handleContinueAndSend}
                  />
                ) : undefined}
              />
            </main>

            {/* Sources panel */}
            <aside className="sources-sidepanel" role="complementary" aria-label="Kilder">
              <SourcesSidepanel
                messages={messages}
                sourceUrl={sourceUrl}
              />
            </aside>
          </div>

          {/* Floating Action Button for sources on narrow screens */}
          {totalSources > 0 && (
            <button
              onClick={() => setIsSourcesSheetOpen(true)}
              className="sources-fab"
              aria-label={`Vis ${totalSources} kilder`}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <span>{totalSources} kilder</span>
            </button>
          )}

          {/* Bottom Sheet for sources on narrow screens */}
          <BottomSheet
            isOpen={isSourcesSheetOpen}
            onClose={() => setIsSourcesSheetOpen(false)}
            title="Kilder"
          >
            <SourcesSidepanel
              messages={messages}
              sourceUrl={sourceUrl}
            />
          </BottomSheet>
        </SourcesPanelProvider>
      )}
    </div>
  );
}

export default App;
