/**
 * Tests for useChat hook.
 *
 * TDD: Tests written to cover chat state management and streaming.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useChat } from '../useChat';
import type { Settings, AskResponse } from '../../types';

// Mock the API module
vi.mock('../../services/api', () => ({
  streamAnswer: vi.fn(),
}));

// Mock the suggestions utility
vi.mock('../../utils/suggestions', () => ({
  extractSuggestedQuestions: vi.fn().mockReturnValue([]),
}));

import { streamAnswer } from '../../services/api';
import { extractSuggestedQuestions } from '../../utils/suggestions';

const mockStreamAnswer = streamAnswer as ReturnType<typeof vi.fn>;
const mockExtractSuggestedQuestions = extractSuggestedQuestions as ReturnType<typeof vi.fn>;

describe('useChat', () => {
  const defaultSettings: Settings = {
    userProfile: 'LEGAL',
    debugMode: false,
    darkMode: false,
    corpusScope: 'single',
    targetCorpora: ['ai-act'],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockExtractSuggestedQuestions.mockReturnValue([]);
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('initial state', () => {
    it('starts with empty messages', () => {
      const { result } = renderHook(() => useChat(defaultSettings));

      expect(result.current.messages).toEqual([]);
      expect(result.current.hasMessages).toBe(false);
    });

    it('starts with isStreaming false', () => {
      const { result } = renderHook(() => useChat(defaultSettings));

      expect(result.current.isStreaming).toBe(false);
    });

    it('starts with no error', () => {
      const { result } = renderHook(() => useChat(defaultSettings));

      expect(result.current.error).toBeNull();
    });
  });

  describe('sendMessage', () => {
    it('does not send empty messages', async () => {
      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('   ');
      });

      expect(result.current.messages).toHaveLength(0);
      expect(mockStreamAnswer).not.toHaveBeenCalled();
    });

    it('adds user message and assistant placeholder when sending', async () => {
      mockStreamAnswer.mockImplementation(() => () => {}); // Returns abort function

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('What is GDPR?');
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.messages[0].role).toBe('user');
      expect(result.current.messages[0].content).toBe('What is GDPR?');
      expect(result.current.messages[1].role).toBe('assistant');
      expect(result.current.messages[1].isStreaming).toBe(true);
    });

    it('sets isStreaming to true while streaming', async () => {
      mockStreamAnswer.mockImplementation(() => () => {});

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test question');
      });

      expect(result.current.isStreaming).toBe(true);
    });

    it('calls streamAnswer with correct parameters', async () => {
      mockStreamAnswer.mockImplementation(() => () => {});

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test question');
      });

      expect(mockStreamAnswer).toHaveBeenCalledWith(
        expect.objectContaining({
          question: 'Test question',
          law: 'ai-act',
          user_profile: 'LEGAL',
          history: [],
        }),
        expect.any(Function), // onChunk
        expect.any(Function), // onResult
        expect.any(Function)  // onError
      );
    });

    it('updates assistant message with chunks', async () => {
      let onChunkCallback: (chunk: string) => void;

      mockStreamAnswer.mockImplementation((_req, onChunk) => {
        onChunkCallback = onChunk;
        return () => {};
      });

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      // Simulate receiving chunks
      act(() => {
        onChunkCallback!('Hello');
      });

      expect(result.current.messages[1].content).toBe('Hello');

      act(() => {
        onChunkCallback!(' World');
      });

      expect(result.current.messages[1].content).toBe('Hello World');
    });

    it('finalizes assistant message on result', async () => {
      let onResultCallback: (response: AskResponse) => void;

      mockStreamAnswer.mockImplementation((_req, _onChunk, onResult) => {
        onResultCallback = onResult;
        return () => {};
      });

      const mockResponse: AskResponse = {
        answer: 'This is the final answer.',
        references: [
          { idx: 1, display: 'Article 5', chunk_text: 'Test' },
        ],
        response_time_seconds: 1.5,
        retrieval_metrics: { chunks_retrieved: 5 },
      };

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      act(() => {
        onResultCallback!(mockResponse);
      });

      const assistantMsg = result.current.messages[1];
      expect(assistantMsg.content).toBe('This is the final answer.');
      expect(assistantMsg.references).toHaveLength(1);
      expect(assistantMsg.responseTime).toBe(1.5);
      expect(assistantMsg.isStreaming).toBe(false);
      expect(result.current.isStreaming).toBe(false);
    });

    it('handles errors gracefully', async () => {
      let onErrorCallback: (error: Error) => void;

      mockStreamAnswer.mockImplementation((_req, _onChunk, _onResult, onError) => {
        onErrorCallback = onError;
        return () => {};
      });

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      act(() => {
        onErrorCallback!(new Error('Network error'));
      });

      expect(result.current.error).toBe('Network error');
      expect(result.current.isStreaming).toBe(false);
      expect(result.current.messages[1].isStreaming).toBe(false);
    });

    it('prevents sending while already streaming', async () => {
      mockStreamAnswer.mockImplementation(() => () => {});

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('First question');
      });

      // Try to send another message while streaming
      await act(async () => {
        await result.current.sendMessage('Second question');
      });

      // Should only have the first message pair
      expect(result.current.messages).toHaveLength(2);
      expect(mockStreamAnswer).toHaveBeenCalledTimes(1);
    });
  });

  describe('stopStreaming', () => {
    it('calls abort function and sets isStreaming to false', async () => {
      const mockAbort = vi.fn();
      mockStreamAnswer.mockImplementation(() => mockAbort);

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      expect(result.current.isStreaming).toBe(true);

      act(() => {
        result.current.stopStreaming();
      });

      expect(mockAbort).toHaveBeenCalled();
      expect(result.current.isStreaming).toBe(false);
    });

    it('does nothing if not streaming', () => {
      const { result } = renderHook(() => useChat(defaultSettings));

      act(() => {
        result.current.stopStreaming();
      });

      expect(result.current.isStreaming).toBe(false);
    });
  });

  describe('clearChat', () => {
    it('clears all messages', async () => {
      let onResultCallback: (response: AskResponse) => void;

      mockStreamAnswer.mockImplementation((_req, _onChunk, onResult) => {
        onResultCallback = onResult;
        return () => {};
      });

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      act(() => {
        onResultCallback!({
          answer: 'Answer',
          references: [],
          response_time_seconds: 1,
        });
      });

      expect(result.current.messages).toHaveLength(2);
      expect(result.current.hasMessages).toBe(true);

      act(() => {
        result.current.clearChat();
      });

      expect(result.current.messages).toHaveLength(0);
      expect(result.current.hasMessages).toBe(false);
    });

    it('clears error state', async () => {
      let onErrorCallback: (error: Error) => void;

      mockStreamAnswer.mockImplementation((_req, _onChunk, _onResult, onError) => {
        onErrorCallback = onError;
        return () => {};
      });

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      act(() => {
        onErrorCallback!(new Error('Test error'));
      });

      expect(result.current.error).toBe('Test error');

      act(() => {
        result.current.clearChat();
      });

      expect(result.current.error).toBeNull();
    });

    it('stops streaming if active', async () => {
      const mockAbort = vi.fn();
      mockStreamAnswer.mockImplementation(() => mockAbort);

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      expect(result.current.isStreaming).toBe(true);

      act(() => {
        result.current.clearChat();
      });

      expect(mockAbort).toHaveBeenCalled();
      expect(result.current.isStreaming).toBe(false);
    });
  });

  describe('hasMessages', () => {
    it('is true when messages exist', async () => {
      mockStreamAnswer.mockImplementation(() => () => {});

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      expect(result.current.hasMessages).toBe(true);
    });

    it('is false after clearing', async () => {
      let onResultCallback: (response: AskResponse) => void;

      mockStreamAnswer.mockImplementation((_req, _onChunk, onResult) => {
        onResultCallback = onResult;
        return () => {};
      });

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      act(() => {
        onResultCallback!({ answer: 'Answer', references: [], response_time_seconds: 1 });
      });

      act(() => {
        result.current.clearChat();
      });

      expect(result.current.hasMessages).toBe(false);
    });
  });

  describe('suggested questions', () => {
    it('extracts suggested questions from response', async () => {
      let onResultCallback: (response: AskResponse) => void;

      mockStreamAnswer.mockImplementation((_req, _onChunk, onResult) => {
        onResultCallback = onResult;
        return () => {};
      });

      mockExtractSuggestedQuestions.mockReturnValue([
        'What about Article 6?',
        'How does this apply to AI systems?',
      ]);

      const { result } = renderHook(() => useChat(defaultSettings));

      await act(async () => {
        await result.current.sendMessage('Test');
      });

      act(() => {
        onResultCallback!({
          answer: 'Answer with suggestions',
          references: [],
          response_time_seconds: 1,
        });
      });

      expect(result.current.messages[1].suggestedQuestions).toEqual([
        'What about Article 6?',
        'How does this apply to AI systems?',
      ]);
    });
  });
});
