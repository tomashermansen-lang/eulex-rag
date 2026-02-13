/**
 * Tests for API client.
 *
 * TDD: Tests written to cover API requests and error handling.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  ApiError,
  askQuestion,
  getCorpora,
  getExamples,
  healthCheck,
  listEvalCases,
  getEvalCase,
  createEvalCase,
  updateEvalCase,
  deleteEvalCase,
  duplicateEvalCase,
  listAnchors,
  runSingleCase,
  streamAnswer,
} from '../api';
import type { AskRequest, EvalCaseCreate, EvalCaseUpdate } from '../../types';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('ApiError', () => {
    it('creates error with message', () => {
      const error = new ApiError('Something went wrong');
      expect(error.message).toBe('Something went wrong');
      expect(error.name).toBe('ApiError');
    });

    it('creates error with status', () => {
      const error = new ApiError('Not found', 404);
      expect(error.status).toBe(404);
    });

    it('creates error with data', () => {
      const error = new ApiError('Validation failed', 400, { field: 'name' });
      expect(error.data).toEqual({ field: 'name' });
    });
  });

  describe('askQuestion', () => {
    const request: AskRequest = {
      question: 'What is GDPR?',
      law: 'gdpr',
      profile: 'LEGAL',
    };

    it('sends POST request with correct body', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ answer: 'GDPR is...' }),
      });

      await askQuestion(request);

      expect(mockFetch).toHaveBeenCalledWith('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
    });

    it('returns response data', async () => {
      const mockResponse = {
        answer: 'GDPR is a regulation',
        references: [],
        retrieval_metrics: {},
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await askQuestion(request);

      expect(result).toEqual(mockResponse);
    });

    it('throws ApiError on failure', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.resolve({ detail: 'Server error' }),
      });

      await expect(askQuestion(request)).rejects.toThrow(ApiError);
      await expect(askQuestion(request)).rejects.toThrow('Server error');
    });

    it('uses statusText when detail is missing', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.resolve({}),
      });

      await expect(askQuestion(request)).rejects.toThrow('Request failed: Internal Server Error');
    });
  });

  describe('getCorpora', () => {
    it('fetches corpora list', async () => {
      const corpora = [
        { id: 'gdpr', name: 'GDPR' },
        { id: 'ai-act', name: 'AI Act' },
      ];

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ corpora }),
      });

      const result = await getCorpora();

      expect(mockFetch).toHaveBeenCalledWith('/api/corpora', {
        headers: { 'Content-Type': 'application/json' },
      });
      expect(result).toEqual(corpora);
    });
  });

  describe('getExamples', () => {
    it('fetches examples', async () => {
      const examples = {
        gdpr: {
          LEGAL: ['What is Article 1?'],
        },
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ examples }),
      });

      const result = await getExamples();

      expect(mockFetch).toHaveBeenCalledWith('/api/examples', {
        headers: { 'Content-Type': 'application/json' },
      });
      expect(result).toEqual(examples);
    });
  });

  describe('healthCheck', () => {
    it('fetches health status', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ status: 'ok', version: '1.0.0' }),
      });

      const result = await healthCheck();

      expect(mockFetch).toHaveBeenCalledWith('/api/health', {
        headers: { 'Content-Type': 'application/json' },
      });
      expect(result).toEqual({ status: 'ok', version: '1.0.0' });
    });
  });

  describe('Eval Case CRUD', () => {
    describe('listEvalCases', () => {
      it('fetches cases for a law', async () => {
        const response = { cases: [], total: 0 };

        mockFetch.mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(response),
        });

        const result = await listEvalCases('ai-act');

        expect(mockFetch).toHaveBeenCalledWith('/api/eval/cases/ai-act', {
          headers: { 'Content-Type': 'application/json' },
        });
        expect(result).toEqual(response);
      });

      it('encodes law parameter', async () => {
        mockFetch.mockResolvedValue({
          ok: true,
          json: () => Promise.resolve({ cases: [], total: 0 }),
        });

        await listEvalCases('law/with/slashes');

        expect(mockFetch).toHaveBeenCalledWith(
          '/api/eval/cases/law%2Fwith%2Fslashes',
          expect.any(Object)
        );
      });
    });

    describe('getEvalCase', () => {
      it('fetches a single case', async () => {
        const evalCase = { id: 'case-1', prompt: 'Test?' };

        mockFetch.mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(evalCase),
        });

        const result = await getEvalCase('ai-act', 'case-1');

        expect(mockFetch).toHaveBeenCalledWith('/api/eval/cases/ai-act/case-1', {
          headers: { 'Content-Type': 'application/json' },
        });
        expect(result).toEqual(evalCase);
      });
    });

    describe('createEvalCase', () => {
      it('creates a new case', async () => {
        const createData: EvalCaseCreate = {
          id: 'new-case',
          profile: 'LEGAL',
          prompt: 'Test prompt?',
          test_types: ['retrieval'],
          expected: {
            behavior: 'answer',
            must_include_any_of: [],
            must_include_any_of_2: [],
            must_include_all_of: [],
            must_not_include_any_of: [],
            contract_check: false,
            min_citations: null,
            max_citations: null,
            allow_empty_references: false,
            must_have_article_support_for_normative: true,
            notes: '',
          },
        };

        const createdCase = { ...createData, origin: 'manual' };

        mockFetch.mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(createdCase),
        });

        const result = await createEvalCase('ai-act', createData);

        expect(mockFetch).toHaveBeenCalledWith('/api/eval/cases/ai-act', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(createData),
        });
        expect(result).toEqual(createdCase);
      });
    });

    describe('updateEvalCase', () => {
      it('updates an existing case', async () => {
        const updateData: EvalCaseUpdate = {
          prompt: 'Updated prompt?',
        };

        const updatedCase = { id: 'case-1', prompt: 'Updated prompt?' };

        mockFetch.mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(updatedCase),
        });

        const result = await updateEvalCase('ai-act', 'case-1', updateData);

        expect(mockFetch).toHaveBeenCalledWith('/api/eval/cases/ai-act/case-1', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(updateData),
        });
        expect(result).toEqual(updatedCase);
      });
    });

    describe('deleteEvalCase', () => {
      it('deletes a case', async () => {
        mockFetch.mockResolvedValue({
          ok: true,
        });

        await deleteEvalCase('ai-act', 'case-1');

        expect(mockFetch).toHaveBeenCalledWith('/api/eval/cases/ai-act/case-1', {
          method: 'DELETE',
        });
      });
    });

    describe('duplicateEvalCase', () => {
      it('duplicates a case', async () => {
        const duplicatedCase = { id: 'case-1-copy', prompt: 'Test?' };

        mockFetch.mockResolvedValue({
          ok: true,
          json: () => Promise.resolve(duplicatedCase),
        });

        const result = await duplicateEvalCase('ai-act', 'case-1');

        expect(mockFetch).toHaveBeenCalledWith('/api/eval/cases/ai-act/case-1/duplicate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });
        expect(result).toEqual(duplicatedCase);
      });
    });
  });

  describe('listAnchors', () => {
    it('fetches anchors for a law', async () => {
      const response = { anchors: [{ id: 'art-1', display: 'Article 1' }] };

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(response),
      });

      const result = await listAnchors('ai-act');

      expect(mockFetch).toHaveBeenCalledWith('/api/admin/corpus/ai-act/anchors', {
        headers: { 'Content-Type': 'application/json' },
      });
      expect(result).toEqual(response);
    });

    it('includes query parameter when provided', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ anchors: [] }),
      });

      await listAnchors('ai-act', 'article');

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/admin/corpus/ai-act/anchors?q=article',
        expect.any(Object)
      );
    });
  });

  describe('runSingleCase', () => {
    it('runs a single eval case', async () => {
      const request = {
        law: 'ai-act',
        run_mode: 'retrieval_only' as const,
        prompt: 'Test?',
        profile: 'LEGAL' as const,
      };

      const response = {
        passed: true,
        duration_ms: 1000,
        scores: {},
        answer: 'Result',
        references: [],
        test_definition: {},
        retrieval_metrics: {},
      };

      mockFetch.mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(response),
      });

      const result = await runSingleCase(request);

      expect(mockFetch).toHaveBeenCalledWith('/api/eval/run-single', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      expect(result).toEqual(response);
    });
  });

  describe('streamAnswer', () => {
    it('returns abort function', () => {
      mockFetch.mockResolvedValue({
        ok: true,
        body: {
          getReader: () => ({
            read: () => Promise.resolve({ done: true, value: undefined }),
          }),
        },
      });

      const abort = streamAnswer(
        { question: 'Test?', law: 'ai-act', profile: 'LEGAL' },
        vi.fn(),
        vi.fn(),
        vi.fn()
      );

      expect(typeof abort).toBe('function');
    });

    it('sends POST request to stream endpoint', async () => {
      let readCalled = false;

      mockFetch.mockResolvedValue({
        ok: true,
        body: {
          getReader: () => ({
            read: () => {
              if (!readCalled) {
                readCalled = true;
                return Promise.resolve({ done: true, value: undefined });
              }
              return Promise.resolve({ done: true, value: undefined });
            },
          }),
        },
      });

      const request: AskRequest = { question: 'Test?', law: 'ai-act', profile: 'LEGAL' };

      streamAnswer(request, vi.fn(), vi.fn(), vi.fn());

      // Wait for async fetch to be called
      await vi.waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          '/api/ask/stream',
          expect.objectContaining({
            method: 'POST',
            body: JSON.stringify(request),
          })
        );
      });
    });

    it('calls onError when response is not ok', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Server Error',
      });

      const onError = vi.fn();

      streamAnswer(
        { question: 'Test?', law: 'ai-act', profile: 'LEGAL' },
        vi.fn(),
        vi.fn(),
        onError
      );

      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalled();
      });
    });

    it('calls onError when no response body', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        body: null,
      });

      const onError = vi.fn();

      streamAnswer(
        { question: 'Test?', law: 'ai-act', profile: 'LEGAL' },
        vi.fn(),
        vi.fn(),
        onError
      );

      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalled();
      });
    });

    it('calls onChunk for chunk events', async () => {
      const encoder = new TextEncoder();
      let readCount = 0;

      mockFetch.mockResolvedValue({
        ok: true,
        body: {
          getReader: () => ({
            read: () => {
              readCount++;
              if (readCount === 1) {
                return Promise.resolve({
                  done: false,
                  value: encoder.encode('data: {"type":"chunk","content":"Hello"}\n'),
                });
              }
              return Promise.resolve({ done: true, value: undefined });
            },
          }),
        },
      });

      const onChunk = vi.fn();

      streamAnswer(
        { question: 'Test?', law: 'ai-act', profile: 'LEGAL' },
        onChunk,
        vi.fn(),
        vi.fn()
      );

      await vi.waitFor(() => {
        expect(onChunk).toHaveBeenCalledWith('Hello');
      });
    });

    it('calls onResult for result events', async () => {
      const encoder = new TextEncoder();
      let readCount = 0;
      const resultData = { answer: 'Result', references: [] };

      mockFetch.mockResolvedValue({
        ok: true,
        body: {
          getReader: () => ({
            read: () => {
              readCount++;
              if (readCount === 1) {
                return Promise.resolve({
                  done: false,
                  value: encoder.encode(`data: {"type":"result","data":${JSON.stringify(resultData)}}\n`),
                });
              }
              return Promise.resolve({ done: true, value: undefined });
            },
          }),
        },
      });

      const onResult = vi.fn();

      streamAnswer(
        { question: 'Test?', law: 'ai-act', profile: 'LEGAL' },
        vi.fn(),
        onResult,
        vi.fn()
      );

      await vi.waitFor(() => {
        expect(onResult).toHaveBeenCalledWith(resultData);
      });
    });

    it('calls onError for error events', async () => {
      const encoder = new TextEncoder();
      let readCount = 0;

      mockFetch.mockResolvedValue({
        ok: true,
        body: {
          getReader: () => ({
            read: () => {
              readCount++;
              if (readCount === 1) {
                return Promise.resolve({
                  done: false,
                  value: encoder.encode('data: {"type":"error","message":"Something went wrong"}\n'),
                });
              }
              return Promise.resolve({ done: true, value: undefined });
            },
          }),
        },
      });

      const onError = vi.fn();

      streamAnswer(
        { question: 'Test?', law: 'ai-act', profile: 'LEGAL' },
        vi.fn(),
        vi.fn(),
        onError
      );

      await vi.waitFor(() => {
        expect(onError).toHaveBeenCalledWith(expect.any(ApiError));
      });
    });

    it('handles [DONE] message', async () => {
      const encoder = new TextEncoder();
      let readCount = 0;

      mockFetch.mockResolvedValue({
        ok: true,
        body: {
          getReader: () => ({
            read: () => {
              readCount++;
              if (readCount === 1) {
                return Promise.resolve({
                  done: false,
                  value: encoder.encode('data: [DONE]\n'),
                });
              }
              return Promise.resolve({ done: true, value: undefined });
            },
          }),
        },
      });

      const onChunk = vi.fn();
      const onResult = vi.fn();

      streamAnswer(
        { question: 'Test?', law: 'ai-act', profile: 'LEGAL' },
        onChunk,
        onResult,
        vi.fn()
      );

      // Wait a bit to ensure processing completes
      await new Promise((r) => setTimeout(r, 100));

      // Neither callback should be called for [DONE]
      expect(onChunk).not.toHaveBeenCalled();
      expect(onResult).not.toHaveBeenCalled();
    });
  });
});
