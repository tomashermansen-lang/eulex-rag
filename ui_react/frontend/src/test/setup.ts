import '@testing-library/jest-dom';
import { vi } from 'vitest';

// Global clipboard mock for jsdom environment
// This creates a mockable clipboard that tests can spy on
const clipboardMock = {
  writeText: vi.fn().mockResolvedValue(undefined),
  readText: vi.fn().mockResolvedValue(''),
};

Object.defineProperty(navigator, 'clipboard', {
  value: clipboardMock,
  writable: true,
  configurable: true,
});

// Export for tests that need to access the mock
export { clipboardMock };
