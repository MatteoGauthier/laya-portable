import { describe, it, expect, vi, afterEach } from 'vitest';
import { fetchJson } from './api.ts';
import { PRESETS } from './presets.ts';

afterEach(() => vi.unstubAllGlobals());

describe('fetchJson', () => {
  it('returns parsed JSON on 200', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: true, json: async () => ({ a: 1 }) })),
    );
    await expect(fetchJson('/reports/x.json')).resolves.toEqual({ a: 1 });
  });
  it('throws on HTTP error (no silent loading…)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => ({ ok: false, status: 404 })),
    );
    await expect(fetchJson('/reports/x.json')).rejects.toThrow('HTTP 404');
  });
  it('throws on network failure', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new Error('down');
      }),
    );
    await expect(fetchJson('/reports/x.json')).rejects.toThrow('failed');
  });
});

describe('presets', () => {
  it('all presets are valid question sets', () => {
    for (const [name, p] of Object.entries(PRESETS)) {
      expect(Object.keys(p.questions).length, name).toBeGreaterThan(0);
      for (const q of Object.values(p.questions)) {
        expect(['choice', 'score', 'noul']).toContain(q.type);
      }
    }
  });
});
