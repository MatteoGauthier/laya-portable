import { describe, it, expect } from 'vitest';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { buildWaterfall, Waterfall } from './waterfall.tsx';
import { computeLag } from '../hooks/useWorker.tsx';

void React;

const timings = { tokenize_ms: 15, inference_ms: 820, postprocess_ms: 8, total_ms: 6649 };
const setup = { download_ms: 5100, session_ms: 700 };

describe('buildWaterfall', () => {
  it('stacks setup then compute phases with offsets', () => {
    const { rows, totalMs, overheadMs } = buildWaterfall(timings, setup);
    expect(totalMs).toBe(6649);
    expect(rows.map((r) => r.label)).toEqual([
      'download',
      'session',
      'tokenize',
      'inference',
      'postprocess',
      'overhead',
    ]);
    // offsets chain: each row starts where the previous ends
    for (let i = 1; i < rows.length; i++) {
      const prev = rows[i - 1];
      const cur = rows[i];
      if (!prev || !cur) throw new Error('missing row');
      const prevEnd = (prev.offsetPct / 100) * totalMs + prev.ms;
      expect(Math.abs((cur.offsetPct / 100) * totalMs - prevEnd)).toBeLessThan(1);
    }
    expect(overheadMs).toBe(6649 - (5100 + 700 + 15 + 820 + 8));
    // widths sum to ~100%
    const widthSum = rows.reduce((a, r) => a + r.widthPct, 0);
    expect(Math.abs(widthSum - 100)).toBeLessThan(0.01);
  });
  it('handles cached runs (zero setup) and zero total without NaN', () => {
    const cached = buildWaterfall(
      { tokenize_ms: 1, inference_ms: 76, postprocess_ms: 6, total_ms: 82 },
      { download_ms: 0, session_ms: 0 },
    );
    expect(cached.rows.filter((r) => r.tone === 'setup').every((r) => r.widthPct === 0)).toBe(true);
    const zero = buildWaterfall(
      { tokenize_ms: 0, inference_ms: 0, postprocess_ms: 0, total_ms: 0 },
      { download_ms: 0, session_ms: 0 },
    );
    expect(zero.rows.every((r) => Number.isFinite(r.widthPct) && Number.isFinite(r.offsetPct))).toBe(true);
  });
});

describe('Waterfall', () => {
  it('renders one row per phase with ms values', () => {
    render(<Waterfall timings={timings} setup={setup} />);
    expect(screen.getByText('download')).toBeInTheDocument();
    expect(screen.getByText('5100ms')).toBeInTheDocument();
    expect(screen.getByText('total 6649ms')).toBeInTheDocument();
  });
});

describe('computeLag', () => {
  it('is zero for on-time ticks and positive for stalls', () => {
    expect(computeLag(1000, 1000)).toBe(0);
    expect(computeLag(1005, 1000)).toBe(5);
    expect(computeLag(800, 1000)).toBe(0);
    expect(computeLag(60_000, 1000)).toBe(59_000);
  });
});
