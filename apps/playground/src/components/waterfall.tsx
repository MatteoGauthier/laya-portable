import React from 'react';
import type { PredictTimings } from '@laya/js/laya-types.ts';

export interface WaterfallSetup {
  download_ms: number;
  session_ms: number;
}

export interface WaterfallRow {
  label: string;
  ms: number;
  tone: 'setup' | 'compute' | 'overhead';
  offsetPct: number;
  widthPct: number;
}

/** Stack setup + compute phases against total; the remainder is honest overhead. */
export function buildWaterfall(
  timings: PredictTimings,
  setup: WaterfallSetup,
): { rows: WaterfallRow[]; totalMs: number; overheadMs: number } {
  const phases: { label: string; ms: number; tone: 'setup' | 'compute' }[] = [
    { label: 'download', ms: Math.max(0, setup.download_ms), tone: 'setup' },
    { label: 'session', ms: Math.max(0, setup.session_ms), tone: 'setup' },
    { label: 'tokenize', ms: Math.max(0, timings.tokenize_ms), tone: 'compute' },
    { label: 'inference', ms: Math.max(0, timings.inference_ms), tone: 'compute' },
    { label: 'postprocess', ms: Math.max(0, timings.postprocess_ms), tone: 'compute' },
  ];
  const totalMs = Math.max(0, timings.total_ms);
  const used = phases.reduce((a, p) => a + p.ms, 0);
  const overheadMs = Math.max(0, totalMs - used);
  const rows: WaterfallRow[] = [];
  let offset = 0;
  const pct = (ms: number): number => (totalMs > 0 ? (ms / totalMs) * 100 : 0);
  for (const p of phases) {
    rows.push({ label: p.label, ms: p.ms, tone: p.tone, offsetPct: pct(offset), widthPct: pct(p.ms) });
    offset += p.ms;
  }
  if (overheadMs > 0) {
    rows.push({
      label: 'overhead',
      ms: overheadMs,
      tone: 'overhead',
      offsetPct: pct(offset),
      widthPct: pct(overheadMs),
    });
  }
  return { rows, totalMs, overheadMs };
}

export function Waterfall({ timings, setup }: { timings: PredictTimings; setup: WaterfallSetup }): React.JSX.Element {
  const { rows, totalMs } = buildWaterfall(timings, setup);
  return (
    <div className="waterfall" role="img" aria-label={`run waterfall, total ${totalMs} milliseconds`}>
      {rows.map((r) => (
        <div className="wf-row" key={r.label}>
          <span className="wf-label">{r.label}</span>
          <div className="wf-track">
            <div
              className={`wf-fill wf-${r.tone}`}
              style={{ marginLeft: `${r.offsetPct}%`, width: `${Math.max(r.widthPct, r.ms > 0 ? 0.5 : 0)}%` }}
            />
          </div>
          <span className="wf-val">{r.ms}ms</span>
        </div>
      ))}
      <div className="wf-total">total {totalMs}ms</div>
    </div>
  );
}
