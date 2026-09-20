import React from 'react';
import type { ReactNode } from 'react';
import type { AccuracyReport, BaselineReport, Fp16Report, ParityReport, PublicBenchmarkReport } from '../lib/reports.ts';
import parityData from '@laya/test-vectors/reports/parity-report.json';
import baselineData from '@laya/test-vectors/reports/mac-baseline.json';
import fp16Data from '@laya/test-vectors/reports/fp16-parity.json';
import accData from '@laya/test-vectors/reports/accuracy-report.json';
import pubData from '@laya/test-vectors/reports/public-benchmark.json';

// Tables render from JSON reports; phase board stays a curated snapshot
// (it summarizes process state, not a single report file).
function ReportTable<T>({
  title,
  data,
  error,
  columns,
  rows,
}: {
  title: string;
  data: T | null;
  error: string | null;
  columns: string[];
  rows: (d: T) => ReactNode;
}): React.JSX.Element {
  return (
    <>
      <h3>{title}</h3>
      {error ? (
        <div className="empty" role="alert">
          failed to load: {error}
        </div>
      ) : !data ? (
        <div className="empty">loading…</div>
      ) : (
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>{rows(data)}</tbody>
        </table>
      )}
    </>
  );
}

export function ProgressTab(): React.JSX.Element {
  // Reports are build-time imports from @laya/test-vectors (regenerate via
  // tools/export, then rebuild). No runtime fetch, no loading states.
  const parity = parityData as unknown as ParityReport;
  const baseline = baselineData as unknown as BaselineReport;
  const fp16 = fp16Data as unknown as Fp16Report;
  const acc = accData as unknown as AccuracyReport;
  const pub = pubData as unknown as PublicBenchmarkReport;
  return (
    <div>
      <h3>Phase board</h3>
      <table>
        <tbody>
          <tr>
            <td>✅ Phase 1</td>
            <td>Faithful FP32 ONNX export, variable-K, 5-fixture CPU parity PASS</td>
          </tr>
          <tr>
            <td>✅ Phase 2</td>
            <td>Browser library core: pure-JS BPE, split WebGPU, worker text→answer</td>
          </tr>
          <tr>
            <td>▶ Phase 3</td>
            <td>Precision and size (FP16 → INT8/4-bit with drift checks)</td>
          </tr>
          <tr>
            <td>⬜ Phase 4</td>
            <td>Native pick: CoreML spike blocked on toolchain (trace + new_ones gaps); MLX/laya.cpp untouched</td>
          </tr>
        </tbody>
      </table>
      <ReportTable
        title="Parity (torch CPU vs ORT CPU)"
        data={parity}
        error={null}
        columns={['fixture', 'max|logit|', 'prob drift', 'labels']}
        rows={(d) =>
          d.fixtures.map((f) => (
            <tr key={f.name}>
              <td>{f.name}</td>
              <td>{f.max_abs_logits.toExponential(1)}</td>
              <td>{f.max_calibrated_prob_drift.toExponential(1)}</td>
              <td>{f.labels_match ? 'OK' : 'DIFF'}</td>
            </tr>
          ))
        }
      />
      <ReportTable
        title="Native latency p50 (from mac-baseline.json)"
        data={baseline}
        error={null}
        columns={['device', 'questions', 'forward', 'end-to-end']}
        rows={(d) =>
          d.runs.map((r, i) => (
            <tr key={i}>
              <td>{r.device}</td>
              <td>{r.questions}</td>
              <td>{r.forward.p50_ms.toFixed(1)}ms</td>
              <td>{r.end_to_end.p50_ms.toFixed(1)}ms</td>
            </tr>
          ))
        }
      />
      <ReportTable
        title="FP16 drift vs torch FP32 (CPU, split model)"
        data={fp16}
        error={null}
        columns={['fixture', 'max|logit|', 'prob drift', 'conf drift', 'flips']}
        rows={(d) =>
          d.fixtures.map((f) => (
            <tr key={f.name}>
              <td>{f.name}</td>
              <td>{f.max_abs_logits_vs_torch.toExponential(1)}</td>
              <td>{f.max_calibrated_prob_drift.toExponential(1)}</td>
              <td>{f.max_confidence_drift.toExponential(1)}</td>
              <td>{f.label_flips.length ? f.label_flips.join(',') : 'none'}</td>
            </tr>
          ))
        }
      />
      <ReportTable
        title="Accuracy harness (13 weak directional checks, CPU)"
        data={acc}
        error={null}
        columns={['variant', 'score', 'fails']}
        rows={(d) =>
          d.adapters.map((a) => (
            <tr key={a.name}>
              <td>{a.name}</td>
              <td>
                {a.score}/{d.cases}
              </td>
              <td>
                {a.rows
                  .filter((r) => !r.pass)
                  .map((r) => r.id)
                  .join(', ') || '—'}
              </td>
            </tr>
          ))
        }
      />
      <ReportTable
        title="Public datasets (labelled, CPU) — community excluded (fixed K=2)"
        data={pub}
        error={null}
        columns={['suite', 'n', 'torch', 'ours fp32', 'ours fp16', 'p50 torch/ours']}
        rows={(d) =>
          Object.entries(d.suites).map(([sname, s]) => (
            <tr key={sname}>
              <td>{sname}</td>
              <td>{s.n}</td>
              <td>{s.adapters['torch-fp32']?.accuracy.toFixed(3) ?? '—'}</td>
              <td>{s.adapters['onnx-laya-split-single']?.accuracy.toFixed(3) ?? '—'}</td>
              <td>{s.adapters['onnx-laya-split-fp16']?.accuracy.toFixed(3) ?? '—'}</td>
              <td>
                {s.adapters['torch-fp32']?.p50_ms.toFixed(0) ?? '—'}/
                {s.adapters['onnx-laya-split-single']?.p50_ms.toFixed(0) ?? '—'}ms
              </td>
            </tr>
          ))
        }
      />
      <h3>Browser (measured snapshot, choice-2 B=1)</h3>
      <table>
        <tbody>
          <tr>
            <td>WASM full fp32</td>
            <td>756ms PASS</td>
          </tr>
          <tr>
            <td>WebGPU-basic full fp32</td>
            <td>1072ms PASS (default FAILs on fusion bug)</td>
          </tr>
          <tr>
            <td>WebGPU-basic split fp32</td>
            <td>306ms cold PASS — fastest accurate path</td>
          </tr>
          <tr>
            <td>WASM split fp16</td>
            <td>762ms warm, 1.57e-03 (matches CPU; no speedup)</td>
          </tr>
          <tr>
            <td>WebGPU-basic split fp16</td>
            <td>180ms warm but 4.11e-02 — fast, NOT shippable as-is</td>
          </tr>
          <tr>
            <td>CPU split int8 (405MB)</td>
            <td>no speedup, 12/13 accuracy (phishing flip) — rejected as-is</td>
          </tr>
          <tr>
            <td>WASM split 4-bit (395MB)</td>
            <td>1088ms, 4.48e-02, harness 13/13 on CPU</td>
          </tr>
          <tr>
            <td>WebGPU-basic split 4-bit</td>
            <td>219ms warm but 3.35e+00 — numerically broken, rejected</td>
          </tr>
        </tbody>
      </table>
      <p className="meta">Sources: @laya/test-vectors/reports · packages/laya-js/README.md</p>
    </div>
  );
}
