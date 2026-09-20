// CPU action head for split-graph: mirrors DecisionModel forward tail exactly.
// Inputs: logits [B][K] (uncalibrated), markerMask [B][K] bool, pooled [B][1024],
// weights {w0:[256][1028], b0:[256], w2:[2][256], b2:[2]} (from export/act_head.npz).
// Output: act_logits [B][2].
// GELU is torch.nn.functional.gelu exact (erf); ENT_EPS 1e-9 mirrors Python
// DecisionModel tail (vs 1e-12 in calibration confidence) — intentional.
import { softmax } from './laya-postprocess.mjs';
import { LayaInferenceError } from './laya-errors.mjs';

export const ACT_ENT_EPS = 1e-9;
export const POOLED_DIM = 1024;
export function erf(x) {
  // Cody rational approximation, |err| < 1.2e-7 (single-precision sufficient; outputs compared at 1e-4)
  const a1 = 0.254829592,
    a2 = -0.284496736,
    a3 = 1.421413741,
    a4 = -1.453152027,
    a5 = 1.061405429,
    p = 0.3275911;
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return s * y;
}
export function gelu(x) {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}
export function actionLogits(logits, markerMask, pooled, w) {
  if (!w?.w0 || !w?.b0 || !w?.w2 || !w?.b2) {
    throw new LayaInferenceError('actionLogits: malformed act_head weights (need w0/b0/w2/b2)');
  }
  if (w.b0.length !== 256 || w.w0.length !== 256 || w.w2.length !== 2) {
    throw new LayaInferenceError(`actionLogits: bad head dims b0=${w.b0.length} w0=${w.w0.length} w2=${w.w2.length}`);
  }
  const B = logits.length;
  const out = [];
  for (let r = 0; r < B; r++) {
    const row = logits[r],
      mm = markerMask[r];
    if (!row?.length || row.length < 1) throw new LayaInferenceError(`actionLogits: row ${r} needs K>=1`);
    if (!pooled[r] || pooled[r].length !== POOLED_DIM) {
      throw new LayaInferenceError(`actionLogits: pooled[${r}] must be ${POOLED_DIM}, got ${pooled[r]?.length}`);
    }
    const p = softmax(row);
    const kk = Math.max(2, mm.filter(Boolean).length);
    let ent = 0;
    for (let i = 0; i < p.length; i++) ent -= p[i] * Math.log(Math.max(ACT_ENT_EPS, p[i]));
    ent /= Math.log(kk);
    const sorted = [...p].sort((a, b) => b - a);
    const top0 = sorted[0] ?? 0;
    const top1 = sorted[1] ?? top0;
    const feats = [top0, top0 - top1, ent, kk / 255];
    const inp = [...pooled[r], ...feats];
    if (inp.length !== POOLED_DIM + 4) throw new LayaInferenceError(`actionLogits: inp ${inp.length} != 1028`);
    const h0 = w.b0.map((b, i) => gelu(w.w0[i].reduce((a, v, j) => a + v * inp[j], b)));
    out.push(w.b2.map((b, i) => w.w2[i].reduce((a, v, j) => a + v * h0[j], b)));
  }
  return out;
}
