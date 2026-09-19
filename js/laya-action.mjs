// CPU action head for split-graph: mirrors DecisionModel forward tail exactly.
// Inputs: logits [B][K] (uncalibrated), markerMask [B][K] bool, pooled [B][1024],
// weights {w0:[256][1028], b0:[256], w2:[2][256], b2:[2]} (from export/act_head.npz).
// Output: act_logits [B][2].
export function erf(x) {
  // Cody rational approximation, |err| < 1.2e-7 (single-precision sufficient; outputs compared at 1e-4)
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return s * y;
}
export function gelu(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }
export function actionLogits(logits, markerMask, pooled, w) {
  const B = logits.length;
  const out = [];
  for (let r = 0; r < B; r++) {
    const row = logits[r], mm = markerMask[r];
    const m = Math.max(...row);
    const e = row.map(v => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    const p = e.map(v => v / s);
    const kk = Math.max(2, mm.filter(Boolean).length);
    let ent = 0;
    for (let i = 0; i < p.length; i++) ent -= p[i] * Math.log(Math.max(1e-9, p[i]));
    ent /= Math.log(kk);
    const sorted = [...p].sort((a, b) => b - a);
    const feats = [sorted[0], sorted[0] - sorted[1], ent, kk / 255];
    const inp = [...pooled[r], ...feats];
    const h0 = w.b0.map((b, i) => gelu(w.w0[i].reduce((a, v, j) => a + v * inp[j], b)));
    out.push(w.b2.map((b, i) => w.w2[i].reduce((a, v, j) => a + v * h0[j], b)));
  }
  return out;
}
