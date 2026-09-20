// Shared calibration/postprocessing: mirrors Agent.system_one exactly.
// Used by Node SDK, checks, and browser worker. Inputs: raw logits/act (nested arrays).
// Single source of truth for softmax/confidence/tempBucket — do not duplicate.
import { LayaEncodeError } from './laya-errors.mjs';

const QTYPE_NAMES = { 0: 'choice', 1: 'score', 2: 'noul' };
export function tempBucket(qtype, k) {
  const size = k <= 2 ? '2' : k <= 5 ? '3-5' : k <= 10 ? '6-10' : '11+';
  return `${QTYPE_NAMES[qtype]}:${size}`;
}
/** Numerically stable softmax (loop max — no spread overflow). */
export function softmax(z) {
  if (!z.length) throw new LayaEncodeError('softmax: empty logits (K>=1 required)');
  let m = z[0];
  for (let i = 1; i < z.length; i++) if (z[i] > m) m = z[i];
  const e = z.map((v) => Math.exp(v - m));
  let s = 0;
  for (const v of e) s += v;
  return e.map((v) => v / s);
}
export const ENT_EPS_CALIBRATION = 1e-12;
export function confidenceFromProbs(p, k) {
  if (k < 2) return 1.0;
  let ent = 0;
  for (let i = 0; i < k; i++) ent -= p[i] * Math.log(Math.max(ENT_EPS_CALIBRATION, Math.min(1.0, p[i])));
  return Math.min(1.0, Math.max(0.0, 1.0 - ent / Math.log(k)));
}
// questions: {qid: {type, instructions, criteria}} (user format); items: [{markers, qtype}];
// logits/act: [B][K] / [B][2]; temp: {temperature:[..], temperature_by_options:{..}}
// Returns {answers, usage} matching agent.predict shape (rounded to 4 like upstream API).
export function predictFromLogits(questions, items, logits, act, temp, nTokens) {
  const ids = Object.keys(questions);
  if (!ids.length) throw new LayaEncodeError('predictFromLogits: no questions');
  const actP = act.map((a) => softmax(a));
  const answers = {};
  for (let r = 0; r < ids.length; r++) {
    const qid = ids[r],
      qdef = questions[qid];
    const k = items[r].markers.length,
      qt = items[r].qtype;
    if (k < 1) throw new LayaEncodeError(`predictFromLogits: K>=1 required for ${qid}`);
    // temp clamp 1e-3 mirrors Python Agent.system_one (verify if upstream changes).
    const tScale = temp.temperature_by_options[tempBucket(qt, k)] ?? temp.temperature[qt];
    const z = logits[r].slice(0, k).map((v) => v / Math.max(1e-3, tScale));
    const p = softmax(z);
    // argmax without spread (first-wins on ties, like np.argmax).
    let best = 0;
    for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
    const ext = { act_probability: Math.round(actP[r][0] * 1e4) / 1e4 };
    if (qdef.type === 'choice') {
      const keys = Array.isArray(qdef.criteria) ? qdef.criteria.map(String) : Object.keys(qdef.criteria);
      if (keys.length !== k) throw new LayaEncodeError(`choice ${qid}: keys ${keys.length} != K ${k}`);
      const probs = {};
      keys.forEach((kk, i) => {
        probs[kk] = Math.round(p[i] * 1e4) / 1e4;
      });
      answers[qid] = {
        type: 'choice',
        choice: keys[best],
        probabilities: probs,
        confidence: Math.round(confidenceFromProbs(p, k) * 1e4) / 1e4,
        action: ext,
      };
    } else if (qdef.type === 'score') {
      if (!Array.isArray(qdef.criteria)) throw new LayaEncodeError(`score ${qid}: criteria must be an array`);
      if (qdef.criteria.length !== k)
        throw new LayaEncodeError(`score ${qid}: criteria ${qdef.criteria.length} != K ${k}`);
      const exp = p.reduce((a, v, i) => a + i * v, 0);
      const probs = {};
      p.forEach((v, i) => {
        probs[String(i)] = Math.round(v * 1e4) / 1e4;
      });
      answers[qid] = {
        type: 'score',
        score: Math.round(exp * 1e4) / 1e4,
        legend: Object.fromEntries(qdef.criteria.map((c, i) => [String(i), c])),
        probabilities: probs,
        confidence: Math.round(confidenceFromProbs(p, k) * 1e4) / 1e4,
        action: ext,
      };
    } else {
      if (k !== 2) throw new LayaEncodeError(`noul ${qid}: K must be 2, got ${k}`);
      const n = p[1];
      answers[qid] = {
        type: 'noul',
        noul: Math.round(n * 1e4) / 1e4,
        confidence: Math.round(Math.max(n, 1 - n) * 1e4) / 1e4,
        action: ext,
      };
    }
  }
  return { model: 'laya-rl-agent', answers, usage: { input_tokens: nTokens, output_tokens: 0 } };
}
