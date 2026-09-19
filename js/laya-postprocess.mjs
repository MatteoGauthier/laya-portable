// Shared calibration/postprocessing: mirrors Agent.system_one exactly.
// Used by Node check and browser worker. Inputs: raw logits/act (nested arrays).
const QTYPE_NAMES = { 0: 'choice', 1: 'score', 2: 'noul' };
export function tempBucket(qtype, k) {
  const size = k <= 2 ? '2' : k <= 5 ? '3-5' : k <= 10 ? '6-10' : '11+';
  return `${QTYPE_NAMES[qtype]}:${size}`;
}
export function softmax(z) {
  const m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}
export function confidenceFromProbs(p, k) {
  if (k < 2) return 1.0;
  let ent = 0;
  for (let i = 0; i < k; i++) ent -= p[i] * Math.log(Math.max(1e-12, Math.min(1.0, p[i])));
  return Math.min(1.0, Math.max(0.0, 1.0 - ent / Math.log(k)));
}
// questions: {qid: {type, instructions, criteria}} (user format); items: [{markers, qtype}];
// logits/act: [B][K] / [B][2]; temp: {temperature:[..], temperature_by_options:{..}}
// Returns {answers, usage} matching agent.predict shape (rounded to 4 like upstream API).
export function predictFromLogits(questions, items, logits, act, temp, nTokens) {
  const ids = Object.keys(questions);
  const actP = act.map(a => softmax(a));
  const answers = {};
  for (let r = 0; r < ids.length; r++) {
    const qid = ids[r], qdef = questions[qid];
    const k = items[r].markers.length, qt = items[r].qtype;
    const tScale = temp.temperature_by_options[tempBucket(qt, k)] ?? temp.temperature[qt];
    const z = logits[r].slice(0, k).map(v => v / Math.max(1e-3, tScale));
    const p = softmax(z);
    const ext = { act_probability: Math.round(actP[r][0] * 1e4) / 1e4 };
    if (qdef.type === 'choice') {
      const keys = Array.isArray(qdef.criteria) ? qdef.criteria.map(String) : Object.keys(qdef.criteria);
      const probs = {};
      keys.forEach((kk, i) => { probs[kk] = Math.round(p[i] * 1e4) / 1e4; });
      answers[qid] = { type: 'choice', choice: keys[p.indexOf(Math.max(...p))], probabilities: probs,
        confidence: Math.round(confidenceFromProbs(p, k) * 1e4) / 1e4, action: ext };
    } else if (qdef.type === 'score') {
      const exp = p.reduce((a, v, i) => a + i * v, 0);
      const probs = {};
      p.forEach((v, i) => { probs[String(i)] = Math.round(v * 1e4) / 1e4; });
      answers[qid] = { type: 'score', score: Math.round(exp * 1e4) / 1e4,
        legend: Object.fromEntries(qdef.criteria.map((c, i) => [String(i), c])),
        probabilities: probs, confidence: Math.round(confidenceFromProbs(p, k) * 1e4) / 1e4, action: ext };
    } else {
      const n = p[1];
      answers[qid] = { type: 'noul', noul: Math.round(n * 1e4) / 1e4,
        confidence: Math.round(Math.max(n, 1 - n) * 1e4) / 1e4, action: ext };
    }
  }
  return { model: 'laya-rl-agent', answers, usage: { input_tokens: nTokens, output_tokens: 0 } };
}
