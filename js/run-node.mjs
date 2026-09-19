// Node.js ORT correctness: run faithful ONNX, compare to Python fixtures,
// replicate temperature/softmax/confidence exactly.
import * as ort from 'onnxruntime-node';
import { readFileSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const QTYPE_NAMES = { 0: 'choice', 1: 'score', 2: 'noul' };

function tempBucket(qtype, k) {
  const size = k <= 2 ? '2' : k <= 5 ? '3-5' : k <= 10 ? '6-10' : '11+';
  return `${QTYPE_NAMES[qtype]}:${size}`;
}
function softmax(z) {
  const m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}
function confidenceFromProbs(p, k) {
  if (k < 2) return 1.0;
  let ent = 0;
  for (let i = 0; i < k; i++) ent -= p[i] * Math.log(Math.max(1e-12, Math.min(1.0, p[i])));
  return Math.min(1.0, Math.max(0.0, 1.0 - ent / Math.log(k)));
}
function toBigInt64(nested) {
  const flat = nested.flat(Infinity).map(v => BigInt(v));
  return BigInt64Array.from(flat);
}
function toBoolU8(nested) {
  const flat = nested.flat(Infinity).map(v => v ? 1 : 0);
  return Uint8Array.from(flat);
}
function maxAbsDiff(a, b, maskOrNull) {
  let m = 0;
  if (maskOrNull) {
    for (let i = 0; i < a.length; i++) for (let j = 0; j < a[0].length; j++) {
      if (!maskOrNull[i][j]) continue;
      m = Math.max(m, Math.abs(a[i][j] - b[i][j]));
    }
  } else {
    for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++)
      m = Math.max(m, Math.abs(a[i][j] - b[i][j]));
  }
  return m;
}

const modelPath = join(root, 'models', 'laya-faithful.onnx');
console.log('loading', modelPath);
const sess = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
console.log('inputs:', sess.inputNames, 'outputs:', sess.outputNames);

const files = readdirSync(join(root, 'js', 'fixtures')).filter(f => f.endsWith('.json')).sort();
let allPass = true;
for (const f of files) {
  const fx = JSON.parse(readFileSync(join(root, 'js', 'fixtures', f), 'utf8'));
  const { batch: B, seq_len: S, kmax: K } = fx;
  const feeds = {
    input_ids: new ort.Tensor('int64', toBigInt64(fx.inputs.input_ids), [B, S]),
    attention_mask: new ort.Tensor('int64', toBigInt64(fx.inputs.attention_mask), [B, S]),
    marker_pos: new ort.Tensor('int64', toBigInt64(fx.inputs.marker_pos), [B, K]),
    marker_mask: new ort.Tensor('bool', toBoolU8(fx.inputs.marker_mask), [B, K]),
    qtype: new ort.Tensor('int64', toBigInt64([fx.inputs.qtype]), [B]),
  };
  const out = await sess.run(feeds);
  const logits = Array.from(out.logits.data);
  const act = Array.from(out.act_logits.data);
  const jsLogits = []; for (let i = 0; i < B; i++) jsLogits.push(logits.slice(i * K, (i + 1) * K));
  const jsAct = []; for (let i = 0; i < B; i++) jsAct.push(act.slice(i * 2, (i + 1) * 2));

  const diffLogits = maxAbsDiff(jsLogits, fx.expected_torch_logits, fx.inputs.marker_mask);
  const diffAct = maxAbsDiff(jsAct, fx.expected_torch_act, null);

  // calibrated comparison (JS postprocessing vs Python expected answers)
  let pdrift = 0, adrift = 0, labelsOk = true;
  for (let r = 0; r < B; r++) {
    const qid = fx.qids[r], k = fx.k_per_q[r], qt = fx.qtypes[r];
    const bucket = tempBucket(qt, k);
    const tScale = fx.temperature.temperature_by_options[bucket] ?? fx.temperature.temperature[qt];
    const z = jsLogits[r].slice(0, k).map(v => v / Math.max(1e-3, tScale));
    const p = softmax(z);
    const actP = softmax(jsAct[r]);
    const exp = fx.expected_answers[qid];
    if (exp.probs) {
      const keys = Object.keys(exp.probs);
      keys.forEach((kk, i) => { pdrift = Math.max(pdrift, Math.abs(p[i] - exp.probs[kk])); });
      if (exp.choice !== undefined) {
        const choice = keys[p.indexOf(Math.max(...p))];
        if (choice !== exp.choice) labelsOk = false;
      }
    } else {
      pdrift = Math.max(pdrift, Math.abs(p[1] - exp.noul));
    }
    adrift = Math.max(adrift, Math.abs(actP[0] - exp.act));
  }
  const pass = diffLogits < 1e-4 && pdrift < 1e-4 && adrift < 1e-4 && labelsOk;
  allPass &&= pass;
  console.log(`${fx.name}: logits max=${diffLogits.toExponential(2)} act max=${diffAct.toExponential(2)} pdrift=${pdrift.toExponential(2)} adrift=${adrift.toExponential(2)} labels=${labelsOk ? 'OK' : 'DIFF'} -> ${pass ? 'PASS' : 'CHECK'}`);
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exit(allPass ? 0 : 1);
