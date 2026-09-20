// @ts-nocheck — legacy harness (covered by tests/ + runtime guards)
// Node.js ORT correctness: run faithful ONNX, compare to Python fixtures,
// replicate temperature/softmax/confidence exactly (via shared modules).
import * as ort from 'onnxruntime-node';
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { softmax, tempBucket } from './laya-postprocess.mjs';
import { buildFeeds } from './laya-feed.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

function confidenceFromProbsLocal(p, k) {
  if (k < 2) return 1.0;
  let ent = 0;
  for (let i = 0; i < k; i++) ent -= p[i] * Math.log(Math.max(1e-12, Math.min(1.0, p[i])));
  return Math.min(1.0, Math.max(0.0, 1.0 - ent / Math.log(k)));
}

function maxAbsDiff(a, b, maskOrNull) {
  let m = 0;
  if (maskOrNull) {
    for (let i = 0; i < a.length; i++)
      for (let j = 0; j < a[0].length; j++) {
        if (!maskOrNull[i][j]) continue;
        m = Math.max(m, Math.abs(a[i][j] - b[i][j]));
      }
  } else {
    for (let i = 0; i < a.length; i++)
      for (let j = 0; j < a[i].length; j++) m = Math.max(m, Math.abs(a[i][j] - b[i][j]));
  }
  return m;
}

const modelArg = process.argv.find((a) => a.startsWith('--model='));
const modelPath = modelArg ? modelArg.slice('--model='.length) : join(root, 'models', 'laya-faithful.onnx');
console.log('loading', modelPath);
const sess = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
console.log('inputs:', sess.inputNames, 'outputs:', sess.outputNames);

const files = readdirSync(join(root, 'js', 'fixtures'))
  .filter((f) => f.endsWith('.json'))
  .sort();
let allPass = true;
for (const f of files) {
  const fx = JSON.parse(readFileSync(join(root, 'js', 'fixtures', f), 'utf8'));
  const { batch: B, seq_len: S, kmax: K } = fx;
  const feeds = buildFeeds(ort, {
    inputIds: fx.inputs.input_ids,
    attentionMask: fx.inputs.attention_mask,
    markerPos: fx.inputs.marker_pos,
    markerMask: fx.inputs.marker_mask,
    qtype: fx.inputs.qtype,
  });
  void B;
  void S;
  void K;
  const out = await sess.run(feeds);
  const logits = Array.from(out.logits.data);
  const act = Array.from(out.act_logits.data);
  const jsLogits = [];
  for (let i = 0; i < B; i++) jsLogits.push(logits.slice(i * K, (i + 1) * K));
  const jsAct = [];
  for (let i = 0; i < B; i++) jsAct.push(act.slice(i * 2, (i + 1) * 2));

  const diffLogits = maxAbsDiff(jsLogits, fx.expected_torch_logits, fx.inputs.marker_mask);
  const diffAct = maxAbsDiff(jsAct, fx.expected_torch_act, null);

  // calibrated comparison (JS postprocessing vs Python expected answers)
  let pdrift = 0,
    adrift = 0,
    labelsOk = true;
  for (let r = 0; r < B; r++) {
    const qid = fx.qids[r],
      k = fx.k_per_q[r],
      qt = fx.qtypes[r];
    const bucket = tempBucket(qt, k);
    const tScale = fx.temperature.temperature_by_options[bucket] ?? fx.temperature.temperature[qt];
    const z = jsLogits[r].slice(0, k).map((v) => v / Math.max(1e-3, tScale));
    const p = softmax(z);
    const actP = softmax(jsAct[r]);
    const exp = fx.expected_answers[qid];
    if (exp.probs) {
      const keys = Object.keys(exp.probs);
      keys.forEach((kk, i) => {
        pdrift = Math.max(pdrift, Math.abs(p[i] - exp.probs[kk]));
      });
      if (exp.choice !== undefined) {
        let best = 0;
        for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
        const choice = keys[best];
        if (choice !== exp.choice) labelsOk = false;
      }
    } else {
      pdrift = Math.max(pdrift, Math.abs(p[1] - exp.noul));
    }
    adrift = Math.max(adrift, Math.abs(actP[0] - exp.act));
  }
  void confidenceFromProbsLocal;
  const pass = diffLogits < 1e-4 && pdrift < 1e-4 && adrift < 1e-4 && labelsOk;
  allPass &&= pass;
  console.log(
    `${fx.name}: logits max=${diffLogits.toExponential(2)} act max=${diffAct.toExponential(2)} pdrift=${pdrift.toExponential(2)} adrift=${adrift.toExponential(2)} labels=${labelsOk ? 'OK' : 'DIFF'} -> ${pass ? 'PASS' : 'CHECK'}`,
  );
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exitCode = allPass ? 0 : 1;
