// Node.js ORT correctness: run faithful ONNX, compare to Python fixtures,
// replicate temperature/softmax/confidence exactly (via shared modules).
// Runs directly on Node >=22 via type-stripping (no build step).
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { softmax, tempBucket } from '../src/laya-postprocess.ts';
import { toFeedData } from '../src/laya-feed.ts';
import { MODELS_DIR } from '../src/laya-paths.ts';
import { FIXTURE_NAMES, fixturePath } from '../tests/test-helpers.ts';
import type { CollatedBatch } from '../src/laya-types.ts';

interface Fixture {
  name: string;
  batch: number;
  seq_len: number;
  kmax: number;
  qids: string[];
  k_per_q: number[];
  qtypes: number[];
  inputs: {
    input_ids: number[][];
    attention_mask: number[][];
    marker_pos: number[][];
    marker_mask: boolean[][];
    qtype: number[];
  };
  expected_torch_logits: number[][];
  expected_torch_act: number[][];
  temperature: { temperature: number[]; temperature_by_options: Record<string, number> };
  expected_answers: Record<string, { probs?: Record<string, number>; choice?: string; noul?: number; act: number }>;
}

function loadFixture(path: string): Fixture {
  return JSON.parse(readFileSync(path, 'utf8')) as Fixture;
}

function maxAbsDiff(a: number[][], b: number[][], maskOrNull: boolean[][] | null): number {
  let m = 0;
  if (maskOrNull) {
    for (let i = 0; i < a.length; i++) {
      const ra = a[i] as number[];
      const rb = b[i] as number[];
      const rm = maskOrNull[i] as boolean[];
      for (let j = 0; j < ra.length; j++) {
        if (!rm[j]) continue;
        m = Math.max(m, Math.abs((ra[j] as number) - (rb[j] as number)));
      }
    }
  } else {
    for (let i = 0; i < a.length; i++) {
      const ra = a[i] as number[];
      const rb = b[i] as number[];
      for (let j = 0; j < ra.length; j++) m = Math.max(m, Math.abs((ra[j] as number) - (rb[j] as number)));
    }
  }
  return m;
}

const modelArg = process.argv.find((a) => a.startsWith('--model='));
const modelPath = modelArg ? modelArg.slice('--model='.length) : join(MODELS_DIR, 'laya-faithful.onnx');
console.log('loading', modelPath);
const sess = await ort.InferenceSession.create(modelPath, { executionProviders: ['cpu'] });
console.log('inputs:', sess.inputNames, 'outputs:', sess.outputNames);

let allPass = true;
for (const name of FIXTURE_NAMES) {
  const fx = loadFixture(fixturePath(name));
  const { batch: B, kmax: K } = fx;
  const batch: CollatedBatch = {
    inputIds: fx.inputs.input_ids,
    attentionMask: fx.inputs.attention_mask,
    markerPos: fx.inputs.marker_pos,
    markerMask: fx.inputs.marker_mask,
    qtype: fx.inputs.qtype,
  };
  const d = toFeedData(batch);
  const out = await sess.run({
    input_ids: new ort.Tensor('int64', d.inputIds, [d.dims.B, d.dims.S]),
    attention_mask: new ort.Tensor('int64', d.attentionMask, [d.dims.B, d.dims.S]),
    marker_pos: new ort.Tensor('int64', d.markerPos, [d.dims.B, d.dims.K]),
    marker_mask: new ort.Tensor('bool', d.markerMask, [d.dims.B, d.dims.K]),
    qtype: new ort.Tensor('int64', d.qtype, [d.dims.B]),
  });
  const logitsTensor = out['logits'];
  const actTensor = out['act_logits'];
  if (!logitsTensor || !actTensor) throw new Error(`run-node: missing outputs in ${fx.name}`);
  const logits = Array.from(logitsTensor.data as ArrayLike<number>);
  const act = Array.from(actTensor.data as ArrayLike<number>);
  const jsLogits: number[][] = [];
  for (let i = 0; i < B; i++) jsLogits.push(logits.slice(i * K, (i + 1) * K));
  const jsAct: number[][] = [];
  for (let i = 0; i < B; i++) jsAct.push(act.slice(i * 2, (i + 1) * 2));

  const diffLogits = maxAbsDiff(jsLogits, fx.expected_torch_logits, fx.inputs.marker_mask);
  const diffAct = maxAbsDiff(jsAct, fx.expected_torch_act, null);

  // calibrated comparison (JS postprocessing vs Python expected answers)
  let pdrift = 0;
  let adrift = 0;
  let labelsOk = true;
  for (let r = 0; r < B; r++) {
    const qid = fx.qids[r] as string;
    const k = fx.k_per_q[r] as number;
    const qt = fx.qtypes[r] as number;
    const bucket = tempBucket(qt, k);
    const tScale = fx.temperature.temperature_by_options[bucket] ?? fx.temperature.temperature[qt];
    if (tScale === undefined) throw new Error(`run-node: no temperature for ${bucket}`);
    const row = jsLogits[r] as number[];
    const z = row.slice(0, k).map((v) => v / Math.max(1e-3, tScale));
    const p = softmax(z);
    const actP = softmax(jsAct[r] as number[]);
    const exp = fx.expected_answers[qid];
    if (exp === undefined) throw new Error(`run-node: no expected answers for ${qid}`);
    if (exp.probs) {
      const probs = exp.probs;
      const keys = Object.keys(probs);
      keys.forEach((kk, i) => {
        pdrift = Math.max(pdrift, Math.abs((p[i] as number) - (probs[kk] as number)));
      });
      if (exp.choice !== undefined) {
        let best = 0;
        for (let i = 1; i < p.length; i++) if ((p[i] as number) > (p[best] as number)) best = i;
        if (keys[best] !== exp.choice) labelsOk = false;
      }
    } else if (exp.noul !== undefined) {
      pdrift = Math.max(pdrift, Math.abs((p[1] as number) - exp.noul));
    }
    adrift = Math.max(adrift, Math.abs((actP[0] as number) - exp.act));
  }
  const pass = diffLogits < 1e-4 && pdrift < 1e-4 && adrift < 1e-4 && labelsOk;
  allPass &&= pass;
  console.log(
    `${fx.name}: logits max=${diffLogits.toExponential(2)} act max=${diffAct.toExponential(2)} pdrift=${pdrift.toExponential(2)} adrift=${adrift.toExponential(2)} labels=${labelsOk ? 'OK' : 'DIFF'} -> ${pass ? 'PASS' : 'CHECK'}`,
  );
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exitCode = allPass ? 0 : 1;
