// Validate split: ORT split (logits+pooled) + JS action head vs torch full.
// Threshold is absolute (like run-node): act logits saturate, so gate on
// calibrated prob drift downstream in tests, raw here at 5e-1 absolute.
// Runs directly on Node >=22 via type-stripping (no build step).
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';
import { actionLogits } from '../src/laya-action.ts';
import { toFeedData, splitOutputs } from '../src/laya-feed.ts';
import { DEFAULT_MODEL } from '../src/laya.ts';
import { DEFAULT_ACT_HEAD_JSON } from '../src/laya-paths.ts';
import { FIXTURE_NAMES, fixturePath } from '../tests/test-helpers.ts';
import type { ActHeadWeights, CollatedBatch } from '../src/laya-types.ts';

const w = JSON.parse(readFileSync(DEFAULT_ACT_HEAD_JSON, 'utf8')) as ActHeadWeights;
const modelArg = process.argv.find((a) => a.startsWith('--model='));
const sess = await ort.InferenceSession.create(modelArg ? modelArg.slice('--model='.length) : DEFAULT_MODEL, {
  executionProviders: ['cpu'],
});

interface SplitFixture {
  name: string;
  batch: number;
  kmax: number;
  inputs: {
    input_ids: number[][];
    attention_mask: number[][];
    marker_pos: number[][];
    marker_mask: boolean[][];
    qtype: number[];
  };
  expected_torch_act: number[][];
}

let allPass = true;
for (const name of FIXTURE_NAMES) {
  const fx = JSON.parse(readFileSync(fixturePath(name), 'utf8')) as SplitFixture;
  const { batch: B, kmax: K } = fx;
  const batch: CollatedBatch = {
    inputIds: fx.inputs.input_ids,
    attentionMask: fx.inputs.attention_mask,
    markerPos: fx.inputs.marker_pos,
    markerMask: fx.inputs.marker_mask,
    qtype: fx.inputs.qtype,
  };
  const d = toFeedData(batch);
  const r = await sess.run({
    input_ids: new ort.Tensor('int64', d.inputIds, [d.dims.B, d.dims.S]),
    attention_mask: new ort.Tensor('int64', d.attentionMask, [d.dims.B, d.dims.S]),
    marker_pos: new ort.Tensor('int64', d.markerPos, [d.dims.B, d.dims.K]),
    marker_mask: new ort.Tensor('bool', d.markerMask, [d.dims.B, d.dims.K]),
    qtype: new ort.Tensor('int64', d.qtype, [d.dims.B]),
  });
  const logitsTensor = r['logits'];
  const pooledTensor = r['pooled'];
  if (!logitsTensor || !pooledTensor) throw new Error(`check-split: missing outputs in ${fx.name}`);
  const { logits: jsLo, pooled: jsPo } = splitOutputs(
    Array.from(logitsTensor.data as ArrayLike<number>),
    Array.from(pooledTensor.data as ArrayLike<number>),
    B,
    K,
  );
  const jsAct = actionLogits(jsLo, fx.inputs.marker_mask, jsPo, w);
  let md = 0;
  for (let i = 0; i < B; i++) {
    for (let j = 0; j < 2; j++) {
      md = Math.max(md, Math.abs((jsAct[i]?.[j] as number) - (fx.expected_torch_act[i]?.[j] as number)));
    }
  }
  // Act logits are O(1000s); absolute 1e-4 is too strict — gate raw at 5e-1
  // and rely on calibrated prob drift (tests/) for the tight check.
  const pass = md < 5e-1;
  allPass &&= pass;
  console.log(`${fx.name}: act max=${md.toExponential(2)} -> ${pass ? 'PASS' : 'CHECK'}`);
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exitCode = allPass ? 0 : 1;
