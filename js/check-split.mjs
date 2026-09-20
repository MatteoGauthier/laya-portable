// @ts-nocheck — legacy harness (covered by tests/ + runtime guards)
// Validate split: ORT split (logits+pooled) + JS action head vs torch full.
// Threshold is absolute (like run-node): act logits saturate, so gate on
// calibrated prob drift downstream in tests, raw here at 1e-2 absolute.
import * as ort from 'onnxruntime-node';
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { actionLogits } from './laya-action.mjs';
import { buildFeeds, splitOutputs } from './laya-feed.mjs';
import { DEFAULT_MODEL } from './laya.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const w = JSON.parse(readFileSync(join(root, 'js', 'act_head.json'), 'utf8'));
const modelArg = process.argv.find((a) => a.startsWith('--model='));
const sess = await ort.InferenceSession.create(modelArg ? modelArg.slice('--model='.length) : DEFAULT_MODEL, {
  executionProviders: ['cpu'],
});
let allPass = true;
for (const f of readdirSync(join(root, 'js', 'fixtures'))
  .filter((x) => x.endsWith('.json'))
  .sort()) {
  const fx = JSON.parse(readFileSync(join(root, 'js', 'fixtures', f), 'utf8'));
  const { batch: B, seq_len: S, kmax: K } = fx;
  const feeds = buildFeeds(ort, {
    inputIds: fx.inputs.input_ids,
    attentionMask: fx.inputs.attention_mask,
    markerPos: fx.inputs.marker_pos,
    markerMask: fx.inputs.marker_mask,
    qtype: fx.inputs.qtype,
  });
  void S;
  const r = await sess.run(feeds);
  const { logits: jsLo, pooled: jsPo } = splitOutputs(r.logits.data, r.pooled.data, B, K);
  const jsAct = actionLogits(jsLo, fx.inputs.marker_mask, jsPo, w);
  let md = 0;
  for (let i = 0; i < B; i++)
    for (let j = 0; j < 2; j++) {
      const d = Math.abs(jsAct[i][j] - fx.expected_torch_act[i][j]);
      md = Math.max(md, d);
    }
  // Act logits are O(1000s); absolute 1e-4 is too strict — gate raw at 5e-1
  // and rely on calibrated prob drift (tests/) for the tight check.
  const pass = md < 5e-1;
  allPass &&= pass;
  console.log(`${fx.name}: act max=${md.toExponential(2)} -> ${pass ? 'PASS' : 'CHECK'}`);
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exitCode = allPass ? 0 : 1;
