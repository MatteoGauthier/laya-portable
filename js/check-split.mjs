// Validate split: ORT split (logits+pooled) + JS action head vs torch full.
import * as ort from 'onnxruntime-node';
import { readFileSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { actionLogits } from './laya-action.mjs';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const w = JSON.parse(readFileSync(join(root, 'js', 'act_head.json'), 'utf8'));
const sess = await ort.InferenceSession.create(join(root, 'models', 'laya-split-single.onnx'), { executionProviders: ['cpu'] });
let allPass = true;
for (const f of readdirSync(join(root, 'js', 'fixtures')).filter(x => x.endsWith('.json')).sort()) {
  const fx = JSON.parse(readFileSync(join(root, 'js', 'fixtures', f), 'utf8'));
  const { batch: B, seq_len: S, kmax: K } = fx;
  const feeds = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(fx.inputs.input_ids.flat(Infinity).map(v => BigInt(v))), [B, S]),
    attention_mask: new ort.Tensor('int64', BigInt64Array.from(fx.inputs.attention_mask.flat(Infinity).map(v => BigInt(v))), [B, S]),
    marker_pos: new ort.Tensor('int64', BigInt64Array.from(fx.inputs.marker_pos.flat(Infinity).map(v => BigInt(v))), [B, K]),
    marker_mask: new ort.Tensor('bool', Uint8Array.from(fx.inputs.marker_mask.flat(Infinity).map(v => v ? 1 : 0)), [B, K]),
    qtype: new ort.Tensor('int64', BigInt64Array.from(fx.inputs.qtype.map(v => BigInt(v))), [B]),
  };
  const r = await sess.run(feeds);
  const lo = Array.from(r.logits.data), po = Array.from(r.pooled.data);
  const jsLo = [], jsPo = [];
  for (let i = 0; i < B; i++) { jsLo.push(lo.slice(i * K, (i + 1) * K)); jsPo.push(po.slice(i * 1024, (i + 1) * 1024)); }
  const jsAct = actionLogits(jsLo, fx.inputs.marker_mask, jsPo, w);
  const expAct = fx.expected_onnx_logits ? null : null; // torch act from npz? fixtures JSON has torch act? No, only logits. Load npz? Use parity-report? Simpler: compare to choice-2 known? Actually js fixtures lack torch_act. Compare JS act to Python ORT full act via... we need expected act. js/fixtures/*.json has expected_torch_act? Check: emit_js_fixtures includes expected_torch_act? Yes: "expected_torch_act": d["torch_act"].tolist(). Good.
  let md = 0, mr = 0;
  for (let i = 0; i < B; i++) for (let j = 0; j < 2; j++) {
    const d = Math.abs(jsAct[i][j] - fx.expected_torch_act[i][j]);
    md = Math.max(md, d);
    mr = Math.max(mr, d / Math.max(1, Math.abs(fx.expected_torch_act[i][j])));
  }
  const pass = mr < 1e-4;
  allPass &&= pass;
  console.log(`${fx.name}: act max=${md.toExponential(2)} rel=${mr.toExponential(2)} -> ${pass ? 'PASS' : 'CHECK'}`);
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exit(allPass ? 0 : 1);
