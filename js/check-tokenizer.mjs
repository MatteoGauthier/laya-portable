// @ts-nocheck — legacy harness (covered by tests/ + runtime guards)
// Validate JS preprocessing vs Python fixtures (token IDs must match exactly).
import { AutoTokenizer } from '@huggingface/transformers';
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toInternal, buildSequence, collateItems, QTYPES } from './laya-preprocess.mjs';
import { STATE, questionsFor } from './test-helpers.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

console.log('loading tokenizer...');
const hf = await AutoTokenizer.from_pretrained(join(root, 'js', 'tokenizer'));
console.log(
  'mask:',
  JSON.stringify(hf.mask_token),
  hf.mask_token_id,
  'cls:',
  hf.cls_token_id,
  'sep:',
  hf.sep_token_id,
  'pad:',
  hf.pad_token_id,
);
const enc = (t) => {
  const r = hf.encode(t, { add_special_tokens: false });
  return Array.from(r.data ?? r);
};
const tok = {
  encode: enc,
  maskToken: hf.mask_token,
  maskId: hf.mask_token_id,
  clsId: hf.cls_token_id ?? enc('[CLS]')[0],
  sepId: hf.sep_token_id,
  padId: hf.pad_token_id,
};
// quick probe
console.log('probe "choice question: hi" ->', tok.encode('choice question: hi').slice(0, 10));

let allPass = true;
for (const f of readdirSync(join(root, 'js', 'fixtures'))
  .filter((x) => x.endsWith('.json'))
  .sort()) {
  const fx = JSON.parse(readFileSync(join(root, 'js', 'fixtures', f), 'utf8'));
  const questions = questionsFor(fx.name);
  const ids = Object.keys(questions);
  const items = ids.map((qid) => {
    const q = toInternal(questions[qid]);
    const { ids: seq, markers } = buildSequence(tok, STATE, q, 512, 192);
    return { ids: seq, markers, qtype: QTYPES[q.t] };
  });
  const batch = collateItems([items], tok.padId);
  const expIds = fx.inputs.input_ids,
    expAtt = fx.inputs.attention_mask;
  const expMp = fx.inputs.marker_pos,
    expMm = fx.inputs.marker_mask;
  let mism = 0,
    firstAt = null;
  for (let i = 0; i < batch.inputIds.length; i++)
    for (let j = 0; j < batch.inputIds[0].length; j++) {
      if (batch.inputIds[i][j] !== expIds[i][j]) {
        mism++;
        firstAt ??= [i, j, batch.inputIds[i][j], expIds[i][j]];
      }
    }
  const attOk = JSON.stringify(batch.attentionMask) === JSON.stringify(expAtt);
  const mpOk = JSON.stringify(batch.markerPos) === JSON.stringify(expMp);
  const mmOk = JSON.stringify(batch.markerMask) === JSON.stringify(expMm);
  const pass = mism === 0 && attOk && mpOk && mmOk;
  allPass &&= pass;
  console.log(
    `${fx.name}: ids ${mism === 0 ? 'OK' : `DIFF ${mism} first@${JSON.stringify(firstAt)}`} att=${attOk ? 'OK' : 'DIFF'} mp=${mpOk ? 'OK' : 'DIFF'} mm=${mmOk ? 'OK' : 'DIFF'} -> ${pass ? 'PASS' : 'CHECK'}`,
  );
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exitCode = allPass ? 0 : 1;
