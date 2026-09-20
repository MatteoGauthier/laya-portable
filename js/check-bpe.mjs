// @ts-nocheck — legacy harness (covered by tests/ + runtime guards)
// Validate pure-JS BPE vs Python fixtures (must match exactly, no transformers.js).
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBpeTokenizer } from './laya-bpe.mjs';
import { toInternal, buildSequence, collateItems, QTYPES } from './laya-preprocess.mjs';
import { STATE, questionsFor } from './test-helpers.mjs';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const tj = JSON.parse(readFileSync(join(root, 'js', 'tokenizer', 'tokenizer.json'), 'utf8'));
const tok = loadBpeTokenizer(tj);
console.log(
  'bpe loaded. probe:',
  tok.encode('choice question: hi').slice(0, 6),
  'mask/cls/sep/pad:',
  tok.maskId,
  tok.clsId,
  tok.sepId,
  tok.padId,
);
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
  let mism = 0,
    firstAt = null;
  for (let i = 0; i < batch.inputIds.length; i++)
    for (let j = 0; j < batch.inputIds[0].length; j++) {
      if (batch.inputIds[i][j] !== fx.inputs.input_ids[i][j]) {
        mism++;
        firstAt ??= [i, j, batch.inputIds[i][j], fx.inputs.input_ids[i][j]];
      }
    }
  const ok =
    mism === 0 &&
    JSON.stringify(batch.attentionMask) === JSON.stringify(fx.inputs.attention_mask) &&
    JSON.stringify(batch.markerPos) === JSON.stringify(fx.inputs.marker_pos) &&
    JSON.stringify(batch.markerMask) === JSON.stringify(fx.inputs.marker_mask);
  allPass &&= ok;
  console.log(
    `${fx.name}: ${mism === 0 ? 'ids OK' : `DIFF ${mism} first@${JSON.stringify(firstAt)}`} -> ${ok ? 'PASS' : 'CHECK'}`,
  );
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exitCode = allPass ? 0 : 1;
