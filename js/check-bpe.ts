// Validate pure-JS BPE vs Python fixtures (must match exactly, no transformers.js).
// Runs directly on Node >=22 via type-stripping (no build step).
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBpeTokenizer } from './laya-bpe.ts';
import { toInternal, buildSequence, collateItems, QTYPES } from './laya-preprocess.ts';
import { STATE, questionsFor } from './test-helpers.ts';
import type { BuiltItem, TokenizerJson } from './laya-types.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const tj = JSON.parse(readFileSync(join(root, 'js', 'tokenizer', 'tokenizer.json'), 'utf8')) as TokenizerJson;
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

interface BpeFixture {
  name: string;
  inputs: {
    input_ids: number[][];
    attention_mask: number[][];
    marker_pos: number[][];
    marker_mask: boolean[][];
  };
}

let allPass = true;
for (const f of readdirSync(join(root, 'js', 'fixtures'))
  .filter((x) => x.endsWith('.json'))
  .sort()) {
  const fx = JSON.parse(readFileSync(join(root, 'js', 'fixtures', f), 'utf8')) as BpeFixture;
  const questions = questionsFor(fx.name);
  const ids = Object.keys(questions);
  const items: BuiltItem[] = ids.map((qid) => {
    const qdef = questions[qid];
    if (qdef === undefined) throw new Error(`check-bpe: missing question ${qid}`);
    const q = toInternal(qdef);
    const { ids: seq, markers } = buildSequence(tok, STATE, q, 512, 192);
    return { ids: seq, markers, qtype: QTYPES[q.t] };
  });
  const batch = collateItems([items], tok.padId);
  let mism = 0;
  let firstAt: [number, number, number, number] | null = null;
  for (let i = 0; i < batch.inputIds.length; i++) {
    const gotRow = batch.inputIds[i] as number[];
    const expRow = fx.inputs.input_ids[i] as number[];
    for (let j = 0; j < gotRow.length; j++) {
      if (gotRow[j] !== expRow[j]) {
        mism++;
        firstAt ??= [i, j, gotRow[j] as number, expRow[j] as number];
      }
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
