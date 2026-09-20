// Validate JS preprocessing vs Python fixtures (token IDs must match exactly).
// Needs @huggingface/transformers (devDependency). Manual harness — not in CI.
// Runs directly on Node >=22 via type-stripping (no build step).
import { AutoTokenizer } from '@huggingface/transformers';
import { readFileSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { toInternal, buildSequence, collateItems, QTYPES } from './laya-preprocess.ts';
import { STATE, questionsFor } from './test-helpers.ts';
import type { BuiltItem, Tokenizer } from './laya-types.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

console.log('loading tokenizer...');
// from_pretrained returns a promise of a tokenizer; narrow loosely (types vary by version).
const hf = (await AutoTokenizer.from_pretrained(join(root, 'js', 'tokenizer'))) as unknown as {
  mask_token: string;
  mask_token_id: number;
  sep_token_id: number;
  pad_token_id: number;
  encode(text: string, opts: { add_special_tokens: boolean }): { data?: ArrayLike<number> } | number[];
};
console.log('mask:', JSON.stringify(hf.mask_token), hf.mask_token_id, 'sep:', hf.sep_token_id, 'pad:', hf.pad_token_id);
const enc = (t: string): number[] => {
  const r = hf.encode(t, { add_special_tokens: false });
  return Array.isArray(r) ? [...r] : Array.from(r.data ?? []);
};
const tok: Tokenizer = {
  encode: enc,
  maskToken: hf.mask_token,
  maskId: hf.mask_token_id,
  clsId: enc('[CLS]')[0] as number,
  sepId: hf.sep_token_id,
  padId: hf.pad_token_id,
};
// quick probe
console.log('probe "choice question: hi" ->', tok.encode('choice question: hi').slice(0, 10));

interface TokFixture {
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
  const fx = JSON.parse(readFileSync(join(root, 'js', 'fixtures', f), 'utf8')) as TokFixture;
  const questions = questionsFor(fx.name);
  const ids = Object.keys(questions);
  const items: BuiltItem[] = ids.map((qid) => {
    const qdef = questions[qid];
    if (qdef === undefined) throw new Error(`check-tokenizer: missing question ${qid}`);
    const q = toInternal(qdef);
    const { ids: seq, markers } = buildSequence(tok, STATE, q, 512, 192);
    return { ids: seq, markers, qtype: QTYPES[q.t] };
  });
  const batch = collateItems([items], tok.padId);
  const expIds = fx.inputs.input_ids;
  let mism = 0;
  let firstAt: [number, number, number, number] | null = null;
  for (let i = 0; i < batch.inputIds.length; i++) {
    const gotRow = batch.inputIds[i] as number[];
    const expRow = expIds[i] as number[];
    for (let j = 0; j < gotRow.length; j++) {
      if (gotRow[j] !== expRow[j]) {
        mism++;
        firstAt ??= [i, j, gotRow[j] as number, expRow[j] as number];
      }
    }
  }
  const attOk = JSON.stringify(batch.attentionMask) === JSON.stringify(fx.inputs.attention_mask);
  const mpOk = JSON.stringify(batch.markerPos) === JSON.stringify(fx.inputs.marker_pos);
  const mmOk = JSON.stringify(batch.markerMask) === JSON.stringify(fx.inputs.marker_mask);
  const pass = mism === 0 && attOk && mpOk && mmOk;
  allPass &&= pass;
  console.log(
    `${fx.name}: ids ${mism === 0 ? 'OK' : `DIFF ${mism} first@${JSON.stringify(firstAt)}`} att=${attOk ? 'OK' : 'DIFF'} mp=${mpOk ? 'OK' : 'DIFF'} mm=${mmOk ? 'OK' : 'DIFF'} -> ${pass ? 'PASS' : 'CHECK'}`,
  );
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exitCode = allPass ? 0 : 1;
