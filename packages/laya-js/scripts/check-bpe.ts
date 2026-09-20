// Validate pure-JS BPE vs Python fixtures (must match exactly, no transformers.js).
// Runs directly on Node >=22 via type-stripping (no build step).
import { readFileSync } from 'node:fs';
import { loadBpeTokenizer } from '../src/laya-bpe.ts';
import { toInternal, buildSequence, collateItems, QTYPES } from '../src/laya-preprocess.ts';
import { STATE, questionsFor, FIXTURE_NAMES, fixturePath } from '../tests/test-helpers.ts';
import { DEFAULT_TOKENIZER } from '../src/laya-paths.ts';
import type { BuiltItem, TokenizerJson } from '../src/laya-types.ts';

const tj = JSON.parse(readFileSync(DEFAULT_TOKENIZER, 'utf8')) as TokenizerJson;
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
for (const name of FIXTURE_NAMES) {
  const fx = JSON.parse(readFileSync(fixturePath(name), 'utf8')) as BpeFixture;
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
