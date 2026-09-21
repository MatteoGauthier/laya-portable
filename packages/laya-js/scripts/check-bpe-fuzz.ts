// Fuzz pure-JS BPE vs Python (206 cases incl. unicode, spaces, added tokens).
// Runs directly on Node >=22 via type-stripping (no build step).
// Defaults gate the english tokenizer; --tokenizer/--fuzz gate any checkpoint:
//   node scripts/check-bpe-fuzz.ts --tokenizer=../../models/laya-multilingual.tokenizer.json \
//     --fuzz=bpe-fuzz.multilingual.json
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadBpeTokenizer } from '../src/laya-bpe.ts';
import { DEFAULT_TOKENIZER, VECTORS_DIR } from '../src/laya-paths.ts';
import { vectorsFile } from '../tests/test-helpers.ts';
import type { TokenizerJson } from '../src/laya-types.ts';

const tokArg = process.argv.find((a) => a.startsWith('--tokenizer='));
const fuzzArg = process.argv.find((a) => a.startsWith('--fuzz='));
const tokPath = tokArg ? tokArg.slice('--tokenizer='.length) : DEFAULT_TOKENIZER;
const fuzzPath = fuzzArg ? join(VECTORS_DIR, fuzzArg.slice('--fuzz='.length)) : vectorsFile('bpe-fuzz.json');

const tj = JSON.parse(readFileSync(tokPath, 'utf8')) as TokenizerJson;
const tok = loadBpeTokenizer(tj);
const cases = JSON.parse(readFileSync(fuzzPath, 'utf8')) as { text: string; ids: number[] }[];
let pass = 0;
const fails: [number, string, string][] = [];
for (let i = 0; i < cases.length; i++) {
  const c = cases[i] as { text: string; ids: number[] };
  const { text, ids: exp } = c;
  let got: number[];
  try {
    got = tok.encode(text);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    fails.push([i, JSON.stringify(text.slice(0, 60)), `THROW ${msg.slice(0, 80)}`]);
    continue;
  }
  if (JSON.stringify(got) === JSON.stringify(exp)) pass++;
  else {
    fails.push([
      i,
      JSON.stringify(text.slice(0, 60)),
      `exp[${exp.slice(0, 8)}] len ${exp.length} vs got[${got.slice(0, 8)}] len ${got.length}`,
    ]);
  }
}
console.log(`${pass}/${cases.length} PASS`);
for (const [i, t, d] of fails.slice(0, 10)) console.log(`FAIL #${i} ${t} :: ${d}`);
process.exitCode = fails.length ? 1 : 0;
