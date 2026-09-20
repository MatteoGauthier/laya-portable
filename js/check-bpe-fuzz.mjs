// @ts-nocheck — legacy harness (covered by tests/ + runtime guards)
// Fuzz pure-JS BPE vs Python (206 cases incl. unicode, spaces, added tokens).
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBpeTokenizer } from './laya-bpe.mjs';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const tok = loadBpeTokenizer(JSON.parse(readFileSync(join(root, 'js', 'tokenizer', 'tokenizer.json'), 'utf8')));
const cases = JSON.parse(readFileSync(join(root, 'js', 'bpe-fuzz.json'), 'utf8'));
let pass = 0;
const fails = [];
for (let i = 0; i < cases.length; i++) {
  const { text, ids: exp } = cases[i];
  let got;
  try {
    got = tok.encode(text);
  } catch (e) {
    fails.push([i, JSON.stringify(text.slice(0, 60)), 'THROW ' + e.message.slice(0, 80)]);
    continue;
  }
  if (JSON.stringify(got) === JSON.stringify(exp)) pass++;
  else
    fails.push([
      i,
      JSON.stringify(text.slice(0, 60)),
      `exp[${exp.slice(0, 8)}] len ${exp.length} vs got[${got.slice(0, 8)}] len ${got.length}`,
    ]);
}
console.log(`${pass}/${cases.length} PASS`);
for (const [i, t, d] of fails.slice(0, 10)) console.log(`FAIL #${i} ${t} :: ${d}`);
process.exitCode = fails.length ? 1 : 0;
