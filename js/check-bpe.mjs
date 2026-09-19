// Validate pure-JS BPE vs Python fixtures (must match exactly, no transformers.js).
import { readFileSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadBpeTokenizer } from './laya-bpe.mjs';
import { toInternal, buildSequence, collateItems, QTYPES } from './laya-preprocess.mjs';
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const STATE = { subject: 'Duplicate charge on invoice 4411', body: 'We were billed twice for March. Please refund the duplicate.' };
function questionsFor(name) {
  const c3 = { type: 'choice', instructions: 'Which team should handle this?', criteria: { billing: 'invoices, payments, refunds', technical: 'bugs and outages', sales: 'pricing' } };
  const s3 = { type: 'score', instructions: 'How urgent is this?', criteria: ['not urgent', 'soon', 'blocking'] };
  const n2 = { type: 'noul', instructions: 'Does the user threaten to cancel?' };
  if (name === 'orig-3q') return { department: c3, urgency: s3, churn_risk: n2 };
  if (name === 'choice-3') return { department: c3 };
  if (name === 'choice-2') return { dept2: { type: 'choice', instructions: 'Billing or tech?', criteria: { billing: 'invoices', technical: 'bugs' } } };
  if (name === 'choice-6') return { dept6: { type: 'choice', instructions: 'Route it', criteria: Object.fromEntries(Array.from({ length: 6 }, (_, i) => [`team${i}`, `area ${i}`])) } };
  if (name === 'mixed-batch') return {
    a_choice5: { type: 'choice', instructions: 'Pick', criteria: Object.fromEntries(Array.from({ length: 5 }, (_, i) => [`o${i}`, `desc ${i}`])) },
    b_noul: n2, c_score2: { type: 'score', instructions: 'Rate', criteria: ['low', 'high'] } };
  throw new Error(name);
}
const tj = JSON.parse(readFileSync(join(root, 'js', 'tokenizer', 'tokenizer.json'), 'utf8'));
const tok = loadBpeTokenizer(tj);
console.log('bpe loaded. probe:', tok.encode('choice question: hi').slice(0, 6), 'mask/cls/sep/pad:', tok.maskId, tok.clsId, tok.sepId, tok.padId);
let allPass = true;
for (const f of readdirSync(join(root, 'js', 'fixtures')).filter(x => x.endsWith('.json')).sort()) {
  const fx = JSON.parse(readFileSync(join(root, 'js', 'fixtures', f), 'utf8'));
  const questions = questionsFor(fx.name);
  const ids = Object.keys(questions);
  const items = ids.map(qid => {
    const q = toInternal(questions[qid]);
    const { ids: seq, markers } = buildSequence(tok, STATE, q, 512, 192);
    return { ids: seq, markers, qtype: QTYPES[q.t] };
  });
  const batch = collateItems([items], tok.padId);
  let mism = 0, firstAt = null;
  for (let i = 0; i < batch.inputIds.length; i++) for (let j = 0; j < batch.inputIds[0].length; j++) {
    if (batch.inputIds[i][j] !== fx.inputs.input_ids[i][j]) { mism++; firstAt ??= [i, j, batch.inputIds[i][j], fx.inputs.input_ids[i][j]]; }
  }
  const ok = mism === 0 && JSON.stringify(batch.attentionMask) === JSON.stringify(fx.inputs.attention_mask)
    && JSON.stringify(batch.markerPos) === JSON.stringify(fx.inputs.marker_pos)
    && JSON.stringify(batch.markerMask) === JSON.stringify(fx.inputs.marker_mask);
  allPass &&= ok;
  console.log(`${fx.name}: ${mism === 0 ? 'ids OK' : `DIFF ${mism} first@${JSON.stringify(firstAt)}`} -> ${ok ? 'PASS' : 'CHECK'}`);
}
console.log(allPass ? 'OVERALL PASS' : 'OVERALL CHECK');
process.exit(allPass ? 0 : 1);
