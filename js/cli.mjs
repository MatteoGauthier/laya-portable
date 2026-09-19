#!/usr/bin/env node
// laya CLI: run calibrated decisions from JSON state + questions.
// Usage:
//   node cli.mjs [--fp16] [--state '{"subject":".."}'] [--questions '{...}']
//   node cli.mjs --state-file s.json --questions-file q.json [--model path.onnx]
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { LayaClient } from './laya.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const get = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : null; };
const DEFAULT_STATE = { subject: 'Duplicate charge on invoice 4411', body: 'We were billed twice for March. Please refund the duplicate.' };
const DEFAULT_QUESTIONS = {
  department: { type: 'choice', instructions: 'Which team should handle this?', criteria: { billing: 'invoices, payments, refunds', technical: 'bugs and outages', sales: 'pricing' } },
  urgency: { type: 'score', instructions: 'How urgent is this?', criteria: ['not urgent', 'soon', 'blocking'] },
  churn_risk: { type: 'noul', instructions: 'Does the user threaten to cancel?' },
};
const load = (inline, file, fallback) => {
  if (inline) return JSON.parse(inline);
  if (file) return JSON.parse(readFileSync(file, 'utf8'));
  return fallback;
};
const model = get('--model')
  ?? join(here, '..', 'models', args.includes('--fp16') ? 'laya-split-fp16.onnx' : 'laya-split-single.onnx');
const t0 = performance.now();
const laya = await LayaClient.open({ model });
console.error(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s: ${model.split('/').pop()}`);
const result = await laya.predict(
  load(get('--state'), get('--state-file'), DEFAULT_STATE),
  load(get('--questions'), get('--questions-file'), DEFAULT_QUESTIONS));
console.log(JSON.stringify(result, null, 1));
