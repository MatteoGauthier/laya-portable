#!/usr/bin/env node
// @ts-nocheck — CLI harness (covered by tests/ + runtime guards)
// laya CLI: run calibrated decisions from JSON state + questions.
//
//   node cli.mjs [--fp16] [--state '{"subject":".."}'] [--questions '{...}']
//   node cli.mjs --state-file s.json --questions-file q.json [--model path.onnx]
//   node cli.mjs --help | --json
import { readFileSync } from 'node:fs';
import { basename, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { LayaClient, DEFAULT_MODEL } from './laya.mjs';

const here = dirname(fileURLToPath(import.meta.url));

const { values } = parseArgs({
  options: {
    model: { type: 'string' },
    fp16: { type: 'boolean', default: false },
    state: { type: 'string' },
    'state-file': { type: 'string' },
    questions: { type: 'string' },
    'questions-file': { type: 'string' },
    json: { type: 'boolean', default: false },
    help: { type: 'boolean', default: false },
  },
  allowPositionals: false,
});

if (values.help) {
  console.log(`laya — calibrated ONNX decisions
Usage:
  node cli.mjs [--model path.onnx | --fp16] [--state JSON] [--questions JSON]
               [--state-file f] [--questions-file f] [--json]
Defaults to ${DEFAULT_MODEL}`);
  process.exit(0);
}

const DEFAULT_STATE = {
  subject: 'Duplicate charge on invoice 4411',
  body: 'We were billed twice for March. Please refund the duplicate.',
};
const DEFAULT_QUESTIONS = {
  department: {
    type: 'choice',
    instructions: 'Which team should handle this?',
    criteria: { billing: 'invoices, payments, refunds', technical: 'bugs and outages', sales: 'pricing' },
  },
  urgency: { type: 'score', instructions: 'How urgent is this?', criteria: ['not urgent', 'soon', 'blocking'] },
  churn_risk: { type: 'noul', instructions: 'Does the user threaten to cancel?' },
};

function load(inline, file, fallback, label) {
  try {
    if (inline) return JSON.parse(inline);
    if (file) return JSON.parse(readFileSync(file, 'utf8'));
    return fallback;
  } catch (err) {
    console.error(`laya: invalid ${label}: ${err.message}`);
    process.exitCode = 1;
    throw err;
  }
}

const model =
  values.model ?? join(here, '..', 'models', values.fp16 ? 'laya-split-fp16.onnx' : 'laya-split-single.onnx');

let laya;
try {
  const t0 = performance.now();
  laya = await LayaClient.open({ model });
  console.error(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s: ${basename(model)}`);
} catch (err) {
  console.error(`laya: open failed: ${err.message}${err.cause ? ` (cause: ${err.cause.message})` : ''}`);
  process.exitCode = 1;
  throw err;
}

try {
  const result = await laya.predict(
    load(values.state, values['state-file'], DEFAULT_STATE, 'state'),
    load(values.questions, values['questions-file'], DEFAULT_QUESTIONS, 'questions'),
  );
  console.log(JSON.stringify(result, null, values.json ? 0 : 1));
} catch (err) {
  console.error(`laya: predict failed: ${err.message}${err.cause ? ` (cause: ${err.cause.message})` : ''}`);
  process.exitCode = 1;
} finally {
  await laya.close?.();
}
