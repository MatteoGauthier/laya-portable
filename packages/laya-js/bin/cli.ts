#!/usr/bin/env node
// laya CLI: calibrated decisions from JSON state + questions.
import { readFileSync } from 'node:fs';
import { basename, join } from 'node:path';
import { parseArgs } from 'node:util';
import { LayaClient, DEFAULT_MODEL } from '../src/laya.ts';
import { CHECKPOINT_CONFIGS, CHECKPOINT_MODELS, CHECKPOINT_TOKENIZERS, MODELS_DIR } from '../src/laya-paths.ts';
import { Router } from '../src/laya-router.ts';
import type { Questions } from '../src/laya-types.ts';

const { values } = parseArgs({
  options: {
    model: { type: 'string' },
    checkpoint: { type: 'string' },
    route: { type: 'boolean', default: false },
    lang: { type: 'string' },
    task: { type: 'string' },
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
  node bin/cli.ts [--model path.onnx | --fp16] [--checkpoint english|multilingual|typed-decisions]
               [--route] [--lang hi] [--task typed_decisions]
               [--state JSON] [--questions JSON]
               [--state-file f] [--questions-file f] [--json]
Defaults to ${DEFAULT_MODEL};
--route auto-selects --checkpoint via the JS Router.`);
  process.exit(0);
}

const DEFAULT_STATE: Record<string, string> = {
  subject: 'Duplicate charge on invoice 4411',
  body: 'We were billed twice for March. Please refund the duplicate.',
};
const DEFAULT_QUESTIONS: Questions = {
  department: {
    type: 'choice',
    instructions: 'Which team should handle this?',
    criteria: { billing: 'invoices, payments, refunds', technical: 'bugs and outages', sales: 'pricing' },
  },
  urgency: { type: 'score', instructions: 'How urgent is this?', criteria: ['not urgent', 'soon', 'blocking'] },
  churn_risk: { type: 'noul', instructions: 'Does the user threaten to cancel?' },
};

function load(inline: string | undefined, file: string | undefined, fallback: unknown, label: string): unknown {
  try {
    if (inline) return JSON.parse(inline) as unknown;
    if (file) return JSON.parse(readFileSync(file, 'utf8')) as unknown;
    return fallback;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`laya: invalid ${label}: ${msg}`);
    process.exitCode = 1;
    throw err;
  }
}

function fail(op: string, err: unknown): never {
  const msg = err instanceof Error ? err.message : String(err);
  const cause =
    err instanceof Error && err.cause ? ` (cause: ${String((err.cause as Error)?.message ?? err.cause)})` : '';
  console.error(`laya: ${op} failed: ${msg}${cause}`);
  process.exitCode = 1;
  throw err;
}

const checkpoint = values.checkpoint as 'english' | 'multilingual' | 'typed-decisions' | undefined;
const model =
  values.model ??
  (checkpoint
    ? CHECKPOINT_MODELS[checkpoint]
    : join(MODELS_DIR, values.fp16 ? 'laya-split-fp16.onnx' : 'laya-split-single.onnx'));
const tokenizer = checkpoint ? CHECKPOINT_TOKENIZERS[checkpoint] : undefined;
const config = checkpoint ? CHECKPOINT_CONFIGS[checkpoint] : undefined;

const state = load(values.state, values['state-file'], DEFAULT_STATE, 'state');
const questions = load(values.questions, values['questions-file'], DEFAULT_QUESTIONS, 'questions') as Questions;

if (values.route) {
  const router = new Router();
  const decision = router.routeDecision(state, questions, { lang: values.lang, task: values.task });
  console.error(`route: ${decision.model} (${decision.reason})`);
  await router.unload();
}

let laya: LayaClient;
try {
  const t0 = performance.now();
  laya = await LayaClient.open({ model, ...(tokenizer ? { tokenizer } : {}), ...(config ? { config } : {}) });
  console.error(`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s: ${basename(model)}`);
} catch (err) {
  fail('open', err);
}

try {
  const result = await laya.predict(state, questions);
  console.log(JSON.stringify(result, null, values.json ? 0 : 1));
} catch (err) {
  fail('predict', err);
} finally {
  await laya.close?.();
}
