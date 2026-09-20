// Minimal Node.js SDK facade: tokenizer + ONNX split model + calibration.
// import { LayaClient } from './laya.mjs';
// const laya = await LayaClient.open(); console.log(await laya.predict(state, questions));
import * as ort from 'onnxruntime-node';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBpeTokenizer } from './laya-bpe.mjs';
import { toInternal, buildSequence, collateItems, QTYPES } from './laya-preprocess.mjs';
import { predictFromLogits } from './laya-postprocess.mjs';
import { actionLogits } from './laya-action.mjs';
import { buildFeeds, splitOutputs } from './laya-feed.mjs';
import { LayaConfigError, LayaEncodeError } from './laya-errors.mjs';

const here = dirname(fileURLToPath(import.meta.url));

/** Canonical default model (single-file split FP32). CLI/worker/checks must match. */
export const DEFAULT_MODEL = join(here, '..', 'models', 'laya-split-single.onnx');
export const DEFAULT_TOKENIZER = join(here, 'tokenizer', 'tokenizer.json');
export const DEFAULT_CONFIG = join(here, 'tokenizer', 'rl_agent_config.json');
export const DEFAULT_ACT_HEAD = join(here, 'act_head.json');

function readJson(path, label) {
  try {
    return JSON.parse(readFileSync(path, 'utf8'));
  } catch (err) {
    throw new LayaConfigError(`cannot read ${label}: ${path}`, { cause: err });
  }
}

export class LayaClient {
  tok;
  sess;
  actW;
  temp;
  maxLen;
  headMaxLen;
  modelPath;
  /**
   * @param {{model?,tokenizer?,config?,actHead?,providers?}} opts
   */
  static async open(opts = {}) {
    const modelPath = opts.model ?? DEFAULT_MODEL;
    const [tokenizerJson, cfg] = await Promise.all([
      (async () => readJson(opts.tokenizer ?? DEFAULT_TOKENIZER, 'tokenizer'))(),
      (async () => readJson(opts.config ?? DEFAULT_CONFIG, 'config'))(),
    ]);
    const c = new LayaClient();
    try {
      c.tok = loadBpeTokenizer(tokenizerJson);
    } catch (err) {
      throw new LayaConfigError('invalid tokenizer.json', { cause: err });
    }
    try {
      c.sess = await ort.InferenceSession.create(modelPath, {
        executionProviders: opts.providers ?? ['cpu'],
      });
    } catch (err) {
      throw new LayaConfigError(`cannot load model: ${modelPath}`, { cause: err });
    }
    // act_head: prefer binary fast path (1.1MB f32) over JSON (5.2MB text).
    const actHeadOpt = opts.actHead;
    if (!actHeadOpt) {
      const bin = join(here, 'act_head.bin');
      const meta = join(here, 'act_head.meta.json');
      if (existsSync(bin) && existsSync(meta)) {
        try {
          const { loadActBin } = await import('./laya-act-bin.mjs');
          c.actW = loadActBin(bin, meta);
        } catch (err) {
          throw new LayaConfigError('invalid act_head.bin', { cause: err });
        }
      } else {
        c.actW = readJson(DEFAULT_ACT_HEAD, 'act_head');
      }
    } else {
      c.actW = readJson(actHeadOpt, 'act_head');
    }
    c.temp = { temperature: cfg.temperature, temperature_by_options: cfg.temperature_by_options };
    if (!c.temp.temperature || !c.temp.temperature_by_options) {
      throw new LayaConfigError('config missing temperature tables');
    }
    c.maxLen = cfg.max_len ?? 512;
    c.headMaxLen = cfg.head_max_len ?? 192;
    c.modelPath = modelPath;
    return c;
  }

  /** Release the ORT session. */
  async close() {
    try {
      await this.sess?.release?.();
    } finally {
      this.sess = null;
    }
  }

  /**
   * @param {object|string} state
   * @param {Record<string, {type:string,instructions:string,criteria?:any}>} questions
   */
  async predict(state, questions) {
    if (!questions || !Object.keys(questions).length) {
      throw new LayaEncodeError('predict: at least one question required');
    }
    if (Object.keys(questions).length > 32) {
      throw new RangeError(`predict: too many questions (${Object.keys(questions).length} > 32)`);
    }
    const t0 = performance.now();
    const ids = Object.keys(questions);
    const items = ids.map((qid) => {
      const q = toInternal(questions[qid]);
      const { ids: seq, markers } = buildSequence(this.tok, state, q, this.maxLen, this.headMaxLen);
      return { ids: seq, markers, qtype: QTYPES[q.t] };
    });
    const b = collateItems([items], this.tok.padId);
    const B = b.inputIds.length;
    const K = b.markerPos[0].length;
    const out = await this.sess.run(buildFeeds(ort, b));
    const { logits, pooled } = splitOutputs(out.logits.data, out.pooled.data, B, K);
    const act = actionLogits(logits, b.markerMask, pooled, this.actW);
    const nTokens = b.attentionMask.flat().reduce((a, v) => a + v, 0);
    const result = predictFromLogits(questions, items, logits, act, this.temp, nTokens);
    result.timings = { total_ms: Math.round(performance.now() - t0) };
    return result;
  }
}
