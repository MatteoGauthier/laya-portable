// Minimal Node.js SDK facade: tokenizer + ONNX split model + calibration.
// import { LayaClient } from './laya.ts';
// const laya = await LayaClient.open(); console.log(await laya.predict(state, questions));
import * as ort from 'onnxruntime-node';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadBpeTokenizer } from './laya-bpe.ts';
import { toInternal, buildSequence, collateItems, QTYPES } from './laya-preprocess.ts';
import { predictFromLogits } from './laya-postprocess.ts';
import { actionLogits } from './laya-action.ts';
import { toFeedData, splitOutputs } from './laya-feed.ts';
import { LayaConfigError, LayaEncodeError } from './laya-errors.ts';
import type {
  ActHeadWeights,
  BuiltItem,
  PredictResult,
  Questions,
  TemperatureConfig,
  Tokenizer,
  TokenizerJson,
} from './laya-types.ts';

const here = dirname(fileURLToPath(import.meta.url));

/** Canonical default model (single-file split FP32). CLI/worker/checks must match. */
export const DEFAULT_MODEL = join(here, '..', 'models', 'laya-split-single.onnx');
export const DEFAULT_TOKENIZER = join(here, 'tokenizer', 'tokenizer.json');
export const DEFAULT_CONFIG = join(here, 'tokenizer', 'rl_agent_config.json');
export const DEFAULT_ACT_HEAD = join(here, 'act_head.json');

function readJson(path: string, label: string): unknown {
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as unknown;
  } catch (err) {
    throw new LayaConfigError(`cannot read ${label}: ${path}`, { cause: err });
  }
}

export interface LayaOpenOptions {
  model?: string;
  tokenizer?: string;
  config?: string;
  actHead?: string;
  providers?: string[];
}

interface AgentConfigJson {
  temperature: number[];
  temperature_by_options: Record<string, number>;
  max_len?: number;
  head_max_len?: number;
}

export class LayaClient {
  tok!: Tokenizer;
  sess!: ort.InferenceSession;
  actW!: ActHeadWeights;
  temp!: TemperatureConfig;
  maxLen!: number;
  headMaxLen!: number;
  modelPath!: string;

  static async open(opts: LayaOpenOptions = {}): Promise<LayaClient> {
    const modelPath = opts.model ?? DEFAULT_MODEL;
    const [tokenizerJson, cfgJson] = await Promise.all([
      Promise.resolve().then(() => readJson(opts.tokenizer ?? DEFAULT_TOKENIZER, 'tokenizer') as TokenizerJson),
      Promise.resolve().then(() => readJson(opts.config ?? DEFAULT_CONFIG, 'config') as AgentConfigJson),
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
    if (!opts.actHead) {
      const bin = join(here, 'act_head.bin');
      const meta = join(here, 'act_head.meta.json');
      if (existsSync(bin) && existsSync(meta)) {
        try {
          const { loadActBin } = await import('./laya-act-bin.ts');
          c.actW = loadActBin(bin, meta);
        } catch (err) {
          throw new LayaConfigError('invalid act_head.bin', { cause: err });
        }
      } else {
        c.actW = readJson(DEFAULT_ACT_HEAD, 'act_head') as ActHeadWeights;
      }
    } else {
      c.actW = readJson(opts.actHead, 'act_head') as ActHeadWeights;
    }
    if (!cfgJson.temperature || !cfgJson.temperature_by_options) {
      throw new LayaConfigError('config missing temperature tables');
    }
    c.temp = { temperature: cfgJson.temperature, temperature_by_options: cfgJson.temperature_by_options };
    c.maxLen = cfgJson.max_len ?? 512;
    c.headMaxLen = cfgJson.head_max_len ?? 192;
    c.modelPath = modelPath;
    return c;
  }

  /** Release the ORT session. */
  async close(): Promise<void> {
    try {
      await this.sess?.release?.();
    } finally {
      this.sess = null as unknown as ort.InferenceSession;
    }
  }

  async predict(state: unknown, questions: Questions): Promise<PredictResult> {
    const qids = Object.keys(questions);
    if (!qids.length) {
      throw new LayaEncodeError('predict: at least one question required');
    }
    if (qids.length > 32) {
      throw new RangeError(`predict: too many questions (${qids.length} > 32)`);
    }
    const t0 = performance.now();
    const items: BuiltItem[] = qids.map((qid) => {
      const qdef = questions[qid];
      if (qdef === undefined) throw new LayaEncodeError(`predict: missing question ${qid}`);
      const q = toInternal(qdef);
      const { ids: seq, markers } = buildSequence(this.tok, state, q, this.maxLen, this.headMaxLen);
      return { ids: seq, markers, qtype: QTYPES[q.t] };
    });
    const b = collateItems([items], this.tok.padId);
    const d = toFeedData(b);
    const out = await this.sess.run({
      input_ids: new ort.Tensor('int64', d.inputIds, [d.dims.B, d.dims.S]),
      attention_mask: new ort.Tensor('int64', d.attentionMask, [d.dims.B, d.dims.S]),
      marker_pos: new ort.Tensor('int64', d.markerPos, [d.dims.B, d.dims.K]),
      marker_mask: new ort.Tensor('bool', d.markerMask, [d.dims.B, d.dims.K]),
      qtype: new ort.Tensor('int64', d.qtype, [d.dims.B]),
    });
    const logitsTensor = out['logits'];
    const pooledTensor = out['pooled'];
    if (!logitsTensor || !pooledTensor) throw new LayaEncodeError('predict: model missing logits/pooled outputs');
    const { logits, pooled } = splitOutputs(
      Array.from(logitsTensor.data as ArrayLike<number>),
      Array.from(pooledTensor.data as ArrayLike<number>),
      d.dims.B,
      d.dims.K,
    );
    const act = actionLogits(logits, b.markerMask, pooled, this.actW);
    const nTokens = b.attentionMask.flat().reduce((a, v) => a + v, 0);
    const result = predictFromLogits(questions, items, logits, act, this.temp, nTokens);
    result.timings = { total_ms: Math.round(performance.now() - t0) };
    return result;
  }
}
