// Node SDK: tokenizer + ONNX split model + calibration.
import * as ort from 'onnxruntime-node';
import { readFileSync, existsSync } from 'node:fs';
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

import {
  CHECKPOINT_ACT_BIN,
  CHECKPOINT_ACT_META,
  DEFAULT_ACT_HEAD_BIN,
  DEFAULT_ACT_HEAD_JSON,
  DEFAULT_ACT_HEAD_META,
  DEFAULT_CONFIG,
  DEFAULT_MODEL,
  DEFAULT_TOKENIZER,
} from './laya-paths.ts';

export { DEFAULT_MODEL };

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
  /** Binary act-head weights (preferred over actHead JSON). Per-checkpoint in B2. */
  actHeadBin?: string;
  actHeadMeta?: string;
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
    const tokenizerJson = readJson(opts.tokenizer ?? DEFAULT_TOKENIZER, 'tokenizer') as TokenizerJson;
    const cfgJson = readJson(opts.config ?? DEFAULT_CONFIG, 'config') as AgentConfigJson;
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
    // Prefer binary act-head (1.1MB) over JSON (5.2MB).
    const ckpt = modelPath.includes('multilingual')
      ? 'multilingual'
      : modelPath.includes('typed-decisions')
        ? 'typed-decisions'
        : null;
    const perModelBin = ckpt ? CHECKPOINT_ACT_BIN[ckpt] : DEFAULT_ACT_HEAD_BIN;
    const perModelMeta = ckpt ? CHECKPOINT_ACT_META[ckpt] : DEFAULT_ACT_HEAD_META;
    if (!opts.actHead && !opts.actHeadBin) {
      const binPath = opts.actHeadBin ?? perModelBin;
      const metaPath = opts.actHeadMeta ?? perModelMeta;
      if (existsSync(binPath) && existsSync(metaPath)) {
        try {
          const { loadActBin } = await import('./laya-act-bin.ts');
          c.actW = loadActBin(binPath, metaPath);
        } catch (err) {
          throw new LayaConfigError('invalid act_head.bin', { cause: err });
        }
      } else {
        c.actW = readJson(DEFAULT_ACT_HEAD_JSON, 'act_head') as ActHeadWeights;
      }
    } else if (opts.actHeadBin) {
      const { loadActBin } = await import('./laya-act-bin.ts');
      c.actW = loadActBin(opts.actHeadBin, opts.actHeadMeta ?? perModelMeta);
    } else {
      c.actW = readJson(opts.actHead as string, 'act_head') as ActHeadWeights;
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
    const tTok = performance.now();
    const items: BuiltItem[] = qids.map((qid) => {
      const qdef = questions[qid];
      if (qdef === undefined) throw new LayaEncodeError(`predict: missing question ${qid}`);
      const q = toInternal(qdef);
      const { ids: seq, markers } = buildSequence(this.tok, state, q, this.maxLen, this.headMaxLen);
      return { ids: seq, markers, qtype: QTYPES[q.t] };
    });
    const b = collateItems([items], this.tok.padId);
    const d = toFeedData(b);
    const tokenizeMs = performance.now() - tTok;
    const tInf = performance.now();
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
    const inferenceMs = performance.now() - tInf;
    const tPost = performance.now();
    const { logits, pooled } = splitOutputs(
      Array.from(logitsTensor.data as ArrayLike<number>),
      Array.from(pooledTensor.data as ArrayLike<number>),
      d.dims.B,
      d.dims.K,
    );
    const act = actionLogits(logits, b.markerMask, pooled, this.actW);
    const nTokens = b.attentionMask.flat().reduce((a, v) => a + v, 0);
    const result = predictFromLogits(questions, items, logits, act, this.temp, nTokens);
    result.timings = {
      tokenize_ms: Math.round(tokenizeMs),
      inference_ms: Math.round(inferenceMs),
      postprocess_ms: Math.round(performance.now() - tPost),
      total_ms: Math.round(performance.now() - t0),
    };
    return result;
  }
}
