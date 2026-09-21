// Playground worker: text→answer (bundled by Vite).
import * as ort from 'onnxruntime-web';
import { loadBpeTokenizer } from '@laya/js/laya-bpe.ts';
import { toInternal, buildSequence, collateItems, QTYPES } from '@laya/js/laya-preprocess.ts';
import { predictFromLogits } from '@laya/js/laya-postprocess.ts';
import { actionLogits } from '@laya/js/laya-action.ts';
import { toFeedData, splitOutputs } from '@laya/js/laya-feed.ts';
import type {
  ActHeadWeights,
  BuiltItem,
  CollatedBatch,
  TemperatureConfig,
  Tokenizer,
  TokenizerJson,
  WorkerRequest,
  WorkerResponse,
} from '@laya/js/laya-types.ts';
import { parseActHeadBin, type ActHeadMeta } from '@laya/js/act-bin-parse.ts';

// Static assets via Vite (?url). Models stay out of the bundle.
import tokenizerUrl from '@laya/js/src/tokenizer/tokenizer.json?url';
import configUrl from '@laya/js/src/tokenizer/rl_agent_config.json?url';
import actBinUrl from '@laya/test-vectors/vectors/act_head.bin?url';
import actMetaUrl from '@laya/test-vectors/vectors/act_head.meta.json?url';
import { route, type CheckpointName } from '@laya/js/laya-router.ts';

type CheckpointSel = 'auto' | CheckpointName;

// Local dev serves repo models/ at /models; hosted builds point at R2.
const MODELS_BASE = (import.meta.env.VITE_MODELS_BASE_URL as string | undefined) ?? '/models';

const MODEL_URLS: Record<CheckpointName, { fp32: string; fp16: string | null }> = {
  english: {
    fp32: `${MODELS_BASE}/laya-split-single.onnx`,
    fp16: `${MODELS_BASE}/laya-split-fp16.onnx`,
  },
  multilingual: { fp32: `${MODELS_BASE}/laya-multilingual-split-single.onnx`, fp16: null },
  'typed-decisions': { fp32: `${MODELS_BASE}/laya-typed-decisions-split-single.onnx`, fp16: null },
};

const TOKENIZER_URLS: Record<CheckpointName, string> = {
  english: tokenizerUrl,
  multilingual: `${MODELS_BASE}/laya-multilingual.tokenizer.json`,
  'typed-decisions': `${MODELS_BASE}/laya-typed-decisions.tokenizer.json`,
};

const CONFIG_URLS: Record<CheckpointName, string> = {
  english: configUrl,
  multilingual: `${MODELS_BASE}/laya-multilingual.rl_agent_config.json`,
  'typed-decisions': `${MODELS_BASE}/laya-typed-decisions.rl_agent_config.json`,
};

const ACT_BIN_URLS: Record<CheckpointName, string> = {
  english: actBinUrl,
  multilingual: `${MODELS_BASE}/laya-multilingual.act_head.bin`,
  'typed-decisions': `${MODELS_BASE}/laya-typed-decisions.act_head.bin`,
};

const ACT_META_URLS: Record<CheckpointName, string> = {
  english: actMetaUrl,
  multilingual: `${MODELS_BASE}/laya-multilingual.act_head.meta.json`,
  'typed-decisions': `${MODELS_BASE}/laya-typed-decisions.act_head.meta.json`,
};

let tok: Tokenizer | null = null;
let sess: ort.InferenceSession | null = null;
let actW: ActHeadWeights | null = null;
let temp: TemperatureConfig | null = null;
let maxLen = 512;
let headMaxLen = 192;
let readyBackend: string | null = null;
let readyModel: string | null = null;
let readyCheckpoint: CheckpointName | null = null;

const workerSelf = self as unknown as {
  postMessage(m: WorkerResponse): void;
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
};
const post = (m: WorkerResponse): void => workerSelf.postMessage(m);

async function fetchWithProgress(url: string, onPct: (pct: number) => void): Promise<Uint8Array | null> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: HTTP ${resp.status}`);
  const total = Number(resp.headers.get('content-length'));
  if (!resp.body || !(total > 0)) {
    return null;
  }
  const rd = resp.body.getReader();
  const bytes = new Uint8Array(total);
  let off = 0;
  let lastPct = -1;
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    if (off + value.length > total) throw new Error(`fetch ${url}: content-length mismatch`);
    bytes.set(value, off);
    off += value.length;
    const pct = Math.floor((off / total) * 100);
    if (pct !== lastPct && pct % 5 === 0) {
      lastPct = pct;
      onPct(pct);
    }
  }
  return bytes.subarray(0, off);
}

const ASSET_CACHE = 'laya-assets-v2';
const noopProgress = (_pct: number): void => undefined;

async function releaseQuietly(sess: ort.InferenceSession | null): Promise<void> {
  try {
    await sess?.release?.();
  } catch {
    // ignore
  }
}

// Cache-first for immutable assets; repeat visits skip the download.
async function cachedBytes(url: string, onPct: (pct: number) => void): Promise<Uint8Array> {
  try {
    const cache = await caches.open(ASSET_CACHE);
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      if (buf.byteLength > 0) return new Uint8Array(buf);
    }
    const bytes = await fetchWithProgress(url, onPct);
    if (bytes) {
      try {
        await cache.put(url, new Response(bytes.slice(), { headers: { 'content-length': String(bytes.length) } }));
      } catch {
        // quota: run uncached
      }
      return bytes;
    }
  } catch {
    // CacheStorage unavailable: fall through to plain fetch
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: HTTP ${resp.status}`);
  return new Uint8Array(await resp.arrayBuffer());
}

workerSelf.onmessage = (e: MessageEvent<WorkerRequest>) => {
  void (async () => {
    const { state, questions, backend = 'auto', precision = 'fp16' } = e.data;
    const sel = (e.data.checkpoint ?? e.data.model ?? 'auto') as CheckpointSel;
    const langHint = e.data.lang;
    const decision =
      sel === 'auto'
        ? route(state, questions, langHint ? { lang: langHint } : {})
        : route(state, questions, { model: sel });
    const checkpoint = decision.model;
    const urls = MODEL_URLS[checkpoint];
    const modelUrl = precision === 'fp16' ? (urls.fp16 ?? urls.fp32) : urls.fp32;
    const modelTag = `${checkpoint}-${precision === 'fp16' && urls.fp16 ? 'fp16' : 'fp32'}`;
    try {
      const t0 = performance.now();
      let downloadMs = 0;
      let sessionMs = 0;
      if (!tok || readyCheckpoint !== checkpoint) {
        tok = null;
        temp = null;
        actW = null;
        if (sess) {
          await releaseQuietly(sess);
          sess = null;
        }
        readyCheckpoint = checkpoint;
        post({ type: 'progress', stage: `tokenizer-${checkpoint}` });
        const [tjBytes, cfgBytes] = await Promise.all([
          cachedBytes(TOKENIZER_URLS[checkpoint], noopProgress),
          cachedBytes(CONFIG_URLS[checkpoint], noopProgress),
        ]);
        const tj = JSON.parse(new TextDecoder().decode(tjBytes)) as TokenizerJson;
        const cfg = JSON.parse(new TextDecoder().decode(cfgBytes)) as {
          temperature: number[];
          temperature_by_options: Record<string, number>;
          max_len?: number;
          head_max_len?: number;
        };
        tok = loadBpeTokenizer(tj);
        temp = { temperature: cfg.temperature, temperature_by_options: cfg.temperature_by_options };
        maxLen = cfg.max_len ?? 512;
        headMaxLen = cfg.head_max_len ?? 192;
      }
      const tokenizer = tok;
      const temps = temp;
      if (!tokenizer || !temps) throw new Error('tokenizer not ready');
      if (!sess || readyModel !== modelTag) {
        if (sess && readyModel !== modelTag) {
          await releaseQuietly(sess);
          sess = null;
        }
        post({ type: 'progress', stage: `model-${backend}-${modelTag}` });
        let bytes: Uint8Array | null = null;
        const tDl = performance.now();
        try {
          bytes = await cachedBytes(modelUrl, (pct) => post({ type: 'download', pct }));
        } catch (err) {
          // Fall back to ORT-direct URL load.
          console.warn('progress download failed, falling back:', err instanceof Error ? err.message : String(err));
          bytes = null;
        }
        downloadMs = performance.now() - tDl;
        const tSess = performance.now();
        // Hosted builds serve the >25MB ORT wasm from R2 (Pages file cap);
        // local dev keeps the default same-origin resolution.
        if (MODELS_BASE.startsWith('http')) {
          ort.env.wasm.wasmPaths = `${MODELS_BASE}/ort/`;
        }
        const tryBackends = backend === 'auto' ? ['webgpu', 'wasm'] : [backend];
        let lastErr: unknown = null;
        for (const ep of tryBackends) {
          try {
            const gpu = (navigator as Navigator & { gpu?: unknown }).gpu;
            if (ep === 'webgpu' && !gpu) throw new Error('no navigator.gpu');
            const opts: ort.InferenceSession.SessionOptions = { executionProviders: [ep] };
            // TODO(ort>1.30): re-test default graphOptimizationLevel; 'basic'
            // works around a SkipLayerNormalization fusion bug in 1.30.
            if (ep === 'webgpu') opts.graphOptimizationLevel = 'basic';
            sess = bytes
              ? await ort.InferenceSession.create(bytes, opts)
              : await ort.InferenceSession.create(modelUrl, opts);
            readyBackend = ep;
            readyModel = modelTag;
            post({ type: 'progress', stage: `backend-${ep}` });
            break;
          } catch (err) {
            lastErr = err;
            sess = null;
          }
        }
        if (!sess) {
          throw new Error(
            `no backend worked: ${String(lastErr instanceof Error ? lastErr.message : lastErr).slice(0, 200)}`,
          );
        }
        // Binary act head over JSON; cached after first visit.
        const [binBytes, metaBytes] = await Promise.all([
          cachedBytes(ACT_BIN_URLS[checkpoint], noopProgress),
          cachedBytes(ACT_META_URLS[checkpoint], noopProgress),
        ]);
        actW = parseActHeadBin(binBytes, JSON.parse(new TextDecoder().decode(metaBytes)) as ActHeadMeta);
        sessionMs = performance.now() - tSess;
      }
      const session = sess;
      const weights = actW;
      if (!session || !weights) throw new Error('session not ready');
      post({ type: 'progress', stage: 'tokenize' });
      const tTok = performance.now();
      const ids = Object.keys(questions);
      if (!ids.length) throw new Error('no questions');
      const items: BuiltItem[] = ids.map((qid) => {
        const qdef = questions[qid];
        if (qdef === undefined) throw new Error(`missing question ${qid}`);
        const q = toInternal(qdef);
        const { ids: seq, markers } = buildSequence(tokenizer, state, q, maxLen, headMaxLen);
        return { ids: seq, markers, qtype: QTYPES[q.t] };
      });
      const b: CollatedBatch = collateItems([items], tokenizer.padId);
      const d = toFeedData(b);
      const feeds: Record<string, ort.Tensor> = {
        input_ids: new ort.Tensor('int64', d.inputIds, [d.dims.B, d.dims.S]),
        attention_mask: new ort.Tensor('int64', d.attentionMask, [d.dims.B, d.dims.S]),
        marker_pos: new ort.Tensor('int64', d.markerPos, [d.dims.B, d.dims.K]),
        marker_mask: new ort.Tensor('bool', d.markerMask, [d.dims.B, d.dims.K]),
        qtype: new ort.Tensor('int64', d.qtype, [d.dims.B]),
      };
      post({ type: 'progress', stage: 'inference' });
      const tokenizeMs = performance.now() - tTok;
      const t1 = performance.now();
      const r = await session.run(feeds);
      const inferenceMs = performance.now() - t1;
      const tPost = performance.now();
      const logitsTensor = r['logits'];
      const pooledTensor = r['pooled'];
      if (!logitsTensor || !pooledTensor) throw new Error('model missing logits/pooled outputs');
      const { logits, pooled } = splitOutputs(
        Array.from(logitsTensor.data as ArrayLike<number>),
        Array.from(pooledTensor.data as ArrayLike<number>),
        d.dims.B,
        d.dims.K,
      );
      const act = actionLogits(logits, b.markerMask, pooled, weights);
      const nTokens = b.attentionMask.flat().reduce((a, v) => a + v, 0);
      const result = predictFromLogits(questions, items, logits, act, temps, nTokens);
      const timings = {
        tokenize_ms: Math.round(tokenizeMs),
        inference_ms: Math.round(inferenceMs),
        postprocess_ms: Math.round(performance.now() - tPost),
        total_ms: Math.round(performance.now() - t0),
      };
      result.timings = timings;
      post({
        type: 'done',
        result,
        timings,
        setup: { download_ms: Math.round(downloadMs), session_ms: Math.round(sessionMs) },
        backend: readyBackend ?? 'unknown',
        model: readyModel ?? modelTag,
        seqLen: d.dims.S,
        kmax: d.dims.K,
        routing: { model: decision.model, reason: decision.reason },
      });
    } catch (err) {
      post({ type: 'error', message: String((err as Error)?.message ?? err).slice(0, 500) });
    }
  })();
};
