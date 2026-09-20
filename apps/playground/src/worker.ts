// Playground worker: end-to-end Laya text→answer (bundled by Vite).
// ort bundled from npm (onnxruntime-web 1.30.0); shared pure-TS modules from @js.
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

// Static assets served through Vite (?url): no public/ symlinks, no absolute
// /js/* paths. Models stay out of the bundle (dev-only /models route).
import tokenizerUrl from '@laya/js/src/tokenizer/tokenizer.json?url';
import configUrl from '@laya/js/src/tokenizer/rl_agent_config.json?url';
import actBinUrl from '@laya/test-vectors/vectors/act_head.bin?url';
import actMetaUrl from '@laya/test-vectors/vectors/act_head.meta.json?url';

let tok: Tokenizer | null = null;
let sess: ort.InferenceSession | null = null;
let actW: ActHeadWeights | null = null;
let temp: TemperatureConfig | null = null;
let readyBackend: string | null = null;
let readyModel: string | null = null;

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
    // No streaming progress (missing content-length); single download, ORT loads from URL.
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

const ASSET_CACHE = 'laya-assets-v1';

// Cache-first for immutable versioned assets (tokenizer, config, act head,
// models). First visit streams with progress and populates the cache;
// repeat visits load from disk with no download events.
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
        /* quota or opaque failures: run uncached */
      }
      return bytes;
    }
  } catch {
    /* CacheStorage unavailable (private mode): fall through */
  }
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: HTTP ${resp.status}`);
  return new Uint8Array(await resp.arrayBuffer());
}

workerSelf.onmessage = (e: MessageEvent<WorkerRequest>) => {
  void (async () => {
    const { state, questions, backend = 'auto', precision = 'fp32' } = e.data;
    const modelUrl = precision === 'fp16' ? '/models/laya-split-fp16.onnx' : '/models/laya-split-single.onnx';
    const modelTag = precision === 'fp16' ? 'fp16' : 'fp32';
    try {
      const t0 = performance.now();
      if (!tok) {
        post({ type: 'progress', stage: 'tokenizer' });
        // Parallelize tokenizer + config (was sequential); cached after first visit.
        const noop = (_pct: number): void => undefined;
        const [tjBytes, cfgBytes] = await Promise.all([
          cachedBytes(tokenizerUrl, noop),
          cachedBytes(configUrl, noop),
        ]);
        const tj = JSON.parse(new TextDecoder().decode(tjBytes)) as TokenizerJson;
        const cfg = JSON.parse(new TextDecoder().decode(cfgBytes)) as {
          temperature: number[];
          temperature_by_options: Record<string, number>;
        };
        tok = loadBpeTokenizer(tj);
        const cfgT = cfg;
        temp = { temperature: cfgT.temperature, temperature_by_options: cfgT.temperature_by_options };
      }
      const tokenizer = tok;
      const temps = temp;
      if (!tokenizer || !temps) throw new Error('tokenizer not ready');
      if (!sess || readyModel !== modelTag) {
        if (sess && readyModel !== modelTag) {
          try {
            await sess.release?.();
          } catch {
            /* ignore */
          }
          sess = null;
        }
        post({ type: 'progress', stage: `model-${backend}-${modelTag}` });
        // Cache-first: repeat visits skip the download entirely (no pct events).
        let bytes: Uint8Array | null = null;
        try {
          bytes = await cachedBytes(modelUrl, (pct) => post({ type: 'download', pct }));
        } catch (err) {
          // Fall back to ORT-direct URL load (double download, but recovers).
          console.warn('progress download failed, falling back:', err instanceof Error ? err.message : String(err));
          bytes = null;
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
        // Binary act head (1.1MB) over JSON (5.2MB); cached after first visit.
        const noopAct = (_pct: number): void => undefined;
        const [binBytes, metaBytes] = await Promise.all([
          cachedBytes(actBinUrl, noopAct),
          cachedBytes(actMetaUrl, noopAct),
        ]);
        actW = parseActHeadBin(
          binBytes,
          JSON.parse(new TextDecoder().decode(metaBytes)) as ActHeadMeta,
        );
      }
      const session = sess;
      const weights = actW;
      if (!session || !weights) throw new Error('session not ready');
      post({ type: 'progress', stage: 'tokenize' });
      const ids = Object.keys(questions);
      if (!ids.length) throw new Error('no questions');
      const items: BuiltItem[] = ids.map((qid) => {
        const qdef = questions[qid];
        if (qdef === undefined) throw new Error(`missing question ${qid}`);
        const q = toInternal(qdef);
        const { ids: seq, markers } = buildSequence(tokenizer, state, q, 512, 192);
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
      const t1 = performance.now();
      const r = await session.run(feeds);
      const inferMs = performance.now() - t1;
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
      post({
        type: 'done',
        result,
        inferMs: Math.round(inferMs),
        totalMs: Math.round(performance.now() - t0),
        backend: readyBackend ?? 'unknown',
        model: readyModel ?? modelTag,
        seqLen: d.dims.S,
        kmax: d.dims.K,
      });
    } catch (err) {
      post({ type: 'error', message: String((err as Error)?.message ?? err).slice(0, 500) });
    }
  })();
};
