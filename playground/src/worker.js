// Playground worker: end-to-end Laya text→answer (bundled by Vite).
// ort bundled from npm (onnxruntime-web 1.30.0); shared pure-JS modules from @js.
import * as ort from 'onnxruntime-web';
import { loadBpeTokenizer } from '@js/laya-bpe.mjs';
import { toInternal, buildSequence, collateItems, QTYPES } from '@js/laya-preprocess.mjs';
import { predictFromLogits } from '@js/laya-postprocess.mjs';
import { actionLogits } from '@js/laya-action.mjs';
import { buildFeeds, splitOutputs } from '@js/laya-feed.mjs';

let tok = null,
  sess = null,
  actW = null,
  temp = null,
  readyBackend = null,
  readyModel = null;
const post = (m) => self.postMessage(m);

async function fetchWithProgress(url, onPct) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: HTTP ${resp.status}`);
  const total = Number(resp.headers.get('content-length'));
  if (!resp.body || !(total > 0)) {
    // No streaming progress (missing content-length); single download, ORT loads from URL.
    return null;
  }
  const rd = resp.body.getReader();
  const bytes = new Uint8Array(total);
  let off = 0,
    lastPct = -1;
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

self.onmessage = async (e) => {
  const { state, questions, backend = 'auto', precision = 'fp32' } = e.data;
  const modelUrl = precision === 'fp16' ? '/models/laya-split-fp16.onnx' : '/models/laya-split-single.onnx';
  const modelTag = precision === 'fp16' ? 'fp16' : 'fp32';
  try {
    const t0 = performance.now();
    if (!tok) {
      post({ type: 'progress', stage: 'tokenizer' });
      // Parallelize tokenizer + config (was sequential).
      const [tj, cfg] = await Promise.all([
        (await fetch('/js/tokenizer/tokenizer.json')).json(),
        (await fetch('/js/tokenizer/rl_agent_config.json')).json(),
      ]);
      tok = loadBpeTokenizer(tj);
      temp = { temperature: cfg.temperature, temperature_by_options: cfg.temperature_by_options };
    }
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
      let bytes = null;
      try {
        bytes = await fetchWithProgress(modelUrl, (pct) => post({ type: 'download', pct }));
      } catch (err) {
        // Fall back to ORT-direct URL load (double download, but recovers).
        console.warn('progress download failed, falling back:', err.message);
        bytes = null;
      }
      const tryBackends = backend === 'auto' ? ['webgpu', 'wasm'] : [backend];
      let lastErr = null;
      for (const ep of tryBackends) {
        try {
          if (ep === 'webgpu' && !navigator.gpu) throw new Error('no navigator.gpu');
          const opts = { executionProviders: [ep] };
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
      if (!sess) throw new Error('no backend worked: ' + String(lastErr?.message || lastErr).slice(0, 200));
      actW = await (await fetch('/js/act_head.json')).json();
    }
    post({ type: 'progress', stage: 'tokenize' });
    const ids = Object.keys(questions);
    if (!ids.length) throw new Error('no questions');
    const items = ids.map((qid) => {
      const q = toInternal(questions[qid]);
      const { ids: seq, markers } = buildSequence(tok, state, q, 512, 192);
      return { ids: seq, markers, qtype: QTYPES[q.t] };
    });
    const b = collateItems([items], tok.padId);
    const B = b.inputIds.length,
      S = b.inputIds[0].length,
      K = b.markerPos[0].length;
    const feeds = buildFeeds(ort, b);
    post({ type: 'progress', stage: 'inference' });
    const t1 = performance.now();
    const r = await sess.run(feeds);
    const inferMs = performance.now() - t1;
    const { logits, pooled } = splitOutputs(r.logits.data, r.pooled.data, B, K);
    const act = actionLogits(logits, b.markerMask, pooled, actW);
    const nTokens = b.attentionMask.flat().reduce((a, v) => a + v, 0);
    const result = predictFromLogits(questions, items, logits, act, temp, nTokens);
    post({
      type: 'done',
      result,
      inferMs: Math.round(inferMs),
      totalMs: Math.round(performance.now() - t0),
      backend: readyBackend,
      model: readyModel,
      seqLen: S,
      kmax: K,
    });
  } catch (err) {
    post({ type: 'error', message: String((err && err.message) || err).slice(0, 500) });
  }
};
