// Module worker: end-to-end Laya text→answer (pure-JS BPE + split ONNX + calibration).
// No bundler, no transformers.js. Tokenizer + model + fixtures served locally;
// ort ESM via pinned CDN.
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.all.min.mjs';
import { loadBpeTokenizer } from '/js/laya-bpe.mjs';
import { toInternal, buildSequence, collateItems, QTYPES } from '/js/laya-preprocess.mjs';
import { predictFromLogits } from '/js/laya-postprocess.mjs';
import { actionLogits } from '/js/laya-action.mjs';

let tok = null, sess = null, actW = null, temp = null;
const post = (m) => self.postMessage(m);

self.onmessage = async (e) => {
  const { state, questions, backend = 'auto' } = e.data;
  try {
    const t0 = performance.now();
    if (!tok) {
      post({ type: 'progress', stage: 'tokenizer' });
      const tj = await (await fetch('/js/tokenizer/tokenizer.json')).json();
      tok = loadBpeTokenizer(tj);
      const cfg = await (await fetch('/js/tokenizer/rl_agent_config.json')).json();
      temp = { temperature: cfg.temperature, temperature_by_options: cfg.temperature_by_options };
    }
    if (!sess) {
      post({ type: 'progress', stage: `model-${backend}` });
      // Manual fetch with byte progress (ORT's internal fetch is opaque).
      // Falls back to direct URL if length unknown.
      let bytes = null;
      try {
        const resp = await fetch('/models/laya-split-single.onnx');
        const total = Number(resp.headers.get('content-length'));
        if (resp.ok && total > 0) {
          const rd = resp.body.getReader();
          bytes = new Uint8Array(total);
          let off = 0, lastPct = -1;
          for (;;) {
            const { done, value } = await rd.read();
            if (done) break;
            bytes.set(value, off); off += value.length;
            const pct = Math.floor((off / total) * 100);
            if (pct !== lastPct && pct % 10 === 0) { lastPct = pct; post({ type: 'download', pct }); }
          }
        }
      } catch { bytes = null; }
      const tryBackends = backend === 'auto' ? ['webgpu', 'wasm'] : [backend];
      let lastErr = null;
      for (const ep of tryBackends) {
        try {
          if (ep === 'webgpu' && !navigator.gpu) throw new Error('no navigator.gpu');
          const opts = { executionProviders: [ep] };
          if (ep === 'webgpu') opts.graphOptimizationLevel = 'basic';
          sess = bytes ? await ort.InferenceSession.create(bytes, opts)
                        : await ort.InferenceSession.create('/models/laya-split-single.onnx', opts);
          post({ type: 'progress', stage: `backend-${ep}` });
          break;
        } catch (err) { lastErr = err; sess = null; }
      }
      if (!sess) throw new Error('no backend worked: ' + String(lastErr?.message || lastErr).slice(0, 200));
      actW = await (await fetch('/js/act_head.json')).json();
    }
    post({ type: 'progress', stage: 'tokenize' });
    const ids = Object.keys(questions);
    const items = ids.map(qid => {
      const q = toInternal(questions[qid]);
      const { ids: seq, markers } = buildSequence(tok, state, q, 512, 192);
      return { ids: seq, markers, qtype: QTYPES[q.t] };
    });
    const b = collateItems([items], tok.padId);
    const B = b.inputIds.length, S = b.inputIds[0].length, K = b.markerPos[0].length;
    const toI64 = (n) => BigInt64Array.from(n.flat(Infinity).map(v => BigInt(v)));
    const toB8 = (n) => Uint8Array.from(n.flat(Infinity).map(v => v ? 1 : 0));
    const feeds = {
      input_ids: new ort.Tensor('int64', toI64(b.inputIds), [B, S]),
      attention_mask: new ort.Tensor('int64', toI64(b.attentionMask), [B, S]),
      marker_pos: new ort.Tensor('int64', toI64(b.markerPos), [B, K]),
      marker_mask: new ort.Tensor('bool', toB8(b.markerMask), [B, K]),
      qtype: new ort.Tensor('int64', toI64([b.qtype]), [B]),
    };
    post({ type: 'progress', stage: 'inference' });
    const t1 = performance.now();
    const r = await sess.run(feeds);
    const inferMs = performance.now() - t1;
    const logits = [], pooled = [];
    const ld = Array.from(r.logits.data), pd = Array.from(r.pooled.data);
    for (let i = 0; i < B; i++) { logits.push(ld.slice(i * K, (i + 1) * K)); pooled.push(pd.slice(i * 1024, (i + 1) * 1024)); }
    const act = actionLogits(logits, b.markerMask, pooled, actW);
    const nTokens = b.attentionMask.flat().reduce((a, v) => a + v, 0);
    const result = predictFromLogits(questions, items, logits, act, temp, nTokens);
    post({ type: 'done', result, inferMs: Math.round(inferMs), totalMs: Math.round(performance.now() - t0) });
  } catch (err) {
    post({ type: 'error', message: String(err && err.message || err).slice(0, 500), stack: String(err?.stack || '').slice(0, 500) });
  }
};
