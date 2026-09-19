// Minimal Node.js SDK facade: tokenizer + ONNX split model + calibration.
// import { LayaClient } from './laya.mjs';
// const laya = await LayaClient.open(); console.log(await laya.predict(state, questions));
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { loadBpeTokenizer } from './laya-bpe.mjs';
import { toInternal, buildSequence, collateItems, QTYPES } from './laya-preprocess.mjs';
import { predictFromLogits } from './laya-postprocess.mjs';
import { actionLogits } from './laya-action.mjs';

const here = dirname(fileURLToPath(import.meta.url));

export class LayaClient {
  static async open(opts = {}) {
    const tokenizerJson = JSON.parse(readFileSync(opts.tokenizer ?? join(here, 'tokenizer', 'tokenizer.json'), 'utf8'));
    const cfg = JSON.parse(readFileSync(opts.config ?? join(here, 'tokenizer', 'rl_agent_config.json'), 'utf8'));
    const c = new LayaClient();
    c.tok = loadBpeTokenizer(tokenizerJson);
    c.sess = await ort.InferenceSession.create(
      opts.model ?? join(here, '..', 'models', 'laya-split-single.onnx'),
      { executionProviders: opts.providers ?? ['cpu'] });
    c.actW = JSON.parse(readFileSync(opts.actHead ?? join(here, 'act_head.json'), 'utf8'));
    c.temp = { temperature: cfg.temperature, temperature_by_options: cfg.temperature_by_options };
    c.maxLen = cfg.max_len ?? 512;
    c.headMaxLen = cfg.head_max_len ?? 192;
    return c;
  }

  async predict(state, questions) {
    const t0 = performance.now();
    const ids = Object.keys(questions);
    const items = ids.map((qid) => {
      const q = toInternal(questions[qid]);
      const { ids: seq, markers } = buildSequence(this.tok, state, q, this.maxLen, this.headMaxLen);
      return { ids: seq, markers, qtype: QTYPES[q.t] };
    });
    const b = collateItems([items], this.tok.padId);
    const B = b.inputIds.length, S = b.inputIds[0].length, K = b.markerPos[0].length;
    const toI64 = (n) => BigInt64Array.from(n.flat(Infinity).map((v) => BigInt(v)));
    const toB8 = (n) => Uint8Array.from(n.flat(Infinity).map((v) => (v ? 1 : 0)));
    const out = await this.sess.run({
      input_ids: new ort.Tensor('int64', toI64(b.inputIds), [B, S]),
      attention_mask: new ort.Tensor('int64', toI64(b.attentionMask), [B, S]),
      marker_pos: new ort.Tensor('int64', toI64(b.markerPos), [B, K]),
      marker_mask: new ort.Tensor('bool', toB8(b.markerMask), [B, K]),
      qtype: new ort.Tensor('int64', toI64([b.qtype]), [B]),
    });
    const logits = [], pooled = [];
    const ld = Array.from(out.logits.data), pd = Array.from(out.pooled.data);
    for (let i = 0; i < B; i++) { logits.push(ld.slice(i * K, (i + 1) * K)); pooled.push(pd.slice(i * 1024, (i + 1) * 1024)); }
    const act = actionLogits(logits, b.markerMask, pooled, this.actW);
    const nTokens = b.attentionMask.flat().reduce((a, v) => a + v, 0);
    const result = predictFromLogits(questions, items, logits, act, this.temp, nTokens);
    result.timings = { total_ms: Math.round(performance.now() - t0) };
    return result;
  }
}
