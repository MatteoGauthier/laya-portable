// Faithful port of upstream laya/common.py preprocessing + agent._to_internal.
// Must produce byte-identical token IDs to Python for the same inputs.
// Differences from naive JS: Python json.dumps separators (', ', ': '),
// insertion-order preservation, leading-space handling, marker sanitization.
//
// Bounds mirror upstream caps: per-option 48 tokens, head window 192,
// minimum 8 head tokens, tail window 512. See export/export_onnx.py contract
// ([B,S] inputs, dynamic B/S/K, K>=2 expected).
import { LayaEncodeError } from './laya-errors.mjs';

export const PER_OPTION_CAP = 48;
export const HEAD_RESERVE = 16;
export const HEAD_MIN_PER_OPTION = 4;
export const HEAD_MIN_TOKENS = 8;
export function pyStringify(value) {
  // json.dumps(value, ensure_ascii=False) with default separators (', ', ': ').
  // Covers str/dict/list/number/bool/None. Non-ASCII preserved (like ensure_ascii=False).
  // For unserializable (Python default=str in render_criterion), fall back to String().
  const seen = new Set();
  function enc(v) {
    if (v === null || v === undefined) return 'null';
    if (typeof v === 'string') return JSON.stringify(v);
    if (typeof v === 'number') {
      // Fail loud: Python json emits NaN/Infinity but tokenizer inputs must not
      // contain non-finite floats (silent 'null' would drift answers).
      if (!isFinite(v)) throw new LayaEncodeError(`non-finite number in state: ${String(v)}`);
      return JSON.stringify(v);
    }
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (Array.isArray(v)) {
      if (seen.has(v)) throw new LayaEncodeError('circular reference in state');
      seen.add(v);
      try {
        return '[' + v.map(enc).join(', ') + ']';
      } finally {
        seen.delete(v);
      }
    }
    if (typeof v === 'object') {
      if (seen.has(v)) throw new Error('circular');
      seen.add(v);
      const parts = [];
      for (const [k, val] of Object.entries(v)) parts.push(JSON.stringify(String(k)) + ': ' + enc(val));
      seen.delete(v);
      return '{' + parts.join(', ') + '}';
    }
    return JSON.stringify(String(v)); // default=str fallback
  }
  return enc(value);
}
export function serializeState(state) {
  if (typeof state === 'string') return state;
  return pyStringify(state);
}
export function renderCriterion(value) {
  if (typeof value === 'string') return value;
  return pyStringify(value);
}
export function renderOptions(q) {
  const t = q.t,
    crit = q.crit;
  if (t === 'choice') {
    return Object.entries(crit).map(([k, v]) =>
      v === null || v === undefined || v === '' ? k : `${k}: ${renderCriterion(v)}`,
    );
  }
  if (t === 'score') return crit.map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  const c = crit || {};
  const f = c['false'],
    tr = c['true'];
  const no = f === null || f === undefined || f === '' ? 'no, the statement does not hold' : renderCriterion(f);
  const yes = tr === null || tr === undefined || tr === '' ? 'yes, the statement holds' : renderCriterion(tr);
  return [`false: ${no}`, `true: ${yes}`];
}
export function toInternal(qdef) {
  if (!qdef || typeof qdef !== 'object')
    throw new LayaEncodeError(`invalid question def: ${JSON.stringify(qdef)?.slice(0, 120)}`);
  const t = qdef.type;
  if (t !== 'choice' && t !== 'score' && t !== 'noul')
    throw new LayaEncodeError(`unknown question type: ${JSON.stringify(t)}`);
  let crit = qdef.criteria;
  if (t === 'choice' && Array.isArray(crit)) crit = Object.fromEntries(crit.map((c) => [String(c), null]));
  if (t === 'score' && !Array.isArray(crit)) throw new LayaEncodeError('score criteria must be an array');
  let ins = qdef.instructions;
  if (typeof ins !== 'string') ins = pyStringify(ins);
  return { t, ins, crit };
}
// tok: { encode(text) -> number[], maskToken, maskId, clsId, sepId, padId }
// optionOrder/truncateLeft are supported (tested in tests/) but default to
// full-order + right-truncate like upstream.
export function buildSequence(tok, state, q, maxLen = 512, headMaxLen = 192, optionOrder = null, truncateLeft = false) {
  if (!tok || typeof tok.encode !== 'function') throw new LayaEncodeError('buildSequence: invalid tokenizer');
  const maskTok = tok.maskToken;
  const opts = renderOptions(q);
  if (!opts.length) throw new LayaEncodeError('buildSequence: question has no options');
  const order = optionOrder ?? opts.map((_, i) => i);
  const ins = String(q.ins).replaceAll(maskTok, ' ');
  let headIds = tok.encode(`${q.t} question: ${ins}`);
  let optIds = order.map((i) => [
    tok.maskId,
    ...tok.encode(' ' + opts[i].replaceAll(maskTok, ' ')).slice(0, PER_OPTION_CAP),
  ]);
  let optBudget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
  if (optBudget < HEAD_RESERVE) {
    const per = Math.max(HEAD_MIN_PER_OPTION, Math.floor((headMaxLen - HEAD_RESERVE) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    optBudget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
  }
  headIds = headIds.slice(0, Math.max(HEAD_MIN_TOKENS, optBudget));
  let ids = [tok.clsId, ...headIds, tok.sepId];
  const markers = [];
  for (const o of optIds) {
    markers.push(ids.length);
    ids.push(...o);
  }
  ids.push(tok.sepId);
  const room = Math.max(0, maxLen - ids.length - 1);
  let st = tok.encode(serializeState(state).replaceAll(maskTok, ' '));
  st = truncateLeft ? st.slice(-room) : st.slice(0, room);
  ids = [...ids, ...st, tok.sepId];
  const outIds = ids.slice(0, maxLen);
  const outMarkers = markers.filter((m) => m < maxLen);
  // Guard: downstream softmax/action head require K>=1 (K>=2 expected by contract).
  if (outMarkers.length < 1) throw new LayaEncodeError(`buildSequence: all markers truncated (maxLen=${maxLen})`);
  return { ids: outIds, markers: outMarkers };
}
export const QTYPES = { choice: 0, score: 1, noul: 2 };
export function collateItems(batch, padId) {
  const items = batch.flat();
  if (!items.length) throw new LayaEncodeError('collateItems: empty batch (no questions?)');
  const L = Math.max(...items.map((it) => it.ids.length));
  const kmax = Math.max(...items.map((it) => it.markers.length));
  const inputIds = [],
    attentionMask = [],
    markerPos = [],
    markerMask = [],
    qtype = [];
  for (const it of items) {
    const ids = [...it.ids, ...Array(L - it.ids.length).fill(padId)];
    const att = [...Array(it.ids.length).fill(1), ...Array(L - it.ids.length).fill(0)];
    const mp = [...it.markers, ...Array(kmax - it.markers.length).fill(0)];
    const mm = [...Array(it.markers.length).fill(true), ...Array(kmax - it.markers.length).fill(false)];
    inputIds.push(ids);
    attentionMask.push(att);
    markerPos.push(mp);
    markerMask.push(mm);
    qtype.push(it.qtype);
  }
  return { inputIds, attentionMask, markerPos, markerMask, qtype };
}
