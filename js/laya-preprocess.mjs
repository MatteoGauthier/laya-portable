// Faithful port of upstream laya/common.py preprocessing + agent._to_internal.
// Must produce byte-identical token IDs to Python for the same inputs.
// Differences from naive JS: Python json.dumps separators (', ', ': '),
// insertion-order preservation, leading-space handling, marker sanitization.
export function pyStringify(value) {
  // json.dumps(value, ensure_ascii=False) with default separators (', ', ': ').
  // Covers str/dict/list/number/bool/None. Non-ASCII preserved (like ensure_ascii=False).
  // For unserializable (Python default=str in render_criterion), fall back to String().
  const seen = new Set();
  function enc(v) {
    if (v === null || v === undefined) return 'null';
    if (typeof v === 'string') return JSON.stringify(v);
    if (typeof v === 'number') {
      if (!isFinite(v)) return 'null'; // Python json allows NaN/Infinity; JS tokenizer inputs never contain them; map to null like JS
      return JSON.stringify(v);
    }
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (Array.isArray(v)) return '[' + v.map(enc).join(', ') + ']';
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
  const t = q.t, crit = q.crit;
  if (t === 'choice') {
    return Object.entries(crit).map(([k, v]) =>
      (v === null || v === undefined || v === '') ? k : `${k}: ${renderCriterion(v)}`);
  }
  if (t === 'score') return crit.map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  const c = crit || {};
  const f = c['false'], tr = c['true'];
  const no = (f === null || f === undefined || f === '') ? 'no, the statement does not hold' : renderCriterion(f);
  const yes = (tr === null || tr === undefined || tr === '') ? 'yes, the statement holds' : renderCriterion(tr);
  return [`false: ${no}`, `true: ${yes}`];
}
export function toInternal(qdef) {
  const t = qdef.type;
  let crit = qdef.criteria;
  if (t === 'choice' && Array.isArray(crit)) crit = Object.fromEntries(crit.map(c => [String(c), null]));
  let ins = qdef.instructions;
  if (typeof ins !== 'string') ins = pyStringify(ins);
  return { t, ins, crit };
}
// tok: { encode(text) -> number[], maskToken, maskId, clsId, sepId, padId }
export function buildSequence(tok, state, q, maxLen = 512, headMaxLen = 192, optionOrder = null, truncateLeft = false) {
  const maskTok = tok.maskToken;
  const opts = renderOptions(q);
  const order = optionOrder ?? opts.map((_, i) => i);
  const ins = String(q.ins).replaceAll(maskTok, ' ');
  let headIds = tok.encode(`${q.t} question: ${ins}`);
  let optIds = order.map(i => [tok.maskId, ...tok.encode(' ' + opts[i].replaceAll(maskTok, ' ')).slice(0, 48)]);
  let optBudget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
  if (optBudget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map(o => o.slice(0, per));
    optBudget = headMaxLen - optIds.reduce((a, o) => a + o.length, 0);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));
  let ids = [tok.clsId, ...headIds, tok.sepId];
  const markers = [];
  for (const o of optIds) { markers.push(ids.length); ids.push(...o); }
  ids.push(tok.sepId);
  const room = Math.max(0, maxLen - ids.length - 1);
  let st = tok.encode(serializeState(state).replaceAll(maskTok, ' '));
  st = truncateLeft ? st.slice(-room) : st.slice(0, room);
  ids = [...ids, ...st, tok.sepId];
  return { ids: ids.slice(0, maxLen), markers: markers.filter(m => m < maxLen) };
}
export const QTYPES = { choice: 0, score: 1, noul: 2 };
export function collateItems(batch, padId) {
  const items = batch.flat();
  if (!items.length) return null;
  const n = items.length;
  const L = Math.max(...items.map(it => it.ids.length));
  const kmax = Math.max(...items.map(it => it.markers.length));
  const inputIds = [], attentionMask = [], markerPos = [], markerMask = [], qtype = [];
  for (const it of items) {
    const ids = [...it.ids, ...Array(L - it.ids.length).fill(padId)];
    const att = [...Array(it.ids.length).fill(1), ...Array(L - it.ids.length).fill(0)];
    const mp = [...it.markers, ...Array(kmax - it.markers.length).fill(0)];
    const mm = [...Array(it.markers.length).fill(true), ...Array(kmax - it.markers.length).fill(false)];
    inputIds.push(ids); attentionMask.push(att); markerPos.push(mp); markerMask.push(mm); qtype.push(it.qtype);
  }
  return { inputIds, attentionMask, markerPos, markerMask, qtype };
}
