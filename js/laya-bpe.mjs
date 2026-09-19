// Pure-JS ByteLevel BPE for Laya tokenizer.json (no bundler, no WASM).
// Matches HF tokenizers: NFC, added-token longest-match split, GPT-2 regex,
// bytes_to_unicode mapping, BPE merge by rank. Throws on missing vocab (loud).
const GPT2_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
function bytesToUnicode() {
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  const bset = new Set(bs);
  for (let b = 0; b < 256; b++) if (!bset.has(b)) { bs.push(b); cs.push(256 + n); n++; }
  return new Map(bs.map((b, i) => [b, String.fromCodePoint(cs[i])]));
}
export function loadBpeTokenizer(tj) {
  const vocab = new Map(Object.entries(tj.model.vocab));
  const rank = new Map(tj.model.merges.map(([a, b], i) => [`${a}\0${b}`, i]));
  const added = new Map(tj.added_tokens.map(a => [a.content, { id: a.id, lstrip: !!a.lstrip, rstrip: !!a.rstrip }]));
  const addedByLen = [...added.keys()].sort((x, y) => y.length - x.length);
  const b2u = bytesToUnicode();
  const te = new TextEncoder();
  function bpe(word) {
    let parts = [...word];
    if (parts.length <= 1) return parts;
    for (;;) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = rank.get(`${parts[i]}\0${parts[i + 1]}`);
        if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
      }
      if (best < 0) break;
      parts = [...parts.slice(0, best), parts[best] + parts[best + 1], ...parts.slice(best + 2)];
      if (parts.length <= 1) break;
    }
    return parts;
  }
  function encodePiece(text) {
    const ids = [];
    GPT2_RE.lastIndex = 0;
    let m;
    while ((m = GPT2_RE.exec(text)) !== null) {
      const bytes = te.encode(m[0]);
      let mapped = '';
      for (const b of bytes) mapped += b2u.get(b);
      for (const tok of bpe(mapped)) {
        const id = vocab.get(tok);
        if (id === undefined) throw new Error(`missing vocab for ${JSON.stringify(tok)}`);
        ids.push(id);
      }
    }
    return ids;
  }
  function encode(text) {
    text = text.normalize('NFC');
    const ids = [];
    let i = 0, buf = '';
    const flush = () => { if (buf) { ids.push(...encodePiece(buf)); buf = ''; } };
    while (i < text.length) {
      let hit = null;
      for (const a of addedByLen) {
        if (text.startsWith(a, i)) { hit = a; break; }
      }
      if (hit !== null) {
        const spec = added.get(hit);
        if (spec.lstrip) buf = buf.replace(/\s+$/u, '');
        flush(); ids.push(spec.id); i += hit.length;
        if (spec.rstrip) { const m = /^\s+/u.exec(text.slice(i)); if (m) i += m[0].length; }
      }
      else { buf += text[i]; i++; }
    }
    flush();
    return ids;
  }
  const id = (s) => { const v = added.get(s)?.id ?? vocab.get(s); if (v === undefined) throw new Error(`no id for ${s}`); return v; };
  return { encode, maskToken: '[MASK]', maskId: id('[MASK]'), clsId: id('[CLS]'), sepId: id('[SEP]'), padId: id('[PAD]') };
}
