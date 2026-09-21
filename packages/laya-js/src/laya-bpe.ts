// Pure-JS BPE for Laya tokenizer.json files (no bundler, no WASM).
// ByteLevel (english, typed-decisions) and Metaspace (multilingual mmBERT,
// space->▁, no NFC) auto-detected from tokenizer.json. Throws on missing vocab.
import { LayaEncodeError } from './laya-errors.ts';
import type { Tokenizer, TokenizerJson } from './laya-types.ts';

// New regex per call: module-global /g would race on shared lastIndex.
const GPT2_SRC = `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+`;
const GPT2_FLAGS = 'gu';
const WS_SENTINEL = '▁';

function bytesToUnicode(): Map<number, string> {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  const bset = new Set(bs);
  for (let b = 0; b < 256; b++) {
    if (!bset.has(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  return new Map(bs.map((b, i) => [b, String.fromCodePoint(cs[i] as number)]));
}

export function loadBpeTokenizer(tj: TokenizerJson): Tokenizer {
  const vocab = new Map(Object.entries(tj.model.vocab));
  const rank = new Map(tj.model.merges.map(([a, b], i) => [`${a}\0${b}`, i]));
  const added = new Map(tj.added_tokens.map((a) => [a.content, { id: a.id, lstrip: !!a.lstrip, rstrip: !!a.rstrip }]));
  const addedByLen = [...added.keys()].sort((x, y) => y.length - x.length);
  const b2u = bytesToUnicode();
  const te = new TextEncoder();
  const isMetaspace = tj.pre_tokenizer?.type === 'Metaspace';
  const spaceToSentinel =
    tj.normalizer?.type === 'Replace' && tj.normalizer.pattern?.String === ' ' && tj.normalizer.content === WS_SENTINEL;
  const byteFallback = tj.model.byte_fallback ?? isMetaspace;
  const fuseUnk = tj.model.fuse_unk ?? false;
  const unkId = vocab.get(tj.model.unk_token ?? '<unk>');
  const pickSpecial = (cands: string[], label: string): { token: string; id: number } => {
    for (const c of cands) {
      const v = added.get(c)?.id ?? vocab.get(c);
      if (v !== undefined) return { token: c, id: v };
    }
    throw new LayaEncodeError(`no id for ${cands[0]} (${label})`);
  };
  const mask = pickSpecial(['[MASK]', '<mask>'], 'mask');
  const cls = pickSpecial(['[CLS]', '<bos>'], 'cls');
  const sep = pickSpecial(['[SEP]', '<eos>'], 'sep');
  const pad = pickSpecial(['[PAD]', '<pad>'], 'pad');
  function bpe(word: string): string[] {
    let parts = [...word];
    if (parts.length <= 1) return parts;
    for (;;) {
      let best = -1;
      let bestRank = Number.POSITIVE_INFINITY;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = rank.get(`${parts[i]}\0${parts[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      parts = [
        ...parts.slice(0, best),
        (parts[best] as string) + (parts[best + 1] as string),
        ...parts.slice(best + 2),
      ];
      if (parts.length <= 1) break;
    }
    return parts;
  }
  function encodePiece(text: string): number[] {
    const ids: number[] = [];
    const re = new RegExp(GPT2_SRC, GPT2_FLAGS);
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      const bytes = te.encode(m[0]);
      let mapped = '';
      for (const b of bytes) mapped += b2u.get(b);
      for (const tok of bpe(mapped)) {
        const id = vocab.get(tok);
        if (id === undefined) {
          throw new LayaEncodeError(`missing vocab for ${JSON.stringify(tok)} in ${JSON.stringify(text.slice(0, 80))}`);
        }
        ids.push(id);
      }
    }
    return ids;
  }
  function byteFallbackIds(ch: string): number[] | null {
    const ids: number[] = [];
    for (const b of te.encode(ch)) {
      const id = vocab.get(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`);
      if (id === undefined) return null;
      ids.push(id);
    }
    return ids;
  }
  function lookupPiece(piece: string, context: string): number[] {
    const hit = vocab.get(piece);
    if (hit !== undefined) return [hit];
    if (byteFallback) {
      const out: number[] = [];
      for (const ch of piece) {
        const h = vocab.get(ch);
        if (h !== undefined) {
          out.push(h);
          continue;
        }
        const fb = byteFallbackIds(ch);
        if (fb === null) {
          if (unkId === undefined) {
            throw new LayaEncodeError(
              `missing vocab for ${JSON.stringify(ch)} in ${JSON.stringify(context.slice(0, 80))}`,
            );
          }
          out.push(unkId);
          continue;
        }
        out.push(...fb);
      }
      return out;
    }
    throw new LayaEncodeError(`missing vocab for ${JSON.stringify(piece)} in ${JSON.stringify(context.slice(0, 80))}`);
  }
  // Metaspace split: \n-runs are own words, lone ▁ stands alone.
  function metaspaceWords(text: string): string[] {
    const words: string[] = [];
    for (const part of text.split(/(\n+)/u)) {
      if (part === '') continue;
      if (/^\n+$/u.test(part)) {
        words.push(part);
        continue;
      }
      let i = 0;
      let first = true;
      while (i < part.length) {
        const ch = part[i] as string;
        if (ch === WS_SENTINEL) {
          const nxt = i + 1 < part.length ? part[i + 1] : undefined;
          if (nxt !== undefined && nxt !== WS_SENTINEL) {
            let j = i + 1;
            while (j < part.length && part[j] !== WS_SENTINEL) j++;
            words.push(part.slice(i, j));
            i = j;
          } else {
            words.push(WS_SENTINEL);
            i++;
          }
        } else {
          let j = i;
          while (j < part.length && part[j] !== WS_SENTINEL) j++;
          const chunk = part.slice(i, j);
          words.push(first && !chunk.startsWith(WS_SENTINEL) ? WS_SENTINEL + chunk : chunk);
          i = j;
        }
        first = false;
      }
    }
    return words;
  }
  function encodeMetaspace(text: string): number[] {
    const ids: number[] = [];
    let lastWasUnk = false;
    const push = (more: number[]): void => {
      for (const id of more) {
        if (fuseUnk && unkId !== undefined && id === unkId && lastWasUnk) continue;
        ids.push(id);
        lastWasUnk = unkId !== undefined && id === unkId;
      }
    };
    for (const word of metaspaceWords(text)) {
      for (const tok of bpe(word)) push(lookupPiece(tok, text));
    }
    return ids;
  }
  function normalize(text: string): string {
    if (spaceToSentinel) return text.replaceAll(' ', WS_SENTINEL);
    return text.normalize('NFC');
  }
  function encodeWith(text: string, encodeChunk: (chunk: string) => number[]): number[] {
    const ids: number[] = [];
    let i = 0;
    let buf = '';
    const flush = (): void => {
      if (buf) {
        ids.push(...encodeChunk(buf));
        buf = '';
      }
    };
    while (i < text.length) {
      let hit: string | null = null;
      for (const a of addedByLen) {
        if (text.startsWith(a, i)) {
          hit = a;
          break;
        }
      }
      if (hit !== null) {
        const spec = added.get(hit);
        if (spec === undefined) throw new LayaEncodeError(`unknown added token ${JSON.stringify(hit)}`);
        if (spec.lstrip) buf = buf.replace(/\s+$/u, '');
        flush();
        ids.push(spec.id);
        i += hit.length;
        if (spec.rstrip) {
          const mm = /^\s+/u.exec(text.slice(i));
          if (mm) i += mm[0].length;
        }
      } else {
        const cp = text.codePointAt(i);
        if (cp === undefined) throw new LayaEncodeError(`encode: bad codepoint at ${i}`);
        buf += String.fromCodePoint(cp);
        i += cp > 0xffff ? 2 : 1;
      }
    }
    flush();
    return ids;
  }
  // Metaspace splits added tokens on raw text (▁ entries never match raw
  // spaces); each chunk is normalized only after splitting.
  function encodeMetaspaceText(text: string): number[] {
    return encodeWith(text, (chunk) => encodeMetaspace(normalize(chunk)));
  }
  function encode(text: string): number[] {
    if (isMetaspace) return encodeMetaspaceText(text);
    return encodeWith(normalize(text), encodePiece);
  }
  return {
    encode,
    maskToken: mask.token,
    maskId: mask.id,
    clsId: cls.id,
    sepId: sep.id,
    padId: pad.id,
  };
}
