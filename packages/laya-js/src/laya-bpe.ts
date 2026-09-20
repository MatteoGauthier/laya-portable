// Pure-JS ByteLevel BPE for Laya tokenizer.json (no bundler, no WASM).
// Matches HF tokenizers: NFC, added-token longest-match split, GPT-2 regex,
// bytes_to_unicode mapping, BPE merge by rank. Throws on missing vocab (loud).
import { LayaEncodeError } from './laya-errors.ts';
import type { Tokenizer, TokenizerJson } from './laya-types.ts';

// Fresh instance per call (module-global /g regex would race under
// interleaved worker/main use via shared lastIndex).
const GPT2_SRC = `'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+`;
const GPT2_FLAGS = 'gu';

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
  // '\0' join is collision-safe: BPE pieces are byte-level unicode chars, never contain NUL.
  const rank = new Map(tj.model.merges.map(([a, b], i) => [`${a}\0${b}`, i]));
  const added = new Map(tj.added_tokens.map((a) => [a.content, { id: a.id, lstrip: !!a.lstrip, rstrip: !!a.rstrip }]));
  const addedByLen = [...added.keys()].sort((x, y) => y.length - x.length);
  const b2u = bytesToUnicode();
  const te = new TextEncoder();
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
  function encode(text: string): number[] {
    text = text.normalize('NFC');
    const ids: number[] = [];
    let i = 0;
    let buf = '';
    const flush = (): void => {
      if (buf) {
        ids.push(...encodePiece(buf));
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
        // Iterate by codepoint so surrogate pairs stay together.
        const cp = text.codePointAt(i);
        if (cp === undefined) throw new LayaEncodeError(`encode: bad codepoint at ${i}`);
        buf += String.fromCodePoint(cp);
        i += cp > 0xffff ? 2 : 1;
      }
    }
    flush();
    return ids;
  }
  const id = (s: string): number => {
    const v = added.get(s)?.id ?? vocab.get(s);
    if (v === undefined) throw new LayaEncodeError(`no id for ${s}`);
    return v;
  };
  return {
    encode,
    maskToken: '[MASK]',
    maskId: id('[MASK]'),
    clsId: id('[CLS]'),
    sepId: id('[SEP]'),
    padId: id('[PAD]'),
  };
}
