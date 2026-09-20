// Browser-safe parser for act_head.bin (see tools/export/emit_act_bin.py).
// Same layout as the Node loader (laya-act-bin.ts) but over a fetched
// Uint8Array instead of node:fs — no Node imports, safe to bundle in workers.
import type { ActHeadWeights } from './laya-types.ts';

export interface ActHeadMeta {
  dtype: string;
  order: string[];
  shapes: number[][];
  bytes: number;
}

export function parseActHeadBin(buf: Uint8Array, meta: ActHeadMeta): ActHeadWeights {
  if (meta.dtype !== 'float32') throw new Error(`parseActHeadBin: dtype ${meta.dtype}`);
  if (buf.byteLength < meta.bytes) throw new Error(`parseActHeadBin: short buffer ${buf.byteLength}`);
  const f32 = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(meta.bytes / 4));
  const [s0, s1, s2, s3] = meta.shapes;
  if (!s0 || !s1 || !s2 || !s3 || s0.length !== 2 || s2.length !== 2) {
    throw new Error('parseActHeadBin: bad shapes');
  }
  let off = 0;
  const take = (n: number): Float32Array => {
    const s = f32.subarray(off, off + n);
    off += n;
    return s;
  };
  const w0flat = take(s0[0]! * s0[1]!);
  const b0 = Array.from(take(s1[0]!));
  const w2flat = take(s2[0]! * s2[1]!);
  const b2 = Array.from(take(s3[0]!));
  const w0: number[][] = [];
  for (let i = 0; i < s0[0]!; i++) w0.push(Array.from(w0flat.subarray(i * s0[1]!, (i + 1) * s0[1]!)));
  const w2: number[][] = [];
  for (let i = 0; i < s2[0]!; i++) w2.push(Array.from(w2flat.subarray(i * s2[1]!, (i + 1) * s2[1]!)));
  return { w0, b0, w2, b2 };
}
