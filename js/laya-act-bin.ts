// Fast binary loader for act_head weights (see export/emit_act_bin.py).
// Falls back to act_head.json when .bin/.meta.json are absent (dev default).
import { readFileSync } from 'node:fs';
import type { ActHeadWeights } from './laya-types.ts';

interface ActHeadMeta {
  dtype: string;
  order: string[];
  shapes: number[][];
  bytes: number;
}

function dim(shape: number[] | undefined, axis: number, label: string): number {
  const v = shape?.[axis];
  if (typeof v !== 'number') throw new Error(`loadActBin: bad shape for ${label}`);
  return v;
}

export function loadActBin(binPath: string, metaPath: string): ActHeadWeights {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8')) as ActHeadMeta;
  const buf = readFileSync(binPath);
  const f32 = new Float32Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength / 4));
  const shapes = meta.shapes;
  if (shapes.length !== 4) throw new Error(`loadActBin: expected 4 shapes, got ${shapes.length}`);
  let off = 0;
  const take = (n: number): Float32Array => {
    const s = f32.subarray(off, off + n);
    off += n;
    return s;
  };
  const w0rows = dim(shapes[0], 0, 'w0');
  const w0cols = dim(shapes[0], 1, 'w0');
  const b0len = dim(shapes[1], 0, 'b0');
  const w2rows = dim(shapes[2], 0, 'w2');
  const w2cols = dim(shapes[2], 1, 'w2');
  const b2len = dim(shapes[3], 0, 'b2');
  const w0flat = take(w0rows * w0cols);
  const b0 = Array.from(take(b0len));
  const w2flat = take(w2rows * w2cols);
  const b2 = Array.from(take(b2len));
  const w0: number[][] = [];
  for (let i = 0; i < w0rows; i++) w0.push(Array.from(w0flat.subarray(i * w0cols, (i + 1) * w0cols)));
  const w2: number[][] = [];
  for (let i = 0; i < w2rows; i++) w2.push(Array.from(w2flat.subarray(i * w2cols, (i + 1) * w2cols)));
  return { w0, b0, w2, b2 };
}
