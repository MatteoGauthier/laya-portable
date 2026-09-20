// Fast binary loader for act_head weights (see export/emit_act_bin.py).
// Falls back to act_head.json when .bin/.meta.json are absent (dev default).
import { readFileSync } from 'node:fs';

/**
 * @param {string} binPath path to act_head.bin
 * @param {string} metaPath path to act_head.meta.json
 */
export function loadActBin(binPath, metaPath) {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const buf = readFileSync(binPath);
  const f32 = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const [w0Shape, b0Shape, w2Shape, b2Shape] = meta.shapes;
  let off = 0;
  const take = (n) => f32.subarray(off, (off += n));
  const w0flat = take(w0Shape[0] * w0Shape[1]);
  const b0 = Array.from(take(b0Shape[0]));
  const w2flat = take(w2Shape[0] * w2Shape[1]);
  const b2 = Array.from(take(b2Shape[0]));
  const w0 = [];
  for (let i = 0; i < w0Shape[0]; i++) w0.push(Array.from(w0flat.subarray(i * w0Shape[1], (i + 1) * w0Shape[1])));
  const w2 = [];
  for (let i = 0; i < w2Shape[0]; i++) w2.push(Array.from(w2flat.subarray(i * w2Shape[1], (i + 1) * w2Shape[1])));
  return { w0, b0, w2, b2 };
}
