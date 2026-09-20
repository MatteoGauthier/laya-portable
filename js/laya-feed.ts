// Single source of truth for ORT feed data.
// Shared by laya.ts (Node SDK), playground worker, and check scripts.
// Returns plain typed arrays + dims; call sites construct `ort.Tensor`
// with their own (fully typed) ORT import — no cross-runtime Tensor types here.

import type { CollatedBatch } from './laya-types.ts';

export function toI64(nested: unknown): BigInt64Array {
  const flat: unknown[] = Array.isArray(nested) ? nested.flat(Number.POSITIVE_INFINITY) : [nested];
  const out = new BigInt64Array(flat.length);
  for (let i = 0; i < flat.length; i++) {
    const v = flat[i];
    if (typeof v !== 'number' && typeof v !== 'bigint' && typeof v !== 'boolean') {
      throw new Error(`toI64: non-numeric feed value ${String(v)}`);
    }
    out[i] = BigInt(v as number);
  }
  return out;
}

export function toB8(nested: unknown): Uint8Array {
  const flat: unknown[] = Array.isArray(nested) ? nested.flat(Number.POSITIVE_INFINITY) : [nested];
  const out = new Uint8Array(flat.length);
  for (let i = 0; i < flat.length; i++) out[i] = flat[i] ? 1 : 0;
  return out;
}

export interface FeedDims {
  B: number;
  S: number;
  K: number;
}

export interface FeedData {
  inputIds: BigInt64Array;
  attentionMask: BigInt64Array;
  markerPos: BigInt64Array;
  markerMask: Uint8Array;
  qtype: BigInt64Array;
  dims: FeedDims;
}

/** Flatten a collated batch into ORT-ready buffers. Throws on empty/truncated batches. */
export function toFeedData(b: CollatedBatch): FeedData {
  if (!b?.inputIds?.length) throw new Error('toFeedData: empty batch (collateItems threw?)');
  const firstRow = b.inputIds[0];
  const firstMarkers = b.markerPos[0];
  if (firstRow === undefined || firstMarkers === undefined) throw new Error('toFeedData: malformed batch');
  const dims: FeedDims = { B: b.inputIds.length, S: firstRow.length, K: firstMarkers.length };
  if (dims.K < 1) throw new Error('toFeedData: K>=1 required (markers truncated away?)');
  return {
    inputIds: toI64(b.inputIds),
    attentionMask: toI64(b.attentionMask),
    markerPos: toI64(b.markerPos),
    markerMask: toB8(b.markerMask),
    qtype: toI64(b.qtype),
    dims,
  };
}

/** Split flat ORT outputs into per-row arrays. Throws on shape mismatch. */
export function splitOutputs(
  logitsData: ReadonlyArray<number>,
  pooledData: ReadonlyArray<number>,
  B: number,
  K: number,
  hiddenDim = 1024,
): { logits: number[][]; pooled: number[][] } {
  if (logitsData.length !== B * K) throw new Error(`splitOutputs: logits len ${logitsData.length} != B*K ${B}*${K}`);
  if (pooledData.length !== B * hiddenDim) {
    throw new Error(`splitOutputs: pooled len ${pooledData.length} != B*${hiddenDim}`);
  }
  const logits: number[][] = [];
  const pooled: number[][] = [];
  for (let i = 0; i < B; i++) {
    logits.push(logitsData.slice(i * K, (i + 1) * K));
    pooled.push(pooledData.slice(i * hiddenDim, (i + 1) * hiddenDim));
  }
  return { logits, pooled };
}
