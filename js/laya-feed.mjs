// Single source of truth for ORT feed construction.
// Shared by laya.mjs (Node SDK), playground worker, and check scripts.
// Replaces the 4 previously duplicated toI64/toB8 inline copies.

/**
 * @param {unknown} nested nested number array
 * @returns {BigInt64Array}
 */
export function toI64(nested) {
  const flat = Array.isArray(nested) ? nested.flat(Infinity) : [nested];
  const out = new BigInt64Array(flat.length);
  for (let i = 0; i < flat.length; i++) out[i] = BigInt(flat[i]);
  return out;
}

/**
 * @param {unknown} nested nested truthy array
 * @returns {Uint8Array}
 */
export function toB8(nested) {
  const flat = Array.isArray(nested) ? nested.flat(Infinity) : [nested];
  const out = new Uint8Array(flat.length);
  for (let i = 0; i < flat.length; i++) out[i] = nested && flat[i] ? 1 : 0;
  return out;
}

/**
 * Build ORT feeds from a collated batch.
 * @param {object} ort onnxruntime module (node or web) exposing Tensor
 * @param {{inputIds:number[][],attentionMask:number[][],markerPos:number[][],markerMask:boolean[][],qtype:number[]}} b
 * @returns {Record<string, object>}
 */
export function buildFeeds(ort, b) {
  if (!b || !b.inputIds?.length) throw new Error('buildFeeds: empty batch (collateItems returned null?)');
  const B = b.inputIds.length;
  const S = b.inputIds[0].length;
  const K = b.markerPos[0].length;
  if (K < 1) throw new Error('buildFeeds: K>=1 required (markers truncated away?)');
  return {
    input_ids: new ort.Tensor('int64', toI64(b.inputIds), [B, S]),
    attention_mask: new ort.Tensor('int64', toI64(b.attentionMask), [B, S]),
    marker_pos: new ort.Tensor('int64', toI64(b.markerPos), [B, K]),
    marker_mask: new ort.Tensor('bool', toB8(b.markerMask), [B, K]),
    qtype: new ort.Tensor('int64', toI64(b.qtype), [B]),
  };
}

/**
 * Split flat ORT outputs into per-row arrays without extra copies beyond slice.
 * @param {ArrayLike<number>} logitsData flat [B*K]
 * @param {ArrayLike<number>} pooledData flat [B*1024]
 * @param {number} B batch, @param {number} K kmax
 */
export function splitOutputs(logitsData, pooledData, B, K, hiddenDim = 1024) {
  const logits = [];
  const pooled = [];
  const ld = Array.isArray(logitsData) ? logitsData : Array.from(logitsData);
  const pd = Array.isArray(pooledData) ? pooledData : Array.from(pooledData);
  if (ld.length !== B * K) throw new Error(`splitOutputs: logits len ${ld.length} != B*K ${B}*${K}`);
  if (pd.length !== B * hiddenDim) throw new Error(`splitOutputs: pooled len ${pd.length} != B*${hiddenDim}`);
  for (let i = 0; i < B; i++) {
    logits.push(ld.slice(i * K, (i + 1) * K));
    pooled.push(pd.slice(i * hiddenDim, (i + 1) * hiddenDim));
  }
  return { logits, pooled };
}
