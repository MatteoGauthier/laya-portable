"""Laya DecisionModel ported to MLX (Apple GPU, no Python-torch at inference).

Faithful to upstream/laya/laya/common.py + HF ModernBERT-large structure:
- ModernBERT encoder x28 (fused Wqkv, GeGLU MLP, per-type RoPE, alternating
  full/sliding attention, exact mask skip rules)
- type_emb + 2 standard TransformerEncoderLayers (norm-first, ReLU)
- scorer (LayerNorm w/ bias, GELU MLPs) + masked_fill(-1e4)
- pooled first-token output (action head itself stays numpy/JS-side, as before)

All FP32. Validate with check_mlx.py (layer-by-layer vs torch ref.npz).
"""
import math
import mlx.core as mx
D = 1024
NHEAD = 16
HD = 64
INTER = 2624
EPS = 1e-5


def layernorm(x, w, b=None):
    m = mx.mean(x, axis=-1, keepdims=True)
    v = mx.mean(mx.square(x - m), axis=-1, keepdims=True)
    y = (x - m) / mx.sqrt(v + EPS) * w
    return y + b if b is not None else y


def gelu_exact(x):
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def rope_cos_sin(s, theta):
    dim = HD
    inv = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    freqs = mx.outer(mx.arange(s, dtype=mx.float32), inv)  # (S,32)
    emb = mx.concatenate([freqs, freqs], axis=-1)  # (S,64)
    return mx.cos(emb), mx.sin(emb)  # (S,64) each


def rotate_half(x):
    x1 = x[..., : HD // 2]
    x2 = x[..., HD // 2:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B,H,S,64); cos/sin: (S,64) broadcast over B,H
    qf = q.astype(mx.float32)
    kf = k.astype(mx.float32)
    qe = qf * cos + rotate_half(qf) * sin
    ke = kf * cos + rotate_half(kf) * sin
    return qe.astype(q.dtype), ke.astype(k.dtype)


def sdpa_eager(q, k, v, add_mask):
    # q,k,v: (B,H,S,64); add_mask: (B,1,S,S) fp32 or None
    s = (q.astype(mx.float32) @ mx.transpose(k.astype(mx.float32), (0, 1, 3, 2))) * (HD ** -0.5)
    if add_mask is not None:
        s = s + add_mask
    p = mx.softmax(s, axis=-1, precise=True)
    o = p.astype(v.dtype) @ v
    return mx.transpose(o, (0, 2, 1, 3)).reshape(o.shape[0], -1, D)


def build_masks(att, layer_types):
    """att: (B,S) int. Returns dict layer_idx -> (B,1,S,S) fp32 additive or None.

    Replicates HF mask factories exactly: full layers mask only when padding
    exists; sliding layers mask when padding exists or S>=64 (|i-j|<=64).
    Masked positions get -inf (torch SDPA bool->additive convention).
    """
    B, S = att.shape
    masks = {}
    has_pad = bool(mx.any(att == 0).item())
    pos = mx.arange(S)
    win = (mx.abs(pos[:, None] - pos[None, :]) <= 64)  # (S,S) bool
    kv_ok = (att == 1)[:, None, None, :]  # (B,1,1,S)
    neginf = mx.array(float("-inf"), dtype=mx.float32)
    for i, lt in enumerate(layer_types):
        if lt == "full_attention":
            if not has_pad:
                masks[i] = None
            else:
                masks[i] = mx.where(kv_ok, mx.zeros((B, 1, S, S), mx.float32),
                                    mx.broadcast_to(neginf, (B, 1, S, S)))
        else:
            if not has_pad and S < 64:
                masks[i] = None
            else:
                keep = kv_ok & win[None, None, :, :]  # (B,1,S,S)
                masks[i] = mx.where(keep, mx.zeros((B, 1, S, S), mx.float32),
                                    mx.broadcast_to(neginf, (B, 1, S, S)))
    return masks


class LayaMLX:
    def __init__(self, weights, layer_types, rope_theta):
        self.w = weights
        self.layer_types = layer_types
        self.rope_theta = rope_theta  # {"full_attention": t, "sliding_attention": t}

    def encoder_layer(self, h, i):
        p = f"encoder.layers.{i}."
        W = self.w
        lt = self.layer_types[i]
        # attention
        an = W.get(p + "attn_norm.weight", None)
        a = layernorm(h, an) if an is not None else h
        qkv = a @ mx.transpose(W[p + "attn.Wqkv.weight"])
        qkv = qkv.reshape(qkv.shape[0], -1, 3, NHEAD, HD)
        q = mx.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        k = mx.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        v = mx.transpose(qkv[:, :, 2], (0, 2, 1, 3))
        cos, sin = rope_cos_sin(q.shape[2], self.rope_theta[lt])
        q, k = apply_rope(q, k, cos, sin)
        o = sdpa_eager(q, k, v, self._masks[i])
        o = o @ mx.transpose(W[p + "attn.Wo.weight"])
        h = h + o
        # mlp (GeGLU)
        m = layernorm(h, W[p + "mlp_norm.weight"])
        wi = m @ mx.transpose(W[p + "mlp.Wi.weight"])  # (B,S,5248)
        x1, gate = wi[..., :INTER], wi[..., INTER:]
        h = h + ((gelu_exact(x1) * gate) @ mx.transpose(W[p + "mlp.Wo.weight"]))
        return h

    def tel_layer(self, h, prefix, pad_add):
        # standard norm-first TransformerEncoderLayer, ReLU, d//64 heads
        W = self.w
        nh = D // 64
        dh = 64
        a = layernorm(h, W[prefix + "norm1.weight"], W.get(prefix + "norm1.bias"))
        qkv = a @ mx.transpose(W[prefix + "self_attn.in_proj_weight"]) + W[prefix + "self_attn.in_proj_bias"]
        qkv = qkv.reshape(qkv.shape[0], -1, 3, nh, dh)
        q = mx.transpose(qkv[:, :, 0], (0, 2, 1, 3))
        k = mx.transpose(qkv[:, :, 1], (0, 2, 1, 3))
        v = mx.transpose(qkv[:, :, 2], (0, 2, 1, 3))
        s = (q @ mx.transpose(k, (0, 1, 3, 2))) * (dh ** -0.5)
        if pad_add is not None:
            s = s + pad_add
        p = mx.softmax(s, axis=-1, precise=True)
        o = mx.transpose(p @ v, (0, 2, 1, 3)).reshape(h.shape[0], -1, D)
        o = o @ mx.transpose(W[prefix + "self_attn.out_proj.weight"]) + W[prefix + "self_attn.out_proj.bias"]
        h = h + o
        m = layernorm(h, W[prefix + "norm2.weight"], W.get(prefix + "norm2.bias"))
        f = m @ mx.transpose(W[prefix + "linear1.weight"]) + W[prefix + "linear1.bias"]
        f = mx.maximum(f, 0)
        f = f @ mx.transpose(W[prefix + "linear2.weight"]) + W[prefix + "linear2.bias"]
        return h + f

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        W = self.w
        h = layernorm(W["encoder.embeddings.tok_embeddings.weight"][input_ids],
                      W["encoder.embeddings.norm.weight"])
        self._masks = build_masks(attention_mask, self.layer_types)
        for i in range(len(self.layer_types)):
            h = self.encoder_layer(h, i)
        h = layernorm(h, W["encoder.final_norm.weight"])
        h = h + W["type_emb.weight"][qtype][:, None, :]
        pad = (attention_mask == 0)
        pad_add = None
        if bool(mx.any(pad).item()):
            neginf = mx.array(float("-inf"), dtype=mx.float32)
            B, S = attention_mask.shape
            pad_add = mx.where(pad[:, None, None, :],
                               mx.broadcast_to(neginf, (B, 1, S, S)),
                               mx.zeros((B, 1, S, S), mx.float32))
        for j in range(2):
            h = self.tel_layer(h, f"head.layers.{j}.", pad_add)
        # scorer gather
        idx = mx.clip(marker_pos, 0, 2**31 - 1)[..., None]
        idx = mx.broadcast_to(idx, (idx.shape[0], idx.shape[1], D))
        m = mx.take_along_axis(h, idx, axis=1)
        s = layernorm(m, W["scorer.0.weight"], W["scorer.0.bias"])
        s = s @ mx.transpose(W["scorer.1.weight"]) + W["scorer.1.bias"]
        s = gelu_exact(s)
        s = mx.squeeze(s @ mx.transpose(W["scorer.3.weight"]) + W["scorer.3.bias"], axis=-1)
        neg = mx.array(-1e4, dtype=mx.float32)
        logits = mx.where(marker_mask, s, mx.broadcast_to(neg, s.shape))
        return logits, h[:, 0, :]


def load_weights():
    from huggingface_hub import snapshot_download
    w = mx.load(snapshot_download("convaiinnovations/laya",
               revision="c5d78730f3493e4fe16d61507ef4b78eef7318cf",
               local_files_only=True) + "/model.safetensors")
    # drop temperature buffer (calibration lives outside, as before); fp16->fp32
    return {k: (v.astype(mx.float32) if v.dtype != mx.bool_ else v)
            for k, v in w.items() if k != "temperature"}
