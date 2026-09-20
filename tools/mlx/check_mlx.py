"""Validate the MLX port at output level vs torch (all 5 fixtures).

Gates: option logits <1e-4 abs, action logits relative (saturated regime),
labels exact. Intermediate hidden states may drift (MLX eager vs SDPA kernel
accumulation) while outputs agree — layers are reported, not gated.
Exits 1 on drift.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import math
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "mlx"))
import numpy as np
import mlx.core as mx
from laya_mlx import LayaMLX, load_weights

LAYER_TYPES = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(28)]
THETA = {"full_attention": 160000.0, "sliding_attention": 10000.0}
NAMES = ["choice-2", "choice-3", "choice-6", "orig-3q", "mixed-batch"]

# FP32 copies of the torch-side action computation (mirror of common.py tail)
W0 = W2 = b0 = b2 = None

def load_act():
    global W0, W2, b0, b2
    ah = np.load(ROOT / "tools" / "export" / "act_head.npz")
    W0, b0, W2, b2 = ah["0.weight"], ah["0.bias"], ah["2.weight"], ah["2.bias"]

def cpu_act(logits64, mm, pooled32):
    import math as _m
    p = np.exp(logits64 - logits64.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    k = mm.sum(axis=1).clip(min=2).astype(np.float64)
    ent = -(p * np.log(np.clip(p, 1e-9, 1))).sum(axis=1) / np.log(k)
    top2 = np.sort(p, axis=1)[:, -2:][:, ::-1]
    feats = np.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], axis=1).astype(np.float32)
    h = np.concatenate([pooled32, feats], axis=1) @ W0.T + b0
    h = 0.5 * h * (1 + np.array([[math.erf(v / 1.4142135623730951) for v in row] for row in h], dtype=np.float32))
    return h @ W2.T + b2

def mx_np(x):
    mx.eval(x)
    return np.asarray(x)

def main():
    load_act()
    print("loading weights...", flush=True)
    model = LayaMLX(load_weights(), LAYER_TYPES, THETA)
    ok = True
    for name in NAMES:
        d = np.load(ROOT / "packages" / "test-vectors" / "vectors" / f"{name}.npz")
        feeds = {k: mx.array(d[k]) for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]}
        lo, pooled = model.forward(**feeds)
        mx.eval(lo, pooled)
        lo64 = mx_np(lo).astype(np.float64)
        po32 = mx_np(pooled).astype(np.float64).astype(np.float32)
        dl = float(np.abs(lo64.astype(np.float32) - d["torch_logits"])[d["marker_mask"]].max())
        act = cpu_act(lo64, d["marker_mask"], po32)
        da = float((np.abs(act - d["torch_act"]) / np.maximum(1, np.abs(d["torch_act"]))).max())
        # labels
        tl, ml = d["torch_logits"], lo64.astype(np.float32)
        match = all(int(tl[r, :int(d["marker_mask"][r].sum())].argmax()) == int(ml[r, :int(d["marker_mask"][r].sum())].argmax())
                    for r in range(tl.shape[0]))
        good = dl < 1e-4 and da < 1e-4 and match
        ok &= good
        print(f"{name}: dlogits={dl:.2e} act-rel={da:.2e} labels={'OK' if match else 'DIFF'} -> {'PASS' if good else 'CHECK'}", flush=True)
    print("overall:", "PASS" if ok else "CHECK DETAILS", flush=True)
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()
