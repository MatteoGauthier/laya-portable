# MLX port (Phase 4 spike)

Native Apple-GPU Laya in `laya_mlx.py` — no torch at inference, no converter
lottery. Mirrors `upstream/laya/laya/common.py` + HF ModernBERT-large exactly:
fused Wqkv split, GeGLU MLP, per-type RoPE (160000/10000), alternating
full/sliding attention, norm-first ReLU head layers, biased scorer LayerNorm.

## Mask rules (pinned empirically against torch factories)

- Full layers: additive mask only when padding exists, else dense.
- Sliding layers: `|i-j|<=64` window ∧ padding, except unpadded S<64 runs dense
  (torch skips the mask when the window would cover everything... verified
  S<=63 → None, S>=64 → MASK; the skip threshold is S<64, not S<=129).
- Masked positions get `-inf` (SDPA bool→additive convention).

## Validate

```sh
.venv/bin/python tools/mlx/capture_ref.py  # torch intermediates → tools/mlx/ref.npz
.venv/bin/python tools/mlx/check_mlx.py    # output parity, exits 1 on drift
```

## Measured (M4 Pro, choice-3 B=1 S=63)

| Variant | Logits vs torch | Action rel. | Forward p50 |
|---|---:|---:|---:|
| MLX FP32 | ≤1.5e-05 (5/5) | ≤1.3e-06 | **19.3ms** |
| Torch MPS | — | — | 21.8ms (baseline doc) |

MLX beats MPS torch unoptimized on first attempt. Intermediate hidden states
drift up to ~0.3 deep in the stack (eager-vs-SDPA accumulation) while outputs
agree — layers are reported, not gated. `ref.npz` + `*.mlpackage` absent:
gitignored, regenerable.
