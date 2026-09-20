# Faithful ONNX export (FP32) + parity fixtures

First milestone from `docs/laya-portability.md`: reproducible tensor-only export,
validated on CPU, Node.js, and browser WASM/WebGPU (see `../js/README.md`).
Not an Android release yet.

## Reproduce

```sh
.venv/bin/python export/export_onnx.py        # -> models/laya-faithful.onnx (+ .data, 1.69 GB)
.venv/bin/python export/to_single_file.py     # -> models/laya-faithful-single.onnx (1.6 GB, for browser)
.venv/bin/python export/check_parity.py       # -> export/parity-report.json + export/fixtures/*.npz
.venv/bin/python export/export_split.py       # -> models/laya-split-single.onnx (1.6 GB) + export/act_head.npz
.venv/bin/python export/to_fp16.py            # -> models/laya-split-fp16.onnx (806 MB)
.venv/bin/python export/check_fp16.py         # -> export/fp16-parity.json
.venv/bin/python export/check_accuracy.py     # -> export/accuracy-report.json (13 weak checks)
.venv/bin/python export/to_int8.py            # -> models/laya-split-int8.onnx (405 MB, rejected as-is)
.venv/bin/python export/to_4bit.py            # -> models/laya-split-4bit.onnx (395 MB, WebGPU-broken)
.venv/bin/python export/emit_act_bin.py        # -> js/act_head.bin (1.1MB f32, fast head load)
.venv/bin/python export/emit_js_fixtures.py    # -> js/fixtures/*.json (needs parity-report)
.venv/bin/python export/emit_bpe_fuzz.py       # -> js/bpe-fuzz.json
```

All converters take `--src/--dst` (`export_onnx.py` also `--opset`,
`to_4bit.py` also `--bits/--block-size`); `check_parity.py` takes
`--model/--onnx` and exits 1 on drift (CI-gatable). Checkpoint resolution uses
`snapshot_download(..., local_files_only=True)` — no hardcoded cache paths.

Offline; reuses pinned checkpoint `c5d78730f3493e4fe16d61507ef4b78eef7318cf` and
source `6a5819129eb220570792e417e49723d697efd76f`. See `export/requirements.txt`.
`models/` is gitignored.

## Contract

Inputs `input_ids`, `attention_mask` [batch, seq], `marker_pos`, `marker_mask`
[batch, options], `qtype` [batch]. Outputs `logits` [batch, options],
`act_logits` [batch, 2]. Dynamic batch/seq/options, `K>=2` (forward uses
topk(2)). Raw logits only; temperature/softmax stay outside the graph.
Fixes community gaps: variable K (not fixed 2) and batched act output
(not `[1,2]`). Metadata: `laya-faithful-metadata.json`.

## Measured parity (torch CPU vs ORT 1.30 CPU, 5 fixtures)

| Fixture | max |logit| | max prob drift | max act prob drift | labels |
|---|---:|---:|---:|:---:|
| orig-3q | 7.7e-06 | 1e-06 | 0.0 | OK |
| choice-3 | 3.8e-06 | 0.0 | 0.0 | OK |
| choice-2 | 7.4e-06 | 0.0 | 0.0 | OK |
| choice-6 | 4.9e-06 | 1e-06 | 0.0 | OK |
| mixed-batch | 6.1e-06 | 1e-06 | 0.0 | OK |

Overall PASS. Act logits differ up to 5.4e-03 absolute but only 1.4e-06
relative (magnitudes ~4000, saturated to prob 1.0). One 4-dec rounding flip
at a 1e-06 boundary (`mixed-batch` c_score2: 0.18295 vs 0.182949) shows why
full-precision drift, not rounded equality, gates parity.

Tested 8 diverse preset/state combos; all `act_probability` 1.0. Uncertain-act
regime not observed; action parity validated in saturated regime only.

## Measured latency (same 63-token inputs, 5 warmup/20 samples)

| Case           | Torch CPU | ONNX CPU | Torch MPS |
| -------------- | --------: | -------: | --------: |
| 1q forward p50 |   82.7 ms |  34.8 ms |   21.8 ms |
| 3q forward p50 |  149.3 ms |  93.9 ms |   40.0 ms |

ONNX CPU is 1.6-2.4x faster than Torch CPU, still slower than Torch MPS.
ORT CoreML EP on this graph: 209.8 ms p50 for 3q (250 partitions, 841/1840
nodes on CoreML), slower than CPU. Direct CoreML conversion remains untested.

## Operators vs browser

Graph (opset 18) includes `TopK:1`, `IsNaN:28`, `GatherND:1`, `GatherElements:1`.
Static WebGPU table omits `TopK`, `IsNaN`, `And`, `Max`, but tiny single-op
probes PASS on both WASM and WebGPU (likely CPU fallback, not proof of GPU
placement). Full-model browser results (single-file, `choice-2`): WASM 756ms
PASS (2.38e-06); WebGPU default FAILs on `SkipLayerNormalization` fusion
(`Beta must be 1D`); WebGPU with `graphOptimizationLevel:'basic'` 1072ms PASS
(3.81e-06), slower than WASM despite Metal hardware. External-data format
fails in browser (`MountedFiles`); single-file required. Details in
`../js/README.md`.

Split variant implements the report's proposal: GPU graph returns
`(logits, pooled)` without `TopK`/`Log`/`ReduceSum`/action head; 1MB
`act_head.npz` runs on CPU/JS. Python parity PASS (logits 7.39e-06, act
relative 3.9e-07). Browser WebGPU-basic: 306ms vs 1072ms full (3.5×),
beating WASM 756ms. JS action head PASS all 5 (relative ≤1.4e-06).
Default WebGPU still FAILs on encoder fusion (split doesn't fix that).

## FP16 (Phase 3, first step)

`to_fp16.py` converts split FP32→FP16 via onnxconverter-common
(`keep_io_types`, `op_block_list=['Cast']` — without the blocklist the model
fails ORT load on a Cast type mismatch). 806MB, half the download. Converter
warns it clamps -3.4e38 (attention -inf-like) to -10000; parity confirms harmless.

CPU/WASM drift vs torch FP32 (`fp16-parity.json`): logits 1.6e-03–9.8e-03,
calibrated probs ≤1.04e-03, confidence ≤1.5e-03, no label flips. Labels and
ranking preserved; 4-dec probabilities move at the 3rd decimal.

Browser (`choice-2`): WASM 762ms warm, 1.57e-03 (matches CPU, no speedup —
WASM upcasts). WebGPU-basic 180ms warm but 4.11e-02 (deterministic across
runs; likely FP16 accumulation in shaders vs FP32 on CPU). Fast but 40× less
accurate — do NOT ship FP16/WebGPU without held-out accuracy validation.
FP32 split remains the accurate WebGPU path. Playground offers both precisions
with this caveat in the selector tooltip.

Retry (2026-09-20): surgical FP32-softmax variant (`keep_softmax_fp32.py`,
30 Softmaxes wrapped — the converter blocklist path never terminates on this
graph, so the casts are hand-placed). CPU parity ≈ plain FP16, no flips;
WebGPU improves 4.11e-02 → 2.94e-02 (167ms) — softmax is ~30% of the gap,
the rest is FP16 matmul accumulation. Still 300× over the 1e-4 gate:
verdict stands.

## Accuracy harness + INT8/4-bit verdicts

`check_accuracy.py` runs 13 weak directional checks (billing intent, phishing,
guardrails, moderation, triage, original anchors) per variant with margins:

| Variant                                | Score | Note                                                                    |
| -------------------------------------- | ----: | ----------------------------------------------------------------------- |
| torch-fp32, onnx-fp32-split, onnx-fp16 | 13/13 | healthy margins (≥0.19)                                                 |
| onnx-int8 (dynamic, 405MB)             | 12/13 | phishing flips 0.845→0.340; logits drift ≤5.76; no CPU speedup (35.3ms) |
| onnx-4bit (weight-only, 395MB)         | 13/13 | CPU/WASM behaviorally OK despite ≤1.06 drift                            |

Browser 4-bit (`choice-2`): WASM 1088ms, 4.48e-02; WebGPU-basic 219ms warm but
3.35e+00 deterministic — numerically broken (dequant + FP16 accumulation or
kernel gap). Sweep (block-64 symmetric, block-32 asymmetric): CPU holds 13/13
both configs, but WebGPU gives 4.34e+00 (sym64, 277ms) and a MatMulNBits
launch failure (asym32, "cannot convert shape") — a kernel gap, not a config
issue. Rejected for WebGPU pending ORT updates.

Net Phase 3: ship FP16 for CPU/WASM (documented 1e-03 drift), FP32 split for
WebGPU. Naive INT8 and 4-bit/WebGPU are out; selective/static quantization
with calibration data is the deeper follow-up, gated by this harness.

## Public datasets (`bench_public.py` → `public-benchmark.json`)

Labelled accuracy on public HF datasets (needs network for data only;
weights stay cached). Community ONNX is interface-excluded everywhere
below — every suite needs K≥4, its graph is fixed K=2:

| Suite (n) | torch | ours fp32 | ours fp16 | Upstream published | Note |
|---|---:|---:|---:|---:|---|
| AG News 4-way (200) | 0.935 | 0.935 | 0.935 | 0.950–0.953 | bare-key prompts vs tuned; credible |
| Emotion 6-way (150) | 0.520 | 0.520 | 0.527 | 0.595–0.600 | gap was sample size: full test n=2000 scores torch 0.587, matching published; prompt variants (descriptions, reword, raw state) never moved the 150-sample result |
| banking77 77-way (154) | 0.435 | 0.435 | 0.429 | 0.425 | reproduces the option-budget ceiling |

Ours matches torch everywhere (FP16: one extra emotion hit from drift noise).
banking77 ECE 0.54 — confident at the ceiling, same theme as upstream's
Khmer 0.000-at-0.952 warning. Per-question p50: AG News torch 113ms / ours
70ms; Emotion 83/46ms; banking77 290/249ms (77-marker sequences). FP16 is
slower on CPU (no FP16 kernels) — its win is download size, not CPU speed.

## CoreML spike (Phase 4, blocked on toolchain)

`to_coreml.py` attempts split→CoreML via
`torch.export` + decompositions (coremltools 9.0, torch 2.14 untested).
TorchScript fails on a float64/int64 dtype conflict inside HF ModernBERT
mask handling. Frontier as of 2026-09-20: default decompositions (not `{}`)
clear the `new_ones` gate; dynamic batch specializes to 1 in HF internals
(`Dim.AUTO` keeps seq dynamic); MIL then rejects `gather_along_axis` with
fp32 indices from the decomposed mask path. No `.mlpackage` yet. Remaining
options: rewrite the marker-gather path with int-safe indexing, static
batch-1/seq-512 bucket, or the MLX port. MPS (21.8ms/1q) remains the Apple
GPU reference.

## Limits

5 parity fixtures + 13 weak accuracy checks; FP32/FP16 shippable per-backend,
naive INT8 and 4-bit/WebGPU rejected with evidence. No Android, no Core ML/MLX
port. Browser validated end-to-end on small fixtures (worker, split WebGPU);
larger batches unmeasured. See main report for the release sequence.
