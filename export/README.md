# Faithful ONNX export (FP32) + parity fixtures

First milestone from `docs/laya-portability.md`: reproducible tensor-only export,
validated on CPU, Node.js, and browser WASM/WebGPU (see `../js/README.md`).
Not an Android release yet.

## Reproduce

```sh
.venv/bin/python export/export_onnx.py        # -> models/laya-faithful.onnx (+ .data, 1.69 GB)
.venv/bin/python export/to_single_file.py     # -> models/laya-faithful-single.onnx (1.6 GB, for browser)
.venv/bin/python export/check_parity.py       # -> export/parity-report.json + export/fixtures/*.npz
```

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

| Case | Torch CPU | ONNX CPU | Torch MPS |
|---|---:|---:|---:|
| 1q forward p50 | 82.7 ms | 34.8 ms | 21.8 ms |
| 3q forward p50 | 149.3 ms | 93.9 ms | 40.0 ms |

ONNX CPU is 1.6-2.4x faster than Torch CPU, still slower than Torch MPS.
ORT CoreML EP on this graph: 209.8 ms p50 for 3q (250 partitions, 841/1840
nodes on CoreML), slower than CPU. Direct CoreML conversion remains untested.

## Operators vs browser

Graph (opset 18) includes `TopK:1`, `IsNaN:28`, `GatherND:1`, `GatherElements:1`.
Static WebGPU table omits `TopK`, `IsNaN`, `And`, `Max`, but tiny single-op
probes PASS on both WASM and WebGPU (likely CPU fallback, not proof of GPU
placement). Full-model browser results (single-file, `choice-2`): WASM 756ms
PASS (2.38e-06); WebGPU default FAILs on `SkipLayerNormalization` fusion
(`Beta must be 1D`); WebGPU with `graphOptimizationLevel:'basic'` 1322ms PASS
(3.81e-06), slower than WASM despite Metal hardware. External-data format
fails in browser (`MountedFiles`); single-file required. Details in
`../js/README.md`. If WebGPU partitioning remains costly, the report's
split-graph proposal (GPU for logits + first-token vector, CPU for top-two
stats + small head) is still the next optimization experiment.

## Limits

5 fixtures, FP32 only. No quantization, no Android, no Core ML/MLX port.
Browser validated for correctness on one small fixture; larger batches and
worker/caching UX unmeasured. See main report for the release sequence.
