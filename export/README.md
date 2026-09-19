# Faithful ONNX export (FP32) + parity fixtures

First milestone from `docs/laya-portability.md`: reproducible tensor-only export,
validated on CPU. Not a browser/Android release yet.

## Reproduce

```sh
.venv/bin/python export/export_onnx.py        # -> models/laya-faithful.onnx (+ .data, 1.69 GB)
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
`TopK` was absent from the inspected ORT WebGPU table; browser provider
assignment and execution are untested. If WebGPU blocks on the action branch,
the report's split-graph proposal (GPU for logits + first-token vector, CPU
for top-two stats + small head) is the next experiment.

## Limits

5 fixtures, FP32 CPU only. No quantization, no WASM/WebGPU run, no Android,
no Core ML/MLX port. See main report for the release sequence.
