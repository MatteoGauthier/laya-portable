# Release notes — v0.1.0 (unpublished draft)

First portable cut: validated FP32 pipeline end-to-end plus a measured FP16
option. Nothing here has been published; paths below are repo-relative.

## Pinned inputs

- Source: `NandhaKishorM/laya @ 6a58191` (`upstream/`, re-cloneable).
- Weights: `convaiinnovations/laya @ c5d7873` (English root; HF cache).
- Runtimes: torch 2.14 / ORT 1.30 / Node 25 / Chrome 148 + Metal (see
  `export/requirements.txt` and `js/package-lock.json`).

## Artifacts and checksums

`models/` is local-only; verify builds against `export/checksums.sha256`:

| Artifact                        |   Size | Status                                           |
| ------------------------------- | -----: | ------------------------------------------------ |
| `laya-split-single.onnx` (FP32) | 1.6 GB | ✅ primary: CPU/WASM/WebGPU-basic                |
| `laya-split-fp16.onnx`          | 806 MB | ✅ CPU/WASM (≤1e-03 drift); ⛔ WebGPU (4.11e-02) |
| `laya-split-int8.onnx`          | 405 MB | ⛔ phishing flip, no speedup                     |
| `laya-split-4bit.onnx`          | 395 MB | ⛔ WebGPU 3.35 error (CPU holds 13/13)           |
| `laya-faithful-single.onnx`     | 1.6 GB | ✅ full-graph reference (pre-split)              |

Plus: `js/tokenizer/` (BPE + config + temperatures), `export/act_head.npz`
and `js/act_head.json` (split head weights), parity/accuracy fixtures.

## Supported contract

- Inputs: `input_ids`, `attention_mask` [B,S], `marker_pos`, `marker_mask`
  [B,K], `qtype` [B]; outputs `logits` [B,K] (+ `pooled` [B,1024] on split).
- Dynamic B/S/K with K≥2 (TopK floor); validated B≤3, S≤74, K≤6 in fixtures.
- Devices: ORT CPU, Node.js, browser WASM, browser WebGPU with
  `graphOptimizationLevel: 'basic'`. No Android runs yet.

## Measured reference (Apple M4 Pro, 24 GB)

- Torch MPS 1q/3q forward p50: 21.8 / 40.0 ms (`packages/test-vectors/reports/mac-baseline.json`).
- ONNX CPU 1q/3q: 34.8 / 93.9 ms. Browser split WebGPU-basic 1q: 306 ms.
- Accuracy harness: 13/13 torch, FP32, FP16, 4-bit-CPU; 12/13 INT8.
- Public sets: AG News 0.935 / Emotion 0.52 / banking77 0.435, ours == torch
  (community excluded, fixed K=2; see `public-benchmark.json`).

## Known limits

- FP16/WebGPU, INT8, 4-bit/WebGPU rejected with evidence (see `export/README.md`).
- Action head saturated (act≈1.0) on all probed inputs; uncertain regime unknown.
- English root checkpoint only; larger batches and mobile unmeasured.
- CoreML direct conversion blocked on toolchain coverage (`export/to_coreml.py`).
