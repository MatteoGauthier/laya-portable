# Release notes — v0.1.0 (unpublished draft)

First portable cut: validated FP32 pipeline end-to-end plus a measured FP16
option. Nothing here has been published; paths below are repo-relative.

B1/B2 update: JS `Router` (`laya-lang.ts` + `laya-router.ts`, exact port of
upstream detection/routing) plus multilingual + typed-decisions split exports
with per-checkpoint parity, BPE, FP16, and accuracy gates (see `export/README.md`).

## Status

- Phase 1 done: faithful FP32 export, variable K, 5-fixture CPU parity PASS
  (see `tools/export/README.md`, Measured parity).
- Phase 2 done: browser core, pure-JS BPE, split WebGPU, worker text to answer
  (see `packages/laya-js/README.md`).
- Phase 3 ongoing: precision and size. FP16 ships for CPU/WASM; INT8 and
  4-bit on WebGPU rejected with evidence (see `tools/export/README.md`).
- Phase 4 blocked: native pick. CoreML direct conversion blocked on toolchain
  coverage; MLX and laya.cpp untouched (see `tools/export/README.md` and
  `tools/mlx/README.md`).

## Pinned inputs

- Source: `NandhaKishorM/laya @ 6a58191` (`upstream/`, re-cloneable).
- Weights: `convaiinnovations/laya @ c5d7873` (bundle: English root +
  `multilingual/` + `typed-decisions/` subfolders; HF cache).
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

B2 checkpoints (same contract, per-checkpoint tokenizer/config/act-head):

| Artifact                                 |   Size | Status                                             |
| ---------------------------------------- | -----: | -------------------------------------------------- |
| `laya-multilingual-split-single.onnx`    | 1.2 GB | ✅ CPU (parity ≤2.4e-05, 12/13 == torch, no flips) |
| `laya-multilingual-split-fp16.onnx`      | 616 MB | ✅ CPU/WASM only (pdrift ≤1.6e-03, no flips)       |
| `laya-multilingual-split-int8.onnx`      | 0.32GB | ⛔ guard-benign flip (True→False)                  |
| `laya-multilingual-split-4bit.onnx`      | 0.86GB | ⚠️ CPU 12/13 == torch; WebGPU untested             |
| `laya-typed-decisions-split-single.onnx` | 1.6 GB | ✅ CPU (parity ≤7.3e-06, 13/13 == torch)           |
| `laya-typed-decisions-split-fp16.onnx`   | 0.85GB | ✅ CPU/WASM only (pdrift ≤3.1e-04, no flips)       |
| `laya-typed-decisions-split-int8.onnx`   | 0.42GB | ⚠️ CPU 13/13, ~14% faster on 1 probe — unshipped   |
| `laya-typed-decisions-split-4bit.onnx`   | 0.41GB | ⚠️ CPU 13/13; WebGPU untested                      |

Plus: `js/tokenizer/` (BPE + config + temperatures), `export/act_head.npz`
and `js/act_head.json` (split head weights), parity/accuracy fixtures.
Pure-JS BPE covers all three tokenizers (206/206 fuzz each, incl. the
multilingual Metaspace mode).

## Supported contract

- Inputs: `input_ids`, `attention_mask` [B,S], `marker_pos`, `marker_mask`
  [B,K], `qtype` [B]; outputs `logits` [B,K] (+ `pooled` [B,H] on split,
  H=1024 ModernBERT-large / H=768 mmBERT-base).
- Dynamic B/S/K with K≥2 (TopK floor); validated B≤3, S≤74, K≤6 in fixtures.
- Devices: ORT CPU, Node.js, browser WASM, browser WebGPU with
  `graphOptimizationLevel: 'basic'`. No Android runs yet.

## Measured reference (Apple M4 Pro, 24 GB)

- Torch MPS 1q/3q forward p50: 21.8 / 40.0 ms (`packages/test-vectors/reports/mac-baseline.json`).
- ONNX CPU 1q/3q: 34.8 / 93.9 ms. Browser split WebGPU-basic 1q: 306 ms.
- Accuracy harness (english): 13/13 torch, FP32, FP16, 4-bit-CPU; 12/13 INT8.
- Accuracy harness (multilingual): torch 12/13 (misses legit-billing phish
  negative — false-positives a polite duplicate-charge email); FP32/FP16/4-bit
  match torch decisions exactly; INT8 flips guard-benign → rejected.
- Accuracy harness (typed-decisions): 13/13 torch, FP32, FP16, INT8, 4-bit —
  INT8 unshipped (single-probe ~14% speedup only, weak-harness evidence).
- Public sets: AG News 0.935 / Emotion 0.52 / banking77 0.435, ours == torch
  (community excluded, fixed K=2; see `public-benchmark.json`).

## Known limits

- FP16/WebGPU, INT8, 4-bit/WebGPU rejected with evidence (see `export/README.md`).
  Per-checkpoint: multilingual INT8 rejected (guard-benign flip); typed INT8
  and both 4-bits are CPU-gated candidates only (WebGPU untested except
  english-4-bit, which is broken).
- Action head saturated (act≈1.0) on all probed inputs; uncertain regime unknown.
- Multilingual torch false-positives one legit-billing email as phishing
  (accuracy harness, english-language cases) — routing is by script/language,
  not by task confidence; fit temperatures per deployment.
- Larger batches and mobile unmeasured.
- CoreML direct conversion blocked on toolchain coverage (`export/to_coreml.py`).
