# Notices and attribution

This project is a portability study and runtime re-implementation around Laya.
It contains no Laya model weights; weights stay in their original distribution
and are loaded from the pinned public checkpoint at build time (see below).

## Upstream sources (all presented under Apache 2.0)

- Laya source: https://github.com/NandhaKishorM/laya — pinned at
  `6a5819129eb220570792e417e49723d697efd76f` for every measurement here.
  See that repository's LICENSE file for its terms.
- Laya weights: https://huggingface.co/convaiinnovations/laya — pinned at
  `c5d78730f3493e4fe16d61507ef4b78eef7318cf` (English root checkpoint).
- ModernBERT base: https://huggingface.co/answerdotai/ModernBERT-large
  (upstream architecture; this project uses Laya's fine-tuned weights, not
  the base weights).

## What this repository adds

- `export/`: reproducible ONNX export, split, and quantization converters
  plus parity/accuracy fixtures and reports.
- `js/`: dependency-free preprocessing, ByteLevel BPE, calibration, and
  action-head ports, with Node.js checks and browser probes.
- `playground/`: Vite dev UI for inference, progress, and inspection.
- `docs/`: investigation notes, runtime comparison, and Mac baseline.

## Redistribution notes

- Keep this NOTICE file and the LICENSE file with any distribution.
- Converted ONNX artifacts are derivative weight packagings; they inherit
  the upstream weights' terms — verify the current Hugging Face model card
  before redistributing weight files.
- Measured numbers in `docs/` and `export/` were taken on one Apple M4 Pro
  Mac; treat them as reference points, not guarantees.
