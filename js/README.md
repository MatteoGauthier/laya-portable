# JS runtime + browser validation

Reuses `export/fixtures/*.npz` via `js/fixtures/*.json` (emitted by
`export/emit_js_fixtures.py`).

```sh
cd js && npm install && npm test && npm run check
# ORT model checks (need models/, gitignored): npm run check:split && npm run check:node
# lint/format/types: npm run lint && npm run format:check && npm run typecheck
# CLI: node cli.mjs --help
```

Shared modules: `laya-feed.mjs` (ORT feed builder, single copy),
`laya-errors.mjs` (typed errors), `laya-act-bin.mjs` (1.1MB binary head —
`export/emit_act_bin.py`; falls back to `act_head.json`). Default model is
`models/laya-split-single.onnx` everywhere (`DEFAULT_MODEL` in `laya.mjs`).

## Node.js (onnxruntime-node 1.30.0)

- `run-node.mjs`: PASS all 5, identical diffs to Python ORT (logits ≤7.7e-06).
- `check-tokenizer.mjs`: PASS all 5, transformers.js token IDs match Python exactly.
- `check-bpe.mjs` / `check-bpe-fuzz.mjs`: PASS 5/5 + 206/206, pure-JS BPE matches Python.
- `check-split.mjs`: PASS all 5, split ORT + JS action head matches torch.

## Browser (onnxruntime-web 1.30.0, headless Chrome, Metal GPU)

```sh
python3 -m http.server 8765  # from workspace root
# open http://localhost:8765/js/web-test/index.html | full.html | split-test.html | worker-demo.html
```

Tiny probes: `add`, `topk`, `isnan`, `and`, `max` all PASS on `wasm` and
`webgpu` (likely CPU fallback for the 4 unlisted ops, not proof of placement).

Full model (`choice-2`, B=1 S=50, single-file required — external-data fails
with `MountedFiles`):

| Backend          |    Run |  max\|logit\| | Note                                                           |
| ---------------- | -----: | ------------: | -------------------------------------------------------------- |
| wasm             |  756ms | 2.38e-06 PASS | baseline                                                       |
| webgpu (default) |   FAIL |             — | `SkipLayerNormalization: Beta must be 1D` (encoder fusion bug) |
| webgpu-basic     | 1072ms | 3.81e-06 PASS | avoids fusion, slower than wasm                                |

Split model (`laya-split-single.onnx`, logits+pooled, JS action head):

| Backend        |   Run |  max\|logit\| | Note                                           |
| -------------- | ----: | ------------: | ---------------------------------------------- |
| webgpu-default |  FAIL |             — | same encoder fusion bug (split doesn't fix it) |
| webgpu-basic   | 306ms | 3.81e-06 PASS | 3.5× faster than full, 2.5× faster than wasm   |

WebGPU is hardware Metal yet full-graph is slower than WASM; split flips the
ranking. Still ~14× slower than native Torch MPS (21.8ms).

Worker (`worker-demo.html`, end-to-end text→answer, split, auto backend):
pure-JS BPE tokenization + inference + calibration off main thread, with
download % and backend selection. Verified billing 0.967 / 63 tokens, matching
`docs/mac-baseline.json` exactly; counter ticks throughout 1.6GB load.

## Tokenizer: pure-JS BPE, no bundler

`js/laya-bpe.mjs` implements ByteLevel BPE from `tokenizer.json` (NFC, added
longest-match with `[MASK]` lstrip, GPT-2 regex, byte mapping, rank merges).
`check-bpe.mjs` PASS 5/5 fixtures; `check-bpe-fuzz.mjs` PASS 206/206
(unicode, spaces, added tokens). transformers.js needs a bundler in browser
(bare `onnxruntime-*` deps; verified FAILs via bare CDN/esm.sh/Hub), so the
worker uses the dependency-free port with `tokenizer.json` + `rl_agent_config.json`
served locally.

## Limits

No Cache Storage pinning (browser HTTP cache applies; explicit pinning is
production follow-up), no Android test, one small fixture in browser.
`act_head.json` is 5.2MB JSON (prototype; use binary for production).
