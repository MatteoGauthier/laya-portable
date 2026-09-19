# JS runtime + browser validation

Reuses `export/fixtures/*.npz` via `js/fixtures/*.json` (emitted by
`export/emit_js_fixtures.py`). No tokenizer yet; inputs are pre-tokenized.

## Node.js (onnxruntime-node 1.30.0)

```sh
cd js && npm install && node run-node.mjs
```

Result: PASS all 5 fixtures, identical diffs to Python ORT
(logits 3.8–7.7e-06, pdrift ≤1.1e-06). Validates graph + JS
temperature/softmax replication.

## Browser probes (onnxruntime-web 1.30.0, headless Chrome, Metal GPU)

```sh
cd js/web-test && python3 -m http.server 8765
# open http://localhost:8765/index.html (tiny ops) or full.html (1.69GB)
```

Tiny probes (`export/build_web_probes.py`, IR 10): `add`, `topk`, `isnan`,
`and`, `max` all PASS on both `wasm` and `webgpu`. Static table lists 4 as
missing, so WebGPU likely falls back to CPU for those single-op graphs
rather than failing. Execution alone does not prove GPU placement.

Full model (`choice-2`, B=1 S=50):

| Backend | Load | Run | max\|logit\| vs torch | Note |
|---|---:|---:|---:|---|
| wasm | 2.8s | 756ms | 2.38e-06 PASS | external-data format fails; single-file required |
| webgpu (default) | 1.9s | FAIL | — | `SkipLayerNormalization: Beta must be 1D` (fusion bug) |
| webgpu-basic | 4.0s | 1322ms | 3.81e-06 PASS | `graphOptimizationLevel:'basic'` avoids fusion, slower than wasm |

External-data `.onnx` + `.data` fails in browser with
`Module.MountedFiles is not available`. Use single-file
`models/laya-faithful-single.onnx` (1.6GB, via `export/to_single_file.py`);
numerics identical to split format (7.39e-06 both).

WebGPU is hardware Metal (`vendor:apple arch:metal-3`) yet slower than WASM
for B=1 (1322 vs 756ms), and 35–60× slower than native Torch MPS (21.8ms).
Browser inference is feasible and correct, but not fast. Partitioning from
CPU-fallback ops + small-batch overhead are the likely causes.

## Limits

Pre-tokenized fixtures only; no JS tokenizer, no worker/caching/progress UI,
no Android test. Full-model test uses one small fixture; larger batches
unmeasured in browser.
