# Laya, portable

Run [Laya](https://huggingface.co/convaiinnovations/laya) decisions outside
Python: native Node.js, browsers (WASM/WebGPU in a worker), and anywhere ONNX
runs. Ships a faithful FP32 pipeline plus a measured FP16 option — same
tokenizer, same calibration, same answers.

## Try it

New here? Start with [`docs/usage.md`](docs/usage.md) (all four entry points,
troubleshooting, porting notes). Making it fast? [`docs/performance.md`](docs/performance.md)
(timings, knob matrix, batching checklist).

```sh
# Interactive dev UI (inference, progress tables, tokenizer inspector)
cd apps/playground && npm install && npm run dev
```

```sh
# CLI (Node.js, split FP32 model; add --fp16 for the 806MB variant)
cd packages/laya-js && npm install && node bin/cli.ts
```

```js
// SDK
import { LayaClient } from '@laya/js';
const laya = await LayaClient.open(); // { model, tokenizer, providers }
console.log(await laya.predict(state, questions));
```

Models are gitignored (1.6GB FP32 / 806MB FP16). Build them with
`tools/export/export_onnx.py` → `to_single_file.py` → `export_split.py`
(`to_fp16.py` for FP16); verify with `tools/export/checksums.sha256`.
Weights come from the pinned public checkpoint; see NOTICE.md.

## Layout

- `apps/playground/` — Vite dev UI for inference, progress, and inspection.
- `packages/laya-js/` — strict-TS runtime: ByteLevel BPE, preprocessing,
  calibration, action head, Node checks, browser probes.
- `packages/test-vectors/` — shared fixtures + parity/accuracy reports (generated).
- `tools/export/` — reproducible converters and emitters.
- `models/` — generated ONNX artifacts (local only, checksummed).
- `docs/` — portability investigation, runtime comparison, Mac baseline.
- `upstream/laya/` — pinned source checkout (gitignored, re-cloneable).

## Validation snapshot

- FP32 ONNX vs torch: logits ≤7.7e-06, 5/5 fixtures, 13/13 accuracy checks.
- FP16 split: 806MB, calibrated drift ≤1.04e-03 on CPU/WASM, 13/13 accuracy.
- Browser: WASM 756ms PASS; split WebGPU-basic 306ms PASS (default graph
  needs `'basic'` to dodge a fusion bug); FP16/4-bit rejected on WebGPU
  accuracy with measured evidence.
- Pure-JS BPE matches Python token-for-token (5/5 fixtures, 206/206 fuzz).

Details: [export notes](tools/export/README.md), [JS notes](packages/laya-js/README.md),
[release notes](RELEASE_NOTES.md), [investigation](docs/laya-portability.md).

## Reproducing measurements (contributors)

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements-benchmark.txt
git clone https://github.com/NandhaKishorM/laya upstream/laya && git -C upstream/laya checkout 6a5819129eb220570792e417e49723d697efd76f
hf download convaiinnovations/laya --revision c5d78730f3493e4fe16d61507ef4b78eef7318cf --include 'model.safetensors' 'rl_agent_config.json' 'encoder/*' 'tokenizer/*'
.venv/bin/python benchmark.py            # Mac CPU/MPS baseline
.venv/bin/python tools/export/check_parity.py  # ONNX parity fixtures (exits 1 on drift)
.venv/bin/python tools/export/check_accuracy.py
```

```sh
uv pip install pytest && python -m pytest tests/ -q  # offline report assertions (no models)
cd packages/laya-js && npm install && npm test && npm run check # JS unit + BPE/fuzz checks
cd apps/playground && npm install && npm test          # UI tests
```

CI (`.github/workflows/ci.yml`): JS tests + report assertions on every push;
heavy export rebuild only on `[export]` commits. `pyproject.toml` is the
packaging source of truth; `Dockerfile` gives a CPU-only repro container.

## License

Apache 2.0 (see LICENSE). Upstream code and weights keep their own
Apache-2.0 terms — see NOTICE.md before redistributing weight files.
