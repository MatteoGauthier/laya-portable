# Laya portable

Run Laya decisions outside Python: Node.js, browsers (WASM/WebGPU in a worker),
and anywhere ONNX runs. FP32 by default, with a measured FP16 option. Same
tokenizer, same calibration, same answers.

## Quickstart

```sh
cd apps/playground && npm install && npm run dev
```

```sh
cd packages/laya-js && npm install && node bin/cli.ts
```

```js
import { LayaClient } from '@laya/js';
const laya = await LayaClient.open();
console.log(await laya.predict(state, questions));
```

Models are gitignored (1.6GB FP32, 806MB FP16 for english). Build once with
`tools/export`, verify with `checksums.sha256`. Weights come from the pinned
public checkpoint, see NOTICE.md. Multilingual and typed-decisions add two more
checkpoints, see `docs/usage.md`.

## Status

Measured, not estimated:

- FP32 ONNX matches torch: max logit drift 7.7e-06, 5/5 fixtures, 13/13 checks.
- FP16 split is half the download: max calibrated drift 1.04e-03 on CPU/WASM, 13/13 checks.
- Browser: WASM 756ms PASS, WebGPU-basic 306ms PASS. FP16 and 4-bit on WebGPU
  rejected on accuracy, with evidence.
- Pure-JS BPE matches Python token for token: 5/5 fixtures, 206/206 fuzz.

Details: `tools/export/README.md`, `packages/laya-js/README.md`, `RELEASE_NOTES.md`.

## Layout

- `apps/playground`: browser dev UI for inference and inspection.
- `packages/laya-js`: TypeScript runtime (BPE, preprocessing, calibration, action head).
- `packages/test-vectors`: shared fixtures and reports, generated.
- `tools/export`: converters and parity checks.
- `models`: generated ONNX artifacts, local only.
- `docs`: usage, performance, and portability notes.
- `upstream/laya`: pinned source checkout, gitignored.

## Reproducing (contributors)

`docs/usage.md` covers the entry points, `docs/performance.md` covers timings.
Full export setup (clone, weights, parity) is in `tools/export/README.md`.

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements-benchmark.txt
.venv/bin/python benchmark.py
.venv/bin/python tools/export/check_parity.py
```

```sh
python -m pytest tests/ -q
cd packages/laya-js && npm install && npm test && npm run check
cd apps/playground && npm install && npm test
```

CI runs JS tests and report assertions on every push. Full export rebuilds only
run on `[export]` commits.

## License

Apache 2.0, see LICENSE. Upstream code and weights keep their own Apache-2.0
terms, see NOTICE.md before redistributing weight files.
