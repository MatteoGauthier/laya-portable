# Using the model

Four entry points, same answers. Pick by platform:

| You want…                        | Use                            | Command                                                |
| -------------------------------- | ------------------------------ | ------------------------------------------------------ |
| Try it visually                  | Playground (browser)           | `cd apps/playground && npm install && npm run dev`     |
| Use it from Node.js              | `@laya/js` SDK                 | `node bin/cli.ts` or `LayaClient` in code              |
| Use it from a shell pipeline     | CLI                            | `node packages/laya-js/bin/cli.ts --state-file s.json` |
| Use it from Python without torch | `tools/export/predict_onnx.py` | `.venv/bin/python tools/export/predict_onnx.py`        |
| Train / reference behavior       | Upstream `laya` (torch)        | `import laya` (see contributor setup in README)        |

All paths need the model files in `models/` (gitignored, built once via
`tools/export/`; verify with `tools/export/checksums.sha256`). The SDK, CLI,
and Python script all default to `models/laya-split-single.onnx` (1.6GB FP32);
pass `--model` / `{ model }` / `--fp16` for the 806MB FP16 variant.

## Node.js SDK

Requires Node ≥22 (sources are strict TypeScript, run via native
type-stripping — no build step).

```js
import { LayaClient } from '@laya/js'; // or '../packages/laya-js/src/laya.ts' from a checkout

const laya = await LayaClient.open(); // { model?, tokenizer?, providers? }, defaults to split FP32 on CPU
try {
  const result = await laya.predict(state, questions);
  console.log(result.answers.department.choice); // 'billing'
  console.log(result.timings); // { tokenize_ms, inference_ms, postprocess_ms, total_ms }
} finally {
  await laya.close();
}
```

- `state`: object (or pre-serialized string) describing the case.
- `questions`: `{ id: { type: 'choice'|'score'|'noul', instructions, criteria } }`.
  Up to 32 questions per call — batch them (see `docs/performance.md`).
- `result.answers[id]`: `{ type, choice|score|noul, probabilities, confidence, action: { act_probability } }`,
  calibrated exactly like the torch reference (verified ≤7.7e-06 logit drift).
- Reuse one client across predictions: `open()` loads ~1.6GB (~2s); steady
  predictions skip that cost. First `predict()` also warms the ORT graph.

## CLI

```sh
cd packages/laya-js
node bin/cli.ts --help
node bin/cli.ts --json # default 3-question example, compact JSON
node bin/cli.ts --state-file s.json --questions-file q.json --model ../models/laya-split-fp16.onnx
```

Exit code is 1 on invalid JSON or inference failure; errors name the cause.

## Browser

The supported pattern is a worker (tokenize + inference off the main thread),
as implemented in `apps/playground/src/worker.ts`:

```js
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
worker.onmessage = (e) => {
  if (e.data.type === 'done') console.log(e.data.result.answers, e.data.timings);
  if (e.data.type === 'error') console.error(e.data.message);
};
worker.postMessage({ state, questions, backend: 'auto', precision: 'fp32' });
// backend: 'auto' (webgpu→wasm fallback) | 'webgpu' | 'wasm'
// precision: 'fp32' (accurate everywhere) | 'fp16' (half the download)
```

- On WebGPU the session must use `graphOptimizationLevel: 'basic'`
  (works around an ORT 1.30 fusion bug); the playground does this for you.
- Models (1.6GB/806MB) are fetched at runtime, never bundled. The playground
  serves local `models/` at `/models` in dev; production must host them at
  `/models` (or any URL you pass to `InferenceSession.create`).
- Result `timings` has the same four phases as Node; the UI shows
  `tok · infer · post · total` per run.

## Python (no torch)

```sh
.venv/bin/python tools/export/predict_onnx.py # default example
.venv/bin/python tools/export/predict_onnx.py --state '{"subject":".."}' --questions '{...}'
```

Tokenizes with the reference implementation, runs the split graph in
onnxruntime on CPU, calibrates with the checkpoint temperature tables in
numpy. Output shape matches the TS SDK (`answers` + per-type fields).

## Porting to a new platform (Android, iOS, edge, …)

No Android/iOS runtime ships yet — but every port implements the same small
contract, so the next one is mechanical, not research:

1. **Tokenizer bytes** — pure-JS BPE in `packages/laya-js/src/laya-bpe.ts`
   (NFC, GPT-2 regex, rank merges); verify with `packages/test-vectors/vectors/bpe-fuzz.json`
   (206 cases) and the 5 fixture `input_ids`.
2. **Preprocessing** — `buildSequence`/`collateItems` caps (`max_len` 512,
   `head_max_len` 192, per-option 48); byte-identical IDs required.
3. **Graph** — any ONNX runtime (ORT has Android/iOS/JS/C# builds) running
   `laya-split-single.onnx`: inputs `input_ids/attention_mask/marker_pos/marker_mask/qtype`
   → outputs `logits/pooled`.
4. **Calibration** — temperature tables from `rl_agent_config.json` +
   CPU action head (`act_head` weights); verify with the 5 parity fixtures
   (logits ≤7.7e-06) and the 13 accuracy checks.

`packages/test-vectors/` is the shared oracle: if a new port reproduces the
fixtures, it reproduces the model. See `docs/explanation/laya-portability.md`
for the six exact porting rules.

## Troubleshooting

- `cannot load model` — `models/` missing or incomplete; rebuild via
  `tools/export/` and check `checksums.sha256`.
- `HTTP 404` on `/models/*` in the browser — models aren't served; use the
  playground dev server or host them at `/models`.
- `no backend worked` — WebGPU unavailable and WASM failed to init; check
  browser support and the error message (first 500 chars are surfaced).
- Answers differ from torch — run `check_parity.py` (Python) or
  `npm run check` (`@laya/js`); drift above 1e-4 means the port diverged.
