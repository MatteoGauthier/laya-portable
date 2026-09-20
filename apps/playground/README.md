# Laya playground (Vite + React + TypeScript)

Dev UI for the portability experiments. Grows with the project: inference today,
quantization/device comparisons next.

Strict TypeScript (`.tsx`/`.ts`, `tsc --noEmit` with `noUncheckedIndexedAccess`

- `erasableSyntaxOnly`); OXC toolchain (`oxlint` + `oxfmt`); Vite 8 + Vitest.

```sh
cd apps/playground && npm install && npm run dev
# open the printed localhost URL (default http://localhost:5173)
```

```sh
npm test        # vitest (api fetchJson, presets, ProbBar/AnswerCard)
npm run lint    # oxlint
npm run typecheck
npm run build   # bundles onnxruntime-web 1.30.0 (offline, ~28MB wasm in dist/)
```

Tabs: **Playground** (run text→answer in a worker, split ONNX + pure-JS BPE),
**Progress** (phase board, parity + latency tables from `export/` + `docs/`),
**Inspector** (tokenizer tester, fixture viewer).

Structure: `src/components/` (tabs + answer cards), `src/hooks/useWorker.tsx`
(single worker instance, isolated heartbeat), `src/lib/` (presets, fetchJson
with error states). Shared inference code is imported from `../js/*.ts` via
the `@js` alias (see `vite.config.ts`).

`public/` symlinks `../models`, `../js`, and the JSON reports — no copies.
Shared inference code is imported from `../js/*.ts` (see `vite.config.ts`
`fs.allow`). First inference run downloads 1.6GB once (dev-machine localhost).
