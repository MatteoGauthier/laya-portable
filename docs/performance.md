# Performance guide

All numbers below are measured, not estimated. Machine specifics are noted;
treat absolute ms as orientation and ratios as guidance.

## Reading the timings

Every prediction reports four phases (`result.timings` in the SDK/CLI/Python,
`tok · infer · post · total` in the playground):

| Phase            | What it is                                    | Typical share         |
| ---------------- | --------------------------------------------- | --------------------- |
| `tokenize_ms`    | BPE + sequence building + feed flattening     | ~1–9ms                |
| `inference_ms`   | `ORT session.run` only                        | 60–90% of steady time |
| `postprocess_ms` | Split outputs + CPU action head + calibration | ~6–8ms                |
| `total_ms`       | Wall clock end-to-end                         | sum + overhead        |

Example — Node SDK, split FP32 on CPU, standard 3-question case: `9 / 100 / 8 / 117`.
Inference dominates; if your `total` is large but `inference` is small, the
cost is outside the model (usually first-run loading — see below).

The playground renders the same phases as a waterfall
(`download → session → tokenize → inference → postprocess`, plus an honest
`overhead` row for the remainder), so per-run costs are visible, not inferred.

## First run vs steady state

| Cost           | Node                                                         | Browser                                              |
| -------------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| Model bytes    | Already on disk (`models/`)                                  | 1.6GB FP32 / 806MB FP16 downloaded once, then cached |
| Session load   | ~2s (`LayaClient.open`)                                      | Session init after download (watch the stage list)   |
| Graph warmup   | First `predict()` compiles/optimizes; later calls are steady | Same, per backend                                    |
| Steady predict | `total_ms` (e.g. 82–125ms CPU, 3Q)                           | `infer` per run                                      |

Practice: reuse one client/session for the process lifetime, and treat the
first prediction as warmup (the playground does one implicitly per session).

## The knob matrix (measured)

Backend × precision × graph variant, standard case:

| Backend      | Variant             | Run                        | Accuracy verdict                     |
| ------------ | ------------------- | -------------------------- | ------------------------------------ |
| Node CPU     | split fp32          | ~100ms infer               | reference parity ≤7.7e-06            |
| WASM         | full fp32           | 756ms                      | PASS                                 |
| WebGPU-basic | split fp32          | 306ms cold                 | PASS — fastest accurate browser path |
| WebGPU-basic | full fp32           | 1072ms                     | PASS but pointless vs split          |
| WASM         | split fp16          | ~762ms, drift 1.57e-03     | matches CPU; no speedup, no gain     |
| WebGPU-basic | split fp16          | ~180ms, drift 4.11e-02     | **rejected** — fast but wrong        |
| CPU          | split int8 (405MB)  | no speedup, 12/13 accuracy | **rejected** (phishing flip)         |
| WebGPU       | split 4-bit (395MB) | 219ms, drift 3.35e+00      | **rejected** — numerically broken    |

Notes: WebGPU requires `graphOptimizationLevel: 'basic'` (ORT 1.30 fusion
bug); FP16 halves the download and is fine on CPU/WASM (13/13 accuracy) but
buys no speed there. The full graph only matters for the faithful-parity
checks — serve the split graph.

## Batch your questions

One `predict()` with N questions beats N × one-question calls (~2× measured:
3×82ms separate vs 125ms batched, split fp32 CPU). The per-call overhead
(tokenize, postprocess, JS↔ORT crossings) is paid once; inference grows
sublinearly. Limits: ≤32 questions per call, `K≥1` markers each.

## Input size

Caps are `max_len` 512 tokens and `head_max_len` 192 (from
`rl_agent_config.json`), per-option 48. Shorter states/questions mean smaller
`S` and proportionally cheaper attention — the `seq` figure in the playground
tells you what you actually paid for. Truncation is right-biased by default
(`truncateLeft` kept for special cases).

## Apple Silicon note

Native options beat the portable ones on Mac: MLX FP32 forwards the same
case in **19.3ms** vs 21.8ms torch MPS (M4 Pro, `tools/mlx/`), while Node CPU
takes ~100ms. If your deployment is Mac-only, MLX is the fast path; if it's
multi-platform, ONNX is the portable path. That's the trade the repo exists to
offer — see `docs/explanation/model-comparison.md`.

## Make-it-fast checklist

1. Serve the **split** graph, FP32, on the fastest accurate backend
   (Node CPU → WebGPU-basic → WASM, in that order per above).
2. **Batch** questions into one `predict()`.
3. **Reuse** the client/session; warm up once.
4. Keep inputs **short**; watch `seq`.
5. Only then consider precision surgery — with the accuracy harness gating
   every step (`13/13` or it doesn't ship).
