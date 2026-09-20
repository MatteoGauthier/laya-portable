# Four-way comparison: Jev vs Laya original vs ours vs community

Honest version of "how do we compete". Three evidence grades — read the
**Source** column before quoting any number:

- **Measured here**: same inputs, same policy, this repo's harnesses.
- **Upstream published**: Laya team's BENCHMARKS.md (their prompts, their samples).
- **Third-party published**: Jev vendor figures via upstream/community cards —
  never measured here, prompts and samples differ. Indicative only.

## Accuracy (higher better)

| Task | Jev (3rd-party publ.) | Laya original | Ours | Community |
|---|---|---|---|---|
| AG News 4-way | 0.910 | 0.950–0.953 (upstream publ.); **0.935 ours meas.** (200, bare prompts) | **0.935** (== torch) | excluded (K=4 > fixed 2) |
| Emotion 6-way | 0.480 | 0.595–0.600 (upstream); **0.587 ours meas.** (2000) | **0.587** (== torch) | excluded (K=6) |
| banking77 77-way | **0.870** | 0.425 (upstream); **0.435 ours meas.** (154) | **0.435** (== torch) | excluded (K=77) |
| typed-decisions 2000 | 0.727 | 0.766 (typed ckpt, upstream) | not run (needs upstream train split) | excluded |
| Intent/routing | ~0.95–0.98 | 0.991, ECE 0.009 (community eval, 1475) | 13/13 harness incl. billing 0.966 margin | K=2 rows only: 7/7 == torch |
| Moderation | ~0.92–0.95 | 0.967 (community eval, 2708) | 13/13 harness | single-noul rows: == torch |

Net: on public sets we reproduce upstream within prompt noise; Jev's one
clear win is banking77 (0.870 vs 0.425 ceiling — architectural option budget,
documented in upstream BENCHMARKS.md).

## Calibration (lower ECE better)

| Setup | ECE |
|---|---|
| Laya + refit temperatures (upstream) | 0.081 |
| Ours (same fitted temps, inherited) | same by construction (≤1e-03 drift) |
| Jev (published) | 0.246 |
| Laya as-shipped (upstream) | 0.466 — over-confident, fit on your own data |

## Latency p50, 1 question (different machines — compare shapes, not digits)

| Setup | p50 | Source |
|---|---|---|
| Jev API | 236–276 ms | published |
| Laya T4 (upstream) | 32.8 ms | published |
| Torch MPS, M4 Pro (ours meas.) | 18–19 ms | head2head.json |
| MLX, M4 Pro (ours meas.) | 19 ms | head2head.json |
| ONNX CPU, M4 Pro (ours meas.) | 33–35 ms | head2head.json |
| Browser WASM / WebGPU-basic (ours meas.) | 756 / 306 ms | browser runs |
| Community 1q | 38.4 ms | their eval card (their hardware) |

## What "ours vs community" actually settles

Numerically identical where community can run (K=2, B=1: 7.39e-06,
7/7 probes to 4 decimals). Everywhere else community cannot run:
fixed-2 interface, batch-1 action branch (B=2 hard FAILs), no calibration
config. Ours is a superset with identical numbers — the comparison ends
at compatibility, not quality.

## Gaps in this table

- Jev: no access, never measurable here. Any Jev-vs-ours claim inherits
  upstream's prompt/sample mismatch caveat.
- typed-decisions: needs upstream's train split; our harness + public sets
  cover the same question with weaker labels.
- Community int8/WebGPU cells: unaudited / broken, listed in export README.
