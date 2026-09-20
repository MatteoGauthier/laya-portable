# Reports schema (`packages/test-vectors/reports/`)

Machine-readable verdicts produced by `tools/export/`. Asserted by
`tools/export/tests/test_reports.py` (CI-gatable, no model load) and rendered
by the playground Progress tab. Regenerate — never hand-edit (see
`packages/test-vectors/README.md`).

## parity-report.json (from `check_parity.py`, exits 1 on drift)

Top level: `model_revision`, `source_commit`, `onnx` (repo-relative path),
`ort_version`, `torch_version`, `overall_pass: bool`, `fixtures[5]`.

Per fixture (`orig-3q`, `choice-3`, `choice-2`, `choice-6`, `mixed-batch`):
`name`, `batch`, `seq_len`, `kmax`, `k_per_q[]`, `qtypes[]`,
`max_abs_logits`, `mean_abs_logits`, `max_abs_act_logits`,
`mean_abs_act_logits`, `max_rel_act_logits`, `max_calibrated_prob_drift`,
`max_act_prob_drift`, `labels_match`, `rounded_4dec_match`,
`torch{}` / `onnx{}` (per-question calibrated answers).

Gates: `overall_pass` true, 5 fixtures, `max_abs_logits < 1e-4`,
`max_calibrated_prob_drift < 1e-4`, `labels_match` true.

## accuracy-report.json (from `check_accuracy.py`, 13 weak checks)

`cases: 13`, `adapters[]` with `name`, `score`, `rows[]`
(`id`, `pass`, `margin`, `got`). Gate: `torch-fp32` scores 13/13;
quantized variants are informational (documented rejections).

## fp16-parity.json (from `check_fp16.py`)

Per fixture: `name`, `max_abs_logits_vs_torch`,
`max_abs_pooled_vs_fp32ort`, `max_rel_act_vs_torch`,
`max_calibrated_prob_drift`, `max_confidence_drift`, `label_flips[]`.

## laya-faithful-metadata.json (from `export_onnx.py`)

`model_id`, `model_revision`, `model_path` (cache dir name only, never an
absolute path), `source_commit`, `versions{}`, `opset` (= 18), `inputs[]`,
`outputs[]`, `operators{}`, `notes`. Gate: opset 18, pinned revision, no
`/Users/` paths anywhere in the JSON.

## mac-baseline.json (from `benchmark.py`)

`platform`, `python`, `versions{}`, `source_commit`, `model_revision`,
`mps_available`, `cpu_threads`, `warmup`, `repeats`, `state`, `questions`,
`runs[]` (`device`, `dtype`, `questions`, `input_shape`, `load_seconds`,
`forward`/`end_to_end` with `p50_ms`/`p95_ms`/`samples_ms`, `answer`).
Informational only (single-machine latency, not asserted).
