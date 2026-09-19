# Laya portability investigation

Start with [the findings and recommended path](docs/laya-portability.md). The [runtime comparison](docs/runtime-options.md) covers llama.cpp/ggml, MLX, Core ML, ONNX Runtime, WebGPU/WASM, Android, and ExecuTorch with primary-source links.

- `index.py`: original example.
- `upstream/laya/`: unchanged shallow checkout of the public source repository, commit `6a5819129eb220570792e417e49723d697efd76f` (gitignored; re-clone with the command below).
- `benchmark.py`: offline CPU/MPS baseline for the original example, using the checked-out source.
- `docs/mac-baseline.json`: timings, environment, and prediction outputs from this Mac.
- `docs/sources/`: pinned Hub file inventories and an audit of the community ONNX graph contracts.
- `requirements-benchmark.txt`: environment versions used for the baseline and ONNX inspection.
- `export/`: faithful FP32 ONNX export + CPU parity fixtures ([notes](export/README.md)).
- `models/laya-faithful.onnx`: generated FP32 graph + external data (gitignored, 1.69 GB).

## Reproduce the baseline

The `.venv` environment is already installed locally. The pinned original weights were already in this machine's Hugging Face cache.

```sh
.venv/bin/python benchmark.py
```

For a fresh environment:

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements-benchmark.txt
git clone https://github.com/NandhaKishorM/laya upstream/laya && git -C upstream/laya checkout 6a5819129eb220570792e417e49723d697efd76f
hf download convaiinnovations/laya --revision c5d78730f3493e4fe16d61507ef4b78eef7318cf --include 'model.safetensors' 'rl_agent_config.json' 'encoder/*' 'tokenizer/*'
.venv/bin/python benchmark.py
```

Run `hf` from the activated environment or use `.venv/bin/hf`. `--model /absolute/path/to/checkpoint` accepts a different local model directory; `--devices cpu` skips MPS. The benchmark records a null model revision for a custom path. On systems where a sandbox hides Metal, run from a regular terminal with GPU access; the script refuses to silently call a CPU fallback a GPU result.

The benchmark uses upstream's loader, which can normalize tokenizer configuration in the supplied model directory. It does not download weights itself. Keep the checkpoint resident for inference; cold load and warm prediction are different measurements.

This folder contains research, the source checkout, a measured baseline, and a validated FP32 ONNX export with CPU parity fixtures. It does not yet contain a new MLX/Core ML/ggml/browser port, quantization, or browser/Android validation.
