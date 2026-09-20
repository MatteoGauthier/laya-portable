"""Dynamic INT8 quantization of the split FP32 model (CPU-oriented).

Weights to int8, activations quantized dynamically at runtime. IO stays float
so the existing adapter contract is unchanged. This is a CPU/WASM path:
dynamic-INT8 ops are not WebGPU-supported.

Usage: .venv/bin/python export/to_int8.py [--src ...] [--dst ...]
"""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
from onnxruntime.quantization import quantize_dynamic, QuantType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "models" / "laya-split-single.onnx")
    ap.add_argument("--dst", type=Path, default=ROOT / "models" / "laya-split-int8.onnx")
    args = ap.parse_args()
    src, dst = str(args.src), str(args.dst)
    print(f"quantizing {src} ...", flush=True)
    quantize_dynamic(src, dst, weight_type=QuantType.QUInt8)
    print(f"done: {Path(dst).stat().st_size/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
