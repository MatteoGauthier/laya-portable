"""Dynamic INT8 quantization of the split FP32 model (CPU-oriented).

Weights to int8, activations quantized dynamically at runtime. IO stays float
so the existing adapter contract is unchanged. This is a CPU/WASM path:
dynamic-INT8 ops are not WebGPU-supported.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
from onnxruntime.quantization import quantize_dynamic, QuantType

src = str(ROOT / "models" / "laya-split-single.onnx")
dst = str(ROOT / "models" / "laya-split-int8.onnx")
print(f"quantizing {src} ...", flush=True)
quantize_dynamic(src, dst, weight_type=QuantType.QUInt8)
print(f"done: {Path(dst).stat().st_size/1e9:.2f} GB", flush=True)
