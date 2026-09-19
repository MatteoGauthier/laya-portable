"""Weight-only 4-bit quantization of the split FP32 model (WebGPU-oriented).

MatMul/Gather weights to 4-bit blocks; activations stay floating-point, which
usually preserves transformer accuracy better than dynamic INT8. Output uses
MatMulNBits, natively supported by the ORT WebGPU backend. Saved single-file
for direct browser use.
"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

src = str(ROOT / "models" / "laya-split-single.onnx")
dst = str(ROOT / "models" / "laya-split-4bit.onnx")
print(f"4-bit quantizing {src} ... (takes minutes)", flush=True)
quant = MatMulNBitsQuantizer(onnx.load(src), bits=4, block_size=128, is_symmetric=True, accuracy_level=4)
quant.process()
quant.model.save_model_to_file(dst, False)
print(f"done: {Path(dst).stat().st_size/1e9:.2f} GB", flush=True)
