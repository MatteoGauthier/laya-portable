"""Weight-only 4-bit quantization of the split FP32 model (WebGPU-oriented).

MatMul/Gather weights to 4-bit blocks; activations stay floating-point, which
usually preserves transformer accuracy better than dynamic INT8. Output uses
MatMulNBits, natively supported by the ORT WebGPU backend. Saved single-file
for direct browser use.

Usage: .venv/bin/python export/to_4bit.py [--src ...] [--dst ...] [--bits 4] [--block-size 128]
"""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "models" / "laya-split-single.onnx")
    ap.add_argument("--dst", type=Path, default=ROOT / "models" / "laya-split-4bit.onnx")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=128)
    args = ap.parse_args()
    src, dst = str(args.src), str(args.dst)
    print(f"4-bit quantizing {src} ... (takes minutes)", flush=True)
    quant = MatMulNBitsQuantizer(onnx.load(src), bits=args.bits, block_size=args.block_size, is_symmetric=True, accuracy_level=4)
    quant.process()
    quant.model.save_model_to_file(dst, False)
    print(f"done: {Path(dst).stat().st_size/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
