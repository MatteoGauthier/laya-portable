"""Convert split single-file FP32 to FP16 (internal only, IO stays FP32).

Standard ORT float16 conversion: weights + compute in FP16, inputs/outputs
FP32 (keep_io_types) so the JS/Python contract is unchanged. Action head
stays FP32 on CPU/JS; only the GPU graph is quantized.

Usage: .venv/bin/python export/to_fp16.py [--src ...] [--dst ...]
"""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import onnx
from onnxconverter_common import float16

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "models" / "laya-split-single.onnx")
    ap.add_argument("--dst", type=Path, default=ROOT / "models" / "laya-split-fp16.onnx")
    args = ap.parse_args()
    src, dst = args.src, args.dst
    print(f"loading {src} ({src.stat().st_size/1e9:.2f} GB)...", flush=True)
    m = onnx.load(str(src))
    print("converting to float16 (keep_io_types=True)...", flush=True)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True,
        # Keep Casts (bool masks, .float() tails) exact: converting the Cast to
        # float left its declared output type inconsistent (ORT load FAIL otherwise).
        op_block_list=['Cast'])
    print(f"saving {dst}...", flush=True)
    onnx.save_model(m16, str(dst))
    print(f"done: {dst.stat().st_size/1e9:.2f} GB", flush=True)

if __name__ == "__main__":
    main()
