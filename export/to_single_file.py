"""Convert external-data ONNX to single-file (<2GB protobuf limit).

ORT Web 1.30 fails to load external data in browser
("Module.MountedFiles is not available"). Single-file 1.6GB works in
WASM/WebGPU, Node, and Python. Numerics identical, packaging only.

Usage: .venv/bin/python export/to_single_file.py
"""
from pathlib import Path
import onnx
ROOT = Path(__file__).resolve().parents[1]
src = ROOT/"models"/"laya-faithful.onnx"
dst = ROOT/"models"/"laya-faithful-single.onnx"
print(f"loading {src} with external data...")
m = onnx.load(str(src), load_external_data=True)
for init in m.graph.initializer:
    del init.external_data[:]
    init.data_location = onnx.TensorProto.DEFAULT
print(f"saving {dst} ...")
onnx.save_model(m, str(dst))
print(f"done: {dst.stat().st_size/1e9:.2f} GB")
