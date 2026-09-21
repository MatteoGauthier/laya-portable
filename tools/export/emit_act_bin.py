"""Emit binary act_head weights (Float32 little-endian) + JSON manifest.

JSON (5.2MB text) is parsed on every worker init; binary loads ~3x faster
and avoids GC-heavy nested arrays. JS loader: js/laya-act-bin.mjs.

Usage: .venv/bin/python export/emit_act_bin.py [--npz export/act_head.npz] [--out js/act_head.bin]
Writes js/act_head.bin + js/act_head.meta.json {shapes, dtype}.
"""
import argparse
from pathlib import Path
import numpy as np
import json

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=ROOT / "tools" / "export" / "act_head.npz")
    ap.add_argument("--out", type=Path, default=ROOT / "packages" / "test-vectors" / "vectors" / "act_head.bin")
    ap.add_argument("--checkpoint", default=None, help="B2 name: write models/laya-<checkpoint>.act_head.bin+meta")
    args = ap.parse_args()
    if args.checkpoint and args.out == ROOT / "packages" / "test-vectors" / "vectors" / "act_head.bin":
        args.out = ROOT / "models" / f"laya-{args.checkpoint}.act_head.bin"
    ah = np.load(args.npz)
    # canonical order: w0 [256,H+4], b0 [256], w2 [2,256], b2 [2]
    # H=1024 english/typed-decisions, H=768 multilingual.
    names = ["0.weight", "0.bias", "2.weight", "2.bias"]
    arrs = [np.ascontiguousarray(ah[n], dtype=np.float32) for n in names]
    shapes = [list(a.shape) for a in arrs]
    assert shapes[1] == [256] and shapes[2] == [2, 256] and shapes[3] == [2], f"unexpected shapes {shapes}"
    assert shapes[0][0] == 256 and shapes[0][1] in (1028, 772), f"unexpected w0 {shapes[0]}"
    buf = b"".join(a.tobytes(order="C") for a in arrs)
    args.out.write_bytes(buf)
    meta = {"dtype": "float32", "order": ["w0", "b0", "w2", "b2"], "shapes": shapes, "bytes": len(buf)}
    meta_path = args.out.parent / (args.out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.out} ({len(buf)/1e6:.1f}MB) + {meta_path.name}")

if __name__ == "__main__":
    main()
