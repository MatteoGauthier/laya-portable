"""Parity: FP16 split ORT vs torch FP32 (and vs FP32 ORT for reference).

Action head stays FP32 numpy (same as split validation). Reports raw drift,
calibrated prob drift, confidence drift, and label flips — not just labels.

Usage:
  .venv/bin/python tools/export/check_fp16.py [--model ...] [--report ...]
  .venv/bin/python tools/export/check_fp16.py --subfolder multilingual \\
      --model models/laya-multilingual-split-fp16.onnx \\
      --ref32 models/laya-multilingual-split-single.onnx \\
      --npz tools/export/act_head.multilingual.npz \\
      --parity-report packages/test-vectors/reports/parity-split-multilingual.json \\
      --report packages/test-vectors/reports/fp16-parity-multilingual.json
"""
import argparse
import json, sys
from pathlib import Path
import math
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "packages" / "test-vectors" / "vectors"
REPORTS = VECTORS.parent / "reports"
sys.path.insert(0, str(ROOT / "upstream" / "laya"))
import onnxruntime as ort

REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
FIXTURE_NAMES = ["orig-3q", "choice-3", "choice-2", "choice-6", "mixed-batch"]

def load_head(npz_path):
    ah = np.load(npz_path)
    return (ah["0.weight"], ah["0.bias"], ah["2.weight"], ah["2.bias"])

def cpu_act(logits64, mm, pooled32):
    p = np.exp(logits64 - logits64.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    k = mm.sum(axis=1).clip(min=2).astype(np.float64)
    ent = -(p * np.log(np.clip(p, 1e-9, 1))).sum(axis=1) / np.log(k)
    top2 = np.sort(p, axis=1)[:, -2:][:, ::-1]
    feats = np.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], axis=1).astype(np.float32)
    h = np.concatenate([pooled32, feats], axis=1) @ W0.T + b0
    h = 0.5 * h * (1 + np.array([[math.erf(v / 1.4142135623730951) for v in row] for row in h], dtype=np.float32))
    return h @ W2.T + b2

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def conf_entropy(p, k):
    if k < 2: return 1.0
    ent = -(p[:k] * np.log(np.clip(p[:k], 1e-12, 1))).sum()
    return float(np.clip(1 - ent / math.log(k), 0, 1))

ap = argparse.ArgumentParser()
ap.add_argument("--revision", default="c5d78730f3493e4fe16d61507ef4b78eef7318cf")
ap.add_argument("--repo", default="convaiinnovations/laya")
ap.add_argument("--subfolder", default=None, help="None (english root), multilingual, typed-decisions")
ap.add_argument("--model", type=Path, default=ROOT / "models" / "laya-split-fp16.onnx")
ap.add_argument("--ref32", type=Path, default=None, help="FP32 split reference (default: english single)")
ap.add_argument("--npz", type=Path, default=None, help="Act-head weights (default: english act_head.npz)")
ap.add_argument("--parity-report", type=Path, default=None, help="Qtypes source (default: english parity-report)")
ap.add_argument("--report", type=Path, default=REPORTS / "fp16-parity.json")
args, _ = ap.parse_known_args()
ckpt = args.subfolder or "english"
suffix = "" if ckpt == "english" else f".{ckpt}"
ref32 = args.ref32 or (ROOT / "models" / "laya-split-single.onnx")
npz = args.npz or (ROOT / "tools" / "export" / "act_head.npz")
parity_report = args.parity_report or (REPORTS / "parity-report.json")
W0, b0, W2, b2 = load_head(npz)
print(f"act head w0 {list(W0.shape)} checkpoint={ckpt}", flush=True)
sess16 = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
sess32 = ort.InferenceSession(str(ref32), providers=["CPUExecutionProvider"])
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from huggingface_hub import snapshot_download
snap = Path(snapshot_download(args.repo, revision=args.revision, local_files_only=True))
snap = snap / args.subfolder if args.subfolder else snap
rep = {"model": Path(args.model).name, "checkpoint": ckpt, "fixtures": []}
for name in FIXTURE_NAMES:
    d = np.load(VECTORS / f"{name}{suffix}.npz")
    feeds = {k: d[k] for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]}
    lo16, po16 = sess16.run(None, feeds)
    lo32, po32 = sess32.run(None, feeds)
    mm = d["marker_mask"]
    tl, ta = d["torch_logits"], d["torch_act"]
    act16 = cpu_act(lo16.astype(np.float64), mm, po16.astype(np.float32))
    # raw drift vs torch
    dl = float(np.abs(lo16 - tl)[mm].max())
    dp = float(np.abs(po16.astype(np.float64) - po32.astype(np.float64)).max())  # pooled has no torch ref; vs fp32 ORT
    da = float((np.abs(act16 - ta) / np.maximum(1, np.abs(ta))).max())
    # calibrated drift vs torch (temperature from checkpoint config)
    import json as js
    cfg = js.load(open(Path(snap) / "rl_agent_config.json"))
    fx = [f for f in js.load(open(parity_report))["fixtures"] if f["name"] == name][0]
    from laya.common import QTYPES, temp_bucket
    qtypes = fx["qtypes"]
    qnames = ["choice", "score", "noul"]
    maxp, maxc, flips = 0.0, 0.0, []
    for r in range(len(qtypes)):
        k = int(mm[r].sum())
        qt = qtypes[r]
        size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        ts = cfg["temperature_by_options"].get(f"{qnames[qt]}:{size}", cfg["temperature"][qt])
        z16 = lo16[r, :k] / max(1e-3, ts)
        z0 = tl[r, :k] / max(1e-3, ts)
        p16 = np.exp(z16 - z16.max()); p16 /= p16.sum()
        p0 = np.exp(z0 - z0.max()); p0 /= p0.sum()
        maxp = max(maxp, float(np.abs(p16 - p0).max()))
        c16 = conf_entropy(p16, k) if qt != 2 else max(p16[1], 1 - p16[1])
        c0 = conf_entropy(p0, k) if qt != 2 else max(p0[1], 1 - p0[1])
        maxc = max(maxc, abs(c16 - c0))
        if qt == 0 and int(p16.argmax()) != int(p0.argmax()):
            flips.append(r)
    print(f"{name}: dlogits={dl:.2e} dpooled(fp32ort)={dp:.2e} actrel={da:.2e} pdrift={maxp:.2e} cdrift={maxc:.2e} flips={flips or 'none'}", flush=True)
    rep["fixtures"].append({"name": name, "max_abs_logits_vs_torch": float(dl), "max_abs_pooled_vs_fp32ort": float(dp),
        "max_rel_act_vs_torch": float(da), "max_calibrated_prob_drift": float(maxp), "max_confidence_drift": float(maxc), "label_flips": [int(x) for x in flips]})
(args.report).write_text(json.dumps(rep, indent=2) + "\n")
print(f"wrote {args.report}")
