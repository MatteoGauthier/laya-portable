"""Head-to-head: MLX vs torch MPS/CPU vs ONNX CPU, same fixtures, same policy.

5 warmup / 20 repeats, explicit device sync, forward-only. Writes
packages/test-vectors/reports/head2head.json. MPS rows are skipped with a
note when the sandbox hides Metal (baseline-doc numbers stay the reference).
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream" / "laya"))
sys.path.insert(0, str(ROOT / "tools" / "mlx"))
import numpy as np

NAMES = ["choice-2", "choice-3", "choice-6", "orig-3q", "mixed-batch"]
WARMUP, REPEATS = 5, 20

def bench(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(ts)), 1), round(float(np.percentile(ts, 95)), 1)

def main():
    import torch
    import laya
    from huggingface_hub import snapshot_download
    import onnxruntime as ort
    import mlx.core as mx
    from laya_mlx import LayaMLX, load_weights

    LT = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(28)]
    TH = {"full_attention": 160000.0, "sliding_attention": 10000.0}
    REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
    mp = snapshot_download("convaiinnovations/laya", revision=REVISION, local_files_only=True)

    print("loading torch cpu + onnx + mlx...", flush=True)
    agent_cpu = laya.load(mp, device="cpu")
    sess = ort.InferenceSession(str(ROOT / "models" / "laya-split-single.onnx"), providers=["CPUExecutionProvider"])
    mlx_model = LayaMLX(load_weights(), LT, TH)
    mps_ok = torch.backends.mps.is_available()
    agent_mps = laya.load(mp, device="mps") if mps_ok else None
    print(f"mps_available={mps_ok}", flush=True)

    report = {"warmup": WARMUP, "repeats": REPEATS, "mps_available": mps_ok, "fixtures": {}}
    for name in NAMES:
        d = np.load(ROOT / "packages" / "test-vectors" / "vectors" / f"{name}.npz")
        bt = {k: torch.from_numpy(d[k]) for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]}
        keys = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
        row = {"batch": int(d["input_ids"].shape[0]), "seq": int(d["input_ids"].shape[1])}
        with torch.no_grad():
            row["torch-cpu"] = bench(lambda: agent_cpu.model(*[bt[k] for k in keys]))
        feeds = {k: d[k] for k in keys}
        row["onnx-cpu"] = bench(lambda: sess.run(None, feeds))
        fm = {k: mx.array(d[k]) for k in keys}
        def mlx_fwd():
            lo, po = mlx_model.forward(**fm)
            mx.eval(lo, po)
        row["mlx"] = bench(mlx_fwd)
        if mps_ok:
            bm = {k: bt[k].to("mps") for k in keys}
            with torch.no_grad():
                def mps_fwd():
                    agent_mps.model(*[bm[k] for k in keys])
                    torch.mps.synchronize()
                row["torch-mps"] = bench(mps_fwd)
        else:
            row["torch-mps"] = "skipped (sandbox hides Metal; baseline doc: 21.8/40.0ms 1q/3q)"
        report["fixtures"][name] = row
        print(f"{name} B={row['batch']} S={row['seq']}: " +
              " ".join(f"{k}={v[0]}ms" if isinstance(v, list) else f"{k}={v}" for k, v in row.items() if k not in ("batch", "seq")),
              flush=True)
    out = ROOT / "packages" / "test-vectors" / "reports" / "head2head.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
