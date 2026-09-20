"""CoreML spike (Phase 4): convert the split model, check parity, benchmark vs MPS.

Traces SplitWrapper via TorchScript and converts with flexible batch/seq/options
ranges. Int64 inputs are cast to int32 at entry (ids fit; CoreML MIL prefers
int32). Saves models/laya-split.mlpackage (gitignored).
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream" / "laya"))
sys.path.insert(0, str(ROOT / "tools" / "export"))
import torch
import torch.nn as nn

REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"

class CoreMLWrapper(nn.Module):
    def __init__(self, split):
        super().__init__()
        self.m = split
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.m(input_ids.to(torch.int32), attention_mask.to(torch.int32),
                      marker_pos.to(torch.int32), marker_mask, qtype.to(torch.int32))

def main():
    import laya
    from huggingface_hub import snapshot_download
    from export_split import SplitWrapper
    import coremltools as ct

    mp = snapshot_download("convaiinnovations/laya", revision=REVISION, local_files_only=True)
    print("loading torch agent (cpu)...", flush=True)
    agent = laya.load(mp, device="cpu")
    split = SplitWrapper(agent.model).eval()
    wrap = CoreMLWrapper(split).eval()
    ex = (torch.randint(0, 50000, (1, 64), dtype=torch.int64), torch.ones((1, 64), dtype=torch.int64),
          torch.tensor([[5, 10, 15]]), torch.tensor([[True, True, True]]), torch.tensor([0]))
    print("exporting via torch.export (strict=False)...", flush=True)
    # NOTE: run_decompositions({}) preserves aten.new_ones, which coremltools 9
    # cannot convert ("Unsupported fx node new_ones"). The default table already
    # contains a new_ones decomposition, so plain run_decompositions() gets past
    # that gate. Next frontier (2026-09-20): dynamic batch specializes to 1
    # inside HF ModernBERT internals (Dim.AUTO keeps seq dynamic), and MIL then
    # rejects gather_along_axis with fp32 indices from the decomposed mask path.
    ep = torch.export.export(wrap, ex, strict=False).run_decompositions()
    print("converting (flexible shapes)...", flush=True)
    mlmodel = ct.convert(
        ep, inputs=[
            ct.TensorType(name="input_ids", shape=(ct.RangeDim(1, 4), ct.RangeDim(16, 128)), dtype=np.int64),
            ct.TensorType(name="attention_mask", shape=(ct.RangeDim(1, 4), ct.RangeDim(16, 128)), dtype=np.int64),
            ct.TensorType(name="marker_pos", shape=(ct.RangeDim(1, 4), ct.RangeDim(2, 8)), dtype=np.int64),
            ct.TensorType(name="marker_mask", shape=(ct.RangeDim(1, 4), ct.RangeDim(2, 8)), dtype=np.bool_),
            ct.TensorType(name="qtype", shape=(ct.RangeDim(1, 4),), dtype=np.int64)],
        outputs=["logits", "pooled"], minimum_deployment_target=ct.target.macOS15)
    out = ROOT / "models" / "laya-split.mlpackage"
    mlmodel.save(str(out))
    print(f"saved {out}", flush=True)

    # Parity on 5 fixtures (CoreML ALL units vs torch)
    for name in ["orig-3q", "choice-3", "choice-2", "choice-6", "mixed-batch"]:
        d = np.load(ROOT / "packages" / "test-vectors" / "vectors" / f"{name}.npz")
        feeds = {k: d[k] for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]}
        r = mlmodel.predict(feeds)
        lo = np.asarray(r["logits"])
        dl = float(np.abs(lo - d["torch_logits"])[d["marker_mask"]].max())
        print(f"{name}: coreml dlogits vs torch {dl:.2e}", flush=True)

    # Latency: CoreML vs torch MPS, choice-3
    d = np.load(ROOT / "packages" / "test-vectors" / "vectors" / "choice-3.npz")
    feeds = {k: d[k] for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]}
    for _ in range(5): mlmodel.predict(feeds)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); mlmodel.predict(feeds); ts.append((time.perf_counter() - t0) * 1000)
    print(f"coreml choice-3 p50 {np.median(ts):.1f}ms p95 {np.percentile(ts, 95):.1f}ms", flush=True)
    if torch.backends.mps.is_available():
        magent = laya.load(mp, device="mps")
        t = [torch.from_numpy(d[k]).to("mps") if k != "marker_mask" else torch.from_numpy(d[k]).to("mps") for k in
             ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]]
        with torch.no_grad():
            for _ in range(5): magent.model(*t)
            torch.mps.synchronize()
            ms = []
            for _ in range(20):
                torch.mps.synchronize(); s = time.perf_counter()
                magent.model(*t); torch.mps.synchronize(); ms.append((time.perf_counter() - s) * 1000)
        print(f"mps choice-3 forward p50 {np.median(ms):.1f}ms (baseline doc: 21.8ms 1q)", flush=True)

if __name__ == "__main__":
    main()
