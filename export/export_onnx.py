"""Faithful ONNX export of Laya DecisionModel (tensor-only, FP32).

Uses the pinned upstream source + checkpoint, offline. Keeps raw logits and
action logits (no temperature/softmax) so calibration stays outside the graph.
Dynamic: batch, sequence, num_options (K>=2, required by topk(2) in forward).
Fixes community export gaps: variable K (not fixed 2) and batched act output
(not [1,2]).

Output: models/laya-faithful.onnx (+ .data) + export/laya-faithful-metadata.json
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, importlib.metadata, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "laya"))

import torch
import torch.nn as nn

REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"

class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        logits, act_logits = self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        return logits, act_logits

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="Local checkpoint dir; default pinned HF cache")
    ap.add_argument("--output", type=Path, default=ROOT/"models"/"laya-faithful.onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--dummy-batch", type=int, default=2)
    ap.add_argument("--dummy-seq", type=int, default=32)
    ap.add_argument("--dummy-options", type=int, default=3)
    args = ap.parse_args()

    import laya
    from huggingface_hub import snapshot_download
    model_path = args.model or snapshot_download("convaiinnovations/laya", revision=REVISION, local_files_only=True)
    print(f"loading {model_path} on cpu ...", flush=True)
    agent = laya.load(model_path, device="cpu")
    assert str(agent.device) == "cpu", agent.device
    model = agent.model.eval()
    print(f"device={agent.device} dtype={next(model.parameters()).dtype}", flush=True)

    wrapper = Wrapper(model).eval()
    B, S, K = args.dummy_batch, args.dummy_seq, args.dummy_options
    assert K >= 2, "forward uses topk(2); K>=2 required"
    input_ids = torch.randint(0, 50000, (B, S), dtype=torch.long)
    attention_mask = torch.ones((B, S), dtype=torch.long)
    attention_mask[1, S*3//4:] = 0
    marker_pos = torch.tensor([[5, 10, 15][:K], [6, 12, 0][:K]], dtype=torch.long)
    if B != 2:  # generic fallback
        marker_pos = (torch.arange(K)[None, :].repeat(B, 1) * 5 + 5).clamp(max=S-1)
        marker_pos[-1, -1] = 0
    marker_mask = torch.ones((B, K), dtype=torch.bool)
    marker_mask[-1, -1] = False  # exercise masked option
    qtype = torch.tensor([0, 1][:B] if B <= 2 else [i % 3 for i in range(B)], dtype=torch.long)

    with torch.no_grad():
        logits, act = wrapper(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    print(f"torch forward ok: logits {tuple(logits.shape)} act {tuple(act.shape)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"exporting opset={args.opset} to {args.output} ...", flush=True)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, marker_pos, marker_mask, qtype),
        str(args.output),
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "act_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "marker_pos": {0: "batch_size", 1: "num_options"},
            "marker_mask": {0: "batch_size", 1: "num_options"},
            "qtype": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "num_options"},
            "act_logits": {0: "batch_size"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    import onnx
    onnx.checker.check_model(str(args.output))
    m = onnx.load(str(args.output), load_external_data=False)
    from collections import Counter
    ops = dict(sorted(Counter(n.op_type for n in m.graph.node).items()))
    print(f"onnx check ok: {args.output.stat().st_size/1e6:.1f} MB graph", flush=True)
    data_file = Path(str(args.output) + ".data")
    if data_file.exists():
        print(f"external data: {data_file.stat().st_size/1e6:.1f} MB", flush=True)

    meta = {
        "model_id": "convaiinnovations/laya",
        "model_revision": REVISION if not args.model else None,
        "model_path": str(model_path),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT/"upstream"/"laya"), "rev-parse", "HEAD"], text=True).strip(),
        "versions": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "onnx", "onnxscript", "numpy"]},
        "opset": args.opset,
        "inputs": [{"name": i.name, "dtype": str(i.type), "shape": [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]} for i in m.graph.input],
        "outputs": [{"name": o.name, "dtype": str(o.type), "shape": [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]} for o in m.graph.output],
        "operators": ops,
        "notes": "FP32 tensor-only; temperature/softmax outside graph; K>=2; act_logits [batch,2]",
    }
    meta_path = ROOT/"export"/"laya-faithful-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_path}", flush=True)

if __name__ == "__main__":
    main()
