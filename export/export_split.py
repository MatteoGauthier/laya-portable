"""Split export: GPU-heavy part returns (logits, pooled); action head runs on CPU/JS.

Removes TopK/Log/ReduceSum + act_head Gemms from the GPU graph. pooled is the
first-token hidden state after head layers (h[:,0]), matching DecisionModel.forward.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "laya"))
import torch
import torch.nn as nn
REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"

class SplitWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.type_emb = model.type_emb
        self.head = model.head
        self.scorer = model.scorer
    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        pooled = h[:, 0].float()
        return logits, pooled

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT/"models"/"laya-split.onnx")
    args = ap.parse_args()
    import laya
    from huggingface_hub import snapshot_download
    mp = snapshot_download("convaiinnovations/laya", revision=REVISION, local_files_only=True)
    agent = laya.load(mp, device="cpu")
    w = SplitWrapper(agent.model).eval()
    B, S, K = 2, 32, 3
    dummy = (torch.randint(0, 50000, (B, S)), torch.ones((B, S), dtype=torch.long),
             torch.tensor([[5, 10, 15], [6, 12, 0]]), torch.tensor([[True]*3, [True, True, False]]),
             torch.tensor([0, 1]))
    with torch.no_grad():
        lo, po = w(*dummy)
    print(f"torch ok: logits {tuple(lo.shape)} pooled {tuple(po.shape)}", flush=True)
    torch.onnx.export(w, dummy, str(args.output),
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["logits", "pooled"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "marker_pos": {0: "batch_size", 1: "num_options"},
            "marker_mask": {0: "batch_size", 1: "num_options"},
            "qtype": {0: "batch_size"}, "logits": {0: "batch_size", 1: "num_options"},
            "pooled": {0: "batch_size"}}, opset_version=18, do_constant_folding=True)
    import onnx
    from collections import Counter
    onnx.checker.check_model(str(args.output))
    m = onnx.load(str(args.output), load_external_data=False)
    ops = dict(sorted(Counter(n.op_type for n in m.graph.node).items()))
    print(f"export ok: {args.output.stat().st_size/1e6:.1f} MB graph, ops removed vs full: "
          f"{set(['TopK','Log','ReduceSum']) - set(ops)}", flush=True)
    # Save act_head weights + metadata for CPU/JS side
    sd = agent.model.act_head.state_dict()
    import numpy as np
    np.savez(ROOT/"export"/"act_head.npz", **{k: v.numpy() for k, v in sd.items()})
    print(f"act_head weights: {list(sd.keys())}", flush=True)
    # Single-file for browser
    m2 = onnx.load(str(args.output), load_external_data=True)
    for i in m2.graph.initializer:
        del i.external_data[:]
        i.data_location = onnx.TensorProto.DEFAULT
    single = str(args.output).replace(".onnx", "-single.onnx")
    onnx.save_model(m2, single)
    print(f"single-file: {Path(single).stat().st_size/1e9:.2f} GB", flush=True)

if __name__ == "__main__":
    main()
