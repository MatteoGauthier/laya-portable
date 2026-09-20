"""Capture torch reference intermediates for the MLX port validation.

Replays the encoder layer-by-layer (same calls as ModernBertModel.forward,
no forward hooks — hooks fight the checkpointing/output wrappers) and records
per-layer attention masks (exact additive values), embeddings, layer outputs,
and final logits/pooled for the choice-2 fixture. Saves tools/mlx/ref.npz.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream" / "laya"))
import numpy as np
import torch

REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"

def main():
    import laya
    from huggingface_hub import snapshot_download
    from transformers.models.modernbert.modeling_modernbert import (
        create_bidirectional_mask, create_bidirectional_sliding_window_mask)
    mp = snapshot_download("convaiinnovations/laya", revision=REVISION, local_files_only=True)
    agent = laya.load(mp, device="cpu")
    model = agent.model.eval()
    enc = model.encoder

    d = np.load(ROOT / "packages" / "test-vectors" / "vectors" / "choice-2.npz")
    input_ids = torch.from_numpy(d["input_ids"])
    attention_mask = torch.from_numpy(d["attention_mask"])
    out = {}
    with torch.no_grad():
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len).unsqueeze(0)
        hidden = enc.embeddings(input_ids=input_ids)
        out["emb"] = hidden.float().numpy()
        mask_kwargs = {"config": enc.config, "inputs_embeds": hidden, "attention_mask": attention_mask}
        mapping = {"full_attention": create_bidirectional_mask(**mask_kwargs),
                   "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs)}
        pos_emb = {lt: enc.rotary_emb(hidden, position_ids, lt) for lt in set(enc.config.layer_types)}
        for i, layer in enumerate(enc.layers):
            am = mapping[layer.attention_type]
            out[f"mask_{i}"] = None if am is None else np.asarray(am, dtype=np.float32)
            hidden = layer(hidden, attention_mask=am,
                           position_embeddings=pos_emb[layer.attention_type])
            out[f"layer_{i}"] = hidden.float().numpy()
        hidden = enc.final_norm(hidden)
        out["enc_last"] = hidden.float().numpy()
        t_logits, t_act = model(*feeds_input_ids(input_ids, attention_mask, d))
        out["torch_logits"] = t_logits.float().numpy()
        out["torch_act"] = t_act.float().numpy()
    for k, v in out.items():
        print(f"{k}: None" if v is None else f"{k}: {v.shape} {v.dtype} finite={np.isfinite(v).all()}")
    np.savez(ROOT / "tools" / "mlx" / "ref.npz", **out)
    print("wrote tools/mlx/ref.npz")

def feeds_input_ids(input_ids, attention_mask, d):
    return (input_ids, attention_mask, torch.from_numpy(d["marker_pos"]),
            torch.from_numpy(d["marker_mask"]), torch.from_numpy(d["qtype"]))

if __name__ == "__main__":
    main()
