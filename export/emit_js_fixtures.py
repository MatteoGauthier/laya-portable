"""Emit small JSON fixtures for the Node.js ORT check.

Reads export/fixtures/*.npz + parity-report + rl_agent_config, writes
js/fixtures/*.json with inputs as nested lists, expected raw outputs,
temperature config, and expected calibrated answers.
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def main():
    parity = json.loads((ROOT/"export"/"parity-report.json").read_text())
    snap = "/Users/matteolemni/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/c5d78730f3493e4fe16d61507ef4b78eef7318cf"
    cfg = json.loads((Path(snap)/"rl_agent_config.json").read_text())
    temp = {"temperature": cfg["temperature"], "temperature_by_options": cfg["temperature_by_options"]}
    outdir = ROOT/"js"/"fixtures"
    outdir.mkdir(parents=True, exist_ok=True)
    for fx in parity["fixtures"]:
        name = fx["name"]
        d = np.load(ROOT/"export"/"fixtures"/f"{name}.npz")
        j = {
            "name": name,
            "batch": fx["batch"], "seq_len": fx["seq_len"], "kmax": fx["kmax"],
            "k_per_q": fx["k_per_q"], "qtypes": fx["qtypes"],
            "qids": list(fx["torch"].keys()),
            "qtype_names": ["choice" if t == 0 else "score" if t == 1 else "noul" for t in fx["qtypes"]],
            "inputs": {k: d[k].tolist() for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]},
            "expected_torch_logits": d["torch_logits"].tolist(),
            "expected_torch_act": d["torch_act"].tolist(),
            "expected_onnx_logits": d["onnx_logits"].tolist(),
            "expected_onnx_act": d["onnx_act"].tolist(),
            "temperature": temp,
            "expected_answers": fx["torch"],  # calibrated from torch, full precision
        }
        (outdir/f"{name}.json").write_text(json.dumps(j) + "\n")
        print(f"wrote {name}.json")
if __name__ == "__main__":
    main()
