"""Accuracy harness through the MLX port (same 13 weak checks as check_accuracy).

Reuses CASES/build_case/evaluate (import-safe: check_accuracy has a main
guard). Action logits via the numpy tail (same as check_mlx), calibrated
with checkpoint temperatures. Exits 1 unless MLX matches torch 13/13.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream" / "laya"))
sys.path.insert(0, str(ROOT / "tools" / "export"))
sys.path.insert(0, str(ROOT / "tools" / "mlx"))
import numpy as np
import mlx.core as mx
from check_accuracy import CASES, QTYPES, build_case, calibrate, evaluate
from check_mlx import load_act, cpu_act
from laya_mlx import LayaMLX, load_weights

LAYER_TYPES = ["full_attention" if i % 3 == 0 else "sliding_attention" for i in range(28)]
THETA = {"full_attention": 160000.0, "sliding_attention": 10000.0}


def main():
    import laya
    from huggingface_hub import snapshot_download
    from laya.common import build_sequence, collate_items
    mp = snapshot_download("convaiinnovations/laya", revision="c5d78730f3493e4fe16d61507ef4b78eef7318cf", local_files_only=True)
    with open(f"{mp}/rl_agent_config.json") as f:
        cfg = json.load(f)
    temp, temp_by = cfg["temperature"], cfg["temperature_by_options"]
    agent = laya.load(mp, device="cpu")
    load_act()
    model = LayaMLX(load_weights(), LAYER_TYPES, THETA)

    score, rows = 0, []
    for case in CASES:
        state, questions = build_case(case)
        ids = list(questions.keys())
        items = []
        for qid in ids:
            q = agent._to_internal(questions[qid])
            seq, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        b = collate_items([items], agent.tok.pad_token_id)
        feeds = {k: mx.array(b[k].numpy()) for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]}
        lo, pooled = model.forward(**feeds)
        mx.eval(lo, pooled)
        lo64 = np.asarray(lo, dtype=np.float64)
        mm = b["marker_mask"].numpy()
        act = cpu_act(lo64, mm, np.asarray(pooled, dtype=np.float64).astype(np.float32))
        _ = act  # action parity covered by check_mlx; accuracy uses option probs
        out = {}
        for r, qid in enumerate(ids):
            q = agent._to_internal(questions[qid])
            k = len(items[r]["markers"])
            p = calibrate(None, temp, temp_by, lo64[r], k, QTYPES[q["t"]])
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                out[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                            "probabilities": {kk: float(v) for kk, v in zip(keys, p)}}
            elif q["t"] == "score":
                out[qid] = {"type": "score", "score": float((np.arange(k) * p).sum()),
                            "probabilities": {str(i): float(v) for i, v in enumerate(p)}}
            else:
                out[qid] = {"type": "noul", "noul": float(p[1])}
        ok, margin, got = evaluate(out, case["check"])
        score += ok
        rows.append({"id": case["id"], "pass": bool(ok), "margin": round(float(margin), 4), "got": got})
        print(f"  {'PASS' if ok else 'FAIL'} {case['id']} margin={margin:.3f} got={got}", flush=True)
    print(f"mlx: {score}/{len(CASES)}", flush=True)
    if score != len(CASES):
        sys.exit(1)


if __name__ == "__main__":
    main()
