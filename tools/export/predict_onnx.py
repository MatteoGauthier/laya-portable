"""Minimal user-facing inference: portable split ONNX + numpy calibration.

Tokenizes with the reference implementation, runs the split graph in
onnxruntime, calibrates with the checkpoint temperature tables — no torch
model load. Mirrors packages/laya-js/src/laya.ts predict().

Usage:
  .venv/bin/python tools/export/predict_onnx.py [--model models/laya-split-single.onnx]
      [--state '{"subject":".."}'] [--questions '{...}']
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "packages" / "test-vectors" / "vectors"
sys.path.insert(0, str(ROOT / "upstream" / "laya"))

REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
DEFAULT_STATE = {"subject": "Duplicate charge on invoice 4411",
                 "body": "We were billed twice for March. Please refund the duplicate."}
DEFAULT_QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs and outages", "sales": "pricing"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"}}

# CPU action head (same math as packages/laya-js/src/laya-action.ts).
def cpu_act(logits64, mm, pooled32, W0, b0, W2, b2):
    import math
    import numpy as np
    p = np.exp(logits64 - logits64.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)
    k = mm.sum(axis=1).clip(min=2).astype(np.float64)
    ent = -(p * np.log(np.clip(p, 1e-9, 1))).sum(axis=1) / np.log(k)
    top2 = np.sort(p, axis=1)[:, -2:][:, ::-1]
    feats = np.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], axis=1).astype(np.float32)
    h = np.concatenate([pooled32, feats], axis=1) @ W0.T + b0
    h = 0.5 * h * (1 + np.array([[math.erf(v / 1.4142135623730951) for v in row] for row in h],
                                dtype=np.float32))
    return h @ W2.T + b2


def main():
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from laya.common import QTYPES, build_sequence, collate_items, confidence_from_probs, temp_bucket

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "laya-split-single.onnx")
    ap.add_argument("--state", default=json.dumps(DEFAULT_STATE))
    ap.add_argument("--questions", default=json.dumps(DEFAULT_QUESTIONS))
    args = ap.parse_args()
    state = json.loads(args.state)
    questions = json.loads(args.questions)

    snap = Path(snapshot_download("convaiinnovations/laya", revision=REVISION, local_files_only=True))
    tok = AutoTokenizer.from_pretrained(str(snap / "tokenizer"))
    cfg = json.loads((snap / "rl_agent_config.json").read_text())
    ah = np.load(ROOT / "tools" / "export" / "act_head.npz")
    W0, b0, W2, b2 = (ah["0.weight"], ah["0.bias"], ah["2.weight"], ah["2.bias"])

    # Upstream build_sequence takes the HF tokenizer directly (callable +
    # mask/cls/sep/pad attributes); no torch model load needed.
    ids = list(questions.keys())
    items, qts = [], []
    for qid in ids:
        q = questions[qid]
        t = q["type"]
        crit = q.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {str(c): None for c in crit}
        ins = q.get("instructions", "")
        internal = {"t": t, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": crit or {}}
        seq, markers = build_sequence(tok, state, internal, cfg["max_len"], cfg["head_max_len"])
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[t]})
        qts.append(QTYPES[t])

    b = collate_items([items], tok.pad_token_id)
    feeds = {k: b[k].numpy() for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]}
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    logits, pooled = sess.run(None, feeds)
    mm = feeds["marker_mask"]
    act = cpu_act(logits.astype(np.float64), mm, pooled.astype(np.float32), W0, b0, W2, b2)
    act_p = np.exp(act - act.max(axis=1, keepdims=True))
    act_p = act_p / act_p.sum(axis=1, keepdims=True)

    answers = {}
    for r, qid in enumerate(ids):
        q = questions[qid]
        k = int(mm[r].sum())
        qt = qts[r]
        ts = cfg["temperature_by_options"].get(f"{['choice', 'score', 'noul'][qt]}:"
                                               f"{'2' if k <= 2 else '3-5' if k <= 5 else '6-10' if k <= 10 else '11+'}",
                                               cfg["temperature"][qt])
        z = logits[r, :k] / max(1e-3, float(ts))
        p = np.exp(z - z.max())
        p = p / p.sum()
        ext = {"act_probability": round(float(act_p[r, 0]), 4)}
        if q["type"] == "choice":
            keys = list(q["criteria"].keys())
            answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                            "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                            "confidence": round(float(confidence_from_probs(p, k)), 4), "action": ext}
        elif q["type"] == "score":
            exp = float((np.arange(k) * p).sum())
            answers[qid] = {"type": "score", "score": round(exp, 4),
                            "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                            "confidence": round(float(confidence_from_probs(p, k)), 4), "action": ext}
        else:
            n = float(p[1])
            answers[qid] = {"type": "noul", "noul": round(n, 4),
                            "confidence": round(max(n, 1 - n), 4), "action": ext}
    print(json.dumps({"model": "laya-rl-agent", "answers": answers}, indent=1))


if __name__ == "__main__":
    main()
