"""Split parity per checkpoint: torch full DecisionModel vs split ONNX + numpy head.

B2 ships split graphs (logits+pooled); there are no faithful (act_logits)
graphs for multilingual/typed-decisions, so check_parity.py cannot gate them.
This script is the split equivalent: same 5 fixtures, same thresholds
(logits<1e-4, calibrated pdrift<1e-4, act prob drift<1e-4, labels match).

The numpy action head takes pooled dim from the weights (1024 ModernBERT-large,
768 mmBERT-base) — never hardcoded. Saves per-checkpoint fixtures as
vectors/<name>.<checkpoint>.npz for the fp16/quant gates, and
reports/parity-split-<checkpoint>.json.

Usage:
  .venv/bin/python tools/export/check_split_parity.py --subfolder multilingual \\
      --onnx models/laya-multilingual-split-single.onnx \\
      --npz tools/export/act_head.multilingual.npz
  (omit --subfolder for the english root)
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "packages" / "test-vectors" / "vectors"
REPORTS = VECTORS.parent / "reports"
sys.path.insert(0, str(ROOT / "upstream" / "laya"))

REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
STATE = {"subject": "Duplicate charge on invoice 4411",
         "body": "We were billed twice for March. Please refund the duplicate."}


def fixtures():
    base_choice3 = {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "invoices, payments, refunds",
                                 "technical": "bugs and outages", "sales": "pricing"}}
    base_score3 = {"type": "score", "instructions": "How urgent is this?",
                   "criteria": ["not urgent", "soon", "blocking"]}
    base_noul = {"type": "noul", "instructions": "Does the user threaten to cancel?"}
    return [
        ("orig-3q", {"department": base_choice3, "urgency": base_score3, "churn_risk": base_noul}),
        ("choice-3", {"department": base_choice3}),
        ("choice-2", {"dept2": {"type": "choice", "instructions": "Billing or tech?",
                                "criteria": {"billing": "invoices", "technical": "bugs"}}}),
        ("choice-6", {"dept6": {"type": "choice", "instructions": "Route it",
                                "criteria": {f"team{i}": f"area {i}" for i in range(6)}}}),
        ("mixed-batch", {"a_choice5": {"type": "choice", "instructions": "Pick",
                                       "criteria": {f"o{i}": f"desc {i}" for i in range(5)}},
                          "b_noul": base_noul,
                          "c_score2": {"type": "score", "instructions": "Rate",
                                       "criteria": ["low", "high"]}}),
    ]


def cpu_act(logits64, mm, pooled32, W0, b0, W2, b2):
    """Numpy mirror of the split action head (pooled dim from weights)."""
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


def postprocess(agent, logits_np, act_np, items, ids, questions):
    from laya.common import QTYPES, confidence_from_probs, temp_bucket
    act_probs = np.exp(act_np - act_np.max(axis=1, keepdims=True))
    act_probs = act_probs / act_probs.sum(axis=1, keepdims=True)
    out = {}
    for r, qid in enumerate(ids):
        q = agent._to_internal(questions[qid])
        k = len(items[r]["markers"])
        qt = QTYPES[q["t"]]
        t_scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
        z = logits_np[r, :k] / max(1e-3, float(t_scale))
        p = np.exp(z - z.max())
        p = p / p.sum()
        conf = round(float(confidence_from_probs(p, k)), 4) if q["t"] != "noul" else round(float(max(p[1], 1 - p[1])), 4)
        ext = round(float(act_probs[r, 0]), 4)
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            out[qid] = {"probs": {kk: round(float(v), 6) for kk, v in zip(keys, p)},
                        "choice": keys[int(p.argmax())], "confidence": conf, "act": ext,
                        "raw_logits": [float(v) for v in logits_np[r, :k]]}
        elif q["t"] == "score":
            exp = float((np.arange(k) * p).sum())
            out[qid] = {"probs": {str(i): round(float(v), 6) for i, v in enumerate(p)},
                        "score": round(exp, 6), "confidence": conf, "act": ext,
                        "raw_logits": [float(v) for v in logits_np[r, :k]]}
        else:
            out[qid] = {"noul": round(float(p[1]), 6), "confidence": conf, "act": ext,
                        "raw_logits": [float(v) for v in logits_np[r, :k]]}
    return out, act_probs


def main():
    import torch
    import laya
    from huggingface_hub import snapshot_download
    from laya.common import QTYPES, build_sequence, collate_items
    import onnxruntime as ort

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="convaiinnovations/laya")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--subfolder", default=None,
                    help="None (english root), multilingual, typed-decisions")
    ap.add_argument("--model", default=None, help="Local checkpoint dir override")
    ap.add_argument("--onnx", type=Path, required=True, help="Split single-file ONNX to gate")
    ap.add_argument("--npz", type=Path, required=True, help="Act-head weights for the numpy head")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    ckpt = args.subfolder or "english"
    if args.model:
        model_path = args.model
    else:
        base = snapshot_download(args.repo, revision=args.revision, local_files_only=True)
        model_path = str(Path(base) / args.subfolder) if args.subfolder else base

    print(f"loading torch {model_path} ...", flush=True)
    agent = laya.load(model_path, device="cpu")
    assert str(agent.device) == "cpu"
    agent.model.eval()

    ah = np.load(args.npz)
    W0, b0, W2, b2 = (ah["0.weight"].astype(np.float32), ah["0.bias"].astype(np.float32),
                      ah["2.weight"].astype(np.float32), ah["2.bias"].astype(np.float32))
    pooled_dim = W0.shape[1] - 4
    print(f"act head pooled_dim={pooled_dim} (w0 {list(W0.shape)})", flush=True)

    print(f"loading onnx {args.onnx} ...", flush=True)
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    assert [o.name for o in sess.get_outputs()] == ["logits", "pooled"], \
        [o.name for o in sess.get_outputs()]

    report = {
        "checkpoint": ckpt,
        "model_revision": args.revision,
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "upstream" / "laya"), "rev-parse", "HEAD"], text=True).strip(),
        "onnx": str(args.onnx.relative_to(ROOT)) if str(args.onnx).startswith(str(ROOT)) else str(args.onnx),
        "ort_version": ort.__version__,
        "torch_version": torch.__version__,
        "pooled_dim": pooled_dim,
        "fixtures": [],
    }
    all_ok = True
    for name, questions in fixtures():
        ids = list(questions.keys())
        items = []
        for qid in ids:
            q = agent._to_internal(questions[qid])
            seq, markers = build_sequence(agent.tok, STATE, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
        b = collate_items([items], agent.tok.pad_token_id)
        feeds_torch = [b[k] for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]]
        with torch.no_grad():
            t_logits, t_act = agent.model(*feeds_torch)
        t_logits_np = t_logits.float().numpy()
        t_act_np = t_act.float().numpy()

        feeds_ort = {k: b[k].numpy() for k in ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]}
        o_logits, o_pooled = sess.run(None, feeds_ort)
        assert o_pooled.shape[-1] == pooled_dim, (o_pooled.shape, pooled_dim)
        o_act = cpu_act(o_logits.astype(np.float64), feeds_ort["marker_mask"],
                        o_pooled.astype(np.float32), W0, b0, W2, b2)

        K = b["marker_mask"].numpy()
        diff_logits = float(np.abs(t_logits_np - o_logits)[K].max())
        mean_logits = float(np.abs(t_logits_np - o_logits)[K].mean())
        diff_act = float(np.abs(t_act_np - o_act).max())
        max_rel_act = float((np.abs(t_act_np - o_act) / np.maximum(1.0, np.abs(t_act_np))).max())

        torch_post, torch_actp = postprocess(agent, t_logits_np, t_act_np, items, ids, questions)
        onnx_post, onnx_actp = postprocess(agent, o_logits, o_act, items, ids, questions)

        max_pdrift = 0.0
        for qid in ids:
            tp, op = torch_post[qid], onnx_post[qid]
            if "probs" in tp:
                for k in tp["probs"]:
                    max_pdrift = max(max_pdrift, abs(tp["probs"][k] - op["probs"][k]))
            else:
                max_pdrift = max(max_pdrift, abs(tp["noul"] - op["noul"]))
        max_adrift = float(np.abs(torch_actp[:, 0] - onnx_actp[:, 0]).max())

        labels_match = all(
            (torch_post[q].get("choice") == onnx_post[q].get("choice"))
            if "choice" in torch_post[q] else True for q in ids
        )
        ok = diff_logits < 1e-4 and max_pdrift < 1e-4 and max_adrift < 1e-4 and labels_match
        all_ok = all_ok and ok
        print(f"{name}: S={b['input_ids'].shape[1]} logits max={diff_logits:.2e} mean={mean_logits:.2e} "
              f"act max={diff_act:.2e} rel={max_rel_act:.2e} pdrift={max_pdrift:.2e} adrift={max_adrift:.2e} "
              f"labels={'OK' if labels_match else 'DIFF'} -> {'PASS' if ok else 'CHECK'}", flush=True)
        report["fixtures"].append({
            "name": name, "batch": len(ids), "seq_len": int(b["input_ids"].shape[1]),
            "kmax": int(K.shape[1]), "k_per_q": [len(it["markers"]) for it in items],
            "qtypes": [it["qtype"] for it in items],
            "max_abs_logits": float(diff_logits), "mean_abs_logits": mean_logits,
            "max_abs_act_logits": diff_act, "max_rel_act_logits": max_rel_act,
            "max_calibrated_prob_drift": float(max_pdrift),
            "max_act_prob_drift": max_adrift,
            "labels_match": labels_match,
            "torch": torch_post, "onnx": onnx_post,
        })
        np.savez(VECTORS / f"{name}.{ckpt}.npz",
                 input_ids=feeds_ort["input_ids"], attention_mask=feeds_ort["attention_mask"],
                 marker_pos=feeds_ort["marker_pos"], marker_mask=feeds_ort["marker_mask"],
                 qtype=feeds_ort["qtype"], torch_logits=t_logits_np, torch_act=t_act_np,
                 onnx_logits=o_logits, onnx_pooled=o_pooled, onnx_act=o_act)

    report["overall_pass"] = bool(all_ok)
    out = args.report or (REPORTS / f"parity-split-{ckpt}.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out} overall={'PASS' if all_ok else 'CHECK DETAILS'}", flush=True)
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
