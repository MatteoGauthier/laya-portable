"""Parity: PyTorch (pinned upstream) vs faithful ONNX (ORT CPU).

Compares raw logits, act logits, calibrated probabilities, scores, action
probs and confidence -- not just winning labels. Saves fixtures + report.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
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

def postprocess(agent, logits_np, act_np, items, ids, questions):
    """Mirror Agent.system_one calibration exactly, given raw logits."""
    from laya.common import QTYPES, confidence_from_probs, temp_bucket
    import numpy as np
    act_probs = np.exp(act_np - act_np.max(axis=1, keepdims=True))
    act_probs = act_probs / act_probs.sum(axis=1, keepdims=True)
    out = {}
    for r, qid in enumerate(ids):
        q = agent._to_internal(questions[qid])
        k = len(items[r]["markers"])
        qt = QTYPES[q["t"]]
        t_scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
        z = logits_np[r, :k] / max(1e-3, float(t_scale))
        p = np.exp(z - z.max()); p = p / p.sum()
        conf = round(float(confidence_from_probs(p, k)), 4) if q["t"] != "noul" else round(float(max(p[1], 1-p[1])), 4)
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

    model_path = snapshot_download("convaiinnovations/laya", revision=REVISION, local_files_only=True)
    print(f"loading torch {model_path} ...", flush=True)
    agent = laya.load(model_path, device="cpu")
    assert str(agent.device) == "cpu"
    agent.model.eval()

    onnx_path = ROOT/"models"/"laya-faithful.onnx"
    print(f"loading onnx {onnx_path} ...", flush=True)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"ort {ort.__version__} inputs {[i.name for i in sess.get_inputs()]}", flush=True)

    report = {
        "model_revision": REVISION,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT/"upstream"/"laya"), "rev-parse", "HEAD"], text=True).strip(),
        "onnx": str(onnx_path),
        "ort_version": ort.__version__,
        "torch_version": torch.__version__,
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
        # ORT expects bool for marker_mask; collate already bool
        o_logits, o_act = sess.run(None, feeds_ort)

        # raw diffs (only valid options for logits)
        K = b["marker_mask"].numpy()
        diff_logits = np.abs(t_logits_np - o_logits)[K].max() if K.any() else float(np.abs(t_logits_np - o_logits).max())
        mean_logits = float(np.abs(t_logits_np - o_logits)[K].mean()) if K.any() else 0.0
        diff_act = float(np.abs(t_act_np - o_act).max())
        mean_act = float(np.abs(t_act_np - o_act).mean())

        torch_post, torch_actp = postprocess(agent, t_logits_np, t_act_np, items, ids, questions)
        onnx_post, onnx_actp = postprocess(agent, o_logits, o_act, items, ids, questions)

        # calibrated prob drift
        max_pdrift = 0.0
        for qid in ids:
            tp, op = torch_post[qid], onnx_post[qid]
            if "probs" in tp:
                for k in tp["probs"]:
                    max_pdrift = max(max_pdrift, abs(tp["probs"][k] - op["probs"][k]))
            else:
                max_pdrift = max(max_pdrift, abs(tp["noul"] - op["noul"]))
        max_adrift = float(np.abs(torch_actp[:, 0] - onnx_actp[:, 0]).max())

        # label/score agreement at 4-dec (like baseline) + exact choice
        labels_match = all(
            (torch_post[q].get("choice") == onnx_post[q].get("choice"))
            if "choice" in torch_post[q] else True for q in ids
        )
        # full 4-dec answer equality (rounded as in benchmark, excluding raw debug)
        def rounded(d):
            import copy
            r = copy.deepcopy(d)
            for q in r.values():
                q.pop("raw_logits", None)
                if "probs" in q:
                    q["probs"] = {k: round(v, 4) for k, v in q["probs"].items()}
                if "score" in q: q["score"] = round(q["score"], 4)
                if "noul" in q: q["noul"] = round(q["noul"], 4)
            return r
        rounded_match = rounded(torch_post) == rounded(onnx_post)

        # Act logits saturate (~4000); absolute 1e-4 is too strict. Use relative + prob drift.
        # Logits are O(1-10), so absolute threshold is meaningful there.
        # Rounded 4-dec equality is informational only: 1e-6 drift can flip the 4th
        # decimal at a rounding boundary (observed in mixed-batch c_score2).
        # Gate on full-precision drift + labels, not on rounding.
        max_rel_act = float((np.abs(t_act_np - o_act) / np.maximum(1.0, np.abs(t_act_np))).max())
        ok = diff_logits < 1e-4 and max_pdrift < 1e-4 and max_adrift < 1e-4 and labels_match
        all_ok = all_ok and ok
        print(f"{name}: B={len(ids)} Kmax={K.shape[1]} S={b['input_ids'].shape[1]} "
              f"logits max={diff_logits:.2e} mean={mean_logits:.2e} act max={diff_act:.2e} rel={max_rel_act:.2e} "
              f"pdrift={max_pdrift:.2e} adrift={max_adrift:.2e} labels={'OK' if labels_match else 'DIFF'} "
              f"rounded={'OK' if rounded_match else 'DIFF'} -> {'PASS' if ok else 'CHECK'}", flush=True)
        report["fixtures"].append({
            "name": name, "batch": len(ids), "seq_len": int(b["input_ids"].shape[1]),
            "kmax": int(K.shape[1]), "k_per_q": [len(it["markers"]) for it in items],
            "qtypes": [it["qtype"] for it in items],
            "max_abs_logits": float(diff_logits), "mean_abs_logits": mean_logits,
            "max_abs_act_logits": diff_act, "mean_abs_act_logits": mean_act,
            "max_rel_act_logits": max_rel_act,
            "max_calibrated_prob_drift": float(max_pdrift),
            "max_act_prob_drift": max_adrift,
            "labels_match": labels_match, "rounded_4dec_match": rounded_match,
            "torch": torch_post, "onnx": onnx_post,
        })
        # save raw inputs for TS port / debugging
        fx_dir = ROOT/"export"/"fixtures"
        fx_dir.mkdir(parents=True, exist_ok=True)
        np.savez(fx_dir/f"{name}.npz",
                 input_ids=feeds_ort["input_ids"], attention_mask=feeds_ort["attention_mask"],
                 marker_pos=feeds_ort["marker_pos"], marker_mask=feeds_ort["marker_mask"],
                 qtype=feeds_ort["qtype"], torch_logits=t_logits_np, torch_act=t_act_np,
                 onnx_logits=o_logits, onnx_act=o_act)

    report["overall_pass"] = bool(all_ok)
    out = ROOT/"export"/"parity-report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out} overall={'PASS' if all_ok else 'CHECK DETAILS'}", flush=True)

if __name__ == "__main__":
    main()
