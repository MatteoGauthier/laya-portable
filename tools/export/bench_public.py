"""Public-dataset accuracy benchmark: torch vs our ONNX on labelled data.

AG News (4-way), DAIR Emotion (6-way), banking77 (77-way stress). Community
ONNX is interface-excluded (fixed K=2) — that exclusion IS the finding.
Compares against upstream BENCHMARKS.md published numbers where they exist.

Usage: .venv/bin/python tools/export/bench_public.py [--n-per-class N] [--banking-per-class N]
Output: packages/test-vectors/reports/public-benchmark.json
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "0")  # needs HF for public datasets (weights come from cache)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "upstream" / "laya"))
sys.path.insert(0, str(ROOT / "tools" / "export"))
import numpy as np

from check_accuracy import TorchAdapter, SplitOnnxAdapter


def sample_per_class(dataset_name, split, text_key, label_key, label_names, n_per_class, seed=0):
    from datasets import load_dataset
    quotas = {n: n_per_class for n in label_names}
    out = []
    ds = load_dataset(dataset_name, split=split, streaming=True)
    rng = __import__("random").Random(seed)
    buf = {}
    for ex in ds:
        lab = ex[label_key]
        name = label_names[lab] if isinstance(lab, int) else lab
        if len([o for o in out if o[1] == name]) >= quotas.get(name, n_per_class):
            continue
        out.append((ex[text_key], name))
        if all(len([o for o in out if o[1] == n]) >= quotas.get(n, n_per_class) for n in label_names):
            break
    rng.shuffle(out)
    return out


def banking_labels():
    from datasets import load_dataset
    seen = {}
    for ex in load_dataset("mteb/banking77", split="test", streaming=True):
        seen[ex["label_text"]] = ex["label"]
        if len(seen) >= 77:
            break
    return sorted(seen)


EMOTION_DESCR = {
    "sadness": "feeling down, sorrowful, unhappy",
    "joy": "feeling happy, delighted, cheerful",
    "love": "feeling affectionate, caring, tender",
    "anger": "feeling mad, furious, irritated",
    "fear": "feeling scared, anxious, afraid",
    "surprise": "feeling astonished by something unexpected",
}


def run_suite(name, samples, criteria, instructions, adapters, raw_state=False):
    questions = {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}}
    rows = {ad.name: {"n": 0, "correct": 0, "confs": [], "briers": [], "ms": []} for ad in adapters}
    for text, gold in samples:
        state = text if raw_state else {"text": text}
        for ad in adapters:
            t0 = time.perf_counter()
            ans = ad.predict(state, questions)["q"]
            ms = (time.perf_counter() - t0) * 1000
            # torch adapter returns {"answers":...}; onnx adapter returns flat dict — normalize
            r = rows[ad.name]
            r["n"] += 1
            r["ms"].append(ms)
            if "choice" in ans:
                pred = ans["choice"]
                probs = ans["probabilities"]
                conf = max(probs.values())
                brier = float(np.mean([(probs.get(k, 0.0) - (1.0 if k == gold else 0.0)) ** 2 for k in criteria]))
            else:
                pred, conf, brier = None, 0.0, 1.0
            hit = pred == gold
            r["correct"] += hit
            r["confs"].append(conf)
            r["briers"].append(brier)
    return rows


def ece(confs, hits, bins=15):
    confs = np.array(confs)
    hits = np.array(hits, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (confs > lo) & (confs <= hi)
        if sel.any():
            e += sel.mean() * abs(confs[sel].mean() - hits[sel].mean())
    return float(e)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--banking-per-class", type=int, default=2)
    ap.add_argument("--models", nargs="*", default=["models/laya-split-single.onnx", "models/laya-split-fp16.onnx"])
    ap.add_argument("--describe-emotion", action="store_true", help="Use one-line criteria descriptions for the emotion suite")
    ap.add_argument("--emotion-only", action="store_true", help="Run only the emotion suite")
    ap.add_argument("--emotion-instruction", default=None, help="Override the emotion suite instruction")
    ap.add_argument("--raw-state", action="store_true", help="Pass state as a raw string instead of {\"text\": ...}")
    ap.add_argument("--output", type=Path, default=ROOT / "packages" / "test-vectors" / "reports" / "public-benchmark.json")
    args = ap.parse_args()

    print("loading adapters...", flush=True)
    adapters = [TorchAdapter()] + [SplitOnnxAdapter(f"onnx-{Path(m).stem}", ROOT / m) for m in args.models]

    report = {"suites": {}, "community": "interface-excluded (fixed K=2; all suites need K>=4)"}
    suites = [
        ("ag_news", "fancyzhx/ag_news", "test", ["World", "Sports", "Business", "Sci/Tech"],
         "Which section does this news article belong to?", args.n_per_class, None),
        ("emotion", "dair-ai/emotion", "test", ["sadness", "joy", "love", "anger", "fear", "surprise"],
         args.emotion_instruction or "Which emotion does this message express?", max(10, args.n_per_class // 2),
         EMOTION_DESCR if args.describe_emotion else None),
    ]
    if args.emotion_only:
        suites = [s for s in suites if s[0] == "emotion"]
        report["suites"] = {}
    for sname, ds, split, labels, ins, npc, descr in suites:
        print(f"sampling {sname} ({len(labels)} classes x {npc})...", flush=True)
        samples = sample_per_class(ds, split, "text", "label", labels, npc)
        print(f"  got {len(samples)}, running {len(adapters)} adapters...", flush=True)
        rows = run_suite(sname, samples, dict(descr) if descr else {l: None for l in labels}, ins, adapters,
                         raw_state=args.raw_state)
        report["suites"][sname] = {"n": len(samples), "labels": labels, "adapters": {}}
        for aname, r in rows.items():
            hits = [1] * r["correct"] + [0] * (r["n"] - r["correct"])
            acc = r["correct"] / r["n"]
            report["suites"][sname]["adapters"][aname] = {
                "accuracy": round(acc, 4), "ece": round(ece(r["confs"], hits), 4),
                "brier": round(float(np.mean(r["briers"])), 4),
                "p50_ms": round(float(np.median(r["ms"])), 1)}
            print(f"  {aname}: acc={acc:.3f} ece={report['suites'][sname]['adapters'][aname]['ece']:.3f} p50={np.median(r['ms']):.0f}ms", flush=True)

    if args.emotion_only:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
        return

    print("resolving banking77 labels...", flush=True)
    blabels = banking_labels()
    print(f"  {len(blabels)} intents", flush=True)
    from datasets import load_dataset
    samples = []
    quotas = {l: args.banking_per_class for l in blabels}
    counts = {l: 0 for l in blabels}
    for ex in load_dataset("mteb/banking77", split="test", streaming=True):
        lt = ex["label_text"]
        if counts.get(lt, 0) >= quotas.get(lt, 0):
            continue
        samples.append((ex["text"], lt))
        counts[lt] += 1
        if all(counts[l] >= quotas[l] for l in blabels):
            break
    import random as _r
    _r.Random(0).shuffle(samples)
    print(f"  got {len(samples)}, running...", flush=True)
    rows = run_suite("banking77", samples, {l: None for l in blabels},
                     "What is the customer's banking intent?", adapters)
    report["suites"]["banking77"] = {"n": len(samples), "n_classes": len(blabels), "adapters": {}}
    for aname, r in rows.items():
        hits = [1] * r["correct"] + [0] * (r["n"] - r["correct"])
        acc = r["correct"] / r["n"]
        report["suites"]["banking77"]["adapters"][aname] = {
            "accuracy": round(acc, 4), "ece": round(ece(r["confs"], hits), 4),
            "brier": round(float(np.mean(r["briers"])), 4),
            "p50_ms": round(float(np.median(r["ms"])), 1)}
        print(f"  {aname}: acc={acc:.3f} ece={report['suites']['banking77']['adapters'][aname]['ece']:.3f} p50={np.median(r['ms']):.0f}ms", flush=True)

    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
