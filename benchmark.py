"""Small, offline CPU/MPS baseline for the original three-question example.

Uses the checked-out upstream implementation and an already downloaded checkpoint.
This measures latency, not model quality or equivalence of alternative runtimes.
"""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "upstream" / "laya"))

import numpy as np
import torch
import laya
from huggingface_hub import snapshot_download
from laya.common import QTYPES, build_sequence, collate_items

REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
STATE = {"subject": "Duplicate charge on invoice 4411",
         "body": "We were billed twice for March. Please refund the duplicate."}
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs and outages", "sales": "pricing"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"},
}


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def measure(fn, device, warmup, repeats):
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95)), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Local checkpoint directory; otherwise use pinned HF cache")
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps"], default=["cpu", "mps"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "packages" / "test-vectors" / "reports" / "mac-baseline.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    model_path = args.model or snapshot_download(
        "convaiinnovations/laya", revision=REVISION, local_files_only=True)
    report = {
        "platform": platform.platform(), "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ["torch", "transformers", "laya", "numpy"]},
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "upstream" / "laya"), "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": REVISION if not args.model else None,
        "mps_available": torch.backends.mps.is_available(),
        "cpu_threads": torch.get_num_threads(), "warmup": args.warmup,
        "repeats": args.repeats, "state": STATE, "questions": QUESTIONS, "runs": [],
    }
    for device in args.devices:
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; refusing to report a CPU fallback as GPU performance")
        for count in [1, 3]:
            start = time.perf_counter()
            agent = laya.load(model_path, device=device)
            synchronize(device)
            load_seconds = time.perf_counter() - start
            if str(agent.device) != device:
                raise RuntimeError(f"Requested {device}; got {agent.device}")
            questions = dict(list(QUESTIONS.items())[:count])
            items = []
            for definition in questions.values():
                q = agent._to_internal(definition)
                ids, markers = build_sequence(agent.tok, STATE, q,
                                              agent.cfg["max_len"], agent.cfg["head_max_len"])
                items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            batch = collate_items([items], agent.tok.pad_token_id)
            inputs = [batch[key].to(device) for key in
                      ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]]
            with torch.no_grad():
                forward = measure(lambda: agent.model(*inputs), device, args.warmup, args.repeats)
                total = measure(lambda: agent.predict(STATE, questions), device, args.warmup, args.repeats)
                answer = agent.predict(STATE, questions)
            if str(agent.device) != device:
                raise RuntimeError("Device changed during inference; results invalid")
            run = {"device": device, "dtype": str(next(agent.model.parameters()).dtype),
                   "questions": count, "input_shape": list(inputs[0].shape),
                   "load_seconds": load_seconds, "forward": forward,
                   "end_to_end": total, "answer": answer}
            report["runs"].append(run)
            print(f"{device} {count} questions: forward {forward['p50_ms']:.1f} ms; "
                  f"end-to-end {total['p50_ms']:.1f} ms", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            del agent, inputs, batch
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
