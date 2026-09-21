"""Emit fuzz strings + Python token IDs for pure-JS BPE validation.

Usage: .venv/bin/python export/emit_bpe_fuzz.py [--revision REV]
"""
import argparse
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "upstream" / "laya"))
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
ap = argparse.ArgumentParser()
ap.add_argument("--revision", default=REVISION)
ap.add_argument("--subfolder", default=None, help="None (english root), multilingual, typed-decisions")
ap.add_argument("--out", type=Path, default=None, help="Output JSON (default vectors/bpe-fuzz[<.subfolder>].json)")
ns = ap.parse_known_args()[0]
SNAP = Path(snapshot_download("convaiinnovations/laya", revision=ns.revision, local_files_only=True))
TOKDIR = SNAP / ns.subfolder / "tokenizer" if ns.subfolder else SNAP / "tokenizer"
tok = AutoTokenizer.from_pretrained(str(TOKDIR))

random.seed(42)
cases = [
    "", " ", "  ", "a", "hi", "Hello, world!", "choice question: hi",
    "'s 't 're 've 'm 'll 'd", "don't can't won't", "  leading and trailing  ",
    "multiple   spaces    here", "\n", "\t", "line1\nline2\ttab",
    "münchen", "naïve café", "日本語テスト", "한국어", "العربية", "हिन्दी", "😀", "😀😀",
    "e\u0301", "é", "Å", "ﬁ", '{"subject": "hi", "n": 123, "b": true, "x": null}',
    "[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]", "[CLS] hello [SEP]",
    "|||IP_ADDRESS|||", "|||EMAIL_ADDRESS|||", "<|padding|>", "<|endoftext|>",
    "[unused0]", "[unused82]", " " * 24, " " * 25, "x" + " " * 30 + "y",
    "level 0: low", "billing: invoices, payments, refunds",
    "false: no, the statement does not hold", "noul question: Does it hold?",
]
# Random ASCII/unicode soup
alphabet = list("abcXYZ019 ,.!?'-\n\t") + ["é", "日", "😀", " ", "  "]
for _ in range(120):
    n = random.randint(0, 60)
    cases.append("".join(random.choice(alphabet) for _ in range(n)))
# Random with occasional added tokens
added = ["[MASK]", "[CLS]", "|||IP_ADDRESS|||", "[unused5]", " " * 20]
for _ in range(40):
    parts = ["".join(random.choice(alphabet) for _ in range(random.randint(0, 20)))]
    if random.random() < 0.7:
        parts.append(random.choice(added))
        parts.append("".join(random.choice(alphabet) for _ in range(random.randint(0, 20))))
    cases.append("".join(parts))

out = []
for s in cases:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    out.append({"text": s, "ids": ids})
suffix = f".{ns.subfolder}" if ns.subfolder else ""
dest = ns.out or (ROOT / "packages" / "test-vectors" / "vectors" / f"bpe-fuzz{suffix}.json")
dest.write_text(json.dumps(out, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} cases -> {dest}")
