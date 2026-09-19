"""Emit fuzz strings + Python token IDs for pure-JS BPE validation."""
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "upstream" / "laya"))
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
SNAP = "/Users/matteolemni/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/c5d78730f3493e4fe16d61507ef4b78eef7318cf"
tok = AutoTokenizer.from_pretrained(f"{SNAP}/tokenizer")

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
(ROOT / "js" / "bpe-fuzz.json").write_text(json.dumps(out, ensure_ascii=False) + "\n")
print(f"wrote {len(out)} cases")
