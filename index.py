"""Pinned demo: same 3-question example as benchmark.py."""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import laya
from huggingface_hub import snapshot_download

REVISION = "c5d78730f3493e4fe16d61507ef4b78eef7318cf"

agent = laya.load(snapshot_download("convaiinnovations/laya", revision=REVISION, local_files_only=True))
result = agent.predict(
    {"subject": "Duplicate charge on invoice 4411",
     "body": "We were billed twice for March. Please refund the duplicate."},
    {"department": {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "invoices, payments, refunds",
                                 "technical": "bugs and outages", "sales": "pricing"}},
     "urgency": {"type": "score", "instructions": "How urgent is this?",
                 "criteria": ["not urgent", "soon", "blocking"]},
     "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel?"}},
)
print(result["answers"]["department"]["choice"])
