"""CI-friendly report assertions (no model load, no network).

Asserts committed export/*.json reports meet thresholds so GitHub Actions
can gate without rebuilding the 1.6GB model.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    p = ROOT / "export" / name
    assert p.exists(), f"missing {p} — run export/check_parity.py first"
    return json.loads(p.read_text())


def test_parity_overall_pass():
    rep = load("parity-report.json")
    assert rep.get("overall_pass") is True, "parity-report overall_pass must be true"
    assert len(rep["fixtures"]) == 5
    for f in rep["fixtures"]:
        assert f["max_abs_logits"] < 1e-4, f"{f['name']} logits drift"
        assert f["max_calibrated_prob_drift"] < 1e-4, f"{f['name']} prob drift"
        assert f["labels_match"] is True, f"{f['name']} labels"


def test_accuracy_thresholds():
    p = ROOT / "export" / "accuracy-report.json"
    if not p.exists():
        import pytest
        pytest.skip("accuracy-report.json not committed")
        return
    rep = json.loads(p.read_text())
    for a in rep["adapters"]:
        # committed baseline: torch 13/13; quantized variants documented as rejected
        if a["name"] == "torch-fp32":
            assert a["score"] == rep["cases"], "torch baseline must be perfect"


def test_metadata_pins():
    meta = load("laya-faithful-metadata.json")
    assert meta["opset"] == 18
    assert meta["model_revision"] == "c5d78730f3493e4fe16d61507ef4b78eef7318cf"
    # no absolute user paths in committed metadata
    assert "/Users/" not in json.dumps(meta), "absolute path leaked into metadata"
