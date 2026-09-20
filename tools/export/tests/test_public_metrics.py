import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_public import ece


def test_ece_preserves_example_pairing():
    assert abs(ece([0.9, 0.1], [1, 0]) - 0.1) < 1e-12
    assert abs(ece([0.9, 0.1], [0, 1]) - 0.9) < 1e-12
