"""Convert split single-file FP32 to FP16 (internal only, IO stays FP32).

Standard ORT float16 conversion: weights + compute in FP16, inputs/outputs
FP32 (keep_io_types) so the JS/Python contract is unchanged. Action head
stays FP32 on CPU/JS; only the GPU graph is quantized.

Usage: .venv/bin/python export/to_fp16.py [--src ...] [--dst ...]
"""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import onnx
from onnxconverter_common import float16

def _instrument(float16):
    """Wrap slow converter stages with stage + per-initializer progress logs.

    The converter emits nothing; on a 1.6GB graph the silent stages run 10+ min
    each. Wrappers resolve through the module namespace, so patching the
    attributes affects convert_float_to_float16's internal calls.
    """
    import time

    def timed(name, fn):
        def wrap(*a, **k):
            print(f"[fp16] {name}... ", flush=True)
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                print(f"[fp16] {name} done in {(time.perf_counter()-t0)/60:.1f}min", flush=True)
        return wrap

    for stage in ["initial_checking", "process_tensor_in_node", "process_node_in_block_list",
                  "process_graph_output", "sort_topology", "remove_unnecessary_cast_node"]:
        setattr(float16, stage, timed(stage, getattr(float16, stage)))

    orig_convert_tensor = float16.convert_tensor_float_to_float16
    state = {"n": 0, "total": 0}

    def counting_convert(initializer, *a, **k):
        from onnx import TensorProto
        nbytes = 0
        if initializer.data_type == TensorProto.FLOAT:
            nbytes = len(initializer.raw_data) or len(initializer.float_data) * 4
        state["n"] += 1
        state["total"] += nbytes
        if nbytes > 10_000_000 or state["n"] % 50 == 0:
            print(f"[fp16] tensors: {state['n']} ({state['total']/1e9:.2f}GB converted)", flush=True)
        return orig_convert_tensor(initializer, *a, **k)

    float16.convert_tensor_float_to_float16 = counting_convert
    # process_initializers looks up convert_tensor_float_to_float16 by module
    # global at call time, so patch the reference it sees:
    import onnxconverter_common.float16 as _m
    _m.convert_tensor_float_to_float16 = counting_convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "models" / "laya-split-single.onnx")
    ap.add_argument("--dst", type=Path, default=ROOT / "models" / "laya-split-fp16.onnx")
    ap.add_argument("--keep-softmax", action="store_true",
                    help="Keep all Softmax nodes in FP32 (attention probabilities are drift-sensitive)")
    ap.add_argument("--keep-scorer", action="store_true",
                    help="Keep the scorer tail MatMuls in FP32 (final decision layer)")
    ap.add_argument("--progress", action="store_true",
                    help="Log converter stages and per-initializer progress (monkeypatches float16 helpers)")
    args = ap.parse_args()
    src, dst = args.src, args.dst
    print(f"loading {src} ({src.stat().st_size/1e9:.2f} GB)...", flush=True)
    m = onnx.load(str(src))
    ops = ['Cast']
    nodes = []
    if args.keep_softmax:
        ops.append('Softmax')
    if args.keep_scorer:
        # scorer = final Linear(d,d)+Linear(d,1) pair (see check: 2 Gemm are head out-projs;
        # scorer linears exported as tail MatMuls)
        tail = [n.name for n in m.graph.node if n.op_type == "MatMul"][-4:]
        nodes.extend(tail)
        print(f"keeping scorer nodes FP32: {tail}", flush=True)
    print(f"converting to float16 (keep_io_types=True, blocked ops={ops}, blocked nodes={len(nodes)})...", flush=True)
    if args.progress:
        _instrument(float16)
    m16 = float16.convert_float_to_float16(m, keep_io_types=True,
        # Keep Casts (bool masks, .float() tails) exact: converting the Cast to
        # float left its declared output type inconsistent (ORT load FAIL otherwise).
        op_block_list=ops, node_block_list=nodes or None)
    print(f"saving {dst}...", flush=True)
    onnx.save_model(m16, str(dst))
    print(f"done: {dst.stat().st_size/1e9:.2f} GB", flush=True)

if __name__ == "__main__":
    main()
