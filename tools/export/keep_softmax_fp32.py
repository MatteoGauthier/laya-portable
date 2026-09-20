"""Surgical FP32-softmax variant of the plain FP16 split model.

The converter's op blocklist path never terminates on this graph (blocking
'Cast' while inserting Cast nodes = unbounded growth; 58+ min observed).
Instead: take the proven plain-FP16 model and wrap each Softmax with
FP16->FP32 in / FP32->FP16 out casts (~62 tiny insertions, seconds).

Usage: .venv/bin/python tools/export/keep_softmax_fp32.py [--src ...] [--dst ...]
"""
import argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import onnx
from onnx import TensorProto, helper


def add_vi(graph, name, elem_type, shape_dims):
    vi = helper.make_tensor_value_info(name, elem_type, shape_dims)
    # copy dim_params if the source had them
    graph.value_info.append(vi)
    return vi


def shape_of(graph, name):
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            tt = vi.type.tensor_type
            dims = []
            for d in tt.shape.dim:
                dims.append(d.dim_param if d.HasField("dim_param") else d.dim_value)
            return tt.elem_type, dims
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "models" / "laya-split-fp16.onnx")
    ap.add_argument("--dst", type=Path, default=ROOT / "models" / "laya-split-fp16-keepsm.onnx")
    args = ap.parse_args()
    print(f"loading {args.src} ...", flush=True)
    m = onnx.load(str(args.src))
    g = m.graph
    sms = [n for n in g.node if n.op_type == "Softmax"]
    print(f"wrapping {len(sms)} Softmax nodes...", flush=True)
    for n in sms:
        # input side: X(f16) -> C1 -> X_s32(f32)
        for i, inp in enumerate(list(n.input)):
            et, dims = shape_of(g, inp)
            if et != TensorProto.FLOAT16:
                continue
            cname = f"{n.name}_in{i}_f32"
            cnode = helper.make_node("Cast", [inp], [cname], name=f"{n.name}_castin{i}", to=TensorProto.FLOAT)
            g.node.append(cnode)
            add_vi(g, cname, TensorProto.FLOAT, dims)
            n.input[i] = cname
        # output side: Y_s32(f32) -> C2 -> Y(f16)
        for i, out in enumerate(list(n.output)):
            et, dims = shape_of(g, out)
            if et != TensorProto.FLOAT16:
                continue
            inner = f"{n.name}_out{i}_f32"
            n.output[i] = inner
            add_vi(g, inner, TensorProto.FLOAT, dims)
            cnode = helper.make_node("Cast", [inner], [out], name=f"{n.name}_castout{i}", to=TensorProto.FLOAT16)
            g.node.append(cnode)
    from onnxconverter_common.float16 import sort_topology
    sort_topology(g)
    onnx.checker.check_model(m)
    print("checker ok, saving...", flush=True)
    onnx.save_model(m, str(args.dst))
    print(f"done: {args.dst.stat().st_size/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
