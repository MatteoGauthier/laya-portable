"""Build tiny ONNX probes for WebGPU blocker ops (TopK, IsNaN, And, Max).

Each <10KB, opset 18, to test ORT Web WebGPU support without loading 1.69GB.
"""
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto

def save(model, name):
    model.ir_version = 10  # match torch export / ORT 1.30 Web support (onnx 1.23 defaults to 14)
    out = Path(__file__).resolve().parents[1]/"js"/"web-test"/"models"/name
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out))
    print(f"{name}: {out.stat().st_size} bytes")

# TopK: [2,5] -> values [2,2], indices [2,2], k=2
X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5])
K = helper.make_tensor("k", TensorProto.INT64, [1], [2])
V = helper.make_tensor_value_info("values", TensorProto.FLOAT, [2, 2])
I = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2])
n = helper.make_node("TopK", ["x", "k"], ["values", "indices"], axis=-1)
g = helper.make_graph([n], "topk", [X], [V, I], [K])
save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), "topk.onnx")

# IsNaN: [2,3] -> [2,3] bool
X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info("y", TensorProto.BOOL, [2, 3])
n = helper.make_node("IsNaN", ["x"], ["y"])
g = helper.make_graph([n], "isnan", [X], [Y])
save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), "isnan.onnx")

# And: [2,3]bool x [2,3]bool -> [2,3]bool
A = helper.make_tensor_value_info("a", TensorProto.BOOL, [2, 3])
B = helper.make_tensor_value_info("b", TensorProto.BOOL, [2, 3])
Y = helper.make_tensor_value_info("y", TensorProto.BOOL, [2, 3])
n = helper.make_node("And", ["a", "b"], ["y"])
g = helper.make_graph([n], "and", [A, B], [Y])
save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), "and.onnx")

# Max (elementwise): [2,3] x [2,3] -> [2,3]
A = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
B = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
n = helper.make_node("Max", ["a", "b"], ["y"])
g = helper.make_graph([n], "max", [A, B], [Y])
save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), "max.onnx")

# Control: Add (known WebGPU-supported) [2,3] -> [2,3]
A = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3])
B = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 3])
Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
n = helper.make_node("Add", ["a", "b"], ["y"])
g = helper.make_graph([n], "add", [A, B], [Y])
save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), "add.onnx")
