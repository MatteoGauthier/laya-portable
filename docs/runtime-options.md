# Laya runtime and portability options

Research checked 2026-09-19 against upstream documentation/source. This is a feasibility assessment, not a benchmark or a completed conversion. Runtime support changes; pin the tested runtime and model revisions before publishing.

## Recommendation

**Laya can plausibly become a native Mac application and a browser model. A complete port must preserve its custom decision head and input/output rules, not merely convert its ModernBERT encoder.** Start with one verified ONNX export and a parity harness, then compare native ONNX Runtime/Core ML against the existing PyTorch MPS baseline. If a small C/C++ library is the primary product, a ggml implementation is technically credible because llama.cpp already implements ModernBERT. MLX is a strong Apple GPU alternative, particularly for a Swift product, but involves another architecture implementation. These are engineering recommendations inferred from the sources below; no speed ranking has been measured.

## The terms describe different layers

| Term | What it supplies | What it does not supply |
|---|---|---|
| Safetensors | Safely serialized weight arrays | An executable model architecture |
| GGUF | Tensor data plus metadata in a deployment-oriented container | Automatic support for arbitrary neural networks |
| ggml | Portable C/C++ tensor computation and quantized operations | Laya's model graph or public prediction API |
| llama.cpp | Implementations of particular model architectures using ggml | An automatic importer for every Hugging Face model |
| ONNX | A serialized computation graph and tensors | A guarantee every accelerator implements every operation |
| ONNX Runtime | An executor with CPU and accelerator backends | Laya's text/schema preparation unless separately implemented |
| MLX | Array operations and neural-network building blocks, with Apple GPU execution | An automatic Torch-to-MLX conversion of custom code |
| Core ML | Apple's model representation and native inference runtime | Browser or Android deployment |
| WASM / WebGPU | Browser CPU execution / browser GPU compute | Model definitions or weights |

Sources: [Safetensors](https://huggingface.co/docs/safetensors/index), [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), [ggml](https://github.com/ggml-org/ggml), [MLX](https://github.com/ml-explore/mlx), [Core ML conversion formats](https://apple.github.io/coremltools/docs-guides/source/target-conversion-formats.html), [ORT WebGPU](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html).

## Exact work a Laya port must preserve

The English checkpoint is approximately 421 million parameters, using ModernBERT-large plus two transformer head layers. It is non-autoregressive: each question becomes an encoder input, options are located at marker tokens, and outputs are distributions rather than generated text. The encoder weights are fine-tuned, so substituting the original ModernBERT weights would change the model. [Laya model card](https://huggingface.co/convaiinnovations/laya)

The pinned local upstream source is commit `6a5819129eb220570792e417e49723d697efd76f`. `DecisionModel.forward` adds a learned question-type embedding to encoder states, runs the extra transformer layers over the whole sequence, gathers option-marker states, and scores them with a LayerNorm/MLP. The separate action head uses the first token and probability statistics. The extra transformer layers use PyTorch's default ReLU feed-forward activation; the option scorer and action head use GELU. Confusing these with ModernBERT's GeGLU would produce a wrong port. [Pinned model implementation](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/laya/common.py)

Preserve tokenizer behavior, JSON serialization, option ordering, marker sanitization, question/option/state token budgets, padding, calibration and result construction. Test identical token IDs before comparing tensors. Passing a raw string to an ordinary text-classification pipeline is not equivalent to the published `predict` API. [Pinned inference implementation](https://github.com/NandhaKishorM/laya/blob/6a5819129eb220570792e417e49723d697efd76f/laya/agent.py)

## llama.cpp and ggml: substantial reusable work exists

Current llama.cpp registers `modern-bert`. Its encoder graph implements symmetric sliding-window attention, local/global RoPE settings, layer normalization and GeGLU; it ends with per-token hidden states. This is useful for Laya because its head needs sequence states, not only a pooled embedding. The code also recognizes the 28-layer, 395M encoder configuration. [ModernBERT graph](https://github.com/ggml-org/llama.cpp/blob/master/src/models/modern-bert.cpp)

Its converter registers `ModernBertModel`, `ModernBertForMaskedLM` and `ModernBertForSequenceClassification`. It writes local-attention and global-attention-pattern metadata. This registration is **not** support for Laya's custom `DecisionModel`; the architecture, checkpoint layout and head tensors still need handling. [Converter](https://github.com/ggml-org/llama.cpp/blob/master/conversion/bert.py)

Proposed implementation, not existing functionality:

1. Export the fine-tuned encoder with a clearly specified mapping from Laya's checkpoint keys.
2. Compare unpooled encoder states against Torch, including local/global attention and padding cases.
3. Implement the type embedding and complete two-layer decision head in ggml. Include all sequence tokens through those head layers; gathering markers before the head changes the computation.
4. Implement option scoring, action output and calibration, plus the existing tokenizer and request preparation semantics.
5. Define a Laya GGUF architecture/schema or a documented companion artifact. A file containing extra tensors does not make an unchanged llama.cpp binary execute those tensors.

A prototype could retrieve encoder outputs through llama.cpp and execute a second head graph. A production integrated graph may avoid the intermediate host transfer; benchmark before choosing. The suggested integration is an inference from the architecture, not a tested patch.

Metal is enabled by default in macOS llama.cpp builds; the project also documents CPU, Android and several GPU backends. ggml explicitly targets WebAssembly and browser backends. This makes a portable `laya.cpp` plausible, but does not establish that all Laya operations work efficiently in any particular browser backend. Backend-specific parity and performance remain required. [Build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md), [ggml platform scope](https://github.com/ggml-org/ggml)

**Assessment:** good long-term C/C++ route, more implementation work than a successful graph export. A claim that ModernBERT is unsupported in current llama.cpp would be incorrect.

## MLX: native Apple GPU without a Python runtime

MLX supplies Python, C++, C and Swift APIs, lazy evaluation and graph transformations. Python can be the development frontend while compiled native operations perform the tensor work. MLX Swift has macOS and iOS examples and can be integrated through SwiftPM/Xcode. MLX uses shared CPU/GPU memory, with operations selecting their execution device. [MLX](https://github.com/ml-explore/mlx), [MLX Swift](https://github.com/ml-explore/mlx-swift), [Unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)

For Laya, implement the backbone and custom head using MLX primitives, map every weight, and validate outputs. No complete, verified upstream Laya MLX recipe was established in this research. MLX-LM support for unrelated language models would not itself solve this custom classifier. A Swift or C++ wrapper can ship without Python; simply changing the file extension to Safetensors does not port the model.

MLX supports function compilation and module quantization. Both are tools to investigate after floating-point parity. Timing must force lazy evaluation and completion; measuring creation of an unevaluated expression reports dispatch rather than inference. [Compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html), [Module quantization](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html)

**Assessment:** appealing for an Apple GPU/Swift implementation; not the natural single deployment stack for browser and Android.

## Core ML: first-class Apple application packaging

Core ML Tools converts PyTorch TorchScript or exported programs into ML Programs, which can be deployed in a native application. Python is used during conversion, not required by the shipped Swift application. Compute-unit selection can allow CPU, GPU and Neural Engine; conversion success does not prove the whole model executes on the Neural Engine. [Conversion formats](https://apple.github.io/coremltools/docs-guides/source/target-conversion-formats.html), [Prediction and compute units](https://apple.github.io/coremltools/docs-guides/source/model-prediction.html)

Export a tensor-only wrapper around the complete decision computation; keep schema parsing and tokenization in native application code. Start with fixed batch/sequence/option shapes, then introduce bounded shapes or buckets such as 128/256/512 tokens. Apple's documentation recommends enumerated shapes for optimization. Multiple enumerated inputs have deployment-version constraints. [Flexible input shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)

ONNX Runtime also offers a CoreML Execution Provider, including MLProgram and compute-unit options. This provides a useful experiment on the ONNX route, but ORT's CoreML operator coverage and graph partitioning are an additional layer compared with direct conversion. Inspect provider assignment and time actual inference. [ORT CoreML provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)

**Assessment:** strong candidate for the simplest native Apple product if conversion and acceleration coverage are good; benchmark against MLX/MPS before calling it fastest.

## Browser: ONNX Runtime Web is the first route to test

ORT Web offers WASM CPU execution and WebGPU execution. Use WebGPU for the GPU experiment, with runtime capability detection and a separately tested WASM fallback. WASM is not itself a GPU API. Static shapes can enable WebGPU graph capture when all compute kernels run on that provider. [ORT WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)

The WebGPU operator list includes MatMul, GatherElements, GELU, LayerNormalization, Softmax, RotaryEmbedding and MatMulNBits. It does **not** list TopK, DynamicQuantizeLinear or MatMulInteger in the version inspected. Therefore full Laya action-head statistics and a CPU-oriented dynamic-int8 export require explicit inspection; they cannot be assumed to execute entirely on WebGPU. Fused attention entries also contain mask limitations. Audit emitted operators and supported types, not just model names. [WebGPU operator table](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)

Possible engineering response: export the GPU-heavy computation to return option logits and the small first-token vector, then evaluate top-two statistics and the small action head on CPU. Preserve the original uncalibrated statistics. This is a proposal, not a measured optimization; avoid a GPU/CPU round trip inside every encoder layer.

The browser's memory budget matters independently of arithmetic speed. ORT documents a 4 GB WASM address-space ceiling and a 2 GB protobuf model-file limit, with external tensor data available for large models. Peak memory also includes loading buffers, activations and GPU allocations. Cache weights locally for subsequent visits and avoid loading all Laya checkpoints at once. [Large-model deployment](https://onnxruntime.ai/docs/tutorials/web/large-models.html)

A community artifact, [Mattepiu/laya-onnx](https://huggingface.co/Mattepiu/laya-onnx), was identified separately during this investigation. Its existence is a starting point for auditing, not evidence of WebGPU compatibility, current upstream parity, or browser performance.

## Android and ExecuTorch

ONNX Runtime has native Android Java/C/C++ packages. CPU is the portability baseline; acceleration can be added where supported. ORT explicitly warns that breaking a graph into accelerator/CPU partitions can reduce performance. Qualcomm QNN is available for supported Snapdragon devices, which is a device-family-specific deployment path rather than universal Android acceleration. [ORT mobile](https://onnxruntime.ai/docs/tutorials/mobile/), [QNN provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)

ExecuTorch is another native deployment route from PyTorch. Its current backend list includes XNNPACK CPU, Apple Core ML/MPS, Android Vulkan, Qualcomm and MediaTek. Export typically generates a backend-specific `.pte` artifact; unsupported delegated operations can fall back to CPU. It is not necessary to rewrite the model in C++ merely to remove the Python runtime. Laya's exportability and delegate coverage still need testing. [ExecuTorch backends](https://docs.pytorch.org/executorch/stable/backends-overview.html)

**Assessment:** use ONNX Runtime first if sharing a graph with the browser is the priority; investigate ExecuTorch if staying close to Torch and optimizing native mobile becomes the priority. Do not build both before identifying a concrete benefit.

## Quantization and performance expectations

For 421M parameters, simple arithmetic gives the following ideal weight payloads. These are not measured file sizes or peak RAM requirements:

| Precision | Ideal payload, decimal MB |
|---|---:|
| Float32 | 1,684 |
| Float16 | 842 |
| 8-bit | 421 |
| 4-bit | 210.5 |

Quantization scales, mixed-precision tensors, metadata and alignment add overhead. Activations and temporary buffers add runtime memory. Whole-model 4-bit storage is not automatically feasible or desirable.

ORT supports weight-only 4-bit conversion for selected constant MatMul/Gather inputs. Its documentation explicitly treats quantization as lossy and warns speed depends on hardware and quantize/dequantize overhead. Smaller files do not guarantee lower latency; CPU dynamic-int8, WebGPU MatMulNBits and ggml quantization are distinct execution paths. [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

Suggested evaluation order: floating-point parity, FP16, then a backend-supported 8-bit or 4-bit candidate. Keep decision-sensitive layers at higher precision if needed. Check probability drift and confidence-gate decisions as well as top-label agreement. Use held-out data for accuracy, Brier score/log loss, calibration and any temperature refit. Identical argmax labels can conceal important probability changes.

Benchmark batch size, token length and option count separately. Record cold load, warm p50/p95, peak memory, energy/thermal behavior, tokenization and complete API latency. Keep the model resident. Synchronize device work when timing and compare the same inputs, precision and warmup policy. Report questions/second or milliseconds/question; autoregressive tokens/second is the wrong primary measure here. These are proposed measurement requirements, not performance claims.

The key question is which kernels and execution graph the model uses. Replacing Python's orchestration alone may offer little gain if matrix multiplication dominates. A good public release should demonstrate measured improvements and faithful decisions, with pinned source revisions, reproducible conversion and parity fixtures.
