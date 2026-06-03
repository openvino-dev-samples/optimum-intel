# Debugging MoE Fusion Failures

## Table of Contents
1. [Diagnostic Workflow](#diagnostic-workflow)
2. [Verifying Model IR](#verifying-model-ir)
3. [Verifying Execution Graph](#verifying-execution-graph)
4. [Common Error Messages](#common-error-messages)
5. [Consumers Count Analysis](#consumers-count-analysis)
6. [Hardware Requirements](#hardware-requirements)

## Diagnostic Workflow

```
1. Export model → check XML (IR structure)
2. If IR wrong → fix patcher, re-export
3. If IR correct → run inference on GPU
4. If crash → check error message (see Common Errors below)
5. If no crash → dump execution graph, check for fused ops
6. If no fused ops → check GPU capabilities, check pass registration
```

## Verifying Model IR

Write a verification script that parses the XML and checks:

```python
import xml.etree.ElementTree as ET

tree = ET.parse("model/openvino_model.xml")
root = tree.getroot()

# Build layer and edge maps
layers = {}
consumers = {}  # layer_id -> outgoing edge count
for layer in root.iter("layer"):
    lid = layer.get("id")
    layers[lid] = {
        "type": layer.get("type", ""),
        "name": layer.get("name", ""),
    }
for edge in root.iter("edge"):
    fl = edge.get("from-layer")
    consumers[fl] = consumers.get(fl, 0) + 1

# Check 1: ReduceSum consumer counts
for lid, info in layers.items():
    if info["type"] == "ReduceSum" and "mlp" in info["name"]:
        count = consumers.get(lid, 0)
        # Find if this ReduceSum feeds a Divide (normalization)
        feeds_divide = False
        for edge in root.iter("edge"):
            if edge.get("from-layer") == lid:
                tl = edge.get("to-layer")
                if layers.get(tl, {}).get("type") == "Divide":
                    feeds_divide = True
        if feeds_divide and count != 1:
            print(f"FAIL: Normalization ReduceSum {lid} has {count} consumers (need 1)")

# Check 2: Expert BMM MatMul transpose_b
for lid, info in layers.items():
    if info["type"] == "MatMul" and "bmm" in info["name"]:
        for layer_el in root.iter("layer"):
            if layer_el.get("id") == lid:
                data = layer_el.find("data")
                if data is not None:
                    tb = data.get("transpose_b", "N/A")
                    print(f"BMM {lid}: transpose_b={tb}")

# Check 3: Router MatMul transpose_b
for lid, info in layers.items():
    if info["type"] == "MatMul" and "linear" in info["name"] and "mlp" in info["name"]:
        for layer_el in root.iter("layer"):
            if layer_el.get("id") == lid:
                data = layer_el.find("data")
                if data is not None:
                    tb = data.get("transpose_b", "N/A")
                    print(f"Router {lid}: transpose_b={tb}")
```

## Verifying Execution Graph

After running inference, dump and check the execution graph:

```python
import openvino as ov

# After model.generate() or model()
compiled_model = model.request.get_compiled_model()
runtime_model = compiled_model.get_runtime_model()
ov.save_model(runtime_model, "exec_graph.xml")

# Parse and check for fusion
tree = ET.parse("exec_graph.xml")
root = tree.getroot()
for layer in root.iter("layer"):
    ltype = layer.get("type", "")
    if "moe" in ltype.lower() or "MOE" in ltype:
        print(f"Found: {ltype} — {layer.get('name', '')}")
    if ltype == "Tile" and "mlp" in layer.get("name", ""):
        print(f"UNFUSED Tile: {layer.get('name', '')}")
```

**Expected results when fusion succeeds:**
- `MOE3GemmFusedCompressed` ops: 1 per MoE layer
- `MOECompressed` ops: 0 (all fused)
- `Tile` ops (MoE): 0 (all fused)

## Common Error Messages

### `input offset too big`
**Meaning:** A kernel stage tried to access an input index beyond the primitive's input count.

**Root cause:** Usually a mismatch between the number of inputs the fused op declares (e.g., 13 vs 14) and what the kernel host code tries to access. Can also occur with optimum-intel's stateful model generation when KV cache handling has issues.

**Fix:** Check `ops/moe.cpp` `validate_inputs_count`, `moe_3gemm_fused_compressed.cpp` expected_inputs, and `moe_3gemm_swiglu_opt.cpp` input index enums.

### `MOECompressed hasn't been found in primitive_ids map`
**Meaning:** Stages 2-3 created an MOECompressed op, but stage 4 (FuseMOE3GemmCompressed) failed to fuse it. MOECompressed(GEMM3_SWIGLU) has no standalone GPU implementation.

**Root cause:** The routing subgraph doesn't match the FuseMOE3GemmCompressed pattern. Usually a `consumers_count(1)` violation.

**Fix:** Check the normalization ReduceSum consumer count. Check all 16 pattern nodes have exactly 1 consumer each.

### No error, but no fusion (Tile ops remain)
**Meaning:** Stage 2 (FuseVectorizedMOE3GEMM) didn't match.

**Possible causes:**
1. `supports_immad` is false for this GPU
2. Expert MatMuls don't have `transpose_b=true` (MOC pass didn't run)
3. Pattern structure mismatch (wrong activation, missing Reshape, etc.)

### NNCF InvalidGroupSizeError
**Meaning:** NNCF tried to INT4-quantize a small constant that shouldn't be quantized.

**Example:** `matmul(weights, ones)` where `ones` is [16,1] — group_size=128 doesn't divide 16.

**Fix:** Don't introduce MatMul ops with small constant inputs. Use CumSum trick instead.

### All NaN output (GPU only, no error message)
**Meaning:** The fused kernel executes but produces garbage.

**Root cause #1:** `keep_moe_3gemm_const_precision.cpp` pattern doesn't match the new input count. The `wrap_type<MOE3GemmFusedCompressed>({...})` pattern has a hardcoded number of `any_input()` entries — if you add a new input (e.g., gate weight as Input 13), update the pattern from 13 to 14 `any_input()` entries for the SigmoidBias branch.

**Root cause #2:** Data type mismatch. If a new input is f32 but the kernel reads it as f16 (`MOE_DTYPE`), the values will be garbage. Use JIT conditionals like `GATE_WEIGHT_IS_F32` to handle different input dtypes.

**Diagnosis:**
```bash
# Check if keep_const_precision pattern matches
grep -n 'SigmoidBias\|any_input' keep_moe_3gemm_const_precision.cpp
# Count the any_input() entries — must equal total inputs
```

## Consumers Count Analysis

FuseMOE3GemmCompressed has 16 nodes with `consumers_count(1)`. Here's every node and what it means:

| Pattern Node | OV Type | What It Is | Why 1 Consumer |
|---|---|---|---|
| matmul | MatMul | Router linear | Only feeds Softmax |
| sm_softmax | Softmax | Score normalization | Only feeds TopK |
| sm_reduce | ReduceSum | Sum of topk weights | Only feeds Divide |
| sm_norm | Divide | Weight normalization | Only feeds Slice/Scatter |
| sm_shape_of | ShapeOf | Shape for Slice | Only feeds Slice |
| sm_norm_slice | Slice | Optional slice (from scatter_) | Only feeds Scatter |
| sig_add | Add | Sigmoid bias (alt path) | Only feeds Sigmoid |
| sig_reduce | ReduceSum | Sigmoid reduce | Only feeds Add |
| sig_add_eps | Add | Epsilon add | Only feeds Divide |
| sig_norm | Divide | Sigmoid normalization | Only feeds Slice |
| sig_slice | Slice | Sigmoid slice | Only feeds Broadcast |
| sig_bc | Broadcast | Sigmoid broadcast | Only feeds Scatter |
| scatter | ScatterElementsUpdate | Route to expert matrix | Only feeds Transpose |
| transpose | Transpose | Reorder dimensions | Only feeds Reshape |
| reshape | Reshape | Shape adjustment | Only feeds Unsqueeze |
| unsqueeze_moe | Unsqueeze | Expand for multiply | Only feeds MOE op |

Not all 16 are active simultaneously — the Softmax and Sigmoid paths are alternatives. MiniCPM5 uses the Softmax path (sm_* nodes).

## Hardware Requirements

**FuseVectorizedMOE3GEMM** requires `supports_immad` (systolic array / XMX support):
- Intel Arc (DG2, Xe-HPG) — supported
- Intel Arc 140V (Xe2-LPG, Lunar Lake) — supported
- Intel Iris Xe (Gen12) — NOT supported (no systolic arrays)
- Intel UHD (Gen9-11) — NOT supported

**FuseVectorizedMOE2GEMM** requires Xe2 or Xe3 architecture specifically.

To check GPU capabilities:
```python
import openvino as ov
core = ov.Core()
print(core.get_property("GPU", "FULL_DEVICE_NAME"))
print(core.get_property("GPU", "DEVICE_ARCHITECTURE"))
```

**Debug flag:** If OV was built with `ENABLE_GPU_DEBUG_CAPS=ON`, you can set `GPU_DISABLE_MOE_OPT=1` to skip all MoE fusion passes (useful for isolating issues).

## Routing Precision Diagnostics

When CPU vs GPU outputs diverge significantly (cosine < 0.5), probe layer-by-layer to locate the source:

```python
import openvino as ov
import numpy as np

core = ov.Core()
model = core.read_model('model.xml')

# Add intermediate outputs at each MoE layer
for op in model.get_ordered_ops():
    name = op.get_friendly_name()
    if 'mlp' in name and op.get_type_name() == 'Add':  # residual add after MLP
        model.add_outputs([op.output(0)])
        break  # Test one layer at a time

# Compare CPU vs GPU for each probed output
model_cpu = core.compile_model(model, 'CPU')
model_gpu = core.compile_model(model, 'GPU')
# ... set inputs, infer, compare outputs
```

**Key indicators:**
- If cosine drops sharply at the first MoE layer → routing precision issue
- If individual expert MatMuls are accurate (cos > 0.998) but combined routed output is bad (cos < 0.1) → wrong experts selected
- If GPU routing scores norm is ~1/5 of CPU → TopK selecting different experts with much lower weights

**Gate MatMul precision check:**
```python
# Check if FP16 accumulation error exceeds inter-expert gap
# For gate MatMul [batch, hidden_dim] x [hidden_dim, num_experts]:
# FP16 accumulation error ~ sqrt(K) * 2^(-11) * avg_product_magnitude
# If this > sigmoid gap between expert rank 16 and 17, routing is unreliable
import math
K = 2048  # hidden_dim
fp16_error = math.sqrt(K) * 2**(-11)  # ~0.022 relative error
# Compare to inter-expert sigmoid gap from CPU reference
```

## Build & Install Tips

**Manual SO copy (fastest iteration):**
```bash
# After `make openvino_intel_gpu_plugin`:
cp bin/intel64/Release/libopenvino_intel_gpu_plugin.so \
  $(python3 -c 'import openvino; print(openvino.__file__.rsplit("/", 1)[0])')/libs/
```

**Build targets:**
- `make openvino_intel_gpu_plugin` — only GPU plugin (enough for kernel/host code changes)
- `make -j$(nproc)` — full build (needed for common transformation changes)
- `make ie_wheel` — wheel packaging (may NOT include latest SO!)
