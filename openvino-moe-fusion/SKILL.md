---
name: openvino-moe-fusion
description: Adapt Mixture-of-Experts (MoE) models for OpenVINO GPU fusion. Use this skill when adding MoE model support to optimum-intel, writing model patchers for MoE architectures (DeepSeek-V2, Mixtral, Qwen-MoE, MiniCPM-MoE, etc.), debugging MoE fusion failures on Intel GPUs, or when the user mentions MOECompressed errors, MoE transformation failures, or wants to enable GPU MoE fusion for any new model. Also trigger when the user asks about OpenVINO's MoE fusion pipeline, consumers_count constraints, or vectorized expert computation patterns.
---

# OpenVINO MoE Fusion — Model Adaptation Guide

This skill guides you through adapting new MoE (Mixture of Experts) models so that the OpenVINO GPU plugin can fuse the entire MoE block into a single high-performance kernel (`MOE3GemmFusedCompressed`). The process involves writing a model patcher in optimum-intel that produces an IR graph matching the GPU plugin's rigid pattern expectations.

## Why This Matters

Without fusion, each MoE layer runs as hundreds of individual ops (Tile, MatMul, Scatter, etc.). With fusion, the entire routing + expert computation collapses into one fused kernel — dramatically improving GPU inference throughput.

## The Fusion Pipeline

The GPU plugin applies 5 transformation passes in sequence. **All must succeed** — if any fails, the MoE block stays unfused and may crash at runtime (the intermediate `MOECompressed` op has no standalone GPU implementation for GEMM3_SWIGLU).

```
Stage 1: VectorizedMOE3GEMMTransposeWeights (GPU plugin, before TransposeMatMul)
  - Converts expert MatMul(transpose_b=false) → MatMul(transpose_b=true)
  - Inserts Transpose on weights, handles decompression Converts
  - Only needed when weights are in [E, K, N] format with tb=false

Stage 2: FuseVectorizedMOE3GEMM (GPU plugin)
  - Fuses expert computation (3×MatMul + SwiGLU + routing multiply + ReduceSum) → MOE(GEMM3_SWIGLU)
  - Requires: supports_immad (systolic array GPUs: Arc, Xe2+)
  - Looks through Transpose nodes on MatMul weight inputs (for ConvertMOEToMOECompressed compatibility)

Stage 3: ConvertMOEToMOECompressed (GPU plugin)
  - MOE → MOECompressed (handles INT4/INT8 compressed weights)
  - Weight Constant must be in [E, N, K/G, G] format for 4D group-compressed weights

Stage 4: FuseMOE3GemmCompressed (GPU plugin)
  - Fuses routing subgraph + MOECompressed → MOE3GemmFusedCompressed
  - 16 nodes with consumers_count(1) — extremely strict
  - Supports both Softmax (11 inputs) and Sigmoid+bias routing (14 inputs)
  - For SigmoidBias: captures gate_routing_weight as Input 13 for FP32 gate GEMV

Stage 5: KeepMOE3GemmConstPrecision (GPU plugin)
  - Marks u4 weight/zp Constants with keep_const_precision to prevent type conversion
  - Pattern must match exact input count (11 for Softmax, 14 for SigmoidBias)
  - If this fails silently, u4 data gets converted → NaN output

Stage 6: TransposeMatMul (GPU plugin, runs AFTER MoE passes)
  - Absorbs standalone Transpose nodes into MatMul by toggling transpose_b
  - MoE passes are registered BEFORE this to prevent it from undoing their work
```

## Critical: Weight Layout Requirements

**The expert weight Constants in the IR MUST be in `[E, N, K]` format** (num_experts, output_features, input_features). This is the layout that `ConvertMOEToMOECompressed` expects when it interprets 4D group-compressed weights as `[E, N, K/G, G]`.

**DO NOT pre-transpose weights** in the patcher's `__enter__`. Instead:
1. Keep weights in their original PyTorch layout: `[E, out_features, in_features]` = `[E, N, K]`
2. Use `.transpose(-1, -2)` in the forward pass to get `[E, K, N]` for `torch.bmm`
3. The `.transpose()` traces as a Transpose node in the IR
4. `TransposeMatMul` (MOC or GPU plugin) absorbs it: `Constant[E,N,K] → MatMul(tb=true)`
5. NNCF quantizes `Constant[E,N,K]` to `u4[E,N,K/G,G]` — matches expected layout

**Wrong approach (causes ConvertMOEToMOECompressed failure):**
```python
# DON'T do this — bakes transposed values into Constant, gives [E,K,N] layout
moe.gate_projs = torch.stack([e.gate_proj.weight for e in experts]).transpose(1, 2).float()
gate = torch.bmm(input, self.gate_projs)
```

**Correct approach:**
```python
# Keep original [E, N, K] layout — ConvertMOEToMOECompressed can match
moe.gate_projs = torch.stack([e.gate_proj.weight for e in experts]).float()
gate = torch.bmm(input, self.gate_projs.transpose(-1, -2))
```

## Step-by-Step Adaptation Process

### Step 1: Analyze the Source Model's MoE Architecture

Before writing any code, understand the model's MoE implementation. Read the model's config and the original MoE forward pass.

**Key config values to extract:**
```python
n_routed_experts      # Total number of experts (e.g., 160)
num_experts_per_tok   # Top-K (e.g., 16)
routed_scaling_factor # Post-routing scaling (e.g., 3.66), may be absent
topk_method           # "greedy", "group_limited_greedy", etc.
n_group               # Number of expert groups (1 = no grouping)
topk_group            # Experts per group
norm_topk_prob        # Whether routing weights are normalized
scoring_func          # "softmax" or "sigmoid" — determines routing branch
```

**What to look for in the original forward pass:**
- How routing weights are computed (softmax vs sigmoid+bias, normalization method)
- Whether there's expert group masking (only matters when `n_group > 1`)
- The scaling strategy (pre-scale, post-scale, or normalization)
- How experts are dispatched (per-expert loop vs batched)

### Step 2: Choose Routing Type

FuseMOE3GemmCompressed supports **two routing patterns**:

#### Softmax routing (most common — DeepSeek-V2, Mixtral, Qwen-MoE):
```
MatMul → Softmax → TopK → ReduceSum → Divide → [Slice] → Scatter → ...
```

#### Sigmoid+bias routing (MiniCPM5-MoE and similar):
```
MatMul → Sigmoid → Add(bias) → TopK
               ↘ GatherElements(topk_indices)
                    → ReduceSum → Add(eps) → Divide → [Slice] → Scatter → ...
```

The sigmoid branch gathers original (unbiased) scores, normalizes with `(sum + eps)`, and the fusion kernel handles the routing internally.

### Step 3: Write the Patcher Function

Use the templates in `references/patcher_template.md`.

**Critical constraints:**

#### Constraint 1: Weight Layout — [E, N, K] Constants
See the "Critical: Weight Layout Requirements" section above. Do NOT pre-transpose.

#### Constraint 2: Normalization ReduceSum Must Have Exactly 1 Consumer
The normalization ReduceSum (`topk_sum`) can ONLY feed the Divide node. See `references/patcher_template.md` for the CumSum trick when you need a second sum.

#### Constraint 3: Inline the Gate When Possible
If the model's gate has complex logic (group masking, score normalization, custom topk), inline it.

#### Constraint 4: Routing Weight Scatter Pattern
Must use `torch.zeros()` → `scatter_()` which traces to `Broadcast + ScatterElementsUpdate`.

#### Constraint 5: Final ReduceSum Must Have keep_dims=false

### Step 4: Register the Model Patcher

In `optimum/exporters/openvino/model_configs.py`:
```python
@register_in_tasks_manager("my_model_type", *["text-generation", "text-generation-with-past"], library_name="transformers")
class MyModelOpenVINOConfig(TextDecoderWithPositionIdsOnnxConfig):
    _MODEL_PATCHER = MyModelMoEPatcher
```

### Step 5: Export and Verify

**Export:**
```bash
optimum-cli export openvino --model <model_path> \
  --task text-generation-with-past --trust-remote-code \
  <output_dir> --weight-format int4
```

**Install order matters:** Always install the custom OV wheel AFTER optimum-intel:
```bash
pip install -e <optimum-intel-dir>  # or pip install <optimum-intel-dir>
pip install --force-reinstall --no-deps <custom-ov-wheel>
```

**CRITICAL: Manual SO copy for iterative development:**
`make ie_wheel` / `pip install` does NOT always pick up the latest `.so`. For rapid iteration, copy the plugin directly:
```bash
cp openvino/bin/intel64/Release/libopenvino_intel_gpu_plugin.so \
  $(python3 -c 'import openvino; print(openvino.__file__.rsplit("/", 1)[0])')/libs/
```

**Verify weight layout (pre-GPU):**
```python
model = core.read_model("model.xml")
for op in model.get_ops():
    if op.get_type_name() == "MatMul" and "bmm" in op.get_friendly_name().lower():
        tb = op.get_attributes().get("transpose_b", "?")
        const = trace_to_constant(op.input(1))
        print(f"{op.get_friendly_name()}: tb={tb}, const_shape={const.shape}")
        # Expected: tb=True, const shape=[E, N, K/G, G] (4D) or [E, N, K] (3D)
```

**Verify GPU execution graph (post-inference):**
- Look for `MOE3GemmFusedCompressed` ops (fusion succeeded)
- No standalone `MOECompressed` ops (would indicate stage 4 failed)
- No unfused `Tile` ops in MoE layers

### Step 6: Debug Fusion Failures

Read `references/debugging.md` for detailed troubleshooting.

**Quick diagnosis:**

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `MOECompressed hasn't been found in primitive_ids map` | Stage 4 failed — routing pattern mismatch | Check consumers_count on all routing nodes |
| `MOE(extension) is not supported` | Stage 3 failed — weight layout mismatch | Verify Constants are [E,N,K/G,G], not [E,K/G,G,N] |
| No MOE ops at all in exec graph | Stage 2 failed — pattern mismatch or `supports_immad=false` | Check GPU supports systolic arrays; verify MatMul transpose_b |
| `Incompatible MatMul matrix dimension` | ConvertMatMulToFullyConnected breaks batched MoE MatMuls | Ensure MoE fusion runs before PostLPT; check convert_matmul_to_fc.cpp Transpose stripping |
| Two ReduceSums merge into one | Tracer CSE deduplication | Use cumsum trick, never clone().sum() |

### Debugging GPU vs CPU Accuracy Gap

| Symptom | Root Cause | Resolution |
|---------|-----------|------------|
| Cosine sim ≈ -0.07, completely wrong tokens | **FP16 gate MatMul routing precision** (FIXED) | Gate GEMV [1,2048]×[2048,160] FP16 accumulation error (~0.07) >> inter-expert gap (~0.0004). Fix: FP32 GEMV in sigmoid_bias_topk kernel. See "Known Fix: Routing Precision" below. |
| Cosine sim ≈ 0.97, top-1 matches | Normal FP16 expert computation variance | Expected — identical text generation in practice |
| GPU FP32 mode fails with layout error | MOE3GemmFusedCompressed only supports FP16 | Cannot force FP32 on GPU for MoE kernel |
| CPU INT4 ≠ Torch BF16 top-1 | INT4 quantization loss | Expected — use larger group_size or sensitivity metrics for better INT4 |
| CPU FP16 model ≠ Torch BF16 top-1 | BF16→FP16 precision conversion | Expected for models with tight logit margins |
| Exec graph shows 0/27 fused ops, or N standalone "MOECompressed" | **Wrong detection method** — friendly-name substring matches leftover Reshape nodes (`.../ReduceSum_1_Reshape/MOECompressed`). | Classify by exec-graph op TYPE: `op.get_rt_info()["layerType"].astype(str) == "moe_3gemm_fused_compressed"`, not by friendly name. |
| GPU degenerate (repeats one token); routing logits wrong (e.g. ~`[-0.35,…]` instead of `[-2.75,…]`) | **Router gate FC horizontally-fused with shared-expert FCs** → MoE op reads wrong VariadicSplit slice as routing logits. See "Fixed Bug: router gate horizontally-fused…" below. | Exclude FCs feeding `MOE3GemmFusedCompressed` from `FullyConnectedHorizontalFusion`. |
| All NaN output after adding new fused input | **KeepMOE3GemmConstPrecision** pattern mismatch | Pattern has hardcoded input count; if it doesn't match, u4 constants lose precision marking → NaN |

## Known Fix: FP16 Gate MatMul Routing Precision

**Problem:** For SigmoidBias routing (MiniCPM5-MoE), the gate MatMul `[batch, 2048] × [2048, 160]` runs in FP16 on GPU. With K=2048, FP16 accumulation produces ~0.07 absolute error in gate logits. After sigmoid, this becomes ~0.01 error — which is 23× larger than the inter-expert gap (~0.0004 after sigmoid). The `e_score_correction_bias` (~92.5) further amplifies the problem by placing biased scores in a range where FP16 ULP = 0.0625.

**Result:** TopK selects completely wrong experts → cosine similarity between CPU and GPU is ~-0.07 (anti-correlated).

**Fix (10 files, commit `c32191f494` on `minicpm5-moe-transpose`):**
1. `moe_3gemm_base.hpp` — Added `ROUTING_GATE_WEIGHT = 13` to input enum
2. `fuse_moe_3gemm_compressed.cpp` — Pattern captures `gate_routing_weight_m` from the routing MatMul, adds as arg[13]
3. `moe_3gemm_fused_compressed.cpp` + `.hpp` (op + primitive) — Updated to 14 inputs for SIGMOID_BIAS
4. `ops/moe.cpp` — `validate_inputs_count` 13 → 14
5. `keep_moe_3gemm_const_precision.cpp` — **Critical:** Updated pattern from 13 to 14 inputs
6. `moe_3gemm_swiglu_fuse.cl` — FP32 GEMV loop in `sigmoid_bias_topk` kernel replaces FP16 gate logit reading
7. `moe_3gemm_swiglu_opt.cpp` — JIT constants `GATE_HIDDEN_DIM`, `GATE_WEIGHT_IS_F32`; passes hidden_states + gate_weight to kernel
8. `fuse_moe_3gemm_compressed_test.cpp` — Updated unit test for 14th arg

**Result after fix:** Cosine 0.06 → 0.97, identical top-1 token, matching text generation.

**Key lesson:** `keep_moe_3gemm_const_precision.cpp` must match the exact input count in its `wrap_type` pattern. If it doesn't match, u4 weight Constants silently lose their `keep_const_precision` attribute → type conversion → NaN.

## Fixed Bug: router gate horizontally-fused with shared-expert FCs (GPU degenerate output)

**Symptom (MiniCPM5-MoE INT4 on GPU Flex 170, 2026-06-03):** OV INT4 matched PyTorch 20/20 on **CPU**,
but the **same IR on GPU was degenerate** (repeated one token, e.g. `" in in in …"`). Fusion itself was
fine (27 `moe_3gemm_fused_compressed` ops).

> **⚠️ FIRST: clear caches between every GPU debug run.** optimum-intel enables OV model caching by
> default (`<model_dir>/model_cache/*.blob`) plus the driver's `~/.cache/neo_compiler_cache`. A cached
> compiled model **skips the whole transformation pipeline**, so kernel/pass edits *and* tensor dumps
> silently reflect the **old** build — this caused several misleading intermediate diagnoses. Always:
> `rm -rf <model_dir>/model_cache ~/.cache/neo_compiler_cache` and pass `ov_config={"CACHE_DIR":""}`.
> Trust only cache-free numbers.

**Root cause:** `FullyConnectedHorizontalFusion` (GPU plugin, `fc_horizontal_fusion.cpp`) runs **after**
`FuseMOE3GemmCompressed` and fuses all `FullyConnectedCompressed` users of a shared input. The **MoE
router gate** FC and the **shared-expert gate/up** FCs all consume the same hidden state, so they were
merged into one MatMul + `VariadicSplit`. The fused MoE op kept its routing-logits input wired to the
merged op and ended up reading the **wrong split slice** — the shared-expert gate_proj output
(~`[-0.35,…]`) instead of the true router logits (~`[-2.75,…]`). Wrong logits → `sigmoid_bias_topk`
selects the wrong experts → degenerate output.

**How it was localized (cache-free):** in-kernel `printf` of `expert_id` (per slot) and the routing
`in_value`/`sigmoid` in `sigmoid_bias_topk` showed GPU logits ≈ `[-0.35,…]` vs the true `[-2.75,…]`
(recomputed as `hidden @ gate_W.T`). Searching the GPU tensor dumps for `[-0.35,…]` found them in
`…shared_experts.gate_proj…_fused_3FCs` — pinpointing the horizontal-fusion mis-wire. (An isolated
compressed transpose_b MatMul was verified correct on GPU, ruling out the INT4 FC kernel itself.)

**Fix (`fc_horizontal_fusion.cpp`):** add `feeds_moe_router(fc)` (true if any consumer is a
`MOE3GemmFusedCompressed`) and skip such FCs in both the `is_target_pattern` predicate and the
collection loop, so the router gate stays out of the horizontal fusion. Result: tiny 3-layer repro
GPU-vs-CPU cosine **0.9995**, argmax identical; full model generates correct text; MoE still fused (27).

**Precision hardening (same change set):** the decode `mlp_reduce` (`moe_3gemm_swiglu_mlp.cl`) and the
prefill `moe_scatter_reduction_opt.cl` summed per-expert outputs in **fp16**; switched both to **fp32**
accumulation (cast back to fp16 on store) to match the CPU reference and reduce borderline greedy flips.

**Residual:** free-running greedy GPU vs CPU still flips occasionally, but **only at near-tie steps**
(CPU top-2 logit margin < 0.2); per-step teacher-forced logit cosine is ~0.99+. This is inherent
FP16-vs-bf16 sensitivity, not a bug. For tightest top-1 agreement use int8 / lower-aggression int4.

## Supported MoE Architectures

| Architecture | Gate Type | Routing | Scaling | Notes |
|-------------|-----------|---------|---------|-------|
| DeepSeek-V2 | Softmax + TopK | Softmax branch (11 inputs) | routed_scaling_factor | May need group mask handling |
| MiniCPM5-MoE | Sigmoid+bias + TopK | Sigmoid branch (14 inputs) | routed_scaling_factor=3.66 | e_score_correction_bias, shared experts, FP32 gate GEMV fix |
| Mixtral | Softmax + TopK | Softmax branch (11 inputs) | norm_topk_prob=true | Already normalized, simpler |
| Qwen-MoE | Softmax + TopK | Softmax branch | Varies | Check specific config |

## GPU MoE Execution Paths & Precision

The fused MoE kernel (`MOE3GemmFusedCompressed`) has **two prefill execution paths** controlled by `MOE_USE_MICRO_GEMM_PREFILL` env var:

### micro_gemm path (default on Xe2+ iGPU with INT4 weights)
- Uses micro_gemm kernels for GEMM (higher performance on systolic array GPUs)
- Scatter-reduce uses `moe_scatter_reduction_opt.cl` — **already accumulates in FP32**
- Enabled when: `arch >= xe2`, iGPU, weights are `u4` (INT4)
- Force with: `MOE_USE_MICRO_GEMM_PREFILL=1`

### onednn path (default on dGPU, pre-Xe2, or non-INT4 weights)
- Uses oneDNN GEMM primitives
- Scatter-reduce uses `index_add_` kernel in `moe_3gemm_swiglu_fuse.cl`
- **Fixed:** Now accumulates into an FP32 buffer (`acc_f32`) and converts to FP16 once at the end via `scatter_f32_to_f16` kernel
- Previously had catastrophic FP16 truncation: each expert's scatter read FP16 dst, added float src, wrote `(half)(dst+src)` — with 16 experts, ~1-2 LSB lost per step
- Force with: `MOE_USE_MICRO_GEMM_PREFILL=0`

### Single-token path (both modes)
- When `token_num == 1`, uses `exec_single_token` — different code path, no scatter accumulation needed

### Verified accuracy
After the FP32 scatter fix, both GPU paths (micro_gemm and onednn) produce **cosine similarity >0.999999** with each other, confirming the fix eliminates the precision gap between the two paths.

**Typical GPU vs CPU maxdiff:** ~5-10 for models with many experts (160). This is inherent to FP16 GEMM computation on GPU and applies to BOTH paths equally. The remaining GPU vs CPU gap comes from different INT4 dequantization implementations and FP16 DPAS/systolic accumulation — not from scatter.

**Practical impact:** For real models with learned representations, the GPU vs CPU token match rate is typically high (>90%). For tiny/test models with random weights and tight logit margins, top-1 token may differ due to cascading FP16 error.

**The sigmoid routing precision fix** (float local memory in `sigmoid_bias_topk`) improves expert selection accuracy. The scatter FP32 accumulation fix eliminates the dominant onednn-path error source. Remaining error is from FP16 GEMM accumulation in the systolic pipeline.

### FP16 Expert Selection Divergence (main GPU vs CPU accuracy gap source)

The gate MatMul (`hidden_states × gate_weight`) runs in FP16 on GPU but FP32 on CPU. Its output determines which experts are selected via `sigmoid(gate_logits) + bias → TopK`. With 160 experts and only 16 selected, the 16th and 17th experts may have biased scores differing by less than FP16 precision (~0.001). On GPU, this can cause **different expert selection** than CPU, leading to completely different MoE outputs.

**Why this is worse for tiny/untrained models:**
- Random weights → flat sigmoid score distribution → many experts have similar scores
- Tight top-1 logit margins (~0.5 difference) → any precision change flips top-1
- Cosine sim ~0.50 between CPU INT4 and GPU INT4 is **expected** for 6-layer random models

**Why real trained models are robust:**
- Learned routing weights → clear winners among 160 experts (large score gaps)
- Well-separated logit distributions → FP16 precision doesn't change top-1
- Expected cosine sim > 0.99 between CPU and GPU INT4

**GPU FP32 mode:** `MOE3GemmFusedCompressed` does **NOT** support `data_type=f32`. Setting `INFERENCE_PRECISION_HINT=f32` will cause a compile error: `[GPU] No layout format available for moe3gemmfusedcompressed (format: bfyx, data_type: f32)`.

### Detecting MoE Fusion in Execution Graph

The GPU execution graph uses `ExecutionNode` as the type name for ALL ops. To detect fused MoE ops, check the **friendly name**:

```python
rm = compiled_model.get_runtime_model()
fused = [op for op in rm.get_ops()
         if "MOE3GemmFusedCompressed" in op.get_friendly_name()]
print(f"Fused MoE layers: {len(fused)}")

# WRONG: op.get_type_name() returns "ExecutionNode" for all GPU ops
# RIGHT: check op.get_friendly_name() for "MOE3GemmFusedCompressed"
```

## OpenVINO Code Changes Required

These OV C++ modifications are needed for MiniCPM5-MoE support (may be upstreamed):

1. **matmul_experts_fusion.cpp** — `FuseVectorizedMOE3GEMM`: Look through Transpose nodes on MatMul weight inputs so ConvertMOEToMOECompressed can match the decompression chain
2. **convert_matmul_to_fc.cpp** — `ConvertMatMulToFullyConnected`: Only strip Transpose from weight when `tb=false` (not when `tb=true`, which would break FullyConnected shape)
3. **fuse_moe_3gemm_compressed.cpp** — `sig_slice` made optional (for sigmoid+bias routing)
4. **transformations_pipeline.cpp** — MoE passes registered BEFORE TransposeMatMul
5. **moe_3gemm_swiglu_fuse.cl** — `sigmoid_bias_topk`: Use float precision for local memory arrays (sigmoid values, selection scores, sorting comparisons) instead of MOE_DTYPE to prevent FP16 routing precision loss
6. **moe_3gemm_swiglu_fuse.cl** — `index_add_`: Scatter into FP32 buffer (`__global float* dst_tok_f32`) instead of FP16 output; added `scatter_f32_to_f16` kernel (gated by `SCATTER_F32_TO_F16_ENABLE`) for final FP32→FP16 conversion
7. **moe_3gemm_swiglu_opt.cpp** — Added `MoE3GemmSwigluScatterF32ToF16` kernel generator class; allocate FP32 accumulation buffer (index 14) in `get_internal_buffer_descs`; zero-fill `acc_f32` before expert loop in `exec_prefill_onednn`; add `scatter_f32_to_f16` stage after expert loop; add dummy buffer slots 9-13 when not using micro_gemm for consistent indexing
8. **moe_3gemm_base.hpp** — Added `#define MOE_INTERNAL_BUFFER_ACC_F32 14`

## Files Reference

**optimum-intel:**
- `optimum/exporters/openvino/model_patcher.py` — All MoE patchers live here
- `optimum/exporters/openvino/model_configs.py` — Patcher registration

**OpenVINO (transformation passes):**
- `src/common/transformations/.../moe_transpose_weights.cpp` — VectorizedMOE3GEMMTransposeWeights
- `src/common/transformations/.../matmul_experts_fusion.cpp` — FuseVectorizedMOE3GEMM
- `src/plugins/intel_gpu/.../transformations_pipeline.cpp` — GPU pass registration & HW gates
- `src/plugins/intel_gpu/.../convert_moe_to_compressed.cpp` — ConvertMOEToMOECompressed
- `src/plugins/intel_gpu/.../fuse_moe_3gemm_compressed.cpp` — FuseMOE3GemmCompressed (the strict one)
- `src/plugins/intel_gpu/.../convert_matmul_to_fc.cpp` — ConvertMatMulToFullyConnected (Transpose fix)
- `src/plugins/intel_gpu/.../moe_3gemm_swiglu_fuse.cl` — Routing & scatter kernels (softmax_topk, sigmoid_bias_topk, gather, index_add_, scatter_f32_to_f16, swiglu)
- `src/plugins/intel_gpu/.../moe/moe_3gemm_swiglu_opt.cpp` — Fused MoE kernel orchestration (prefill onednn & micro_gemm paths)
- `src/plugins/intel_gpu/.../moe/moe_3gemm_base.hpp` — Internal buffer index definitions
- `src/plugins/intel_gpu/.../moe/moe_scatter_reduction_opt.cl` — micro_gemm scatter-reduce (FP32 accumulation)
