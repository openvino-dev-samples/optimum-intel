# MoE Patcher Template

This is an annotated template for writing MoE forward patchers in optimum-intel that produce IR graphs compatible with OpenVINO's GPU MoE fusion pipeline.

## Table of Contents
1. [Forward Function Template](#forward-function-template)
2. [Model Patcher Class Template](#model-patcher-class-template)
3. [PyTorch → OV IR Mapping](#pytorch--ov-ir-mapping)
4. [Common Pitfalls](#common-pitfalls)

## Forward Function Template

```python
def my_model_moe_forward_patched(self, hidden_states):
    """Vectorized MoE forward producing a graph matching OV GPU MoE fusion.

    Fusion pipeline:
    1. VectorizedMOE3GEMMTransposeWeights (MOC) — MatMul(tb=false→true)
    2. FuseVectorizedMOE3GEMM (GPU) — expert computation → MOE(GEMM3_SWIGLU)
    3. ConvertMOEToMOECompressed (GPU) — compressed weights
    4. FuseMOE3GemmCompressed (GPU) — routing + MOECompressed → fused

    Computational equivalence:
      Original: output = scaling * sum_i(expert_i(x) * w_i)
      Patched:  output = [sum_i(expert_i(x) * w_i/sum_w)] * sum_w * scaling
    """
    # ── Extract config ──
    num_experts = self.config.n_routed_experts
    top_k = self.config.num_experts_per_tok
    # scaling_factor may not exist for all models (e.g., Mixtral normalizes instead)
    scaling_factor = getattr(self.config, 'routed_scaling_factor', 1.0)
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # ── Reshape 3D→2D ──
    # This becomes `hidden_state_reshape` in the fusion pattern.
    # Both the router MatMul and the expert Tile consume this node.
    hidden_states_flat = hidden_states.view(-1, hidden_dim)

    # ════════════════════════════════════════════════
    # ROUTING SUBGRAPH — must match FuseMOE3GemmCompressed
    # ════════════════════════════════════════════════

    # Router linear: F.linear traces to MatMul(transpose_b=true)
    # IMPORTANT: Do NOT cast hidden_states_flat — a Convert between
    # hidden_state_reshape and matmul breaks the pattern match.
    # Only cast the gate weight if needed for precision.
    logits = torch.nn.functional.linear(
        hidden_states_flat, self.gate.weight.float()
    )

    # Softmax: traces to ov::Softmax
    scores = logits.softmax(dim=-1)

    # TopK: traces to ov::TopK (produces values + indices)
    topk_weights, topk_indices = torch.topk(
        scores, k=top_k, dim=-1, sorted=False
    )

    # ── Normalization: ReduceSum → Divide ──
    # The ReduceSum (sm_reduce) MUST have exactly 1 consumer (the Divide).
    # This is the #1 failure point for fusion.
    topk_sum = topk_weights.sum(dim=-1, keepdim=True)
    topk_norm = topk_weights / topk_sum

    # ── Scatter routing weights to full expert matrix ──
    # torch.zeros → Broadcast, scatter_ → ScatterElementsUpdate + Slice
    new_routing_weights = torch.zeros(
        batch_size * seq_len, num_experts, dtype=topk_norm.dtype
    )
    new_routing_weights.scatter_(
        dim=1, index=topk_indices, src=topk_norm
    )

    # ── Correction factor (if model doesn't already normalize) ──
    # Uses CumSum + Slice instead of ReduceSum to avoid merging with
    # the normalization ReduceSum. cumsum(w)[:,-1:] == w.sum(keepdim=True)
    correction_sum = topk_weights.cumsum(dim=-1)[:, -1:]
    correction = (correction_sum * scaling_factor).view(
        batch_size, seq_len, 1
    )

    # ══════════════════════════════════════════════════════════════
    # ALTERNATIVE: SIGMOID+BIAS ROUTING (MiniCPM5-MoE style)
    # Uncomment this block and comment the softmax block above
    # if the model uses sigmoid+bias routing.
    # ══════════════════════════════════════════════════════════════
    # # Sigmoid: logits → Sigmoid (consumers: Add for biased TopK, GatherElements for raw scores)
    # sigmoid_scores = torch.sigmoid(logits)
    #
    # # Add bias for TopK selection (traces: Sigmoid → Add → TopK)
    # biased_scores = sigmoid_scores + self.e_score_correction_bias
    # topk_weights, topk_indices = torch.topk(biased_scores, k=top_k, dim=-1, sorted=False)
    #
    # # Gather ORIGINAL (unbiased) sigmoid scores for normalization
    # # traces: Sigmoid → GatherElements(indices) — separate consumer of Sigmoid
    # original_topk = torch.gather(sigmoid_scores, dim=1, index=topk_indices)
    #
    # # Normalize: sum + eps → divide (traces: ReduceSum → Add(eps) → Divide)
    # topk_sum = original_topk.sum(dim=-1, keepdim=True)
    # topk_norm = original_topk / (topk_sum + 1e-20)
    #
    # # Scatter (same as softmax branch)
    # new_routing_weights = torch.zeros(batch_size * seq_len, num_experts, dtype=topk_norm.dtype)
    # new_routing_weights.scatter_(dim=1, index=topk_indices, src=topk_norm)
    #
    # # Correction factor
    # correction_sum = original_topk.cumsum(dim=-1)[:, -1:]
    # correction = (correction_sum * scaling_factor).view(batch_size, seq_len, 1)

    # ════════════════════════════════════════════════
    # SHARED EXPERTS (if model has them)
    # ════════════════════════════════════════════════
    # Compute shared experts separately — they're outside the fused pattern.
    if hasattr(self, 'shared_experts'):
        shared_output = self.shared_experts(hidden_states_flat)

    # ════════════════════════════════════════════════
    # EXPERT COMPUTATION — must match FuseVectorizedMOE3GEMM
    # ════════════════════════════════════════════════

    # Tile + Reshape: .repeat() → Tile, .view() → Reshape
    hidden_states_rep = hidden_states_flat.repeat(num_experts, 1).view(
        num_experts, -1, hidden_dim
    )

    # 3× BMM with explicit transpose: traces to MatMul + Transpose on weight input.
    # Weight Constants stay in [E, N, K] format (original PyTorch layout).
    # The .transpose(-1,-2) traces as a Transpose node in IR.
    # VectorizedMOE3GEMMTransposeWeights or TransposeMatMul absorbs it → MatMul(tb=true).
    # This preserves [E, N, K] layout so NNCF quantizes to [E, N, K/G, G] —
    # exactly what ConvertMOEToMOECompressed expects.
    gate = torch.bmm(hidden_states_rep, self.gate_projs.transpose(-1, -2))
    up = torch.bmm(hidden_states_rep, self.up_projs.transpose(-1, -2))

    # SwiGLU: silu(gate) * up
    # silu → ov::Swish, * → ov::Multiply
    gate_up = torch.nn.functional.silu(gate) * up

    down = torch.bmm(gate_up, self.down_projs.transpose(-1, -2))

    # Reshape(4D) → Multiply(routing) → ReduceSum(keep_dims=false)
    # This is the pattern tail that FuseVectorizedMOE3GEMM matches.
    down = down.view(num_experts, batch_size, -1, hidden_dim)
    down = down * new_routing_weights.transpose(0, 1).view(
        num_experts, batch_size, -1
    )[..., None]
    result = down.sum(dim=0)  # keep_dims=false (default)

    # ════════════════════════════════════════════════
    # POST-FUSION CORRECTION
    # ════════════════════════════════════════════════
    result = result * correction

    # Add shared experts if present
    if hasattr(self, 'shared_experts'):
        shared_output = shared_output.view(batch_size, -1, hidden_dim)
        result = shared_output + result

    return result.view(batch_size, seq_len, hidden_dim)
```

## Model Patcher Class Template

```python
class MyModelMoEPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        for layer in self._model.model.layers:
            if hasattr(layer.mlp, "experts"):
                moe = layer.mlp

                # Stack expert weights into batched matrices
                experts = moe.experts
                num_experts = len(experts)

                # CRITICAL: Keep original [E, N, K] layout (no transpose!)
                # PyTorch nn.Linear weights are [out_features, in_features] = [N, K]
                # Stacking gives [E, N, K] — this is what ConvertMOEToMOECompressed expects.
                # The transpose to [E, K, N] for bmm happens in the forward pass,
                # which traces as a Transpose node that TransposeMatMul absorbs.
                # NNCF then quantizes [E, N, K] → [E, N, K/G, G] — correct layout.
                gate_projs = torch.stack(
                    [e.gate_proj.weight for e in experts]
                ).float()

                up_projs = torch.stack(
                    [e.up_proj.weight for e in experts]
                ).float()

                down_projs = torch.stack(
                    [e.down_proj.weight for e in experts]
                ).float()

                # Register as buffers so they trace correctly
                moe.register_buffer("gate_projs", gate_projs)
                moe.register_buffer("up_projs", up_projs)
                moe.register_buffer("down_projs", down_projs)

                # Patch the forward method
                moe._orig_forward = moe.forward
                moe.forward = types.MethodType(
                    my_model_moe_forward_patched, moe
                )

    def __exit__(self, exc_type, exc_val, exc_tb):
        for layer in self._model.model.layers:
            if hasattr(layer.mlp, "experts"):
                moe = layer.mlp
                if hasattr(moe, "_orig_forward"):
                    moe.forward = moe._orig_forward
                    del moe._orig_forward
                for attr in ["gate_projs", "up_projs", "down_projs"]:
                    if hasattr(moe, attr):
                        delattr(moe, attr)
        super().__exit__(exc_type, exc_val, exc_tb)
```

## PyTorch → OV IR Mapping

| PyTorch Operation | OV IR Node | Notes |
|---|---|---|
| `F.linear(x, w)` | `MatMul(transpose_b=true)` | Weight already transposed |
| `torch.bmm(a, b.transpose(-1,-2))` | `MatMul + Transpose` → `MatMul(transpose_b=true)` | Transpose absorbed by TransposeMatMul or VectorizedMOE3GEMMTransposeWeights |
| `x.softmax(dim=-1)` | `Softmax` | |
| `torch.topk(x, k)` | `TopK` | Produces values + indices outputs |
| `x.sum(dim, keepdim)` | `ReduceSum` | keepdim maps to keep_dims attr |
| `x / y` | `Divide` | |
| `torch.zeros(shape)` | `Broadcast` | With Const(0) input |
| `x.scatter_(dim, idx, src)` | `ScatterElementsUpdate` | Frontend adds Slice node |
| `x.repeat(n, 1)` | `Tile` | |
| `x.view(shape)` | `Reshape` | |
| `x.transpose(a, b)` | `Transpose` | |
| `x.unsqueeze(dim)` | `Unsqueeze` | |
| `F.silu(x)` | `Swish` | |
| `x * y` | `Multiply` | |
| `x.cumsum(dim)` | `CumSum` | Used for correction trick |
| `x[:, -1:]` | `Slice` / `StridedSlice` | |

## Common Pitfalls

### 1. ReduceSum Merging
**Problem:** Two `.sum()` calls on the same tensor merge into one node.
**Failed attempts:** `.clone().sum()` (clone optimized away), `matmul(x, ones)` (NNCF quantization error)
**Solution:** `cumsum(dim=-1)[:, -1:]` — different op type, can't merge.

### 2. Gate Weight Casting
**Problem:** `F.linear(hidden.float(), weight)` inserts Convert before router MatMul.
**Solution:** Cast only the weight: `F.linear(hidden, weight.float())`

### 3. Per-Expert Loop
**Problem:** Looping over experts produces 160 separate MatMul subgraphs.
**Solution:** Stack weights, use `torch.bmm()` for vectorized computation.

### 4. Complex Gate Subgraph
**Problem:** Group masking, score normalization create extra nodes between Softmax and TopK.
**Solution:** Inline the gate computation when n_group=1. For n_group>1, carefully restructure.

### 5. Custom OV Wheel Overwritten
**Problem:** `pip install optimum-intel` pulls openvino from PyPI, overwriting custom build.
**Solution:** Always install custom wheel AFTER optimum-intel with `--no-deps`.

### 6. Pre-Transposed Weights (Most Critical!)
**Problem:** `.transpose(1, 2)` in `__enter__` bakes transposed values into Constants as `[E, K, N]`. NNCF then quantizes to `[E, K/G, G, N]` — wrong layout for ConvertMOEToMOECompressed which expects `[E, N, K/G, G]`.
**Symptom:** `MOE(extension) is not supported` or Reshape shape inference error in ConvertMOEToMOECompressed.
**Solution:** Keep weights in original `[E, N, K]` layout in `__enter__`, use `.transpose(-1, -2)` in forward's `torch.bmm()` calls. The Transpose node gets absorbed by TransposeMatMul.

### 7. Wrong Activation Function
**Problem:** Model uses GELU instead of SiLU — pattern expects Swish.
**Solution:** Check which activation the model uses. The current fusion only supports SwiGLU (SiLU * up). If the model uses a different activation, the pattern won't match and you may need to modify the OV transformation pass.

### 8. GPU FP16 Routing Precision (SigmoidBias models)
**Problem:** For SigmoidBias routing (MiniCPM5-MoE), the gate MatMul `[batch, hidden_dim] x [hidden_dim, num_experts]` runs in FP16 on GPU. With K=2048, FP16 accumulation error (~0.07) far exceeds the inter-expert sigmoid gap (~0.0004), causing TopK to select completely wrong experts. The `e_score_correction_bias` (~92.5) amplifies the issue — biased scores cluster where FP16 ULP = 0.0625.
**Symptom:** Cosine similarity between CPU and GPU is ~-0.07 (anti-correlated). GPU generates garbage text while CPU is fine.
**Fix:** The OV GPU plugin computes the gate GEMV in FP32 inside the fused `sigmoid_bias_topk` kernel. Gate weight is passed as Input 13 (ROUTING_GATE_WEIGHT). All files with hardcoded input counts must be updated (see checklist in pitfall #9).
**Result after fix:** Cosine 0.06 → 0.97, identical top-1 token, matching text generation.

### 9. KeepMOE3GemmConstPrecision Pattern Mismatch
**Problem:** Adding a new input to MOE3GemmFusedCompressed without updating `keep_moe_3gemm_const_precision.cpp` causes silent failure — u4 weight/zp Constants lose `keep_const_precision`, get type-converted, produce all-NaN output.
**Symptom:** All NaN output, no error message. Kernel runs but reads garbage data.
**Fix:** Update the `wrap_type<MOE3GemmFusedCompressed>({...})` pattern to match exact input count (11 for Softmax, 14 for SigmoidBias).
**Checklist when adding a new fused input:**
1. `moe_3gemm_base.hpp` — input enum
2. `moe_3gemm_fused_compressed.cpp` — expected_inputs validation
3. `moe_3gemm_fused_compressed.hpp` — op + primitive docs
4. `fuse_moe_3gemm_compressed.cpp` — pattern capture
5. `keep_moe_3gemm_const_precision.cpp` — **input count in wrap_type pattern**
6. `ops/moe.cpp` — validate_inputs_count
7. Kernel `.cl` file — new parameter
8. Host `.cpp` file — JIT constants + execute_stage inputs
9. Unit test — push new arg

### 10. INT4 Export via Python API Does Not Quantize
**Problem:** Using `main_export()` with `weight_format='int4'` produces INT8 instead of INT4.
**Symptom:** Model file size matches INT8 size, GPU INT4 == GPU INT8 outputs exactly.
**Solution:** Use the CLI: `optimum-cli export openvino --weight-format int4`. The CLI properly calls NNCF weight compression. When using Python API, pass `OVConfig(quantization_config={...})` but quantization may only apply via the `_main_quantize` path, not `main_export`.
