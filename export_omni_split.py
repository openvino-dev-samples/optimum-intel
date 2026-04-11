"""
Export the Z-Image-Omni transformer as split sub-models.

The transformer contains dynamic logic (variable-length sequences, data-dependent
noise masks) that OpenVINO static graph tracing cannot handle correctly.

Solution: Split into 5 sub-models, keeping dynamic logic in Python pipeline code.

Sub-models:
1. patch_embed   - t_embedder + x_embedder + cap_embedder + siglip_embedder + pad tokens
2. noise_refiner - 2 ZImageTransformerBlock (modulation=True)
3. context_refiner - 2 ZImageTransformerBlock (modulation=False)
4. siglip_refiner - 2 ZImageTransformerBlock (modulation=False)  
5. main_transformer - 30 ZImageTransformerBlock + FinalLayer (modulation=True)

RoPE embedding + patchify + sequence building + unpatchify = Python pipeline code.
"""

import sys
import os
import gc
import time
import shutil
import types
import json
import subprocess

sys.path.insert(0, r"d:\optimum-intel")

OMNI_SRC = r"d:\optimum-intel\Z-Image-Omni-Base"
OMNI_OV = r"d:\optimum-intel\Z-Image-Omni-Base-OV"
PYTHON = r"d:\optimum-intel\py_env\Scripts\python.exe"


def export_submodel(step_name):
    """Run a subprocess to export one sub-model."""
    print(f"\n{'='*70}")
    print(f"Exporting: {step_name}")
    print(f"{'='*70}", flush=True)
    
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, __file__, step_name],
        cwd=r"d:\optimum-intel",
        timeout=7200,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED: {step_name} (exit code {result.returncode}) in {elapsed:.1f}s")
        return False
    print(f"DONE: {step_name} in {elapsed:.1f}s")
    return True


def _load_transformer():
    """Load the PyTorch transformer model."""
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
    import torch
    
    print("Loading transformer from source...", flush=True)
    transformer = ZImageTransformer2DModel.from_pretrained(
        os.path.join(OMNI_SRC, "transformer"),
        torch_dtype=torch.float32,
    )
    transformer.eval()
    print(f"Loaded transformer: {sum(p.numel() for p in transformer.parameters())/1e9:.2f}B params", flush=True)
    return transformer


def _export_ov(model_wrapper, dummy_inputs, output_dir, input_names=None, output_names=None, dynamic_axes=None):
    """Export a model wrapper to OpenVINO IR."""
    import torch
    import openvino as ov
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Tracing with torch.export + OV convert...", flush=True)
    
    # Use openvino.convert_model with torch module
    example_inputs = tuple(dummy_inputs[k] for k in dummy_inputs)
    
    ov_model = ov.convert_model(
        model_wrapper,
        example_input=example_inputs,
    )
    
    # Rename inputs/outputs
    if input_names:
        for i, name in enumerate(input_names):
            ov_model.inputs[i].set_names({name})
    if output_names:
        for i, name in enumerate(output_names):
            ov_model.outputs[i].set_names({name})
    
    # Set dynamic shapes if specified
    if dynamic_axes:
        for inp in ov_model.inputs:
            name = next(iter(inp.get_names()))
            if name in dynamic_axes:
                shape = list(inp.get_partial_shape())
                for axis_idx in dynamic_axes[name]:
                    shape[axis_idx] = -1
                inp.get_node().set_partial_shape(ov.PartialShape(shape))
        ov_model.validate_nodes_and_infer_types()
    
    # Save
    xml_path = os.path.join(output_dir, "openvino_model.xml")
    ov.save_model(ov_model, xml_path, compress_to_fp16=False)
    print(f"  Saved to {output_dir}", flush=True)
    
    return ov_model


def export_patch_embed():
    """Export patch_embed sub-model: embedders + pad token application."""
    import torch
    import torch.nn as nn
    
    transformer = _load_transformer()
    
    class PatchEmbedModel(nn.Module):
        def __init__(self, t_embedder, x_embedder_key, all_x_embedder, cap_embedder, 
                     siglip_embedder, x_pad_token, cap_pad_token, siglip_pad_token, t_scale):
            super().__init__()
            self.t_embedder = t_embedder
            self.x_embedder = all_x_embedder[x_embedder_key]
            self.cap_embedder = cap_embedder
            self.siglip_embedder = siglip_embedder
            self.x_pad_token = x_pad_token
            self.cap_pad_token = cap_pad_token
            self.siglip_pad_token = siglip_pad_token
            self.t_scale = t_scale
            
        def forward(self, x_patches, x_pad_mask, cap_feats, cap_pad_mask,
                    sig_feats, sig_pad_mask, timestep):
            # Embed timestep
            t_noisy = self.t_embedder(timestep * self.t_scale)
            t_clean = self.t_embedder(torch.ones_like(timestep) * self.t_scale)
            
            # Embed and apply pad tokens
            x_emb = self.x_embedder(x_patches)
            x_emb = torch.where(x_pad_mask.unsqueeze(-1), self.x_pad_token, x_emb)
            
            cap_emb = self.cap_embedder(cap_feats)
            cap_emb = torch.where(cap_pad_mask.unsqueeze(-1), self.cap_pad_token, cap_emb)
            
            sig_emb = self.siglip_embedder(sig_feats)
            sig_emb = torch.where(sig_pad_mask.unsqueeze(-1), self.siglip_pad_token, sig_emb)
            
            return x_emb, cap_emb, sig_emb, t_noisy, t_clean
    
    model = PatchEmbedModel(
        transformer.t_embedder,
        "2-1",
        transformer.all_x_embedder,
        transformer.cap_embedder,
        transformer.siglip_embedder,
        transformer.x_pad_token,
        transformer.cap_pad_token,
        transformer.siglip_pad_token,
        transformer.t_scale,
    ).eval()
    
    dim = transformer.dim  # 3840
    cap_dim = transformer.config.cap_feat_dim  # 2560
    sig_dim = transformer.config.siglip_feat_dim  # 1152
    in_ch = transformer.config.in_channels  # 16
    patch_dim = 2 * 2 * 1 * in_ch  # 64
    
    dummy = {
        "x_patches": torch.randn(256, patch_dim),
        "x_pad_mask": torch.zeros(256, dtype=torch.bool),
        "cap_feats": torch.randn(128, cap_dim),
        "cap_pad_mask": torch.zeros(128, dtype=torch.bool),
        "sig_feats": torch.randn(288, sig_dim),
        "sig_pad_mask": torch.zeros(288, dtype=torch.bool),
        "timestep": torch.tensor([0.5]),
    }
    
    out_dir = os.path.join(OMNI_OV, "transformer_patch_embed")
    _export_ov(
        model, dummy, out_dir,
        input_names=["x_patches", "x_pad_mask", "cap_feats", "cap_pad_mask",
                      "sig_feats", "sig_pad_mask", "timestep"],
        output_names=["x_emb", "cap_emb", "sig_emb", "t_noisy", "t_clean"],
        dynamic_axes={"x_patches": [0], "x_pad_mask": [0], "cap_feats": [0], 
                       "cap_pad_mask": [0], "sig_feats": [0], "sig_pad_mask": [0]},
    )
    
    # Save config
    config = {
        "dim": dim, "cap_feat_dim": cap_dim, "siglip_feat_dim": sig_dim,
        "in_channels": in_ch, "patch_dim": patch_dim,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("patch_embed exported successfully", flush=True)


def _make_refiner_wrapper(blocks, has_modulation):
    """Create a wrapper module for refiner blocks."""
    import torch
    import torch.nn as nn
    
    class RefinerWithModulation(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = nn.ModuleList(blocks)
            
        def forward(self, x, attn_mask, freqs_cis, noise_mask, adaln_noisy, adaln_clean):
            for block in self.blocks:
                x = block(x, attn_mask, freqs_cis, noise_mask=noise_mask,
                         adaln_noisy=adaln_noisy, adaln_clean=adaln_clean)
            return x
    
    class RefinerNoModulation(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.blocks = nn.ModuleList(blocks)
            
        def forward(self, x, attn_mask, freqs_cis):
            for block in self.blocks:
                x = block(x, attn_mask, freqs_cis)
            return x
    
    if has_modulation:
        return RefinerWithModulation(blocks)
    else:
        return RefinerNoModulation(blocks)


def export_noise_refiner():
    """Export noise_refiner sub-model: 2 blocks with dual modulation."""
    import torch
    
    transformer = _load_transformer()
    
    # Patch attention processors for OV-compatible RoPE
    _patch_attention_processors(transformer)
    
    model = _make_refiner_wrapper(list(transformer.noise_refiner), has_modulation=True).eval()
    
    dim = transformer.dim
    adaln_dim = min(dim, 256)
    n_heads = transformer.config.n_heads
    head_dim = dim // n_heads  # 128
    freq_dim = sum(transformer.config.axes_dims)  # 128 = 32+48+48
    seq_len = 320  # example
    
    dummy = {
        "x": torch.randn(1, seq_len, dim),
        "attn_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "freqs_cis": torch.randn(1, seq_len, freq_dim, 2),
        "noise_mask": torch.randint(0, 2, (1, seq_len), dtype=torch.long),
        "adaln_noisy": torch.randn(1, adaln_dim),
        "adaln_clean": torch.randn(1, adaln_dim),
    }
    
    out_dir = os.path.join(OMNI_OV, "transformer_noise_refiner")
    _export_ov(
        model, dummy, out_dir,
        input_names=["x", "attn_mask", "freqs_cis", "noise_mask", "adaln_noisy", "adaln_clean"],
        output_names=["output"],
        dynamic_axes={"x": [1], "attn_mask": [1], "freqs_cis": [1], "noise_mask": [1]},
    )
    print("noise_refiner exported successfully", flush=True)


def export_context_refiner():
    """Export context_refiner sub-model: 2 blocks without modulation."""
    import torch
    
    transformer = _load_transformer()
    _patch_attention_processors(transformer)
    
    model = _make_refiner_wrapper(list(transformer.context_refiner), has_modulation=False).eval()
    
    dim = transformer.dim
    freq_dim = sum(transformer.config.axes_dims)  # 128
    seq_len = 128
    
    dummy = {
        "x": torch.randn(1, seq_len, dim),
        "attn_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "freqs_cis": torch.randn(1, seq_len, freq_dim, 2),
    }
    
    out_dir = os.path.join(OMNI_OV, "transformer_context_refiner")
    _export_ov(
        model, dummy, out_dir,
        input_names=["x", "attn_mask", "freqs_cis"],
        output_names=["output"],
        dynamic_axes={"x": [1], "attn_mask": [1], "freqs_cis": [1]},
    )
    print("context_refiner exported successfully", flush=True)


def export_siglip_refiner():
    """Export siglip_refiner sub-model: 2 blocks without modulation."""
    import torch
    
    transformer = _load_transformer()
    _patch_attention_processors(transformer)
    
    model = _make_refiner_wrapper(list(transformer.siglip_refiner), has_modulation=False).eval()
    
    dim = transformer.dim
    freq_dim = sum(transformer.config.axes_dims)  # 128
    seq_len = 288
    
    dummy = {
        "x": torch.randn(1, seq_len, dim),
        "attn_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "freqs_cis": torch.randn(1, seq_len, freq_dim, 2),
    }
    
    out_dir = os.path.join(OMNI_OV, "transformer_siglip_refiner")
    _export_ov(
        model, dummy, out_dir,
        input_names=["x", "attn_mask", "freqs_cis"],
        output_names=["output"],
        dynamic_axes={"x": [1], "attn_mask": [1], "freqs_cis": [1]},
    )
    print("siglip_refiner exported successfully", flush=True)


def export_main_transformer():
    """Export main_transformer: 30 blocks + final_layer with dual modulation."""
    import torch
    import torch.nn as nn
    
    transformer = _load_transformer()
    _patch_attention_processors(transformer)
    
    class MainTransformerModel(nn.Module):
        def __init__(self, layers, final_layer):
            super().__init__()
            self.layers = layers
            self.final_layer = final_layer
            
        def forward(self, x, attn_mask, freqs_cis, noise_mask, adaln_noisy, adaln_clean):
            for layer in self.layers:
                x = layer(x, attn_mask, freqs_cis, noise_mask=noise_mask,
                         adaln_noisy=adaln_noisy, adaln_clean=adaln_clean)
            x = self.final_layer(x, noise_mask=noise_mask, c_noisy=adaln_noisy, c_clean=adaln_clean)
            return x
    
    model = MainTransformerModel(
        transformer.layers,
        transformer.all_final_layer["2-1"],
    ).eval()
    
    dim = transformer.dim
    adaln_dim = min(dim, 256)
    freq_dim = sum(transformer.config.axes_dims)  # 128
    out_ch = 2 * 2 * 1 * transformer.out_channels  # 64
    seq_len = 800  # example
    
    dummy = {
        "x": torch.randn(1, seq_len, dim),
        "attn_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "freqs_cis": torch.randn(1, seq_len, freq_dim, 2),
        "noise_mask": torch.randint(0, 2, (1, seq_len), dtype=torch.long),
        "adaln_noisy": torch.randn(1, adaln_dim),
        "adaln_clean": torch.randn(1, adaln_dim),
    }
    
    out_dir = os.path.join(OMNI_OV, "transformer_main")
    _export_ov(
        model, dummy, out_dir,
        input_names=["x", "attn_mask", "freqs_cis", "noise_mask", "adaln_noisy", "adaln_clean"],
        output_names=["output"],
        dynamic_axes={"x": [1], "attn_mask": [1], "freqs_cis": [1], "noise_mask": [1]},
    )
    print("main_transformer exported successfully", flush=True)


def _patch_attention_processors(transformer):
    """Patch attention processors to use OV-compatible RoPE (cos/sin instead of complex)."""
    from diffusers.models.transformers import transformer_z_image
    from optimum.exporters.openvino.model_patcher import (
        _zimage_rope_embedder_precompute_freqs_cis,
        _zimage_rope_embedder_call,
        _zimage_attn_processor_call,
    )
    
    orig_cls = transformer_z_image.ZSingleStreamAttnProcessor
    
    class PatchedAttnProcessor(orig_cls):
        def __call__(self_inner, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, freqs_cis=None):
            return _zimage_attn_processor_call(
                self_inner, attn, hidden_states, encoder_hidden_states,
                attention_mask, freqs_cis
            )
    
    # Patch all blocks
    for block_list in [transformer.noise_refiner, transformer.context_refiner, 
                       transformer.layers]:
        for layer in block_list:
            layer.attention.processor = PatchedAttnProcessor()
    
    if transformer.siglip_refiner is not None:
        for layer in transformer.siglip_refiner:
            layer.attention.processor = PatchedAttnProcessor()


def verify_all():
    """Verify all sub-models can be loaded."""
    import openvino as ov
    
    core = ov.Core()
    sub_models = [
        "transformer_patch_embed",
        "transformer_noise_refiner", 
        "transformer_context_refiner",
        "transformer_siglip_refiner",
        "transformer_main",
    ]
    
    for name in sub_models:
        xml = os.path.join(OMNI_OV, name, "openvino_model.xml")
        if os.path.exists(xml):
            model = core.read_model(xml)
            bin_size = os.path.getsize(xml.replace('.xml', '.bin')) / (1024*1024)
            print(f"✓ {name}: {len(model.inputs)} inputs, {len(model.outputs)} outputs, {bin_size:.1f} MB")
            for inp in model.inputs:
                n = next(iter(inp.get_names()))
                print(f"    input: {n} {inp.get_partial_shape()}")
            for out in model.outputs:
                n = next(iter(out.get_names()))
                print(f"    output: {n} {out.get_partial_shape()}")
        else:
            print(f"✗ {name}: NOT FOUND")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Master process: run all exports in subprocesses
        steps = [
            "patch_embed",
            "noise_refiner",
            "context_refiner",
            "siglip_refiner",
            "main_transformer",
        ]
        
        results = {}
        for step in steps:
            ok = export_submodel(step)
            results[step] = "PASS" if ok else "FAIL"
        
        print(f"\n{'='*70}")
        print("EXPORT SUMMARY")
        print(f"{'='*70}")
        for step, status in results.items():
            print(f"  {step}: {status}")
        
        # Verify
        print(f"\n{'='*70}")
        print("VERIFICATION")
        print(f"{'='*70}")
        
        subprocess.run([PYTHON, __file__, "verify"])
        
    else:
        step = sys.argv[1]
        dispatch = {
            "patch_embed": export_patch_embed,
            "noise_refiner": export_noise_refiner,
            "context_refiner": export_context_refiner,
            "siglip_refiner": export_siglip_refiner,
            "main_transformer": export_main_transformer,
            "verify": verify_all,
        }
        if step in dispatch:
            dispatch[step]()
        else:
            print(f"Unknown step: {step}")
            sys.exit(1)
