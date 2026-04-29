"""Step-by-step decode diagnostic for GLM-OCR on OV vs HF PyTorch.

For a given image + prompt, runs 8 manual decode steps on both backends and
reports per-step top-1 and logits max-diff. Also dumps the ``position_ids``
tensor that each backend hands to its language-model forward on each step.

Usage:
    python examples/openvino/glm_ocr/diag_decode.py \\
        --ov_model glm_ocr_ov --pt_model D:/optimum-intel/GLM-OCR \\
        --image sample.png --steps 8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForImageTextToText
from optimum.intel.openvino import OVModelForVisualCausalLM


def build_inputs(processor, image_path: str, prompt: str):
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs


def _install_dump_hook(pytorch_model):
    """Record position_ids that HF GlmOcrTextModel.forward sees per step."""
    from transformers.models.glm_ocr import modeling_glm_ocr

    recorded: list[torch.Tensor] = []
    orig = modeling_glm_ocr.GlmOcrTextModel.forward

    def wrapper(self, *args, **kwargs):
        pids = kwargs.get("position_ids")
        if pids is not None:
            recorded.append(pids.detach().clone())
        return orig(self, *args, **kwargs)

    modeling_glm_ocr.GlmOcrTextModel.forward = wrapper
    return recorded, lambda: setattr(modeling_glm_ocr.GlmOcrTextModel, "forward", orig)


def _install_ov_dump_hook(ov_model):
    """Record position_ids that the OV LM sees per step (numpy array)."""
    from optimum.intel.openvino.modeling_visual_language import OVModelWithEmbedForCausalLM

    recorded: list[np.ndarray] = []
    orig = OVModelWithEmbedForCausalLM.prepare_inputs

    def wrapper(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        pid = out.get("position_ids")
        if pid is not None:
            recorded.append(np.asarray(pid).copy())
        return out

    OVModelWithEmbedForCausalLM.prepare_inputs = wrapper
    return recorded, lambda: setattr(OVModelWithEmbedForCausalLM, "prepare_inputs", orig)


@torch.no_grad()
def pt_decode(model, inputs, steps: int):
    out = model(**inputs, use_cache=True, return_dict=True)
    logits = out.logits.float()[:, -1:, :]
    all_logits = [logits]
    next_tok = logits.argmax(-1)
    tokens = [int(next_tok.item())]

    input_ids = next_tok
    attention_mask = torch.cat(
        [inputs["attention_mask"], torch.ones_like(next_tok)], dim=1
    )
    past_key_values = out.past_key_values

    for _ in range(steps - 1):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        logits = out.logits.float()[:, -1:, :]
        all_logits.append(logits)
        next_tok = logits.argmax(-1)
        tokens.append(int(next_tok.item()))
        input_ids = next_tok
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_tok)], dim=1)
        past_key_values = out.past_key_values
    return torch.cat(all_logits, dim=1), tokens


def ov_decode(model, inputs, steps: int):
    """Run ``steps`` decode steps using the public ``.generate()`` path but
    capture per-step logits via a hook on the LM forward."""
    from optimum.intel.openvino.modeling_visual_language import OVModelWithEmbedForCausalLM

    captured: list[torch.Tensor] = []
    orig_forward = OVModelWithEmbedForCausalLM.forward

    def wrapper(self, *args, **kwargs):
        res = orig_forward(self, *args, **kwargs)
        captured.append(res.logits[:, -1:, :].float().cpu())
        return res

    OVModelWithEmbedForCausalLM.forward = wrapper
    try:
        gen = model.generate(**inputs, max_new_tokens=steps, do_sample=False)
    finally:
        OVModelWithEmbedForCausalLM.forward = orig_forward

    prefill_len = inputs["input_ids"].shape[1]
    new_tokens = gen[0, prefill_len:].tolist()
    logits = torch.cat(captured, dim=1)  # [1, steps, V]; step 0 is prefill last token
    return logits, new_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ov_model", required=True)
    parser.add_argument("--pt_model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Text Recognition:")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--skip_pt", action="store_true")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.pt_model, trust_remote_code=True)
    inputs = build_inputs(processor, args.image, args.prompt)
    print("input_ids.shape:", inputs["input_ids"].shape)
    if "image_grid_thw" in inputs:
        print("image_grid_thw:", inputs["image_grid_thw"].tolist())

    print("\n=== OV CPU (position_ids dump + logits) ===")
    ov_model = OVModelForVisualCausalLM.from_pretrained(args.ov_model, device="CPU")
    ov_pids, ov_restore = _install_ov_dump_hook(ov_model)
    try:
        t0 = time.time()
        ov_logits, ov_tokens = ov_decode(ov_model, {k: v for k, v in inputs.items()}, args.steps)
        ov_dt = time.time() - t0
    finally:
        ov_restore()
    print(f"OV decoded {args.steps} tokens in {ov_dt:.1f}s "
          f"({ov_dt / max(1, args.steps - 1) * 1000:.0f} ms/tok incl prefill)")
    print("OV tokens:", ov_tokens)
    for i, pid in enumerate(ov_pids[: args.steps + 1]):
        print(f"  step {i}: position_ids shape={pid.shape} first-row[:6]={pid[0, 0, :6] if pid.ndim == 3 else pid[0, :6]}")

    if args.skip_pt:
        return

    print("\n=== PyTorch CPU (position_ids dump + logits) ===")
    pt_model = AutoModelForImageTextToText.from_pretrained(
        args.pt_model, torch_dtype=torch.float32, trust_remote_code=True
    ).eval()
    pt_pids, pt_restore = _install_dump_hook(pt_model)
    try:
        t0 = time.time()
        pt_logits, pt_tokens = pt_decode(pt_model, inputs, args.steps)
        pt_dt = time.time() - t0
    finally:
        pt_restore()
    print(f"PT decoded {args.steps} tokens in {pt_dt:.1f}s")
    print("PT tokens:", pt_tokens)
    for i, pid in enumerate(pt_pids[: args.steps + 1]):
        print(f"  step {i}: position_ids shape={tuple(pid.shape)}")

    print("\n=== Per-step divergence ===")
    for i in range(min(ov_logits.shape[1], pt_logits.shape[1])):
        ov_l = ov_logits[:, i, :]
        pt_l = pt_logits[:, i, :]
        diff = (ov_l - pt_l).abs()
        ov_top1 = int(ov_l.argmax(-1).item())
        pt_top1 = int(pt_l.argmax(-1).item())
        match = "✓" if ov_top1 == pt_top1 else "✗"
        print(
            f" step {i}: {match} top1 ov={ov_top1} pt={pt_top1} "
            f"maxdiff={diff.max().item():.3f} meandiff={diff.mean().item():.4f}"
        )


if __name__ == "__main__":
    main()
