"""Compare OV-CPU / OV-GPU / HF-PyTorch-CPU outputs of GLM-OCR on a sample image.

Reports:
- per-logit maxdiff / meandiff / top-1 match rate
- decoded text side-by-side

Usage:
    python compare_cpu_gpu.py --ov_model glm_ocr_ov --pt_model D:/optimum-intel/GLM-OCR --image sample.png
"""
import argparse
import json
import sys
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


@torch.no_grad()
def run_pt_forward(model, inputs):
    out = model(**inputs, use_cache=False)
    return out.logits.float().cpu().numpy()


def run_ov_forward(model, inputs):
    out = model(**{k: v for k, v in inputs.items()}, use_cache=False)
    logits = out.logits
    if isinstance(logits, torch.Tensor):
        logits = logits.float().cpu().numpy()
    return np.asarray(logits)


def diff_report(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> dict:
    # match seq-length (some prefill vs incremental skew)
    seq = min(a.shape[1], b.shape[1])
    a = a[:, :seq].astype(np.float32)
    b = b[:, :seq].astype(np.float32)
    diff = np.abs(a - b)
    top1_a = a.argmax(-1)
    top1_b = b.argmax(-1)
    match = (top1_a == top1_b).mean()
    return {
        "pair": f"{name_a} vs {name_b}",
        "shape": list(a.shape),
        "maxdiff": float(diff.max()),
        "meandiff": float(diff.mean()),
        "top1_match": float(match),
    }


def decode_generation(name, model, processor, inputs, max_new_tokens, t0=None):
    start = time.time() if t0 is None else t0
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    dt = time.time() - start
    new_tokens = gen[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    print(f"\n[{name}] generated in {dt:.1f}s")
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ov_model", required=True)
    parser.add_argument("--pt_model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Text Recognition:")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--skip_gpu", action="store_true")
    parser.add_argument("--skip_pt", action="store_true")
    parser.add_argument("--output", default="compare_results.txt")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.pt_model, trust_remote_code=True)
    inputs = build_inputs(processor, args.image, args.prompt)
    print(f"input_ids.shape = {inputs['input_ids'].shape}")
    if "pixel_values" in inputs:
        print(f"pixel_values.shape = {inputs['pixel_values'].shape}")
    if "image_grid_thw" in inputs:
        print(f"image_grid_thw = {inputs['image_grid_thw'].tolist()}")

    results = {}

    # --- OV CPU ---
    print("\n=== OV CPU ===")
    ov_cpu = OVModelForVisualCausalLM.from_pretrained(args.ov_model, device="CPU")
    text_ov_cpu = decode_generation("OV-CPU", ov_cpu, processor, inputs, args.max_new_tokens)
    results["ov_cpu_text"] = text_ov_cpu

    # --- OV GPU ---
    text_ov_gpu = None
    if not args.skip_gpu:
        print("\n=== OV GPU ===")
        try:
            ov_gpu = OVModelForVisualCausalLM.from_pretrained(args.ov_model, device="GPU")
            text_ov_gpu = decode_generation("OV-GPU", ov_gpu, processor, inputs, args.max_new_tokens)
            results["ov_gpu_text"] = text_ov_gpu
        except Exception as e:
            print(f"GPU load/run failed: {e}")
            results["ov_gpu_error"] = str(e)

    # --- PyTorch CPU ---
    text_pt = None
    if not args.skip_pt:
        print("\n=== PyTorch CPU ===")
        pt = AutoModelForImageTextToText.from_pretrained(
            args.pt_model, torch_dtype=torch.float32, trust_remote_code=True
        ).eval()
        text_pt = decode_generation("PyTorch-CPU", pt, processor, inputs, args.max_new_tokens)
        results["pt_cpu_text"] = text_pt

        # logits-level comparison (single forward)
        print("\n=== Logits comparison ===")
        pt_logits = run_pt_forward(pt, inputs)

        ov_cpu_logits = run_ov_forward(ov_cpu, inputs)
        results["diff_ov_cpu_vs_pt"] = diff_report("OV-CPU", ov_cpu_logits, "PT-CPU", pt_logits)
        print(json.dumps(results["diff_ov_cpu_vs_pt"], indent=2))

        if not args.skip_gpu and "ov_gpu_error" not in results:
            ov_gpu_logits = run_ov_forward(ov_gpu, inputs)
            results["diff_ov_gpu_vs_pt"] = diff_report("OV-GPU", ov_gpu_logits, "PT-CPU", pt_logits)
            print(json.dumps(results["diff_ov_gpu_vs_pt"], indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
