"""
ERNIE-Image OpenVINO Export & Inference Test Script

Tests:
1. Export model via optimum-cli (FP32 and INT4)
2. Run inference with OVErnieImagePipeline
3. Compare with original PyTorch pipeline output
4. Measure latency

Usage:
    python test_ernie_image.py --model-path /path/to/ERNIE-Image
    python test_ernie_image.py --model-path /path/to/ERNIE-Image --skip-export  # reuse exported models
    python test_ernie_image.py --model-path /path/to/ERNIE-Image --skip-pytorch  # skip pytorch comparison
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Register ministral3 config early (needed for ERNIE-Image text encoder)
from transformers import MistralConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

if "ministral3" not in getattr(CONFIG_MAPPING, "_extra_content", {}):
    CONFIG_MAPPING.register("ministral3", MistralConfig)


def export_model(model_path: str, output_dir: str, weight_format: str = "fp32"):
    """Export ERNIE-Image model via optimum-cli."""
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli",
        "export", "openvino",
        "--model", model_path,
        output_dir,
        "--task", "text-to-image",
        "--weight-format", weight_format,
    ]
    print(f"\n{'='*60}")
    print(f"Exporting model ({weight_format}): {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f"Export failed with return code {result.returncode}")
    print(f"Export ({weight_format}) completed successfully!")


def run_pytorch_inference(model_path: str, prompt: str, num_steps: int, height: int, width: int, seed: int):
    """Run inference with original PyTorch model."""
    from diffusers import ErnieImagePipeline, ErnieImageTransformer2DModel, AutoencoderKLFlux2
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoModel, PreTrainedTokenizerFast

    print(f"\n{'='*60}")
    print("Loading PyTorch model...")
    print(f"{'='*60}")

    text_encoder = AutoModel.from_pretrained(f"{model_path}/text_encoder", torch_dtype=torch.float32)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{model_path}/tokenizer")
    transformer = ErnieImageTransformer2DModel.from_pretrained(f"{model_path}/transformer", torch_dtype=torch.float32)
    vae = AutoencoderKLFlux2.from_pretrained(f"{model_path}/vae", torch_dtype=torch.float32)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{model_path}/scheduler")

    pipe = ErnieImagePipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        pe=None,
        pe_tokenizer=None,
    )
    pipe.to("cpu")

    generator = torch.Generator("cpu").manual_seed(seed)

    print(f"Generating image with PyTorch: prompt='{prompt}', steps={num_steps}, {width}x{height}")
    start = time.time()
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        generator=generator,
    )
    elapsed = time.time() - start
    image = result.images[0]
    print(f"PyTorch inference: {elapsed:.2f}s")

    del pipe, text_encoder, transformer, vae
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    import gc; gc.collect()

    return image, elapsed


def run_ov_inference(export_dir: str, prompt: str, num_steps: int, height: int, width: int, seed: int, label: str = "OV"):
    """Run inference with exported OpenVINO model."""
    from optimum.intel import OVErnieImagePipeline

    print(f"\n{'='*60}")
    print(f"Loading {label} model from {export_dir}...")
    print(f"{'='*60}")

    pipe = OVErnieImagePipeline.from_pretrained(export_dir, device="CPU")

    generator = torch.Generator("cpu").manual_seed(seed)

    print(f"Generating image with {label}: prompt='{prompt}', steps={num_steps}, {width}x{height}")
    start = time.time()
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        height=height,
        width=width,
        generator=generator,
    )
    elapsed = time.time() - start
    image = result.images[0]
    print(f"{label} inference: {elapsed:.2f}s")

    del pipe
    import gc; gc.collect()

    return image, elapsed


def compute_image_similarity(img1: Image.Image, img2: Image.Image) -> dict:
    """Compute similarity metrics between two images."""
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)

    if arr1.shape != arr2.shape:
        # Resize to match
        img2 = img2.resize(img1.size)
        arr2 = np.array(img2).astype(np.float32)

    mse = np.mean((arr1 - arr2) ** 2)
    psnr = 10 * np.log10(255.0**2 / max(mse, 1e-10))
    mae = np.mean(np.abs(arr1 - arr2))

    return {"MSE": mse, "PSNR_dB": psnr, "MAE": mae}


def main():
    parser = argparse.ArgumentParser(description="Test ERNIE-Image with OpenVINO")
    parser.add_argument("--model-path", type=str, default="/home/ethan/intel/ERNIE-image/ERNIE-Image",
                        help="Path to ERNIE-Image model")
    parser.add_argument("--output-dir", type=str, default="/tmp/ernie_image_ov_test",
                        help="Base output directory for exported models")
    parser.add_argument("--prompt", type=str, default="a cute cat sitting on a colorful cushion, studio lighting, high quality",
                        help="Text prompt for image generation")
    parser.add_argument("--num-steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--height", type=int, default=256, help="Image height")
    parser.add_argument("--width", type=int, default=256, help="Image width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-export", action="store_true", help="Skip export, reuse existing models")
    parser.add_argument("--skip-pytorch", action="store_true", help="Skip PyTorch comparison")
    parser.add_argument("--skip-int4", action="store_true", help="Skip INT4 quantized export")
    args = parser.parse_args()

    fp32_dir = os.path.join(args.output_dir, "FP32")
    int4_dir = os.path.join(args.output_dir, "INT4")

    results = {}

    # ──────────────── Step 1: Export ────────────────
    if not args.skip_export:
        # FP32 export
        if os.path.exists(fp32_dir):
            subprocess.run(["rm", "-rf", fp32_dir])
        export_model(args.model_path, fp32_dir, "fp32")

        # INT4 export
        if not args.skip_int4:
            if os.path.exists(int4_dir):
                subprocess.run(["rm", "-rf", int4_dir])
            export_model(args.model_path, int4_dir, "int4")

    # ──────────────── Step 2: PyTorch Inference ────────────────
    if not args.skip_pytorch:
        pt_image, pt_time = run_pytorch_inference(
            args.model_path, args.prompt, args.num_steps, args.height, args.width, args.seed,
        )
        pt_path = os.path.join(args.output_dir, "pytorch_output.png")
        pt_image.save(pt_path)
        print(f"PyTorch output saved: {pt_path}")
        results["pytorch"] = {"time": pt_time, "image": pt_image}

    # ──────────────── Step 3: OV FP32 Inference ────────────────
    if os.path.exists(fp32_dir):
        fp32_image, fp32_time = run_ov_inference(
            fp32_dir, args.prompt, args.num_steps, args.height, args.width, args.seed, "OV-FP32",
        )
        fp32_path = os.path.join(args.output_dir, "ov_fp32_output.png")
        fp32_image.save(fp32_path)
        print(f"OV-FP32 output saved: {fp32_path}")
        results["ov_fp32"] = {"time": fp32_time, "image": fp32_image}

    # ──────────────── Step 4: OV INT4 Inference ────────────────
    if not args.skip_int4 and os.path.exists(int4_dir):
        int4_image, int4_time = run_ov_inference(
            int4_dir, args.prompt, args.num_steps, args.height, args.width, args.seed, "OV-INT4",
        )
        int4_path = os.path.join(args.output_dir, "ov_int4_output.png")
        int4_image.save(int4_path)
        print(f"OV-INT4 output saved: {int4_path}")
        results["ov_int4"] = {"time": int4_time, "image": int4_image}

    # ──────────────── Step 5: Comparison ────────────────
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Resolution: {args.width}x{args.height}, Steps: {args.num_steps}, Seed: {args.seed}")
    print()

    # Latency table
    print(f"{'Model':<15} {'Time (s)':<12}")
    print("-" * 27)
    for name, data in results.items():
        print(f"{name:<15} {data['time']:<12.2f}")

    # Similarity comparison
    if "pytorch" in results:
        pt_img = results["pytorch"]["image"]
        print()
        print(f"{'Comparison':<25} {'MSE':<12} {'PSNR (dB)':<12} {'MAE':<12}")
        print("-" * 61)
        for name in ["ov_fp32", "ov_int4"]:
            if name in results:
                metrics = compute_image_similarity(pt_img, results[name]["image"])
                print(f"PyTorch vs {name:<14} {metrics['MSE']:<12.2f} {metrics['PSNR_dB']:<12.2f} {metrics['MAE']:<12.2f}")

        if "ov_fp32" in results and "ov_int4" in results:
            metrics = compute_image_similarity(results["ov_fp32"]["image"], results["ov_int4"]["image"])
            print(f"OV-FP32 vs OV-INT4       {metrics['MSE']:<12.2f} {metrics['PSNR_dB']:<12.2f} {metrics['MAE']:<12.2f}")

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
