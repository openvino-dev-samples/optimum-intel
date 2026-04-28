"""Convert PP-DocLayout-V3 (PaddlePaddle) to OpenVINO IR.

Steps:
1. Download the PP-DocLayout-V3 inference package from PaddleX if not present.
2. Convert the `.pdmodel` + `.pdiparams` to ONNX via paddle2onnx.
3. Convert the ONNX model to OpenVINO IR via openvino.convert_model.

Usage:
    python convert.py --output pp_doclayout_v3_ov
"""
import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import openvino as ov


DEFAULT_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/"
    "paddle3.0.0/PP-DocLayout_plus-L_infer.tar"
)


def download(url: str, dest: Path):
    if dest.exists():
        print(f"{dest} already exists, skipping download")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def extract_tar(tar_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dest_dir)
    # Find the extracted directory (the tar usually contains a single top-level dir)
    subdirs = [p for p in dest_dir.iterdir() if p.is_dir()]
    if subdirs:
        return subdirs[0]
    return dest_dir


def paddle_to_onnx(pd_dir: Path, onnx_path: Path):
    if onnx_path.exists():
        print(f"{onnx_path} already exists, skipping paddle2onnx")
        return
    print(f"Converting Paddle model in {pd_dir} to ONNX at {onnx_path}")
    cmd = [
        sys.executable,
        "-m",
        "paddle2onnx",
        "--model_dir",
        str(pd_dir),
        "--model_filename",
        "inference.pdmodel",
        "--params_filename",
        "inference.pdiparams",
        "--save_file",
        str(onnx_path),
        "--opset_version",
        "16",
        "--enable_onnx_checker",
        "True",
    ]
    subprocess.check_call(cmd)


def onnx_to_ir(onnx_path: Path, ov_dir: Path):
    ov_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting ONNX {onnx_path} -> OpenVINO IR in {ov_dir}")
    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, ov_dir / "pp_doclayout_v3.xml", compress_to_fp16=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="PP-DocLayout-V3 inference tarball URL")
    parser.add_argument("--work_dir", default="pp_doclayout_v3_src")
    parser.add_argument("--output", default="pp_doclayout_v3_ov")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    tar_path = work / Path(args.url).name
    download(args.url, tar_path)
    pd_dir = extract_tar(tar_path, work / "extracted")

    onnx_path = work / "model.onnx"
    paddle_to_onnx(pd_dir, onnx_path)

    onnx_to_ir(onnx_path, Path(args.output))
    print(f"Done. OV IR saved to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
