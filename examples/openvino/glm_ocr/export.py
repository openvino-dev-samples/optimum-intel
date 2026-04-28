"""Export GLM-OCR to OpenVINO IR.

Usage:
    python export.py --model_path D:/optimum-intel/GLM-OCR --output glm_ocr_ov
"""
import argparse
from pathlib import Path

from optimum.exporters.openvino import main_export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Local path or HF id of GLM-OCR checkpoint")
    parser.add_argument("--output", default="glm_ocr_ov", help="Output directory for OpenVINO IR")
    parser.add_argument("--weight_format", default="fp16", choices=["fp32", "fp16", "int8", "int4"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    main_export(
        model_name_or_path=args.model_path,
        output=out,
        task="image-text-to-text",
        weight_format=args.weight_format,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Exported to {out.resolve()}")


if __name__ == "__main__":
    main()
