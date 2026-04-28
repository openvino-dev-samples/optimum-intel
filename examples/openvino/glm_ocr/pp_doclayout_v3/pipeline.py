"""End-to-end document parsing: PP-DocLayout-V3 (layout) + GLM-OCR (recognition).

Both models run on OpenVINO. Layout detection classifies each box into
text / table / formula / image / ..., and GLM-OCR is queried with the matching
prompt for each text/table/formula region. Other region types are kept as
placeholders.

Usage:
    python pipeline.py --layout_model pp_doclayout_v3_ov \\
        --ocr_model glm_ocr_ov --image document.png --device GPU
"""
import argparse
import json
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

from transformers import AutoProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM


# PP-DocLayout-V3 class indices (from PaddleX config). These are the labels the
# model was trained with; adjust if using a custom checkpoint.
LAYOUT_CLASSES = [
    "paragraph_title",
    "image",
    "text",
    "number",
    "abstract",
    "content",
    "figure_title",
    "formula",
    "table",
    "table_title",
    "reference",
    "doc_title",
    "footnote",
    "header",
    "algorithm",
    "footer",
    "seal",
    "chart_title",
    "chart",
    "formula_number",
    "header_image",
    "footer_image",
    "aside_text",
]

PROMPT_BY_CLASS = {
    "text": "Text Recognition:",
    "paragraph_title": "Text Recognition:",
    "doc_title": "Text Recognition:",
    "figure_title": "Text Recognition:",
    "table_title": "Text Recognition:",
    "chart_title": "Text Recognition:",
    "abstract": "Text Recognition:",
    "content": "Text Recognition:",
    "reference": "Text Recognition:",
    "footnote": "Text Recognition:",
    "header": "Text Recognition:",
    "footer": "Text Recognition:",
    "aside_text": "Text Recognition:",
    "number": "Text Recognition:",
    "algorithm": "Text Recognition:",
    "formula": "Formula Recognition:",
    "formula_number": "Formula Recognition:",
    "table": "Table Recognition:",
}


class LayoutDetector:
    """Thin OV wrapper for PP-DocLayout-V3.

    Assumes the exported model expects a 3xHxW float tensor normalized with
    ImageNet stats. The network outputs bounding boxes + class scores; we
    post-process the common PP-DocLayout output head (x0, y0, x1, y1, class, score).
    """

    def __init__(self, model_dir: str, device: str, input_size: int = 800):
        core = ov.Core()
        xml = next(Path(model_dir).glob("*.xml"))
        self.model = core.compile_model(str(xml), device)
        self.input_size = input_size
        self.input_name = self.model.inputs[0].get_any_name()

    def preprocess(self, image: Image.Image):
        img = image.convert("RGB")
        w, h = img.size
        scale = self.input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = img.resize((new_w, new_h))
        canvas = Image.new("RGB", (self.input_size, self.input_size), (127, 127, 127))
        canvas.paste(resized, (0, 0))
        arr = np.asarray(canvas).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)[None]
        return arr, scale, (w, h)

    def __call__(self, image: Image.Image, score_thr: float = 0.3):
        inp, scale, (w, h) = self.preprocess(image)
        outputs = self.model({self.input_name: inp})
        # The exact tensor layout depends on the paddle2onnx conversion; we
        # expect one output of shape [N, 6]: [x0, y0, x1, y1, class, score].
        # If the model head differs, adjust accordingly.
        main_output = list(outputs.values())[0]
        if main_output.ndim == 3:
            main_output = main_output[0]
        detections = []
        for row in main_output:
            if row.shape[0] < 6:
                continue
            x0, y0, x1, y1, cls, score = row[:6]
            if score < score_thr:
                continue
            x0, x1 = x0 / scale, x1 / scale
            y0, y1 = y0 / scale, y1 / scale
            x0, y0 = max(0, min(w, x0)), max(0, min(h, y0))
            x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
            cls_idx = int(cls)
            if cls_idx < 0 or cls_idx >= len(LAYOUT_CLASSES):
                continue
            detections.append({
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "class": LAYOUT_CLASSES[cls_idx],
                "score": float(score),
            })
        # Sort top-to-bottom, left-to-right (reading order heuristic)
        detections.sort(key=lambda d: (round(d["bbox"][1] / 20), d["bbox"][0]))
        return detections


def recognize_region(ov_ocr, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    gen = ov_ocr.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = gen[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_model", required=True)
    parser.add_argument("--ocr_model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--score_thr", type=float, default=0.3)
    parser.add_argument("--output", default="pipeline_output.md")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    detector = LayoutDetector(args.layout_model, args.device)
    detections = detector(image, score_thr=args.score_thr)
    print(f"Detected {len(detections)} regions")

    processor = AutoProcessor.from_pretrained(args.ocr_model, trust_remote_code=True)
    ocr = OVModelForVisualCausalLM.from_pretrained(args.ocr_model, device=args.device)

    md_parts = []
    for i, det in enumerate(detections):
        cls = det["class"]
        prompt = PROMPT_BY_CLASS.get(cls)
        x0, y0, x1, y1 = det["bbox"]
        crop = image.crop((x0, y0, x1, y1))
        if prompt is None:
            md_parts.append(f"![{cls} region {i}](region_{i}.png)")
            crop.save(f"region_{i}.png")
            continue
        text = recognize_region(ocr, processor, crop, prompt, args.max_new_tokens)
        if cls in {"doc_title", "paragraph_title"}:
            md_parts.append(f"# {text}" if cls == "doc_title" else f"## {text}")
        elif cls == "formula":
            md_parts.append(f"$$\n{text}\n$$")
        elif cls == "table":
            md_parts.append(text)
        else:
            md_parts.append(text)
        print(f"[{i}] ({cls}) -> {text[:80]}")

    output = "\n\n".join(md_parts)
    Path(args.output).write_text(output, encoding="utf-8")
    print(f"\nMarkdown written to {args.output}")


if __name__ == "__main__":
    main()
