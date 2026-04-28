# GLM-OCR on OpenVINO

End-to-end scripts to export and validate `zai-org/GLM-OCR` with optimum-intel,
plus an optional PP-DocLayout-V3 + GLM-OCR document parsing pipeline.

## Files

- `export.py` — exports the local GLM-OCR checkpoint to OpenVINO IR using
  `optimum.exporters.openvino.main_export`.
- `compare_cpu_gpu.py` — loads the IR on CPU and GPU, loads the original HF
  PyTorch model on CPU, runs the same image+prompt through all three, and
  reports logits diff and decoded-text diff.
- `pp_doclayout_v3/convert.py` — converts PP-DocLayout-V3 (PaddlePaddle) to
  OpenVINO IR via paddle2onnx + `ov.convert_model`.
- `pp_doclayout_v3/pipeline.py` — full document parsing: PP-DocLayout-V3
  detects regions, GLM-OCR recognises each region, result is composed into
  Markdown.

## Quick start

```bash
# 1. Export GLM-OCR
python export.py --model_path D:/optimum-intel/GLM-OCR --output glm_ocr_ov

# 2. Compare CPU/GPU against PyTorch reference
python compare_cpu_gpu.py \
    --ov_model glm_ocr_ov \
    --pt_model D:/optimum-intel/GLM-OCR \
    --image sample.png

# 3. (Optional) PP-DocLayout-V3 pipeline
python pp_doclayout_v3/convert.py --output pp_doclayout_v3_ov
python pp_doclayout_v3/pipeline.py \
    --layout_model pp_doclayout_v3_ov \
    --ocr_model glm_ocr_ov \
    --image document.pdf \
    --device GPU
```
