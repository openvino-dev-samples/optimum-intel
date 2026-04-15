# ERNIE-Image with OpenVINO — 使用指南

## 概述

本指南介绍如何从零开始完成 ERNIE-Image（Turbo）的环境搭建、模型转换（FP16 / INT4）、
OpenVINO 推理，以及与 PyTorch 原始模型的精度对比验证。

> **2026-04-15 更新**：新增 PE（Prompt Enhancer）模型的导出、量化及推理支持。
> 使用 `use_pe=True` 可自动增强 prompt，与原始 `ErnieImagePipeline` 用法完全一致。

> **2026-04-14 更新**：ERNIE-Image 已合并到官方 diffusers（[PR #13432](https://github.com/huggingface/diffusers/pull/13432)），
> 不再需要使用 HsiaWinter fork。本指南已更新为使用 **官方 diffusers main 分支**。

---

## 1. 环境准备

### 1.1 系统要求

- Python 3.10+
- Linux / Windows（推荐 Linux）
- 磁盘空间：原始模型 ~30 GB，转换输出约 7–22 GB（视格式而定）
- 内存：建议 64 GB+（FP16 转换时峰值约 48 GB）

### 1.2 克隆仓库并创建虚拟环境

```bash
# 克隆含 ERNIE-Image 支持的 optimum-intel fork
git clone https://github.com/openvino-dev-samples/optimum-intel.git
cd optimum-intel
git checkout ernie-image
# 锁定到已验证的 commit（可选，确保可复现）
git checkout 6a55b811e51bf0dd2e09dc9f4f826c99704cf457

# 创建 Python 3.10 虚拟环境
python3.10 -m venv py_env
source py_env/bin/activate   # Windows: py_env\Scripts\activate
```

### 1.3 安装依赖

```bash
# 1. 安装 optimum（锁定到已验证的 commit）
pip install "git+https://github.com/huggingface/optimum.git@ec676fd4e0b1440e91549e7a1aa82e0de85e79b5"

# 2. 安装 diffusers（官方 main 分支，已包含 ERNIE-Image 支持）
pip install "git+https://github.com/huggingface/diffusers.git@6a339ce637db184c2e1a10ec90ac0e292beb76ac"

# 3. 安装 transformers
pip install transformers==4.57.6

# 4. 安装 OpenVINO 及配套工具
pip install openvino==2026.1.0
pip install openvino-tokenizers==2026.1.0.0

# 5. 安装 NNCF（量化支持，INT4/INT8 需要）
pip install nncf==3.0.0

# 6. 安装 PyTorch（CPU 版即可，如需 GPU 去掉 --index-url）
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# 7. 以可编辑模式安装 optimum-intel（ernie-image 分支）
pip install -e ".[openvino,nncf,diffusers]"
```

> **已在干净环境中验证通过的版本（2026-04-14）：**
> | 包 | 版本 | 来源 | commit |
> |---|------|------|--------|
> | `optimum-intel` | 1.27.0.dev0 | `openvino-dev-samples/optimum-intel` ernie-image 分支 | `d0e7fc2a` |
> | `optimum` | 2.1.0.dev0 | `huggingface/optimum` main | `ec676fd4` |
> | `diffusers` | 0.38.0.dev0 | `huggingface/diffusers` **main 分支**（含 PR #13432） | `6a339ce6` |
> | `transformers` | 4.57.6 | PyPI | — |
> | `openvino` | 2026.0.0 | PyPI | — |
> | `openvino-tokenizers` | 2026.0.0.0 | PyPI | — |
> | `nncf` | 3.0.0 | PyPI | — |
> | `torch` | 2.10.0 | PyTorch CPU wheel | — |

---

## 2. 获取 ERNIE-Image-Turbo 原始模型

下载 ERNIE-Image-Turbo PyTorch 权重，解压后目录结构如下：

```
ERNIE-Image-Turbo/
├── model_index.json
├── scheduler/
├── pe/                 # Prompt Encoder (Ministral3ForCausalLM, 可选)
├── pe_tokenizer/       # PE Tokenizer (可选)
├── text_encoder/       # Mistral3 语言模型（~7.2 GB）
├── tokenizer/
├── transformer/        # ErnieImageTransformer2DModel（~15 GB, 2-shard）
└── vae/                # AutoencoderKLFlux2（~161 MB）
```

> **注意**：`pe/` 和 `pe_tokenizer/` 是 Prompt Enhancer 组件（Ministral3ForCausalLM），
> 导出时会自动检测并转换为 OpenVINO IR。如源模型包含这些组件，推理时可通过 `use_pe=True` 启用。

---

## 3. 模型转换（导出为 OpenVINO IR）

### 3.1 导出 FP16（推荐基准格式）

```bash
optimum-cli export openvino \
  --model /path/to/ERNIE-Image-Turbo \
  --task text-to-image \
  --weight-format fp16 \
  ./ernie_image_turbo_fp16
```

### 3.2 导出 INT4（体积最小，速度最快）

```bash
optimum-cli export openvino \
  --model /path/to/ERNIE-Image-Turbo \
  --task text-to-image \
  --weight-format int4 \
  --ratio 0.8 \
  ./ernie_image_turbo_int4
```

> **INT4 各子模型量化策略：**
> | 子模型 | 量化方式 | FP16 大小 | INT4 大小 | 说明 |
> |--------|----------|-----------|-----------|------|
> | `text_encoder` | INT4 asym (ratio=1.0, 87% INT4 / 13% INT8) | 6.4 GB | 1.9 GB | 语言模型，INT4 精度损失小 |
> | `transformer` | INT4 asym (ratio=0.8, 80% INT4 / 20% INT8) | 15 GB | 4.7 GB | 20% 权重保留 INT8 以确保精度 |
> | `vae_encoder` | INT8 sym (100%) | 66 MB | 33 MB | VAE 对量化敏感，保留 INT8 |
> | `vae_decoder` | INT8 sym (100%) | 95 MB | 48 MB | VAE 对量化敏感，保留 INT8 |
> | `pe` | INT4 asym (ratio=1.0, 87% INT4 / 13% INT8) | 6.4 GB | 1.9 GB | Prompt Enhancer，自动检测并量化 |

转换完成后，输出目录结构：

```
ernie_image_turbo_int4/
├── model_index.json
├── openvino_config.json
├── vae_bn_stats.npz        # VAE BatchNorm 统计量（推理时自动加载）
├── scheduler/
├── tokenizer/
├── pe/                      # Prompt Enhancer (OVModelForCausalLM)
│   ├── openvino_model.xml
│   ├── openvino_model.bin   # ~1.9 GB (INT4)
│   └── config.json
├── pe_tokenizer/            # PE Tokenizer
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── chat_template.jinja
├── text_encoder/
│   ├── openvino_model.xml
│   └── openvino_model.bin   # ~1.9 GB (INT4)
├── transformer/
│   ├── openvino_model.xml
│   └── openvino_model.bin   # ~4.7 GB (INT4)
├── vae_encoder/
│   ├── openvino_model.xml
│   └── openvino_model.bin   # ~33 MB (INT8)
└── vae_decoder/
    ├── openvino_model.xml
    └── openvino_model.bin   # ~48 MB (INT8)
```


## 4. OpenVINO 推理

### 4.1 基础推理示例

```python
from optimum.intel import OVErnieImagePipeline
import torch

# 加载导出好的 OpenVINO 模型
pipe = OVErnieImagePipeline.from_pretrained(
    "./ernie_image_turbo_int4",   # 可替换为 fp16 目录
    device="CPU",                  # 也可以用 "GPU"（需要 Intel GPU）
)

generator = torch.Generator("cpu").manual_seed(42)

result = pipe(
    prompt="a cute cat sitting on a colorful cushion, studio lighting, high quality",
    num_inference_steps=20,
    height=256,
    width=256,
    generator=generator,
)

result.images[0].save("output.png")
```

### 4.2 使用 Prompt Enhancer（PE）

如果导出的模型包含 PE 组件，可通过 `use_pe=True` 自动增强 prompt：

```python
from optimum.intel import OVErnieImagePipeline
import torch

pipe = OVErnieImagePipeline.from_pretrained(
    "./ernie_image_turbo_int4",
    device="CPU",
)

generator = torch.Generator("cpu").manual_seed(42)

# use_pe=True 会自动使用 PE 模型将简短 prompt 扩展为详细的视觉描述
result = pipe(
    prompt="a cute cat",
    height=1024,
    width=1024,
    num_inference_steps=8,
    guidance_scale=1.0,
    generator=generator,
    use_pe=True,  # 启用 Prompt Enhancer
)

result.images[0].save("output_with_pe.png")

# 查看增强后的 prompt
if result.revised_prompts:
    print(f"Enhanced prompt: {result.revised_prompts[0]}")
```

> **说明**：
> - `use_pe=True` 时，PE 模型会将简短的 prompt 自动扩展为丰富的视觉描述。
> - 如果导出的模型不包含 PE 组件，`use_pe=True` 会被自动忽略，不影响推理。
> - PE 模型使用 `OVModelForCausalLM`，支持 OpenVINO 加速的自回归生成。
> - 与原始 `ErnieImagePipeline` 的 `use_pe` 参数用法完全一致。

### 4.3 常用参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | — | 文本描述 |
| `num_inference_steps` | 28 | 去噪步数，越多越慢越精细，推荐 20–30 |
| `height` / `width` | 512 | 输出尺寸，建议为 64 的整数倍 |
| `guidance_scale` | 3.5 | CFG 强度，建议不超过 5.0 |
| `generator` | None | 随机种子，用于复现结果 |
| `use_pe` | True | 是否使用 Prompt Enhancer 增强 prompt |

---

## 5. 精度验证与性能对比

### 5.1 使用内置测试脚本

```bash
# 完整测试：导出 FP16 + INT4，与 PyTorch 做精度对比
python test_ernie_image.py \
  --model-path /path/to/ERNIE-Image-Turbo \
  --num-steps 20 \
  --height 256 \
  --width 256 \
  --seed 42

# 跳过导出，复用已有模型
python test_ernie_image.py \
  --skip-export \
  --num-steps 20 \
  --height 256 \
  --width 256

# 仅测试 OV 推理，跳过 PyTorch 对比（节省时间）
python test_ernie_image.py \
  --skip-export \
  --skip-pytorch \
  --num-steps 20
```

### 5.2 精度指标参考（ERNIE-Image-Turbo，256×256，20 步，seed=42，2026-04-14 验证）

| 对比 | MSE | PSNR | MAE | 结论 |
|------|-----|------|-----|------|
| PyTorch FP32 vs OV-FP16 | 1113.95 | 17.66 dB | 22.49 | 精度损失较小 |
| PyTorch FP32 vs OV-INT4 | 4063.62 | 12.04 dB | 52.35 | 可接受，有量化噪声 |
| OV-FP16 vs OV-INT4 | 5644.76 | 10.61 dB | 61.13 | INT4 与 FP16 差异 |

> **说明**：ERNIE-Image-Turbo 由于模型结构更复杂（Mistral3 文本编码器 + 36 层 Transformer），
> FP16 导出与 PyTorch FP32 推理之间存在一定的数值差异（PSNR 17.66 dB），主要来自
> FP32→FP16 权重精度截断。生成的图像在视觉上仍然连贯且高质量。

### 5.3 性能对比（ERNIE-Image-Turbo，256×256，20 步，Intel CPU，2026-04-14 验证）

| 格式 | 推理耗时 | 相对加速 |
|------|---------|---------|
| PyTorch FP32 | ~133 s | 1× |
| OV FP16 | ~127 s | ~1.0× |
| OV INT4 | ~49 s | ~2.7× |

> INT4 量化在保持可接受图像质量的前提下，推理速度提升约 2.7 倍。

### 5.4 手动精度验证脚本

```python
import numpy as np
from PIL import Image

def psnr(img1_path, img2_path):
    a = np.array(Image.open(img1_path)).astype(np.float32)
    b = np.array(Image.open(img2_path)).astype(np.float32)
    mse = np.mean((a - b) ** 2)
    return 10 * np.log10(255.0 ** 2 / max(mse, 1e-10))

print(f"FP16 PSNR: {psnr('pytorch_output.png', 'ov_fp16_output.png'):.2f} dB")
print(f"INT4 PSNR: {psnr('pytorch_output.png', 'ov_int4_output.png'):.2f} dB")
```

---

## 6. 常见问题

### Q：导出时报 `TokenizersBackend` 错误？

A：ERNIE-Image 的 tokenizer 使用了 HuggingFace tokenizers v5 才支持的后端。
`OVErnieImagePipeline` 已内置处理逻辑，会自动回退到 `PreTrainedTokenizerFast`，无需手动处理。

### Q：`ImportError: cannot import name 'ErnieImagePipeline' from 'diffusers'`？

A：diffusers 版本过旧。ERNIE-Image 支持已于 2026-04-11 合并到官方 diffusers main 分支
（[PR #13432](https://github.com/huggingface/diffusers/pull/13432)）。请安装最新版：
```bash
pip install "git+https://github.com/huggingface/diffusers.git@6a339ce637db184c2e1a10ec90ac0e292beb76ac"
```

### Q：出现 `ministral3 not found in config mapping`？

A：`Mistral3TextEncoder` 的子配置 `ministral3` 在部分 transformers 版本中未注册。
`OVErnieImagePipeline` 会在加载时自动注册，无需手动操作。

### Q：导出时报 `The library name could not be automatically inferred`？

A：确保模型路径指向包含 `model_index.json` 的目录。如果目录内没有此文件，
optimum-cli 无法判断是哪种 pipeline。可手动添加 `--library diffusers`。

### Q：INT4 图像质量明显差于 FP16？

A：请确认使用的是包含以下修复的 `ernie-image` 分支（commit `475d437b` 及以后）：
- VAE encoder/decoder 保留 INT8（而非 INT4）
- transformer ratio 上限为 0.8

### Q：`pe/` 和 `pe_tokenizer/` 是什么？

A：这些是 Prompt Enhancer 组件，用于将简短 prompt 自动扩展为丰富的视觉描述。
导出时会自动检测源模型中的 PE 组件并转换为 OpenVINO IR（`OVModelForCausalLM`）。
推理时通过 `use_pe=True` 启用。如果模型不包含 PE 或设置 `use_pe=False`，推理正常进行。

### Q：磁盘空间不足导致导出失败（XML 文件为 0 字节）？

A：转换期间峰值磁盘占用约 100+ GB（FP32 临时文件）。确认磁盘有足够空间后重试。

```bash
df -h .   # 查看当前目录所在分区剩余空间
```
