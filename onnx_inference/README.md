# ResShift ONNX Full Inference

本目录目标：基于已导出的 `weights/resshift_model.onnx`（UNet 单步），补齐完整 ONNX 推理链路：

1. Encoder ONNX：将上采样后的 LQ 图像编码到 latent。
2. UNet ONNX：在扩散时间步中迭代预测。
3. Decoder ONNX：将最终 latent 解码为 SR 图像。

## 设计思考

`export_model/export2onnx.py` 当前只导出了单步 UNet，不包含 autoencoder。完整流程在原始 PyTorch 推理中是：

1. `encode_first_stage(y)` 得到 `z_y`。
2. `p_sample_loop` 在 `t=T-1...0` 上迭代更新 `z_t`。
3. `decode_first_stage(z_0)` 得到 SR 图像。

因此这里采用“3 个 ONNX 模型 + Python 采样调度”的方式，保持与原公式一致，并且便于替换后端（ONNX Runtime 或 ReferenceEvaluator）。

## Step 1: 导出 Encoder / Decoder ONNX

### 实现

新增脚本：
- `onnx_inference/export_autoencoder_onnx.py`

实现点：
- 从完整配置（默认 `configs/realsr_swinunet_realesrgan256.yaml`）加载 autoencoder。
- 导出两个图：
  - `EncoderExportWrapper`: `image -> latent`
  - `DecoderExportWrapper`: `latent -> image`
- CPU 导出场景下强制禁用 xformers memory-efficient attention，避免导出阶段报错。
- 输出动态轴（batch/height/width）以支持可变分辨率。

### 验证

执行命令：

```bash
/home/ubuntu/miniconda3/envs/ResShift/bin/python onnx_inference/export_autoencoder_onnx.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --encoder_output onnx_inference/models/autoencoder_encoder.onnx \
  --decoder_output onnx_inference/models/autoencoder_decoder.onnx \
  --device cpu
```

结果：
- 导出成功，生成
  - `onnx_inference/models/autoencoder_encoder.onnx`（约 86 MB）
  - `onnx_inference/models/autoencoder_decoder.onnx`（约 126 MB）
- 使用 `onnx.checker.check_model` 校验通过。

---

## Step 2: 实现 ONNX 端扩散迭代脚本

### 实现

新增脚本：
- `onnx_inference/run_onnx_pipeline.py`

实现点：
- 完整流程：
  1. 读取 LQ 图像，归一化到 `[-1,1]`。
  2. 进行与原仓库一致的 padding 处理（默认按 `lq_size` 对齐）。
  3. `bicubic` 上采样后送入 Encoder ONNX 得到 `z_y`。
  4. 按 diffusion 参数初始化 `z_T`。
  5. 迭代 `t=T-1...0`：
     - UNet ONNX 输入：`x`、`lq`、`timesteps`
     - 按原公式计算 `pred_xstart`、`posterior mean`、`sigma` 并更新 `z_t`
  6. `z_0` 输入 Decoder ONNX 得到 SR 图。
  7. 反归一化并写出图像。
- 支持参数：
  - `--max_steps`：快速验证只跑前 N 个反向步。
  - `--dry_run`：加载模型并校验接口，不执行推理。
  - `--allow_reference`：允许无 `onnxruntime` 时使用 `onnx.reference`（极慢）。
- 兼容性处理：
  - 小图大补边时 `reflect` 不合法，自动退化为 `replicate` padding。
  - 明确添加 runtime guard：无 `onnxruntime` 且非 dry-run 时直接报错。

### 验证

1) 干跑接口验证（成功）

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/ubuntu/miniconda3/envs/ResShift/bin/python onnx_inference/run_onnx_pipeline.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --unet_onnx weights/resshift_model.onnx \
  --encoder_onnx onnx_inference/models/autoencoder_encoder.onnx \
  --decoder_onnx onnx_inference/models/autoencoder_decoder.onnx \
  --input onnx_inference/outputs/test_lq.png \
  --output onnx_inference/outputs/test_sr_step1.png \
  --dry_run
```

输出要点：
- 识别到模型接口：
  - UNet 输入 `['x', 'lq', 'timesteps']`，输出 `output`
  - Encoder 输入 `['image']`，输出 `latent`
  - Decoder 输入 `['latent']`，输出 `image`
- 形状链路正确：`LQ (1,3,64,64) -> upsampled (1,3,256,256)`

2) 依赖保护验证（成功）

在当前环境缺少 `onnxruntime` 时执行非 dry-run：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/ubuntu/miniconda3/envs/ResShift/bin/python onnx_inference/run_onnx_pipeline.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --unet_onnx weights/resshift_model.onnx \
  --encoder_onnx onnx_inference/models/autoencoder_encoder.onnx \
  --decoder_onnx onnx_inference/models/autoencoder_decoder.onnx \
  --input onnx_inference/outputs/test_lq.png \
  --output onnx_inference/outputs/test_sr_step1.png
```

输出要点：
- 抛出预期错误：`onnxruntime is not installed...`。

说明：
- `onnx.reference` 在当前机器上执行大图 UNet/Decoder 极慢，不具备工程实用性；
- 完整端到端推理应安装 `onnxruntime`（或 `onnxruntime-gpu`）后执行。

## Step 3: 端到端验证与使用说明

### 实现

新增脚本：
- `onnx_inference/validate_onnx_pipeline.py`

作用：
- 校验 3 个 ONNX 模型结构（`onnx.checker`）。
- 读取并输出三者输入/输出名称。
- 执行一次 Encoder ONNX 的真实前向（可在无 `onnxruntime` 时用 `onnx.reference`）。
- 生成结构化报告：`onnx_inference/outputs/validation_report.json`。

### 验证

执行命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/ubuntu/miniconda3/envs/ResShift/bin/python onnx_inference/validate_onnx_pipeline.py
```

结果摘要：
- `onnxruntime_available=false`（当前环境未安装）
- 模型接口检查通过：
  - UNet: `x, lq, timesteps -> output`
  - Encoder: `image -> latent`
  - Decoder: `latent -> image`
- Encoder ONNX 前向执行成功：
  - backend: `onnx-reference`
  - output shape: `[1, 3, 64, 64]`
- 报告文件已生成：`onnx_inference/outputs/validation_report.json`

### 完整推理使用方式

1) 先确保存在三个 ONNX：
- `weights/resshift_model.onnx`
- `onnx_inference/models/autoencoder_encoder.onnx`
- `onnx_inference/models/autoencoder_decoder.onnx`

2) 安装 `onnxruntime`（或 `onnxruntime-gpu`）后，执行：

```bash
/home/ubuntu/miniconda3/envs/ResShift/bin/python onnx_inference/run_onnx_pipeline.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --unet_onnx weights/resshift_model.onnx \
  --encoder_onnx onnx_inference/models/autoencoder_encoder.onnx \
  --decoder_onnx onnx_inference/models/autoencoder_decoder.onnx \
  --input <LQ_IMAGE_PATH> \
  --output <SR_OUTPUT_PATH>
```

3) 当前机器无 `onnxruntime` 时，可先做接口验证：

```bash
/home/ubuntu/miniconda3/envs/ResShift/bin/python onnx_inference/run_onnx_pipeline.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --unet_onnx weights/resshift_model.onnx \
  --encoder_onnx onnx_inference/models/autoencoder_encoder.onnx \
  --decoder_onnx onnx_inference/models/autoencoder_decoder.onnx \
  --input onnx_inference/outputs/test_lq.png \
  --output onnx_inference/outputs/test_sr.png \
  --dry_run
```
