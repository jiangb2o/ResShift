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

待实现。

## Step 3: 端到端验证与使用说明

待实现。
