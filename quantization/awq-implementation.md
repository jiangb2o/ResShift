# AWQ Quantization Flow

## 任务目标

实现一套完整的 AWQ 量化流程，要求：
- 不影响 ResShift 现有训练与推理入口的默认行为。
- 量化过程可选择性运行，不强制介入训练/推理。
- 所有新增文件都放在 `quantization/` 目录下。
- 支持：校准样本生成、AWQ 量化、量化权重保存、量化模型加载与前向验证。

## 实现思路

- 不改动原始 `models/`、`sampler.py`、`inference_resshift.py` 的默认执行路径。
- 在 `quantization/` 下实现独立 AWQ 工具链：
  - 使用真实图像生成 `UNetModelSwin(x, timesteps, lq)` 的校准输入。
  - 对 `nn.Linear` 层执行 AWQ 风格的 activation-aware weight-only 量化。
  - 生成独立的量化 checkpoint，不覆盖原始 checkpoint。
  - 提供独立加载器，在需要时将原模型替换为量化线性层。
- 量化默认只处理 `UNetModelSwin` 中的 `nn.Linear`，避免破坏卷积和 autoencoder 路径。
- `use_linfusion`、训练、普通推理都保持原样；只有显式运行 `quantization/quantize_awq.py` 和 `quantization/verify_awq.py` 时，AWQ 流程才会介入。

## 实施步骤

1. [x] 设计并实现 AWQ 核心模块与量化层。
2. [x] 实现校准数据生成和量化 CLI。
3. [x] 实现量化模型加载与真实验证，并记录结果。

## 步骤执行记录

### 步骤 1
- 变更内容：
  - 新增 `quantization/awq.py`，实现：
    - `AWQConfig`
    - `AWQLinear` 量化线性层
    - 分组对称权重量化 / 反量化
    - 激活采集器 `ActivationCollector`
    - AWQ 比例搜索 `_search_awq_scale`
    - `collect_awq_activations()`
    - `quantize_model_awq()`
    - `save_awq_checkpoint()`
    - `load_awq_quantized_model()`
  - 量化逻辑默认只匹配 `qkv`、`proj`、`reduction` 等 `nn.Linear` 层，可通过 include/exclude 模式调整。
- 验证操作：
  - `python -m py_compile quantization/*.py`
- 验证输出摘要：
  - 所有 `quantization/*.py` 语法检查通过。
- 结果：通过

### 步骤 2
- 变更内容：
  - 新增 `quantization/calibration.py`：
    - 读取图像并缩放到 `lq_size`
    - 使用真实 `autoencoder + diffusion` 生成 `x / timesteps / lq` 校准样本
  - 新增 `quantization/quantize_awq.py`：
    - 加载配置、UNet checkpoint、autoencoder checkpoint
    - 构建校准样本
    - 收集激活并执行 AWQ
    - 保存独立量化 checkpoint
  - 为 CPU 校准路径增加 `xformers` 关闭逻辑，避免 autoencoder 在 CPU 上因 memory-efficient attention 不可用而失败。
- 验证操作：
  - 首次执行：
```bash
python quantization/quantize_awq.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --checkpoint weights/resshift_realsrx4_s15_v1_default.pth \
  --calibration_dir onnx_inference/outputs/test_lq.png \
  --output quantization/artifacts/test_awq.pth \
  --device cpu \
  --num_images 1 \
  --samples_per_image 1 \
  --max_tokens_per_layer 64
```
  - 观测到失败：CPU 上 autoencoder 仍走 `xformers.ops.memory_efficient_attention`
  - 修复后再次执行同一命令
- 验证输出摘要：
  - 修复后输出：
    - `calibration_batches 1`
    - `collected_layers 36`
    - `quantized_layers 36`
    - `saved_to /home/ubuntu/ResShift/quantization/artifacts/test_awq.pth`
- 结果：通过

### 步骤 3
- 变更内容：
  - 新增 `quantization/verify_awq.py`：
    - 加载原始模型结构
    - 用量化 checkpoint 替换对应 `nn.Linear`
    - 执行随机前向验证
- 验证操作：
```bash
python quantization/verify_awq.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --checkpoint quantization/artifacts/test_awq.pth \
  --device cpu
```
- 验证输出摘要：
  - `loaded_quantized_layers 36`
  - `forward_ok (1, 3, 64, 64) torch.float32 True`
- 结果：通过

## 最终验证
- 端到端检查：
  - 已可显式运行 `quantization/quantize_awq.py` 生成独立 AWQ checkpoint。
  - 已可显式运行 `quantization/verify_awq.py` 加载量化 checkpoint 并完成前向。
  - 正常训练与普通推理入口未改动，因此默认行为不受影响。
- 已知限制：
  - 当前 AWQ 仅覆盖 `UNetModelSwin` 中的 `nn.Linear`，不量化卷积层和 autoencoder。
  - 当前 `AWQLinear` 在前向时会即时反量化权重，优先保证兼容性和正确性，不提供自定义低比特 kernel，因此推理加速收益有限。
  - 校准样本已尽量贴近真实 `x/lq/timestep` 分布，但目前仍是轻量实现，适合作为工程版 AWQ 起点，不是 AutoAWQ 那类高度优化实现。
