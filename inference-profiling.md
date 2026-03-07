# Inference Profiling

## 任务目标

为 ResShift 推理增加可选 profiling，优先分析：
- `UNet` 耗时
- `autoencoder.encode` 耗时
- `autoencoder.decode` 耗时

要求：
- 默认推理行为不变
- profiling 可选择性开启
- 统计语义与真实执行单元一致：`_process_per_image()` 实际处理的是一个 batch，因此 profiling 需要按 batch 统计，并同时给出 per-image 归一化结果

## 实现思路

- 在推理入口增加 profiling 开关。
- 在 `sampler.py` 中实现轻量 profiler：
  - 对 `self.model` 的每次前向计时，统计 `UNet` 总耗时、调用次数、平均耗时
  - 对 `self.autoencoder.encode` / `decode` 分别计时
- 统计维度：
  - per-batch 汇总
  - per-image 归一化汇总
  - whole-run 汇总
- 计时方式：
  - CUDA 可用时在关键前后 `torch.cuda.synchronize()`，保证计时真实
  - CPU 时直接使用 `time.perf_counter()`
- warmup 完成后重置 profiling 统计，避免 warmup 污染正式推理结果。

## 实施步骤

1. [x] 设计并实现 batch-aware profiling 数据结构。
2. [x] 在 `sampler.py` 中接入 batch size 统计和 per-batch/per-image 日志输出。
3. [x] 运行真实验证并记录结果。

## 步骤执行记录

### 步骤 1
- 变更内容：
  - 在 `sampler.py` 中新增：
    - `_TimerStat`
    - `_InferenceProfiler`
    - `_ProfiledModule`
    - `_ProfiledAutoencoder`
  - `_InferenceProfiler` 改为 batch-aware：
    - `begin_batch(batch_size)`
    - `end_batch()`
    - `global_images`
    - `current_images`
- 验证操作：
  - `python -m py_compile inference_resshift.py sampler.py`
- 验证输出摘要：
  - 语法检查通过。
- 结果：通过

### 步骤 2
- 变更内容：
  - 在 `inference_resshift.py` 中新增 `--profile_inference` 开关。
  - 在 `ResShiftSampler` 初始化时传递 `profile_inference`。
  - 在 `sampler.py` 中：
    - `_process_per_image()` 改为按 batch 开始/结束统计
    - 日志从 `Image profile` 改为 `Batch profile`
    - 每项统计同时输出：
      - `batch_total`
      - `per_image`
      - `calls`
      - `per_call`
    - warmup 后调用 `reset_global()` 清空统计
- 验证操作：
  - 代码路径复查
- 验证输出摘要：
  - profiling 默认关闭，不影响原推理路径
  - 开启后会输出 Batch/Global 两级统计，且带 per-image 归一化值
- 结果：通过

### 步骤 3
- 变更内容：
  - 使用实际模型和 autoencoder，在 CPU 环境下验证 batch-aware profiler 包装器会记录真实耗时。
  - 为避免 CPU 上 `xformers` 不可用，验证时显式关闭 autoencoder 的 xformers 分支。
- 验证操作：
```bash
python - <<'PY'
import torch
from omegaconf import OmegaConf
from utils import util_common, util_net
from sampler import _InferenceProfiler, _ProfiledModule, _ProfiledAutoencoder
from ldm.modules.diffusionmodules import model as diffusion_model

diffusion_model.XFORMERS_IS_AVAILBLE = False
cfg = OmegaConf.load('configs/realsr_swinunet_realesrgan256.yaml')
profiler = _InferenceProfiler(enabled=True)

model = util_common.instantiate_from_config(cfg.model).eval()
ckpt = torch.load('weights/resshift_realsrx4_s15_v1_default.pth', map_location='cpu')
state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
util_net.reload_model(model, state)
model = _ProfiledModule(model, profiler, 'unet')

ae = util_common.instantiate_from_config(cfg.autoencoder).eval()
ae_ckpt = torch.load(cfg.autoencoder.ckpt_path, map_location='cpu')
ae_state = ae_ckpt['state_dict'] if 'state_dict' in ae_ckpt else ae_ckpt
util_net.reload_model(ae, ae_state)
ae = _ProfiledAutoencoder(ae, profiler)

profiler.begin_batch(2)
with torch.no_grad():
    _ = ae.encode(torch.randn(2, 3, 256, 256))
    _ = ae.decode(torch.randn(2, 3, 64, 64))
    _ = model(x=torch.randn(2,3,64,64), timesteps=torch.tensor([10, 10]), lq=torch.randn(2,3,64,64), mask=None)
stats, batch_images = profiler.end_batch()
for line in profiler.summary_lines(stats, prefix='test | ', image_count=batch_images):
    print(line)
PY
```
- 验证输出摘要：
  - `test | images=2`
  - `test | UNet: batch_total=721.79 ms, per_image=360.90 ms, calls=1, per_call=721.79 ms`
  - `test | AE.encode: batch_total=1390.98 ms, per_image=695.49 ms, calls=1, per_call=1390.98 ms`
  - `test | AE.decode: batch_total=2176.66 ms, per_image=1088.33 ms, calls=1, per_call=2176.66 ms`
  - `test | Profiled subtotal: batch_total=4289.43 ms, per_image=2144.71 ms`
- 结果：通过

## 最终验证
- 端到端检查：
  - 已支持通过 `--profile_inference True` 开启推理 profiling。
  - 已可输出 `UNet`、`AE.encode`、`AE.decode` 的 batch 总耗时和 per-image 归一化耗时。
  - 统计语义已与 `_process_per_image()` 的真实 batch 执行方式对齐。
- 已知限制：
  - 当前环境无可用 CUDA，未直接跑 `inference_resshift.py` 的整条 GPU 推理链；但 profiler 包装逻辑和真实模型前向计时已在 CPU 环境下验证。
  - 目前 profiling 聚焦 `UNet` 与 autoencoder，不含数据加载、切图拼接、图像保存等外围耗时拆分。
