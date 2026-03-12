# Hybrid AWQ INT8 PTQ

## 任务目标

在 ResShift 现有 AWQ 方案基础上，补充一套针对 AWQ 未覆盖参数的 INT8 PTQ 方案。

约束：
- 保持原始训练与普通推理路径不变
- 量化过程可选执行
- 继续兼容当前 AWQ 量化的 `nn.Linear`
- 为 `Conv2d` 以及未被 AWQ 处理的 `Linear` 增加 INT8 PTQ
- 推理时默认采用“加载时一次性反量化缓存”，不引入新的低比特 kernel 依赖
- 新增文件放在 `quantization/` 下
- 推理加载时根据 checkpoint 类型自动选择普通/AWQ/Hybrid PTQ 路径

## 实现思路

- 保留当前 AWQ 方案：
  - 对指定 `Linear` 层执行 W4 weight-only 量化
  - 推理时默认缓存反量化到 CUDA `fp16` / CPU `fp32`
- 新增 INT8 PTQ 方案：
  - 对 AWQ 未覆盖的 `Conv2d` 与剩余 `Linear` 执行对称 per-output-channel INT8 weight-only 量化
  - 推理时同样在加载阶段一次性反量化为缓存浮点权重
- 新增 hybrid checkpoint：
  - 同时保存 AWQ 层与 INT8 PTQ 层的量化状态
  - 通过 payload `format` 区分普通 checkpoint / AWQ checkpoint / hybrid checkpoint
- 新增独立脚本：
  - hybrid 量化脚本
  - hybrid 验证脚本
- 修改推理加载入口：
  - 自动识别 hybrid checkpoint 并替换对应模块

## 实施步骤

1. [ ] 设计 hybrid checkpoint 结构与 INT8 PTQ 模块边界。
2. [ ] 实现 `quantization/int8_ptq.py` 与 `quantization/hybrid_ptq.py`。
3. [ ] 实现 hybrid 量化脚本与验证脚本。
4. [ ] 集成 `sampler.py` 的 hybrid checkpoint 自动加载。
5. [ ] 完成真实量化、加载、前向验证，并记录结果。

## 步骤执行记录

### 步骤 1
- 变更内容：
  - 明确 hybrid checkpoint 结构：
    - `format: reshift-hybrid-ptq-v1`
    - `awq_config`
    - `int8_config`
    - `model_state`
    - `awq_layers`
    - `int8_layers`
    - `source_config`
  - 明确层覆盖策略：
    - AWQ：仅处理现有规则命中的 `nn.Linear`
    - INT8 PTQ：处理 AWQ 未覆盖的 `nn.Conv2d` 与剩余 `nn.Linear`
  - 明确推理策略：
    - AWQ 与 INT8 PTQ 均默认在加载阶段一次性反量化缓存
    - CUDA 下缓存为 `fp16`
    - CPU 下缓存为 `fp32`
- 验证操作：
  - `git log --oneline --decorate -- quantization sampler.py inference_resshift.py | head -n 20`
  - `sed -n '1,260p' quantization/awq.py`
  - `sed -n '1,260p' quantization/calibration.py`
  - `sed -n '220,320p' sampler.py`
- 验证输出摘要：
  - 已确认当前 AWQ 仅覆盖选中的 `nn.Linear`
  - 已确认 AWQ checkpoint 现有格式与 `sampler.py` 的加载入口
  - 已确认当前推理默认是缓存反量化路径，适合作为 INT8 PTQ 的统一加载语义
- 结果：通过

### 步骤 2
- 变更内容：
  - 新增 `quantization/int8_ptq.py`
    - `INT8PTQConfig`
    - `INT8Linear`
    - `INT8Conv2d`
    - `collect_int8_target_layers()`
    - `quantize_model_int8_ptq()`
    - `materialize_int8_caches()`
  - 新增 `quantization/hybrid_ptq.py`
    - `quantize_model_hybrid()`
    - `save_hybrid_checkpoint()`
    - `is_hybrid_checkpoint_payload()`
    - `load_hybrid_quantized_model_from_payload()`
    - `load_hybrid_quantized_model()`
  - 实现方式：
    - AWQ 保持原有 `Linear` 路线
    - INT8 PTQ 对 AWQ 未覆盖的 `Conv2d` 与剩余 `Linear` 做 per-output-channel 对称量化
    - 推理时统一采用缓存反量化
- 验证操作：
  - `bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python -m py_compile quantization/int8_ptq.py quantization/hybrid_ptq.py sampler.py'`
- 验证输出摘要：
  - 语法检查通过，无报错
- 结果：通过

### 步骤 3
- 变更内容：
  - 新增 `quantization/quantize_hybrid_ptq.py`
  - 新增 `quantization/verify_hybrid_ptq.py`
  - hybrid 量化脚本会：
    - 构建 AWQ 校准样本
    - 对命中的 `Linear` 执行 AWQ
    - 对剩余 `Conv2d` 与 `Linear` 执行 INT8 PTQ
    - 保存 `reshift-hybrid-ptq-v1` checkpoint
- 验证操作：
```bash
bash -lc 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python quantization/quantize_hybrid_ptq.py --config configs/realsr_swinunet_realesrgan256.yaml --checkpoint weights/resshift_realsrx4_s15_v1_default.pth --calibration_dir quantization/calibration64 --output quantization/artifacts/test_hybrid_ptq.pth --device cpu --num_images 1 --samples_per_image 1 --use_linfusion True'

bash -lc 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python quantization/verify_hybrid_ptq.py --config configs/realsr_swinunet_realesrgan256.yaml --checkpoint quantization/artifacts/test_hybrid_ptq.pth --reference_checkpoint weights/resshift_realsrx4_s15_v1_default.pth --device cpu'
```
- 验证输出摘要：
  - 量化脚本输出：
    - `calibration_batches 1`
    - `collected_awq_layers 36`
    - `quantized_awq_layers 36`
    - `quantized_int8_layers 144`
    - `saved_to /home/ubuntu/ResShift/quantization/artifacts/test_hybrid_ptq.pth`
  - 验证脚本输出：
    - `loaded_awq_layers 36`
    - `loaded_int8_layers 144`
    - `module_counts awq=36 int8_linear=24 int8_conv=120`
    - `forward_ok (1, 3, 64, 64) torch.float32 True`
    - `max_abs_diff 0.15812265872955322`
    - `mean_abs_diff 0.024523476138710976`
- 结果：通过

### 步骤 4
- 变更内容：
  - 修改 `sampler.py`
    - 新增 hybrid checkpoint 识别
    - 在 `build_model()` 中优先处理 `reshift-hybrid-ptq-v1`
    - 自动替换并加载 AWQ + INT8 PTQ 模块
- 验证操作：
```bash
bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && export CUDA_VISIBLE_DEVICES=6 && python - <<\"PY\"
import torch
from omegaconf import OmegaConf
from sampler import ResShiftSampler
from quantization.awq import AWQLinear
from quantization.int8_ptq import INT8Conv2d, INT8Linear

cfg = OmegaConf.load(\"configs/realsr_swinunet_realesrgan256.yaml\")
cfg.model.ckpt_path = \"quantization/artifacts/test_hybrid_ptq.pth\"
sampler = ResShiftSampler(
    configs=cfg,
    sf=4,
    use_amp=False,
    chop_size=64,
    chop_stride=64,
    chop_bs=1,
    seed=123,
    profile_inference=False,
)
with torch.no_grad():
    x = torch.randn(1, 3, 64, 64, device=\"cuda\")
    lq = torch.randn(1, 3, 64, 64, device=\"cuda\")
    t = torch.tensor([10], dtype=torch.long, device=\"cuda\")
    out = sampler.model(x=x, timesteps=t, lq=lq, mask=None)
print(
    \"sampler_hybrid_ok\",
    tuple(out.shape),
    out.dtype,
    bool(torch.isfinite(out).all().item()),
    sum(1 for m in sampler.model.modules() if isinstance(m, AWQLinear)),
    sum(1 for m in sampler.model.modules() if isinstance(m, INT8Linear)),
    sum(1 for m in sampler.model.modules() if isinstance(m, INT8Conv2d)),
)
PY'
```
- 验证输出摘要：
  - `Loaded hybrid PTQ checkpoint with 36 AWQ linear layers and 144 INT8 PTQ layers.`
  - `sampler_hybrid_ok (1, 3, 64, 64) torch.float32 True 36 24 120`
- 结果：通过

### 步骤 5
- 变更内容：
  - 检查 hybrid checkpoint 的覆盖范围和实际体积
- 验证操作：
```bash
ls -lh weights/resshift_realsrx4_s15_v1_default.pth quantization/artifacts/test_awq.pth quantization/artifacts/test_hybrid_ptq.pth

bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python - <<\"PY\"
import torch
payload = torch.load(\"quantization/artifacts/test_hybrid_ptq.pth\", map_location=\"cpu\")
print(\"format\", payload[\"format\"])
print(\"awq_layers\", len(payload[\"awq_layers\"]))
print(\"int8_layers\", len(payload[\"int8_layers\"]))
print(\"int8_conv\", sum(1 for v in payload[\"int8_layers\"].values() if v[\"module_type\"] == \"conv2d\"))
print(\"int8_linear\", sum(1 for v in payload[\"int8_layers\"].values() if v[\"module_type\"] == \"linear\"))
PY'
```
- 验证输出摘要：
  - 文件大小：
    - 原始 checkpoint：`456M`
    - 纯 AWQ checkpoint：`449M`
    - hybrid checkpoint：`118M`
  - payload 结构：
    - `format reshift-hybrid-ptq-v1`
    - `awq_layers 36`
    - `int8_layers 144`
    - `int8_conv 120`
    - `int8_linear 24`
- 结果：通过

## 最终验证
- 端到端检查：
  - hybrid 量化 checkpoint 已能在 CPU 下完成加载和前向
  - `sampler.py` 已能在真实 CUDA 环境按 checkpoint 类型自动走 hybrid 加载分支
  - 当前测试产物位于 `quantization/artifacts/test_hybrid_ptq.pth`
- 已知限制：
  - 当前 INT8 PTQ 是 weight-only 路线，不执行原生 INT8 activation/kernel 推理
  - 当前未量化 `LayerNorm`、相对位置偏置等非 `Conv2d/Linear` 参数
  - 数值验证使用的是最小规模校准样本，正式量化应扩大校准集
  - 当 INT8 `Conv2d` 在 CUDA 下恢复为普通 `fp16 Conv2d` 后，直接裸调 `model(x)` 需要 AMP 或半精度输入；标准 `sampler/inference_resshift.py` 推理路径默认已启用 AMP

## 增量更新：INT8 Conv 恢复为普通 Conv2d

### 目标

将 hybrid checkpoint 中的 INT8 `Conv2d` 在加载时一次性反量化并恢复为普通 `nn.Conv2d`，避免推理阶段继续经过 `INT8Conv2d` wrapper。

### 实现

- 在 `quantization/int8_ptq.py` 中新增：
  - `restore_int8_conv_modules(model, device, dtype=None)`
- 在 `quantization/hybrid_ptq.py` 中新增加载参数：
  - `restore_int8_conv=True`
- 当前默认行为：
  - 加载 hybrid checkpoint
  - 用量化模块承接 state_dict
  - `model.to(device)`
  - 将所有 `INT8Conv2d` 一次性反量化并替换为普通 `nn.Conv2d`
  - 对剩余 `INT8Linear` 仍保留缓存反量化
- 在 `sampler.py` 中更新日志：
  - 明确打印恢复成普通 `Conv2d` 的层数

### 验证 1：CPU 下恢复版与 wrapper 版数值一致

- 验证操作：
```bash
bash -lc 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python - <<\"PY\"
import torch
from omegaconf import OmegaConf
from utils import util_common
from quantization.hybrid_ptq import load_hybrid_quantized_model
from quantization.int8_ptq import INT8Conv2d

cfg = OmegaConf.load(\"configs/realsr_swinunet_realesrgan256.yaml\")
device = torch.device(\"cpu\")

model_restore = util_common.instantiate_from_config(cfg.model)
model_restore, _, _ = load_hybrid_quantized_model(model_restore, \"quantization/artifacts/test_hybrid_ptq.pth\", device, restore_int8_conv=True)
model_wrap = util_common.instantiate_from_config(cfg.model)
model_wrap, _, _ = load_hybrid_quantized_model(model_wrap, \"quantization/artifacts/test_hybrid_ptq.pth\", device, restore_int8_conv=False)

x = torch.randn(1, 3, 64, 64)
lq = torch.randn(1, 3, 64, 64)
t = torch.tensor([10], dtype=torch.long)
with torch.no_grad():
    out_restore = model_restore(x=x, timesteps=t, lq=lq, mask=None)
    out_wrap = model_wrap(x=x, timesteps=t, lq=lq, mask=None)
diff = (out_restore - out_wrap).abs()
print(\"int8_conv_restore_count\", sum(1 for m in model_restore.modules() if isinstance(m, INT8Conv2d)))
print(\"int8_conv_wrapper_count\", sum(1 for m in model_wrap.modules() if isinstance(m, INT8Conv2d)))
print(\"max_abs_diff\", diff.max().item())
print(\"mean_abs_diff\", diff.mean().item())
PY'
```
- 验证输出摘要：
  - `int8_conv_restore_count 0`
  - `int8_conv_wrapper_count 120`
  - `max_abs_diff 0.0`
  - `mean_abs_diff 0.0`
- 结果：通过

### 验证 2：真实 CUDA 下恢复版比 wrapper 版更快

- 验证操作：
```bash
bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && export CUDA_VISIBLE_DEVICES=6 && python - <<\"PY\"
import time
import torch
from omegaconf import OmegaConf
from utils import util_common
from quantization.hybrid_ptq import load_hybrid_quantized_model
from quantization.int8_ptq import INT8Conv2d

cfg = OmegaConf.load(\"configs/realsr_swinunet_realesrgan256.yaml\")
device = torch.device(\"cuda\")

def load_model(restore):
    model = util_common.instantiate_from_config(cfg.model)
    model, _, _ = load_hybrid_quantized_model(
        model,
        \"quantization/artifacts/test_hybrid_ptq.pth\",
        device,
        restore_int8_conv=restore,
    )
    return model

model_restore = load_model(True)
model_wrap = load_model(False)

x = torch.randn(1, 3, 64, 64, device=device)
lq = torch.randn(1, 3, 64, 64, device=device)
t = torch.tensor([10], dtype=torch.long, device=device)

for _ in range(3):
    with torch.no_grad(), torch.cuda.amp.autocast():
        model_restore(x=x, timesteps=t, lq=lq, mask=None)
        model_wrap(x=x, timesteps=t, lq=lq, mask=None)
torch.cuda.synchronize()

def bench(model, iters=20):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for _ in range(iters):
            out = model(x=x, timesteps=t, lq=lq, mask=None)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters, out

ms_restore, out_restore = bench(model_restore)
ms_wrap, out_wrap = bench(model_wrap)
diff = (out_restore.float() - out_wrap.float()).abs()
print(\"cuda_restore_conv_wrappers\", sum(1 for m in model_restore.modules() if isinstance(m, INT8Conv2d)))
print(\"cuda_wrap_conv_wrappers\", sum(1 for m in model_wrap.modules() if isinstance(m, INT8Conv2d)))
print(\"restore_ms\", round(ms_restore, 3))
print(\"wrap_ms\", round(ms_wrap, 3))
print(\"speedup\", round(ms_wrap / ms_restore, 4))
print(\"max_abs_diff\", diff.max().item())
print(\"mean_abs_diff\", diff.mean().item())
PY'
```
- 验证输出摘要：
  - `cuda_restore_conv_wrappers 0`
  - `cuda_wrap_conv_wrappers 120`
  - `restore_ms 26.676`
  - `wrap_ms 29.532`
  - `speedup 1.1071`
  - `max_abs_diff 0.001708984375`
  - `mean_abs_diff 0.00030418415553867817`
- 结果：通过

### 验证 3：真实 sampler CUDA 加载路径已使用恢复版 Conv2d

- 验证操作：
```bash
bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && export CUDA_VISIBLE_DEVICES=6 && python - <<\"PY\"
import torch
from omegaconf import OmegaConf
from sampler import ResShiftSampler
from quantization.int8_ptq import INT8Conv2d, INT8Linear
from quantization.awq import AWQLinear

cfg = OmegaConf.load(\"configs/realsr_swinunet_realesrgan256.yaml\")
cfg.model.ckpt_path = \"quantization/artifacts/test_hybrid_ptq.pth\"
sampler = ResShiftSampler(
    configs=cfg,
    sf=4,
    use_amp=True,
    chop_size=64,
    chop_stride=64,
    chop_bs=1,
    seed=123,
    profile_inference=False,
)
with torch.no_grad(), torch.cuda.amp.autocast():
    x = torch.randn(1, 3, 64, 64, device=\"cuda\")
    lq = torch.randn(1, 3, 64, 64, device=\"cuda\")
    t = torch.tensor([10], dtype=torch.long, device=\"cuda\")
    out = sampler.model(x=x, timesteps=t, lq=lq, mask=None)
print(
    \"sampler_restore_ok\",
    tuple(out.shape),
    out.dtype,
    bool(torch.isfinite(out).all().item()),
    sum(1 for m in sampler.model.modules() if isinstance(m, AWQLinear)),
    sum(1 for m in sampler.model.modules() if isinstance(m, INT8Linear)),
    sum(1 for m in sampler.model.modules() if isinstance(m, INT8Conv2d)),
)
PY'
```
- 验证输出摘要：
  - `Loaded hybrid PTQ checkpoint with 36 AWQ linear layers, 24 INT8 linear layers, and restored 120 INT8 conv layers to plain Conv2d.`
  - `sampler_restore_ok (1, 3, 64, 64) torch.float16 True 36 24 0`
- 结果：通过

## 增量更新：`with_awq` 开关

### 目标

为 hybrid 量化脚本增加 `with_awq` 布尔选项：

- `with_awq=True`
  - 保持当前行为
  - 先执行 AWQ，再对剩余层做 INT8 PTQ
- `with_awq=False`
  - 不执行 AWQ 量化流程
  - 不收集 AWQ 激活
  - 不构建 AWQ 所需校准样本
  - 所有 `Conv2d + Linear` 全部走 INT8 PTQ

### 实现

- 在 `quantization/quantize_hybrid_ptq.py` 中新增：
  - `--with_awq`
- 当 `with_awq=False` 时：
  - `--calibration_dir` 不再是必填
  - 跳过 `autoencoder` 加载
  - 跳过 `diffusion` 构建
  - 跳过 `build_calibration_batches()`
  - 跳过 `collect_awq_activations()`
- 在 `quantization/hybrid_ptq.py` 中：
  - `quantize_model_hybrid(..., with_awq: bool = True)`
  - `save_hybrid_checkpoint(..., with_awq: bool)`
  - payload 中新增：
    - `with_awq`

### 验证 1：纯 INT8 PTQ 导出

- 验证操作：
```bash
bash -lc 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python quantization/quantize_hybrid_ptq.py --config configs/realsr_swinunet_realesrgan256.yaml --checkpoint weights/resshift_realsrx4_s15_v1_default.pth --output quantization/artifacts/test_int8_only_ptq.pth --device cpu --with_awq False --quantize_conv True --quantize_linear True --use_linfusion True'
```
- 验证输出摘要：
  - `calibration_batches 0`
  - `collected_awq_layers 0`
  - `quantized_awq_layers 0`
  - `quantized_int8_layers 180`
  - `saved_to /home/ubuntu/ResShift/quantization/artifacts/test_int8_only_ptq.pth`
- 结果：通过

### 验证 2：checkpoint payload 确认

- 验证操作：
```bash
bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python - <<\"PY\"
import torch
payload = torch.load(\"quantization/artifacts/test_int8_only_ptq.pth\", map_location=\"cpu\")
print(\"format\", payload[\"format\"])
print(\"with_awq\", payload[\"with_awq\"])
print(\"awq_layers\", len(payload[\"awq_layers\"]))
print(\"int8_layers\", len(payload[\"int8_layers\"]))
PY'
```
- 验证输出摘要：
  - `format reshift-hybrid-ptq-v1`
  - `with_awq False`
  - `awq_layers 0`
  - `int8_layers 180`
- 结果：通过

### 验证 3：CPU 加载与前向

- 验证操作：
```bash
bash -lc 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python quantization/verify_hybrid_ptq.py --config configs/realsr_swinunet_realesrgan256.yaml --checkpoint quantization/artifacts/test_int8_only_ptq.pth --reference_checkpoint weights/resshift_realsrx4_s15_v1_default.pth --device cpu'
```
- 验证输出摘要：
  - `loaded_awq_layers 0`
  - `loaded_int8_layers 180`
  - `module_counts awq=0 int8_linear=60 int8_conv=0`
  - `forward_ok (1, 3, 64, 64) torch.float32 True`
  - `max_abs_diff 0.04266566038131714`
  - `mean_abs_diff 0.006817747373133898`
- 结果：通过

### 验证 4：真实 sampler CUDA 加载

- 验证操作：
```bash
bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && export CUDA_VISIBLE_DEVICES=6 && python - <<\"PY\"
import torch
from omegaconf import OmegaConf
from sampler import ResShiftSampler
from quantization.int8_ptq import INT8Conv2d, INT8Linear
from quantization.awq import AWQLinear

cfg = OmegaConf.load(\"configs/realsr_swinunet_realesrgan256.yaml\")
cfg.model.ckpt_path = \"quantization/artifacts/test_int8_only_ptq.pth\"
sampler = ResShiftSampler(
    configs=cfg,
    sf=4,
    use_amp=True,
    chop_size=64,
    chop_stride=64,
    chop_bs=1,
    seed=123,
    profile_inference=False,
)
with torch.no_grad(), torch.cuda.amp.autocast():
    x = torch.randn(1, 3, 64, 64, device=\"cuda\")
    lq = torch.randn(1, 3, 64, 64, device=\"cuda\")
    t = torch.tensor([10], dtype=torch.long, device=\"cuda\")
    out = sampler.model(x=x, timesteps=t, lq=lq, mask=None)
print(
    \"sampler_int8_only_ok\",
    tuple(out.shape),
    out.dtype,
    bool(torch.isfinite(out).all().item()),
    sum(1 for m in sampler.model.modules() if isinstance(m, AWQLinear)),
    sum(1 for m in sampler.model.modules() if isinstance(m, INT8Linear)),
    sum(1 for m in sampler.model.modules() if isinstance(m, INT8Conv2d)),
)
PY'
```
- 验证输出摘要：
  - `Loaded hybrid PTQ checkpoint with 0 AWQ linear layers, 60 INT8 linear layers, and restored 120 INT8 conv layers to plain Conv2d.`
  - `sampler_int8_only_ok (1, 3, 64, 64) torch.float16 True 0 60 0`
- 结果：通过
