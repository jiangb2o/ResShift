# AWQ FP16 Dequant

## 任务目标

将 AWQ 推理路径的默认行为固定为：
- CUDA 上：量化权重反量化为 `fp16` 后执行线性层
- CPU 上：量化权重反量化为 `fp32` 后执行线性层

要求：
- 保持与现有 AWQ checkpoint 格式兼容
- 普通 checkpoint 路径不受影响
- 在真实 CUDA 环境中验证该行为

## 实现思路

- 不改动 AWQ checkpoint 存储格式。
- 仅修改 `AWQLinear.forward()` 的计算 dtype 选择：
  - `x.is_cuda == True` -> 反量化权重和 bias 使用 `fp16`
  - 否则使用 `fp32`
- 为了兼容当前模型其他部分，线性层计算后再把输出 cast 回输入 `x.dtype`。

## 实施步骤

1. [x] 修改 `AWQLinear` 的默认反量化 dtype 选择逻辑。
2. [x] 验证 AWQ checkpoint 在 CPU/CUDA 下都能正常前向。
3. [x] 将真实 CUDA 结果记录到文档中。

## 步骤执行记录

### 步骤 1
- 变更内容：
  - 在 `quantization/awq.py` 中新增 `AWQLinear._compute_dtype()`：
    - CUDA -> `torch.float16`
    - CPU -> `torch.float32`
  - 修改 `AWQLinear.forward()`：
    - 输入先转到计算 dtype
    - 量化权重反量化到计算 dtype
    - `F.linear` 后将输出转回原输入 dtype
- 验证操作：
  - `python -m py_compile quantization/awq.py`
- 验证输出摘要：
  - 语法检查通过。
- 结果：通过

### 步骤 2
- 变更内容：
  - 验证 CPU 路径保持 `fp32` 反量化。
  - 在真实 CUDA 环境中验证 AWQ 默认走 `fp16` 反量化。
- 验证操作：
```bash
python - <<'PY'
import torch
from omegaconf import OmegaConf
from utils import util_common
from quantization.awq import load_awq_quantized_model, AWQLinear

cfg = OmegaConf.load('configs/realsr_swinunet_realesrgan256.yaml')
model = util_common.instantiate_from_config(cfg.model)
model, layers = load_awq_quantized_model(model, 'quantization/artifacts/test_awq.pth', torch.device('cpu'))
layer = next(m for m in model.modules() if isinstance(m, AWQLinear))
print('cpu_compute_dtype', layer._compute_dtype(torch.randn(1, 192)))
with torch.no_grad():
    out = model(x=torch.randn(1,3,64,64), timesteps=torch.tensor([10]), lq=torch.randn(1,3,64,64), mask=None)
print('cpu_forward_ok', tuple(out.shape), out.dtype, torch.isfinite(out).all().item(), len(layers))
PY
```

```bash
bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && export CUDA_VISIBLE_DEVICES=4 && python - <<"PY"
import torch
from omegaconf import OmegaConf
from utils import util_common
from quantization.awq import load_awq_quantized_model, AWQLinear

cfg = OmegaConf.load("configs/realsr_swinunet_realesrgan256.yaml")
device = torch.device("cuda")
model = util_common.instantiate_from_config(cfg.model)
model, layers = load_awq_quantized_model(model, "quantization/artifacts/test_awq.pth", device)
layer = next(m for m in model.modules() if isinstance(m, AWQLinear))
print("cuda_compute_dtype", layer._compute_dtype(torch.randn(1, 192, device=device)))
with torch.no_grad():
    out = model(x=torch.randn(1,3,64,64, device=device), timesteps=torch.tensor([10], dtype=torch.long, device=device), lq=torch.randn(1,3,64,64, device=device), mask=None)
    torch.cuda.synchronize()
print("cuda_forward_ok", tuple(out.shape), out.dtype, torch.isfinite(out).all().item(), len(layers))
PY'
```
- 验证输出摘要：
  - `cpu_compute_dtype torch.float32`
  - `cpu_forward_ok (1, 3, 64, 64) torch.float32 True 36`
  - `cuda_compute_dtype torch.float16`
  - `cuda_forward_ok (1, 3, 64, 64) torch.float32 True 36`
- 结果：通过

### 步骤 3
- 变更内容：
  - 将真实 CUDA 验证结果回填到本文档。
- 验证操作：
  - 文档回填
- 验证输出摘要：
  - 已记录 CPU 和真实 CUDA 的实际运行结果。
- 结果：通过

## 最终验证
- 端到端检查：
  - AWQ checkpoint 现在默认在 CUDA 上以 `fp16` 反量化执行。
  - CPU 路径保持 `fp32` 反量化执行。
  - 普通 checkpoint 路径不受影响。
- 已知限制：
  - 当前输出仍会 cast 回输入 dtype，因此如果整网主体仍以 `fp32` 运行为主，AWQ 线性层只是内部用 `fp16` 计算，不代表整网已经完全转成 `fp16` 推理。
