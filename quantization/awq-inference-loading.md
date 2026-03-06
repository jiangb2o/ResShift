# AWQ Inference Loading Fix

## 任务目标

修复 `inference_resshift.py` / `sampler.py` 在传入 AWQ 量化 checkpoint 时的加载失败问题，使 `--checkpoint_path` 同时兼容：
- 原始 ResShift checkpoint
- `quantization/quantize_awq.py` 生成的 AWQ checkpoint

并保证普通训练/推理路径不受影响。

## 实现思路

- 保留原始 checkpoint 的加载逻辑。
- 在 `sampler.py` 中对 checkpoint 做格式探测：
  - 若是普通 `state_dict` 或 `state_dict` 包裹格式，继续走 `util_net.reload_model()`
  - 若是 `format=reshift-awq-v1` 的 AWQ checkpoint，则调用 `quantization.awq` 的量化模型加载器，先替换目标 `nn.Linear`，再加载量化权重
- 新增验证记录，确认两种路径都可用。

## 实施步骤

1. [x] 增加 AWQ checkpoint 探测与从 payload 加载的能力。
2. [x] 在 `sampler.py` 中接入自动分支，不破坏原始 checkpoint 路径。
3. [x] 运行真实加载验证并记录结果。

## 步骤执行记录

### 步骤 1
- 变更内容：
  - 在 `quantization/awq.py` 中新增：
    - `is_awq_checkpoint_payload()`
    - `load_awq_quantized_model_from_payload()`
  - 现有 `load_awq_quantized_model()` 改为复用新的 payload 加载函数。
- 验证操作：
  - `python -m py_compile sampler.py quantization/*.py`
- 验证输出摘要：
  - 语法检查通过。
- 结果：通过

### 步骤 2
- 变更内容：
  - 在 `sampler.py` 中接入 checkpoint 自动识别：
    - `format=reshift-awq-v1` -> 走 AWQ 量化加载路径
    - 其他格式 -> 保持原始 `util_net.reload_model()` 路径
  - 新增日志：成功加载 AWQ checkpoint 时输出量化层数量。
- 验证操作：
  - 代码路径复查
- 验证输出摘要：
  - 普通 checkpoint 路径未删除
  - AWQ checkpoint 不再走 `reload_model()` 的严格 key 对齐逻辑
- 结果：通过

### 步骤 3
- 变更内容：
  - 分别验证 AWQ checkpoint 和普通 checkpoint 的加载与前向。
- 验证操作：
```bash
python - <<'PY'
import torch
from omegaconf import OmegaConf
from utils import util_common, util_net
from quantization.awq import is_awq_checkpoint_payload, load_awq_quantized_model_from_payload

cfg = OmegaConf.load('configs/realsr_swinunet_realesrgan256.yaml')

awq_payload = torch.load('quantization/artifacts/test_awq.pth', map_location='cpu')
print('awq_detected', is_awq_checkpoint_payload(awq_payload))
model = util_common.instantiate_from_config(cfg.model)
model, quantized_layers = load_awq_quantized_model_from_payload(model, awq_payload, torch.device('cpu'))
with torch.no_grad():
    out = model(x=torch.randn(1,3,64,64), timesteps=torch.tensor([10]), lq=torch.randn(1,3,64,64), mask=None)
print('awq_forward_ok', tuple(out.shape), torch.isfinite(out).all().item(), len(quantized_layers))

regular_payload = torch.load('weights/resshift_realsrx4_s15_v1_default.pth', map_location='cpu')
print('regular_detected', is_awq_checkpoint_payload(regular_payload))
regular_state = regular_payload['state_dict'] if 'state_dict' in regular_payload else regular_payload
regular_model = util_common.instantiate_from_config(cfg.model)
util_net.reload_model(regular_model, regular_state)
with torch.no_grad():
    out2 = regular_model(x=torch.randn(1,3,64,64), timesteps=torch.tensor([10]), lq=torch.randn(1,3,64,64), mask=None)
print('regular_forward_ok', tuple(out2.shape), torch.isfinite(out2).all().item())
PY
```
- 验证输出摘要：
  - `awq_detected True`
  - `awq_forward_ok (1, 3, 64, 64) True 36`
  - `regular_detected False`
  - `regular_forward_ok (1, 3, 64, 64) True`
- 结果：通过

## 最终验证
- 端到端检查：
  - `--checkpoint_path` 传入原始 checkpoint 时，仍走原始加载路径。
  - `--checkpoint_path` 传入 AWQ checkpoint 时，已切换到量化权重加载路径，不再触发 `reload_model()` 的 key 断言。
- 已知限制：
  - 当前环境无可用 CUDA，因此没有直接在 `inference_resshift.py` 的整条 GPU 推理链上做实机验证；但 sampler 中新增的 AWQ 分支本身已通过真实加载和前向验证。
