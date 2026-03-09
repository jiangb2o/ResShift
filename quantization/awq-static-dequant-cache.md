# AWQ Static Dequant Cache

## 任务目标

分析 ResShift 的 AWQ 推理中反量化开销，并将当前“每次 forward 动态反量化”改为“加载 checkpoint 时一次性恢复反量化缓存”。

要求：
- 默认使用加载时一次性恢复的缓存模式
- 保持与现有 AWQ checkpoint 格式兼容
- 保留动态反量化模式，仅用于基准对比
- 在真实 CUDA 环境下给出动态模式与缓存模式的推理速度对比

## 实现思路

- 当前动态反量化发生在 `AWQLinear.forward()`：每次调用都会执行 `_dequantize_weight()`，逐 group 重建浮点权重，再执行 `F.linear`。
- 改造为：
  - `AWQLinear` 支持两种模式：
    - `dynamic`：每次 forward 现算反量化权重
    - `cached`：在加载时一次性构建 `fp16/fp32` 权重缓存，forward 直接使用缓存
  - AWQ checkpoint 默认使用 `cached`
- 用真实 CUDA 环境测三组结果：
  - 动态模式模型前向平均耗时
  - 缓存模式模型前向平均耗时
  - 一次性缓存构建的纯耗时

## 缓存反量化实现细节

缓存反量化的核心目标是：把原来每次 `forward()` 都做一次的 `_dequantize_weight()`，提前到模型加载阶段只做一次。

### 1. 在 `AWQLinear` 中增加缓存张量

文件：
- `quantization/awq.py`

新增成员：
- `weight_cache`
- `bias_cache`
- `dequant_mode`

其中：
- `weight_cache`：保存一次性恢复后的浮点权重
- `bias_cache`：保存一次性转换后的 bias
- `dequant_mode`：控制当前层使用
  - `dynamic`
  - `cached`

### 2. 将原本的反量化公式封装保留

反量化公式仍然是原来的：
- 先按 group 执行 `qweight.float() * scales`
- 再整体除以 `awq_scale`

对应函数：
- `AWQLinear._dequantize_weight()`

也就是说，缓存模式没有改变数值逻辑，只是改变了执行时机。

### 3. 新增 `materialize_cache()`

新增函数：
- `AWQLinear.materialize_cache(device, dtype)`

它在加载阶段做两件事：
1. 调用 `_dequantize_weight()` 生成完整浮点权重
2. 直接把结果放到目标设备和目标 dtype 上

即：
- CUDA -> `fp16`
- CPU -> `fp32`

并把结果写入：
- `self.weight_cache`
- `self.bias_cache`

### 4. 新增整模型缓存恢复入口

新增函数：
- `materialize_awq_caches(model, device)`

它会遍历整棵模型：
- 找到所有 `AWQLinear`
- 对每个 `AWQLinear` 调用 `materialize_cache(...)`

这一步是在 AWQ checkpoint 加载完成后统一执行的，因此所有量化线性层都会在推理开始前准备好缓存。

### 5. 在 checkpoint 加载流程里默认触发缓存恢复

函数：
- `load_awq_quantized_model_from_payload(..., dequant_mode=\"cached\")`

当前默认行为：
1. 先把原始 `nn.Linear` 替换为 `AWQLinear`
2. 加载 `qweight/scales/awq_scale/bias`
3. `model.to(device)`
4. 如果 `dequant_mode == \"cached\"`：
   - 调用 `materialize_awq_caches(model, device)`

这意味着：
- 缓存是在“模型已经放到目标设备之后”生成的
- 因此不会先在 CPU 生成再搬到 GPU，而是直接在目标推理设备上生成最终缓存

### 6. `forward()` 如何切换动态/缓存模式

在 `AWQLinear.forward()` 中：

- 如果 `dequant_mode == \"cached\"`
  - 优先直接使用 `weight_cache / bias_cache`
  - 如果缓存为空，或者 device/dtype 不匹配，再即时补一次 `materialize_cache()`

- 如果 `dequant_mode == \"dynamic\"`
  - 每次前向都重新执行 `_dequantize_weight()`
  - 不使用缓存

因此当前默认推理路径实际上是：
- 加载时一次性恢复缓存
- 推理时直接消费缓存

### 7. 为什么数值和动态模式完全一致

因为：
- 动态模式和缓存模式调用的是同一套反量化公式
- 区别只在于执行时机不同

所以这次真实 CUDA 对比里：
- `max_abs_diff 0.0`
- `mean_abs_diff 0.0`

这说明缓存模式只是把“重复做的事情”前移到了加载时，并没有改变结果。

## 实施步骤

1. [x] 实现 `AWQLinear` 的缓存反量化机制，并将默认模式切到 `cached`。
2. [x] 更新 AWQ checkpoint 加载流程，在加载时一次性恢复缓存。
3. [x] 在真实 CUDA 环境下对比 `dynamic` 与 `cached` 的推理速度，并记录结果。

## 步骤执行记录

### 步骤 1
- 变更内容：
  - 在 `quantization/awq.py` 中为 `AWQLinear` 新增：
    - `weight_cache`
    - `bias_cache`
    - `dequant_mode`
    - `set_dequant_mode()`
    - `materialize_cache()`
    - `clear_cache()`
  - `forward()` 现在支持两种路径：
    - `dynamic`：每次执行 `_dequantize_weight()`
    - `cached`：直接使用已恢复的缓存权重
  - 默认模式为 `cached`。
- 验证操作：
  - `bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python -m py_compile quantization/awq.py sampler.py'`
- 验证输出摘要：
  - 语法检查通过。
- 结果：通过

### 步骤 2
- 变更内容：
  - 在 `load_awq_quantized_model_from_payload(..., dequant_mode="cached")` 中：
    - 默认将量化层设置为 `cached`
    - 在 `model.to(device)` 后调用 `materialize_awq_caches(model, device)`，一次性恢复所有 AWQLinear 的缓存权重
  - 为兼容当前 `use_linfusion=True` 模型结构，加载时会过滤掉 payload 中当前模型不存在的额外键（例如旧 checkpoint 中的 `relative_position_bias_table`）。
- 验证操作：
```bash
bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && python - <<"PY"
import torch
from omegaconf import OmegaConf
from utils import util_common
from quantization.awq import load_awq_quantized_model, AWQLinear

cfg = OmegaConf.load("configs/realsr_swinunet_realesrgan256.yaml")
for mode in ["dynamic", "cached"]:
    model = util_common.instantiate_from_config(cfg.model)
    model, layers = load_awq_quantized_model(model, "quantization/artifacts/test_awq.pth", torch.device("cpu"), dequant_mode=mode)
    layer = next(m for m in model.modules() if isinstance(m, AWQLinear))
    print("mode", mode, "cache_numel", int(layer.weight_cache.numel()), "dtype", layer.weight_cache.dtype if layer.weight_cache.numel() else "empty")
    with torch.no_grad():
        out = model(x=torch.randn(1,3,64,64), timesteps=torch.tensor([10]), lq=torch.randn(1,3,64,64), mask=None)
    print("forward_ok", mode, tuple(out.shape), out.dtype, torch.isfinite(out).all().item(), len(layers))
PY'
```
- 验证输出摘要：
  - `mode dynamic cache_numel 0 dtype empty`
  - `forward_ok dynamic (1, 3, 64, 64) torch.float32 True 36`
  - `mode cached cache_numel 110592 dtype torch.float32`
  - `forward_ok cached (1, 3, 64, 64) torch.float32 True 36`
- 结果：通过

### 步骤 3
- 变更内容：
  - 在真实 CUDA 环境中对比 `dynamic` 和 `cached` 两种 AWQ 反量化模式。
  - 额外测量纯缓存构建时间，单独分析“一次性恢复”的成本。
- 验证操作：
```bash
bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && export CUDA_VISIBLE_DEVICES=4 && python - <<"PY"
import time
import torch
from omegaconf import OmegaConf
from utils import util_common
from quantization.awq import load_awq_quantized_model, AWQLinear

torch.manual_seed(0)
cfg = OmegaConf.load("configs/realsr_swinunet_realesrgan256.yaml")
device = torch.device("cuda")

start = time.perf_counter()
model_cached = util_common.instantiate_from_config(cfg.model)
model_cached, layers_c = load_awq_quantized_model(model_cached, "quantization/artifacts/test_awq.pth", device, dequant_mode="cached")
torch.cuda.synchronize()
cache_load_ms = (time.perf_counter() - start) * 1000.0

model_dynamic = util_common.instantiate_from_config(cfg.model)
model_dynamic, layers_d = load_awq_quantized_model(model_dynamic, "quantization/artifacts/test_awq.pth", device, dequant_mode="dynamic")

x = torch.randn(4, 3, 64, 64, device=device)
lq = torch.randn(4, 3, 64, 64, device=device)
t = torch.tensor([10, 10, 10, 10], dtype=torch.long, device=device)

for _ in range(3):
    with torch.no_grad():
        model_dynamic(x=x, timesteps=t, lq=lq, mask=None)
        model_cached(x=x, timesteps=t, lq=lq, mask=None)
torch.cuda.synchronize()

with torch.no_grad():
    out_dyn = model_dynamic(x=x, timesteps=t, lq=lq, mask=None)
    out_cache = model_cached(x=x, timesteps=t, lq=lq, mask=None)
torch.cuda.synchronize()
diff = (out_dyn - out_cache).abs()

def bench(model, iters=20):
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            model(x=x, timesteps=t, lq=lq, mask=None)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters

ms_dynamic = bench(model_dynamic)
ms_cached = bench(model_cached)
first_layer_dynamic = next(m for m in model_dynamic.modules() if isinstance(m, AWQLinear))
first_layer_cached = next(m for m in model_cached.modules() if isinstance(m, AWQLinear))
print("cuda_dequant_compare")
print("batch_size", x.shape[0])
print("quantized_layers", len(layers_d), len(layers_c))
print("cache_load_ms", round(cache_load_ms, 3))
print("dynamic_ms", round(ms_dynamic, 3))
print("cached_ms", round(ms_cached, 3))
print("speedup", round(ms_dynamic / ms_cached, 4))
print("dynamic_cache_numel", int(first_layer_dynamic.weight_cache.numel()))
print("cached_cache_numel", int(first_layer_cached.weight_cache.numel()))
print("cached_dtype", first_layer_cached.weight_cache.dtype)
print("max_abs_diff", float(diff.max()))
print("mean_abs_diff", float(diff.mean()))
PY'
```

```bash
bash -lc 'source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate ResShift && export CUDA_VISIBLE_DEVICES=4 && python - <<"PY"
import time
import torch
from omegaconf import OmegaConf
from utils import util_common
from quantization.awq import load_awq_quantized_model, materialize_awq_caches, AWQLinear

cfg = OmegaConf.load("configs/realsr_swinunet_realesrgan256.yaml")
device = torch.device("cuda")
model = util_common.instantiate_from_config(cfg.model)
model, _ = load_awq_quantized_model(model, "quantization/artifacts/test_awq.pth", device, dequant_mode="dynamic")
first = next(m for m in model.modules() if isinstance(m, AWQLinear))
print("before_cache_numel", int(first.weight_cache.numel()))
torch.cuda.synchronize()
start = time.perf_counter()
materialize_awq_caches(model, device)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1000.0
print("materialize_only_ms", round(elapsed_ms, 3))
print("after_cache_numel", int(first.weight_cache.numel()), first.weight_cache.dtype)
PY'
```
- 验证输出摘要：
  - `cuda_dequant_compare`
  - `batch_size 4`
  - `quantized_layers 36 36`
  - `cache_load_ms 4436.807`
  - `dynamic_ms 35.447`
  - `cached_ms 32.305`
  - `speedup 1.0973`
  - `dynamic_cache_numel 0`
  - `cached_cache_numel 110592`
  - `cached_dtype torch.float16`
  - `max_abs_diff 0.0`
  - `mean_abs_diff 0.0`
  - `before_cache_numel 0`
  - `materialize_only_ms 27.834`
  - `after_cache_numel 110592 torch.float16`
- 结果：通过

## 最终验证
- 端到端检查：
  - AWQ checkpoint 现在默认在加载时一次性恢复反量化缓存。
  - 动态模式仍保留，仅用于基准对比。
  - 在真实 CUDA 环境中，缓存模式比动态模式更快，且数值完全一致。
- 已知限制：
  - `cache_load_ms` 包含了模型实例化、checkpoint 加载和缓存恢复，若只看纯缓存恢复成本，应以 `materialize_only_ms` 为准。
  - 当前缓存模式仍然保存量化参数和反量化缓存并存，内存占用会高于纯动态模式。
