# LinFusion Window Attention

## 任务目标

在 `models/swin_transformer.py` 的 `WindowAttention` 中启用完整的 LinFusion 线性注意力分支，使 `UNetModelSwin` 在 `use_linfusion=True` 时无论是否存在 shifted-window mask，都不再回退到 softmax 注意力，而是统一使用 LinFusion 风格的注意力计算，同时保持默认行为和 checkpoint 兼容性不变。

## 实现思路

- 保留现有 `qkv` 投影、`proj` 投影和参数结构，避免破坏 checkpoint key。
- `use_linfusion=True` 时，`WindowAttention` 整体切换为 LinFusion：
  - `phi(q)=elu(q)+1`
  - `phi(k)=elu(k)+1`
  - `z = q @ mean(k)^T + eps`
  - `kv = (k^T / sqrt(N)) @ (v / sqrt(N))`
  - `out = (q @ kv) / z`
- 对 `mask is not None` 的 Swin shifted-window，不能再回退 softmax；改为利用 mask 的块对角结构，把每个窗口拆成若干可见 token 组，在每个组内部独立执行 LinFusion，再按原 token 顺序 scatter 回去。
- 修复当前 `WindowAttention` 中会阻断 softmax 前向的明显错误（相对位置偏置变量拼写错误），确保 `use_linfusion=False` 时默认行为仍可用。

## 实施步骤

1. [x] 核对现有 `WindowAttention` 与 `use_linfusion` 参数链路，并记录改造范围。
2. [x] 在 `models/swin_transformer.py` 中实现完整的 `use_linfusion=True` 线性注意力分支，覆盖有无 mask 两种情况。
3. [x] 运行窗口级和 `UNetModelSwin` 级前向验证，并写回真实结果。

## 步骤执行记录

### 步骤 1
- 变更内容：
  - 确认 `UNetModelSwin -> BasicLayer -> SwinTransformerBlock -> WindowAttention` 的 `use_linfusion` / `linfusion_eps` 参数链路已经存在，无需额外改 `models/unet.py`。
  - 确认当前真正缺失的是 `WindowAttention._linear_attention()` 的可执行实现。
  - 确认 softmax 分支里存在 `relaztive_position_bias` 拼写错误，会导致前向异常。
- 验证操作：
  - `grep -RIn "use_linfusion\|linfusion" . --exclude-dir=.git`
  - `sed -n '1,280p' models/swin_transformer.py`
  - `sed -n '620,880p' models/unet.py`
- 验证输出摘要：
  - 参数已从 `UNetModelSwin` 透传到 `WindowAttention`
  - `WindowAttention` 中仅有注释版 `_linear_attention`
  - softmax 分支存在相对位置偏置变量拼写错误
- 结果：通过

### 步骤 2
- 变更内容：
  - 在 `models/swin_transformer.py` 中新增可执行的 `_linear_attention(q, k, v)`。
  - 实现公式与 `LinearAttention/linfusion.py` 的 torch 分支对齐：
    - `F.elu(.) + 1`
    - `z = q @ mean(k)^T + eps`
    - `kv = (k^T / sqrt(N)) @ (v / sqrt(N))`
    - `out = (q @ kv) / z`
  - 新增 `_masked_linear_attention(q, k, v, mask)`：
    - 从 `mask` 中提取每个 shifted-window 的可见 token 分组
    - 每个分组内部独立执行 LinFusion
    - 将结果回填到原 token 顺序
  - `use_linfusion=True` 时，`WindowAttention` 无论 `mask` 是否存在都统一走线性分支。
  - 修复 `relative_position_bias` 变量拼写错误，恢复 `use_linfusion=False` 的原 softmax 分支可用性。
- 验证操作：
  - 代码 diff 复查
- 验证输出摘要：
  - 仅改动 `models/swin_transformer.py`
  - 未新增参数，不影响现有 checkpoint key
  - `use_linfusion=True` 下已不存在“mask 时回退 softmax”的路径
- 结果：通过

### 步骤 3
- 变更内容：
  - 运行无 mask 的窗口级前向验证，确认 LinFusion 分支可执行且与 softmax 分支数值不同。
  - 运行带 shifted-window mask 的窗口级前向验证，确认 `mask != None` 也在线性分支内完成。
  - 运行 `UNetModelSwin` 级前向验证，确认配置透传和整网 shape 正常。
- 验证操作：
  - 无 mask 窗口级验证：
```bash
python - <<'PY'
import torch
from models.swin_transformer import WindowAttention

torch.manual_seed(0)
attn = WindowAttention(dim=192, window_size=(8, 8), num_heads=6, use_linfusion=True)
x = torch.randn(4, 64, 192)
out = attn(x)
print('window_ok', tuple(out.shape), out.dtype, torch.isfinite(out).all().item())

attn_softmax = WindowAttention(dim=192, window_size=(8, 8), num_heads=6, use_linfusion=False)
attn_softmax.load_state_dict(attn.state_dict())
out_softmax = attn_softmax(x)
print('window_mean_abs_diff', float((out - out_softmax).abs().mean()))
PY
```
  - 带 mask 窗口级验证：
```bash
python - <<'PY'
import torch
from models.swin_transformer import WindowAttention, SwinTransformerBlock

torch.manual_seed(0)
block = SwinTransformerBlock(dim=192, input_resolution=(16, 16), num_heads=6, window_size=8, shift_size=4, use_linfusion=True)
mask = block.attn_mask
attn = block.attn
x = torch.randn(8, 64, 192)
out = attn(x, mask=mask)
print('masked_window_ok', tuple(out.shape), out.dtype, torch.isfinite(out).all().item())
print('mask_shape', tuple(mask.shape))
print('mask_unique', sorted(mask.unique().tolist()))
PY
```
  - 整网验证：
```bash
python - <<'PY'
import torch
from omegaconf import OmegaConf
from utils import util_common

cfg = OmegaConf.load('configs/realsr_swinunet_realesrgan256.yaml')
model = util_common.instantiate_from_config(cfg.model)
model.eval()
with torch.no_grad():
    x = torch.randn(1, 3, 64, 64)
    lq = torch.randn(1, 3, 64, 64)
    t = torch.tensor([10], dtype=torch.long)
    out = model(x=x, timesteps=t, lq=lq, mask=None)
print('unet_ok', tuple(out.shape), out.dtype, torch.isfinite(out).all().item())
print('use_linfusion', cfg.model.params.use_linfusion)
PY
```
- 验证输出摘要：
  - `window_ok (4, 64, 192) torch.float32 True`
  - `window_mean_abs_diff 0.012113490141928196`
  - `masked_window_ok (8, 64, 192) torch.float32 True`
  - `mask_shape (4, 64, 64)`
  - `mask_unique [-100.0, 0.0]`
  - `unet_ok (1, 3, 64, 64) torch.float32 True`
  - `use_linfusion True`
  - 额外观测：整网验证时出现一条 CUDA 初始化 warning，但 CPU 前向仍正常完成，不影响本次功能验证结论。
- 结果：通过

## 最终验证
- 端到端检查：
  - `WindowAttention` 在 `use_linfusion=True` 时，已不再依赖 softmax 注意力。
  - 无 mask 和带 shifted-window mask 两种场景都统一走 LinFusion 线性注意力。
  - `UNetModelSwin` 可通过现有配置 `use_linfusion=True` 正常实例化并完成前向。
  - `use_linfusion=False` 时原 softmax 分支仍可用。
- 已知限制：
  - 当前 `use_linfusion=True` 分支未使用原 `relative_position_bias_table`，因为 LinFusion 的核化公式无法直接复用 softmax 注意力中的成对相对位置偏置项。
  - 当前实现是 LinFusion 的 torch 公式分支，没有接入外部仓库中的 Triton fused kernel。
