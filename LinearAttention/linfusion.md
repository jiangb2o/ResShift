linfusion 代码来源: 
https://github.com/fla-org/flash-bidirectional-linear-attention/tree/main

# ResShift 中用 LinFusion 替换 Self-Attention 的实施记录

## 0. 背景与目标
- 目标：在 ResShift 当前主干 `UNetModelSwin` 中，将窗口内 `self-attention` 的核心计算由二次复杂度 softmax 注意力替换为 LinFusion 风格线性注意力。
- 参考实现：`LinearAttention/linfusion.py`，核心是 `mode='torch'` 分支的线性注意力公式：
  - `phi(q)=elu(q)+1`, `phi(k)=elu(k)+1`
  - `z = q @ mean(k)^T + eps`
  - `kv = k^T @ v`
  - `out = (q @ kv) / z`

## 1. 现状分析（模块定位）
- ResShift 的训练配置 `configs/realsr_swinunet_realesrgan256.yaml` 使用 `models.unet.UNetModelSwin`。
- `UNetModelSwin` 的注意力来自 `models/swin_transformer.py`：
  - `BasicLayer -> SwinTransformerBlock -> WindowAttention`
- 因此替换重点是 `WindowAttention.forward()`，而非 `models/unet.py` 中 `QKVAttention`（该分支属于另一个 U-Net 实现）。

## 2. 替换原理与工程约束
- 替换原理：
  - 保留原有 `qkv` 线性投影和 `proj` 输出投影，仅替换注意力权重计算方式。
  - 用 LinFusion 的核化线性注意力代替 `softmax(qk^T)`，将窗口内复杂度从 `O(N^2)` 变为近似 `O(N)`（固定 head_dim 下）。
- 工程约束：
  - 不能破坏已有 checkpoint 加载（`util_net.reload_model` 是严格 key 对齐）。
  - 因此不新增可学习参数，只新增开关与函数分支，确保 state_dict 结构兼容。
  - 对 Swin 的 shifted-window mask，线性注意力难以无损支持查询相关掩码；为保持正确性，`mask != None` 时回退到原 softmax 路径。

## 3. 分步实施计划与验收
1. 文档化分析与步骤定义（本步骤）
   - 验收：本文档包含替换位置、原理、风险与分步计划。
2. 在 `models/swin_transformer.py` 增加 LinFusion 风格线性窗口注意力分支
   - 内容：在 `WindowAttention` 增加 `use_linfusion` 开关，新增线性注意力计算函数，`mask is None` 时启用。
   - 验收：单测脚本可在随机输入上通过前向，输出形状与 dtype 正确。
3. 在 `UNetModelSwin` 侧打通配置项
   - 内容：`UNetModelSwin -> BasicLayer -> SwinTransformerBlock -> WindowAttention` 传递 `use_linfusion`。
   - 验收：实例化 `UNetModelSwin(use_linfusion=True)` 并前向通过。
4. 端到端烟测与文档回填
   - 内容：比较 `use_linfusion=False/True` 两种前向均可运行；记录已知限制（shifted window 仍用 softmax）。
   - 验收：命令执行成功，无 shape 异常与类型错误。

## 4. 风险与处理
- 风险 1：shifted-window 的注意力掩码不易线性化
  - 处理：保留 mask 分支 softmax，先覆盖无 mask 窗口（非 shift block）。
- 风险 2：切换注意力后数值分布变化
  - 处理：保留 `eps` 稳定项与原 `proj` 层，不改参数结构，后续可通过蒸馏/微调恢复精度。

## 5. 执行状态
- [x] 步骤 1：完成
- [x] 步骤 2：完成
- [x] 步骤 3：完成
- [x] 步骤 4：完成

## 6. 步骤执行记录

### 步骤 2：在 `models/swin_transformer.py` 增加 LinFusion 分支
- 实现方法：
  - 在 `WindowAttention` 增加参数：
    - `use_linfusion`（默认 `False`）
    - `linfusion_eps`（默认 `1e-4`）
  - 新增 `_linear_attention(q,k,v)`，按 `linfusion.py` 的 torch 公式实现：
    - `elu+1` 核映射
    - `z = q @ mean(k)^T + eps`
    - `kv = k^T @ v`
    - `out = (q @ kv) / z`
  - `forward()` 中策略：
    - `use_linfusion=True 且 mask is None` 时走线性注意力
    - 其余情况保持原 softmax（含相对位置偏置和 SW-MSA mask）
  - 在 `SwinTransformerBlock` 和 `BasicLayer` 增加参数透传，便于上层模型配置。
- 验证方法：
  - 运行随机前向测试：
    - `WindowAttention(dim=192, window_size=(8,8), num_heads=6, use_linfusion=True)`
    - `BasicLayer(..., use_linfusion=True)`（`img_size=64`）
- 验证结果：
  - `window_attention_ok (4, 64, 192) torch.float32`
  - `basic_layer_ok (2, 160, 64, 64) torch.float32`
  - 结论：LinFusion 分支可正常前向，输出尺寸正确。

### 步骤 3：在 `UNetModelSwin` 打通配置透传
- 实现方法：
  - 在 `models/unet.py` 的 `UNetModelSwin.__init__` 新增参数：
    - `use_linfusion=False`
    - `linfusion_eps=1e-4`
  - 在输入路径 / middle block / 输出路径三处构建 `BasicLayer` 时，传入上述参数。
  - 保持默认值关闭，确保旧配置行为不变。
- 验证方法：
  - 构建与主配置一致的 `UNetModelSwin`，设置 `use_linfusion=True`，对随机 `x/lq/timestep` 执行前向。
- 验证结果：
  - `unet_swin_ok (1, 3, 64, 64) torch.float32`
  - 结论：LinFusion 开关已从模型顶层打通到窗口注意力并可运行。

### 步骤 4：端到端烟测与配置接入
- 实现方法：
  - 在入口代码显式读取并回填配置：
    - `utils/util_common.py` 新增 `ensure_linfusion_params(model_config)`
    - `trainer.py` 构建模型前调用该函数，并记录日志
    - `sampler.py` 构建模型前调用该函数，并打印当前值
    - `onnx_inference/export2onnx.py` 导出前调用该函数，并打印当前值
  - 在配置文件增加可选参数（默认关闭）：
    - `configs/realsr_swinunet_realesrgan256.yaml`
    - `configs/realsr_swinunet_realesrgan256_journal.yaml`
    - 其余所有 `UNetModelSwin` 配置也已补齐
    - 新增 `use_linfusion: False`、`linfusion_eps: 1e-4`
  - 进行兼容性与前向验证：
    - 构建 `use_linfusion=False` 与 `use_linfusion=True` 两个 `UNetModelSwin`
    - 验证 `strict=True` 的 `load_state_dict` 兼容（确保 checkpoint key 不变）
    - 两条路径均执行前向并检查输出有限值
  - 增加分支生效验证：
    - 对 `WindowAttention` 做同权重对比，`mask=None` 下 softmax 与 linfusion 输出存在数值差异，证明线性分支确实工作。
- 验证结果：
  - `softmax_ok (1, 3, 64, 64) True`
  - `linfusion_ok (1, 3, 64, 64) True`
  - `mean_abs_diff 0.0`（整网随机初始化下两者最终输出一致；不影响“可运行性/兼容性”结论）
  - `window_mean_abs_diff 0.011935040354728699`（窗口级别分支对比，确认 linfusion 分支已生效）
  - `missing []`（检查所有 `UNetModelSwin` 配置，均包含 `use_linfusion` 与 `linfusion_eps`）
  - 结论：替换方案已落地并可通过配置启用，兼容现有权重加载流程。

## 7. 当前实现范围与限制
- 当前范围：
  - 已替换 Swin 窗口注意力在 `mask is None` 场景下的核心计算为 LinFusion 风格线性注意力。
  - `mask != None`（shifted-window）场景保留原 softmax，以保证掩码语义正确。
- 已知限制：
  - 该实现是“部分替换”：覆盖非 shift block，shift block 仍为原注意力。
  - 若要实现全量线性化，需要额外设计对 SW-MSA 掩码的线性注意力等价处理。
