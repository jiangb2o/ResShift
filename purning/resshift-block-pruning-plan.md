# ResShift Block Pruning

## 任务目标

- 为 `ResShift` 增加一种训后块剪枝方案，剪枝对象是 `UNetModelSwin` 中堆叠的 `SwinTransformerBlock`。
- 重要性指标以“块输出残差的 L2 大小”为核心，优先删除残差贡献最小的块。
- 本轮只输出实现思路，不改动模型代码、不生成可执行剪枝脚本。

## 已确认的架构事实

- `models/unet.py` 中的 `UNetModelSwin` 会在 U-Net 的输入分支、middle block、输出分支插入 `BasicLayer`。
- `models/swin_transformer.py` 中的 `BasicLayer` 内部通过 `self.blocks = nn.ModuleList([...])` 堆叠 `SwinTransformerBlock`。
- `SwinTransformerBlock.forward()` 的块级残差可以定义为 `block_out - block_in`，其中 `block_out` 是整个 block 返回值，`block_in` 是进入该 block 前的特征。
- 当前配置文件 `configs/realsr_swinunet_realesrgan256.yaml` 中 `attention_resolutions=[64,32,16,8]`，`swin_depth=2`。基于源码静态推断，当前模型会在输入侧放 4 个 `BasicLayer`、middle 放 1 个、输出侧放 4 个，总计 9 个 `BasicLayer`，即 18 个 `SwinTransformerBlock`。
- `trainer.py` 和 `utils/util_net.py` 当前默认按完整结构严格加载权重；`reload_model()` 会逐键匹配，所以如果直接改变 block 数量，现有加载路径会失效。

## 实现思路

### 1. 先明确“块”的粒度

- 剪枝粒度建议先定义为单个 `SwinTransformerBlock`，而不是整个 `BasicLayer`。
- 每个 block 用一个稳定 ID 标识，例如：
  - `input_blocks.1.1.blocks.0`
  - `middle_block.1.blocks.1`
  - `output_blocks.3.1.blocks.0`
- 这里的路径规则直接对应 `named_modules()` 路径，方便后续记录分数、保存 prune spec、重放剪枝结果。

### 2. 用块输出残差的 L2 作为重要性指标

- 对每个 block，在 forward 时记录：
  - 输入特征 `h_in`
  - 输出特征 `h_out`
  - 块残差 `r = h_out - h_in`
- 基础分数可以定义为：

```text
score_block = ||r||_2
```

- 但如果要在不同分辨率、不同通道数的 block 之间做全局排序，直接用原始 L2 会偏向大特征图。更稳妥的比较口径建议改为下面两种之一：

```text
score_block = ||r||_2 / sqrt(numel(r))
score_block = ||r||_2 / (||h_in||_2 + eps)
```

- 如果严格按用户当前想法落第一版，排序主指标可以保留“残差 L2 越小越先删”；但实现时建议同时把归一化分数也记录下来，避免跨 stage 排序失真。

### 3. 分数采集方式建议走“真实采样分布”

- `ResShift` 是扩散式恢复模型，同一个 U-Net 会在不同 diffusion step 多次调用；块的重要性会随时间步变化。
- 因此分数统计不建议只看单次前向，而应在真实推理/采样轨迹上做聚合。
- 建议流程：
  1. 选一小批 calibration 图像，例如 `testdata/Val_SR/lq` 中 10 到 50 张。
  2. 运行正常的 ResShift 推理流程。
  3. 对每次 U-Net forward 内的每个 block 记录一次残差分数。
  4. 对所有样本、所有 diffusion step 求平均，得到最终 `importance_score`。

- 推荐聚合方式：

```text
importance_score(block) = mean_{image, step}(score_block)
```

- 同时记录方差或分位数，后面可以识别“均值很小但波动很大”的块，避免误删偶发关键块。

### 4. 剪枝策略建议分两层

- 第一层是全局候选排序：
  - 将所有 block 按 `importance_score` 从小到大排序。
  - 从最小分数开始尝试删除。

- 第二层是结构约束：
  - 每个 `BasicLayer` 至少保留 1 个 block，避免整层退化成只有 patch embed/unembed。
  - 对 Swin 的交替窗口机制要加保护。因为 block 默认按 `shift_size=0` 和 `shift_size=window_size//2` 交替堆叠，如果只删掉其中一半，效果可能比预期差。
  - 第一版可以保留“单块剪枝”能力，但建议额外支持“成对剪枝”模式，把同一 `BasicLayer` 中的偶数/奇数 block 作为一组一起删。

- 推荐的实际落地顺序：
  1. 先做单块 one-shot 排序，观察最低几个块。
  2. 先删除 1 个块验证效果。
  3. 如果质量下降很小，再尝试删 2 个、3 个。
  4. 若 one-shot 不稳定，再改成 greedy：每删 1 个块后重新统计一次分数。

### 5. 当前代码基础上，最省改动的落地路径是“运行时替换为 Identity”

- 由于现有模型构造使用统一的 `swin_depth`，且 checkpoint 加载默认严格逐键匹配，直接“物理删除并重建更浅模型”会牵涉：
  - 配置结构变更
  - `BasicLayer` 构造逻辑变更
  - state_dict 重新映射
  - 训练/推理/导出路径同步改造

- 第一版更适合走下面的后处理方案：
  1. 先按完整模型加载 checkpoint。
  2. 根据 prune spec 找到待删 block。
  3. 把对应的 `SwinTransformerBlock` 替换成 `nn.Identity()`。
  4. 后续推理、导出 ONNX、测速都基于这个已剪枝模型执行。

- 这样做的优点：
  - 不需要修改 checkpoint 存储格式。
  - 不需要重排权重键名。
  - 删除的是实际计算图里的 block，推理时重计算会消失。
  - 便于快速试验不同 prune ratio。

- 这样做的代价：
  - 如果把剪枝结果单独存成 checkpoint，无法再用当前“原始结构 + strict reload”直接恢复。
  - 更适合保存为“原始 checkpoint + prune spec”二件套，而不是单独保存一个已经瘦身后的纯权重文件。

### 6. 如果后续追求“真正瘦身后的独立模型”，再做第二阶段结构化改造

- 第二阶段再考虑把 `BasicLayer` 改成支持显式保留索引，例如：
  - `depth_per_layer`
  - `keep_block_indices`
  - `prune_spec_path`

- 到那时需要做的事情：
  1. 模型构造时按保留索引只实例化未剪枝 block。
  2. 从原始 checkpoint 提取保留 block 的权重并重排索引。
  3. 更新训练/推理/导出脚本，保证新结构可直接加载。

- 这条路更“干净”，但明显比第一版重很多，不适合作为当前第一步。

## 推荐的第一版落地方案

1. 增加 block registry，枚举所有 `SwinTransformerBlock` 的模块路径。
2. 增加 calibration scoring 逻辑，在真实推理轨迹中统计 `block_out - block_in` 的 L2 分数。
3. 生成 `prune_spec.json`，内容至少包括：
   - block ID
   - 原始排序分数
   - 是否被剪
   - 剪枝批次/时间
4. 在模型加载后应用 prune spec，把目标 block 替换成 `nn.Identity()`。
5. 对剪枝后模型做质量与速度验证。
6. 若验证通过，再考虑导出 ONNX 或进一步做结构化瘦身。

## 验证方案

### 功能正确性

- 剪枝后模型可以正常跑完整个 ResShift 推理流程。
- block 被替换后，forward shape 不变。
- 未剪枝时输出与基线一致；剪枝后输出变化可控。

### 质量评估

- 在 `testdata/Val_SR` 上比较：
  - PSNR
  - LPIPS
  - 如项目已有，也可补充 NIQE / MANIQA

- 建议设置一条简单门槛：
  - 延迟下降明显
  - PSNR 下降不超过预设阈值，例如 `0.1dB` 到 `0.3dB`

### 性能评估

- 比较剪枝前后的：
  - 单张平均推理耗时
  - 吞吐
  - ONNX 导出后运行时延迟

## 关键风险

- 仅按单次统计得到的“小残差块”不一定真的不重要，尤其在扩散模型里，不同时间步的作用差异会很大。
- 跨分辨率 block 直接比较原始 L2 会有尺度偏置，建议至少同步记录归一化分数。
- Swin block 有 shift/non-shift 交替特性，删单个 block 可能破坏局部窗口信息混合节奏。
- one-shot 全局排序删除多个块后，块间相互作用会变化，因此多块剪枝更适合 greedy 迭代。

## 实施步骤

1. [ ] 设计 `block registry` 与 `prune spec` 格式，保证每个 block 都有稳定 ID。
2. [ ] 在推理路径中加入 block 残差统计逻辑，并跑一轮 calibration 收集分数。
3. [ ] 按分数生成待删列表，并增加运行时 `Identity` 替换能力。
4. [ ] 完成质量、速度、导出验证，确认删块收益。
5. [ ] 如果第一版收益稳定，再考虑做真正的结构化 block 删除与 checkpoint 重排。

## 步骤执行记录

### 步骤 1

- 变更内容：
  - 阅读并分析 `models/unet.py`、`models/swin_transformer.py`、`trainer.py`、`utils/util_net.py`、`configs/realsr_swinunet_realesrgan256.yaml`。
  - 识别 block 插入位置、残差定义位置和权重加载约束。
- 验证操作：
  - `sed -n '216,520p' models/swin_transformer.py`
  - `sed -n '700,870p' models/unet.py`
  - `sed -n '1,220p' utils/util_net.py`
  - `sed -n '200,250p' trainer.py`
  - `sed -n '1,220p' configs/realsr_swinunet_realesrgan256.yaml`
- 验证输出摘要：
  - `BasicLayer` 内部使用 `ModuleList` 堆叠 `SwinTransformerBlock`。
  - `SwinTransformerBlock` 输出是带残差连接的整块结果，因此块残差可定义为 `out - in`。
  - 当前加载路径默认按完整结构严格对齐权重，不适合直接改 block 数后原样加载。
- 结果：通过

### 步骤 2

- 变更内容：
  - 尝试在本地实例化 `UNetModelSwin` 做动态核验。
- 验证操作：
  - `python - <<'PY' ... from models.unet import UNetModelSwin ... PY`
  - `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/ubuntu/miniconda3/envs/ResShift/bin/python - <<'PY' ... PY`
- 验证输出摘要：
  - 系统默认 `python` 缺少 `numpy`，无法直接导入模型。
  - 切换到项目 Conda 环境后，又因为当前沙箱里的 OpenMP SHM 权限限制而中断。
  - 因此“9 个 BasicLayer / 18 个 block”的结论目前来自源码静态推断，不是运行时枚举结果。
- 结果：失败

### 步骤 3

- 变更内容：
  - 创建当前方案文档，整理第一版推荐实现路径与风险。
- 验证操作：
  - 创建后重新检查本文档内容是否包含任务目标、实现思路、实施步骤和执行记录。
- 验证输出摘要：
  - 文档已覆盖剪枝粒度、评分方法、候选筛选、运行时 Identity 替换方案、验证方案和当前阻塞。
- 结果：通过

## 最终验证

- 端到端检查：
  - 本轮只完成方案设计文档，未修改 `ResShift` 代码，未执行剪枝实验。
- 已知限制：
  - 动态实例化模型的验证受当前环境依赖与 OpenMP SHM 权限限制，部分数量结论仍属于基于源码的静态推断。
  - 还没有真实跑过 calibration scoring，也没有验证“删掉最小残差块”对画质和速度的实际影响。
