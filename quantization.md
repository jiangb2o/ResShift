# ResShift PTQ 量化方案

## 1. 目标与约束

本文给出 `ResShift` 的离线量化方案，仅输出方案，不改动现有代码。

当前仓库的推理链路是三段式：
- `encoder`：自编码器编码器
- `unet`：扩散过程中的单步去噪网络，多次循环调用
- `decoder`：自编码器解码器

对应文件：
- `onnx_inference/export2onnx.py`
- `onnx_inference/run_onnx_pipeline.py`
- `configs/realsr_swinunet_realesrgan256.yaml`

本次要求采用 PTQ，避免从头训练。候选方案是 `AWQ` 和 `GPTQ`。这两种方法都更适合处理以 `Linear/MatMul` 为主的权重压缩；而 `ResShift` 实际上是一个混合结构：
- `UNetModelSwin` 内部同时包含大量 `Conv2d` 和 `Swin` 中的 `Linear`
- `autoencoder` 以 `Conv2d` 为主

因此需要先明确一个关键判断：
- `AWQ/GPTQ` 可以优先用于 `UNet` 中 `Swin` 注意力和 MLP 的 `Linear` 权重
- 对 `Conv2d` 密集的部分，尤其是 `autoencoder`，更适合保留 FP16/FP32，或者后续用 ONNX Runtime 的常规 INT8/QDQ PTQ 做补充

结论先行：
- 主方案建议：`UNet` 采用 PTQ，优先尝试 `GPTQ`，`encoder/decoder` 保持 FP16 或 FP32
- `AWQ` 作为对照实验方案，用于比较精度和吞吐
- 若最终目标是稳定导出并运行在 `ONNX Runtime`，需要把“PyTorch 量化”和“ONNX 量化表示”拆开看，不能假设 AWQ/GPTQ 的 PyTorch 权重能直接无损导出成当前工程可运行的 ONNX

## 2. 量化对象划分

### 2.1 优先量化对象

优先量化 `UNet`，原因如下：
- `UNet` 在扩散推理中会被重复调用 15 次，累计耗时最高
- `UNet` 中存在 `WindowAttention.qkv`、`proj`、`reduction` 等 `Linear` 层，适合 AWQ/GPTQ
- 即使只压缩 `UNet`，也能先拿到较明显的显存和时延收益

### 2.2 暂不作为第一阶段量化对象

`encoder/decoder` 暂不作为第一阶段重点量化对象：
- 这两部分以 `Conv2d` 为主，不是 AWQ/GPTQ 的典型高收益场景
- 它们在整条链路中只各运行一次，压缩收益不如 `UNet` 明显
- 量化后如果产生细节损失，容易直接体现在最终图像纹理和颜色上

建议：
- 第一阶段：只量化 `UNet`
- 第二阶段：若还需继续压缩，再评估 `encoder/decoder` 是否用 ONNX Runtime INT8 PTQ 单独处理

## 3. 基线建立

在正式量化前，先固定一个可复现实验基线。

### 3.1 固定模型与配置

基于当前仓库默认超分模型：
- 配置：`configs/realsr_swinunet_realesrgan256.yaml`
- UNet 权重：`weights/resshift_realsrx4_s15_v1_default.pth`
- Autoencoder 权重：`weights/autoencoder_vq_f4.pth`

### 3.2 固定验证集与校准集

建议准备两个数据子集：
- `calibration set`：用于 PTQ 校准，建议 100 到 500 张低分辨率输入图像
- `evaluation set`：用于最终验收，建议 50 到 100 张，与校准集不重叠

数据要求：
- 分布尽量贴近实际业务图像，不要只使用 ImageNet 风格自然图
- 覆盖平滑区域、细纹理、边缘、高对比度、噪声图像
- 输入尺寸建议覆盖常见分辨率，并包含需要 padding 的样本

### 3.3 基线指标

量化前先记录以下指标：
- 端到端平均推理时延
- 仅 `UNet` 单步推理时延
- 显存占用
- 模型文件大小
- 图像质量指标：`PSNR`、`SSIM`、`LPIPS`
- 主观观察项：锐度、过平滑、伪影、色偏、重复纹理

验收方式建议分为两层：
- 算法层：与 FP32/FP16 基线对比
- 工程层：在 PyTorch 和 ONNX 两条链路分别做一致性对比

## 4. 公共 PTQ 流程

无论采用 AWQ 还是 GPTQ，都建议先执行下面的公共步骤。

### 4.1 拆分模块

按以下粒度量化和验证：
1. 先只量化 `UNet`
2. 验证多步扩散后的最终 SR 输出
3. 通过后再考虑是否扩展到 `encoder/decoder`

不要一开始就同时量化三段模型，否则定位误差来源会很困难。

### 4.2 选择量化范围

优先纳入量化的层：
- `models/swin_transformer.py` 中的 `qkv`
- `proj`
- `reduction`
- attention/MLP 内部的 `Linear`

默认排除或延后处理的层：
- 首尾卷积
- 输出层
- 时间步嵌入相关的小型投影层
- `encoder/decoder` 的关键卷积层

原因：
- 首尾层和输出敏感度通常更高
- 扩散模型误差会逐步累积，输出端更需要保守处理

### 4.3 量化粒度建议

建议从以下配置起步：
- 权重量化位宽：`W4`
- 激活保持：`FP16` 或 `FP32`
- 分组粒度：`group_size = 128` 或 `64`
- 对异常敏感层允许回退到 `FP16`

第一轮不建议直接追求极限压缩，比如全模型 `W3` 或更小位宽。

### 4.4 校准输入的构造

由于 `UNet` 的真实输入不是原图，而是扩散过程中的：
- `x`：当前 latent 状态
- `lq`：条件图
- `timesteps`

所以校准数据不能只喂普通图像。推荐做法是：
1. 用现有 FP32 链路跑一遍采样过程
2. 在多个时间步抓取真实的 `(x_t, lq, t)` 样本
3. 将这些样本作为 AWQ/GPTQ 的校准输入

建议覆盖：
- 早期时间步
- 中期时间步
- 后期时间步
- 平滑图像与高频纹理图像

这是 ResShift 场景里最重要的校准要求，否则量化器会偏离真实分布。

## 5. AWQ 方案

### 5.1 方案定位

`AWQ` 的核心思想是利用少量校准样本，保留对激活更敏感的权重通道，从而在低比特权重量化时尽量维持精度。

对当前 `ResShift`，AWQ 更适合作为：
- `UNet` 中 `Swin` 相关 `Linear` 层的权重量化方案
- 不适合作为整个模型统一量化方案

### 5.2 AWQ 实施步骤

1. 构建 FP32 基线模型
2. 只抽取 `UNet` 作为量化对象
3. 准备真实扩散输入形式的校准集 `(x_t, lq, t)`
4. 为 `Swin` 中的 `Linear` 层建立待量化白名单
5. 跳过首层、尾层、输出投影和明显精度敏感层
6. 执行 `W4` AWQ 搜索，优先尝试 `group_size=128`
7. 对量化后的 `UNet` 单步输出做逐层误差分析
8. 将量化 `UNet` 放回完整扩散链路，评估最终 SR 图像质量
9. 若精度下降明显，则按层回退：
   - 先回退输出相关层
   - 再回退注意力投影层
   - 必要时只量化 MLP 层

### 5.3 AWQ 的优点

- 一般比朴素 W4 更稳
- 更适合激活分布差异大的层
- 如果只做 `UNet` 的部分 `Linear` 压缩，通常比较容易保住画质

### 5.4 AWQ 的风险

- `ResShift` 并不是纯 Transformer/LLM 结构，AWQ 工具链可能默认假设 HuggingFace 风格模块，需要额外适配
- `Conv2d` 无法直接获得同等收益
- 扩散多步误差会累计，单步误差很小也可能在最终 SR 图像上放大
- 当前仓库的 ONNX 导出路径是标准 `torch.onnx.export`，不保证能直接导出 AWQ 的压缩模块表示

### 5.5 AWQ 的验收标准

建议以以下门限作为第一轮目标：
- 模型大小显著下降
- `UNet` 单步速度有可测收益
- 端到端图像质量基本不出现明显伪影
- `LPIPS` 不显著恶化
- 主观纹理不出现块状、振铃或条纹化

## 6. GPTQ 方案

### 6.1 方案定位

`GPTQ` 是典型的权重后训练量化方法，通过近似二阶信息逐层求解量化误差。对 `ResShift` 来说，它同样主要适用于 `UNet` 中的 `Linear/MatMul` 路径。

与 AWQ 相比，GPTQ 更适合作为本项目的主试方案，原因是：
- ONNX Runtime 已经提供 `MatMul` 的 4bit weight-only 量化能力，并明确支持 `GPTQ` 算法
- `Swin` 里的注意力和投影天然对应 `MatMul/Linear` 路径
- 从“最终需要 ONNX 落地”这个目标看，GPTQ 比 AWQ 更容易和 ONNX Runtime 的 weight-only int4 路线对齐

### 6.2 GPTQ 实施步骤

1. 构建 FP32 基线模型
2. 拆出 `UNet`
3. 收集真实扩散输入形式的校准样本 `(x_t, lq, t)`
4. 为 `Swin` 模块中的 `Linear` 建立量化清单
5. 从 `W4, group_size=128` 开始做 GPTQ
6. 先验证单步 `UNet(x, lq, t)` 输出误差
7. 再验证 15 步完整扩散输出的画质劣化
8. 对误差大的层做混合精度回退
9. 固化最优配置，形成可复用量化模板

### 6.3 GPTQ 的优点

- 与 ONNX Runtime 的 `MatMul` 4bit weight-only 方向更一致
- 对 `Linear` 密集子模块更有工程落地优势
- 更容易形成“先导出 FP32 ONNX，再对 MatMul 做 ONNX 图级 4bit 量化”的闭环

### 6.4 GPTQ 的风险

- 仍然无法覆盖 `Conv2d` 主导部分
- 多步扩散累积误差依旧需要重点验证
- 如果直接用 PyTorch 侧 GPTQ 压缩模块替换原始层，未必能顺利导出到当前 ONNX 图

### 6.5 GPTQ 的验收标准

比 AWQ 更关注以下两点：
- `UNet` 的 `MatMul` 是否能稳定映射到 ONNX 图中的可量化节点
- 量化后 ONNX 模型能否在 `onnxruntime` 中直接执行，而不是只停留在 PyTorch 侧可运行

## 7. ONNX 转换方案

这里必须把两种不同路线分开。

### 7.1 路线 A：先做 PyTorch AWQ/GPTQ，再导出 ONNX

这是最直观的路线，但风险最高。

步骤：
1. 在 PyTorch 中对 `UNet` 应用 AWQ 或 GPTQ
2. 用量化后的模块替换原始 `UNet`
3. 调整 `onnx_inference/export2onnx.py`，尝试导出量化后 `UNet`
4. 检查导出的 ONNX 是否仍是标准算子图
5. 用 `run_onnx_pipeline.py` 验证可运行性与精度

主要问题：
- 很多 AWQ/GPTQ 实现依赖自定义量化层、打包权重格式或运行时 kernel
- 这些表示通常不是当前 `torch.onnx.export` 能稳定表达的标准 ONNX 结构
- 即使导出成功，也可能被反解成普通浮点算子，失去真正的量化收益

因此，这条路线更适合作为实验路线，不建议作为第一落地路线。

### 7.2 路线 B：先导出标准 FP32 ONNX，再对 ONNX 图做量化

这是更推荐的工程路线。

步骤：
1. 保持当前仓库方式，先导出标准 FP32 ONNX
2. 对 `UNet ONNX` 做图级量化，而不是先改 PyTorch 模块
3. 对 `MatMul` 节点使用 weight-only int4 量化
4. 对 `Conv` 节点按需考虑 INT8/QDQ PTQ
5. 用 `onnxruntime` 重新验证端到端推理

这条路线的优点：
- 与当前仓库现有导出脚本兼容度最高
- 更容易保持 `encoder/unet/decoder` 三段式结构不变
- 更符合 ONNX Runtime 的部署方式

### 7.3 AWQ 与 ONNX 的结合方式

对 AWQ，需要特别保守。

建议分两种处理：
- 方案 B1：AWQ 只作为 PyTorch 侧可行性验证工具，用来筛选“哪些层适合 W4”
- 方案 B2：确定这些层后，不强求保留 AWQ 原始压缩格式，而是在 ONNX 侧重新落到标准量化表示

换句话说：
- AWQ 可以用于“找敏感层、定混合精度策略”
- 但最终部署到 ONNX 时，更现实的是转成 ONNX Runtime 可执行的标准量化图，而不是要求 ONNX 原生理解 AWQ 模块

### 7.4 GPTQ 与 ONNX 的结合方式

GPTQ 更适合映射到 ONNX Runtime 的 `MatMul` weight-only int4 量化流程。

推荐步骤：
1. 先导出 FP32 `UNet ONNX`
2. 分析图中 `MatMul` 节点，确认它们对应 `Swin` 的 `Linear`
3. 对这些 `MatMul` 执行 ONNX 图级 4bit weight-only 量化
4. 优先保留 `Conv` 为 FP16/FP32
5. 用 `onnxruntime` 验证端到端结果

如果后续要进一步压缩：
- 再单独尝试对 `encoder/decoder` 做常规 ONNX INT8 PTQ
- 不建议第一阶段就对三段模型全部做 4bit

### 7.5 三段模型的最终建议格式

建议第一阶段采用如下混合格式：
- `encoder.onnx`: FP16 或 FP32
- `unet.onnx`: `MatMul` 走 4bit weight-only，其他保持 FP16/FP32
- `decoder.onnx`: FP16 或 FP32

原因：
- 最符合当前工程结构
- 风险最低
- 出问题时容易快速回退到浮点 `encoder/decoder`

## 8. 推荐实施顺序

建议按以下顺序推进。

### 第一步：建立基线

- 固定权重、配置、数据集、随机种子
- 跑通当前 PyTorch 和 ONNX 基线
- 记录时延、显存、画质指标

### 第二步：只量化 `UNet`

- 不动 `encoder/decoder`
- 从 `Swin` 的 `Linear` 层开始
- 首轮只做 `W4`

### 第三步：并行试 AWQ 与 GPTQ

- 使用同一批校准样本
- 比较单步误差和端到端画质
- 统计哪些层最敏感

### 第四步：优先收敛 GPTQ/MatMul 路线

- 以 ONNX Runtime 可落地为主
- 先保证 `UNet ONNX` 的 4bit weight-only 跑通
- 再谈更多层的覆盖率

### 第五步：必要时引入混合精度回退

- 对明显敏感的 `qkv/proj/output` 层回退到 FP16
- 只保留收益高、误差小的层为 W4

### 第六步：最后再评估 `encoder/decoder`

- 若端到端收益仍不足，再单独评估自编码器是否做 INT8 PTQ
- 这里建议优先考虑 ONNX Runtime 常规 INT8/QDQ，而不是 AWQ/GPTQ

## 9. 风险与注意事项

### 9.1 扩散模型误差累积

ResShift 的 `UNet` 会被循环调用多次，量化误差不是一次性暴露，而是会随步数累积。因此：
- 不能只看单步输出误差
- 必须看完整 15 步后的最终 SR 图像

### 9.2 校准集必须贴近真实推理状态

如果校准集只包含普通图像，而不包含真实 `x_t` 分布，量化结果容易失真。

### 9.3 AWQ/GPTQ 不应直接覆盖全部模块

特别是：
- 首层
- 尾层
- 输出投影层
- 自编码器卷积主干

这些部分应优先保留浮点。

### 9.4 ONNX 导出不能想当然

当前仓库的 ONNX 导出脚本基于标准 `torch.onnx.export`。如果 PyTorch 侧引入 AWQ/GPTQ 的自定义量化层：
- 可能导不出
- 可能导出后变回浮点图
- 可能需要额外图重写步骤

因此，最终部署格式建议以 ONNX 图级量化为主，而不是把 PyTorch 压缩模块直接搬进 ONNX。

## 10. 最终建议

对于当前 `ResShift` 工程，建议采用下面的主次方案。

### 主方案

- 量化对象：`UNet`
- PTQ 方法：优先 `GPTQ` 思路
- 实际部署：先导出 `FP32 ONNX`，再在 ONNX 图上对 `MatMul` 做 4bit weight-only 量化
- `encoder/decoder`：先保持 FP16/FP32

这是最符合当前工程结构、最容易落到 `onnxruntime` 的方案。

### 备选方案

- 用 `AWQ` 对 `UNet` 中的 `Linear` 层做离线实验
- 目标主要是筛选敏感层和验证 W4 可行性
- 若 AWQ 在 PyTorch 侧效果优于 GPTQ，再考虑是否把其结果重新映射到 ONNX 标准量化表示

### 不建议的方案

- 一开始就对 `encoder + unet + decoder` 全量做 AWQ/GPTQ
- 直接假设 AWQ/GPTQ 压缩后的 PyTorch 模块能无缝导出为当前可运行 ONNX
- 未做真实扩散输入校准，就直接量化 `UNet`

## 11. 后续落地时需要补充的实现项

虽然本次不写代码，但后续真正落地时，至少需要补以下能力：
- 生成 `UNet` 校准样本 `(x_t, lq, t)` 的脚本
- `UNet` 层级白名单/黑名单配置
- PyTorch 侧 AWQ/GPTQ 实验脚本
- ONNX 图级量化脚本
- 量化前后端到端评测脚本
- ONNX Runtime 兼容性检查与基准测试

## 12. 参考说明

本文方案结合了当前仓库结构和 ONNX Runtime 的量化能力边界，核心判断如下：
- 当前工程的三段式导出和推理链路，决定了量化最好按模块分治
- `ResShift` 是 Conv + Swin 混合结构，AWQ/GPTQ 不适合作为全模型统一方案
- 若最终部署目标是 ONNX Runtime，则 `GPTQ/MatMul int4 + Conv 保持浮点或单独 INT8` 是更稳妥的组合
