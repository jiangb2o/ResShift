# LinFusion 原理与在 ResShift 中的迁移说明

## 1. 先看原始代码在做什么（`LinearAttention/linfusion.py`）

LinFusion 在这个文件中的核心是 `GeneralizedLinearAttention.forward()`。  
它本质上替换了传统 `softmax(QK^T)V`，改为一个核化的线性注意力近似。

---

## 2. 逐段解释 `linfusion.py`

### 2.1 非线性投影层 `get_none_linear_projection`

```python
torch.nn.Sequential(
    Linear(query_dim, mid_dim or query_dim),
    LayerNorm(...),
    LeakyReLU(),
    Linear(..., query_dim),
)
```

这部分给 Q/K 的输入增加一个可学习非线性残差投影：

- 若 `mid_dim == -1`：用 `Identity()`，不做非线性投影。
- 否则：构造两层 MLP 投影，用于增强表示能力。

在 `forward` 里它被这样用：

```python
query = self.to_q(hidden_states + self.to_q_(hidden_states))
key   = self.to_k(encoder_hidden_states + self.to_k_(encoder_hidden_states))
value = self.to_v(encoder_hidden_states)
```

即 Q/K 的输入是 “原输入 + 非线性投影残差”。

### 2.2 多头重排

`head_to_batch_dim` 把 `[B, L, H*D]` 变成 `[(B*H), L, D]`。  
后续注意力在每个 head 上独立计算。

### 2.3 核映射（最关键）

```python
query = F.elu(query) + 1.0
key = F.elu(key) + 1.0
```

这是把 Q/K 映射到非负空间（`ELU + 1`），记为 `phi(q), phi(k)`。  
目的：构造可线性化的注意力形式，并让分母稳定可正。

### 2.4 线性注意力公式（`mode='torch'`）

代码：

```python
z = query @ key.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-4
kv = (key.transpose(-2, -1) * (sequence_length**-0.5)) @ (value * (sequence_length**-0.5))
hidden_states = query @ kv / z
```

可理解为：

1. 先聚合 `K^T V`（`kv`）
2. 再左乘 `Q`
3. 用归一化项 `z` 做缩放，防止数值爆炸/偏移

与标准注意力相比：

- 标准：先算 `QK^T`（复杂度 `O(N^2)`）再乘 `V`
- 这里：绕开显式 `QK^T` 矩阵，变成 “先 `K^T V` 后 `Q(...)`”

当序列长度 N 很大时，内存和计算更友好（尤其避免大 `N x N` 权重矩阵）。

### 2.5 `mode='triton'`

`linfusion.py` 还支持 Triton fused kernel：

```python
hidden_states = linear_attention(query, key, value, eps=1e-4)
```

这是工程加速路径，数学目标与 torch 路径一致。

---

## 3. LinFusion 与标准 Self-Attention 的差异总结

- 标准 MHA：
  - `Attn = softmax(QK^T / sqrt(d))`
  - `Out = Attn V`
  - 需要构造 `N x N` 注意力矩阵

- LinFusion 风格线性注意力：
  - `Q/K` 先做核映射 `phi(.) = elu(.) + 1`
  - 通过 `K^T V` 和归一化项 `z` 近似替代 softmax attention
  - 不再显式构造全量 `N x N` 权重矩阵

优点：

- 更低的显存和更好的长序列可扩展性

代价：

- 与 softmax 注意力不是严格等价，效果依赖任务和训练策略

---

## 4. 我是如何迁移到 ResShift 的

## 4.1 先确定“要替换谁”

ResShift 的主配置使用 `models.unet.UNetModelSwin`，注意力路径是：

- `UNetModelSwin` -> `BasicLayer` -> `SwinTransformerBlock` -> `WindowAttention`

所以应替换的是 `models/swin_transformer.py` 里的 `WindowAttention`，而不是 `models/unet.py` 里的 `QKVAttention` 分支。

## 4.2 迁移策略

迁移时遵循了两个原则：

1. 不改已有参数结构（保证历史 checkpoint 严格加载）
2. 保留可回退路径（可通过配置开关启停）

因此实现方式是：

- 保留原 `qkv` 线性层和 `proj` 输出层
- 仅在 `WindowAttention.forward` 内新增一个分支：
  - `use_linfusion=True` 且 `mask is None` -> 走线性注意力
  - 否则 -> 走原 softmax 注意力

对应代码在：

- `models/swin_transformer.py` 的 `WindowAttention._linear_attention`
- `models/swin_transformer.py` 的 `if self.use_linfusion and mask is None:` 分支

## 4.3 迁移后的线性分支公式

在 ResShift 中实现的是 `linfusion.py` 的 torch 版本同构公式：

```python
q = F.elu(q) + 1.0
k = F.elu(k) + 1.0
z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + eps
kv = (k.transpose(-2, -1) * (seq_len ** -0.5)) @ (v * (seq_len ** -0.5))
out = (q @ kv) / z
```

输入输出形状（窗口内）：

- `q,k,v`: `[B_, heads, N, head_dim]`
- `out`: `[B_, heads, N, head_dim]`

最后再回到 `[B_, N, C]`，并经过原 `proj/proj_drop`。

## 4.4 为什么 `mask != None` 时不替换

Swin 的 shifted-window block 会用 attention mask（SW-MSA）。  
这类掩码在 softmax 框架下是直接加到 `QK^T` logits 上的。  
线性注意力要无损表达这个掩码并不直接，若强行替换容易破坏语义。

所以当前设计：

- 非 shift（无 mask）窗口：LinFusion
- shift（有 mask）窗口：原 softmax

这是一个“安全可用的分阶段替换”。

## 4.5 配置与调用链如何打通

我在 `UNetModelSwin` 新增并透传了两个参数：

- `use_linfusion`（bool）
- `linfusion_eps`（float）

传递链：

- `UNetModelSwin` -> `BasicLayer` -> `SwinTransformerBlock` -> `WindowAttention`

另外在训练/推理/导出入口显式读取并打印配置，确保运行时可见：

- `trainer.py`
- `sampler.py`
- `onnx_inference/export2onnx.py`

并在 `utils/util_common.py` 增加 `ensure_linfusion_params`，给缺省配置自动补默认值：

- `use_linfusion=False`
- `linfusion_eps=1e-4`

---

## 5. 数值与效果上你需要知道的点

1. `linfusion_eps`
- 用于分母稳定（避免除零或过小）
- 太小可能不稳定，太大可能影响归一化精度
- 当前默认 `1e-4`，与 `linfusion.py` 对齐

2. 为什么“整网输出差异可能看起来很小”
- 当前模型是随机初始化测试时，后续层可能把差异“冲淡”
- 但在窗口级别比较中，线性分支与 softmax 分支输出是可区分的（说明分支确实生效）

3. 与 checkpoint 兼容性
- 本次改动没有新增可学习参数张量
- state_dict key 结构保持兼容，旧权重可严格加载

---

## 6. 当前版本的边界

这次迁移是“工程安全优先”的版本：

- 已替换：无 mask 的窗口注意力
- 未替换：有 mask（shifted-window）的注意力

若你要“全量 LinFusion 化”：

- 需要进一步设计 SW-MSA mask 在线性注意力下的等价处理
- 或者改为不依赖该 mask 的架构策略（这会触及模型行为，不是纯算子替换）

---

## 7. 一句话总结

LinFusion 的核心是把 `softmax(QK^T)V` 改写为核映射后的线性注意力计算；  
我在 ResShift 中把这个替换落在了 `WindowAttention`，并用配置开关实现可控启停，同时保留 masked（shift）路径的原始注意力来保证稳定性与兼容性。
