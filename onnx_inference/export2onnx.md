# ResShift 模型导出和转换

## 概述

ONNX (Open Neural Network Exchange)  
存储了网络的拓扑结构与权重, 重点在于其通用性与开放性, 可以在不同的深度学习框架中迁移. 此外 ONNX 格式支持多种硬件加速库以及推理引擎(ONNX Runtime), 使模型推理使能够根据硬件进行优化, 从而提升推理性能(端侧选择ONNX格式的原因).

ONNX使用计算图来表示模型, 每个节点代表一个操作, 边代表节点之间的数据流.

## 导出为 ONNX

而实际的推理为三段式:
* encoder
* unet (多步)
* decoder  
因此, 在使用导出 onnx 格式进行推理时, 先对 encoder 和 decoder 进行导出. 再导出 unet. 并设计循环控制、步数以及噪声注入. 
如果将这些步骤都一并导出, 会造成模型显著变大(unet 循环)且难以维护.

```bash
bash export2onnx.sh
```

## Encoder 导出问题  

### 不支持的操作符
AutoEncoder 模块中使用了 `xformers.ops.memory_efficient_attention`  
导致以下错误: RuntimeError: unsupported output type: int, from operator:xformers::efficient_attention_forward_cutlass  

解决方案: 禁用 xformer, 从 efficient_attention 转为普通的 attention 实现

### TracerWarning  
/home/ubuntu/ResShift/ldm/modules/diffusionmodules/model.py:192  
Converting a tensor to a Python integer might cause the trace to be incorrect. We can't record the data flow of  Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!  
w_ = w_ * (int(c)**(-0.5))  

解决:  
其中的c为某个卷积的输出通道数, 从shape获取是一个维度为1的tensor. 转为int无法记录, 导致计算图中该值不会根据输入动态改变. Warning, 只需查到对应卷积的输出通道设置, 直接使用该int值即可解决问题  

## UNet 导出问题
### UserWarning  
/home/ubuntu/miniconda3/envs/ResShift/lib/python3.10/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]

解决:  
在meshgrid调用处加上索引参数:  
models/swin_transformer.py, line: 95
```py
coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij")) 
```

### TracerWarning 将tensor转换为int导致
/home/ubuntu/ResShift/models/swin_transformer.py:60: TracerWarning: Converting a tensor to a Python integer
  might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be
  treated as a constant in the future. This means that the trace might not generalize to other inputs!
    B = int(windows.shape[0] / (H * W / window_size / window_size))

同样是转为int无法记录的问题, 导致计算图中该值不会根据输入动态改变, 改为等价实现即可:  
```py
# B = int(windows.shape[0] / (H * W / window_size / window_size))
# int类型强制转换导致 onnx 导出时将其转换为常量. 因此改为动态计算B
num_windows = (H * W) // (window_size * window_size)
B = windows.shape[0] // num_windows
```

## `run_onnx_pipeline.py` 详细流程说明

该脚本对应的是 ResShift 原生推理链路的 ONNX 版本实现：  
`encode_first_stage -> p_sample_loop -> decode_first_stage`

---

### 1. 读取配置与构建 diffusion 调度器

脚本先加载 `configs/realsr_swinunet_realesrgan256.yaml`，并调用：

```py
diffusion = create_gaussian_diffusion(**cfg.diffusion.params)
```

对应原 ResShift：
- `sampler.py` 中 `self.base_diffusion = instantiate_from_config(self.configs.diffusion)`

作用：
- 得到完整扩散参数：`etas / sqrt_etas / kappa / posterior_mean_coef1 / posterior_mean_coef2 / posterior_log_variance_clipped / model_mean_type` 等。

---

### 2. 图像预处理与尺寸对齐

流程：
1. 读图并归一化到 `[-1, 1]`。
2. 将输入 padding 到 `lq_size`（默认 64）的整数倍。
3. 按超分倍率 `sf` 做 bicubic 上采样（4x 时从 LQ 到 256 分辨率）。

对应原 ResShift：
- `sampler.py::sample_func()` 的 padding 逻辑。
- `GaussianDiffusion.encode_first_stage(..., up_sample=True)` 中的 `F.interpolate(..., scale_factor=self.sf, mode='bicubic')`。

---

### 3. 加载三个 ONNX 子模型

加载：
- `encoder_onnx`
- `unet_onnx`
- `decoder_onnx`

其输入/输出语义：
- Encoder: `image -> latent`
- UNet: `x, lq, timesteps -> output`
- Decoder: `latent -> image`

---

### 4. Encoder 推理与 latent 缩放

步骤：
1. 将上采样后的图送入 encoder，得到 `z_y`。
2. 执行 `z_y = z_y * scale_factor`。

对应原 ResShift：
- `GaussianDiffusion.encode_first_stage()`：
  - `z_y = first_stage_model.encode(y)`
  - `out = z_y * self.scale_factor`

---

### 5. 初始化反向扩散起点 `z_t`

脚本实现：

```py
t_last = diffusion.num_timesteps - 1
z_t = z_y + kappa * sqrt_etas[t_last] * noise
```

对应原 ResShift：
- `GaussianDiffusion.prior_sample(y, noise)`：
  - `y + kappa * sqrt_etas[t_last] * noise`

意义：
- 这是反向采样的起始状态，等价于从 `q(x_T | y)` 采样。

---

### 6. 时间步 `timesteps` 的来源

脚本中先构造：

```py
indices = list(range(diffusion.num_timesteps))[::-1]
```

即从 `T-1 -> 0` 反向迭代。每次循环：
- `i` 是当前反向步索引；
- `model_t` 是喂给 UNet 的时间步。

若启用了 `SpacedDiffusion`，则通过 `timestep_map` 做映射：

```py
model_t = timestep_map[i] if timestep_map is not None else i
```

对应原 ResShift：
- `p_sample_loop_progressive()` 中 `indices = list(range(self.num_timesteps))[::-1]`
- `respace.py::_WrappedModel.__call__()` 中对 `ts` 的映射逻辑。

---

### 7. 每一步 UNet 输入是什么

每一步给 UNet 的 3 个输入：
- `x`: 当前状态 `x_t`（脚本变量 `z_t`），先经过 `_scale_input` 归一化；
- `lq`: 条件图像（脚本使用原始 `y0`，与 `model_kwargs['lq']` 对齐）；
- `timesteps`: 当前时刻 `model_t`。

对应原 ResShift：
- `GaussianDiffusion.p_mean_variance()`：
  - `model(self._scale_input(x_t, t), t, **model_kwargs)`
- `sampler.py` 中 `model_kwargs={'lq': y0}`

---

### 8. `model_mean_type` 是什么

`model_mean_type` 决定了 UNet 输出代表什么目标，来自 diffusion 配置中的 `predict_type`。

脚本支持：
- `START_X`：UNet 直接预测 `x_0`
- `RESIDUAL`：UNet 预测 `y - x_0`
- `EPSILON` / `EPSILON_SCALE`：UNet 预测噪声形式，再按公式恢复 `x_0`

对应原 ResShift：
- `GaussianDiffusion.p_mean_variance()` 里对 `ModelMeanType` 的分支处理
- `_predict_xstart_from_residual / _predict_xstart_from_eps / _predict_xstart_from_eps_scale`

---

### 9. `coef1 / coef2` 是什么

这两个系数来自扩散后验分布：
- `posterior_mean_coef1[t]`
- `posterior_mean_coef2[t]`

并用于计算：

```py
mean = coef1 * x_t + coef2 * x0_pred
```

对应原 ResShift：
- `GaussianDiffusion.q_posterior_mean_variance()`

物理意义：
- 它们定义了 `q(x_{t-1} | x_t, x_0)` 的后验均值。

---

### 10. 噪声注入与最终一步

非最后一步（`i != 0`）：

```py
sigma = exp(0.5 * posterior_log_variance_clipped[i])
x_{t-1} = mean + sigma * noise
```

最后一步（`i == 0`）：
- 不再注入噪声，直接 `x_0 = mean`

对应原 ResShift：
- `p_sample()` 中 `nonzero_mask = (t != 0)` 的逻辑（`t=0` 不加噪声）

---

### 11. Decoder 推理与后处理

步骤：
1. 先除以 `scale_factor`：`z_out = z_t / scale_factor`
2. 输入 decoder ONNX，得到 SR 图像（`[-1,1]`）
3. 去除 padding 区域
4. 反归一化到 `[0,255]` 保存

对应原 ResShift：
- `GaussianDiffusion.decode_first_stage()` 中先执行 `z_sample = 1 / self.scale_factor * z_sample`，再调用 `first_stage_model.decode(...)`

---

## 小结

`run_onnx_pipeline.py` 把原 ResShift 的关键推理路径逐项映射到了 ONNX 子模型调用中：

1. 与原流程一致的 `encode_first_stage`  
2. 与原公式一致的 `p_sample_loop`（包括 timestep、model_mean_type、posterior coef、sigma）  
3. 与原流程一致的 `decode_first_stage`

因此，只要导出的三个 ONNX 与对应权重匹配，其结果应与原 PyTorch 推理在数值上保持一致趋势（允许有小量数值误差）。
