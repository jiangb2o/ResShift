
实现了 保持参数的 resshift 线性注意力改造  

SwinTransformer结构, 采用的是WindowAttention
1. 相对位置偏置
$$
SoftMax(\frac{QK^T}{\sqrt{d}} + B) V
$$
2. 在 SW-MSA 阶段要加入 mask 掩码  
$$
SoftMax(\frac{QK^T}{\sqrt{d}} + B + Mask) V
$$

测试改造后的原模型推理性能:  
未改造:         0.1213s per image
线性注意力改造:  0,1204s per image

导出未onnx格式后的性能:  
未改造:        0.51s per image
线性注意力改造: 0.42s per image  

现象: 在原模型推理中, 线性注意力并未带来推理速度上的提升. 但是在onnx格式下进行推理时, 线性注意力改造能够带来约 25% 的推理速度提升.

TODO: 线性注意力训练: 采样微调方法, 使用 1% 的训练数据, 使用原模型参数, 冻结除线性注意力以外的参数, 对线性注意力参数进行训练.  