
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

TODO: 线性注意力模型重新训练


TODO: 量化没有加速效果, 是不是推理时还是使用了原参数类型? 没有使用 量化后的 int 类型. 量化主要是降低模型的内存占用, 在资源受限平台能够加速推理. 在实际jetson nano推理时, 还是需要将参数反量化为fp16精度.  