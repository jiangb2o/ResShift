# ResShift 模型导出和转换

## 概述

本指南展示如何快速将 ResShift 模型导出为 ONNX 格式

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
AutoEncoder 模块中使用了 `xformers.ops.memory_efficient_attention`  
导致以下错误: RuntimeError: unsupported output type: int, from operator:xformers::efficient_attention_forward_cutlass  
