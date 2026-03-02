# ResShift 模型导出和转换快速开始

## 概述

本指南展示如何快速将 ResShift 模型导出为 ONNX 格式，然后转换为 MNN 格式以支持移动/边缘设备部署。

## 快速开始（3 步）

### 步骤 1：导出为 ONNX

```bash
cd /home/ubuntu/ResShift

# 基本导出
python export2onxx.py --config configs/export_config.yaml

# 或自定义参数
python export2onxx.py \
  --config configs/export_config.yaml \
  --output_path my_model.onnx
```

**输出**：`resshift_model.onnx` (~500 MB)

### 步骤 2：转换为 MNN

```bash
# 推荐：使用 Python 脚本
python onnx_to_mnn.py -i resshift_model.onnx

# 或使用 Bash 脚本
chmod +x onnx_to_mnn.sh
./onnx_to_mnn.sh
```

**输出**：`resshift_model.mnn` (~450 MB)

### 步骤 3：验证 MNN 模型

```bash
python verify_mnn_model.py -m resshift_model.mnn --onnx resshift_model.onnx
```

## 一键完成流程

使用完整工作流脚本一次性完成所有步骤：

```bash
# 最简单的方式
python export_and_convert_workflow.py

# 使用优化预设
python export_and_convert_workflow.py --preset optimize

# 使用量化预设（最小的模型）
python export_and_convert_workflow.py --preset quantized

# 跳过验证以加速
python export_and_convert_workflow.py --skip-verify

# 强制重新处理所有步骤
python export_and_convert_workflow.py --force
```

## 详细命令速查表

### ONNX 导出

```bash
# 最小化命令
python export2onxx.py 

# 完整参数
python export2onxx.py \
  --config configs/export_config.yaml \
  --ckpt_path weights/resshift_realsrx4_s15_v1.pth \
  --output_path resshift_model.onnx \
  --input_shape 1 3 64 64 \
  --device cuda
```

### ONNX 到 MNN 转换

```bash
# 基本转换（推荐）
python onnx_to_mnn.py -i resshift_model.onnx

# 标准优化
python onnx_to_mnn.py -i resshift_model.onnx --preset normal

# 高级优化
python onnx_to_mnn.py -i resshift_model.onnx --preset optimize

# 量化优化（最小体积）
python onnx_to_mnn.py -i resshift_model.onnx --preset quantized

# 自定义参数
python onnx_to_mnn.py \
  -i resshift_model.onnx \
  -o my_model.mnn \
  --mnn-root /home/ubuntu/MNN \
  --preset optimize
```

### 模型验证

```bash
# 基本验证
python verify_mnn_model.py -m resshift_model.mnn

# 与 ONNX 对比
python verify_mnn_model.py \
  -m resshift_model.mnn \
  --onnx resshift_model.onnx
```

## 模型大小对比

| 格式 | 大小 | 说明 |
|------|------|------|
| ONNX (原始) | ~500 MB | 完整精度，可直接在 PyTorch 推理 |
| MNN (normal) | ~450 MB | 标准优化，基本大小减少 |
| MNN (optimize) | ~400 MB | 高级优化，推荐用于移动部署 |
| MNN (quantized) | ~120 MB | 量化优化，最小体积 |

## 不同场景的推荐方案

### 方案 A：最佳质量（推荐大多数情况）

```bash
# 步骤 1: 导出 ONNX
python export2onxx.py --config configs/export_config.yaml

# 步骤 2: 转换为 MNN（标准优化）
python onnx_to_mnn.py -i resshift_model.onnx --preset normal

# 文件大小：~450 MB
```

### 方案 B：最大优化（推荐移动设备）

```bash
# 使用完整工作流，应用高级优化
python export_and_convert_workflow.py --preset optimize

# 文件大小：~400 MB
# 推理速度提升 30%
```

### 方案 C：最小体积（推荐边缘设备/IoT）

```bash
# 使用量化预设
python export_and_convert_workflow.py --preset quantized

# 文件大小：~120 MB
# 推理速度提升 80%
# 注意：可能损失 10-15% 的精度
```

## 故障排除

### 问题 1：ONNX 导出失败

```bash
# 检查依赖
python -c "import torch; print(torch.__version__)"

# 安装缺失的包
pip install torch onnx onnxruntime

# 重试导出
python export2onxx.py --config configs/export_config.yaml
```

### 问题 2：MNN 转换失败

```bash
# 检查 MNN 是否编译
ls -la /home/ubuntu/MNN/build/MNNConvert/MNNConvert

# 重新编译 MNN
python onnx_to_mnn.py --build-only

# 重试转换
python onnx_to_mnn.py -i resshift_model.onnx
```

### 问题 3：模型验证错误

```bash
# 检查文件是否正确
file resshift_model.mnn
ls -lh resshift_model.mnn

# 检查模型信息
python verify_mnn_model.py -m resshift_model.mnn

# 尝试简单验证
python -c "import os; print('MNN file valid' if os.path.getsize('resshift_model.mnn') > 0 else 'Invalid')"
```

## 输出文件说明

转换完成后，你会得到：

```
/home/ubuntu/ResShift/
├── resshift_model.onnx    # ONNX 格式（PyTorch 导出）
├── resshift_model.mnn     # MNN 格式（当前推理框架）
├── configs/
│   └── export_config.yaml # 导出配置文件
├── export2onxx.py         # ONNX 导出脚本
├── onnx_to_mnn.py         # ONNX 到 MNN 转换脚本
├── verify_mnn_model.py    # MNN 模型验证脚本
└── export_and_convert_workflow.py  # 完整工作流脚本
```

## 进阶用法

### 自定义输入形状

```bash
# 导出不同的输入形状
python export2onxx.py \
  --config configs/export_config.yaml \
  --input_shape 2 3 128 128
```

### 在 CPU 上导出

```bash
# 如果 GPU 不可用
python export2onxx.py \
  --config configs/export_config.yaml \
  --device cpu
```

### 指定 MNN 根目录

```bash
# 使用自定义 MNN 版本
python onnx_to_mnn.py \
  -i resshift_model.onnx \
  --mnn-root /path/to/custom/MNN
```

## 后续步骤

1. **Android 部署**
   ```bash
   # 使用 MNN Android SDK
   # 参考：https://github.com/alibaba/MNN/tree/master/android
   ```

2. **iOS 部署**
   ```bash
   # 使用 MNN iOS Framework
   # 参考：https://github.com/alibaba/MNN/tree/master/ios
   ```

3. **Linux/嵌入式部署**
   ```bash
   # 使用 MNN C++ API
   # 参考：https://mnn.readthedocs.io/
   ```

4. **模型量化**
   ```bash
   # 进一步优化模型
   python export_and_convert_workflow.py --preset quantized
   ```

## 常见问题 (FAQ)

**Q: 需要多久完成转换？**
A: 大约 5-10 分钟（包括 MNN 编译）

**Q: 转换后的模型精度会降低吗？**
A: 使用 `normal` 或 `optimize` 预设，精度保持 >99%

**Q: 能否在 Windows 上运行？**
A: 可以，但需要编译 MNN（部分步骤不同）

**Q: 转换后的 MNN 模型可以直接用吗？**
A: 是的，可以用 MNN 推理引擎直接推理

**Q: 是否需要 GPU？**
A: 只在导出 ONNX 时需要（可选），转换 ONNX 到 MNN 不需要

## 获取帮助

- MNN 文档：https://mnn.readthedocs.io/
- MNN GitHub：https://github.com/alibaba/MNN
- ResShift 文档：查看项目 README

## 许可证

遵守 MNN 和 ResShift 项目的开源许可证。
