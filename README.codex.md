### codex 注意事项

1. python环境, 当你想要真实运行python时, 请先激活ResShift conda环境: `conda activate ResShift`  
2. GPU内存不足. 当你运行python出现GPU内存不足的错误时, 你可以先执行 `nvidia-smi` 来查看当前GPU的使用情况, 选择一个内存使用较少的GPU, 然后设置环境变量 `export CUDA_VISIBLE_DEVICES=0` (假设你选择了GPU 0) 来指定使用哪个GPU. 这样可以避免内存不足的问题.  
**注意: 不要通过kill其他正在运行的进程来使GPU空闲!!!**
**注意: 不要通过kill其他正在运行的进程来使GPU空闲!!!**
**注意: 不要通过kill其他正在运行的进程来使GPU空闲!!!**