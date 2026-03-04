# train
MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 --nnodes=1 main.py --cfg_path configs/realsr_swinunet_realesrgan256.yaml --save_dir result


#
MKL_THREADING_LAYER=GNU 解决以下报错
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library. Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.


tmux new -s [name]
tmux -a -t [name]
tmux set -g mouse on

# 
LOCAL_RANK=0 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o inference_result/imagenet_test --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64

LOCAL_RANK=0 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o inference_result/imagenet_test_mymodel --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64

# 评估
python cal_metrics_sr.py --gt_dir database/imagenet256_srx4/imagenet256/gt --sr_dir inference_result/imagenet_test --device cuda:0

# 文件夹图片数量
 find inference_result/imagenet_test -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l

 # 论文中结果
 PSNR: 25.01  
 SSIM: 0.677  
 LPIPS: 0.231


 # 自己练的模型
Images evaluated: 3000
PSNR: 25.5603  (std: 3.4862)
SSIM: 0.7635  (std: 0.1150)
LPIPS-VGG: 0.1442
LPIPS-AlexNet: 0.0792

# 作者模型结果
Images evaluated: 3000
PSNR: 23.0134  (std: 3.3612)
SSIM: 0.6182  (std: 0.1554)
LPIPS-VGG: 0.3428
LPIPS-AlexNet: 0.2385



### 导出为 ONXX
```sh
python export2onxx.py --config configs/export_config.yaml
```