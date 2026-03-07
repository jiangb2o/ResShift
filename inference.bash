
# result_path="inference_result/imagenet_test"

# LOCAL_RANK=0 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o $result_path --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64
# LOCAL_RANK=1 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o $result_path --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64
# LOCAL_RANK=2 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o $result_path --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64
# LOCAL_RANK=3 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o $result_path --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64
# LOCAL_RANK=4 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o $result_path --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64
# LOCAL_RANK=5 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o $result_path --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64
# LOCAL_RANK=6 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o $result_path --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64
# LOCAL_RANK=7 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/lq -o $result_path --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64

# find $result_path -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l


# psrn, ssim, lpips 指标
# python cal_metrics_sr.py --gt_dir database/imagenet256_srx4/imagenet256/gt --sr_dir $result_path --device cuda:0

# iqa 指标
# python cal_iqa.py --sr_dir $result_path --device cuda:0


# 单图推理
# CUDA_VISIBLE_DEVICES=3 python inference_resshift.py -i onnx_inference/outputs/shu.jpg \
#  --use_linfusion True -o onnx_inference/outputs/shu_sr --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64

# 单卡推理性能测试
result_path="inference_result/runtime_test"
result_path2="inference_result/runtime_test2"
result_path3="inference_result/runtime_quant"

# default
# CUDA_VISIBLE_DEVICES=3 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/test_lq_10 -o $result_path \
#   --use_linfusion False --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64 \
#   --profile_inference True

CUDA_VISIBLE_DEVICES=3 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/test_lq_50 -o $result_path \
  --use_linfusion False --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 24 \
  --profile_inference True

# linfusion
#  CUDA_VISIBLE_DEVICES=4 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/test_lq_10 -o $result_path2 \
#   --use_linfusion True --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64 
# 量化
# CUDA_VISIBLE_DEVICES=4 python inference_resshift.py -i database/imagenet256_srx4/imagenet256/test_lq_10 -o $result_path3 \
#  --use_linfusion False --task realsr --scale 4 --version v1 --chop_size 64 --chop_stride 64 --bs 64 \
#  --checkpoint_path quantization/artifacts/test_awq.pth \
#  --profile_inference True
