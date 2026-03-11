# run
CUDA_VISIBLE_DEVICES=1 python quantization/quantize_awq.py \
   --config configs/realsr_swinunet_realesrgan256.yaml \
   --checkpoint weights/resshift_realsrx4_s15_linfusion.pth \
   --calibration_dir quantization/calibration64 \
   --output quantization/artifacts/test_awq_linfusion.pth \
   --samples_per_image 4 \
   --use_linfusion True

# verify
# python quantization/verify_awq.py \
#   --config configs/realsr_swinunet_realesrgan256.yaml \
#   --checkpoint quantization/artifacts/test_awq.pth \