# run
# CUDA_VISIBLE_DEVICES=4 python quantization/quantize_awq.py \
#    --config configs/realsr_swinunet_realesrgan256.yaml \
#    --checkpoint weights/resshift_realsrx4_s15_v1_default.pth \
#    --calibration_dir quantization/calibration64 \
#    --output quantization/artifacts/test_awq.pth \
#    --samples_per_image 4 \

# verify
python quantization/verify_awq.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --checkpoint quantization/artifacts/test_awq.pth \