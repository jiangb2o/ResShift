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

# INT8 PTQ
  python quantization/quantize_hybrid_ptq.py \
    --config configs/realsr_swinunet_realesrgan256.yaml \
    --checkpoint weights/resshift_realsrx4_s15_v1_default.pth \
    --calibration_dir quantization/calibration64 \
    --output quantization/artifacts/test_hybrid_ptq.pth \
    --device cpu \

  python quantization/verify_hybrid_ptq.py \
    --config configs/realsr_swinunet_realesrgan256.yaml \
    --checkpoint quantization/artifacts/test_hybrid_ptq.pth \
    --reference_checkpoint weights/resshift_realsrx4_s15_v1_default.pth \