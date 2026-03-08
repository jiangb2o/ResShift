cd ~/ResShift/onnx_inference

# CUDA_VISIBLE_DEVICES=3 python run_onnx_pipeline.py \
#   --config configs/realsr_swinunet_realesrgan256.yaml \
#   --encoder_onnx onnx_inference/models/autoencoder_encoder.onnx \
#   --decoder_onnx onnx_inference/models/autoencoder_decoder.onnx \
#   --input onnx_inference/outputs/test_lq.png \
#   --output onnx_inference/outputs/test_sr.png \
#   --unet_onnx onnx_inference/models/resshift_model.onnx

# CUDA_VISIBLE_DEVICES=3 python run_onnx_pipeline.py \
#   --config configs/realsr_swinunet_realesrgan256.yaml \
#   --encoder_onnx onnx_inference/models/autoencoder_encoder.onnx \
#   --decoder_onnx onnx_inference/models/autoencoder_decoder.onnx \
#   --input onnx_inference/outputs/test_lq.png \
#   --output onnx_inference/outputs/test_sr.png \
#   --unet_onnx onnx_inference/models_linear_attn/resshift_model.onnx \

# 多图onnx推理 非 linear atten
CUDA_VISIBLE_DEVICES=7 python run_onnx_pipeline.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --encoder_onnx onnx_inference/models/autoencoder_encoder.onnx \
  --decoder_onnx onnx_inference/models/autoencoder_decoder.onnx \
  --input database/imagenet256_srx4/imagenet256/test_lq_100 \
  --output onnx_inference/outputs/runtime_test \
  --unet_onnx onnx_inference/models/resshift_model.onnx

# linear atten
CUDA_VISIBLE_DEVICES=7 python run_onnx_pipeline.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --encoder_onnx onnx_inference/models/autoencoder_encoder.onnx \
  --decoder_onnx onnx_inference/models/autoencoder_decoder.onnx \
  --input database/imagenet256_srx4/imagenet256/test_lq_100 \
  --output onnx_inference/outputs/runtime_test2 \
  --unet_onnx onnx_inference/models_linear_attn/resshift_model.onnx \
