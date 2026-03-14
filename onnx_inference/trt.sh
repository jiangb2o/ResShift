cd ~/ResShift/onnx_inference

# build
# default
# CUDA_VISIBLE_DEVICES=4 python build_trt_engines.py \
#     --models unet \
#     --unet_onnx onnx_inference/models/resshift_model.onnx \
#     --latent_min_hw 64 64 \
#     --latent_opt_hw 64 64 \
#     --latent_max_hw 64 64 \
#     --fp16

# linfusion
CUDA_VISIBLE_DEVICES=3 python build_trt_engines.py \
    --models unet \
    --unet_onnx onnx_inference/models_linear_attn/linfusion.onnx \
    --latent_min_hw 64 64 \
    --latent_opt_hw 64 64 \
    --latent_max_hw 64 64 \
    --fp16

# awq + linfusion
CUDA_VISIBLE_DEVICES=3 python build_trt_engines.py \
    --models unet \
    --unet_onnx onnx_inference/models_linear_attn/awq_linfusion_unet.onnx \
    --latent_min_hw 64 64 \
    --latent_opt_hw 64 64 \
    --latent_max_hw 64 64 \
    --fp16

# int8 + linfusion
CUDA_VISIBLE_DEVICES=3 python build_trt_engines.py \
    --models unet \
    --unet_onnx onnx_inference/models_linear_attn/int8_only_unet.onnx \
    --latent_min_hw 64 64 \
    --latent_opt_hw 64 64 \
    --latent_max_hw 64 64 \
    --fp16

# int8 + awq + linfusion
CUDA_VISIBLE_DEVICES=3 python build_trt_engines.py \
    --models unet \
    --unet_onnx onnx_inference/models_linear_attn/hybrid_ptq_unet.onnx \
    --latent_min_hw 64 64 \
    --latent_opt_hw 64 64 \
    --latent_max_hw 64 64 \
    --fp16

#input=database/imagenet256_srx4/imagenet256/test_lq_10
input=database/imagenet256_srx4/imagenet256/lq
output=onnx_inference/outputs_trt/default
output2=onnx_inference/outputs_trt/linfusion
output3=onnx_inference/outputs_trt/awq_linfusion
output4=onnx_inference/outputs_trt/int8_linfusion
output5=onnx_inference/outputs_trt/awq_int8_linfusion

# 推理
# CUDA_VISIBLE_DEVICES=3 python run_tensorrt_pipeline.py \
#     --input $input \
#     --output $output \
#     --encoder_engine onnx_inference/engines/autoencoder_encoder.engine \
#     --decoder_engine onnx_inference/engines/autoencoder_decoder.engine \
#     --unet_engine onnx_inference/engines/resshift_model.engine \

# # linfusion
# CUDA_VISIBLE_DEVICES=3 python run_tensorrt_pipeline.py \
#     --input $input \
#     --output $output2 \
#     --encoder_engine onnx_inference/engines/autoencoder_encoder.engine \
#     --decoder_engine onnx_inference/engines/autoencoder_decoder.engine \
#     --unet_engine onnx_inference/engines/linfusion.engine \

# # awq
# CUDA_VISIBLE_DEVICES=3 python run_tensorrt_pipeline.py \
#     --input $input \
#     --output $output3 \
#     --encoder_engine onnx_inference/engines/autoencoder_encoder.engine \
#     --decoder_engine onnx_inference/engines/autoencoder_decoder.engine \
#     --unet_engine onnx_inference/engines/awq_linfusion_unet.engine  \

# # int8
# CUDA_VISIBLE_DEVICES=3 python run_tensorrt_pipeline.py \
#     --input $input \
#     --output $output4 \
#     --encoder_engine onnx_inference/engines/autoencoder_encoder.engine \
#     --decoder_engine onnx_inference/engines/autoencoder_decoder.engine \
#     --unet_engine onnx_inference/engines/int8_only_unet.engine \

# CUDA_VISIBLE_DEVICES=3 python run_tensorrt_pipeline.py \
#     --input $input \
#     --output $output5 \
#     --encoder_engine onnx_inference/engines/autoencoder_encoder.engine \
#     --decoder_engine onnx_inference/engines/autoencoder_decoder.engine \
#     --unet_engine onnx_inference/engines/hybrid_ptq_unet.engine \
