python  onnx_inference/build_trt_engines.py \
    --models all \
    --fp16

# 推理
CUDA_VISIBLE_DEVICES=5 run_tensorrt_pipeline.py \
    --input onnx_inference/outputs/test_lq.png \
    --output onnx_inference/outputs/test_sr_trt.png \
    --encoder_engine onnx_inference/engines/autoencoder_encoder.engine \
    --decoder_engine default="onnx_inference/engines/autoencoder_decoder.engine \
    --unet_engine default="onnx_inference/engines/resshift_model.engine \