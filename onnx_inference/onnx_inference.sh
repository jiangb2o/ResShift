cd ~/ResShift/onnx_inference
python run_onnx_pipeline.py \
  --config configs/realsr_swinunet_realesrgan256.yaml \
  --unet_onnx onnx_inference/models/resshift_model.onnx \
  --encoder_onnx onnx_inference/models/autoencoder_encoder.onnx \
  --decoder_onnx onnx_inference/models/autoencoder_decoder.onnx \
  --input onnx_inference/outputs/test_lq.png \
  --output onnx_inference/outputs/test_sr.png \
