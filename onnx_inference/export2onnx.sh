cd ~/ResShift/onnx_inference
# export autoencoder and Unet
#python export2onnx.py --config ~/ResShift/configs/realsr_swinunet_realesrgan256.yaml

# export UNet only
python export2onnx.py --config ~/ResShift/configs/realsr_swinunet_realesrgan256.yaml --autoencoder False

# export Encoder/Decoder only
#python export2onnx.py --config ~/ResShift/configs/realsr_swinunet_realesrgan256.yaml --unet False