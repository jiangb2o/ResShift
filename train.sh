MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES=0,1,2,3,6,7 \
 torchrun --standalone --nproc_per_node=6 --nnodes=1 main.py --cfg_path configs/realsr_swinunet_realesrgan256.yaml --save_dir result