gt_dir="database/imagenet256_srx4/imagenet256/gt"
#sr_dir="inference_result/imagenet_test"
sr_dir="inference_result/imagenet_test_linfusion"
#sr_dir="inference_result/runtime_quant"

#gt_dir="inference_result/runtime_test"
#sr_dir="inference_result/runtime_quant"
CUDA_VISIBLE_DEVICES=1 python cal_metrics_sr.py --gt_dir $gt_dir --sr_dir $sr_dir --device cuda:0
