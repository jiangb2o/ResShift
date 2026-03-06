#$gt_dir="database/imagenet256_srx4/imagenet256/gt"
#sr_dir="inference_result/imagenet_test"

gt_dir="inference_result/runtime_test"
sr_dir="inference_result/runtime_quant"
python cal_metrics_sr.py --gt_dir $gt_dir --sr_dir $sr_dir --device cuda:0