#!/bin/bash
# Benchmark: dronev4_2 with DroneSplat-style dynamic densify threshold scheduling
# Dataset: 333 images, 266 train / 67 test, SAM masks + human annotations
# Target: Solidago canadensis (加拿大一枝黄花) segmentation
# Base: true SOTA config (Dual-Feature + no_opacity_detach) + scheduled densify threshold

DATA="/mnt/data/liufengyang/data/dataset/dronev4_2"
EXP="dronev4_2_sota_dualfeat_nodetach_scheduled_densify"

echo "=== Training ${EXP} on dronev4_2 ==="

python train.py \
  -s ${DATA} \
  -m output/${EXP} \
  --num_classes 1 \
  --eval \
  --resolution 2 \
  --white_background \
  --no_opacity_detach \
  --dual_feature \
  --seg_feature_dim 8 \
  --seg_decoder_hidden 64 \
  --seg_decoder_layers 2 \
  --mask_weight 0.2 \
  --start_semantic_iter 500 \
  --mask_every 5 \
  --update_until 15000 \
  --schedule_densify_grad_threshold \
  --densify_grad_threshold_final 0.001 \
  --knn_adaptive \
  --knn_late_stage_iter 10000 \
  --knn_late_stage_factor 5 \
  --knn_min_samples 256 \
  --knn_max_samples 1024 \
  --knn_weight 0.02 \
  --knn_every 100 \
  --knn_offset 55 \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --iterations 30000 \
  --test_iterations 7000 12000 15000 30000 \
  --save_iterations 30000 \
  --port 6232

echo "=== Evaluating ${EXP} on myvideo human annotations ==="
python eval_myvideo.py -m output/${EXP} --iteration 30000

echo "=== Training complete: ${EXP} ==="
