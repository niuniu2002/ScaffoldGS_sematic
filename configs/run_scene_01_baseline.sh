#!/bin/bash
# Benchmark: SW_scenes/scene_01
# Dataset: 479 images (sparse/0), 3840x2160, SegmentationClass masks
# Binary segmentation (num_classes=1)
# NOTE: No train/test split by default; run generate_split.py first if needed.

DATA="/mnt/data/liufengyang/data/dataset/SW_scenes/scene_01"
EXP="sw_scene_01_baseline"

echo "=== Training ${EXP} on SW_scenes/scene_01 ==="

python train.py \
  -s ${DATA} \
  -m output/${EXP} \
  --start_semantic_iter 5000 \
  --mask_weight 0.1 \
  --mask_warmup 1000 \
  --mask_ramp 3000 \
  --knn_weight 0.02 \
  --knn_warmup 2000 \
  --knn_ramp 3000 \
  --knn_every 100 \
  --knn_offset 55 \
  --focal_alpha 0.25 \
  --update_until 0 \
  --iterations 30000

echo "=== Evaluating ${EXP} ==="
python eval_myvideo.py -m output/${EXP} --iteration 30000
