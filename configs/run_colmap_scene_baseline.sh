#!/bin/bash
# Benchmark: lfy/colmap_scene
# Dataset: 200 images, 1920x1080, ISAT annotation masks
# Binary segmentation (num_classes=1)

DATA="/mnt/data/liufengyang/data/dataset/lfy/colmap_scene"
EXP="lfy_colmap_scene_baseline"

echo "=== Training ${EXP} on lfy/colmap_scene ==="

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
