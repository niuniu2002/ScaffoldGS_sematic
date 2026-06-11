#!/bin/bash
# Benchmark: dronev4_2
# Dataset: 333 images, 266 train / 67 test, SAM masks + human annotations
# Target: Solidago canadensis (加拿大一枝黄花) segmentation
# SOTA config: no_opacity_detach + mask_weight=0.4

DATA="/mnt/data/liufengyang/data/dataset/dronev4_2"
EXP="dronev4_2_sota_nodetach"

echo "=== Training ${EXP} on dronev4_2 ==="

python train.py \
  -s ${DATA} \
  -m output/${EXP} \
  --start_semantic_iter 5000 \
  --mask_weight 0.4 \
  --knn_weight 0.05 \
  --knn_every 100 \
  --knn_offset 55 \
  --focal_alpha 0.25 \
  --update_until 15000 \
  --no_opacity_detach \
  --iterations 30000

echo "=== Evaluating ${EXP} on test set ==="
python eval_myvideo.py -m output/${EXP} --iteration 30000
