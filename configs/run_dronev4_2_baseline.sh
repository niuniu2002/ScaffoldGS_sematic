#!/bin/bash
# Benchmark: dronev4_2
# Dataset: 333 images, 266 train / 67 test, SAM masks + human annotations
# Target: Solidago canadensis (加拿大一枝黄花) segmentation

DATA="/mnt/data/liufengyang/data/dataset/dronev4_2"
EXP="dronev4_2_baseline"

echo "=== Training ${EXP} on dronev4_2 ==="

# Best balance: mIoU ~0.75, PSNR gap ~1.0 dB
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

echo "=== Evaluating ${EXP} on human-annotated myvideo ==="
python eval_myvideo.py -m output/${EXP} --iteration 30000
