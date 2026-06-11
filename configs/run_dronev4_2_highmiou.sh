#!/bin/bash
# Benchmark: dronev4_2 (High mIoU configuration)
# Warning: Higher PSNR gap (~3.5-4.0 dB) due to stronger semantic loss

DATA="/mnt/data/liufengyang/data/dataset/dronev4_2"
EXP="dronev4_2_highmiou"

echo "=== Training ${EXP} on dronev4_2 ==="

python train.py \
  -s ${DATA} \
  -m output/${EXP} \
  --start_semantic_iter 0 \
  --mask_weight 0.2 \
  --mask_warmup 1000 \
  --mask_ramp 3000 \
  --knn_weight 0.05 \
  --knn_warmup 2000 \
  --knn_ramp 3000 \
  --knn_every 100 \
  --knn_offset 55 \
  --focal_alpha 0.25 \
  --update_until 15000 \
  --iterations 33000

echo "=== Evaluating ${EXP} ==="
python eval_myvideo.py -m output/${EXP} --iteration 33000
