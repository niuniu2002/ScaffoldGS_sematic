#!/usr/bin/env bash
set -e

# ============================================================
# 实验：exp15_startsem3000_mask02
# 目的：highpsnr 实验用了 start_semantic_iter=3000 + mask_weight=0.05，结果 mIoU 崩盘。
#      这里保持 start_semantic_iter=3000（比 exp06 更早引入语义），但 mask_weight 提升到 0.2，
#      测试是否能同时拿到高 PSNR（>25.0）和高 mIoU（>0.77）。
# ============================================================

SOURCE_PATH="data/dronev4_2"
MODEL_PATH="output/dronev4_2_exp15_startsem3000_mask02"
LOG_PATH="autoresearch_scaffoldgs/logs/exp15_startsem3000_mask02_train.log"

mkdir -p "$(dirname "$LOG_PATH")"

/mnt/data/liufengyang/envs/scaffold_gslfy/bin/python train.py \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --iterations 30000 \
  --resolution 2 \
  --white_background \
  --start_semantic_iter 3000 \
  --mask_weight 0.2 \
  --mask_warmup 0 \
  --mask_ramp 0 \
  --knn_weight 0.05 \
  --knn_every 100 \
  --knn_offset 55 \
  --knn_warmup 0 \
  --knn_ramp 0 \
  --focal_alpha 0.75 \
  --focal_gamma 2.0 \
  --uncertainty_min 0.1 \
  --uncertainty_power 2.0 \
  --update_until 15000 \
  --test_iterations 3000 7000 15000 20000 25000 30000 \
  --save_iterations 7000 15000 30000 \
  --eval \
  2>&1 | tee "$LOG_PATH"
