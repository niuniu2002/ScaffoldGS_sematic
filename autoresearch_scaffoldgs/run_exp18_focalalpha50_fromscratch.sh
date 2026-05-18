#!/usr/bin/env bash
set -e

# ============================================================
# 实验：exp18_focalalpha50_fromscratch
# 目的：基于 exp06 最佳基准，调整 focal_alpha 从 0.25 到 0.5，
#      测试更平衡的前景/背景权重是否能提升 FG IoU 和整体 mIoU。
# ============================================================

SOURCE_PATH="data/dronev4_2"
MODEL_PATH="output/dronev4_2_exp18_focalalpha50_fromscratch"
LOG_PATH="autoresearch_scaffoldgs/logs/exp18_focalalpha50_fromscratch_train.log"

mkdir -p "$(dirname "$LOG_PATH")"

/mnt/data/liufengyang/envs/scaffold_gslfy/bin/python train.py \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --iterations 30000 \
  --resolution 2 \
  --white_background \
  --start_semantic_iter 5000 \
  --mask_weight 0.2 \
  --mask_warmup 1000 \
  --mask_ramp 3000 \
  --knn_weight 0.05 \
  --knn_every 100 \
  --knn_offset 55 \
  --knn_warmup 2000 \
  --knn_ramp 3000 \
  --focal_alpha 0.5 \
  --focal_gamma 2.0 \
  --uncertainty_min 0.1 \
  --uncertainty_power 2.0 \
  --update_until 15000 \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000 \
  --eval \
  2>&1 | tee "$LOG_PATH"
