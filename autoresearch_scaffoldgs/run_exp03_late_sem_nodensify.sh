#!/usr/bin/env bash
set -e

# ============================================================
# 实验：exp03_late_sem_nodensify
# 目的：验证关闭 densification（update_until=0）对语义分割和 RGB 重建的影响
# ============================================================

SOURCE_PATH="data/dronev4_2"
# 注意：output/dronev4_2_exp03_late_sem_nodensify 已存在且有结果，
# 如需重新训练，请手动修改 MODEL_PATH
MODEL_PATH="output/dronev4_2_exp03_late_sem_nodensify_reproduce"
LOG_PATH="autoresearch_scaffoldgs/logs/exp03_late_sem_nodensify_train.log"

mkdir -p "$(dirname "$LOG_PATH")"

python train.py \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --iterations 30000 \
  --start_semantic_iter 5000 \
  --mask_weight 0.2 \
  --mask_warmup 1000 \
  --mask_ramp 3000 \
  --knn_weight 0.05 \
  --knn_warmup 2000 \
  --knn_ramp 3000 \
  --knn_every 100 \
  --knn_offset 55 \
  --update_until 0 \
  --test_iterations 7000 10000 30000 \
  --save_iterations 7000 10000 30000 \
  --eval \
  2>&1 | tee "$LOG_PATH"
