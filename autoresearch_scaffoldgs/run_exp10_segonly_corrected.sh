#!/usr/bin/env bash
set -e

# ============================================================
# 实验：exp10_segonly_corrected
# 目的：修正 exp09 的配置错误，匹配 stage1 的 resolution=2 和 white_background=True，
#      并调整 focal_alpha=0.25 以更好地处理前景-背景不平衡。
# ============================================================

SOURCE_PATH="data/dronev4_2"
STAGE1_PATH="output/dronev4_2_stage1"
MODEL_PATH="output/dronev4_2_exp10_segonly_corrected"
LOG_PATH="autoresearch_scaffoldgs/logs/exp10_segonly_corrected_train.log"

mkdir -p "$(dirname "$LOG_PATH")"

# 复制 checkpoint（load_iteration 只从当前 model_path 读取）
if [ ! -d "$MODEL_PATH/point_cloud/iteration_15000" ]; then
    mkdir -p "$MODEL_PATH/point_cloud"
    cp -r "$STAGE1_PATH/point_cloud/iteration_15000" "$MODEL_PATH/point_cloud/"
    echo "[INFO] Copied stage1 iter 15000 checkpoint to $MODEL_PATH"
fi

/mnt/data/liufengyang/envs/scaffold_gslfy/bin/python train.py \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --load_iteration 15000 \
  --iterations 30000 \
  --resolution 2 \
  --white_background \
  --seg_only \
  --seg_only_reuse_head False \
  --start_semantic_iter 0 \
  --mask_weight 0.4 \
  --mask_warmup 2000 \
  --mask_ramp 5000 \
  --knn_weight 0.05 \
  --knn_every 100 \
  --knn_offset 55 \
  --knn_warmup 2000 \
  --knn_ramp 5000 \
  --focal_alpha 0.25 \
  --focal_gamma 2.0 \
  --uncertainty_min 0.1 \
  --uncertainty_power 2.0 \
  --test_iterations 5000 10000 15000 20000 25000 30000 \
  --save_iterations 10000 20000 30000 \
  --eval \
  2>&1 | tee "$LOG_PATH"
