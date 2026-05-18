#!/usr/bin/env bash
set -e

# ============================================================
# 实验：exp09_segonly_long
# 目的：长时 seg_only 两阶段训练。第一阶段用 stage1 @ 15000 (PSNR 25.34) 的纯 RGB
#       checkpoint，冻结几何，重新初始化语义头，用大权重长时训练语义分支。
# ============================================================

SOURCE_PATH="data/dronev4_2"
STAGE1_PATH="output/dronev4_2_stage1"
MODEL_PATH="output/dronev4_2_exp09_segonly_long"
LOG_PATH="autoresearch_scaffoldgs/logs/exp09_segonly_long_train.log"

mkdir -p "$(dirname "$LOG_PATH")"

# 复制 checkpoint（load_iteration 只从当前 model_path 读取）
if [ ! -d "$MODEL_PATH/point_cloud/iteration_15000" ]; then
    mkdir -p "$MODEL_PATH/point_cloud"
    cp -r "$STAGE1_PATH/point_cloud/iteration_15000" "$MODEL_PATH/point_cloud/"
    echo "[INFO] Copied stage1 iter 15000 checkpoint to $MODEL_PATH"
fi

python train.py \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --load_iteration 15000 \
  --iterations 30000 \
  --seg_only \
  --seg_only_reuse_head False \
  --start_semantic_iter 0 \
  --mask_weight 0.4 \
  --mask_warmup 2000 \
  --mask_ramp 5000 \
  --knn_weight 0.1 \
  --knn_every 50 \
  --knn_offset 55 \
  --knn_warmup 2000 \
  --knn_ramp 5000 \
  --focal_alpha 0.5 \
  --focal_gamma 2.0 \
  --uncertainty_min 0.1 \
  --uncertainty_power 2.0 \
  --test_iterations 10000 20000 30000 \
  --save_iterations 10000 20000 30000 \
  --eval \
  2>&1 | tee "$LOG_PATH"
