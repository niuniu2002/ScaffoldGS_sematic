#!/bin/bash
set -e

# ============================================================================
# Semantic Weight Heatmap 辅助语义分割 — 对比实验脚本
# 
# 前置条件：
#   1. /mnt/data/liufengyang/data/dronev4_2/semantic_weights/ 已生成完毕
#   2. 已有 baseline 几何模型（如 dronev4_2_highpsnr iter 30000）
#
# 实验分组：
#   A. Two-Stage Baseline（不加语义权重）
#   B. Two-Stage + Semantic Weight（hard 策略）
#   C. Two-Stage + Semantic Weight（smooth 策略）
#   D. Full Training + Semantic Weight（从头联合训练）
# ============================================================================

SOURCE="/mnt/data/liufengyang/data/dronev4_2"
OUTPUT_ROOT="/mnt/data/liufengyang/data/Scaffold-GSLFY/output"

# --- 基线模型配置（用你已有的最佳几何） ---
BASE_MODEL="$OUTPUT_ROOT/dronev4_2_highpsnr"
BASE_ITER=30000

# 通用参数（与 exp06/twostage 一致）
COMMON_ARGS="
  --source_path $SOURCE
  --resolution 2
  --white_background
  --appearance_dim 32
  --start_semantic_iter 0
  --mask_weight 0.2
  --knn_weight 0.05
  --knn_every 100
  --knn_offset 55
  --focal_alpha 0.25
"

# ============================================================================
# 实验 A：Two-Stage Baseline（不加语义权重，复刻你的 twostage 结果）
# ============================================================================
echo "========================================"
echo "[Exp A] Two-Stage Baseline"
echo "========================================"

EXP_A="$OUTPUT_ROOT/dronev4_2_twostage_baseline"
mkdir -p "$EXP_A/point_cloud"
cp -r "$BASE_MODEL/point_cloud/iteration_$BASE_ITER" "$EXP_A/point_cloud/"

python train.py \
  -s "$SOURCE" -m "$EXP_A" \
  $COMMON_ARGS \
  --seg_only \
  --load_iteration $BASE_ITER \
  --iterations 10000

# ============================================================================
# 实验 B：Two-Stage + Semantic Weight（hard 策略）
# ============================================================================
echo "========================================"
echo "[Exp B] Two-Stage + Semantic Weight (hard)"
echo "========================================"

EXP_B="$OUTPUT_ROOT/dronev4_2_twostage_semweight_hard"
mkdir -p "$EXP_B/point_cloud"
cp -r "$BASE_MODEL/point_cloud/iteration_$BASE_ITER" "$EXP_B/point_cloud/"

python train.py \
  -s "$SOURCE" -m "$EXP_B" \
  $COMMON_ARGS \
  --seg_only \
  --load_iteration $BASE_ITER \
  --iterations 10000 \
  --use_semantic_weight \
  --sem_weight_strategy hard \
  --sem_weight_high 0.7 \
  --sem_weight_low 0.1 \
  --sem_weight_boost 2.0

# ============================================================================
# 实验 C：Two-Stage + Semantic Weight（smooth 策略）
# ============================================================================
echo "========================================"
echo "[Exp C] Two-Stage + Semantic Weight (smooth)"
echo "========================================"

EXP_C="$OUTPUT_ROOT/dronev4_2_twostage_semweight_smooth"
mkdir -p "$EXP_C/point_cloud"
cp -r "$BASE_MODEL/point_cloud/iteration_$BASE_ITER" "$EXP_C/point_cloud/"

python train.py \
  -s "$SOURCE" -m "$EXP_C" \
  $COMMON_ARGS \
  --seg_only \
  --load_iteration $BASE_ITER \
  --iterations 10000 \
  --use_semantic_weight \
  --sem_weight_strategy smooth \
  --sem_weight_smooth_peak 0.5

# ============================================================================
# 实验 D：Full Training + Semantic Weight（从头联合训练，验证联合训练效果）
#   注意：这个实验时间较长（~8-10小时），建议 A/B/C 有结果后再决定是否跑
# ============================================================================
# echo "========================================"
# echo "[Exp D] Full Training + Semantic Weight"
# echo "========================================"
#
# EXP_D="$OUTPUT_ROOT/dronev4_2_full_semweight"
# python train.py \
#   -s "$SOURCE" -m "$EXP_D" \
#   --resolution 2 \
#   --white_background \
#   --appearance_dim 32 \
#   --start_semantic_iter 5000 \
#   --update_until 15000 \
#   --mask_weight 0.2 \
#   --knn_weight 0.05 \
#   --knn_every 100 \
#   --knn_offset 55 \
#   --focal_alpha 0.25 \
#   --iterations 30000 \
#   --use_semantic_weight \
#   --sem_weight_strategy hard

echo "========================================"
echo "All experiments finished!"
echo "========================================"
