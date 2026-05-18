#!/bin/bash
# ============================================
# Focal Alpha + Anchor Voting 消融实验脚本
# 基准: exp06 (late_sem, stop_densify@15000)
# 变量: focal_alpha x anchor_fg_weight(on/off)
# ============================================

SCENE="data/dronev4_2"
GPU="-1"

# ---- 基准参数 (from exp06) ----
BASE_ARGS=(
  --resolution 2
  --white_background
  --eval
  --appearance_dim 32
  --start_semantic_iter 5000
  --mask_weight 0.2
  --mask_warmup 1000
  --mask_ramp 3000
  --knn_weight 0.05
  --knn_every 100
  --knn_offset 55
  --knn_warmup 2000
  --knn_ramp 3000
  --focal_gamma 2.0
  --update_until 15000
  --test_iterations 7000 30000
  --save_iterations 7000 30000
)

# focal_alpha 取值列表
ALPHAS=(0.15 0.18 0.20 0.28 0.30)

# 投票机制参数 (打开时)
VOTE_ON_ARGS=(
  --anchor_fg_weight 0.1
  --anchor_fg_start_iter 5000
  --anchor_fg_every 10
  --anchor_fg_ratio_thr 0.3
  --anchor_fg_max_samples 4096
  --anchor_fg_detach_xyz
  --anchor_fg_ramp 1000
)

# ============================================
# 顺序执行 10 个实验
# ============================================

for alpha in "${ALPHAS[@]}"; do
  # ---- 1) 投票关闭 ----
  EXP_NAME="dronev4_2_focalalpha${alpha}_voteoff"
  echo "========================================"
  echo "Starting: ${EXP_NAME}"
  echo "========================================"
  python train.py -s "${SCENE}" \
    -m "output/${EXP_NAME}" \
    --gpu "${GPU}" \
    "${BASE_ARGS[@]}" \
    --focal_alpha "${alpha}" \
    --anchor_fg_weight 0.0
  echo "Finished: ${EXP_NAME}"
  echo ""

  # ---- 2) 投票打开 ----
  EXP_NAME="dronev4_2_focalalpha${alpha}_voton"
  echo "========================================"
  echo "Starting: ${EXP_NAME}"
  echo "========================================"
  python train.py -s "${SCENE}" \
    -m "output/${EXP_NAME}" \
    --gpu "${GPU}" \
    "${BASE_ARGS[@]}" \
    --focal_alpha "${alpha}" \
    "${VOTE_ON_ARGS[@]}"
  echo "Finished: ${EXP_NAME}"
  echo ""
done

echo "All 10 experiments completed!"
