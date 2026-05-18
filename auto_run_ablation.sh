#!/bin/bash
# ============================================
# Auto-run ablation after exp18 finishes
# ============================================

PYTHON="/mnt/data/liufengyang/envs/scaffold_gslfy/bin/python"
SCENE="data/dronev4_2"
GPU="-1"

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

ALPHAS=(0.15 0.18 0.20 0.28 0.30)

VOTE_ON_ARGS=(
  --anchor_fg_weight 0.1
  --anchor_fg_start_iter 5000
  --anchor_fg_every 10
  --anchor_fg_ratio_thr 0.3
  --anchor_fg_max_samples 4096
  --anchor_fg_detach_xyz
  --anchor_fg_ramp 1000
)

# ---- Run 10 experiments sequentially ----
for alpha in "${ALPHAS[@]}"; do
  # 1) vote off
  EXP_NAME="dronev4_2_focalalpha${alpha}_voteoff"
  echo "[$(date)] ======================================="
  echo "[$(date)] Starting: ${EXP_NAME}"
  "$PYTHON" train.py -s "${SCENE}" -m "output/${EXP_NAME}" --gpu "${GPU}" "${BASE_ARGS[@]}" --focal_alpha "${alpha}" --anchor_fg_weight 0.0
  echo "[$(date)] Finished: ${EXP_NAME}"

  # 2) vote on
  EXP_NAME="dronev4_2_focalalpha${alpha}_voton"
  echo "[$(date)] ======================================="
  echo "[$(date)] Starting: ${EXP_NAME}"
  "$PYTHON" train.py -s "${SCENE}" -m "output/${EXP_NAME}" --gpu "${GPU}" "${BASE_ARGS[@]}" --focal_alpha "${alpha}" "${VOTE_ON_ARGS[@]}"
  echo "[$(date)] Finished: ${EXP_NAME}"
done

echo "[$(date)] All 10 experiments completed!"
