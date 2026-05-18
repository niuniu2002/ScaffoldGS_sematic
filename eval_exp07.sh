#!/bin/bash
# Evaluate exp07 after training completes

cd /mnt/data/liufengyang/data/Scaffold-GSLFY

MODEL="output/dronev4_2_exp07_anchor_region_w002"
ITER=30000

echo "=== Evaluating $MODEL at iteration $ITER ==="

# Use eval_myvideo.py (same as exp02 baseline evaluation)
conda run -n scaffold_gslfy python eval_myvideo.py \
  --model_path "$MODEL" \
  --iteration "$ITER" \
  --white_background \
  2>&1 | tee "$MODEL/myvideo_eval_iter${ITER}.log"

echo "=== Evaluation complete ==="
