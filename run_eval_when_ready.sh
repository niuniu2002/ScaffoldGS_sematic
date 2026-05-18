#!/bin/bash
# Wait for exp07 training to complete, then evaluate

cd /mnt/data/liufengyang/data/Scaffold-GSLFY

MODEL="output/dronev4_2_exp07_anchor_region_w002"
ITER=30000
LOG="output/dronev4_2_exp07.log"

echo "Waiting for training to complete..."
while ! grep -q "30000/30000" "$LOG" 2>/dev/null; do
    iter=$(grep -oP '\d+(?=/30000)' "$LOG" 2>/dev/null | tail -1)
    echo "Current iteration: $iter / 30000"
    sleep 60
done

echo ""
echo "=== Training complete! Evaluating... ==="

# Evaluate
conda run -n scaffold_gslfy python eval_myvideo.py \
  --model_path "$MODEL" \
  --iteration "$ITER" \
  --white_background \
  2>&1 | tee "$MODEL/myvideo_eval_iter${ITER}.log"

echo "=== Done ==="
