#!/bin/bash
# Auto-evaluate exp07 when training completes

MODEL="output/dronev4_2_exp07_anchor_region_w002"
ITER=30000
LOG="/mnt/data/liufengyang/data/Scaffold-GSLFY/output/dronev4_2_exp07.log"

echo "Waiting for training to complete..."
echo "Monitoring $LOG for '30000/30000'..."

while true; do
    if grep -q "30000/30000" "$LOG" 2>/dev/null; then
        echo "Training completed! Starting evaluation..."
        break
    fi
    sleep 60
done

cd /mnt/data/liufengyang/data/Scaffold-GSLFY

# Evaluate using eval_myvideo.py (same as baseline)
conda run -n scaffold_gslfy python eval_myvideo.py \
  --model_path "$MODEL" \
  --iteration "$ITER" \
  --white_background \
  2>&1 | tee "$MODEL/myvideo_eval_iter${ITER}.log"

echo "=== Evaluation complete ==="
