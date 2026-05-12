#!/bin/bash
set -e

SOURCE="/mnt/data/liufengyang/data/myvideo"
OUTPUT_ROOT="/mnt/data/liufengyang/data/Scaffold-GSLFY/output"

echo "========================================"
echo "Batch evaluating all experiments on myvideo"
echo "========================================"

for exp_dir in "$OUTPUT_ROOT"/dronev4_2*/; do
    exp_name=$(basename "$exp_dir")

    # Skip gaussian-grouping (it's in another repo)
    if [[ "$exp_name" == "dronev4_2" ]]; then
        # This is the base model, but check if it has semantic head
        : # keep it
    fi

    # Find latest iteration
    latest_iter=$(ls "$exp_dir"point_cloud/ 2>/dev/null | grep iteration | sed 's/iteration_//' | sort -n | tail -1)
    if [ -z "$latest_iter" ]; then
        echo "[$exp_name] No checkpoint found, skipping"
        continue
    fi

    # Skip if myvideo result already exists
    if [ -f "$exp_dir/myvideo_iou_iter${latest_iter}.txt" ]; then
        echo "[$exp_name] Already evaluated at iter $latest_iter, skipping"
        continue
    fi

    echo "[$exp_name] Evaluating iter $latest_iter ..."
    cd /mnt/data/liufengyang/data/Scaffold-GSLFY
    conda run -n scaffold_gslfy python eval_myvideo.py "$exp_dir" "$latest_iter" > "$exp_dir/myvideo_eval_iter${latest_iter}.log" 2>&1 || {
        echo "[$exp_name] FAILED (see $exp_dir/myvideo_eval_iter${latest_iter}.log)"
    }
    echo "[$exp_name] Done"
done

echo "========================================"
echo "All evaluations finished"
echo "========================================"
