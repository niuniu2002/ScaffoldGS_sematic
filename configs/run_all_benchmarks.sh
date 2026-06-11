#!/bin/bash
# Run all three benchmark experiments sequentially.
# Adjust GPU assignment or run in parallel as needed.

set -e

echo "========================================="
echo "Running all benchmark experiments"
echo "========================================="

# 1. dronev4_2 - Best Balance
echo "[1/3] dronev4_2 baseline..."
bash configs/run_dronev4_2_baseline.sh

# 2. lfy/colmap_scene
echo "[2/3] lfy/colmap_scene baseline..."
bash configs/run_colmap_scene_baseline.sh

# 3. SW_scenes/scene_01
# NOTE: If scene_01 has no train/test split, generate it first:
# python configs/generate_split.py /mnt/data/liufengyang/data/dataset/SW_scenes/scene_01 --ratio 0.8
echo "[3/3] SW_scenes/scene_01 baseline..."
bash configs/run_scene_01_baseline.sh

echo "========================================="
echo "All benchmarks completed!"
echo "========================================="
