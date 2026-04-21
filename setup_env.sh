#!/bin/bash
# Setup script for Scaffold-GSLFY environment
# Run after PyTorch is installed

set -e

ENV="scaffold_gslfy"
source /mnt/data/liufengyang/envs/miniforge3/etc/profile.d/conda.sh
conda activate $ENV

echo "=== Installing Python dependencies ==="
pip install \
    einops \
    tqdm \
    plyfile \
    wandb \
    lpips \
    laspy \
    opencv-python \
    scipy \
    imageio \
    colorama \
    tqdm

echo "=== Installing pytorch-scatter (for PyTorch 2.1 + CUDA 12.1) ==="
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

echo "=== Building CUDA submodules ==="
cd submodules/diff-gaussian-rasterization
pip install -e .
cd ../simple-knn
pip install -e .

echo "=== Environment setup complete ==="
conda run -n $ENV python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
