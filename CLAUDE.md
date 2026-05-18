# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of [Scaffold-GS](https://github.com/city-super/Scaffold-GS) extended with **semantic segmentation** capabilities for **3D reconstruction, semantic rendering, and volume estimation of Solidago canadensis (加拿大一枝黄花)** from UAV aerial imagery.

The base Scaffold-GS framework is an **anchor-based 3D Gaussian splatting** method. Each anchor point predicts view-dependent Gaussian attributes (color, opacity, scale, rotation) via lightweight MLPs. This project adds a **segmentation head** on top of the existing anchor features to predict semantic logits, which are rendered into 2D segmentation masks through the differentiable rasterizer.

### What This Project Is / Is Not

- **This is NOT a from-scratch 3DGS implementation.** It inherits the full Scaffold-GS geometry engine (anchors, offsets, MLPs for view-dependent attributes, densification).
- **This IS an extension that adds a semantic segmentation branch** to an existing Scaffold-GS codebase. The core contribution is the segmentation head and its training protocol, not the 3D representation itself.

### Semantic Supervision Signal

- **2D masks / pseudo-labels are supervisory signals**, NOT direct inputs to the semantic branch.
- The semantic branch takes **anchor features (`_anchor_feat`, dim=32)** as input.
- It outputs **anchor-level or Gaussian-level semantic logits**.
- These logits are fed into the differentiable rasterizer along with Gaussian positions, opacities, and covariances to produce a **rendered 2D mask**.
- The rendered mask is then compared against the GT/pseudo-label mask via focal loss + dice loss.
- This is a **render-and-compare** paradigm, not a feed-forward CNN that directly consumes images or masks.

## Environment Setup

Dependencies are managed via Conda. CUDA extensions must be built from source.

```bash
# Create environment
conda env create --file environment.yml
conda activate scaffold_gs

# Build CUDA submodules (diff-gaussian-rasterization, simple-knn)
cd submodules/diff-gaussian-rasterization && pip install -e .
cd ../simple-knn && pip install -e .
```

The `setup_env.sh` script provides an alternative setup path for a specific pre-existing conda environment (`scaffold_gslfy`) with PyTorch 2.1 + CUDA 12.1.

## Current Experiment Priority

The following experiments are ordered by current effectiveness and recommended next steps:

1. **`exp01_sem_ramp`** (semantic ramp): Currently the most effective configuration. Semantic loss ramps up gradually alongside geometry training.
2. **`exp02_late_sem`** (late semantic): Performance is roughly on par with `sem_ramp`. Semantic training starts later in the schedule.
3. **`late_sem + no_densify`**: **Next highest-priority experiment to run.** This disables densification during the semantic training phase to test the hypothesis that anchor splitting/merging perturbs geometry and degrades RGB reconstruction quality (PSNR).

**Rationale for `no_densify`:** If PSNR drops when semantic loss is active, densification may be the culprit. Freezing the anchor distribution while training the segmentation head isolates whether the semantic loss itself harms geometry or whether the interaction with densification does.

## Common Training Commands

### `sem_ramp` Baseline (Joint Training, Semantic Loss Ramps Early)

```bash
python train.py -s data/<scene> -m outputs/exp01_sem_ramp \
  --start_semantic_iter 5000 \
  --mask_weight 0.2 --knn_weight 0.05 \
  --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25
```

### `late_sem` (Semantic Loss Starts Later)

```bash
python train.py -s data/<scene> -m outputs/exp02_late_sem \
  --start_semantic_iter 15000 \
  --mask_weight 0.2 --knn_weight 0.05 \
  --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25
```

### `late_sem + no_densify` (Late Semantic, No Densification During Semantic Phase)

```bash
python train.py -s data/<scene> -m outputs/exp03_late_sem_no_densify \
  --start_semantic_iter 15000 \
  --update_until 15000 \
  --mask_weight 0.2 --knn_weight 0.05 \
  --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25
```

> Note: `--update_until 15000` stops densification exactly when semantic loss starts. Adjust this value to match your `--start_semantic_iter`.

### Two-Stage Segmentation Training (`seg_only`, Freeze Geometry)

```bash
python train.py -s data/<scene> -m outputs/exp04_seg_only \
  --seg_only \
  --load_iteration 30000 --iterations 10000 \
  --start_semantic_iter 0 \
  --mask_weight 0.2 --knn_weight 0.05
```

> **Important:** See [Loading Checkpoints](#loading-checkpoints) for `--load_iteration` behavior and pitfalls.

### Single Scene (Quick Test via Shell Script)

```bash
bash single_train.sh
```

> Edit variables inside `single_train.sh` before running.

Batch training scripts for public datasets (Mip-NeRF 360, Tanks & Temples) are in `scripts/` (e.g., `train_mip360.sh`, `train_tnt.sh`).

## Loading Checkpoints

The `--load_iteration` argument is used to resume from or initialize from a previous checkpoint. There are critical constraints:

1. **Scope:** `--load_iteration` loads from the **current** `--model_path` (`-m`) directory. It does **NOT** automatically pull checkpoints from a different experiment directory.
2. **Directory Structure Expected:**
   ```
   <model_path>/
     point_cloud/
       iteration_<load_iteration>/
         point_cloud.ply
   ```
3. **If starting a new experiment from an old checkpoint:** You **must manually copy** the old experiment's `point_cloud/iteration_<N>/` directory into the new experiment's `--model_path` before running. Example:
   ```bash
   mkdir -p outputs/new_exp/point_cloud
   cp -r outputs/old_exp/point_cloud/iteration_30000 outputs/new_exp/point_cloud/
   python train.py -s data/<scene> -m outputs/new_exp --load_iteration 30000 ...
   ```
4. **Failure mode:** If the `point_cloud.ply` is missing at the expected path, training will crash with a file-not-found error.

## High-Level Architecture

### Anchor-Based Gaussian Representation

The scene is represented by anchor points (`_anchor`) stored as explicit tensors, not a neural network. Each anchor has:
- A feature vector (`_anchor_feat`, dim=32)
- Learnable offsets (`_offset`) that spawn `n_offsets` (default 10) child Gaussians per anchor
- Global scales/rotations

View-dependent Gaussian attributes (color, opacity, covariance, segmentation) are predicted on-the-fly by MLPs conditioned on viewing direction and distance. This happens in `gaussian_renderer/__init__.py:generate_neural_gaussians()`.

### Core Rendering Pipeline

`gaussian_renderer/__init__.py` is the heart of the system:

1. **View frustum filtering**: `prefilter_voxel()` culls anchors outside the camera view.
2. **Neural Gaussian generation**: `generate_neural_gaussians()` runs the MLPs to produce per-Gaussian position, color, opacity, scale, rotation, and segmentation.
3. **Two rasterizer passes**:
   - First pass renders RGB with dataset background color.
   - Second pass renders the segmentation mask with a black background. Crucially, mask rendering uses `opacities = opacity.detach()` to prevent mask loss gradients from backpropagating into the opacity MLP and interfering with RGB reconstruction.

### Semantic Segmentation Extension

- **SegmentationHead** (`scene/gaussian_model.py`): A 3-layer residual MLP with 128 hidden units and LayerNorm. It consumes anchor features and outputs logits.
- **Per-anchor vs per-Gaussian**: `use_per_gaussian_seg=False` shares one seg value across all offsets of an anchor (via `repeat_interleave`). `use_per_gaussian_seg=True` predicts independent logits per offset Gaussian.
- **Mask supervision**: Binary focal loss + dice loss on the rendered mask, with optional uncertainty-based pixel weighting for soft/pseudo labels.
- **KNN spatial consistency**: Every `knn_every` iterations, a symmetric BCE loss in probability space encourages neighboring anchors to have similar segmentation values. This avoids the "push-to-0.5" problem of MSE.

### Two-Stage Training (`--seg_only`)

When `seg_only=True` (`train.py`):
1. The segmentation head is reinitialized with the deep 3-layer architecture (replacing whatever was loaded from checkpoint).
2. All explicit parameters (`_anchor`, `_offset`, etc.) and all MLPs except `mlp_segmentation` are frozen.
3. Densification is disabled.
4. The optimizer is replaced with Adam tracking only `mlp_segmentation` parameters (~55K params).

### Densification & Anchor Management

Anchor densification happens in `scene/gaussian_model.py` via `training_statis()` and `adjust_anchor()`. It is controlled by:
- `update_until`: iteration at which densification stops.
- `update_from`, `update_interval`, `start_stat`: scheduling.
- `densify_grad_threshold`, `min_opacity`, `success_threshold`: criteria for splitting/merging anchors.

In two-stage mode (`seg_only`), densification is automatically disabled.

## Data Layout

Scenes follow COLMAP structure:

```
data/my_scene/
  images/
    IMG_001.jpg
    ...
  masks/              # optional: binary segmentation GT (0=bg, 255=fg)
    IMG_001.png
    ...
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

The dataset reader (`scene/dataset_readers.py`) automatically detects `masks/` and pairs masks with images by basename.

## Key Files

| File | Role |
|---|---|
| `train.py` | Main training loop, loss computation (RGB + mask + KNN), two-stage logic |
| `scene/gaussian_model.py` | Anchor Gaussian model, MLP definitions, densification, checkpoint save/load |
| `gaussian_renderer/__init__.py` | Neural Gaussian generation, dual-pass rasterization (RGB + mask) |
| `scene/dataset_readers.py` | COLMAP/Blender data loading, optional mask loading |
| `arguments/__init__.py` | All hyperparameters and CLI arguments |
| `eval_myvideo.py` | Standalone segmentation IoU evaluation script |
| `tools/render.py` | Standard RGB + mask rendering |
| `tools/render1.py` | Rendering with semantic heatmap overlay |

## Claude Code Operating Rules

When modifying code in this repository, adhere to the following constraints:

1. **Do NOT modify the CUDA rasterizer** (`submodules/diff-gaussian-rasterization/`) unless there is an explicit, justified reason. The rasterizer is a stable, compiled dependency. Changes here require recompilation and can introduce hard-to-debug rendering artifacts.
2. **Do NOT alter the COLMAP data structure or dataset reader conventions.** The `scene/dataset_readers.py` loader assumes standard COLMAP output (cameras.bin, images.bin, points3D.bin) and an optional `masks/` folder paired by basename. Deviations break data loading for all scenes.
3. **When modifying loss functions or training schedules, monitor BOTH mIoU and PSNR.** A change that boosts segmentation accuracy but collapses RGB quality is not acceptable.
4. **mIoU is the PRIMARY metric; PSNR is the GUARDRAIL metric.** Optimize for segmentation IoU first, but do not allow PSNR to degrade significantly below the no-segmentation baseline.
5. **Do NOT trade RGB reconstruction quality for semantic performance.** Semantic gains achieved by destroying geometry (e.g., allowing densification to run unchecked, weakening opacity regularization) are invalid. If PSNR drops sharply, the modification is harmful regardless of mIoU improvement.

## Notable Implementation Details

- The rasterizer (`submodules/diff-gaussian-rasterization`) is a modified CUDA extension that accepts pre-computed colors and does not use spherical harmonics (`sh_degree=1` is a placeholder).
- `torch-scatter` is used for anchor-level aggregation during densification statistics.
- `lpips` is used for evaluation but not training loss.
- Training logs are saved to `outputs/` (or the path specified by `-m`). Each run backs up the source code to the output directory for reproducibility.
- TensorBoard logging is available if `torch.utils.tensorboard` is installed; masks are logged every 1000 iterations.
