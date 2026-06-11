# Scaffold-GS-Semantic: Anchor-Based 3D Gaussian Splatting with Semantic Segmentation

> **An extended fork of [Scaffold-GS](https://github.com/city-super/Scaffold-GS) for UAV aerial-image 3D reconstruction and semantic segmentation.**
>
> This project targets **3D semantic segmentation of Solidago canadensis (加拿大一枝黄花)** from UAV imagery, enabling downstream applications such as volume estimation, precision spraying, and quantitative governance.

[![Scaffold-GS](https://img.shields.io/badge/Base-Scaffold--GS-blue)](https://github.com/city-super/Scaffold-GS)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CUDA](https://img.shields.io/badge/CUDA-11.6%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)

---

## What is Different?

| Feature | Scaffold-GS (Original) | This Fork |
|---|---|---|
| Semantic Segmentation | ❌ | ✅ Binary & Multi-class |
| Segmentation Head | — | ✅ 3-layer residual MLP (128 hidden, LayerNorm) |
| Per-Gaussian Seg | — | ✅ Optional independent logits per offset |
| KNN Consistency Loss | — | ✅ Symmetric BCE (binary) / Symmetric KL (multi-class) |
| Opacity Detach | — | ✅ Mask pass uses `opacity.detach()` to protect geometry |
| Multi-class Rendering | — | ✅ CUDA rasterizer supports up to 128 channels |
| Two-Stage Training | — | ✅ `seg_only` freeze geometry, train head only |
| Auto Two-Stage | — | ✅ Automatic geometry pretraining + semantic fine-tuning |
| Label-Shift Robustness | — | ✅ Uncertainty-aware pixel weighting |

---

## Benchmark Datasets

We evaluate on **three UAV-captured scenes** with binary segmentation masks:

| Dataset | Images | Train / Test | Resolution | Mask Source | Path |
|---|---|---|---|---|---|
| `dronev4_2` | 333 | 266 / 67 | — | SAM + Human | `data/dronev4_2` |
| `lfy/colmap_scene` | 200 | — / 40 (test_list) | 1920×1080 | ISAT annotation | `data/lfy/colmap_scene` |
| `SW_scenes/scene_01` | 479 (sparse/0) | TBD | 3840×2160 | SegmentationClass | `data/SW_scenes/scene_01` |

### Download

All benchmark datasets and 2D segmentation weights are packaged here:

🔗 **Quark Netdisk**: https://pan.quark.cn/s/0833e64fe708

Contents include:
- `dronev4_2/` — UAV aerial images + SAM masks + human annotations
- `lfy/colmap_scene/` — COLMAP scene with ISAT-labeled masks
- `SW_scenes/scene_01/` — High-res UAV scene (3840×2160) + segmentation masks
- 2D segmentation model weights (for pseudo-label generation)

### Data Layout

Each scene follows standard COLMAP structure:

```
scene_name/
├── images/
│   ├── IMG_001.jpg
│   └── ...
├── masks/              # optional: binary masks (0=bg, 255=fg) or class indices
│   ├── IMG_001.png
│   └── ...
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── train_list.txt      # optional: one basename per line
└── test_list.txt       # optional: one basename per line
```

---

## Installation

```bash
# 1. Clone
git clone https://github.com/niuniu2002/ScaffoldGS_sematic.git
cd ScaffoldGS_sematic

# 2. Create env
conda env create --file environment.yml
conda activate scaffold_gs

# 3. Build CUDA submodules
cd submodules/diff-gaussian-rasterization && pip install -e .
cd ../simple-knn && pip install -e .
```

---

## Quick Start

### Binary Segmentation (Single Scene)

```bash
python train.py \
  -s /path/to/dronev4_2 \
  -m output/dronev4_2_baseline \
  --start_semantic_iter 5000 \
  --mask_weight 0.1 \
  --mask_warmup 1000 \
  --mask_ramp 3000 \
  --knn_weight 0.02 \
  --knn_warmup 2000 \
  --knn_ramp 3000 \
  --knn_every 100 \
  --knn_offset 55 \
  --focal_alpha 0.25 \
  --update_until 0 \
  --iterations 30000
```

### Multi-class Segmentation

```bash
python train.py \
  -s /path/to/scene \
  -m output/scene_multiclass \
  --num_classes 8 \
  --start_semantic_iter 5000 \
  --mask_weight 0.1 \
  --knn_weight 0.02 \
  --focal_alpha 0.25 \
  --update_until 0 \
  --iterations 30000
```

### Evaluation

```bash
# Evaluation is integrated into training; test-set PSNR + mIoU are reported
# automatically at scheduled testing iterations.

# For standalone multi-class evaluation:
python eval_multiclass.py \
  --model_path output/scene_multiclass \
  --source_path /path/to/scene \
  --iteration 30000 \
  --num_classes 8
```

---

## Experiment Configurations

All scripts below are located in `configs/` and can be run directly:

```bash
bash configs/run_dronev4_2_baseline.sh
```

### Verified Baselines on `dronev4_2`

All metrics are reported on the **test set** (67 images).

#### Test Set PSNR & mIoU

| Config | `start_sem` | `update_until` | `focal_alpha` | `mask_weight` | ITER 7000 PSNR | ITER 7000 mIoU | ITER 30000 PSNR | ITER 30000 mIoU |
|---|---|---|---|---|---|---|---|---|
| **exp01_sem_ramp** | 0 | 15000 | 0.75 | 0.2 | 24.88 | 0.7809 | 24.71 | 0.7880 |
| **exp02_late_sem** | 5000 | 15000 | 0.75 | 0.2 | **24.90** | 0.7752 | 24.68 | 0.7891 |
| **exp03_nodensify** | 5000 | 0 | 0.25 | 0.2 | 24.83 | 0.7767 | 24.63 | 0.7885 |
| **exp03_scratch** | 5000 | 0 | 0.25 | 0.2 | 24.03 | 0.6477 | 24.41 | 0.7570 |
| **exp04_stop5000** | 5000 | 5000 | 0.25 | 0.2 | 24.47 | 0.6561 | 24.79 | 0.7617 |
| **exp05_stop10000** | 5000 | 10000 | 0.25 | 0.2 | 24.47 | 0.6661 | 24.87 | 0.7741 |
| **exp06_stop15000** | 5000 | 15000 | 0.25 | 0.2 | 24.45 | 0.6631 | **25.04** | 0.7778 |
| **🔥 mw04_nodetach** | 5000 | 15000 | 0.25 | **0.4** | — | — | 23.91 | **0.8300** |

#### Final Rendering Quality (Test Set)

| Config | SSIM ↑ | PSNR ↑ | LPIPS ↓ | FPS ↑ | Avg Visible Count |
|---|---|---|---|---|---|
| exp01_sem_ramp | 0.6902 | 24.7041 | 0.2969 | 29.23 | 390,896 |
| exp02_late_sem | 0.6886 | 24.6374 | 0.2964 | 46.36 | 390,532 |
| **exp03_nodensify** | 0.6872 | 24.6282 | 0.2961 | **85.20** | 386,277 |
| exp03_scratch | 0.6403 | 24.4052 | 0.3884 | **143.15** | 174,975 |
| exp04_stop5000 | 0.6743 | 24.7904 | 0.3416 | 118.80 | 258,774 |
| exp05_stop10000 | 0.6915 | 24.8688 | 0.2997 | 105.31 | 396,850 |
| **exp06_stop15000** | **0.7017** | **25.0408** | **0.2870** | 92.72 | 474,184 |
| **🔥 mw04_nodetach** | — | 23.91 | — | — | — |

**Key Insights** (from ablation studies):
- **🔥 mw04_nodetach** achieves the highest test mIoU (**0.8300**) by allowing mask gradients into the opacity MLP (`no_opacity_detach`) with a stronger mask weight (0.4).
- **exp06_stop15000** achieves the best rendering quality among detached configs (PSNR 25.04, SSIM 0.702, LPIPS 0.287).
- **exp03_nodensify** offers the best quality-speed balance (PSNR 24.63, FPS 85.2) by disabling densification on a pre-trained model.
- **exp01/exp02** use `focal_alpha=0.75` and suffer from slower inference (~29–46 FPS) compared to `focal_alpha=0.25` variants.
- `update_until` strongly affects model size and speed; earlier stop = smaller model but lower quality.

### Recommended Configurations

#### 🔥 A. SOTA: No-Detach Joint Training (Recommended)
```bash
--start_semantic_iter 5000 \
--mask_weight 0.4 \
--knn_weight 0.05 \
--knn_every 100 --knn_offset 55 \
--focal_alpha 0.25 \
--update_until 15000 \
--no_opacity_detach \
--iterations 30000
```
> **Test mIoU 0.8300** | PSNR 23.91 | BG IoU 0.9921
> Allowing mask gradients into the opacity MLP (`no_opacity_detach`) with a higher `mask_weight` pushes segmentation accuracy to SOTA while preserving acceptable RGB quality.

#### B. Quality-Speed Balance (No Densify)
```bash
--start_semantic_iter 5000 \
--mask_weight 0.1 --mask_warmup 1000 --mask_ramp 3000 \
--knn_weight 0.02 --knn_warmup 2000 --knn_ramp 3000 \
--knn_every 100 --knn_offset 55 \
--focal_alpha 0.25 \
--update_until 0 \
--iterations 30000
```
> Test mIoU ≈ 0.79, FPS 85.2. Best for real-time inference.

#### C. Two-Stage (`seg_only`) — Freeze Geometry
```bash
--seg_only \
--load_iteration 30000 \
--start_semantic_iter 0 \
--mask_weight 0.2 \
--knn_weight 0.05 \
--iterations 10000
```
> **Note**: `seg_only` consistently underperforms joint training. Prefer joint training with controlled `mask_weight`.

---

## Full Parameter Reference

### Semantic Parameters

| Parameter | Default | Description |
|---|---|---|
| `--num_classes` | `1` | `1`=binary (sigmoid), `>=2`=multi-class (softmax) |
| `--use_per_gaussian_seg` | `False` | Independent seg logit per offset Gaussian |
| `--start_semantic_iter` | `7000` | Hard warmup: skip mask/knn losses before this iter |
| `--mask_weight` | `0.01` | Weight for mask loss (focal + dice) |
| `--mask_warmup` | `0` | Iterations to keep mask weight at 0 |
| `--mask_ramp` | `0` | Iterations to linearly ramp to full weight |
| `--mask_weight_final` | `-1` | Final mask weight for decay schedule (-1=disabled) |
| `--knn_weight` | `0.05` | Weight for KNN consistency loss |
| `--knn_every` | `100` | Compute KNN every N iterations |
| `--knn_offset` | `55` | Scheduling offset within KNN period |
| `--knn_warmup` | `0` | KNN weight ramp start |
| `--knn_ramp` | `0` | KNN weight ramp length |
| `--focal_alpha` | `0.25` | Focal loss alpha (class imbalance) |
| `--focal_gamma` | `2.0` | Focal loss gamma (hard example mining) |
| `--uncertainty_min` | `0.1` | Minimum pixel weight for soft pseudo-labels |
| `--uncertainty_power` | `2.0` | Confidence-to-weight curve shape |
| `--seg_only` | `False` | Two-stage: freeze geometry, train seg head only |
| `--seg_only_reuse_head` | `False` | Keep original seg head in two-stage mode |
| `--auto_twostage` | `False` | Automatic geometry pretraining + semantic fine-tuning |
| `--no_opacity_detach` | `False` | Allow mask gradients into opacity MLP |
| `--opacity_grad_until` | `-1` | Gradual opacity detach (-1=use flag) |

### Anchor Densification

| Parameter | Default | Description |
|---|---|---|
| `--start_stat` | `500` | Start accumulating gradient statistics |
| `--update_from` | `1500` | First densification iteration |
| `--update_interval` | `100` | Densification check interval |
| `--update_until` | `15000` | Stop densification (-1=never stop, 0=disable) |
| `--densify_grad_threshold` | `0.0002` | Gradient threshold for anchor growing |
| `--min_opacity` | `0.005` | Opacity threshold for pruning |

---

## Architecture

### Dual-Pass Rasterization

```
Pass 1: RGB
  rasterizer(means3D, colors_precomp=color, opacities=opacity)
  -> rendered_image

Pass 2: Semantic Mask
  seg_feature = pad(segmentation_probs, 128 channels)
  rasterizer_mask(means3D, semantic_feature=seg_feature, opacities=opacity.detach())
  -> rendered_mask
```

> **Critical**: Mask pass uses `opacity.detach()` to prevent semantic loss gradients from flowing into the opacity MLP, protecting RGB reconstruction quality.

### Segmentation Head

```python
SegmentationHead(feat_dim=32, hidden_dim=128, num_layers=3, dropout=0.0, num_outputs=1)
# input_proj + LayerNorm + ReLU
# 3x residual blocks (Linear + LayerNorm + ReLU)
# logit_head -> sigmoid (binary) or softmax (multi-class)
```

### KNN Consistency Loss

- **Binary**: Symmetric BCE in probability space
  ```
  L = -0.5 * [ p_i*log(p_j) + (1-p_i)*log(1-p_j) + p_j*log(p_i) + (1-p_j)*log(1-p_i) ]
  ```
- **Multi-class**: Symmetric KL divergence in logit space

---

## Evaluation Notes

### Train vs Test Sample Size Bias

In `training_report()`:
- **Train set**: Only **5 cameras** evaluated (`range(5, 30, 5)`)
- **Test set**: **All** test cameras evaluated

**Train mIoU is NOT directly comparable to test mIoU.** Use test mIoU as the ground truth.

---

## Project Structure

```
Scaffold-GSLFY/
├── train.py                          # Main training loop
├── scene/
│   ├── gaussian_model.py             # GaussianModel, SegmentationHead, densification
│   ├── dataset_readers.py            # COLMAP/Blender data loading, mask loading
│   └── __init__.py                   # Scene class
├── gaussian_renderer/__init__.py     # Neural Gaussian generation + dual-pass rasterization
├── arguments/__init__.py             # All CLI hyperparameters
├── eval_myvideo.py                   # Binary IoU evaluation (test set)
├── eval_multiclass.py                # Standalone multi-class IoU evaluation
├── eval_scene_multiclass.py          # Per-scene multi-class evaluation
├── tools/
│   ├── render.py                     # Standard RGB + mask rendering
│   ├── render1.py                    # Rendering with semantic heatmap overlay
│   └── metrics.py                    # PSNR / SSIM / LPIPS
├── configs/                          # Benchmark experiment scripts
└── submodules/
    ├── diff-gaussian-rasterization/  # CUDA rasterizer (128-channel semantic_feature)
    └── simple-knn/                   # KNN search utility
```

---

## Citation

If you use this work, please cite both the original Scaffold-GS and acknowledge this extension:

```bibtex
@inproceedings{scaffoldgs,
  title={Scaffold-gs: Structured 3d gaussians for view-adaptive rendering},
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo and Wang, Limin and Lin, Dahua and Dai, Bo},
  booktitle={CVPR},
  pages={20654--20664},
  year={2024}
}
```

## License

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).
