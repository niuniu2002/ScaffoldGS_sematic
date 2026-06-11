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

> **Note on `dronev4_2`**: Training uses SAM-generated masks; evaluation on `myvideo` (37 images, human-annotated) reveals a **label-shift gap** (SAM vs Human IoU ≈ 0.62). See [Evaluation](#evaluation) for mitigation strategies.

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
# Standard evaluation (PSNR + mIoU on test set)
python eval_myvideo.py -m output/dronev4_2_baseline --iteration 30000

# Multi-class standalone evaluation
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

| Config | `start_sem` | `mask_w` | `knn_w` | `focal_a` | `update_until` | PSNR | mIoU | FG IoU | BG IoU |
|---|---|---|---|---|---|---|---|---|---|
| **exp01_sem_ramp** | 0 | 0.2 | 0.05 | 0.25 | 15000 | 24.71 | **0.7880** | 0.6216 | 0.9804 |
| **exp02_late_sem** | 5000 | 0.2 | 0.05 | 0.25 | 15000 | 24.68 | **0.7891** | 0.6236 | 0.9807 |
| **exp03_nodensify** | 5000 | 0.2 | 0.05 | 0.25 | 0 | 24.63 | 0.7885 | 0.6178 | 0.9798 |
| **user_params** | 5000 | 0.1 | 0.02 | 0.25 | 0 | 24.54 | 0.7459 | 0.6164 | 0.9809 |
| **exp06_stopdensify15k** | 5000 | 0.2 | 0.05 | 0.25 | 15000 | **25.04** | 0.7778 | 0.6197 | 0.9810 |

**Key Insights** (from ablation studies):
- `focal_alpha=0.25` is the sweet spot. `0.75` biases toward background and locks mIoU low.
- `mask_weight=0.1` balances mIoU and PSNR. Above `0.2`, PSNR gap exceeds 3 dB (geometry collapse).
- `knn_weight=0.02` gives spatial consistency without over-smoothing.
- `update_until` is **not** the primary factor for mIoU-PSNR trade-off; `focal_alpha` + `mask_weight` dominate.

### Recommended Configurations

#### A. Best mIoU (Joint Training)
```bash
--start_semantic_iter 5000 \
--mask_weight 0.1 --mask_warmup 1000 --mask_ramp 3000 \
--knn_weight 0.02 --knn_warmup 2000 --knn_ramp 3000 \
--knn_every 100 --knn_offset 55 \
--focal_alpha 0.25 \
--update_until 0 \
--iterations 30000
```
> Test mIoU ≈ 0.75, PSNR gap ≈ 1.0 dB (minimal overfitting).

#### B. High mIoU with Acceptable PSNR Gap
```bash
--start_semantic_iter 0 \
--mask_weight 0.2 --mask_warmup 1000 --mask_ramp 3000 \
--knn_weight 0.05 --knn_warmup 2000 --knn_ramp 3000 \
--knn_every 100 --knn_offset 55 \
--focal_alpha 0.25 \
--update_until 15000 \
--iterations 33000
```
> Test mIoU ≈ 0.79, PSNR gap ≈ 3.5–4.0 dB.

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

### SAM Label Shift vs Human Annotations

The `dronev4_2` dataset uses **SAM-generated masks** for training. The `myvideo` subset (37 images) uses **human annotations**.

- SAM vs Human mean IoU: **0.62**
- 30/37 images have IoU < 0.7
- This is a **label-shift** problem, not traditional overfitting.

**Mitigation**:
- Lower `mask_weight` (0.05–0.1)
- Add label smoothing
- Use `uncertainty_min` / `uncertainty_power` to down-weight edge pixels

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
├── eval_myvideo.py                   # Binary IoU evaluation (human-annotated test)
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
