# Scaffold-GS with 3D Semantic Segmentation

This project extends [Scaffold-GS](https://github.com/city-super/Scaffold-GS) with **anchor-based 3D semantic segmentation** capabilities.

## Core Contributions (3D Segmentation)

### 1. Anchor Segmentation Head
A lightweight MLP (`mlp_segmentation`) is attached to each anchor point in the Gaussian model. It predicts a per-anchor binary segmentation confidence that is splatted to per-Gaussian level during rendering.

**File:** `scene/gaussian_model.py`

### 2. Rendering with Mask Output
The rasterizer pipeline is extended to output a rendered semantic mask alongside the RGB image. The mask is supervised against 2D semantic annotations.

**File:** `gaussian_renderer/__init__.py`

### 3. Multi-Task Training Loss
The training loop (`train.py`) jointly optimizes:
- **RGB reconstruction** (L1 + SSIM)
- **Mask supervision** (pixel-wise focal loss with uncertainty weighting)
- **KNN spatial consistency** (encourages smooth segmentation logits among spatially neighboring anchors)
- **Scaling regularization** (optional, preserves thin structures)

**File:** `train.py`

### 4. Mask-Aware Dataset Reader
COLMAP dataset reader is extended to optionally load 2D semantic masks from a `masks/` folder (PNG/JPG) alongside RGB images.

**File:** `scene/dataset_readers.py`

### 5. Visualization Scripts
- `render1.py` — Renders RGB + semi-transparent semantic overlay for qualitative inspection.
- `render_video.py` — Generates smooth camera-trajectory videos with segmentation visualization.

## Data Layout

Place your COLMAP-processed scene under `data/<scene>/` with the following structure:

```
data/
└── my_scene/
    ├── images/
    │   ├── 0001.jpg
    │   └── ...
    ├── masks/           # <-- semantic masks (optional, same basename as images)
    │   ├── 0001.png
    │   └── ...
    └── sparse/
        └── 0/
```

## Training

Single scene:
```bash
bash single_train.sh
```

Batch scripts for public datasets are in `scripts/`.

## Key Hyperparameters (in `arguments/__init__.py`)

| Parameter | Default | Description |
|---|---|---|
| `start_semantic_iter` | 7000 | First iteration to enable mask/KNN losses |
| `mask_weight` | 0.01 | Weight for rendered-mask BCE loss |
| `knn_weight` | 0.05 | Weight for KNN spatial-consistency loss |
| `focal_alpha` | 0.25 | Focal-loss class-imbalance parameter |
| `focal_gamma` | 2.0 | Focal-loss hard-example-mining parameter |
| `uncertainty_min` | 0.1 | Minimum pixel weight for uncertain pseudo-labels |
| `knn_every` | 100 | Compute KNN loss every N iterations |
| `use_semantic_weight` | False | Enable semantic weight map from external models |
| `sem_weight_strategy` | "hard" | "hard"=three-stage threshold, "smooth"=continuous curve |
| `sem_weight_high` | 0.7 | High-confidence threshold (abs_conf >= 0.7 => weight=1.0) |
| `sem_weight_low` | 0.1 | Low-confidence threshold (abs_conf < 0.1 => ignored) |
| `sem_weight_boost` | 2.0 | Weight boost factor for medium-confidence pixels |

## Semantic Weight Map (Optional)

To leverage external semantic models (e.g., SegFormer, Mask2Former) for better segmentation supervision:

1. **Generate weight maps**: Run your semantic model on each image and save the per-pixel confidence (softmax probability) as a grayscale image.
2. **Place in `semantic_weights/`**: Create a `semantic_weights/` folder alongside `images/` and `masks/`. Files should share the same basename as their corresponding RGB images.
3. **Pixel value convention**:
   - `0`   = background confidence is 100% (abs_conf = 1.0)
   - `128` = completely uncertain (abs_conf = 0.0)
   - `255` = foreground confidence is 100% (abs_conf = 1.0)
4. **Enable in training**: Add `--use_semantic_weight` to your training command.

### Weight Strategy Explained

The semantic weight map modulates the per-pixel mask loss:
- **High confidence** (abs_conf >= `sem_weight_high`): normal weight (`1.0`) — reliable supervision.
- **Medium confidence** (`sem_weight_low` <= abs_conf < `sem_weight_high`): boosted weight (`sem_weight_boost`) — "focus attention" on ambiguous boundaries.
- **Low confidence** (abs_conf < `sem_weight_low`): weight = `0.0` — ignored to avoid noise propagation.

## File Map

| File | Role |
|---|---|
| `train.py` | Main training loop with segmentation losses |
| `scene/gaussian_model.py` | Anchor Gaussian model + segmentation MLP |
| `gaussian_renderer/__init__.py` | Rasterization pipeline with mask output |
| `scene/dataset_readers.py` | COLMAP reader + mask loading |
| `arguments/__init__.py` | CLI arguments & hyperparameters |
| `render.py` | Standard rendering (RGB + mask) |
| `render1.py` | Rendering with semantic overlay visualization |
| `render_video.py` | Camera-path video rendering |
| `metrics.py` | PSNR/SSIM/LPIPS evaluation |
| `scripts/*.sh` | Batch training scripts for public datasets |
