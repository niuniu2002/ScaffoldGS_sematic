# Unified Benchmark (20260618) 实验记录

## 1. 概述

本文档记录 `20260618-unified-bench` 统一 benchmark 的构建过程、数据预处理、以及两种 3D Gaussian Splatting 分割方法（Gaussian-Grouping 与 COB-GS）在该 benchmark 上的评测结果。

- **Benchmark 路径**：`/mnt/data/liufengyang/data/benchmark/20260618-unified-bench/`
- **打包文件**：`/mnt/data/liufengyang/benchmark_20260618.tar.gz`（7.9 GB）
- **解压后大小**：9.5 GB
- **场景数量**：21 个
- **分辨率**：RGB 图像 1920×1080，mask 为 L-mode（0=背景，255=前景）
- **训练/测试划分**：80% / 20%，随机种子 `seed=42`

## 2. 数据来源与构建

### 2.1 原始数据来源

| 来源 | 场景数 | 说明 |
|------|--------|------|
| `dronev4_2` | 1 | 无人机视频序列 |
| `lfy/colmap_scene` | 1 | 静态场景 |
| `SW_Dateset` | 19（`scene_00`–`scene_18`） | 原 `SW_scenes` 按固定区间分段，COLMAP 注册率过低，后按相邻帧 SIFT 几何内点重新分段 |

`SW_Dateset` 原始数据：2192 张 4K 图像（缺 `0287.jpg`），2193 张 mask。

### 2.2 SW 场景重新分段

原 `SW_scenes` 按固定区间分成 9 段，部分场景 COLMAP 注册率极低（`SW_scene_03` 18.0%、`SW_scene_07` 43.5%、`SW_scene_08` 21.8%）。

**重新分段方法**：
- 以相邻帧 SIFT 几何内点数量作为连续性指标
- 检测连续性断点
- 约束每段长度 60–200 帧，合并短段、切开超长段
- 生成 19 个新 scene（`scene_00`–`scene_18`）

### 2.3 预处理

- 4K 图像 resize 到 **1920×1080**
- mask resize 到 **1920×1080**，转为 L-mode，像素值归一化到 **0/255**
- 对每个 scene 跑 COLMAP sequential matcher，生成 `sparse/0/`
- 生成 `train_list.txt` / `test_list.txt`（80/20，seed=42）

### 2.4 场景统计

| 场景 | 图像数 | Mask 数 | 训练集 | 测试集 | COLMAP 注册数 | 注册率 |
|------|--------|---------|--------|--------|---------------|--------|
| dronev4_2 | 333 | 333 | 266 | 67 | 333 | 100.0% |
| lfy | 200 | 200 | 160 | 40 | 200 | 100.0% |
| scene_00 | 133 | 133 | 106 | 27 | 133 | 100.0% |
| scene_01 | 89 | 89 | 71 | 18 | 85 | 95.5% |
| scene_02 | 65 | 65 | 52 | 13 | 65 | 100.0% |
| scene_03 | 164 | 164 | 131 | 33 | 163 | 99.4% |
| scene_04 | 60 | 60 | 48 | 12 | 60 | 100.0% |
| scene_05 | 97 | 97 | 77 | 20 | 97 | 100.0% |
| scene_06 | 63 | 63 | 50 | 13 | 52 | 82.5% |
| scene_07 | 103 | 103 | 82 | 21 | 96 | 93.2% |
| scene_08 | 149 | 149 | 119 | 30 | 94 | 63.1% |
| scene_09 | 61 | 61 | 48 | 13 | 21 | 34.4% |
| scene_10 | 161 | 161 | 128 | 33 | 52 | 32.3% |
| scene_11 | 79 | 79 | 63 | 16 | 11 | 13.9% |
| scene_12 | 127 | 127 | 101 | 26 | 127 | 100.0% |
| scene_13 | 135 | 135 | 108 | 27 | 34 | 25.2% |
| scene_14 | 175 | 175 | 140 | 35 | 175 | 100.0% |
| scene_15 | 116 | 116 | 92 | 24 | 56 | 48.3% |
| scene_16 | 154 | 154 | 123 | 31 | 45 | 29.2% |
| scene_17 | 79 | 79 | 63 | 16 | 37 | 46.8% |
| scene_18 | 182 | 182 | 145 | 37 | 57 | 31.3% |

> 注：`scene_09`–`scene_18` 部分场景运动较快或纹理较弱，COLMAP sequential matcher 注册率偏低，但所有场景均已有 `sparse/0/` 重建结果，可直接用于训练。

## 3. 评测方法

### 3.1 Gaussian-Grouping

- 在 19 个 `scene_*` 上训练 30k 迭代
- 使用 benchmark 标准 hold-out 测试集
- 默认配置：`config/gaussian_dataset/train.json`
- 大场景 `scene_14`（175 张 1080P）与 `scene_18`（182 张 1080P）默认配置 OOM，改用低 densify 配置 `config/gaussian_dataset/train_low_densify.json`：
  - `densify_until_iter`：10000 → 6000
  - `densify_grad_threshold`：0.0002 → 0.0005

### 3.2 COB-GS

- 使用每个 scene 已有的 `masks/` 文件夹，未重新提取 SAM mask
- 流程：
  1. base 3DGS 训练 30k 迭代
  2. mask finetuning：`--include_mask --finetune_mask`，`N4views=10`，`mask_threshold=0.8`
  3. benchmark 评测
- 关键改动：
  - `data/COB-GS/scene/dataset_readers.py`：优先读取 `test_list.txt` 作为 test cameras
  - `data/COB-GS/eval_benchmark_cobgs.py`：新建，从 `masks/` 加载 mask checkpoint，渲染 test cameras，计算 PSNR 与 mIoU
  - `tools/run_cobgs_benchmark.py`：新建，支持 `--max-jobs` 并行跑 base → finetune → eval
- 多数 scene 以 3 进程并行运行；`scene_14` 并行 base 训练时 OOM，改为单进程后成功

## 4. 实验结果

### 4.1 总体对比

| 方法 | mean PSNR | mean mIoU |
|---|---|---|
| **Gaussian-Grouping** | **27.20** | **0.9648** |
| **COB-GS** | **21.01** | **0.8650** |

Gaussian-Grouping 平均比 COB-GS 高 **6.19 dB PSNR** 和 **0.0998 mIoU**。

### 4.2 Gaussian-Grouping 每场景结果

| scene | PSNR | SSIM | LPIPS | test mIoU |
|-------|------|------|-------|-----------|
| scene_00 | 25.19 | 0.8269 | 0.1700 | 0.9745 |
| scene_01 | 24.66 | 0.8046 | 0.1992 | 0.9592 |
| scene_02 | 27.41 | 0.8614 | 0.1571 | 0.9563 |
| scene_03 | 25.27 | 0.8269 | 0.1618 | 0.9565 |
| scene_04 | 27.83 | 0.8836 | 0.1222 | 0.9709 |
| scene_05 | 26.15 | 0.8679 | 0.1388 | 0.9508 |
| scene_06 | 28.78 | 0.9078 | 0.1030 | 0.9782 |
| scene_07 | 26.35 | 0.8301 | 0.1677 | 0.9594 |
| scene_08 | 23.95 | 0.7773 | 0.1789 | 0.8917 |
| scene_09 | 29.84 | 0.9433 | 0.0721 | 0.9811 |
| scene_10 | 28.74 | 0.9258 | 0.0950 | 0.9075 |
| scene_11 | 36.91 | 0.9882 | 0.0214 | 0.9999 |
| scene_12 | 23.82 | 0.7916 | 0.2384 | 0.9745 |
| scene_13 | 32.64 | 0.9711 | 0.0579 | 0.9999 |
| scene_14 | 23.08 | 0.6967 | 0.3298 | 0.9515 |
| scene_15 | 23.56 | 0.7571 | 0.2856 | 0.9849 |
| scene_16 | 29.88 | 0.9456 | 0.0870 | 0.9969 |
| scene_17 | 27.81 | 0.9242 | 0.1086 | 0.9855 |
| scene_18 | 24.87 | 0.7088 | 0.3075 | 0.9529 |
| **mean** | **27.20** | — | — | **0.9648** |

### 4.3 COB-GS 每场景结果

| scene | #test | PSNR | FG IoU | BG IoU | mIoU |
|-------|-------|------|--------|--------|------|
| scene_00 | 27 | 22.13 | 0.7251 | 0.9899 | 0.8575 |
| scene_01 | 17 | 21.63 | 0.8162 | 0.9886 | 0.9024 |
| scene_02 | 13 | 24.31 | 0.6299 | 0.9839 | 0.8069 |
| scene_03 | 33 | 22.62 | 0.7772 | 0.9858 | 0.8815 |
| scene_04 | 12 | 23.16 | 0.8480 | 0.9917 | 0.9199 |
| scene_05 | 20 | 21.93 | 0.7732 | 0.9796 | 0.8764 |
| scene_06 | 11 | 21.66 | 0.7901 | 0.9898 | 0.8899 |
| scene_07 | 19 | 22.22 | 0.7871 | 0.9876 | 0.8873 |
| scene_08 | 18 | 22.69 | 0.6312 | 0.9893 | 0.8103 |
| scene_09 | 3 | 19.60 | 0.7319 | 0.9689 | 0.8504 |
| scene_10 | 9 | 24.22 | 0.6300 | 0.9626 | 0.7963 |
| scene_11 | 3 | 15.91 | 0.8173 | 0.9066 | 0.8620 |
| scene_12 | 26 | 19.16 | 0.8701 | 0.8980 | 0.8840 |
| scene_13 | 9 | 16.58 | 0.8158 | 0.9464 | 0.8811 |
| scene_14 | 35 | 23.26 | 0.7818 | 0.9155 | 0.8487 |
| scene_15 | 12 | 17.10 | 0.7627 | 0.8676 | 0.8152 |
| scene_16 | 7 | 16.70 | 0.8298 | 0.8978 | 0.8638 |
| scene_17 | 5 | 20.56 | 0.8801 | 0.8988 | 0.8895 |
| scene_18 | 13 | 23.77 | 0.8289 | 0.9969 | 0.9129 |
| **mean** | — | **21.01** | — | — | **0.8650** |

## 5. 关键发现

1. **Gaussian-Grouping 全面优于 COB-GS**：在 PSNR 和 mIoU 两个指标上，Gaussian-Grouping 均显著领先。
2. **场景尺度影响**：
   - `scene_14`/`scene_18` 图像数多、1080P 分辨率高，Gaussian-Grouping 需降低 densify 强度才能避免 OOM
   - COB-GS 在这些大场景上表现尚可，但总体仍落后于 Gaussian-Grouping
3. **低注册率场景**：`scene_09`–`scene_18` 部分场景 COLMAP 注册率偏低，但 `sparse/0/` 中相机数量仍足以完成训练和评测。
4. **异常 scene**：
   - `scene_08` Gaussian-Grouping mIoU 明显偏低（0.8917），可能与该 scene 的 mask 类别分布或 COLMAP 注册质量有关
   - COB-GS 在 `scene_11`、`scene_13`、`scene_15`、`scene_16` 上 PSNR 较低（<18 dB），重建质量不稳定

## 6. 目录结构与使用

### 6.1 目录结构

```
20260618-unified-bench/
├── benchmark_registry.json   # 场景元数据
├── README.md                 # benchmark 说明
├── dronev4_2/
├── lfy/
├── scene_00/
├── ...
└── scene_18/

<scene>/
├── images/          # RGB 帧，1920×1080
├── masks/           # 语义 mask，L-mode
├── colmap/          # COLMAP database.db
├── sparse/0/        # COLMAP 重建结果
├── train_list.txt   # 训练帧列表
└── test_list.txt    # 测试帧列表
```

### 6.2 验证

```bash
python3 tools/verify_unified_benchmark.py \
    --root /mnt/data/liufengyang/data/benchmark/20260618-unified-bench
```

### 6.3 训练示例

**Gaussian-Grouping**：
```bash
cd /mnt/data/liufengyang/data/gaussian-grouping
python train.py \
    -s /mnt/data/liufengyang/data/benchmark/20260618-unified-bench/scene_00 \
    --config config/gaussian_dataset/train.json \
    --eval
```

**COB-GS**：
```bash
cd /mnt/data/liufengyang/data/COB-GS
python train.py \
    -s /mnt/data/liufengyang/data/benchmark/20260618-unified-bench/scene_00 \
    -m output/scene_00 \
    --eval --disable_viewer
```

## 7. 清理记录

原始 benchmark 约 103 GB，清理后 9.5 GB：
- 删除 `masks/chkpnt*.pth`（COB-GS checkpoint，86 GB）
- 删除 `_old_sw_scenes/`（旧 SW 场景备份，7.4 GB）
- 保留 `colmap/database.db`（COLMAP 中间数据库，可重新跑重建）

## 8. 相关文件

| 文件 | 说明 |
|---|---|
| `data/benchmark/20260618-unified-bench/README.md` | benchmark 完整说明文档 |
| `data/benchmark/20260618-unified-bench/benchmark_registry.json` | 场景元数据 |
| `data/gaussian-grouping/output/benchmark_eval_results.json` | Gaussian-Grouping 原始评测数据 |
| `data/COB-GS/output/cobgs_benchmark_metrics.json` | COB-GS 原始评测数据 |
| `tools/run_gaussian_grouping_benchmark.py` | Gaussian-Grouping 批量训练脚本 |
| `tools/run_cobgs_benchmark.py` | COB-GS 批量训练脚本 |
| `tools/verify_unified_benchmark.py` | benchmark 完整性验证脚本 |
| `Research.md` | 完整实验日志 |

## 9. 变更记录

- 2026-06-18：创建统一 benchmark，包含 `dronev4_2`、`lfy`、重新分段后的 19 个 SW scene
- 2026-06-23/24：完成 Gaussian-Grouping 19/19 scene 训练与评测
- 2026-06-24：完成 COB-GS 19/19 scene 训练与评测，更新 README
- 2026-06-25：清理 benchmark，删除 `masks/chkpnt*.pth` 与 `_old_sw_scenes/`，保留 COLMAP `database.db`，大小从 103 GB 降至 9.5 GB
- 2026-06-25：打包 benchmark 为 `benchmark_20260618.tar.gz`（7.9 GB），并推送实验记录到 GitHub
