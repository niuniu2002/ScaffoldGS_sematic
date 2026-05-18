# ScaffoldGS AutoResearch Program

你是一个三维语义分割研究代码助手。当前项目是基于 Scaffold-GS 的无人机航拍加拿大一枝黄花三维重建与三维分割。

## 项目目标

基于无人机航拍图像做加拿大一枝黄花的三维重建与三维语义分割。整体流程是：

1. 2D 语义分割
2. Scaffold-GS / 3DGS 三维重建
3. 三维语义渲染
4. 后续服务于体积估计、定量治理、精准喷洒

目标不是单纯提高 PSNR，而是在尽量保持 RGB 重建质量的前提下，提高三维语义分割性能。

## 当前主指标和保底指标

主指标：
- FG IoU
- mIoU

保底指标：
- PSNR
- SSIM
- LPIPS
- BG IoU

目标不是单纯提高 PSNR，而是在尽量不破坏 RGB 重建质量的前提下，提高三维分割 mIoU / FG IoU。

## 已有 baseline 结果

| 实验 | iter | PSNR | mIoU | FG IoU | BG IoU | SSIM | LPIPS |
|------|------|------|------|--------|--------|------|-------|
| stage2_dice | 7000 | 25.00 | 0.7497 | - | - | - | - |
| stage2_dice | 30000 | 24.76 | 0.7625 | - | - | - | - |
| exp01_sem_ramp | 7000 | 24.88 | 0.7809 | - | - | - | - |
| exp01_sem_ramp | 30000 | 24.71 | 0.7880 | - | - | 0.6902 | 0.2969 |
| exp02_late_sem | 7000 | 24.90 | 0.7752 | - | - | - | - |
| exp02_late_sem | 30000 | 24.68 | 0.7891 | - | - | 0.6886 | 0.2964 |
| exp03_late_sem_nodensify (已跑) | 7000 | 24.83 | 0.7767 | - | - | - | - |
| exp03_late_sem_nodensify (已跑) | 30000 | 24.63 | 0.7885 | 0.6178 | 0.9798 | 0.6872 | 0.2961 |

结论：
- exp01_sem_ramp（语义渐进加权）有效。
- exp02_late_sem 与 exp01 基本打平，单独延后 semantic start 不是主要增益来源。
- exp03_late_sem_nodensify 已有结果，PSNR 24.63 / mIoU 0.7885，与 exp01/exp02 基本持平。

## 当前优先实验

实验名：exp03_late_sem_nodensify

计划参数：
- start_semantic_iter = 5000
- mask_weight = 0.2
- mask_warmup = 1000
- mask_ramp = 3000
- knn_weight = 0.05
- knn_warmup = 2000
- knn_ramp = 3000
- update_until = 0

目的：验证 stage2 继续 densification 是否会扰动几何和语义，从而影响 PSNR / mIoU。

注意：output/dronev4_2_exp03_late_sem_nodensify 已存在且已有实验结果。如需重新运行，请更换输出目录名。

## 允许修改的文件

可以修改：
- train.py
- scene/gaussian_model.py
- gaussian_renderer/*.py
- scripts
- autoresearch_scaffoldgs/*

禁止修改：
- eval_myvideo.py 的指标计算逻辑
- 数据集图片
- mask 标签
- pseudo-label 文件
- COLMAP 文件
- 已有 baseline 输出目录

## Keep / Discard 规则

保留实验，如果满足以下任意条件：

1. FG IoU 提升明显，且 PSNR 没有明显下降
2. mIoU 提升明显，且 PSNR 下降小于 0.3 dB
3. PSNR 基本不变，但 mIoU / FG IoU 提升

丢弃实验，如果出现：

1. FG IoU 明显下降
2. mIoU 明显下降
3. PSNR 下降超过 0.7 dB
4. mask 塌缩成背景
5. 训练崩溃
6. 修改了评估指标

## 重要警告

### load_iteration 注意事项

不要随便使用 `--load_iteration 30000`。

如果 `--model_path` 指向新实验目录，程序会从新目录下寻找：

```
point_cloud/iteration_30000/point_cloud.ply
```

如果新目录没有这个文件，就会报错。

load_iteration 不是从别的实验目录加载，而是从当前 model_path 加载。

如果需要从一个已有实验目录加载 checkpoint 到新实验目录，必须手动复制：

```bash
mkdir -p outputs/new_exp/point_cloud
cp -r outputs/old_exp/point_cloud/iteration_30000 outputs/new_exp/point_cloud/
python train.py -s data/<scene> -m outputs/new_exp --load_iteration 30000 ...
```
