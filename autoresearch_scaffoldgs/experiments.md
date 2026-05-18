# 实验摘要与计划

## 已有实验结果

### stage2_dice

- 7000 iter: PSNR 25.00, mIoU 0.7497
- 30000 iter: PSNR 24.76, mIoU 0.7625

### exp01_sem_ramp

- start_semantic_iter = 0
- mask_warmup = 1000
- mask_ramp = 3000
- knn_warmup = 2000
- knn_ramp = 3000
- 7000 iter: PSNR 24.88, mIoU 0.7809
- 30000 iter: PSNR 24.71, mIoU 0.7880
- final evaluate: SSIM 0.6902, PSNR 24.7041, LPIPS 0.2969

结论：语义渐进加权有效。

### exp02_late_sem

- start_semantic_iter = 5000
- update_until = 15000
- 7000 iter: PSNR 24.90, mIoU 0.7752
- 30000 iter: PSNR 24.68, mIoU 0.7891
- final evaluate: SSIM 0.6886, PSNR 24.6374, LPIPS 0.2964

结论：单独延后 semantic start 收益不明显，基本和 exp01_sem_ramp 打平。

### exp03_late_sem_nodensify（已跑过）

- start_semantic_iter = 5000
- update_until = 0
- 7000 iter (test): PSNR 24.83, mIoU 0.7767
- 30000 iter (test): PSNR 24.63, mIoU 0.7885
- final evaluate: SSIM 0.6872, PSNR 24.6282, LPIPS 0.2961
- myvideo_iou_iter30000: PSNR 23.19, BG IoU 0.9798, FG IoU 0.6178, mIoU 0.7988

注意：该实验使用了 `--load_iteration 30000`，初始高斯从 checkpoint 加载，不是从零训练。

---

## 当前计划实验

### exp03_late_sem_nodensify（ reproduce / 从零训练验证）

**Hypothesis：**
stage2 继续 densification 会扰动 anchor 分布，导致几何不稳定，从而同时影响 PSNR 和 mIoU。关闭 densification（update_until=0）可能保持更稳定的 geometry，进而提升或至少保持分割精度。

**计划参数：**
- start_semantic_iter = 5000
- mask_weight = 0.2
- mask_warmup = 1000
- mask_ramp = 3000
- knn_weight = 0.05
- knn_warmup = 2000
- knn_ramp = 3000
- update_until = 0

**当前状态：** not started（已有同名目录结果，但为 load_iteration 模式；如需从零训练验证，需更换输出目录名）
