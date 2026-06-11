# Scaffold-GSLFY Research Log

## 2026-06-11 Recovery after Disk Migration

### 数据丢失范围

- **最新同步记录**：仅到 2026-06-07（daily_report_2026-06-07.md）。
- **丢失内容**：2026-06-08 至 2026-06-11 的本地训练日志、checkpoint、代码 diff、部分实验结果，因硬盘迁移未同步。
- **处理原则**：没有日志文件（outputs.log）支撑的 30k final 结果一律不编；所有缺失结果统一标记为：

```text
lost due to disk migration; need rerun
```

### 已恢复结果（来自聊天记录与现存 checkpoint）

#### Phase C 7k 中期结果

| 实验 | mask_weight | Test PSNR | Test mIoU | Train PSNR | PSNR gap | fg_ratio |
| -- | ----------: | --------: | --------: | ---------: | -------: | -------: |
| C1 |         0.1 |     24.49 |     0.861 |      24.97 |     0.48 |    ~0.03 |
| C2 |         0.2 |     24.40 |     0.869 |      24.91 |     0.51 |    ~0.06 |

> C1/C2 的 30k final 结果：lost due to disk migration; need rerun。

#### 三个数据集诊断结果

| 数据集       | Test PSNR | Test mIoU | PSNR gap | 说明                          |
| --------- | --------: | --------: | -------: | --------------------------- |
| dronev4_2 |     25.15 |     0.890 |     2.62 | nodetach 0.15 历史较优结果        |
| lfy       |     21.71 |     0.877 |     6.05 | mIoU 不低，但新视角泛化差             |
| scene_01  |     20.32 |     0.375 |     2.84 | RGB-only baseline 低，基础重建是瓶颈 |

**结论**：
- `dronev4_2` 是当前最稳定的数据集，可作为方法验证主战场。
- `lfy` 的问题是新视角泛化差（PSNR gap 6.05），不是单纯分割精度低。
- `scene_01` 暂时作为诊断数据集，不能作为主方法结论核心；其 RGB-only baseline 低说明基础重建是瓶颈。

### 实验范围限定

- **暂时不要管 AMtown01**。
- 只围绕以下三个数据集：
  ```text
  dronev4_2
  lfy / colmap_scene
  SW_scenes / scene_01
  ```
- **主实验优先级**：`dronev4_2 + lfy > scene_01`
- `scene_01` 暂时只做诊断或补充，不要抢主线。

### 脚本修改记录

- `configs/run_colmap_scene_baseline.sh` 和 `configs/run_scene_01_baseline.sh` 在恢复过程中被改为 nodetach SOTA 配置（mask_weight=0.4, no_opacity_detach, update_until=15000）。
- 这些修改与原始 baseline（detach, mask_weight=0.1, update_until=0）不同，容易混淆。
- git diff 已保存到 `docs/diffs/` 和 `docs/recovery_git_status_*.txt`。
- **后续如需跑公平对照，必须区分 baseline 与 nodetach SOTA，不可混用脚本名。**

### 当前主线：Dual-Feature 恢复

**核心思想**：
```text
RGB branch: _anchor_feat
Seg branch:  _anchor_feat_seg
```

**目的**：让语义分支拥有独立 feature，不再通过 `no_opacity_detach` 直接干扰 RGB opacity / RGB feature。

**只允许修改的文件**：
```text
arguments/__init__.py
scene/gaussian_model.py
gaussian_renderer/__init__.py
train.py
```

**关键逻辑**（必须保证）：
```python
feat_seg = pc._anchor_feat_seg[visible_mask] if pc.dual_feature else feat.detach()
```

- `dual_feature=False` 时，旧 baseline 完全保持 `feat.detach()`，行为不变。
- `dual_feature=True` 时，seg branch 使用 `_anchor_feat_seg`。
- RGB branch 继续使用原来的 `_anchor_feat`。
- 不要默认打开 `no_opacity_detach`。

### D2 公平对照计划

D2 只跑 `dronev4_2 + lfy`，同一套参数，不允许给 lfy 单独调参：

| 方法                    | mask_weight | no_opacity_detach | dual_feature | 数据集             |
| --------------------- | ----------: | ----------------- | ------------ | --------------- |
| sem_ramp 0.2          |         0.2 | False             | False        | dronev4_2 + lfy |
| no_opacity_detach 0.2 |         0.2 | True              | False        | dronev4_2 + lfy |
| Dual-Feature 0.2      |         0.2 | False             | True         | dronev4_2 + lfy |

> 历史 `no_opacity_detach 0.4` 只能作为参考，不作为公平主对照。

### Smoke Test 计划

Dual-Feature 代码恢复后先跑 smoke test，不要直接跑 30k：

**Smoke（3k）**：
```bash
conda run -n scaffold_gslfy --no-capture-output python train.py \
  -s /mnt/data/liufengyang/data/dataset/dronev4_2 \
  -m outputs/dronev4_dualfeat_smoke_20260611_rebuild \
  --num_classes 1 --eval --resolution 2 --white_background \
  --dual_feature \
  --mask_weight 0.2 \
  --start_semantic_iter 500 --update_until 15000 \
  --knn_weight 0.02 --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --iterations 3000 \
  --test_iterations 1000 3000 \
  --save_iterations 3000 \
  --port 6211
```

**Resume test（3100）**：
```bash
conda run -n scaffold_gslfy --no-capture-output python train.py \
  -s /mnt/data/liufengyang/data/dataset/dronev4_2 \
  -m outputs/dronev4_dualfeat_smoke_20260611_rebuild \
  --num_classes 1 --eval --resolution 2 --white_background \
  --dual_feature \
  --mask_weight 0.2 \
  --start_semantic_iter 500 --update_until 15000 \
  --knn_weight 0.02 --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --iterations 3100 \
  --load_iteration 3000 \
  --test_iterations 3100 \
  --port 6212
```

通过 smoke/resume 后，再跑正式 D2。

### Dual-Feature 代码恢复状态

**恢复时间**：2026-06-11

**已修改文件**：
- `arguments/__init__.py`：添加 `dual_feature=False`
- `scene/gaussian_model.py`：添加 `_anchor_feat_seg` 及其完整生命周期（create / save / load / prune / grow / optimizer）
- `gaussian_renderer/__init__.py`：`feat_seg = pc._anchor_feat_seg[visible_mask] if pc.dual_feature else feat.detach()`，seg MLP 输入从 `feat.detach()` 切换为 `feat_seg`
- `train.py`：`GaussianModel` 实例化传入 `dual_feature`；seg_only 模式下冻结 `_anchor_feat_seg`

**兼容性保证**：
- `dual_feature=False` 时，所有旧路径行为完全一致（`feat.detach()` 不变）。
- 旧 PLY checkpoint 无 `f_anchor_feat_seg_*` 属性时，自动从 `anchor_feat` 初始化。
- 语法检查已通过。

**Smoke Test 结果（2026-06-11）**：

| 指标 | Test | Train | Gap |
| --- | --- | --- | --- |
| PSNR | 23.29 | 23.52 | 0.23 dB |
| mIoU | 0.8290 | 0.8445 | 0.0155 |
| FG_IoU | 0.6747 | 0.6991 | — |

- PSNR gap 仅 0.23 dB，说明 Dual-Feature 有效隔离了语义分支对 RGB geometry 的干扰。
- 3k checkpoint 已成功保存。

**Resume Test 结果（2026-06-11）**：

| 指标 | Test | Train | Gap |
| --- | --- | --- | --- |
| PSNR | 23.79 | 23.80 | 0.01 dB |
| mIoU | 0.8307 | 0.8552 | 0.0245 |
| FG_IoU | 0.6778 | 0.7198 | — |

- Resume 后 PSNR gap 几乎为 0，checkpoint 兼容性验证通过。

**Bug 修复记录**：
- `scene/gaussian_model.py` `load_ply_sparse_gaussian` 中 `anchor_feat_names` 的过滤条件 `startswith("f_anchor_feat")` 会误匹配 `f_anchor_feat_seg_*`，导致 `_anchor_feat` 被加载为 64 维。
- 修复为：`startswith("f_anchor_feat_") and not startswith("f_anchor_feat_seg_")`。
- 该 bug 仅在 `dual_feature=True` 且从 PLY resume 时触发；smoke test 从头训练不受影响，但 resume 会崩溃。

**下一步**：启动 D2 公平对照（dronev4_2 + lfy，30k）。

### D2 启动记录（2026-06-11）

**lfy 数据集路径修正**：实际 COLMAP 数据位于 `/mnt/data/liufengyang/data/dataset/lfy/colmap_scene`，而非 `/mnt/data/liufengyang/data/dataset/lfy`（后者为原始标注目录，无 `sparse/`）。此前 lfy 实验因路径错误报 `Could not recognize scene type!`，已修正。

**lfy 图片补充**：`lfy/colmap_scene/images/` 为空，COLMAP `images.bin` 引用的 `.jpg` 实际存放于 `lfy/lfy/`。已将 200 张 `.jpg` 复制到 `lfy/colmap_scene/images/`，实验已正常启动。

**批次策略**：僵尸进程已清理，显存恢复。改为分两轮跑，每轮 3 个：先 dronev4_2，后 lfy。输出目录统一加 `YYYYMMDD` 日期前缀。

**第一轮（dronev4_2，2026-06-12 启动）**：

| 数据集 | 方法 | 输出目录 | 端口 | 状态 |
| --- | --- | --- | --- | --- |
| dronev4_2 | sem_ramp | outputs/20260612_d2_dronev4_sem_ramp | 6213 | running |
| dronev4_2 | no_opacity_detach | outputs/20260612_d2_dronev4_nodetach | 6214 | running |
| dronev4_2 | Dual-Feature | outputs/20260612_d2_dronev4_dualfeat | 6215 | running |

**第二轮（lfy）**：待 dronev4_2 三组合部跑完后启动，目录将使用当日日期前缀。

所有实验统一参数：`mask_weight=0.2`, `start_semantic_iter=500`, `update_until=15000`, `knn_weight=0.02`, `focal_alpha=0.25`, `resolution=2`。

### 已在跑实验标记

- `output/sw_scene_01_sota_nodetach` 已启动（VGG16 下载中），但标记为：
  ```text
  extra diagnostic / low priority
  ```
- 不要将其写入主线。当前主线仍然是：
  ```text
  Reserch.md 恢复
  Dual-Feature 代码恢复
  3k smoke test
  3100 resume test
  dronev4_2 + lfy 公平对照
  ```
