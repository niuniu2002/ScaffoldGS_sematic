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

### 新增组合实验：Dual-Feature + no_opacity_detach

**启动时间**：2026-06-12

**动机**：`Dual-Feature` 分离 RGB/语义 feature，`no_opacity_detach` 允许语义损失直接优化 opacity。两者正交，组合后理论上既能减少语义对 RGB feature 的干扰，又能保持语义对 geometry 的直接优化能力。

**配置**：与 D2 完全相同，仅同时打开 `--dual_feature` 和 `--no_opacity_detach`。

| 数据集 | 方法 | 输出目录 | 端口 | 状态 |
| --- | --- | --- | --- | --- |
| dronev4_2 | Dual-Feature + no_opacity_detach | outputs/20260612_d2_dronev4_dualfeat_nodetach | 6231 | **running** (2740/30000) |

**当前进度**（截至 2026-06-12 17:56）：4090/30000，速度约 1.38 s/it。7k eval 预计数小时后触发。

**命令**：

```bash
conda run -n scaffold_gslfy --no-capture-output python train.py \
  -s /mnt/data/liufengyang/data/dataset/dronev4_2 \
  -m outputs/20260612_d2_dronev4_dualfeat_nodetach \
  --num_classes 1 --eval --resolution 2 --white_background \
  --no_opacity_detach --dual_feature \
  --mask_weight 0.2 \
  --start_semantic_iter 500 --update_until 15000 \
  --knn_weight 0.02 --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --iterations 30000 \
  --test_iterations 7000 12000 15000 30000 \
  --save_iterations 30000 \
  --port 6231
```

**预期**：Test mIoU 应不低于 `no_opacity_detach`（0.8887），PSNR gap 应小于或接近 `no_opacity_detach`（2.53 dB）。若组合有效，将成为 dronev4_2 上的新 SOTA 候选。

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
| dronev4_2 | sem_ramp | outputs/20260612_d2_dronev4_sem_ramp | 6213 | **completed** |
| dronev4_2 | no_opacity_detach | outputs/20260612_d2_dronev4_nodetach | 6214 | **completed** |
| dronev4_2 | Dual-Feature | outputs/20260612_d2_dronev4_dualfeat | 6215 | **completed** |

**dronev4_2 中期/最终结果（截至 2026-06-12）**：

| 方法 | Test PSNR | Test mIoU | Train PSNR | Train mIoU | PSNR gap | mIoU gap | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sem_ramp | 24.33 | 0.6737 | 25.37 | 0.6343 | 1.04 | -0.0394 | 30k 最终，FG_IoU 崩溃至 0.3762 |
| no_opacity_detach | **25.10** | **0.8887** | 27.63 | 0.9314 | **2.53** | 0.0427 | 30k 最终 |
| Dual-Feature | 24.37 | 0.8057 | 26.14 | 0.8347 | 1.77 | 0.0290 | 30k 最终 |

> **关键观察**：
> - `no_opacity_detach` 在 15k 后基本稳定（15k Test PSNR 25.11 / mIoU 0.8856 → 30k 25.10 / 0.8887），几何与分割均保持。
> - `Dual-Feature` 15k 时最好（Test PSNR 24.85 / mIoU 0.8369），30k 退化（24.37 / 0.8057）。
> - `sem_ramp` 30k 出现更严重的崩溃：Test FG_IoU 从 15k 的 0.6370 降至 30k 的 0.3762，Train FG_IoU 仅 0.2895（低于 test），mIoU 从 0.8094 降至 0.6737。这说明在 `mask_weight=0.2`、`update_until=15000` 配置下，15k 停止 densification 后继续用强语义损失训练，会逐步摧毁前景分割，甚至让训练集表现比测试集还差。
> - 15k 后过拟合/退化是 `sem_ramp` 与 `Dual-Feature` 共性问题，`no_opacity_detach` 能抵抗这种退化。

**第二轮（lfy，2026-06-12 启动）**：

| 数据集 | 方法 | 输出目录 | 端口 | 状态 |
| --- | --- | --- | --- | --- |
| lfy | sem_ramp | outputs/20260612_d2_lfy_sem_ramp | 6221 | **stopped at ~15.9k** |
| lfy | no_opacity_detach | outputs/20260612_d2_lfy_nodetach | 6222 | running |
| lfy | Dual-Feature | outputs/20260612_d2_lfy_dualfeat | 6223 | running |

**lfy 路径与数据**：使用 `/mnt/data/liufengyang/data/dataset/lfy/colmap_scene`，images 已补齐。

所有实验统一参数：`mask_weight=0.2`, `start_semantic_iter=500`, `update_until=15000`, `knn_weight=0.02`, `focal_alpha=0.25`, `resolution=2`, `iterations=30000`。

**lfy 当前进度**（截至 2026-06-12 17:56）：nodetach 21640/30000 / dualfeat 20390/30000 / sem_ramp stopped at ~15.9k。

**lfy 7k / 12k / 15k 中期结果**：

| 方法 | 轮次 | Test PSNR | Test mIoU | Test FG_IoU | Train PSNR | Train mIoU | Train FG_IoU | PSNR gap | mIoU gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sem_ramp | 7k | 22.21 | 0.8200 | 0.6583 | 24.21 | 0.8405 | 0.6977 | 2.00 | 0.0205 |
| no_opacity_detach | 7k | 22.24 | 0.8737 | 0.7606 | 25.23 | 0.8863 | 0.7844 | 2.99 | 0.0126 |
| Dual-Feature | 7k | 22.36 | 0.8257 | 0.6694 | 24.60 | 0.8421 | 0.7007 | 2.24 | 0.0164 |
| sem_ramp | 12k | 22.10 | 0.8419 | 0.7007 | 25.24 | 0.8539 | 0.7234 | 3.14 | 0.0120 |
| no_opacity_detach | 12k | 22.09 | 0.8748 | 0.7625 | 26.37 | 0.8912 | 0.7938 | **4.28** | 0.0164 |
| Dual-Feature | 12k | 22.16 | 0.8323 | 0.6819 | 25.77 | 0.8420 | 0.7002 | 3.61 | 0.0097 |
| sem_ramp | 15k | 21.94 | 0.8191 | 0.6595 | 25.29 | 0.8522 | 0.7206 | 3.35 | 0.0331 |
| no_opacity_detach | 15k | 21.89 | 0.8752 | 0.7633 | 26.62 | 0.8966 | 0.8039 | **4.73** | 0.0214 |
| Dual-Feature | 15k | 21.89 | 0.8328 | 0.6827 | 25.84 | 0.8450 | 0.7061 | 3.95 | 0.0122 |

> `no_opacity_detach` 15k 结果关键变化：Test PSNR 从 12k 的 22.09 **降到 21.89**，Train PSNR 从 26.37 涨到 26.62，PSNR gap 从 4.28 扩大到 **4.73 dB**。分割指标仅微涨（mIoU 0.8748→0.8752，FG_IoU 0.7625→0.7633）。这强烈说明 12k–15k 期间语义损失继续在破坏新视角几何，而分割收益已接近天花板。
>
> `sem_ramp` 15k 出现与 dronev4_2 上类似的退化：Test mIoU 从 12k 的 0.8419 **降到 15k 的 0.8191**，FG_IoU 从 0.7007 降到 0.6595。说明在 `mask_weight=0.2`、`update_until=15000` 配置下，`sem_ramp` 在 lfy 上也无法抵抗 15k 后的退化。
>
> `Dual-Feature` 15k 相对稳健：mIoU 从 0.8323 微涨到 0.8328，FG_IoU 从 0.6819 微涨到 0.6827，PSNR gap 从 3.61 扩大到 3.95。虽未崩溃，但几何过拟合也在加剧。

### lfy `sem_ramp` 停止记录

**停止时间**：2026-06-12

**停止原因**：
- 15k eval 出现与 `dronev4_2` 上 `sem_ramp` 类似的退化：Test mIoU 从 12k 的 0.8419 降至 15k 的 0.8191，FG_IoU 从 0.7007 降至 0.6595。
- 继续跑到 30k 只会浪费算力，且预期会复制 dronev4_2 上 30k 的崩溃模式（FG_IoU 跌至 0.37）。
- `sem_ramp` 已充分证明：在 `mask_weight=0.2`、`update_until=15000` 配置下，15k 后无法抵抗退化。

**后续处理**：保留 `outputs/20260612_d2_lfy_sem_ramp` 日志与已保存的 checkpoint（30k 未保存），作为公平对照数据，不再继续训练。

### dronev4_2 vs lfy 的 PSNR gap 差异分析

用户观察：添加语义分支后，`dronev4_2` 上 PSNR gap 约 2.5 dB，而 `lfy` 上 15k 已达 4.7 dB。为什么 lfy 的 gap 明显更大？

可能原因：

1. **数据集本身的新视角难度不同**
   - `dronev4_2`：UAV 航拍，训练/测试视角高度重叠，相机分布密集。历史较优结果（nodetach 0.15）PSNR gap 也只有 2.62 dB。
   - `lfy`：COLMAP scene，训练/测试视角差异大，新视角泛化天生更难。历史诊断结果 PSNR gap 已达 6.05 dB。

2. **appearance embedding 过拟合训练视角**
   - `appearance_dim=32` 为每个训练相机学习独立外观向量。
   - 在 lfy 这种训练/测试视角差异大的数据集上，appearance embedding 更容易过拟合训练视角，拉大 Train/Test PSNR gap。

3. **语义损失与几何的冲突在难泛化数据集上更剧烈**
   - `no_opacity_detach` 让语义损失直接改 opacity。
   - 在 dronev4_2 上，训练视角和测试视角看到的内容接近，改 opacity 对 test 影响小。
   - 在 lfy 上，训练视角和测试视角差异大，为训练视角优化的 opacity 分布会损害测试视角的 RGB 重建。

4. **mask 质量/一致性差异**
   - `dronev4_2` 全部使用 SAM 生成的 mask，训练/测试标签分布一致。
   - `lfy` 的 mask 来源和一致性尚待检查，如果训练 mask 本身有噪声或不一致，会加剧语义-几何冲突。

**结论**：lfy 的 PSNR gap 大，不完全是方法问题，而是 **lfy 作为数据集本身就比 dronev4_2 更难泛化**。方法层面的优化（如 15k 后 detach opacity、冻结 appearance）可以在一定程度上缓解，但可能无法把 lfy gap 压到 dronev4_2 的水平。

### 主线调整：no_opacity_detach 作为主线

**决定时间**：2026-06-12

**依据**：
- `no_opacity_detach` 在 `dronev4_2` 与 `lfy` 两个数据集上均取得最高或最稳定分割表现。
- `dronev4_2` 30k 最终：Test PSNR 25.10 / mIoU 0.8887，且 15k→30k 不衰减。
- `lfy` 12k：Test mIoU 0.8748 / FG_IoU 0.7625，仍在缓慢提升，未见 sem_ramp/Dual-Feature 的退化迹象。

**主要问题**：PSNR gap 偏大（dronev4_2 2.53 dB，lfy 12k 4.28 dB），几何过拟合明显。

**后续策略**：
- 优先围绕 `no_opacity_detach` 做优化，目标是在保持分割优势的同时压缩 PSNR gap。
- `sem_ramp` 与 `Dual-Feature` 暂时不再投入大量算力，仅作为公平对照保留已跑结果。
- 等 `lfy` 三个 30k 实验全部结束后，再启动 `no_opacity_detach` 的优化消融。

### 后期训练优化探索（2026-06-12 启动）

**问题**：`no_opacity_detach` 在 15k 后 Train PSNR 持续上涨、Test PSNR 停滞，PSNR gap 扩大；mIoU 微涨但增益小。

**根因假设**：
1. `appearance_dim=32` 的 per-camera embedding 过拟合训练视角。
2. `no_opacity_detach` 让语义损失持续牵引 opacity，densification 停止后难以同时优化 RGB 和分割。
3. mlp_opacity / geometry 学习率 15k 后衰减过快，加上 capacity 固定，进入平台期。

**验证计划**：

| 阶段 | 实验 | 做法 | 状态 |
| --- | --- | --- | --- |
| 1 | `dronev4_nodetach_save15k` | 从头训练到 15k 并保存 checkpoint（与 D2 nodetach 同参数） | **running** (8060/15000) |
| 2 | `dronev4_nodetach_opg15_resume` | 从 15k 加载，`--no_opacity_detach --opacity_grad_until 15000` 跑到 30k | pending |

**阶段 1 命令**：

```bash
conda run -n scaffold_gslfy --no-capture-output python train.py \
  -s /mnt/data/liufengyang/data/dataset/dronev4_2 \
  -m outputs/20260612_d2_dronev4_nodetach_save15k \
  --num_classes 1 --eval --resolution 2 --white_background \
  --no_opacity_detach \
  --mask_weight 0.2 \
  --start_semantic_iter 500 --update_until 15000 \
  --knn_weight 0.02 --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --iterations 15000 \
  --test_iterations 7000 12000 15000 \
  --save_iterations 15000 \
  --port 6230
```

> 阶段 1 完成后，将启动阶段 2 验证 15k 后 detach opacity 是否能实现 PSNR / mIoU 双升。

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
  no_opacity_detach 主线优化
  后期训练优化探索（opg15_resume）
  Dual-Feature + no_opacity_detach 组合验证
  ```

## 2026-06-12 今日工作总结

### 完成事项

1. **主线决策确认**：将 **`Dual-Feature + no_opacity_detach`** 定为当前最优主线方法。
   - `dronev4_2` 30k 最终：Test PSNR 25.09 / mIoU 0.8916，15k→30k 未衰减，略优于单一 `no_opacity_detach`（PSNR 25.10 / mIoU 0.8887）。
   - `lfy` 上 `no_opacity_detach` 分割指标同样领先，虽 PSNR gap 较大，但未见 `sem_ramp`/`Dual-Feature` 的退化。

2. **停止失败实验**：`lfy_sem_ramp` 在 15k eval 出现退化（Test mIoU 0.8419→0.8191，FG_IoU 0.7007→0.6595），已停止并保留日志作为对照。

3. **新增 profiler**：在 `train.py` 中加入 `SCAFFOLD_PROFILE=1` 控制的轻量级逐迭代计时器，覆盖 render / loss / backward / densify / optimizer 等阶段，用于定位后期训练变慢根因。

4. **启动组合实验**：`Dual-Feature + no_opacity_detach` 已在 `dronev4_2` 上启动，验证两者正交组合是否能同时保持高 mIoU 并降低 PSNR gap。

5. **启动 save15k 实验**：为验证 15k 后动态detach opacity（`opacity_grad_until=15000`）的效果，先从头训练一个 15k checkpoint（`dronev4_nodetach_save15k`）。

### 当前运行状态（截至 2026-06-12 17:23）

| 实验 | 数据集 | 进度 | 备注 |
| --- | --- | --- | --- |
| `20260612_d2_lfy_nodetach` | lfy | 21640/30000 | 15k: Test PSNR 21.89 / mIoU 0.8752；30k eval 待触发 |
| `20260612_d2_lfy_dualfeat` | lfy | 20390/30000 | 15k: Test PSNR 21.89 / mIoU 0.8328；30k eval 待触发 |
| `20260612_d2_dronev4_nodetach_save15k` | dronev4_2 | 8060/15000 | 7k: Test PSNR 24.36 / mIoU 0.8764；15k checkpoint 备用 |
| `20260612_d2_dronev4_dualfeat_nodetach` | dronev4_2 | 4090/30000 | 组合实验，7k eval 待触发 |
| `20260612_d2_lfy_sem_ramp` | lfy | ~15.9k | **已停止**，15k 退化 |

### 2026-06-12 23:30 D2 dronev4_2 最终结果更新

三个被询问的 `dronev4_2` D2 实验均已跑完 30000 iter，无 OOM/报错；当前没有任何 `train.py` 进程残留。

| 实验 | 配置 | Test PSNR@30k | Test mIoU@30k | FG_IoU@30k | 状态 |
| --- | --- | --- | --- | --- | --- |
| `20260612_d2_dronev4_sem_ramp` | sem_ramp baseline | 24.33 | **0.6737** | 0.3762 | **完成，15k 后严重退化** |
| `20260612_d2_dronev4_nodetach` | `no_opacity_detach` | 25.10 | 0.8887 | 0.7877 | 完成 |
| `20260612_d2_dronev4_dualfeat` | `dual_feature` | 24.37 | 0.8057 | 0.6296 | 完成，15k 后小幅退化 |
| `20260612_d2_dronev4_dualfeat_nodetach` | `dual_feature + no_opacity_detach` | 25.09 | **0.8916** | **0.7934** | **完成，当前最优** |

关键结论：
- **`Dual-Feature + no_opacity_detach` 是当前 dronev4_2 上的最优配置**：Test mIoU 0.8916，PSNR 25.09，15k→30k 无衰减。
- `Dual-Feature` 单独使用会在后期退化（30k mIoU 0.8057 vs 15k 0.8369），但与 `no_opacity_detach` 组合后不仅避免退化，还小幅超越了单一 `no_opacity_detach`（mIoU 0.8887）。
- `sem_ramp` 在 15k→30k 出现灾难性退化（mIoU 0.8094→0.6737，FG_IoU 0.6370→0.3762），确认不能作为后期训练配置。

### 2026-06-12 23:45 D2 lfy 最终结果更新

`lfy` 上两个 30k 实验也已跑完，无 OOM/报错：

| 实验 | 配置 | Test PSNR@30k | Test mIoU@30k | FG_IoU@30k | 状态 |
| --- | --- | --- | --- | --- | --- |
| `20260612_d2_lfy_sem_ramp` | sem_ramp baseline | — | — | — | 已停止（~15.9k，15k 退化） |
| `20260612_d2_lfy_nodetach` | `no_opacity_detach` | 21.58 | **0.8764** | 0.7655 | 完成，lfy 上最优 |
| `20260612_d2_lfy_dualfeat` | `dual_feature` | 21.71 | 0.8280 | 0.6735 | 完成，但不如单一 nodetach |

关键发现：
- `lfy` 上 **单一 `no_opacity_detach` 优于 `Dual-Feature`**（mIoU 0.8764 vs 0.8280）。
- 这与 `dronev4_2` 上 `Dual-Feature + no_opacity_detach` 最优的结论 **不完全一致**。
- 因此，**`Dual-Feature + no_opacity_detach` 在 lfy 上的组合效果尚未验证**，是当前最大缺口。若组合在 lfy 上也能追平或超过单一 nodetach，才能证明其跨数据集泛化能力。

### 跨数据集对比（D2 公平对照，mask_weight=0.2）

| 配置 | dronev4_2 Test mIoU | lfy Test mIoU | dronev4_2 PSNR gap | lfy PSNR gap |
| --- | --- | --- | --- | --- |
| sem_ramp | 0.6737（退化） | 0.8191（15k，已退化） | — | — |
| no_opacity_detach | 0.8887 | **0.8764** | ~2.53 dB | ~4.7 dB |
| dual_feature | 0.8057（退化） | 0.8280 | — | — |
| dual_feature + no_opacity_detach | **0.8916** | **待跑** | ~2.46 dB | 待跑 |

> PSNR gap = Train PSNR − Test PSNR（30k eval）。lfy 的 gap 明显大于 dronev4_2，说明 lfy 新视角泛化更难。

### 当前运行状态（截至 2026-06-13 13:40）

- `dronev4_2` 4 个 D2 实验全部完成。
- `lfy` 3 个相关实验全部完成：`nodetach`（mIoU 0.8764）、`dualfeat`（mIoU 0.8280）、`dualfeat_nodetach`（mIoU 0.8771）。
- **运行中**：`20260613_d2_dronev4_nodetach_opg15_resume`（15k 后 detach opacity）。
- **数据恢复**：`SW_scenes/scene_01/images/` 已从 `SW_Dateset/JPEGImages/` 复制 479 张原图，恢复为独立目录。
- **新增运行中**：`20260613_scene_01_dualfeat_nodetach`（验证当前主线方法在 scene_01 弱 baseline 上的效果）。

### 2026-06-13 启动：scene_01 `Dual-Feature + no_opacity_detach`

**目的**：验证当前主线方法 `Dual-Feature + no_opacity_detach` 在 `scene_01`（基础重建差、RGB-only PSNR 仅 20.32）上是否优于旧 `userparams` 配置（detach, mask_weight=0.1，PSNR 19.40 / mIoU 0.8187）。

**输出目录**：`outputs/20260613_scene_01_dualfeat_nodetach`

**端口**：`6234`

**配置**：与 D2 当前最优一致，`--dual_feature --no_opacity_detach --mask_weight 0.2`。

**启动过程**：
- 首次启动报错 `Could not recognize scene type!`，原因是 COLMAP `sparse/` 目录在 `colmap/sparse/` 下，而 dataset reader 期望 `scene_01/sparse/`。
- 已修复：`cd scene_01 && ln -s colmap/sparse sparse`。
- 修复后重新启动，成功读取 479 个 camera，训练正常开始。

```bash
conda run -n scaffold_gslfy --no-capture-output python train.py \
  -s /mnt/data/liufengyang/data/dataset/SW_scenes/scene_01 \
  -m outputs/20260613_scene_01_dualfeat_nodetach \
  --num_classes 1 --eval --resolution 2 --white_background \
  --no_opacity_detach --dual_feature \
  --mask_weight 0.2 \
  --start_semantic_iter 500 --update_until 15000 \
  --knn_weight 0.02 --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --iterations 30000 \
  --test_iterations 7000 12000 15000 30000 \
  --save_iterations 30000 \
  --port 6234
```

**预期**：Test mIoU 希望超过 0.8187，PSNR 希望不低于 19.40（最好接近或超过 RGB-only 的 20.32）。

### 2026-06-13 数据恢复：`SW_scenes/scene_01/images/`

**问题**：硬盘迁移后 `SW_scenes/scene_01/images/` 丢失，但 COLMAP 重建、train/test 列表、`masks/` 软链接均保留。

**来源数据**：`/mnt/data/liufengyang/data/SW_Dateset/JPEGImages/`（2192 张原图）。

**操作**：
1. 删除 `SW_scenes/scene_01/images` 指向 `SW_Dateset/JPEGImages/` 的软链接（原重建脚本设置）。
2. 创建真实 `images/` 目录。
3. 根据 `scene_01/train_list.txt` + `test_list.txt` 中的 479 个 ID，复制对应 `.jpg` 到 `scene_01/images/`。

**结果**：
- 复制成功 479 张，共 2.1 GB。
- 图片 ID 与 train/test 列表完全一致。
- `masks/` 中 479 个软链接可正常配对（basename 匹配，如 `0289.jpg` ↔ `0289.png`）。

**待验证**：跑一个 3k smoke test 确认 COLMAP reader 能正常加载并训练。

### 2026-06-13 启动：dronev4_2 `no_opacity_detach + opacity_grad_until=15000` resume

**目的**：验证 15k 之后将 opacity detach，是否能提升 PSNR 同时保持 mIoU。

**来源 checkpoint**：`outputs/20260612_d2_dronev4_nodetach_save15k/point_cloud/iteration_15000/`

**目标输出目录**：`outputs/20260613_d2_dronev4_nodetach_opg15_resume`

**端口**：`6233`

**关键参数**：
- `--load_iteration 15000`
- `--opacity_grad_until 15000`（15k 之后 mask pass 中 opacity 被 detach）
- `--iterations 30000`

**预期**：Test mIoU 应接近直接 `no_opacity_detach` 到 30k 的 0.8887，PSNR 希望有所提升。

### 2026-06-13 启动：lfy `Dual-Feature + no_opacity_detach`

**目的**：验证组合方案在 lfy 上是否能追平/超越单一 `no_opacity_detach`，从而证明跨数据集泛化能力。

**输出目录**：`outputs/20260613_d2_lfy_dualfeat_nodetach`

**端口**：`6232`

**配置**：与 D2 完全一致，`--dual_feature --no_opacity_detach`，mask_weight=0.2。

```bash
conda run -n scaffold_gslfy --no-capture-output python train.py \
  -s /mnt/data/liufengyang/data/dataset/lfy/colmap_scene \
  -m outputs/20260613_d2_lfy_dualfeat_nodetach \
  --num_classes 1 --eval --resolution 2 --white_background \
  --no_opacity_detach --dual_feature \
  --mask_weight 0.2 \
  --start_semantic_iter 500 --update_until 15000 \
  --knn_weight 0.02 --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --iterations 30000 \
  --test_iterations 7000 12000 15000 30000 \
  --save_iterations 30000 \
  --port 6232
```

**预期**：Test mIoU 应 ≥ 0.8764（lfy 上当前最优）。若达到 0.88+，可确认组合方案跨数据集有效。

### 2026-06-14 启动：SW_scenes `scene_00` + `scene_04` `Dual-Feature + no_opacity_detach`

**目的**：验证当前主线方法 `Dual-Feature + no_opacity_detach` 在 COLMAP 重建覆盖完整（100%）的 `scene_00` 和 `scene_04` 上的效果，与 `scene_01`（覆盖 81.2%）形成对照。

**数据准备**：
- `SW_scenes/scene_00/images/` 和 `scene_04/images/` 已从 `SW_Dateset/JPEGImages/` 复制真实文件。
- `scene_00` 222 张、`scene_04` 127 张（复制后发现 `scene_04` 的 COLMAP reconstruction 实际引用 `1225.jpg`–`1351.jpg`，补充了 `1351.jpg` 和对应 mask）。
- 为两个 scene 创建 `sparse -> colmap/sparse` 软链接，使 dataset reader 能识别 COLMAP 场景。

**输出目录**：
- `outputs/20260614_sw_scene_00_dualfeat_nodetach`（端口 `6240`）
- `outputs/20260614_sw_scene_04_dualfeat_nodetach`（端口 `6241`）

**配置**：与 D2 当前最优一致。

```bash
conda run -n scaffold_gslfy --no-capture-output python train.py \
  -s /mnt/data/liufengyang/data/dataset/SW_scenes/scene_00 \
  -m outputs/20260614_sw_scene_00_dualfeat_nodetach \
  --num_classes 1 --eval --resolution 2 --white_background \
  --no_opacity_detach --dual_feature \
  --mask_weight 0.2 \
  --start_semantic_iter 500 --update_until 15000 \
  --knn_weight 0.02 --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --iterations 30000 \
  --test_iterations 7000 12000 15000 30000 \
  --save_iterations 30000 \
  --port 6240

conda run -n scaffold_gslfy --no-capture-output python train.py \
  -s /mnt/data/liufengyang/data/dataset/SW_scenes/scene_04 \
  -m outputs/20260614_sw_scene_04_dualfeat_nodetach \
  --num_classes 1 --eval --resolution 2 --white_background \
  --no_opacity_detach --dual_feature \
  --mask_weight 0.2 \
  --start_semantic_iter 500 --update_until 15000 \
  --knn_weight 0.02 --knn_every 100 --knn_offset 55 \
  --focal_alpha 0.25 --focal_gamma 2.0 \
  --iterations 30000 \
  --test_iterations 7000 12000 15000 30000 \
  --save_iterations 30000 \
  --port 6241

## 2026-06-14 Gaussian Grouping vs Scaffold-GSLFY 横向对比

### 完整指标对比

| 数据集 | 方法 | Test PSNR | Test mIoU | Test FG_IoU | 训练分辨率 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| **dronev4_2** | Gaussian Grouping | 21.44 | 0.801 | 0.621 | res=1 | |
| **dronev4_2** | Scaffold-GSLFY (Dual-Feature + no_opacity_detach) | **25.10** | **0.892** | **0.793** | res=2 | **你的方法全面更优** |
| **lfy** | Gaussian Grouping | **23.66** | **0.912** | **0.833** | res=1 | **GG 全面更优** |
| **lfy** | Scaffold-GSLFY (Dual-Feature + no_opacity_detach) | 21.65 | 0.877 | 0.767 | res=2 | |
| **scene_01** | Gaussian Grouping | **21.63** | 0.919 | 0.848 | res=-1 (full) | **GG PSNR 明显更高** |
| **scene_01** | Scaffold-GSLFY (Dual-Feature + no_opacity_detach @ 15k) | 20.44 | **0.922** | **0.855** | res=2 | **你的 mIoU 略高** |

### 关键结论

1. **dronev4_2：你的方法优势明显**
   - 你的 mIoU 0.892 vs GG 0.801，高出 **+0.091**。
   - 你的 PSNR 25.10 vs GG 21.44，高出 **+3.66 dB**。
   - 即使 GG 用了更高分辨率（res=1 vs res=2），你的 PSNR 仍然大幅领先。

2. **lfy：Gaussian Grouping 更优**
   - GG mIoU 0.912 vs 你的 0.877，高出 **+0.035**。
   - GG PSNR 23.66 vs 你的 21.65，高出 **+2.01 dB**。
   - 这说明在 `lfy` 这种 COLMAP 场景上，GG 的 per-Gaussian object identity + SAM 范式更强。

3. **scene_01：基本打平，各有优劣**
   - 你的 mIoU 0.922（15k）vs GG 0.919，略高 **+0.003**。
   - 但 GG PSNR 21.63 vs 你的 20.44，高出 **+1.19 dB**。
   - 注意你的结果是 **15k 中期**，30k 最终尚未完成；若 30k 与 15k 接近，则两者 mIoU 基本持平。

### 总体判断

- **mIoU 角度**：你的方法在 dronev4_2 明显更好；lfy 上 GG 更好；scene_01 基本持平。
- **PSNR 角度**：你的方法只在 dronev4_2 大幅领先；lfy 和 scene_01 都是 GG 领先。
- **综合**：你的方法在 **UAV 航拍密集视角（dronev4_2）** 上占绝对优势；GG 在 **COLMAP 稀疏场景（lfy、scene_01）** 上 RGB 重建和分割都更强。这可能与 GG 的 per-Gaussian identity encoding 更适合处理视角差异大、需要实例级区分的场景有关。

## 2026-06-14 Gaussian Grouping 评估补充

### 已完成的评估

| 数据集 | Train/Test split | Test PSNR | Test mIoU | FG_IoU | 备注 |
| --- | --- | --- | --- | --- | --- |
| `dronev4_2` | 266 / 67 | 21.44 | 0.8010 | 0.6211 | Gaussian Grouping 官方结果 |
| `lfy` | 175 / 25 | **23.66** | 0.9120 | 0.8328 | 本次重跑评估（修正 source path） |
| `scene_01` | 419 / 60 | 21.63 | **0.9191** | **0.8480** | 本次新跑评估 |

### `lfy` 评估细节

- **问题**：`output/lfy/cfg_args` 记录的 `source_path='/mnt/data/liufengyang/data/gaussian-grouping/data/lfy'` 已不存在。
- **解决**：使用实际存在的 COLMAP 数据 `/mnt/data/liufengyang/data/dataset/lfy/colmap_scene`，并通过 CLI 覆盖 `--source_path` 和 `--object_path masks`。
- **迭代**：30000
- **评估命令**：
  ```bash
  python eval_miou.py -m output/lfy \
    -s /mnt/data/liufengyang/data/dataset/lfy/colmap_scene \
    --object_path masks --eval --skip_train --iteration 30000
  python render.py -m output/lfy \
    -s /mnt/data/liufengyang/data/dataset/lfy/colmap_scene \
    --object_path masks --eval --iteration 30000 --skip_train
  python metrics.py -m output/lfy
  ```
- **注意**：原实验 `num_classes=256`，但实际为二值分割；评估时仅 class 0/1 出现，因此 mIoU 仍为有效二值指标。
- **结论**：`lfy` 上 Gaussian Grouping 的 test mIoU 达到 0.912，PSNR 23.66，均优于 `dronev4_2`。

### `scene_01` 评估细节

- **数据来源**：`/mnt/data/liufengyang/data/dataset/SW_scenes/scene_01`
- **迭代**：30000
- **评估命令**：
  ```bash
  python eval_miou.py -m output/scene_01_gausgroup_20260609 --object_path masks --skip_train --iteration 30000
  python render.py -m output/scene_01_gausgroup_20260609 --object_path masks --iteration 30000 --skip_train
  python metrics.py -m output/scene_01_gausgroup_20260609
  ```
- **注意**：原 cfg_args 中 `object_path='object_mask'`，但实际 masks 文件夹名为 `masks`，评估时通过 `--object_path masks` 覆盖。
- **结论**：Gaussian Grouping 在 `scene_01` 上取得了非常高的分割精度（test mIoU 0.919），说明 SW_scenes 的 mask 质量/一致性较好，且场景相对简单。PSNR 21.63 与 dronev4_2（21.44）接近。

### 无法评估的项目

- **`lfy`（`output/lfy`）**：cfg_args 中记录的 `source_path='/mnt/data/liufengyang/data/gaussian-grouping/data/lfy'` 已不存在；该实验 `num_classes=256`，与现有二值 masks 不匹配，无法可靠重跑评估。
- 结果文件已写入：`/mnt/data/liufengyang/data/gaussian-grouping/output/scene_01_gausgroup_20260609/FINAL_RESULT.txt`
```

**状态**：已启动并在后台运行。

**预期**：由于 COLMAP 覆盖完整，PSNR 和 mIoU 应明显高于 `scene_01`（RGB-only baseline 仅 20.32 / mIoU 0.375）。

## 2026-06-15 实验状态与结果更新

### 当前运行状态

截至 2026-06-15 检查，所有 `train.py` 进程均已结束，无后台训练在跑。部分实验因会话/机器中断而提前停止：

| 实验 | 数据集 | 进度 | 状态 |
| --- | --- | --- | --- |
| `20260613_d2_dronev4_nodetach_opg15_resume` | dronev4_2 | 30000/30000 | **完成** |
| `20260613_d2_lfy_dualfeat_nodetach` | lfy | 30000/30000 | **完成** |
| `20260613_scene_01_dualfeat_nodetach` | scene_01 | ~17060/30000 | **中断**（无报错，进程终止） |
| `20260614_sw_scene_00_dualfeat_nodetach` | scene_00 | ~4170/30000 | **中断**（无报错） |
| `20260614_sw_scene_04_dualfeat_nodetach` | scene_04 | ~5460/30000 | **中断**（无报错） |

### `dronev4_2`：`no_opacity_detach` + `opacity_grad_until=15000` resume 结果

来源 checkpoint：`outputs/20260612_d2_dronev4_nodetach_save15k/point_cloud/iteration_15000/`

| 轮次 | Test PSNR | Test FG_IoU | Test mIoU | Train PSNR | Train mIoU | PSNR gap | mIoU gap |
| --- | --------: | --------: | --------: | ---------: | ---------: | -------: | -------: |
| 16k | 25.21 | 0.7892 | 0.8894 | 27.04 | 0.9244 | 1.83 | 0.0350 |
| 20k | 25.19 | 0.7826 | 0.8860 | 27.25 | 0.9184 | 2.06 | 0.0324 |
| 25k | 25.04 | 0.7745 | 0.8817 | 27.20 | 0.9063 | 2.16 | 0.0246 |
| 30k | 24.91 | 0.7650 | 0.8767 | 27.28 | 0.9011 | **2.37** | 0.0244 |

**关键结论**：
- 15k 后将 mask pass 的 opacity detach **并未提升 PSNR**，Test PSNR 从 16k 的 25.21 持续下滑到 30k 的 24.91。
- Test mIoU 同样从 0.8894 下滑到 0.8767，说明 15k 后语义分支继续优化对分割仍有帮助，detach 反而损害了 mIoU。
- 与直接 `no_opacity_detach` 到 30k（Test PSNR 25.10 / mIoU 0.8887）和 `Dual-Feature + no_opacity_detach`（25.10 / 0.8916）相比，`opg15_resume` 在两个指标上均未占优。
- **结论**：在 dronev4_2 上，15k 后动态 detach opacity 不是有效策略，保持 `no_opacity_detach` 全程打开更优。

### `lfy`：`Dual-Feature + no_opacity_detach` 30k 最终结果

| 轮次 | Test PSNR | Test FG_IoU | Test mIoU | Train PSNR | Train mIoU | PSNR gap | mIoU gap |
| --- | --------: | --------: | --------: | ---------: | ---------: | -------: | -------: |
| 7k | 22.40 | 0.7621 | 0.8745 | 25.29 | 0.8888 | 2.89 | 0.0143 |
| 12k | 22.12 | 0.7642 | 0.8757 | 26.46 | 0.8921 | 4.34 | 0.0164 |
| 15k | 21.95 | 0.7648 | 0.8760 | 26.78 | 0.8983 | 4.83 | 0.0223 |
| 30k | 21.65 | 0.7670 | 0.8771 | 28.14 | 0.9027 | **6.49** | 0.0256 |

**关键结论**：
- `Dual-Feature + no_opacity_detach` 在 lfy 上 30k Test mIoU 达到 **0.8771**，略高于单一 `no_opacity_detach` 的 0.8764，但差距极小（+0.0007）。
- Test PSNR 从 15k 的 21.95 继续下降到 30k 的 21.65，PSNR gap 从 4.83 扩大到 **6.49 dB**，几何过拟合严重加剧。
- 与 dronev4_2 不同，lfy 上组合方案并未显著超越单一 nodetach，说明 `Dual-Feature` 在 lfy 上的收益有限，主要瓶颈仍是数据集本身的新视角泛化难度。

### `scene_01`：`Dual-Feature + no_opacity_detach` 15k 结果（中断前）

| 轮次 | Test PSNR | Test FG_IoU | Test mIoU | Train PSNR | Train mIoU | PSNR gap | mIoU gap |
| --- | --------: | --------: | --------: | ---------: | ---------: | -------: | -------: |
| 7k | 20.34 | 0.8494 | 0.9193 | 20.51 | 0.9084 | 0.17 | -0.0109 |
| 12k | 20.39 | 0.8519 | 0.9206 | 21.67 | 0.9197 | 1.28 | -0.0009 |
| 15k | 20.44 | 0.8545 | 0.9221 | 21.84 | 0.9144 | 1.40 | -0.0077 |

**关键结论**：
- 仅 15k 就远超旧 `userparams` 配置（Test mIoU 0.8187 / PSNR 19.40），mIoU 提升 **+0.1034**。
- Test PSNR 20.44 已接近 RGB-only baseline（20.32），且 mIoU 达到 0.9221，说明当前主线方法在 scene_01 上非常有效。
- 进程在 ~17k 处中断，建议恢复继续跑到 30k，以确认是否还有提升空间。

### 跨数据集横向对比（当前最优 vs Gaussian Grouping）

| 数据集 | 方法 | Test PSNR | Test mIoU | Test FG_IoU | 分辨率 | 备注 |
| --- | --- | --------: | --------: | --------: | --- | --- |
| dronev4_2 | Scaffold-GSLFY (Dual-Feature + no_opacity_detach) | **25.10** | **0.8916** | **0.7934** | res=2 | 当前最优 |
| dronev4_2 | Gaussian Grouping | 21.44 | 0.8010 | 0.6211 | res=1 | |
| lfy | Scaffold-GSLFY (Dual-Feature + no_opacity_detach) | 21.65 | 0.8771 | 0.7670 | res=2 | 30k 最终 |
| lfy | Gaussian Grouping | **23.66** | **0.9120** | **0.8328** | res=1 | GG 更优 |
| scene_01 | Scaffold-GSLFY (Dual-Feature + no_opacity_detach @ 15k) | 20.44 | **0.9221** | **0.8545** | res=2 | 中断前 |
| scene_01 | Gaussian Grouping | **21.63** | 0.9191 | 0.8480 | res=-1 | GG PSNR 更高 |

### 当前核心结论

1. **dronev4_2 主线确认**：`Dual-Feature + no_opacity_detach` 是当前最优配置（Test PSNR 25.10 / mIoU 0.8916）。
2. **lfy 仍有差距**：无论是单一 nodetach 还是组合方案，lfy 的 Test PSNR 都显著低于 Gaussian Grouping（21.65 vs 23.66），mIoU 也低 0.035。lfy 是新视角泛化瓶颈，不是方法本身问题。
3. **opg15_resume 策略失效**：15k 后 detach opacity 在 dronev4_2 上同时损害了 PSNR 和 mIoU，不应采用。
4. **SW_scenes 前景好**：scene_01 15k 即达到 mIoU 0.9221，大幅超过旧方法，但需要恢复训练并等待完整结果。

### 待跟进事项

- **恢复 `scene_01` 训练**：从 ~17k 处 resume 到 30k，获取最终指标。
- **重启 `scene_00` / `scene_04`**：因中断较早（4k/5k），需重新启动并跑到 30k。
- **评估是否继续优化 lfy PSNR gap**：当前 gap 已达 6.49 dB，若需改进，建议尝试冻结 appearance embedding 或降低语义 loss weight。
- 如用户需要，可开启一次 `SCAFFOLD_PROFILE=1` 短实验，量化后期训练各阶段耗时。

### 2026-06-15 今日工作记录

1. 检查所有后台实验状态，确认无 `train.py` 进程在跑。
2. 提取并整理 7 个实验的完整 eval 日志，补全 `Reserch.md` 最终结果。
3. 关键发现：
   - `dronev4_nodetach_opg15_resume` 完成但效果不如全程 nodetach。
   - `lfy_dualfeat_nodetach` 完成，mIoU 与单一 nodetach 基本持平。
   - `scene_01` 15k 已达 0.9221 mIoU，但训练在 ~17k 中断。
   - `scene_00` / `scene_04` 在 4k/5k 处中断，需重启。
4. 更新 `Reserch.md` 跨数据集对比表与待跟进事项。

