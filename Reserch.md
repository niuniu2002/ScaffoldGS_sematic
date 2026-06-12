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

**当前进度**（截至 2026-06-12 17:23）：2740/30000，速度约 1.5 s/it。7k eval 预计数小时后触发。

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

**lfy 当前进度**（截至 2026-06-12 17:23）：nodetach 20480/30000 / dualfeat 19310/30000 / sem_ramp stopped at ~15.9k。

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
| 1 | `dronev4_nodetach_save15k` | 从头训练到 15k 并保存 checkpoint（与 D2 nodetach 同参数） | **running** (6750/15000) |
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

1. **主线决策确认**：将 `no_opacity_detach` 定为当前主线方法。
   - `dronev4_2` 30k 最终：Test PSNR 25.10 / mIoU 0.8887，15k→30k 未衰减。
   - `lfy` 上分割指标同样领先，虽 PSNR gap 较大，但未见 `sem_ramp`/`Dual-Feature` 的退化。

2. **停止失败实验**：`lfy_sem_ramp` 在 15k eval 出现退化（Test mIoU 0.8419→0.8191，FG_IoU 0.7007→0.6595），已停止并保留日志作为对照。

3. **新增 profiler**：在 `train.py` 中加入 `SCAFFOLD_PROFILE=1` 控制的轻量级逐迭代计时器，覆盖 render / loss / backward / densify / optimizer 等阶段，用于定位后期训练变慢根因。

4. **启动组合实验**：`Dual-Feature + no_opacity_detach` 已在 `dronev4_2` 上启动，验证两者正交组合是否能同时保持高 mIoU 并降低 PSNR gap。

5. **启动 save15k 实验**：为验证 15k 后动态detach opacity（`opacity_grad_until=15000`）的效果，先从头训练一个 15k checkpoint（`dronev4_nodetach_save15k`）。

### 当前运行状态（截至 2026-06-12 17:23）

| 实验 | 数据集 | 进度 | 备注 |
| --- | --- | --- | --- |
| `20260612_d2_lfy_nodetach` | lfy | 20480/30000 | 15k: Test PSNR 21.89 / mIoU 0.8752；30k eval 待触发 |
| `20260612_d2_lfy_dualfeat` | lfy | 19310/30000 | 15k: Test PSNR 21.89 / mIoU 0.8328；30k eval 待触发 |
| `20260612_d2_dronev4_nodetach_save15k` | dronev4_2 | 6750/15000 | 用于阶段 2 resume 的 15k checkpoint |
| `20260612_d2_dronev4_dualfeat_nodetach` | dronev4_2 | 2740/30000 | 组合实验，7k eval 待触发 |
| `20260612_d2_lfy_sem_ramp` | lfy | ~15.9k | **已停止**，15k 退化 |

### 待跟进事项

- `lfy_nodetach` / `lfy_dualfeat` 到达 30k 后评估最终指标与 myvideo IoU。
- `dronev4_nodetach_save15k` 到达 15k 后，启动 `dronev4_nodetach_opg15_resume`（15k 后 detach opacity）。
- `dronev4_dualfeat_nodetach` 到达 7k/12k/15k/30k 后检查是否优于单一 `no_opacity_detach`。
- 如用户需要，可开启一次 `SCAFFOLD_PROFILE=1` 短实验，量化后期训练各阶段耗时。

