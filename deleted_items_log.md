# 删除记录

> 删除时间：2026-05-11
> 操作：清理跑失败的实验目录、垃圾日志文件、备份文件及缓存

---

## 一、跑失败的实验目录

### 1. `output/dronev4_2_debug_anchor_region/`
- **状态**：训练未完成（无 point_cloud 输出，无 eval 结果）
- **大小**：4.4M
- **删除原因**：启动后即刻终止，未产生有效模型
- **实验参数**：
  - `sh_degree=3, feat_dim=32, n_offsets=10`
  - `voxel_size=0.001, update_depth=3, update_init_factor=16, update_hierachy_factor=4`
  - `appearance_dim=32, resolution=2, white_background=True`
  - `use_per_gaussian_seg=False`
  - `model_path='output/dronev4_2_debug_anchor_region'`

### 2. `output/dronev4_2_debug_anchor_region_v2/`
- **状态**：训练未完成（仅含 outputs.log，无模型输出）
- **大小**：4.5K
- **删除原因**：启动后即刻终止，无 cfg_args，无有效模型
- **实验参数**：无（目录内未生成 cfg_args）

### 3. `output/dronev4_2_debug_anchor_region_v3/`
- **状态**：训练到 Iter 3000 后渲染阶段崩溃
- **大小**：84M
- **删除原因**：`RuntimeError: forward() expected at most 2 argument(s) but received 3 argument(s)`，评估失败，无有效结果
- **实验参数**：
  - `sh_degree=3, feat_dim=32, n_offsets=10`
  - `voxel_size=0.001, update_depth=3, update_init_factor=16, update_hierachy_factor=4`
  - `appearance_dim=32, resolution=2, white_background=True`
  - `use_per_gaussian_seg=False`
  - `model_path='output/dronev4_2_debug_anchor_region_v3'`

---

## 二、崩溃/废弃日志文件

| 文件路径 | 大小 | 删除原因 |
|---------|------|---------|
| `output/dronev4_2_debug_anchor_region.log` | 19K | 对应未完成实验的残留日志 |
| `output/dronev4_2_debug_v2_stderr.log` | 3.6K | 空/残次 stderr 输出 |
| `output/dronev4_2_debug_v2_stdout.log` | 210 bytes | 空 stdout 输出 |
| `output/dronev4_2_debug_v3.log` | 81K | `debug_anchor_region_v3` 崩溃日志 |
| `output/dronev4_2_exp07.log` | 678K | `exp07` 主实验在 Iter 30000 渲染阶段崩溃日志 |

---

## 三、垃圾文件

| 文件路径 | 大小 | 删除原因 |
|---------|------|---------|
| `train.py.bak` | 37K | 旧备份文件（2026-04-20） |
| `__pycache__/` | 109K | Python 字节码缓存 |

---

## 四、保留说明

以下实验**成功完成**，予以保留：
- `output/dronev4_2_exp07_anchor_region_w002/` — Iter 30000，mIoU 70.81%
- `output/dronev4_2_debug_anchor_voting/` — Iter 7000，mIoU 70.76%
