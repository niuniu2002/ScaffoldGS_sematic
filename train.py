#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Extended for 3D semantic segmentation (see PROJECT.md for details).
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np

import subprocess
try:
    cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
except:
    pass

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torch.nn.functional as F
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import geom_transform_points
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import SegmentationHead
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')


def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """Binary Dice loss for segmentation.

    Dice = 2 * |pred ∩ target| / (|pred| + |target|)
    Loss = 1 - Dice

    Particularly effective for highly imbalanced foreground/background.
    """
    pred = pred.float().view(-1)
    target = target.float().view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1.0 - dice


def focal_loss(pred: torch.Tensor,
               target: torch.Tensor,
               alpha: float = 0.25,
               gamma: float = 2.0,
               reduction: str = 'mean') -> torch.Tensor:
    """Binary focal loss on probabilities.

    Formula:
        Loss = - alpha_t * (1 - pt)^gamma * log(pt)
    where pt = p if y=1 else (1-p), and alpha_t = alpha if y=1 else (1-alpha).

    Supports reduction='none' for pixel-wise weighting.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    pred = pred.float()
    target = target.float()

    pt = pred * target + (1.0 - pred) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = -alpha_t * torch.pow(1.0 - pt, gamma) * torch.log(pt)

    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def compute_anchor_fg_voting_loss(
    xyz: torch.Tensor,
    seg_logits: torch.Tensor,
    opacity: torch.Tensor,
    anchor_idx: torch.Tensor,
    viewpoint_cam,
    gt_mask: torch.Tensor,
    fg_ratio_thr: float = 0.3,
    max_samples: int = 4096,
    detach_xyz: bool = True,
):
    """
    Foreground-only Anchor Voting Loss.

    For each anchor, projects its child Gaussians to 2D and samples the GT mask.
    Computes a per-anchor foreground ratio using opacity*confidence weighting.
    Only anchors with anchor_fg_ratio > fg_ratio_thr are supervised (target=1).
    Background supervision is left to the Rendered Mask Loss (focal+dice).

    Args:
        xyz:          [M, 3]   child Gaussian world coordinates
        seg_logits:   [M, 1]   pre-sigmoid segmentation logits
        opacity:      [M, 1]   child Gaussian opacity
        anchor_idx:   [M]      anchor index for each child Gaussian
        viewpoint_cam: Camera object with full_proj_transform
        gt_mask:      [1, H, W] ground-truth semantic mask (0=bg, 1=fg)
        fg_ratio_thr: float, threshold to qualify as a foreground anchor
        max_samples:  int, max number of FG anchors to supervise per iteration
        detach_xyz:   bool, whether to detach xyz before projection

    Returns:
        loss: scalar tensor (0.0 if no FG anchors)
        info: dict with fg_anchor_count, anchor_fg_ratio_mean,
              pred_fg_mean_on_fg_anchor, anchor_fg_loss
    """
    info = {
        "fg_anchor_count": 0,
        "anchor_fg_ratio_mean": 0.0,
        "pred_fg_mean_on_fg_anchor": 0.0,
        "anchor_fg_loss": 0.0,
    }

    if xyz.shape[0] == 0 or gt_mask is None:
        return torch.tensor(0.0, device=xyz.device), info

    if detach_xyz:
        xyz = xyz.detach()

    n_anchors = int(anchor_idx.max().item()) + 1

    # 1. Project child Gaussians to NDC
    p_ndc = geom_transform_points(xyz, viewpoint_cam.full_proj_transform)  # [M, 3]

    in_image = (p_ndc[:, 0] >= -1.0) & (p_ndc[:, 0] <= 1.0) & \
               (p_ndc[:, 1] >= -1.0) & (p_ndc[:, 1] <= 1.0) & \
               (p_ndc[:, 2] >= 0.0)  & (p_ndc[:, 2] <= 1.0)

    # 2. Sample GT mask at projected child positions
    gt_mask_norm = gt_mask.float()
    if gt_mask_norm.max() > 1.0:
        gt_mask_norm = gt_mask_norm / 255.0

    grid_2d = torch.stack([p_ndc[:, 0], -p_ndc[:, 1]], dim=-1)  # [M, 2]
    grid_2d = grid_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 2]

    sampled_mask = torch.nn.functional.grid_sample(
        gt_mask_norm.unsqueeze(0),  # [1, 1, H, W]
        grid_2d,  # [1, 1, M, 2]
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    ).squeeze()  # [M]

    # 3. Per-child confidence (how close to 0 or 1)
    conf = torch.abs(sampled_mask - 0.5) * 2.0  # [M]

    # 4. Per-child weight = opacity * confidence
    opacity_squeezed = opacity.view(-1)  # [M]
    weight = opacity_squeezed * conf  # [M]

    # Only keep children that are in image and have non-zero weight
    valid_child = in_image & (weight > 1e-6)
    valid_count = int(valid_child.sum().item())

    if valid_count == 0:
        return torch.tensor(0.0, device=xyz.device), info

    # 5. Aggregate per-anchor foreground ratio
    valid_idx = anchor_idx[valid_child]
    valid_sampled_mask = sampled_mask[valid_child]
    valid_weight = weight[valid_child]

    fg_weight = torch.zeros(n_anchors, device=xyz.device)
    total_weight = torch.zeros(n_anchors, device=xyz.device)

    fg_weight = fg_weight.index_add(0, valid_idx, valid_sampled_mask * valid_weight)
    total_weight = total_weight.index_add(0, valid_idx, valid_weight)

    valid_anchors = total_weight > 1e-6
    anchor_fg_ratio = torch.zeros(n_anchors, device=xyz.device)
    anchor_fg_ratio[valid_anchors] = fg_weight[valid_anchors] / total_weight[valid_anchors].clamp(min=1e-6)

    # 6. Select foreground anchors (anchor_fg_ratio > fg_ratio_thr)
    fg_mask = anchor_fg_ratio > fg_ratio_thr
    fg_anchor_count = int(fg_mask.sum().item())

    info["fg_anchor_count"] = fg_anchor_count

    if fg_anchor_count == 0:
        return torch.tensor(0.0, device=xyz.device), info

    # 7. Get per-anchor seg logits (average over all children; they should be identical for per-anchor seg)
    anchor_logits_sum = torch.zeros(n_anchors, device=xyz.device)
    anchor_logits_count = torch.zeros(n_anchors, device=xyz.device)
    anchor_logits_sum = anchor_logits_sum.index_add(0, anchor_idx, seg_logits.view(-1))
    anchor_logits_count = anchor_logits_count.index_add(0, anchor_idx, torch.ones_like(anchor_idx, dtype=torch.float))
    anchor_logits = anchor_logits_sum / anchor_logits_count.clamp(min=1)

    fg_logits = anchor_logits[fg_mask]
    fg_targets = torch.ones_like(fg_logits)

    # 8. Subsample if too many
    if fg_logits.numel() > max_samples:
        perm = torch.randperm(fg_logits.numel(), device=fg_logits.device)[:max_samples]
        fg_logits = fg_logits[perm]
        fg_targets = fg_targets[perm]
        info["fg_anchor_count"] = fg_logits.numel()

    # 9. BCEWithLogitsLoss on FG anchors only (target=1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(fg_logits, fg_targets, reduction='mean')

    # 10. Debug info
    with torch.no_grad():
        pred_fg_mean = torch.sigmoid(fg_logits).mean().item()
        anchor_fg_ratio_mean = anchor_fg_ratio[fg_mask].mean().item()

    info["anchor_fg_ratio_mean"] = float(anchor_fg_ratio_mean)
    info["pred_fg_mean_on_fg_anchor"] = float(pred_fg_mean)
    info["anchor_fg_loss"] = float(loss.item())

    return loss, info

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None, load_iteration=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                              use_per_gaussian_seg=getattr(dataset, 'use_per_gaussian_seg', False))
    # 如果提供了 load_iteration，则从对应的 point_cloud/iteration_X 加载已有高斯作为初始化
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # [Two-Stage] 如果只训练 segmentation，冻结所有参数并替换 optimizer
    if getattr(opt, 'seg_only', False):
        reuse_head = getattr(opt, 'seg_only_reuse_head', False)
        if reuse_head:
            logger.info("\n[Two-Stage Mode] Reusing original seg head and freezing all other parameters.")
        else:
            logger.info("\n[Two-Stage Mode] Replacing seg head with deep MLP and freezing all other parameters.")
            # 重新初始化更深的分割头（替换从 checkpoint 加载的浅 MLP）
            seg_outputs = gaussians.n_offsets if gaussians.use_per_gaussian_seg else 1
            gaussians.mlp_segmentation = SegmentationHead(
                feat_dim=gaussians.feat_dim,
                hidden_dim=128,
                num_layers=3,
                dropout=0.0,
                num_outputs=seg_outputs,
            ).cuda()
        # Freeze all explicit parameters
        for attr in ('_anchor', '_offset', '_anchor_feat', '_scaling', '_rotation', '_opacity'):
            p = getattr(gaussians, attr, None)
            if p is not None and hasattr(p, 'requires_grad'):
                p.requires_grad = False
        # Freeze all MLPs except the new seg head
        for mlp in (gaussians.mlp_opacity, gaussians.mlp_cov, gaussians.mlp_color):
            for p in mlp.parameters():
                p.requires_grad = False
        if gaussians.use_feat_bank:
            for p in gaussians.mlp_feature_bank.parameters():
                p.requires_grad = False
        if gaussians.appearance_dim > 0 and gaussians.embedding_appearance is not None:
            for p in gaussians.embedding_appearance.parameters():
                p.requires_grad = False
        # New seg head remains trainable (default requires_grad=True)
        for param in gaussians.mlp_segmentation.parameters():
            param.requires_grad = True
        # 替换为只包含 segmentation 的 optimizer
        gaussians.optimizer = torch.optim.Adam(
            gaussians.mlp_segmentation.parameters(),
            lr=0.001,
            eps=1e-15
        )
        logger.info(f"  Optimizable params: {sum(p.numel() for p in gaussians.mlp_segmentation.parameters())}")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    def _ramp_factor(iteration: int, warmup: int, ramp: int) -> float:
        if warmup is None:
            warmup = 0
        if ramp is None:
            ramp = 0
        if iteration < warmup:
            return 0.0
        if ramp <= 0:
            return 1.0
        return float(min(1.0, (iteration - warmup) / float(ramp)))

    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        # Scaling regularization: exponentially decay weight over training to preserve thin structures.
        scaling_reg_start = float(getattr(opt, "scaling_reg_start", 0.01))
        scaling_reg_end = float(getattr(opt, "scaling_reg_end", 0.001))
        total_iters = max(1, int(getattr(opt, "iterations", 30_000)))
        t = float(iteration) / float(total_iters)
        # Exponential interpolation: w(t)=w0*(w1/w0)^t
        scaling_reg_w = scaling_reg_start * ((scaling_reg_end / max(1e-12, scaling_reg_start)) ** t)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + scaling_reg_w * scaling_reg

        # =========================================================================================
        # [王子殿下的专属修正版] Mask Loss + KNN Consistency Loss
        # =========================================================================================

        # 1. Mask Loss (权重可配置 + 硬性语义预热 + 可升温)
        pred_mask = render_pkg.get("mask", None)
        gt_mask = viewpoint_cam.semantic_mask.cuda() if getattr(viewpoint_cam, "semantic_mask", None) is not None else None

        # Hard semantic warmup: before start_semantic_iter, skip mask/knn losses entirely.
        start_semantic_iter = int(getattr(opt, "start_semantic_iter", 7000))
        semantic_enabled = (iteration >= start_semantic_iter)
        semantic_phase_iter = iteration - start_semantic_iter  # 0 at first semantic iteration
        
        if semantic_enabled and pred_mask is not None and gt_mask is not None:
            # [王子殿下专属修正] 
            # 加上 epsilon (1e-6) 防止 pred_mask 为 0 时 log(0) 导致 Loss=NaN
            pred_mask_clamped = torch.clamp(pred_mask, min=1e-6, max=1.0-1e-6)
            
            # 确保 gt_mask 也是 float 类型以便计算 loss
            gt_mask_float = gt_mask.float()

            if pred_mask_clamped.shape == gt_mask_float.shape:
                # Uncertainty-aware pixel weighting (edge-robust):
                # gt close to 0.5 => very low weight; gt close to 0/1 => weight -> 1.0
                conf = torch.clamp(torch.abs(gt_mask_float - 0.5) * 2.0, 0.0, 1.0)
                uncertainty_min = float(getattr(opt, "uncertainty_min", 0.1))
                uncertainty_power = float(getattr(opt, "uncertainty_power", 2.0))
                uncertainty_weight = uncertainty_min + (1.0 - uncertainty_min) * torch.pow(conf, uncertainty_power)

                # [新增] 语义权重图加权（Semantic Weight Map）
                sem_weight_map = viewpoint_cam.semantic_weight if hasattr(viewpoint_cam, 'semantic_weight') else None
                if getattr(opt, 'use_semantic_weight', False) and sem_weight_map is not None:
                    sem_w = sem_weight_map.cuda().float()
                    # 语义权重图: 像素值表示外部模型对该像素的置信度
                    # 0.5 为决策边界，越接近 0 或 1 越确定
                    abs_conf = torch.abs(sem_w - 0.5) * 2.0  # [0, 1], 1=最确定

                    strategy = getattr(opt, 'sem_weight_strategy', 'hard')
                    if strategy == 'hard':
                        high_t = float(getattr(opt, 'sem_weight_high', 0.7))
                        low_t = float(getattr(opt, 'sem_weight_low', 0.1))
                        boost = float(getattr(opt, 'sem_weight_boost', 2.0))
                        sem_pixel_weight = torch.zeros_like(abs_conf)
                        sem_pixel_weight[abs_conf >= high_t] = 1.0
                        sem_pixel_weight[(abs_conf >= low_t) & (abs_conf < high_t)] = boost
                        # abs_conf < low_t 保持 0（忽略）
                    elif strategy == 'conservative_hard':
                        high_t = float(getattr(opt, 'sem_weight_high', 0.7))
                        boost = float(getattr(opt, 'sem_weight_boost', 2.0))
                        # [实验4] 保守策略: 不忽略任何像素, 低置信度保持1.0, 高置信度增强
                        sem_pixel_weight = torch.ones_like(abs_conf)
                        sem_pixel_weight[abs_conf >= high_t] = boost
                    else:  # smooth
                        peak = float(getattr(opt, 'sem_weight_smooth_peak', 0.5))
                        # 倒抛物线: 在 peak 处权重最大
                        denom = peak if peak > 1e-6 else 1e-6
                        sem_pixel_weight = 1.0 - 4.0 * ((abs_conf - peak) / denom) ** 2
                        sem_pixel_weight = torch.clamp(sem_pixel_weight, min=0.0)
                        # 高置信度区域保底权重 1.0
                        high_t = float(getattr(opt, 'sem_weight_high', 0.7))
                        sem_pixel_weight = torch.where(abs_conf >= high_t, torch.ones_like(sem_pixel_weight), sem_pixel_weight)

                    # 将语义权重与现有 uncertainty_weight 相乘（叠加外部先验）
                    uncertainty_weight = uncertainty_weight * sem_pixel_weight

                    if iteration % 100 == 0:
                        active_ratio = (sem_pixel_weight > 0).float().mean().item()
                        boost_ratio = (sem_pixel_weight > 1.0).float().mean().item()
                        print(f"Iter {iteration}: SemanticWeight active={active_ratio:.2%} boost={boost_ratio:.2%}")

                # Pixel-wise focal loss (handles class imbalance)
                raw_loss = focal_loss(
                    pred_mask_clamped,
                    gt_mask_float,
                    alpha=float(getattr(opt, "focal_alpha", 0.25)),
                    gamma=float(getattr(opt, "focal_gamma", 2.0)),
                    reduction='none',
                )
                loss_focal = (raw_loss * uncertainty_weight).mean()

                # Dice loss (complements focal loss for small foreground regions)
                loss_dice = dice_loss(pred_mask_clamped, gt_mask_float, smooth=1.0)

                # Combined: focal + dice (equal weighting by default)
                dice_weight = float(getattr(opt, "dice_weight", 1.0))
                loss_seg = loss_focal + dice_weight * loss_dice

                seg_w = float(getattr(opt, "mask_weight", 0.0)) * _ramp_factor(
                    semantic_phase_iter,
                    int(getattr(opt, "mask_warmup", 0)),
                    int(getattr(opt, "mask_ramp", 0)),
                )
                if seg_w > 0:
                    loss = loss + seg_w * loss_seg

                if iteration % 100 == 0:
                    print(f"Iter {iteration}: Mask Loss = {loss_seg.item():.5f} (focal={loss_focal.item():.5f}, dice={loss_dice.item():.5f}, w={seg_w:.4f})")

        # 2. KNN Consistency Loss (权重可配置 + 硬性语义预热 + 可升温 + 可调频率)
        knn_every = int(getattr(opt, "knn_every", 0))
        knn_offset = int(getattr(opt, "knn_offset", 0))
        do_knn = (knn_every > 0) and ((iteration - knn_offset) % knn_every == 0)
        if semantic_enabled and gt_mask is not None and do_knn:
            try:
                # 获取锚点数据 (保持在 CUDA)
                anchor_xyz = gaussians.get_anchor.detach()
                # [改进] 使用 logit-space 的 segmentation 值，避免概率空间 MSE 推向 0.5
                anchor_seg_logit = gaussians.get_anchor_seg_logits_raw()

                # 长度一致性检查 (防止剪枝后显存未同步)
                if anchor_xyz.shape[0] != anchor_seg_logit.shape[0]:
                    pass
                else:
                    N = anchor_xyz.shape[0]
                    if N > 1:
                        # 更保守的采样上限，减少内存压力
                        n_samples = min(1024, N)
                        indices = torch.randperm(N, device=anchor_xyz.device)[:n_samples]

                        # anchor_xyz 作为位置只需副本，不影响梯度路径
                        anchor_xyz_safe = anchor_xyz.detach().clone().contiguous()
                        sample_xyz = anchor_xyz_safe[indices].contiguous()

                        # anchor_seg_logit 保留梯度，用于更新 mlp_segmentation
                        sample_logit = anchor_seg_logit[indices].contiguous()

                        # 分块计算距离以降低峰值显存占用
                        chunk_size = 512
                        dists_chunks = []
                        try:
                            for i in range(0, N, chunk_size):
                                part = anchor_xyz_safe[i:i+chunk_size].contiguous()
                                # 这里分配 (n_samples x chunk_size) 的临时张量
                                d = torch.cdist(sample_xyz, part)
                                dists_chunks.append(d)

                            dists = torch.cat(dists_chunks, dim=1)  # [n_samples, N]

                            # 找最近的 k 个 (包含自己)
                            k = min(6, N)
                            try:
                                topk_val, topk_inds = torch.topk(dists, k=k, dim=1, largest=False)
                            except Exception as e_topk:
                                print(f"[Warning] topk failed in KNN at iter {iteration}: {e_topk}")
                                del dists, dists_chunks
                                raise e_topk

                            if k > 1:
                                neighbor_inds = topk_inds[:, 1:]
                                neighbor_logit = anchor_seg_logit[neighbor_inds]

                                # [改进 v2] 概率空间对称 BCE KNN：
                                # 数值范围可控，不会推向 0.5，而是强制相邻 anchor 趋向一致（都 0 或都 1）
                                sample_prob = torch.sigmoid(sample_logit).unsqueeze(1)      # [n_samples, 1]
                                neighbor_prob = torch.sigmoid(neighbor_logit)               # [n_samples, k-1]
                                eps = 1e-6
                                # 对称 BCE = [BCE(p_i||p_j) + BCE(p_j||p_i)] / 2
                                loss_knn = -0.5 * (
                                    sample_prob * torch.log(neighbor_prob + eps)
                                    + (1.0 - sample_prob) * torch.log(1.0 - neighbor_prob + eps)
                                    + neighbor_prob * torch.log(sample_prob + eps)
                                    + (1.0 - neighbor_prob) * torch.log(1.0 - sample_prob + eps)
                                )
                                loss_knn = loss_knn.mean()

                                knn_w = float(getattr(opt, "knn_weight", 0.0)) * _ramp_factor(
                                    semantic_phase_iter,
                                    int(getattr(opt, "knn_warmup", 0)),
                                    int(getattr(opt, "knn_ramp", 0)),
                                )
                                if knn_w > 0:
                                    loss = loss + knn_w * loss_knn

                                print(f"Iter {iteration}: KNN(BCE) Loss = {loss_knn.item():.6f} (w={knn_w:.4f}, every={knn_every}, offset={knn_offset})")

                            # 及时清理中间变量并释放显存
                            del dists, topk_inds, topk_val, dists_chunks
                        except RuntimeError as rte:
                            print(f"[Warning] KNN computation runtime error at iter {iteration}: {rte}")
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            # 跳过本次 KNN
                        except Exception as e_any:
                            print(f"[Warning] KNN computation skipped at iter {iteration}: {e_any}")
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass

            except Exception as e:
                print(f"[Warning] KNN Loss skipped at iter {iteration}: {e}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        
        # =========================================================================================

        # 3. Foreground-only Anchor Voting Loss
        ar_weight = float(getattr(opt, "anchor_fg_weight", 0.0))
        ar_start = int(getattr(opt, "anchor_fg_start_iter", 5000))
        ar_every = int(getattr(opt, "anchor_fg_every", 10))
        ar_fg_ratio_thr = float(getattr(opt, "anchor_fg_ratio_thr", 0.3))
        ar_max_samples = int(getattr(opt, "anchor_fg_max_samples", 4096))
        ar_detach_xyz = bool(getattr(opt, "anchor_fg_detach_xyz", True))
        ar_ramp = int(getattr(opt, "anchor_fg_ramp", 1000))

        do_ar = (ar_weight > 0.0) and (iteration >= ar_start) and ((iteration - ar_start) % ar_every == 0)

        if do_ar and gt_mask is not None:
            try:
                neural_xyz = render_pkg.get("neural_xyz")
                neural_seg_logits = render_pkg.get("neural_seg_logits")
                neural_opacity_filtered = render_pkg.get("neural_opacity_filtered")
                neural_anchor_idx = render_pkg.get("neural_anchor_idx")

                if (neural_xyz is not None and neural_seg_logits is not None and
                    neural_opacity_filtered is not None and neural_anchor_idx is not None):
                    loss_ar, ar_info = compute_anchor_fg_voting_loss(
                        xyz=neural_xyz,
                        seg_logits=neural_seg_logits,
                        opacity=neural_opacity_filtered,
                        anchor_idx=neural_anchor_idx,
                        viewpoint_cam=viewpoint_cam,
                        gt_mask=gt_mask,
                        fg_ratio_thr=ar_fg_ratio_thr,
                        max_samples=ar_max_samples,
                        detach_xyz=ar_detach_xyz,
                    )

                    # Safe linear ramp
                    ar_w = ar_weight * min(1.0, max(0.0, (iteration - ar_start) / max(ar_ramp, 1)))

                    if ar_w > 0 and loss_ar.item() > 0:
                        loss = loss + ar_w * loss_ar

                    if iteration % 100 == 0:
                        fg_count = ar_info.get("fg_anchor_count", 0)
                        fg_ratio_mean = ar_info.get("anchor_fg_ratio_mean", 0.0)
                        pred_fg_mean = ar_info.get("pred_fg_mean_on_fg_anchor", 0.0)
                        ar_loss_val = ar_info.get("anchor_fg_loss", 0.0)
                        print(
                            f"Iter {iteration}: FG-Anchor-Vote Loss={ar_loss_val:.5f} "
                            f"w={ar_w:.4f} fg_count={fg_count} "
                            f"fg_ratio_mean={fg_ratio_mean:.3f} pred_fg_mean={pred_fg_mean:.3f}"
                        )
                        if fg_count == 0:
                            print(
                                f"[WARNING] Iter {iteration}: FG-Anchor fg_count=0! "
                                f"Check projection/fg_ratio_thr."
                            )
            except Exception as e:
                print(f"[Warning] FG-Anchor-Vote Loss skipped at iter {iteration}: {e}")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)

            # TensorBoard: visualize current view's GT / predicted masks during training
            # Log masks only every 1000 iterations to avoid excessive I/O
            if tb_writer is not None and pred_mask is not None and gt_mask is not None and iteration % 1000 == 0:
                try:
                    # add_images expects NCHW; both gt_mask and pred_mask are [1, H, W]
                    tb_writer.add_images(f"{dataset_name}/train_view/mask_gt", gt_mask.unsqueeze(0), iteration)
                    tb_writer.add_images(f"{dataset_name}/train_view/mask_pred", pred_mask.clamp(0.0, 1.0).unsqueeze(0), iteration)
                    # Optionally visualize masked RGB (prediction overlay); image is [3, H, W]
                    masked_image = image * pred_mask.clamp(0.0, 1.0)
                    tb_writer.add_images(f"{dataset_name}/train_view/masked_rgb", masked_image.unsqueeze(0), iteration)
                except Exception as e:
                    logger.warning(f"TensorBoard mask logging failed at iter {iteration}: {e}")
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if not getattr(opt, 'seg_only', False) and iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    """
    王子殿下专属增强版报告系统：
    - 实时监控 PSNR (峰值信噪比) 变化曲线
    - 实时监控 mIoU (语义分割精度) 变化曲线
    - 自动同步高清渲染图与误差图
    """
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss": Ll1, 'train_total_loss': loss})
    
    # 到了测试节点（如 7000, 30000 步），开始深度检阅
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()}, 
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                miou_test = 0.0 # 语义精度指标
                
                for idx, viewpoint in enumerate(config['cameras']):
                    # 执行渲染
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                    
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # 1. 计算重建精度 (PSNR)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    # 2. 计算语义精度 (mIoU)
                    pred_mask = render_pkg.get("mask", None)
                    gt_mask = getattr(viewpoint, "semantic_mask", None)
                    if pred_mask is not None and gt_mask is not None:
                        gt_mask_cuda = gt_mask.cuda().float()
                        # 二值化处理：大于 0.5 视为目标
                        p_mask = (pred_mask > 0.5).float()
                        g_mask = (gt_mask_cuda > 0.5).float()
                        
                        intersection = (p_mask * g_mask).sum()
                        union = p_mask.sum() + g_mask.sum() - intersection
                        miou_test += (intersection + 1e-6) / (union + 1e-6)

                    # 向 TensorBoard 实时推送前 30 张渲染图
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/' + config['name'] + f"_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/' + config['name'] + f"_view_{viewpoint.image_name}/errormap", (gt_image[None] - image[None]).abs(), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/' + config['name'] + f"_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)

                # 计算所有评估相机均值
                cam_count = len(config['cameras'])
                psnr_test /= cam_count
                l1_test /= cam_count
                miou_test /= cam_count
                
                # TensorBoard 曲线绘制
                if tb_writer:
                    # PSNR 曲线
                    tb_writer.add_scalar(f'{dataset_name}/{config["name"]}/Accuracy_PSNR', psnr_test, iteration)
                    # mIoU 曲线（有语义时才记录）
                    if miou_test > 0:
                        tb_writer.add_scalar(f'{dataset_name}/{config["name"]}/Semantic_mIoU', miou_test, iteration)
                    # 原有 L1 曲线
                    tb_writer.add_scalar(f'{dataset_name}/' + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)

                logger.info(f"\n[ITER {iteration}] 检阅 {config['name']}: PSNR {psnr_test:.2f} dB, mIoU {miou_test:.4f}")
                
                if wandb is not None:
                    wandb.log({f"{config['name']}_PSNR": psnr_test, f"{config['name']}_mIoU": miou_test})

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/total_points', scene.gaussians.get_anchor.shape[0], iteration)
        
        torch.cuda.empty_cache()
        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
                              use_per_gaussian_seg=getattr(dataset, 'use_per_gaussian_seg', False))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # 默认在 7000 和 30000 两个迭代点做测试 / 保存
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default=None, help="Resume training from existing point_cloud/iteration_X as initialization")
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    # [王子殿下专属修正] 注释掉下面这块代码，禁止自动备份代码
    # try:
    #     saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    # except:
    #     logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger, load_iteration=args.load_iteration)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path, load_iteration=args.load_iteration)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")