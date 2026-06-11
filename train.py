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


def multiclass_dice_loss(pred_probs: torch.Tensor,
                         target: torch.Tensor,
                         num_classes: int,
                         smooth: float = 1.0,
                         ignore_index: int = 255) -> torch.Tensor:
    """Multi-class Dice loss.

    Args:
        pred_probs: [C, H, W] or [C, N] softmax probabilities
        target: [H, W] or [N] long class indices
        num_classes: number of classes
        smooth: Laplace smoothing
        ignore_index: pixels with this label are ignored
    """
    # Flatten to support both [C, H, W] and [C, N]
    pred_probs = pred_probs.reshape(num_classes, -1)
    target = target.reshape(-1)

    valid = target != ignore_index
    if not valid.any():
        return torch.tensor(0.0, device=pred_probs.device, dtype=pred_probs.dtype)

    pred_probs = pred_probs[:, valid]
    target = target[valid]

    target_one_hot = torch.nn.functional.one_hot(target.long(), num_classes).permute(1, 0).float()
    intersection = (pred_probs * target_one_hot).sum(dim=1)
    union = pred_probs.sum(dim=1) + target_one_hot.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


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


def multiclass_focal_loss(pred_probs: torch.Tensor,
                          target: torch.Tensor,
                          num_classes: int,
                          gamma: float = 2.0,
                          class_weights: torch.Tensor = None,
                          ignore_index: int = 255) -> torch.Tensor:
    """Multi-class focal loss on softmax probabilities.

    For each pixel, only the target class probability contributes:
        Loss = -(1 - pt)^gamma * log(pt)
    where pt is the predicted probability of the target class.
    Optional per-class weights (alpha_t) can be applied via class_weights.

    Args:
        pred_probs: [C, H, W] or [C, N] softmax probabilities
        target: [H, W] or [N] long class indices
        num_classes: number of classes
        gamma: focal gamma (higher = more down-weighting of easy examples)
        class_weights: [num_classes] per-class weight, used as alpha_t
        ignore_index: pixels with this label are ignored
    """
    pred_probs = pred_probs.reshape(num_classes, -1)
    target = target.reshape(-1)

    valid = target != ignore_index
    if not valid.any():
        return torch.tensor(0.0, device=pred_probs.device, dtype=pred_probs.dtype)

    pred_probs = pred_probs[:, valid]
    target = target[valid]

    # Gather pt = predicted probability of the target class
    pt = pred_probs[target, torch.arange(target.numel(), device=target.device)]
    pt = pt.clamp(min=1e-7, max=1.0)

    focal_weight = torch.pow(1.0 - pt, gamma)
    loss = -focal_weight * torch.log(pt)

    if class_weights is not None:
        alpha_t = class_weights[target]
        loss = alpha_t * loss

    return loss.mean()


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


def compute_anchor_fg_soft_curve_loss(
    xyz: torch.Tensor,
    seg_logits: torch.Tensor,
    opacity: torch.Tensor,
    anchor_idx: torch.Tensor,
    viewpoint_cam,
    gt_mask: torch.Tensor,
    tau: float = 0.35,
    temp: float = 0.10,
    min_ratio: float = 0.10,
    max_samples: int = 4096,
    detach_xyz: bool = True,
):
    """
    Anchor-level Soft Foreground Curve Loss (exp08).

    1. 保留原有 anchor_fg_ratio 统计逻辑（投影、采样、按 opacity*conf 加权聚合）。
    2. 对每个 anchor 计算 soft_weight = sigmoid((fg_ratio - tau) / temp)。
    3. 只保留 fg_ratio > min_ratio 的 anchor。
    4. target = 1 (foreground)。
    5. loss = BCEWithLogits(pred_logit, ones) * soft_weight。
    6. 返回未乘以 time_weight 的 raw loss，由调用方乘 time_weight。

    Returns:
        loss: scalar tensor (0.0 if no valid anchors)
        info: dict with selected_count, fg_ratio_mean, soft_weight_mean, pred_fg_mean, anchor_soft_loss
    """
    info = {
        "selected_count": 0,
        "fg_ratio_mean": 0.0,
        "soft_weight_mean": 0.0,
        "pred_fg_mean": 0.0,
        "anchor_soft_loss": 0.0,
    }

    if xyz.shape[0] == 0 or gt_mask is None:
        return torch.tensor(0.0, device=xyz.device), info

    if detach_xyz:
        xyz = xyz.detach()

    n_anchors = int(anchor_idx.max().item()) + 1

    # 1. Project child Gaussians to NDC
    p_ndc = geom_transform_points(xyz, viewpoint_cam.full_proj_transform)
    in_image = (p_ndc[:, 0] >= -1.0) & (p_ndc[:, 0] <= 1.0) & \
               (p_ndc[:, 1] >= -1.0) & (p_ndc[:, 1] <= 1.0) & \
               (p_ndc[:, 2] >= 0.0)  & (p_ndc[:, 2] <= 1.0)

    # 2. Sample GT mask at projected child positions
    gt_mask_norm = gt_mask.float()
    if gt_mask_norm.max() > 1.0:
        gt_mask_norm = gt_mask_norm / 255.0

    grid_2d = torch.stack([p_ndc[:, 0], -p_ndc[:, 1]], dim=-1)
    grid_2d = grid_2d.unsqueeze(0).unsqueeze(0)

    sampled_mask = torch.nn.functional.grid_sample(
        gt_mask_norm.unsqueeze(0),
        grid_2d,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False,
    ).squeeze()

    # 3. Per-child confidence
    conf = torch.abs(sampled_mask - 0.5) * 2.0

    # 4. Per-child weight = opacity * confidence
    opacity_squeezed = opacity.view(-1)
    weight = opacity_squeezed * conf

    valid_child = in_image & (weight > 1e-6)
    valid_count = int(valid_child.sum().item())

    if valid_count == 0:
        return torch.tensor(0.0, device=xyz.device), info

    # 5. Aggregate per-anchor foreground ratio (same as hard version)
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

    # 6. Soft selection: fg_ratio > min_ratio
    soft_mask = anchor_fg_ratio > min_ratio
    selected_count = int(soft_mask.sum().item())
    info["selected_count"] = selected_count

    if selected_count == 0:
        return torch.tensor(0.0, device=xyz.device), info

    # 7. Get per-anchor seg logits
    anchor_logits_sum = torch.zeros(n_anchors, device=xyz.device)
    anchor_logits_count = torch.zeros(n_anchors, device=xyz.device)
    anchor_logits_sum = anchor_logits_sum.index_add(0, anchor_idx, seg_logits.view(-1))
    anchor_logits_count = anchor_logits_count.index_add(0, anchor_idx, torch.ones_like(anchor_idx, dtype=torch.float))
    anchor_logits = anchor_logits_sum / anchor_logits_count.clamp(min=1)

    selected_logits = anchor_logits[soft_mask]
    selected_fg_ratio = anchor_fg_ratio[soft_mask]

    # 8. Soft weight = sigmoid((fg_ratio - tau) / temp)
    soft_weight = torch.sigmoid((selected_fg_ratio - tau) / temp)

    # 9. BCEWithLogitsLoss (target=1) weighted by soft_weight
    targets = torch.ones_like(selected_logits)
    loss_per_anchor = torch.nn.functional.binary_cross_entropy_with_logits(
        selected_logits, targets, reduction='none'
    )
    loss = (loss_per_anchor * soft_weight).sum() / soft_weight.sum().clamp(min=1e-6)

    # 10. Subsample if too many
    if selected_logits.numel() > max_samples:
        perm = torch.randperm(selected_logits.numel(), device=selected_logits.device)[:max_samples]
        selected_logits = selected_logits[perm]
        soft_weight = soft_weight[perm]
        targets = targets[perm]
        loss_per_anchor = torch.nn.functional.binary_cross_entropy_with_logits(
            selected_logits, targets, reduction='none'
        )
        loss = (loss_per_anchor * soft_weight).sum() / soft_weight.sum().clamp(min=1e-6)
        info["selected_count"] = max_samples

    # 11. Debug info
    with torch.no_grad():
        pred_fg_mean = torch.sigmoid(selected_logits).mean().item()
        fg_ratio_mean = selected_fg_ratio.mean().item()
        soft_weight_mean = soft_weight.mean().item()

    info["fg_ratio_mean"] = float(fg_ratio_mean)
    info["soft_weight_mean"] = float(soft_weight_mean)
    info["pred_fg_mean"] = float(pred_fg_mean)
    info["anchor_soft_loss"] = float(loss.item())

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
                              use_per_gaussian_seg=getattr(dataset, 'use_per_gaussian_seg', False),
                              num_classes=getattr(dataset, 'num_classes', 1),
                              no_opacity_detach=getattr(dataset, 'no_opacity_detach', False))
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
            if gaussians.use_per_gaussian_seg:
                seg_outputs = gaussians.num_classes * gaussians.n_offsets
            else:
                seg_outputs = gaussians.num_classes
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
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad,
                            iteration=iteration, opacity_grad_until=int(getattr(opt, 'opacity_grad_until', -1)))
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        # [超大场景适配] 某些 camera 在场景外，导致 scaling 为空 tensor
        if scaling.numel() > 0:
            scaling_reg = scaling.prod(dim=1).mean()
        else:
            scaling_reg = torch.tensor(0.0, device="cuda")
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
            num_classes = gaussians.num_classes
            dice_weight = float(getattr(opt, "dice_weight", 1.0))
            seg_w = float(getattr(opt, "mask_weight", 0.0)) * _ramp_factor(
                semantic_phase_iter,
                int(getattr(opt, "mask_warmup", 0)),
                int(getattr(opt, "mask_ramp", 0)),
            )

            # Mask weight decay schedule: linearly decay seg_w from mask_weight -> mask_weight_final
            # between mask_decay_start and mask_decay_end (absolute iterations).
            mask_weight_final = float(getattr(opt, "mask_weight_final", -1))
            mask_decay_start = int(getattr(opt, "mask_decay_start", -1))
            mask_decay_end = int(getattr(opt, "mask_decay_end", -1))
            if mask_weight_final >= 0 and mask_decay_start > 0 and mask_decay_end > mask_decay_start:
                if iteration >= mask_decay_end:
                    seg_w = mask_weight_final
                elif iteration >= mask_decay_start:
                    t_decay = (iteration - mask_decay_start) / max(1, mask_decay_end - mask_decay_start)
                    seg_w = seg_w + t_decay * (mask_weight_final - seg_w)

            if num_classes == 1:
                # ========== Binary Segmentation (sigmoid + focal + dice) ==========
                pred_mask_clamped = torch.clamp(pred_mask, min=1e-6, max=1.0-1e-6)
                gt_mask_float = gt_mask.float()
                if gt_mask_float.max() > 1.0:
                    gt_mask_float = gt_mask_float / 255.0

                if pred_mask_clamped.shape == gt_mask_float.shape:
                    # Uncertainty-aware pixel weighting (edge-robust)
                    conf = torch.clamp(torch.abs(gt_mask_float - 0.5) * 2.0, 0.0, 1.0)
                    uncertainty_min = float(getattr(opt, "uncertainty_min", 0.1))
                    uncertainty_power = float(getattr(opt, "uncertainty_power", 2.0))
                    uncertainty_weight = uncertainty_min + (1.0 - uncertainty_min) * torch.pow(conf, uncertainty_power)

                    # Semantic weight map
                    sem_weight_map = viewpoint_cam.semantic_weight if hasattr(viewpoint_cam, 'semantic_weight') else None
                    if getattr(opt, 'use_semantic_weight', False) and sem_weight_map is not None:
                        sem_w = sem_weight_map.cuda().float()
                        abs_conf = torch.abs(sem_w - 0.5) * 2.0
                        strategy = getattr(opt, 'sem_weight_strategy', 'hard')
                        if strategy == 'hard':
                            high_t = float(getattr(opt, 'sem_weight_high', 0.7))
                            low_t = float(getattr(opt, 'sem_weight_low', 0.1))
                            boost = float(getattr(opt, 'sem_weight_boost', 2.0))
                            sem_pixel_weight = torch.zeros_like(abs_conf)
                            sem_pixel_weight[abs_conf >= high_t] = 1.0
                            sem_pixel_weight[(abs_conf >= low_t) & (abs_conf < high_t)] = boost
                        elif strategy == 'conservative_hard':
                            high_t = float(getattr(opt, 'sem_weight_high', 0.7))
                            boost = float(getattr(opt, 'sem_weight_boost', 2.0))
                            sem_pixel_weight = torch.ones_like(abs_conf)
                            sem_pixel_weight[abs_conf >= high_t] = boost
                        else:  # smooth
                            peak = float(getattr(opt, 'sem_weight_smooth_peak', 0.5))
                            denom = peak if peak > 1e-6 else 1e-6
                            sem_pixel_weight = 1.0 - 4.0 * ((abs_conf - peak) / denom) ** 2
                            sem_pixel_weight = torch.clamp(sem_pixel_weight, min=0.0)
                            high_t = float(getattr(opt, 'sem_weight_high', 0.7))
                            sem_pixel_weight = torch.where(abs_conf >= high_t, torch.ones_like(sem_pixel_weight), sem_pixel_weight)
                        uncertainty_weight = uncertainty_weight * sem_pixel_weight

                    raw_loss = focal_loss(
                        pred_mask_clamped, gt_mask_float,
                        alpha=float(getattr(opt, "focal_alpha", 0.25)),
                        gamma=float(getattr(opt, "focal_gamma", 2.0)),
                        reduction='none',
                    )
                    loss_focal = (raw_loss * uncertainty_weight).mean()
                    loss_dice = dice_loss(pred_mask_clamped, gt_mask_float, smooth=1.0)
                    loss_seg = loss_focal + dice_weight * loss_dice
                    if seg_w > 0:
                        loss = loss + seg_w * loss_seg
                    if iteration % 100 == 0:
                        op_until = int(getattr(opt, 'opacity_grad_until', -1))
                        use_nd = getattr(gaussians, 'no_opacity_detach', False)
                        if use_nd and op_until > 0:
                            op_status = "grad" if iteration < op_until else "detach"
                        else:
                            op_status = "grad" if use_nd else "detach"
                        print(f"Iter {iteration}: Mask Loss = {loss_seg.item():.5f} (focal={loss_focal.item():.5f}, dice={loss_dice.item():.5f}, w={seg_w:.4f}) [opacity={op_status}]")
            else:
                # ========== Multi-class Segmentation (prob + NLL + dice) ==========
                # pred_mask: [C, H, W] PROBABILITIES from rasterizer (alpha-blended softmax)
                # gt_mask:  [1, H, W] or [H, W] long class indices
                gt_labels = gt_mask.squeeze(0).long()  # [H, W]
                ignore_index = 255
                max_valid = gt_labels[gt_labels != ignore_index].max() if (gt_labels != ignore_index).any() else torch.tensor(0, device=gt_labels.device)
                if max_valid > num_classes - 1:
                    raise ValueError(
                        f"Mask contains label {max_valid.item()} which exceeds num_classes-1={num_classes-1}. "
                        f"Please ensure mask labels are in [0, {num_classes-1}] or use {ignore_index} as ignore_index."
                    )

                pred_probs = pred_mask  # [C, H, W] probabilities (already softmaxed by rasterizer)
                
                # Normalize probabilities (rasterizer output may not sum to 1 due to alpha blending)
                pred_probs = pred_probs / pred_probs.sum(dim=0, keepdim=True).clamp_min(1e-6)

                # Compute class weights: inverse sqrt frequency, normalized to mean=1
                valid_labels = gt_labels[gt_labels != ignore_index]
                if valid_labels.numel() > 0:
                    class_counts = torch.bincount(valid_labels, minlength=num_classes).float()
                    # Avoid division by zero for missing classes - use median count instead of 1.0
                    # to prevent extreme weights for absent classes
                    median_count = class_counts[class_counts > 0].median() if (class_counts > 0).any() else torch.tensor(1.0)
                    class_counts = torch.where(class_counts > 0, class_counts, median_count)
                    class_weights = 1.0 / torch.sqrt(class_counts)
                    class_weights = class_weights / class_weights.mean()  # normalize to mean=1
                    # Clamp weights to prevent extreme values
                    class_weights = torch.clamp(class_weights, min=0.1, max=3.0)
                    class_weights = class_weights.to(pred_probs.device)
                else:
                    class_weights = torch.ones(num_classes, device=pred_probs.device)

                # NLL loss (negative log-likelihood) with class weights
                pred_probs_clamped = torch.clamp(pred_probs, min=1e-7, max=1.0)  # avoid log(0)
                log_probs = torch.log(pred_probs_clamped)  # [C, H, W]
                
                if (gt_labels == ignore_index).any():
                    valid_mask = gt_labels != ignore_index
                    loss_nll = F.nll_loss(
                        log_probs[:, valid_mask].unsqueeze(0),  # [1, C, N]
                        gt_labels[valid_mask].unsqueeze(0),      # [1, N]
                        weight=class_weights,
                        ignore_index=ignore_index,
                        reduction='mean'
                    )
                else:
                    loss_nll = F.nll_loss(
                        log_probs.unsqueeze(0),   # [1, C, H*W]
                        gt_labels.unsqueeze(0),   # [1, H*W]
                        weight=class_weights,
                        ignore_index=ignore_index,
                        reduction='mean'
                    )

                # Multi-class dice (ignore invalid pixels)
                if (gt_labels == ignore_index).any():
                    valid_mask = gt_labels != ignore_index
                    loss_dice = multiclass_dice_loss(pred_probs[:, valid_mask], gt_labels[valid_mask], num_classes, smooth=1.0)
                else:
                    loss_dice = multiclass_dice_loss(pred_probs, gt_labels, num_classes, smooth=1.0)

                loss_focal = multiclass_focal_loss(
                    pred_probs, gt_labels, num_classes,
                    gamma=float(getattr(opt, 'focal_gamma', 2.0)),
                    class_weights=class_weights,
                    ignore_index=ignore_index
                )

                loss_seg = loss_nll + dice_weight * loss_dice + loss_focal
                if seg_w > 0:
                    loss = loss + seg_w * loss_seg
                if iteration % 100 == 0:
                    op_until = int(getattr(opt, 'opacity_grad_until', -1))
                    op_status = "grad" if (op_until > 0 and iteration < op_until) or (getattr(gaussians, 'no_opacity_detach', False) and op_until <= 0) else "detach"
                    print(f"Iter {iteration}: Mask Loss = {loss_seg.item():.5f} (nll={loss_nll.item():.5f}, dice={loss_dice.item():.5f}, focal={loss_focal.item():.5f}, w={seg_w:.4f}) [opacity={op_status}]")

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

                                num_classes_knn = gaussians.num_classes
                                if num_classes_knn == 1:
                                    # [改进 v2] 概率空间对称 BCE KNN（二分类）
                                    sample_prob = torch.sigmoid(sample_logit).unsqueeze(1)      # [n_samples, 1]
                                    neighbor_prob = torch.sigmoid(neighbor_logit)               # [n_samples, k-1]
                                    eps = 1e-6
                                    loss_knn = -0.5 * (
                                        sample_prob * torch.log(neighbor_prob + eps)
                                        + (1.0 - sample_prob) * torch.log(1.0 - neighbor_prob + eps)
                                        + neighbor_prob * torch.log(sample_prob + eps)
                                        + (1.0 - neighbor_prob) * torch.log(1.0 - sample_prob + eps)
                                    )
                                    loss_knn = loss_knn.mean()
                                    knn_loss_name = "KNN(BCE)"
                                else:
                                    # 多分类 KNN：symmetric KL divergence (stronger than MSE)
                                    sample_prob = torch.nn.functional.softmax(sample_logit, dim=-1).unsqueeze(1)   # [n_samples, 1, C]
                                    neighbor_prob = torch.nn.functional.softmax(neighbor_logit, dim=-1)              # [n_samples, k-1, C]
                                    eps = 1e-7
                                    # KL(P||Q) + KL(Q||P)
                                    kl_sn = (sample_prob * (torch.log(sample_prob + eps) - torch.log(neighbor_prob + eps))).sum(dim=-1)
                                    kl_ns = (neighbor_prob * (torch.log(neighbor_prob + eps) - torch.log(sample_prob + eps))).sum(dim=-1)
                                    loss_knn = 0.5 * (kl_sn + kl_ns).mean()
                                    knn_loss_name = "KNN(KL)"

                                knn_w = float(getattr(opt, "knn_weight", 0.0)) * _ramp_factor(
                                    semantic_phase_iter,
                                    int(getattr(opt, "knn_warmup", 0)),
                                    int(getattr(opt, "knn_ramp", 0)),
                                )
                                if knn_w > 0:
                                    loss = loss + knn_w * loss_knn

                                print(f"Iter {iteration}: {knn_loss_name} Loss = {loss_knn.item():.6f} (w={knn_w:.4f}, every={knn_every}, offset={knn_offset})")

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

        # Anchor voting losses are currently designed for binary segmentation only
        if do_ar and gt_mask is not None and gaussians.num_classes == 1:
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

        # 4. Anchor-level Soft Foreground Curve Loss (exp08)
        #    与 hard FG voting 互斥；若 anchor_soft_enable=True 则启用 soft，忽略 hard。
        #    当前仅支持二分类。
        if getattr(opt, "anchor_soft_enable", False) and gt_mask is not None and gaussians.num_classes == 1:
            try:
                as_weight = float(getattr(opt, "anchor_soft_weight", 0.002))
                as_start = int(getattr(opt, "anchor_soft_start_iter", 5000))
                as_ramp = int(getattr(opt, "anchor_soft_ramp", 1000))
                as_decay_start = int(getattr(opt, "anchor_soft_decay_start", 10000))
                as_decay_end = int(getattr(opt, "anchor_soft_decay_end", 15000))
                as_tau = float(getattr(opt, "anchor_soft_tau", 0.35))
                as_temp = float(getattr(opt, "anchor_soft_temp", 0.10))
                as_min_ratio = float(getattr(opt, "anchor_soft_min_ratio", 0.10))
                as_max_samples = int(getattr(opt, "anchor_soft_max_samples", 4096))
                as_detach_xyz = bool(getattr(opt, "anchor_soft_detach_xyz", True))

                # Time weight: 0 -> ramp up -> 1 -> decay -> 0
                if iteration < as_start:
                    time_weight = 0.0
                elif iteration < as_start + as_ramp:
                    time_weight = (iteration - as_start) / max(as_ramp, 1)
                elif iteration < as_decay_start:
                    time_weight = 1.0
                elif iteration < as_decay_end:
                    time_weight = 1.0 - (iteration - as_decay_start) / max(as_decay_end - as_decay_start, 1)
                else:
                    time_weight = 0.0

                if time_weight > 0.0:
                    neural_xyz = render_pkg.get("neural_xyz")
                    neural_seg_logits = render_pkg.get("neural_seg_logits")
                    neural_opacity_filtered = render_pkg.get("neural_opacity_filtered")
                    neural_anchor_idx = render_pkg.get("neural_anchor_idx")

                    if (neural_xyz is not None and neural_seg_logits is not None and
                        neural_opacity_filtered is not None and neural_anchor_idx is not None):
                        loss_as, as_info = compute_anchor_fg_soft_curve_loss(
                            xyz=neural_xyz,
                            seg_logits=neural_seg_logits,
                            opacity=neural_opacity_filtered,
                            anchor_idx=neural_anchor_idx,
                            viewpoint_cam=viewpoint_cam,
                            gt_mask=gt_mask,
                            tau=as_tau,
                            temp=as_temp,
                            min_ratio=as_min_ratio,
                            max_samples=as_max_samples,
                            detach_xyz=as_detach_xyz,
                        )

                        total_w = as_weight * time_weight
                        if total_w > 0.0 and loss_as.item() > 0.0:
                            loss = loss + total_w * loss_as

                        if iteration % 100 == 0:
                            sel_count = as_info.get("selected_count", 0)
                            fg_ratio_mean = as_info.get("fg_ratio_mean", 0.0)
                            soft_w_mean = as_info.get("soft_weight_mean", 0.0)
                            pred_fg_mean = as_info.get("pred_fg_mean", 0.0)
                            as_loss_val = as_info.get("anchor_soft_loss", 0.0)
                            print(
                                f"Iter {iteration}: Anchor-Soft Loss={as_loss_val:.5f} "
                                f"time_w={time_weight:.4f} total_w={total_w:.6f} "
                                f"selected_count={sel_count} fg_ratio_mean={fg_ratio_mean:.3f} "
                                f"soft_weight_mean={soft_w_mean:.3f} pred_fg_mean={pred_fg_mean:.3f}"
                            )
            except Exception as e:
                print(f"[Warning] Anchor-Soft-Curve Loss skipped at iter {iteration}: {e}")
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
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger, auto_twostage=getattr(opt, 'auto_twostage', False), model_path=dataset.model_path)

            # TensorBoard: visualize current view's GT / predicted masks during training
            # Log masks only every 1000 iterations to avoid excessive I/O
            if tb_writer is not None and pred_mask is not None and gt_mask is not None and iteration % 1000 == 0:
                try:
                    if num_classes == 1:
                        # add_images expects NCHW; both gt_mask and pred_mask are [1, H, W]
                        tb_writer.add_images(f"{dataset_name}/train_view/mask_gt", gt_mask.unsqueeze(0), iteration)
                        tb_writer.add_images(f"{dataset_name}/train_view/mask_pred", pred_mask.clamp(0.0, 1.0).unsqueeze(0), iteration)
                        # Optionally visualize masked RGB (prediction overlay); image is [3, H, W]
                        masked_image = image * pred_mask.clamp(0.0, 1.0)
                        tb_writer.add_images(f"{dataset_name}/train_view/masked_rgb", masked_image.unsqueeze(0), iteration)
                    else:
                        # Multi-class: visualize argmax label map normalized to [0,1]
                        pred_label_vis = pred_mask.argmax(dim=0).float() / max(num_classes - 1, 1)
                        gt_label_raw = gt_mask.squeeze(0).float()
                        # Map ignore_index=255 to background (0) for visualization
                        gt_label_raw[gt_label_raw == 255.0] = 0.0
                        gt_label_vis = gt_label_raw / max(num_classes - 1, 1)
                        tb_writer.add_images(f"{dataset_name}/train_view/mask_gt", gt_label_vis.unsqueeze(0).unsqueeze(0), iteration)
                        tb_writer.add_images(f"{dataset_name}/train_view/mask_pred", pred_label_vis.unsqueeze(0).unsqueeze(0), iteration)
                        # Skip masked_rgb overlay for multi-class (ambiguous which channel to use)
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

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None, auto_twostage=False, model_path=None):
    """
    王子殿下专属增强版报告系统：
    - 实时监控 PSNR (峰值信噪比) 变化曲线
    - 实时监控 mIoU (语义分割精度) 变化曲线
    - 自动同步高清渲染图与误差图
    - 新增 SSIM 监控（用于 auto-twostage geometry selection）
    """
    metrics = {}
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
                ssim_test = 0.0
                miou_test = 0.0  # semantic metric: fg_iou for binary, mIoU for multi-class
                miou_accum = None  # [ [inter, union], ... ] per class, only for multi-class
                miou_views = 0     # count of views with valid masks (multi-class)
                # For binary: accumulate bg and fg inter/union across all views
                binary_inter_union = None  # {'bg': [inter, union], 'fg': [inter, union]}

                for idx, viewpoint in enumerate(config['cameras']):
                    # 执行渲染
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    # 1. 计算重建精度 (PSNR + SSIM)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                    # 2. 计算语义精度 (mIoU)
                    pred_mask = render_pkg.get("mask", None)
                    gt_mask = getattr(viewpoint, "semantic_mask", None)
                    if pred_mask is not None and gt_mask is not None:
                        gt_mask = gt_mask.cuda()
                        if scene.gaussians.num_classes == 1:
                            gt_mask_cuda = gt_mask.cuda().float()
                            p_mask = (pred_mask > 0.5).float()
                            g_mask = (gt_mask_cuda > 0.5).float()
                            # Binary: compute both bg and fg IoU, accumulate globally
                            if binary_inter_union is None:
                                binary_inter_union = {'bg': [0.0, 0.0], 'fg': [0.0, 0.0]}
                            # FG
                            inter_fg = (p_mask * g_mask).sum().item()
                            union_fg = p_mask.sum().item() + g_mask.sum().item() - inter_fg
                            binary_inter_union['fg'][0] += inter_fg
                            binary_inter_union['fg'][1] += union_fg
                            # BG
                            p_bg = (pred_mask <= 0.5).float()
                            g_bg = (gt_mask_cuda <= 0.5).float()
                            inter_bg = (p_bg * g_bg).sum().item()
                            union_bg = p_bg.sum().item() + g_bg.sum().item() - inter_bg
                            binary_inter_union['bg'][0] += inter_bg
                            binary_inter_union['bg'][1] += union_bg
                            # Legacy: per-view fg_iou average for miou_test (kept for backward compat)
                            miou_test += (inter_fg + 1e-6) / (union_fg + 1e-6)
                        else:
                            # Multi-class mIoU: accumulate per-class intersection/union across all test views
                            num_classes_mc = scene.gaussians.num_classes
                            if miou_accum is None:
                                miou_accum = [[0.0, 0.0] for _ in range(num_classes_mc)]
                            pred_labels = pred_mask.argmax(dim=0)  # [H, W]
                            gt_labels_mc = gt_mask.squeeze(0).long()  # [H, W]
                            ignore_index = 255
                            valid_mask = gt_labels_mc != ignore_index
                            pred_valid = pred_labels[valid_mask]
                            gt_valid = gt_labels_mc[valid_mask]
                            if pred_valid.device != gt_valid.device:
                                gt_valid = gt_valid.to(pred_valid.device)
                            for c in range(num_classes_mc):
                                p_c = (pred_valid == c).float()
                                g_c = (gt_valid == c).float()
                                inter = (p_c * g_c).sum().item()
                                union = p_c.sum().item() + g_c.sum().item() - inter
                                # accumulate globally
                                miou_accum[c][0] += inter
                                miou_accum[c][1] += union
                            miou_views += 1

                    # 向 TensorBoard 实时推送前 30 张渲染图
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/' + config['name'] + f"_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/' + config['name'] + f"_view_{viewpoint.image_name}/errormap", (gt_image[None] - image[None]).abs(), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/' + config['name'] + f"_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)

                # 计算所有评估相机均值
                cam_count = len(config['cameras'])
                psnr_test /= cam_count
                ssim_test /= cam_count
                l1_test /= cam_count
                # mIoU: binary per-view average; multi-class global per-class average
                fg_iou = 0.0
                bg_iou = 0.0
                binary_miou = 0.0
                if miou_accum is not None and miou_views > 0:
                    # Multi-class
                    class_ious = []
                    for inter, union in miou_accum:
                        if union > 0:
                            class_ious.append(inter / (union + 1e-6))
                    miou_test = float(np.mean(class_ious)) if class_ious else 0.0
                elif binary_inter_union is not None:
                    # Binary: compute true binary mIoU = (bg_iou + fg_iou) / 2
                    inter_bg, union_bg = binary_inter_union['bg']
                    inter_fg, union_fg = binary_inter_union['fg']
                    bg_iou = inter_bg / (union_bg + 1e-6)
                    fg_iou = inter_fg / (union_fg + 1e-6)
                    binary_miou = (bg_iou + fg_iou) / 2.0
                    # miou_test historically tracked per-view fg_iou average; keep for compat
                    miou_test = miou_test / cam_count if cam_count > 0 else 0.0
                else:
                    miou_test /= cam_count

                # TensorBoard 曲线绘制
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/{config["name"]}/Accuracy_PSNR', psnr_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/{config["name"]}/Accuracy_SSIM', ssim_test, iteration)
                    if miou_test > 0:
                        if binary_inter_union is not None:
                            tb_writer.add_scalar(f'{dataset_name}/{config["name"]}/Semantic_FG_IoU', fg_iou, iteration)
                            tb_writer.add_scalar(f'{dataset_name}/{config["name"]}/Semantic_Binary_mIoU', binary_miou, iteration)
                        else:
                            tb_writer.add_scalar(f'{dataset_name}/{config["name"]}/Semantic_mIoU', miou_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/' + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)

                if binary_inter_union is not None:
                    logger.info(f"\n[ITER {iteration}] 检阅 {config['name']}: PSNR {psnr_test:.2f} dB, SSIM {ssim_test:.4f}, FG_IoU {fg_iou:.4f}, Binary_mIoU {binary_miou:.4f} (BG={bg_iou:.4f}, FG={fg_iou:.4f})")
                else:
                    logger.info(f"\n[ITER {iteration}] 检阅 {config['name']}: PSNR {psnr_test:.2f} dB, SSIM {ssim_test:.4f}, mIoU {miou_test:.4f}")

                if wandb is not None:
                    log_dict = {f"{config['name']}_PSNR": psnr_test, f"{config['name']}_SSIM": ssim_test}
                    if binary_inter_union is not None:
                        log_dict[f"{config['name']}_FG_IoU"] = fg_iou
                        log_dict[f"{config['name']}_Binary_mIoU"] = binary_miou
                    else:
                        log_dict[f"{config['name']}_mIoU"] = miou_test
                    wandb.log(log_dict)

                metrics[config['name']] = {
                    'psnr': psnr_test.item() if isinstance(psnr_test, torch.Tensor) else float(psnr_test),
                    'ssim': ssim_test.item() if isinstance(ssim_test, torch.Tensor) else float(ssim_test),
                }
                if binary_inter_union is not None:
                    metrics[config['name']]['fg_iou'] = float(fg_iou)
                    metrics[config['name']]['bg_iou'] = float(bg_iou)
                    metrics[config['name']]['binary_miou'] = float(binary_miou)
                else:
                    metrics[config['name']]['miou'] = miou_test.item() if isinstance(miou_test, torch.Tensor) else float(miou_test)

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/total_points', scene.gaussians.get_anchor.shape[0], iteration)

        # Auto-twostage: persist geometry-stage metrics for best-checkpoint selection
        if auto_twostage and model_path is not None:
            log_path = os.path.join(model_path, "geometry_stage_metrics.jsonl")
            try:
                with open(log_path, "a") as f:
                    record = {"iteration": iteration, "metrics": metrics}
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                if logger is not None:
                    logger.warning(f"Failed to write geometry metrics log: {e}")

        torch.cuda.empty_cache()
        scene.gaussians.train()

    return metrics


def select_best_geometry_from_logs(model_path, tie_delta=0.1, logger=None):
    """
    读取 geometry_stage_metrics.jsonl，按 test PSNR 优先 + SSIM 平局的规则选出最佳 geometry iteration。
    返回 best_iter (int)。
    """
    log_path = os.path.join(model_path, "geometry_stage_metrics.jsonl")
    if not os.path.exists(log_path):
        if logger is not None:
            logger.warning(f"Geometry metrics log not found at {log_path}; falling back to final iteration.")
        return None

    records = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                iteration = rec.get("iteration")
                metrics = rec.get("metrics", {})
                test_metrics = metrics.get("test", {})
                psnr = test_metrics.get("psnr", 0.0)
                ssim_val = test_metrics.get("ssim", 0.0)
                records.append({"iteration": iteration, "psnr": psnr, "ssim": ssim_val})
    except Exception as e:
        if logger is not None:
            logger.warning(f"Failed to parse geometry metrics log: {e}; falling back to final iteration.")
        return None

    if not records:
        return None

    # Sort by PSNR desc, then SSIM desc, then iteration asc (earlier preferred)
    records.sort(key=lambda r: (-r["psnr"], -r["ssim"], r["iteration"]))

    best = records[0]
    # Tie-breaker: if top records have very close PSNR, prefer higher SSIM
    for rec in records[1:]:
        if abs(rec["psnr"] - best["psnr"]) < tie_delta:
            if rec["ssim"] > best["ssim"]:
                best = rec
        else:
            break

    if logger is not None:
        logger.info(f"[Auto Two-Stage] Best geometry selected: iter={best['iteration']}, PSNR={best['psnr']:.4f}, SSIM={best['ssim']:.4f}")
    return best["iteration"]


def copy_best_geometry_checkpoint(model_path, best_iter, best_geometry_dir, logger=None):
    """
    将最佳 geometry PLY 复制到 best_geometry_dir，并附带 metadata JSON。
    """
    src_ply = os.path.join(model_path, "point_cloud", f"iteration_{best_iter}", "point_cloud.ply")
    dst_dir = os.path.join(model_path, best_geometry_dir)
    os.makedirs(dst_dir, exist_ok=True)
    dst_ply = os.path.join(dst_dir, "point_cloud.ply")

    if not os.path.exists(src_ply):
        if logger is not None:
            logger.error(f"Best geometry PLY not found: {src_ply}")
        return False

    shutil.copy2(src_ply, dst_ply)

    # Read metrics for metadata
    log_path = os.path.join(model_path, "geometry_stage_metrics.jsonl")
    meta = {"best_iter": best_iter, "best_psnr": None, "best_ssim": None, "selection_rule": "psnr_first_ssim_tiebreaker"}
    try:
        with open(log_path, "r") as f:
            for line in f:
                rec = json.loads(line.strip())
                if rec.get("iteration") == best_iter:
                    test_m = rec.get("metrics", {}).get("test", {})
                    meta["best_psnr"] = test_m.get("psnr")
                    meta["best_ssim"] = test_m.get("ssim")
                    break
    except Exception:
        pass

    meta_path = os.path.join(dst_dir, "best_geometry_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    if logger is not None:
        logger.info(f"[Auto Two-Stage] Copied best geometry to {dst_ply}")
    return True


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
        gt = view.original_image[0:3, :, :].cuda()

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
                              use_per_gaussian_seg=getattr(dataset, 'use_per_gaussian_seg', False),
                              num_classes=getattr(dataset, 'num_classes', 1),
                              no_opacity_detach=getattr(dataset, 'no_opacity_detach', False))
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
    if args.auto_twostage:
        logger.info("\n[Auto Two-Stage] Stage 1: RGB Geometry Pretraining")
        # Stage 1: disable semantic losses, keep original densification schedule
        opt_stage1 = op.extract(args)
        opt_stage1.auto_twostage = True
        opt_stage1.start_semantic_iter = 999_999_999  # force semantic off
        opt_stage1.mask_weight = 0.0
        opt_stage1.knn_weight = 0.0
        opt_stage1.iterations = args.geometry_stage_iters
        # update_until keeps original value for densification during geometry stage
        stage1_testing = list(range(args.geometry_eval_interval, args.geometry_stage_iters + 1, args.geometry_eval_interval))
        stage1_saving = sorted(set(stage1_testing + [args.geometry_stage_iters]))
        # Clean previous geometry metrics log if exists
        geo_log_path = os.path.join(args.model_path, "geometry_stage_metrics.jsonl")
        if os.path.exists(geo_log_path):
            os.remove(geo_log_path)

        training(lp.extract(args), opt_stage1, pp.extract(args), dataset, stage1_testing, stage1_saving, [], args.start_checkpoint, args.debug_from, wandb, logger, load_iteration=args.load_iteration)

        # Select best geometry checkpoint
        best_iter = select_best_geometry_from_logs(args.model_path, args.geometry_tie_psnr_delta, logger=logger)
        if best_iter is None:
            best_iter = args.geometry_stage_iters
            logger.info(f"[Auto Two-Stage] No metrics log found; falling back to final iteration {best_iter}")
        if args.save_best_geometry:
            copy_best_geometry_checkpoint(args.model_path, best_iter, args.best_geometry_dir, logger=logger)

        # Stage 2: Semantic Fine-tuning from best geometry
        logger.info("\n[Auto Two-Stage] Stage 2: Semantic Fine-tuning")
        opt_stage2 = op.extract(args)
        opt_stage2.iterations = args.semantic_stage_iters
        opt_stage2.update_until = args.semantic_update_until  # default 0 (no densification)
        stage2_ply = os.path.join(args.model_path, args.best_geometry_dir, "point_cloud.ply")
        # Ensure final semantic iteration is saved
        stage2_saving = sorted(set(args.save_iterations + [args.semantic_stage_iters]))

        training(lp.extract(args), opt_stage2, pp.extract(args), dataset, args.test_iterations, stage2_saving, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=stage2_ply)
    else:
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