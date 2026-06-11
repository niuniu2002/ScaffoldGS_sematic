#
# render1.py — Semantic-aware rendering script (modified from render.py)
#
# Usage:  python render1.py -m <path_to_model> [options]
#
# Differences from render.py:
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#   - Adds semantic overlay visualization (create_semantic_overlay)
#   - Replaces matplotlib heatmap with a semi-transparent orange mask
#   - Outputs debug images with semantic probabilities overlaid on RGB
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import os
import torch
import numpy as np
import subprocess
import json
import time
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render, prefilter_voxel
from scene import Scene
# [移除] 不再需要 matplotlib
# import matplotlib.pyplot as plt

# --- 自动显存分配 ---
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
try:
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
except Exception as e:
    pass

# --- [新增] 创建半透明叠加图的核心函数 ---
def create_semantic_overlay(rgb_image, semantic_map, alpha=0.6, threshold=0.5, num_classes=1):
    """
    创建一个半透明的语义掩码覆盖在 RGB 图像上。
    rgb_image: [3, H, W], 原始 RGB 图像
    semantic_map: [1, H, W] logits (binary) or [C, H, W] logits (multi-class)
    alpha: 掩码透明度 (0.0 - 1.0), 越大越不透明
    threshold: 判定为目标的阈值 (0.0 - 1.0)，仅用于二分类
    num_classes: 1=binary (sigmoid+threshold), >1=multi-class (softmax+argmax)
    """
    if semantic_map is None:
        return rgb_image

    sem = semantic_map.detach()

    # 增加调试输出，方便定位语义图是否恒定
    try:
        smin = float(torch.min(sem))
        smax = float(torch.max(sem))
        smean = float(torch.mean(sem))
        print(f"[render] semantic_map stats: min={smin:.6f}, max={smax:.6f}, mean={smean:.6f}, num_classes={num_classes}")
    except Exception:
        pass

    if num_classes == 1:
        # 二分类：sigmoid + threshold
        if sem.max() <= 1.0 and sem.min() >= 0.0:
            prob = sem
        else:
            prob = torch.sigmoid(sem)
        mask = (prob > threshold).float()
        if mask.dim() == 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)
        color = torch.tensor([1.0, 0.3, 0.0], device=rgb_image.device, dtype=rgb_image.dtype).view(3, 1, 1)
        mask3 = mask.unsqueeze(0).repeat(3, 1, 1)
        overlay = color * mask3
        out = rgb_image * (1.0 - alpha * mask3) + overlay * (alpha * mask3)
        return torch.clamp(out, 0.0, 1.0)
    else:
        # 多分类：直接对 logits 做 argmax（argmax(logits) == argmax(softmax(logits))）
        # 每个非背景类用不同颜色叠加
        pred_labels = sem.argmax(dim=0)  # [H, W]
        palette = torch.tensor([
            [0.0, 0.0, 0.0],      # class 0: no overlay
            [1.0, 0.3, 0.0],      # class 1: orange-red
            [0.0, 0.6, 1.0],      # class 2: sky blue
            [0.2, 0.8, 0.2],      # class 3: green
            [1.0, 0.0, 1.0],      # class 4: magenta
            [1.0, 0.8, 0.0],      # class 5: yellow
        ], device=rgb_image.device, dtype=rgb_image.dtype)
        if num_classes > palette.shape[0]:
            extra = num_classes - palette.shape[0]
            rand_colors = torch.rand(extra, 3, device=rgb_image.device, dtype=rgb_image.dtype)
            palette = torch.cat([palette, rand_colors], dim=0)
        overlay_rgb = palette[pred_labels].permute(2, 0, 1)  # [3, H, W]
        fg_mask = (pred_labels > 0).float()  # [H, W]
        mask3 = fg_mask.unsqueeze(0).repeat(3, 1, 1)
        out = rgb_image * (1.0 - alpha * mask3) + overlay_rgb * (alpha * mask3)
        return torch.clamp(out, 0.0, 1.0)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_semantic=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # [修改] 将输出文件夹命名为 overlay，更准确
    overlay_path = os.path.join(model_path, name, "ours_{}".format(iteration), "overlay")

    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)
    if render_semantic and not os.path.exists(overlay_path):
        os.makedirs(overlay_path)

    t_list = []
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        t_list.append(t1-t0)

        # 保存 RGB
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        # 保存语义叠加图
        if render_semantic:
            # 从 render_pkg 中安全地取出语义结果
            sem_out = None
            possible_keys = ["semantic", "sem", "mask", "segmentation", "render_semantic"]
            for key in possible_keys:
                if key in render_pkg and render_pkg[key] is not None:
                    sem_out = render_pkg[key]
                    break

            if sem_out is not None:
                overlay_vis = create_semantic_overlay(rendering, sem_out, alpha=0.6, threshold=0.5, num_classes=gaussians.num_classes)
                # 保存到 overlay 文件夹
                torchvision.utils.save_image(overlay_vis, os.path.join(overlay_path, '{0:05d}'.format(idx) + ".png"))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_semantic : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, num_classes=getattr(dataset, 'num_classes', 1))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_semantic)
        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_semantic)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--semantic", action="store_true", help="Render semantic segmentation masks")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.semantic)