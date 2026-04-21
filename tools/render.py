#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
import matplotlib.pyplot as plt # 用于生成热力图颜色

# --- 自动显存分配 ---
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
try:
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
except Exception as e:
    pass

# --- 核心修改：更加强大的可视化函数 ---
def visualize_semantic(semantic_map):
    if semantic_map is None:
        return None
    
    # 1. 数据标准化：将任意范围的数据映射到 [0, 1]
    # 这样无论是 logits 还是概率，都能显示颜色
    sem_min = semantic_map.min()
    sem_max = semantic_map.max()
    
    # 防止分母为0
    if sem_max - sem_min < 1e-6:
        norm_map = torch.zeros_like(semantic_map)
    else:
        norm_map = (semantic_map - sem_min) / (sem_max - sem_min)
    
    # 2. 应用热力图颜色映射 (Turbo/Jet 风格: 蓝->绿->红)
    # 形状变换: [1, H, W] -> [H, W]
    heatmap = norm_map.squeeze().cpu().numpy()
    
    # 使用 matplotlib 的 colormap
    colormap = plt.get_cmap('turbo') # 'turbo' 对视觉更友好，也可以换 'jet'
    colored_image = colormap(heatmap) # 输出形状 [H, W, 4] (RGBA)
    
    # 转换为 Tensor [3, H, W] 并去掉 Alpha 通道
    colored_tensor = torch.from_numpy(colored_image[:, :, :3]).permute(2, 0, 1).float().to(semantic_map.device)
    
    return colored_tensor

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_semantic=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    sem_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantic")

    if not os.path.exists(render_path):
        os.makedirs(render_path)
    if not os.path.exists(gts_path):
        os.makedirs(gts_path)
    if render_semantic and not os.path.exists(sem_path):
        os.makedirs(sem_path)

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
        
        # 保存语义图
        if render_semantic:
            # 自动寻找语义 Key
            sem_out = None
            possible_keys = ['semantic', 'sem', 'mask', 'segmentation', 'render_semantic']
            for key in possible_keys:
                if key in render_pkg and render_pkg[key] is not None:
                    sem_out = render_pkg[key]
                    break
            
            if sem_out is not None:
                # [数值探针] 打印第一帧的数值范围，极为关键！
                if idx == 0:
                    print(f"\n[Semantic Probe] Frame 0 stats: Min={sem_out.min().item():.4f}, Max={sem_out.max().item():.4f}, Mean={sem_out.mean().item():.4f}")
                    if sem_out.max() == 0 and sem_out.min() == 0:
                        print("⚠️ 警告：语义输出全为0！这可能是全黑的原因。")

                sem_vis = visualize_semantic(sem_out)
                torchvision.utils.save_image(sem_vis, os.path.join(sem_path, '{0:05d}'.format(idx) + ".png"))

    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_semantic : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
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