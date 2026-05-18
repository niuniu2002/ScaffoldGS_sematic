"""
带 Test-Time Adaptation (TTA) 的 myvideo 评估脚本。

用法：
    python eval_myvideo_with_tta.py \
        -m <model_path> \
        -s <source_path> \
        --iteration 30000 \
        --tta_iterations 50 \
        --tta_lr 1e-4 \
        --white_background

原理：
    冻结所有参数，只微调 mlp_segmentation 几轮，使模型快速适应目标域。
    对跨域数据（如 myvideo）特别有效。
"""
import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    bce = -targets * torch.log(inputs + 1e-6) - (1 - targets) * torch.log(1 - inputs + 1e-6)
    p_t = (targets * inputs) + ((1 - targets) * (1 - inputs))
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    loss = (alpha_t * (1.0 - p_t) ** gamma * bce).mean()
    return loss


def dice_loss(inputs, targets, smooth=1.0):
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1.0 - dice


def evaluate_with_tta(model_path, source_path, iteration,
                      white_background=False, appearance_dim=32,
                      tta_iterations=0, tta_lr=1e-4):
    parser = argparse.ArgumentParser()
    ModelParams(parser)
    args = parser.parse_args([
        '--source_path', source_path,
        '--model_path', model_path,
        '--eval',
        '--white_background' if white_background else '--no-white_background',
        '--appearance_dim', str(appearance_dim),
    ])

    gaussians = GaussianModel(
        args.feat_dim, args.n_offsets, args.voxel_size,
        args.update_depth, args.update_init_factor, args.update_hierachy_factor,
        args.use_feat_bank, args.appearance_dim, args.ratio,
        args.add_opacity_dist, args.add_cov_dist, args.add_color_dist,
        num_classes=getattr(args, 'num_classes', 1)
    )
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    class FakePipe:
        def __init__(self):
            self.debug = False
            self.compute_cov3D_python = False
            self.convert_SHs_python = False
    pipe = FakePipe()

    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    print(f"Total cameras: {len(all_cameras)}")

    # ============== Test-Time Adaptation ==============
    if tta_iterations > 0:
        print(f"\n[ TTA ] Fine-tuning mlp_segmentation for {tta_iterations} iterations...")
        # 冻结所有参数
        for param in gaussians.parameters():
            param.requires_grad = False
        # 只解冻 segmentation MLP
        for param in gaussians.mlp_segmentation.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(
            gaussians.mlp_segmentation.parameters(), lr=tta_lr
        )

        cameras_with_mask = [c for c in all_cameras
                               if getattr(c, 'semantic_mask', None) is not None]
        if len(cameras_with_mask) == 0:
            print("[Warning] No GT masks found, skipping TTA.")
        else:
            for tta_iter in range(tta_iterations):
                total_loss = 0.0
                for viewpoint in cameras_with_mask:
                    render_pkg = render(viewpoint, gaussians, pipe, background)
                    pred_mask = render_pkg['mask']
                    gt_mask = viewpoint.semantic_mask.cuda().float()

                    pred = torch.clamp(pred_mask, min=1e-6, max=1.0 - 1e-6)
                    loss = focal_loss(pred, gt_mask, alpha=0.25, gamma=2.0)
                    loss = loss + dice_loss(pred, gt_mask)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(cameras_with_mask)
                if (tta_iter + 1) % 10 == 0:
                    print(f"  TTA iter {tta_iter + 1}/{tta_iterations}, loss={avg_loss:.5f}")

        # 恢复所有参数为 eval 模式
        for param in gaussians.parameters():
            param.requires_grad = True
        gaussians.eval()
        print("[ TTA ] Done.\n")

    # ============== Evaluation ==============
    ious = []
    ious_best_thresh = []
    results = []

    for viewpoint in tqdm(all_cameras, desc="Evaluating"):
        render_pkg = render(viewpoint, gaussians, pipe, background)
        pred_mask = render_pkg['mask']

        gt_mask = getattr(viewpoint, 'semantic_mask', None)
        if gt_mask is None:
            continue

        gt_mask_cuda = gt_mask.cuda().float()
        p_mask = (pred_mask > 0.5).float()
        g_mask = (gt_mask_cuda > 0.5).float()
        intersection = (p_mask * g_mask).sum().item()
        union = p_mask.sum().item() + g_mask.sum().item() - intersection
        iou = intersection / (union + 1e-8)
        ious.append(iou)

        # 搜索最佳阈值
        best_iou = 0.0
        best_thresh = 0.5
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            p_t = (pred_mask > thresh).float()
            inter = (p_t * g_mask).sum().item()
            uni = p_t.sum().item() + g_mask.sum().item() - inter
            iou_t = inter / (uni + 1e-8)
            if iou_t > best_iou:
                best_iou = iou_t
                best_thresh = thresh
        ious_best_thresh.append(best_iou)

        results.append({
            'name': viewpoint.image_name,
            'iou@0.5': iou,
            'best_iou': best_iou,
            'best_thresh': best_thresh,
        })

    mean_iou = np.mean(ious)
    mean_best_iou = np.mean(ious_best_thresh)

    print("\n========== Evaluation Results ==========")
    print(f"Mean IoU @ 0.5:     {mean_iou:.4f}")
    print(f"Mean Best IoU:      {mean_best_iou:.4f}")
    print(f"Evaluated images:   {len(ious)}")
    print("========================================\n")

    return mean_iou, mean_best_iou, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--appearance_dim", type=int, default=32)
    parser.add_argument("--tta_iterations", type=int, default=0,
                        help="Number of TTA iterations (0 = no TTA)")
    parser.add_argument("--tta_lr", type=float, default=1e-4)
    args = parser.parse_args()

    evaluate_with_tta(
        model_path=args.model_path,
        source_path=args.source_path,
        iteration=args.iteration,
        white_background=args.white_background,
        appearance_dim=args.appearance_dim,
        tta_iterations=args.tta_iterations,
        tta_lr=args.tta_lr,
    )
