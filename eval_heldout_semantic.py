"""
Held-out Semantic Evaluation Script
====================================
Only evaluates test cameras (or a manual test list), never mixing train cameras.
Outputs: PSNR, FG IoU, Binary mIoU, Multi-class mIoU, per-class IoU.
Supports ignore_index=255.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.image_utils import psnr as compute_psnr
from train import decode_rendered_mask


class FakePipe:
    def __init__(self):
        self.debug = False
        self.compute_cov3D_python = False
        self.convert_SHs_python = False


def evaluate_heldout(model_path, source_path, iteration, num_classes=1,
                     white_background=False, use_per_gaussian_seg=False,
                     manual_test_list=None, ignore_index=255,
                     appearance_dim=0, save_per_view=False,
                     seg_feature_dim=0, seg_decoder_hidden=64, seg_decoder_layers=2,
                     dual_feature=False, resolution=-1):
    """
    Evaluate a trained model on held-out test cameras.

    Args:
        model_path: path to output directory containing point_cloud/
        source_path: path to scene data (COLMAP format)
        iteration: checkpoint iteration to load
        num_classes: 1 for binary, >=2 for multi-class
        white_background: dataset uses white background
        use_per_gaussian_seg: per-Gaussian segmentation flag
        manual_test_list: path to .txt file with image names (one per line) to evaluate.
                          If None, uses scene.getTestCameras().
        ignore_index: label value to ignore in metrics (default 255)
        appearance_dim: appearance embedding dimension
        save_per_view: if True, save per-view results to JSON

    Returns:
        dict with metrics
    """
    parser = argparse.ArgumentParser()
    ModelParams(parser)
    arg_list = [
        '--source_path', source_path,
        '--model_path', model_path,
        '--eval',
        '--appearance_dim', str(appearance_dim),
    ]
    if white_background:
        arg_list.append('--white_background')
    if use_per_gaussian_seg:
        arg_list.append('--use_per_gaussian_seg')
    if resolution != -1:
        arg_list.extend(['--resolution', str(resolution)])
    args = parser.parse_args(arg_list)

    gaussians = GaussianModel(
        args.feat_dim, args.n_offsets, args.voxel_size,
        args.update_depth, args.update_init_factor, args.update_hierachy_factor,
        args.use_feat_bank, args.appearance_dim, args.ratio,
        args.add_opacity_dist, args.add_cov_dist, args.add_color_dist,
        use_per_gaussian_seg=args.use_per_gaussian_seg if hasattr(args, 'use_per_gaussian_seg') else False,
        num_classes=num_classes,
        dual_feature=dual_feature,
        seg_feature_dim=seg_feature_dim,
        seg_decoder_hidden=seg_decoder_hidden,
        seg_decoder_layers=seg_decoder_layers,
    )

    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')
    pipe = FakePipe()

    # Determine camera list
    if manual_test_list is not None and os.path.exists(manual_test_list):
        with open(manual_test_list, 'r') as f:
            test_names = set(line.strip() for line in f if line.strip())
        all_cameras = scene.getTestCameras() + scene.getTrainCameras()
        cameras = [c for c in all_cameras if getattr(c, 'image_name', '') in test_names]
        print(f"Manual test list: {len(test_names)} names, matched {len(cameras)} cameras")
    else:
        cameras = scene.getTestCameras()
        print(f"Using scene test cameras: {len(cameras)}")

    if len(cameras) == 0:
        print("ERROR: No cameras to evaluate!")
        return {}

    # Accumulators
    psnrs = []
    per_view_results = []

    if num_classes == 1:
        # Binary accumulators
        binary_inter_union = {'bg': [0.0, 0.0], 'fg': [0.0, 0.0]}
    else:
        # Multi-class accumulators
        inter_union = [[0.0, 0.0] for _ in range(num_classes)]
        binary_inter_union = {'bg': [0.0, 0.0], 'fg': [0.0, 0.0]}  # collapsed binary
        gt_class_counts = np.zeros(num_classes, dtype=np.int64)
        pred_class_counts = np.zeros(num_classes, dtype=np.int64)
        total_pixels = 0

    for viewpoint in tqdm(cameras, desc="Evaluating"):
        render_pkg = render(viewpoint, gaussians, pipe, background)
        pred_features = render_pkg['mask']  # [seg_feature_dim, H, W] or [C, H, W] legacy
        pred_mask = decode_rendered_mask(gaussians, pred_features)
        rendered_image = render_pkg['render']

        gt_mask = getattr(viewpoint, 'semantic_mask', None)
        if gt_mask is None:
            print(f"Warning: no GT mask for {viewpoint.image_name}, skipping")
            continue

        gt_labels = gt_mask.cuda().squeeze(0).long()  # [H, W]
        gt_image = viewpoint.original_image.cuda()

        # PSNR
        mse = torch.mean((rendered_image - gt_image) ** 2).item()
        psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
        psnrs.append(psnr)

        valid_mask = gt_labels != ignore_index
        n_valid = valid_mask.sum().item()

        if num_classes == 1:
            # Binary evaluation
            pred_bin = (pred_mask[0] > 0.5).float()
            gt_bin = (gt_mask.cuda().float().squeeze(0) > 0.5).float()
            # FG
            inter_fg = (pred_bin * gt_bin).sum().item()
            union_fg = pred_bin.sum().item() + gt_bin.sum().item() - inter_fg
            binary_inter_union['fg'][0] += inter_fg
            binary_inter_union['fg'][1] += union_fg
            # BG
            pred_bg = (pred_mask[0] <= 0.5).float()
            gt_bg = (gt_mask.cuda().float().squeeze(0) <= 0.5).float()
            inter_bg = (pred_bg * gt_bg).sum().item()
            union_bg = pred_bg.sum().item() + gt_bg.sum().item() - inter_bg
            binary_inter_union['bg'][0] += inter_bg
            binary_inter_union['bg'][1] += union_bg

            fg_iou_view = inter_fg / (union_fg + 1e-8)
            per_view_results.append({
                'image': viewpoint.image_name,
                'psnr': psnr,
                'fg_iou': fg_iou_view,
            })
        else:
            # Multi-class evaluation
            pred_labels = pred_mask.argmax(dim=0)  # [H, W]
            pred_valid = pred_labels[valid_mask]
            gt_valid = gt_labels[valid_mask]
            total_pixels += n_valid

            # Per-class IoU
            view_ious = []
            for c in range(num_classes):
                p_c = (pred_valid == c).float()
                g_c = (gt_valid == c).float()
                inter = (p_c * g_c).sum().item()
                union = p_c.sum().item() + g_c.sum().item() - inter
                inter_union[c][0] += inter
                inter_union[c][1] += union
                view_ious.append(inter / (union + 1e-8) if union > 0 else float('nan'))
                gt_class_counts[c] += (gt_valid == c).sum().item()
                pred_class_counts[c] += (pred_valid == c).sum().item()

            # Collapsed binary: class 0 = BG, classes 1..C-1 = FG
            p_fg = (pred_valid > 0).float()
            g_fg = (gt_valid > 0).float()
            inter_fg = (p_fg * g_fg).sum().item()
            union_fg = p_fg.sum().item() + g_fg.sum().item() - inter_fg
            binary_inter_union['fg'][0] += inter_fg
            binary_inter_union['fg'][1] += union_fg

            p_bg = (pred_valid == 0).float()
            g_bg = (gt_valid == 0).float()
            inter_bg = (p_bg * g_bg).sum().item()
            union_bg = p_bg.sum().item() + g_bg.sum().item() - inter_bg
            binary_inter_union['bg'][0] += inter_bg
            binary_inter_union['bg'][1] += union_bg

            miou_view = np.nanmean(view_ious) if any(not np.isnan(x) for x in view_ious) else 0.0
            per_view_results.append({
                'image': viewpoint.image_name,
                'psnr': psnr,
                'miou': miou_view,
                'ious': view_ious,
            })

    # Global metrics
    mean_psnr = np.mean(psnrs) if psnrs else 0.0

    iou_fg = binary_inter_union['fg'][0] / (binary_inter_union['fg'][1] + 1e-8)
    iou_bg = binary_inter_union['bg'][0] / (binary_inter_union['bg'][1] + 1e-8)
    binary_miou = (iou_fg + iou_bg) / 2.0

    print(f"\n{'='*60}")
    print(f"Held-out Evaluation: {len(cameras)} cameras")
    print(f"Mean PSNR        : {mean_psnr:.4f}")
    print(f"{'='*60}")
    print(f"Binary FG IoU    : {iou_fg:.4f}")
    print(f"Binary BG IoU    : {iou_bg:.4f}")
    print(f"Binary mIoU      : {binary_miou:.4f}")

    if num_classes > 1:
        class_ious = []
        for c in range(num_classes):
            inter, union = inter_union[c]
            iou_c = inter / (union + 1e-8) if union > 0 else float('nan')
            class_ious.append(iou_c)
        multiclass_miou = np.nanmean(class_ious) if any(not np.isnan(x) for x in class_ious) else 0.0

        print(f"\n--- Multi-class Metrics ---")
        for c in range(num_classes):
            print(f"  Class {c} IoU    : {class_ious[c]:.4f}")
        print(f"  multiclass mIoU: {multiclass_miou:.4f}")

        gt_dist = gt_class_counts / total_pixels if total_pixels > 0 else gt_class_counts
        pred_dist = pred_class_counts / total_pixels if total_pixels > 0 else pred_class_counts
        print(f"\n--- GT Class Distribution ---")
        for c in range(num_classes):
            print(f"  Class {c}        : {gt_dist[c]:.4f} ({gt_class_counts[c]} px)")
        print(f"\n--- Pred Class Distribution ---")
        for c in range(num_classes):
            print(f"  Class {c}        : {pred_dist[c]:.4f} ({pred_class_counts[c]} px)")
    else:
        multiclass_miou = None
        class_ious = None
        gt_dist = None
        pred_dist = None

    # Save to file
    out_file = os.path.join(model_path, f'heldout_semantic_iter{iteration}.txt')
    with open(out_file, 'w') as f:
        f.write(f"Held-out Evaluation: {len(cameras)} cameras\n")
        f.write(f"Mean PSNR        : {mean_psnr:.4f}\n")
        f.write(f"Binary FG IoU    : {iou_fg:.4f}\n")
        f.write(f"Binary BG IoU    : {iou_bg:.4f}\n")
        f.write(f"Binary mIoU      : {binary_miou:.4f}\n")
        if num_classes > 1:
            f.write(f"multiclass mIoU  : {multiclass_miou:.4f}\n")
            for c in range(num_classes):
                f.write(f"Class {c} IoU    : {class_ious[c]:.4f}\n")
            f.write(f"\nGT Distribution:\n")
            for c in range(num_classes):
                f.write(f"  Class {c}        : {gt_dist[c]:.4f}\n")
            f.write(f"\nPred Distribution:\n")
            for c in range(num_classes):
                f.write(f"  Class {c}        : {pred_dist[c]:.4f}\n")
        f.write(f"\nPer-View Results:\n")
        for r in per_view_results:
            if num_classes > 1:
                iou_str = " ".join([f"{x:.4f}" for x in r['ious']])
                f.write(f"{r['image']}	PSNR={r['psnr']:.2f}	mIoU={r['miou']:.4f}	IoUs=[{iou_str}]\n")
            else:
                f.write(f"{r['image']}\tPSNR={r['psnr']:.2f}\tFG_IoU={r['fg_iou']:.4f}\n")
    print(f"\nResults saved to {out_file}")

    if save_per_view:
        json_out = os.path.join(model_path, f'heldout_semantic_iter{iteration}.json')
        with open(json_out, 'w') as f:
            json.dump({
                'num_cameras': len(cameras),
                'mean_psnr': mean_psnr,
                'binary_fg_iou': iou_fg,
                'binary_bg_iou': iou_bg,
                'binary_miou': binary_miou,
                'multiclass_miou': multiclass_miou,
                'class_ious': class_ious,
                'per_view': per_view_results,
            }, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"JSON saved to {json_out}")

    return {
        'psnr': mean_psnr,
        'binary_fg_iou': iou_fg,
        'binary_bg_iou': iou_bg,
        'binary_miou': binary_miou,
        'multiclass_miou': multiclass_miou,
        'class_ious': class_ious,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Held-out semantic evaluation")
    ModelParams(parser)
    parser.add_argument('--iteration', type=int, default=30000, help="Checkpoint iteration")
    parser.add_argument('--manual_test_list', type=str, default=None, help="Path to .txt with image names to evaluate")
    parser.add_argument('--ignore_index', type=int, default=255, help="Label value to ignore in metrics")
    parser.add_argument('--save_per_view', action='store_true', help="Save per-view results to JSON")
    args = parser.parse_args()

    # Fall back to saved cfg_args for source_path / scene config
    cfg_path = os.path.join(args.model_path, 'cfg_args')
    if (not args.source_path or args.source_path == '') and os.path.exists(cfg_path):
        from argparse import Namespace as _NS
        with open(cfg_path) as f:
            cfg_ns = eval(f.read(), {"Namespace": _NS})
        args.source_path = cfg_ns.source_path
    if os.path.exists(cfg_path):
        import re
        with open(cfg_path) as f:
            content = f.read()
        if 'white_background=True' in content:
            args.white_background = True
        if 'use_per_gaussian_seg=True' in content:
            args.use_per_gaussian_seg = True
        if 'dual_feature=True' in content:
            args.dual_feature = True
        m = re.search(r'appearance_dim=(\d+)', content)
        if m:
            args.appearance_dim = int(m.group(1))
        m_nc = re.search(r'num_classes=(\d+)', content)
        if m_nc:
            args.num_classes = int(m_nc.group(1))
        m_sfd = re.search(r'seg_feature_dim=(\d+)', content)
        if m_sfd:
            args.seg_feature_dim = int(m_sfd.group(1))
        m_sdh = re.search(r'seg_decoder_hidden=(\d+)', content)
        if m_sdh:
            args.seg_decoder_hidden = int(m_sdh.group(1))
        m_sdl = re.search(r'seg_decoder_layers=(\d+)', content)
        if m_sdl:
            args.seg_decoder_layers = int(m_sdl.group(1))

    evaluate_heldout(
        model_path=args.model_path,
        source_path=args.source_path,
        iteration=args.iteration,
        num_classes=args.num_classes,
        white_background=args.white_background,
        use_per_gaussian_seg=args.use_per_gaussian_seg,
        manual_test_list=args.manual_test_list,
        ignore_index=args.ignore_index,
        appearance_dim=args.appearance_dim,
        save_per_view=args.save_per_view,
        seg_feature_dim=getattr(args, 'seg_feature_dim', 0),
        seg_decoder_hidden=getattr(args, 'seg_decoder_hidden', 64),
        seg_decoder_layers=getattr(args, 'seg_decoder_layers', 2),
        dual_feature=getattr(args, 'dual_feature', False),
        resolution=getattr(args, 'resolution', -1),
    )
