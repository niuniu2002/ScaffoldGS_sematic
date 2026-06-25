"""
Evaluate trained model on multi-class segmentation.
Outputs:
  1. per-class IoU
  2. multiclass mIoU
  3. collapsed binary BG/FG/binary mIoU
  4. GT class distribution
  5. Pred class distribution
"""
import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from train import decode_rendered_mask


def evaluate_multiclass(model_path, source_path, iteration, white_background=False,
                        appearance_dim=32, use_per_gaussian_seg=False, num_classes=2,
                        seg_feature_dim=0, seg_decoder_hidden=64, seg_decoder_layers=2, dual_feature=False):
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

    class FakePipe:
        def __init__(self):
            self.debug = False
            self.compute_cov3D_python = False
            self.convert_SHs_python = False
    pipe = FakePipe()

    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    print(f"Total cameras to evaluate: {len(all_cameras)}")

    # Accumulators for multi-class IoU (global across all views)
    inter_union = [[0.0, 0.0] for _ in range(num_classes)]
    # Binary collapse accumulators
    binary_inter_union = {'fg': [0.0, 0.0], 'bg': [0.0, 0.0]}
    # Class distributions
    gt_class_counts = np.zeros(num_classes, dtype=np.int64)
    pred_class_counts = np.zeros(num_classes, dtype=np.int64)
    total_pixels = 0

    psnrs = []
    per_view_results = []

    ignore_index = 255

    for viewpoint in tqdm(all_cameras, desc="Evaluating"):
        render_pkg = render(viewpoint, gaussians, pipe, background)
        pred_features = render_pkg['mask']  # [seg_feature_dim, H, W] or [C, H, W] legacy
        pred_logits = decode_rendered_mask(gaussians, pred_features)
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

        # Predicted labels from argmax
        pred_labels = pred_logits.argmax(dim=0)  # [H, W]

        # Valid mask (ignore 255)
        valid_mask = gt_labels != ignore_index
        pred_valid = pred_labels[valid_mask]
        gt_valid = gt_labels[valid_mask]
        n_valid = valid_mask.sum().item()
        total_pixels += n_valid

        # Update class distributions
        for c in range(num_classes):
            gt_class_counts[c] += (gt_valid == c).sum().item()
            pred_class_counts[c] += (pred_valid == c).sum().item()

        # Per-class IoU (global accumulation)
        view_ious = []
        for c in range(num_classes):
            p_c = (pred_valid == c).float()
            g_c = (gt_valid == c).float()
            inter = (p_c * g_c).sum().item()
            union = p_c.sum().item() + g_c.sum().item() - inter
            inter_union[c][0] += inter
            inter_union[c][1] += union
            view_ious.append(inter / (union + 1e-8) if union > 0 else float('nan'))

        # Collapsed binary: treat class 0 as BG, classes 1..C-1 as FG
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
    class_ious = []
    for c in range(num_classes):
        inter, union = inter_union[c]
        iou_c = inter / (union + 1e-8) if union > 0 else float('nan')
        class_ious.append(iou_c)

    multiclass_miou = np.nanmean(class_ious) if any(not np.isnan(x) for x in class_ious) else 0.0

    iou_fg = binary_inter_union['fg'][0] / (binary_inter_union['fg'][1] + 1e-8)
    iou_bg = binary_inter_union['bg'][0] / (binary_inter_union['bg'][1] + 1e-8)
    binary_miou = (iou_fg + iou_bg) / 2.0

    gt_dist = gt_class_counts / total_pixels if total_pixels > 0 else gt_class_counts
    pred_dist = pred_class_counts / total_pixels if total_pixels > 0 else pred_class_counts

    mean_psnr = np.mean(psnrs)

    print(f"\n{'='*60}")
    print(f"Mean PSNR        : {mean_psnr:.4f}")
    print(f"{'='*60}")
    print(f"\n--- Multi-class Metrics ---")
    for c in range(num_classes):
        print(f"  Class {c} IoU    : {class_ious[c]:.4f}")
    print(f"  multiclass mIoU: {multiclass_miou:.4f}")

    print(f"\n--- Collapsed Binary Metrics ---")
    print(f"  BG IoU         : {iou_bg:.4f}")
    print(f"  FG IoU         : {iou_fg:.4f}")
    print(f"  binary mIoU    : {binary_miou:.4f}")

    print(f"\n--- GT Class Distribution ---")
    for c in range(num_classes):
        print(f"  Class {c}        : {gt_dist[c]:.4f} ({gt_class_counts[c]} px)")

    print(f"\n--- Pred Class Distribution ---")
    for c in range(num_classes):
        print(f"  Class {c}        : {pred_dist[c]:.4f} ({pred_class_counts[c]} px)")

    print(f"\n{'='*60}")
    print(f"{'Image':<25} {'PSNR':>8} {'mIoU':>8} " + " ".join([f"IoU{c}">8 for c in range(num_classes)]))
    print("-" * (60 + num_classes * 9))
    for r in per_view_results:
        iou_str = " ".join([f"{x:>8.4f}" for x in r['ious']])
        print(f"{r['image']:<25} {r['psnr']:>8.2f} {r['miou']:>8.4f} {iou_str}")

    out_file = os.path.join(model_path, f'multiclass_iou_iter{iteration}.txt')
    with open(out_file, 'w') as f:
        f.write(f"Mean PSNR        : {mean_psnr:.4f}\n")
        f.write(f"multiclass mIoU  : {multiclass_miou:.4f}\n")
        for c in range(num_classes):
            f.write(f"Class {c} IoU    : {class_ious[c]:.4f}\n")
        f.write(f"binary BG IoU    : {iou_bg:.4f}\n")
        f.write(f"binary FG IoU    : {iou_fg:.4f}\n")
        f.write(f"binary mIoU      : {binary_miou:.4f}\n")
        f.write(f"\nGT Distribution:\n")
        for c in range(num_classes):
            f.write(f"  Class {c}        : {gt_dist[c]:.4f}\n")
        f.write(f"\nPred Distribution:\n")
        for c in range(num_classes):
            f.write(f"  Class {c}        : {pred_dist[c]:.4f}\n")
        f.write(f"\n{'Image':<25} {'PSNR':>8} {'mIoU':>8} " + " ".join([f"IoU{c}">8 for c in range(num_classes)]) + "\n")
        f.write("-" * (60 + num_classes * 9) + "\n")
        for r in per_view_results:
            iou_str = " ".join([f"{x:>8.4f}" for x in r['ious']])
            f.write(f"{r['image']:<25} {r['psnr']:>8.2f} {r['miou']:>8.4f} {iou_str}\n")
    print(f"\nResults saved to {out_file}")

    return {
        'psnr': mean_psnr,
        'multiclass_miou': multiclass_miou,
        'class_ious': class_ious,
        'binary_bg_iou': iou_bg,
        'binary_fg_iou': iou_fg,
        'binary_miou': binary_miou,
        'gt_dist': gt_dist,
        'pred_dist': pred_dist,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ModelParams(parser)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()

    model_path = args.model_path
    iteration = args.iteration
    num_classes = args.num_classes

    cfg_path = os.path.join(model_path, 'cfg_args')
    white_background = False
    appearance_dim = 32
    use_per_gaussian_seg = False
    seg_feature_dim = getattr(args, 'seg_feature_dim', 0)
    seg_decoder_hidden = getattr(args, 'seg_decoder_hidden', 64)
    seg_decoder_layers = getattr(args, 'seg_decoder_layers', 2)
    dual_feature = False
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            content = f.read()
            if 'white_background=True' in content:
                white_background = True
            if 'use_per_gaussian_seg=True' in content:
                use_per_gaussian_seg = True
            import re
            m = re.search(r'appearance_dim=(\d+)', content)
            if m:
                appearance_dim = int(m.group(1))
            m_nc = re.search(r'num_classes=(\d+)', content)
            if m_nc:
                cfg_num_classes = int(m_nc.group(1))
                if args.num_classes == 2 and cfg_num_classes > 2:
                    num_classes = cfg_num_classes
            m_sfd = re.search(r'seg_feature_dim=(\d+)', content)
            if m_sfd:
                seg_feature_dim = int(m_sfd.group(1))
            m_sdh = re.search(r'seg_decoder_hidden=(\d+)', content)
            if m_sdh:
                seg_decoder_hidden = int(m_sdh.group(1))
            m_sdl = re.search(r'seg_decoder_layers=(\d+)', content)
            if m_sdl:
                seg_decoder_layers = int(m_sdl.group(1))
            if 'dual_feature=True' in content:
                dual_feature = True

    print(f"Model: {model_path}, Iteration: {iteration}, num_classes: {num_classes}")

    evaluate_multiclass(
        model_path=model_path,
        source_path='/mnt/data/liufengyang/data/myvideo',
        iteration=iteration,
        white_background=white_background,
        appearance_dim=appearance_dim,
        use_per_gaussian_seg=use_per_gaussian_seg,
        num_classes=num_classes,
        seg_feature_dim=seg_feature_dim,
        seg_decoder_hidden=seg_decoder_hidden,
        seg_decoder_layers=seg_decoder_layers,
        dual_feature=dual_feature,
    )
