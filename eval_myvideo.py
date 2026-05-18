"""
Evaluate trained model on myvideo dataset (human-annotated masks).
Computes per-image IoU and mean IoU across all views.
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


def evaluate_on_myvideo(model_path, source_path, iteration, white_background=False, appearance_dim=32, use_per_gaussian_seg=False, num_classes=1):
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
    )

    # Load scene (this also loads MLP checkpoints automatically)
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    # Background
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    class FakePipe:
        def __init__(self):
            self.debug = False
            self.compute_cov3D_python = False
            self.convert_SHs_python = False
    pipe = FakePipe()

    # Collect all cameras (train + test)
    all_cameras = scene.getTrainCameras() + scene.getTestCameras()
    print(f"Total cameras to evaluate: {len(all_cameras)}")

    ious = []
    psnrs = []
    results = []

    for viewpoint in tqdm(all_cameras, desc="Evaluating"):
        render_pkg = render(viewpoint, gaussians, pipe, background)
        pred_mask = render_pkg['mask']
        rendered_image = render_pkg['render']

        gt_mask = getattr(viewpoint, 'semantic_mask', None)
        if gt_mask is None:
            print(f"Warning: no GT mask for {viewpoint.image_name}, skipping")
            continue

        gt_mask_cuda = gt_mask.cuda().float()
        gt_image = viewpoint.original_image.cuda()

        # Compute PSNR
        mse = torch.mean((rendered_image - gt_image) ** 2).item()
        psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
        psnrs.append(psnr)

        if num_classes == 1:
            # ========== Binary evaluation ==========
            p_mask = (pred_mask > 0.5).float()
            g_mask = (gt_mask_cuda > 0.5).float()

            intersection_fg = (p_mask * g_mask).sum().item()
            union_fg = p_mask.sum().item() + g_mask.sum().item() - intersection_fg
            iou_fg = intersection_fg / (union_fg + 1e-8)

            p_bg = 1.0 - p_mask
            g_bg = 1.0 - g_mask
            intersection_bg = (p_bg * g_bg).sum().item()
            union_bg = p_bg.sum().item() + g_bg.sum().item() - intersection_bg
            iou_bg = intersection_bg / (union_bg + 1e-8)

            miou = (iou_bg + iou_fg) / 2.0

            # Best threshold search
            best_iou_fg = 0.0
            best_thresh = 0.5
            for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                p_mask_t = (pred_mask > thresh).float()
                inter_t = (p_mask_t * g_mask).sum().item()
                union_t = p_mask_t.sum().item() + g_mask.sum().item() - inter_t
                iou_t = inter_t / (union_t + 1e-8)
                if iou_t > best_iou_fg:
                    best_iou_fg = iou_t
                    best_thresh = thresh

            result = {
                'image': viewpoint.image_name,
                'iou_fg': iou_fg,
                'iou_bg': iou_bg,
                'miou': miou,
                'psnr': psnr,
                'best_iou_fg': best_iou_fg,
                'best_thresh': best_thresh,
                'pred_mean': pred_mask.mean().item(),
                'gt_mean': gt_mask_cuda.mean().item(),
            }
        else:
            # ========== Multi-class evaluation ==========
            # pred_mask: [C, H, W] logits
            # gt_mask: [1, H, W] or [H, W]
            pred_labels = pred_mask.argmax(dim=0)  # [H, W]
            gt_labels = gt_mask_cuda.squeeze(0).long()  # [H, W]
            if gt_labels.max() > num_classes - 1:
                gt_labels = (gt_labels > 0).long()

            iou_per_class = []
            for c in range(num_classes):
                p_c = (pred_labels == c).float()
                g_c = (gt_labels == c).float()
                inter = (p_c * g_c).sum().item()
                union = p_c.sum().item() + g_c.sum().item() - inter
                iou_c = inter / (union + 1e-8)
                iou_per_class.append(iou_c)

            miou = np.mean(iou_per_class)
            result = {
                'image': viewpoint.image_name,
                'miou': miou,
                'psnr': psnr,
                'iou_per_class': iou_per_class,
                'pred_mean': pred_mask.mean().item(),
                'gt_mean': gt_mask_cuda.mean().item(),
            }

        ious.append(miou)
        results.append(result)

    mean_miou = np.mean(ious)
    mean_psnr = np.mean(psnrs)
    print(f"\n{'='*60}")
    print(f"Mean PSNR        : {mean_psnr:.4f}")
    print(f"Mean mIoU        : {mean_miou:.4f} ({mean_miou*100:.2f}%)")
    print(f"{'='*60}")

    if num_classes == 1:
        mean_fg = np.mean([r['iou_fg'] for r in results])
        mean_bg = np.mean([r['iou_bg'] for r in results])
        print(f"Mean BG IoU      : {mean_bg:.4f}")
        print(f"Mean FG IoU      : {mean_fg:.4f}")
        print(f"\n{'Image':<25} {'PSNR':>8} {'IoU_bg':>8} {'IoU_fg':>8} {'mIoU':>8} {'BestFG':>8} {'BestThr':>8}")
        print("-" * 90)
        for r in results:
            print(f"{r['image']:<25} {r['psnr']:>8.2f} {r['iou_bg']:>8.4f} {r['iou_fg']:>8.4f} {r['miou']:>8.4f} {r['best_iou_fg']:>8.4f} {r['best_thresh']:>8.2f}")
    else:
        # Print per-class IoU
        for c in range(num_classes):
            class_iou = np.mean([r['iou_per_class'][c] for r in results])
            print(f"Mean IoU class {c} : {class_iou:.4f}")
        print(f"\n{'Image':<25} {'PSNR':>8} {'mIoU':>8}")
        print("-" * 50)
        for r in results:
            print(f"{r['image']:<25} {r['psnr']:>8.2f} {r['miou']:>8.4f}")

    # Save results to file
    out_file = os.path.join(model_path, f'myvideo_iou_iter{iteration}.txt')
    with open(out_file, 'w') as f:
        f.write(f"Mean PSNR        : {mean_psnr:.4f}\n")
        f.write(f"Mean mIoU        : {mean_miou:.4f} ({mean_miou*100:.2f}%)\n")
        if num_classes == 1:
            mean_fg = np.mean([r['iou_fg'] for r in results])
            mean_bg = np.mean([r['iou_bg'] for r in results])
            f.write(f"Mean BG IoU      : {mean_bg:.4f}\n")
            f.write(f"Mean FG IoU      : {mean_fg:.4f}\n")
        else:
            for c in range(num_classes):
                class_iou = np.mean([r['iou_per_class'][c] for r in results])
                f.write(f"Mean IoU class {c} : {class_iou:.4f}\n")
        f.write("\n")
        if num_classes == 1:
            f.write(f"{'Image':<25} {'PSNR':>8} {'IoU_bg':>8} {'IoU_fg':>8} {'mIoU':>8} {'BestFG':>8} {'BestThr':>8}\n")
            f.write("-" * 90 + "\n")
            for r in results:
                f.write(f"{r['image']:<25} {r['psnr']:>8.2f} {r['iou_bg']:>8.4f} {r['iou_fg']:>8.4f} {r['miou']:>8.4f} {r['best_iou_fg']:>8.4f} {r['best_thresh']:>8.2f}\n")
        else:
            f.write(f"{'Image':<25} {'PSNR':>8} {'mIoU':>8}\n")
            f.write("-" * 50 + "\n")
            for r in results:
                f.write(f"{r['image']:<25} {r['psnr']:>8.2f} {r['miou']:>8.4f}\n")
    print(f"\nResults saved to {out_file}")

    if num_classes == 1:
        mean_fg = np.mean([r['iou_fg'] for r in results])
        mean_bg = np.mean([r['iou_bg'] for r in results])
        return mean_miou, mean_fg, mean_bg, mean_psnr
    else:
        return mean_miou, 0.0, 0.0, mean_psnr


if __name__ == '__main__':
    import sys
    # Support both positional args and argparse-style args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='output/dronev4_2_highpsnr')
    parser.add_argument('--source_path', type=str, default='/mnt/data/liufengyang/data/myvideo')
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--white_background', action='store_true')
    parser.add_argument('--appearance_dim', type=int, default=32)
    parser.add_argument('--use_per_gaussian_seg', action='store_true')
    args = parser.parse_args()
    
    model_path = args.model_path
    iteration = args.iteration

    # Read cfg_args to get white_background, appearance_dim, use_per_gaussian_seg
    cfg_path = os.path.join(model_path, 'cfg_args')
    white_background = False
    appearance_dim = 32
    use_per_gaussian_seg = False
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            content = f.read()
            if 'white_background=True' in content:
                white_background = True
            if 'use_per_gaussian_seg=True' in content:
                use_per_gaussian_seg = True
            # Try to extract appearance_dim
            import re
            m = re.search(r'appearance_dim=(\d+)', content)
            if m:
                appearance_dim = int(m.group(1))

    print(f"Model: {model_path}, Iteration: {iteration}")
    print(f"white_background={white_background}, appearance_dim={appearance_dim}, use_per_gaussian_seg={use_per_gaussian_seg}")

    evaluate_on_myvideo(
        model_path=model_path,
        source_path='/mnt/data/liufengyang/data/myvideo',
        iteration=iteration,
        white_background=white_background,
        appearance_dim=appearance_dim,
        use_per_gaussian_seg=use_per_gaussian_seg,
    )
