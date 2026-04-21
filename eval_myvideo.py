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


def evaluate_on_myvideo(model_path, source_path, iteration, white_background=False, appearance_dim=32):
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
        args.add_opacity_dist, args.add_cov_dist, args.add_color_dist
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
    results = []

    for viewpoint in tqdm(all_cameras, desc="Evaluating"):
        render_pkg = render(viewpoint, gaussians, pipe, background)
        pred_mask = render_pkg['mask']

        gt_mask = getattr(viewpoint, 'semantic_mask', None)
        if gt_mask is None:
            print(f"Warning: no GT mask for {viewpoint.image_name}, skipping")
            continue

        gt_mask_cuda = gt_mask.cuda().float()

        # Compute IoU at threshold 0.5
        p_mask = (pred_mask > 0.5).float()
        g_mask = (gt_mask_cuda > 0.5).float()
        intersection = (p_mask * g_mask).sum().item()
        union = p_mask.sum().item() + g_mask.sum().item() - intersection
        iou = intersection / (union + 1e-8)

        # Also try best threshold
        best_iou = 0.0
        best_thresh = 0.5
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            p_mask_t = (pred_mask > thresh).float()
            inter_t = (p_mask_t * g_mask).sum().item()
            union_t = p_mask_t.sum().item() + g_mask.sum().item() - inter_t
            iou_t = inter_t / (union_t + 1e-8)
            if iou_t > best_iou:
                best_iou = iou_t
                best_thresh = thresh

        ious.append(iou)
        results.append({
            'image': viewpoint.image_name,
            'iou_0.5': iou,
            'best_iou': best_iou,
            'best_thresh': best_thresh,
            'pred_mean': pred_mask.mean().item(),
            'gt_mean': gt_mask_cuda.mean().item(),
        })

    mean_iou = np.mean(ious)
    print(f"\n{'='*60}")
    print(f"Mean IoU @ 0.5: {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"{'='*60}")

    # Print per-image results
    print(f"\n{'Image':<25} {'IoU@0.5':>8} {'Best IoU':>8} {'Best Thr':>8} {'PredMean':>10} {'GTMean':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['image']:<25} {r['iou_0.5']:>8.4f} {r['best_iou']:>8.4f} {r['best_thresh']:>8.2f} {r['pred_mean']:>10.4f} {r['gt_mean']:>10.4f}")

    # Save results to file
    out_file = os.path.join(model_path, f'myvideo_iou_iter{iteration}.txt')
    with open(out_file, 'w') as f:
        f.write(f"Mean IoU @ 0.5: {mean_iou:.4f} ({mean_iou*100:.2f}%)\n\n")
        f.write(f"{'Image':<25} {'IoU@0.5':>8} {'Best IoU':>8} {'Best Thr':>8} {'PredMean':>10} {'GTMean':>10}\n")
        f.write("-" * 80 + "\n")
        for r in results:
            f.write(f"{r['image']:<25} {r['iou_0.5']:>8.4f} {r['best_iou']:>8.4f} {r['best_thresh']:>8.2f} {r['pred_mean']:>10.4f} {r['gt_mean']:>10.4f}\n")
    print(f"\nResults saved to {out_file}")

    return mean_iou


if __name__ == '__main__':
    import sys
    # Default: evaluate latest checkpoint on myvideo
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'output/dronev4_2_highpsnr'
    iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 30000

    # Read cfg_args to get white_background and appearance_dim
    cfg_path = os.path.join(model_path, 'cfg_args')
    white_background = False
    appearance_dim = 32
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            content = f.read()
            if 'white_background=True' in content:
                white_background = True
            # Try to extract appearance_dim
            import re
            m = re.search(r'appearance_dim=(\d+)', content)
            if m:
                appearance_dim = int(m.group(1))

    print(f"Model: {model_path}, Iteration: {iteration}")
    print(f"white_background={white_background}, appearance_dim={appearance_dim}")

    evaluate_on_myvideo(
        model_path=model_path,
        source_path='/mnt/data/liufengyang/data/myvideo',
        iteration=iteration,
        white_background=white_background,
        appearance_dim=appearance_dim,
    )
