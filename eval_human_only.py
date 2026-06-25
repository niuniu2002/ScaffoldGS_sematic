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
from train import decode_rendered_mask


def evaluate_on_myvideo(model_path, source_path, iteration, white_background=False, appearance_dim=32, use_per_gaussian_seg=False, num_classes=1, resolution=-1,
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

    # Load scene (this also loads MLP checkpoints automatically)
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    # Replace masks with human-annotated masks if available
    human_masks_dir = os.path.join(source_path, "masks_human")
    if os.path.exists(human_masks_dir):
        print(f"[INFO] Replacing masks with human annotations from {human_masks_dir}")
        for cam in scene.getTrainCameras() + scene.getTestCameras():
            human_mask_path = os.path.join(human_masks_dir, cam.image_name + ".png")
            if os.path.exists(human_mask_path):
                mask_pil = Image.open(human_mask_path)
                if mask_pil.size != (cam.width, cam.height):
                    mask_pil = mask_pil.resize((cam.width, cam.height), Image.NEAREST)
                mask_np = np.array(mask_pil)
                # 兼容 0/255 -> 0/1
                if np.array_equal(np.unique(mask_np), [0, 255]):
                    mask_np = mask_np // 255
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).long()
                cam._semantic_mask = mask_tensor
                cam._semantic_mask_path = human_mask_path
            else:
                print(f"[WARNING] No human mask found for {cam.image_name}, keeping original mask")
    else:
        print(f"[INFO] No masks_human/ directory found at {human_masks_dir}, using default masks")

    # Background
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

    class FakePipe:
        def __init__(self):
            self.debug = False
            self.compute_cov3D_python = False
            self.convert_SHs_python = False
    pipe = FakePipe()

    # Collect only cameras that have human-annotated masks
    all_cameras = []
    for cam in scene.getTrainCameras() + scene.getTestCameras():
        if cam._semantic_mask_path is not None and "masks_human" in cam._semantic_mask_path:
            all_cameras.append(cam)
    print(f"Total cameras to evaluate (human annotated only): {len(all_cameras)}")

    ious = []
    psnrs = []
    results = []

    for viewpoint in tqdm(all_cameras, desc="Evaluating"):
        render_pkg = render(viewpoint, gaussians, pipe, background)
        pred_features = render_pkg['mask']
        pred_mask = decode_rendered_mask(gaussians, pred_features)
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
            ignore_index = 255
            max_valid = gt_labels[gt_labels != ignore_index].max() if (gt_labels != ignore_index).any() else torch.tensor(0, device=gt_labels.device)
            if max_valid > num_classes - 1:
                raise ValueError(
                    f"Mask contains label {max_valid.item()} which exceeds num_classes-1={num_classes-1}. "
                    f"Please ensure mask labels are in [0, {num_classes-1}] or use {ignore_index} as ignore_index."
                )

            iou_per_class = []
            valid_mask = gt_labels != ignore_index
            pred_valid = pred_labels[valid_mask]
            gt_valid = gt_labels[valid_mask]
            for c in range(num_classes):
                p_c = (pred_valid == c).float()
                g_c = (gt_valid == c).float()
                inter = (p_c * g_c).sum().item()
                union = p_c.sum().item() + g_c.sum().item() - inter
                if union > 0:
                    iou_per_class.append(inter / (union + 1e-8))

            miou = np.mean(iou_per_class) if iou_per_class else 0.0
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
        # Print per-class IoU (handle missing classes gracefully)
        max_classes = max(len(r['iou_per_class']) for r in results)
        for c in range(num_classes):
            class_ious = [r['iou_per_class'][c] if c < len(r['iou_per_class']) else 0.0 for r in results]
            class_iou = np.mean(class_ious)
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
                class_ious = [r['iou_per_class'][c] if c < len(r['iou_per_class']) else 0.0 for r in results]
                class_iou = np.mean(class_ious)
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
    ModelParams(parser)
    parser.add_argument('--iteration', type=int, default=30000)
    args = parser.parse_args()
    
    model_path = args.model_path
    iteration = args.iteration

    # Read cfg_args to get white_background, appearance_dim, use_per_gaussian_seg, num_classes
    cfg_path = os.path.join(model_path, 'cfg_args')
    white_background = False
    appearance_dim = 32
    use_per_gaussian_seg = False
    num_classes = args.num_classes
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
            # Try to extract appearance_dim
            import re
            m = re.search(r'appearance_dim=(\d+)', content)
            if m:
                appearance_dim = int(m.group(1))
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

    print(f"Model: {model_path}, Iteration: {iteration}")
    print(f"white_background={white_background}, appearance_dim={appearance_dim}, use_per_gaussian_seg={use_per_gaussian_seg}, num_classes={num_classes}")
    print(f"seg_feature_dim={seg_feature_dim}, seg_decoder_hidden={seg_decoder_hidden}, seg_decoder_layers={seg_decoder_layers}, dual_feature={dual_feature}")

    evaluate_on_myvideo(
        model_path=model_path,
        source_path=args.source_path,
        iteration=iteration,
        white_background=white_background,
        appearance_dim=appearance_dim,
        use_per_gaussian_seg=use_per_gaussian_seg,
        num_classes=num_classes,
        resolution=args.resolution if hasattr(args, 'resolution') else -1,
        seg_feature_dim=seg_feature_dim,
        seg_decoder_hidden=seg_decoder_hidden,
        seg_decoder_layers=seg_decoder_layers,
        dual_feature=dual_feature,
    )
