"""
Standalone evaluation: load a trained checkpoint, render test images from test_list.txt,
and compute PSNR / SSIM / LPIPS / mIoU.

Usage:
    cd /mnt/data/liufengyang/data/Scaffold-GSLFY
    python eval_from_checkpoint.py -m output/20260528_mw01_sem5k_upd15k --iteration 30000
    python eval_from_checkpoint.py -m output/20260528_mw04_sem5k_upd15k --iteration 30000
"""

import os, sys, json, torch, torchvision, numpy as np
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(__file__))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render, prefilter_voxel
from scene.dataset_readers import (
    readColmapCameras, read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text, CameraInfo,
)
from scene.cameras import Camera
from utils.camera_utils import loadCam
from utils.loss_utils import ssim
from utils.image_utils import psnr
import lpips
from train import decode_rendered_mask


def readColmapSceneInfo_with_testlist(path, images, test_list_file):
    """Read COLMAP scene and split into train/test using test_list.txt."""
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(path, "sparse/0", "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(path, "sparse/0", "cameras.bin"))
    except:
        cam_extrinsics = read_extrinsics_text(os.path.join(path, "sparse/0", "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(path, "sparse/0", "cameras.txt"))

    reading_dir = "images" if images is None else images
    cam_infos = sorted(
        readColmapCameras(cam_extrinsics, cam_intrinsics, os.path.join(path, reading_dir)),
        key=lambda x: x.image_name,
    )

    with open(test_list_file) as f:
        test_names_raw = [line.strip() for line in f if line.strip()]
    # Strip extensions: COLMAP image_name uses stem, test_list.txt may have .png/.jpg
    test_names = set(Path(n).stem for n in test_names_raw)

    test_cam_infos = [c for c in cam_infos if c.image_name in test_names]
    train_cam_infos = [c for c in cam_infos if c.image_name not in test_names]

    print(f"Total cameras: {len(cam_infos)}")
    print(f"Test cameras (from test_list.txt): {len(test_cam_infos)}")
    print(f"Train cameras: {len(train_cam_infos)}")

    return train_cam_infos, test_cam_infos


def main():
    parser = ArgumentParser(description="Evaluate a trained checkpoint using test_list.txt")
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="Checkpoint iteration (-1 for latest)")
    parser.add_argument("--skip_render", action="store_true", help="Only compute metrics from existing renders")
    parser.add_argument("--skip_mask_metrics", action="store_true", help="Skip mIoU computation")
    args = get_combined_args(parser)

    # Fix: get_combined_args may override cfg_args source_path with empty default
    from argparse import Namespace as _NS
    cfg_path = os.path.join(args.model_path, "cfg_args")
    if (not args.source_path or args.source_path == "") and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg_ns = eval(f.read(), {"Namespace": _NS})
        args.source_path = cfg_ns.source_path

    # Pull segmentation config from saved cfg_args when available
    seg_feature_dim = getattr(args, "seg_feature_dim", 8)
    seg_decoder_hidden = getattr(args, "seg_decoder_hidden", 64)
    seg_decoder_layers = getattr(args, "seg_decoder_layers", 2)
    dual_feature = getattr(args, "dual_feature", False)
    if os.path.exists(cfg_path):
        import re
        with open(cfg_path) as f:
            content = f.read()
        m = re.search(r"seg_feature_dim=(\d+)", content)
        if m:
            seg_feature_dim = int(m.group(1))
        m = re.search(r"seg_decoder_hidden=(\d+)", content)
        if m:
            seg_decoder_hidden = int(m.group(1))
        m = re.search(r"seg_decoder_layers=(\d+)", content)
        if m:
            seg_decoder_layers = int(m.group(1))
        if 'dual_feature=True' in content:
            dual_feature = True

    source_path = args.source_path
    model_path = args.model_path
    test_list_file = os.path.join(source_path, "test_list.txt")

    if not os.path.exists(test_list_file):
        print(f"ERROR: {test_list_file} not found")
        print(f"  source_path={source_path}")
        print(f"  cwd={os.getcwd()}")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Source: {source_path}")

    # --- Load checkpoint ---
    gaussians = GaussianModel(
        args.feat_dim, args.n_offsets, args.voxel_size,
        args.update_depth, args.update_init_factor, args.update_hierachy_factor,
        args.use_feat_bank, args.appearance_dim, args.ratio,
        args.add_opacity_dist, args.add_cov_dist, args.add_color_dist,
        use_per_gaussian_seg=getattr(args, "use_per_gaussian_seg", False),
        num_classes=getattr(args, "num_classes", 1),
        no_opacity_detach=getattr(args, "no_opacity_detach", False),
        dual_feature=dual_feature,
        seg_feature_dim=seg_feature_dim,
        seg_decoder_hidden=seg_decoder_hidden,
        seg_decoder_layers=seg_decoder_layers,
    )

    # Find iteration
    if args.iteration == -1:
        from utils.system_utils import searchForMaxIteration
        loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    else:
        loaded_iter = args.iteration
    print(f"Loading iteration: {loaded_iter}")

    # Need to set appearance embedding size before loading MLPs
    # Count train cameras from COLMAP
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(source_path, "sparse/0", "images.bin"))
    except:
        cam_extrinsics = read_extrinsics_text(os.path.join(source_path, "sparse/0", "images.txt"))
    num_cameras = len(cam_extrinsics)
    gaussians.set_appearance(num_cameras)

    # Load weights
    ckpt_dir = os.path.join(model_path, "point_cloud", f"iteration_{loaded_iter}")
    ply_path = os.path.join(ckpt_dir, "point_cloud.ply")
    gaussians.load_ply_sparse_gaussian(ply_path)
    gaussians.load_mlp_checkpoints(ckpt_dir)
    print(f"Loaded checkpoint from {ckpt_dir}")

    gaussians.eval()

    # --- Load test cameras ---
    if not args.skip_render:
        train_cam_infos, test_cam_infos = readColmapSceneInfo_with_testlist(
            source_path, args.images, test_list_file
        )
        if len(test_cam_infos) == 0:
            print("ERROR: No test cameras found!")
            sys.exit(1)

        # Create Camera objects using loadCam
        class FakeArgs:
            resolution = args.resolution
            data_device = args.data_device

        test_cameras = []
        for id, cam_info in enumerate(tqdm(test_cam_infos, desc="Loading test cameras")):
            cam = loadCam(FakeArgs(), id, cam_info, resolution_scale=1.0)
            test_cameras.append(cam)

        # --- Render ---
        test_dir = os.path.join(model_path, "test", f"ours_{loaded_iter}")
        renders_dir = os.path.join(test_dir, "renders")
        gt_dir = os.path.join(test_dir, "gt")
        errors_dir = os.path.join(test_dir, "errors")
        masks_rendered_dir = os.path.join(test_dir, "masks_rendered")
        os.makedirs(renders_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(errors_dir, exist_ok=True)
        os.makedirs(masks_rendered_dir, exist_ok=True)

        bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        pipe = pipeline_params.extract(args)

        print(f"\nRendering {len(test_cameras)} test views...")
        name_mapping = []  # [(idx, image_name), ...]
        with torch.no_grad():
            for idx, view in enumerate(tqdm(test_cameras, desc="Rendering")):
                voxel_mask = prefilter_voxel(view, gaussians, pipe, background)
                render_pkg = render(view, gaussians, pipe, background, visible_mask=voxel_mask)
                rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt = view.original_image[0:3, :, :].cuda()
                error = (rendering - gt).abs()

                name = f"{idx:05d}.png"
                torchvision.utils.save_image(rendering, os.path.join(renders_dir, name))
                torchvision.utils.save_image(gt, os.path.join(gt_dir, name))
                torchvision.utils.save_image(error, os.path.join(errors_dir, name))

                # Save rendered segmentation mask (probability map)
                rendered_mask = decode_rendered_mask(gaussians, render_pkg["mask"])  # [1, H, W]
                torchvision.utils.save_image(rendered_mask, os.path.join(masks_rendered_dir, name))

                # Record mapping from index to original image name
                name_mapping.append((idx, view.image_name))

        # Save name mapping for mask evaluation
        mapping_file = os.path.join(test_dir, "name_mapping.json")
        with open(mapping_file, "w") as f:
            json.dump(name_mapping, f)
        print(f"Rendered to {test_dir}")

    # --- Compute metrics ---
    test_dir = os.path.join(model_path, "test", f"ours_{loaded_iter}")
    renders_dir = os.path.join(test_dir, "renders")
    gt_dir = os.path.join(test_dir, "gt")

    render_files = sorted(Path(renders_dir).glob("*.png"))
    gt_files = sorted(Path(gt_dir).glob("*.png"))

    if len(render_files) == 0:
        print("ERROR: No rendered images found! Run without --skip_render first.")
        sys.exit(1)

    print(f"\nComputing metrics on {len(render_files)} images...")
    lpips_fn = lpips.LPIPS(net="vgg").to("cuda")

    ssims, psnrs, lpipss = [], [], []
    for r_path, g_path in zip(render_files, gt_files):
        r_img = torchvision.io.read_image(str(r_path)).float() / 255.0
        g_img = torchvision.io.read_image(str(g_path)).float() / 255.0

        r_img = r_img.unsqueeze(0).cuda().contiguous()
        g_img = g_img.unsqueeze(0).cuda().contiguous()

        ssims.append(ssim(r_img, g_img).item())
        mse = ((r_img - g_img) ** 2).view(r_img.shape[0], -1).mean(1, keepdim=True)
        psnrs.append((20 * torch.log10(1.0 / torch.sqrt(mse + 1e-8))).item())
        lpipss.append(lpips_fn(r_img * 2 - 1, g_img * 2 - 1).item())

    results = {
        "model_path": model_path,
        "iteration": loaded_iter,
        "num_images": len(render_files),
        "PSNR": float(np.mean(psnrs)),
        "SSIM": float(np.mean(ssims)),
        "LPIPS": float(np.mean(lpipss)),
    }

    # --- Compute mIoU on dronev4_2 test set ---
    if not args.skip_mask_metrics:
        masks_rendered_dir = os.path.join(test_dir, "masks_rendered")
        mapping_file = os.path.join(test_dir, "name_mapping.json")

        if not os.path.exists(mapping_file):
            print("WARNING: name_mapping.json not found, skipping mIoU. Re-run without --skip_render.")
        else:
            with open(mapping_file) as f:
                name_mapping = json.load(f)

            gt_masks_dir = os.path.join(source_path, "masks")
            if not os.path.isdir(gt_masks_dir):
                print(f"WARNING: GT masks not found at {gt_masks_dir}, skipping mIoU")
            else:
                print(f"\nComputing mIoU on {len(name_mapping)} test images...")
                ious_fg, ious_bg = [], []
                valid_count = 0
                for idx, img_name in name_mapping:
                    rendered_mask_path = os.path.join(masks_rendered_dir, f"{idx:05d}.png")
                    gt_mask_path = os.path.join(gt_masks_dir, f"{img_name}.png")

                    if not os.path.exists(gt_mask_path):
                        continue

                    # Load rendered mask (probability in [0,1])
                    r_mask = torchvision.io.read_image(rendered_mask_path).float() / 255.0  # [1, H, W]
                    # Load GT mask (0=bg, 255=fg)
                    g_mask_raw = torchvision.io.read_image(gt_mask_path)  # [1, H, W] uint8
                    g_mask = (g_mask_raw > 127).float()  # binary: 1=fg, 0=bg

                    # Resize rendered mask to match GT if needed
                    if r_mask.shape != g_mask.shape:
                        r_mask = torch.nn.functional.interpolate(
                            r_mask.unsqueeze(0), size=g_mask.shape[1:], mode="bilinear", align_corners=False
                        ).squeeze(0)

                    # Threshold at 0.5
                    pred = (r_mask > 0.5).float()

                    # IoU for foreground (1) and background (0)
                    for cls_val, cls_iou_list in [(1.0, ious_fg), (0.0, ious_bg)]:
                        pred_cls = (pred == cls_val)
                        gt_cls = (g_mask == cls_val)
                        intersection = (pred_cls & gt_cls).sum().item()
                        union = (pred_cls | gt_cls).sum().item()
                        if union > 0:
                            cls_iou_list.append(intersection / union)

                    valid_count += 1

                if valid_count > 0:
                    miou_fg = float(np.mean(ious_fg)) if ious_fg else 0.0
                    miou_bg = float(np.mean(ious_bg)) if ious_bg else 0.0
                    miou = (miou_fg + miou_bg) / 2.0
                    results["mIoU"] = miou
                    results["mIoU_fg"] = miou_fg
                    results["mIoU_bg"] = miou_bg
                    results["mask_eval_images"] = valid_count
                    print(f"  FG IoU: {miou_fg:.4f}")
                    print(f"  BG IoU: {miou_bg:.4f}")
                    print(f"  mIoU:   {miou:.4f}")
                else:
                    print("WARNING: No valid GT mask pairs found for mIoU computation")

    print(f"\n{'='*40}")
    print(f"  Results: {model_path}")
    print(f"  Iteration: {loaded_iter}")
    print(f"  Test images: {results['num_images']}")
    print(f"  PSNR:  {results['PSNR']:.4f}")
    print(f"  SSIM:  {results['SSIM']:.4f}")
    print(f"  LPIPS: {results['LPIPS']:.4f}")
    if "mIoU" in results:
        print(f"  mIoU:  {results['mIoU']:.4f} (FG={results['mIoU_fg']:.4f}, BG={results['mIoU_bg']:.4f})")
    print(f"{'='*40}")

    out_file = os.path.join(model_path, f"eval_results_{loaded_iter}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
