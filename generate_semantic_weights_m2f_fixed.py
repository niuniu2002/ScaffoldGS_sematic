"""
Generate semantic weight maps using SegFormer.

Usage:
    # Default (myvideo)
    cd /mnt/data/liufengyang/mmsegmentation
    /mnt/data/liufengyang/envs/segformer/bin/python \
        /mnt/data/liufengyang/data/Scaffold-GSLFY/generate_semantic_weights.py

    # Custom dataset (e.g., dronev4_2)
    /mnt/data/liufengyang/envs/segformer/bin/python \
        /mnt/data/liufengyang/data/Scaffold-GSLFY/generate_semantic_weights.py \
        --image_dir /mnt/data/liufengyang/data/dronev4_2/images \
        --output_dir /mnt/data/liufengyang/data/dronev4_2/semantic_weights

Output:
    semantic_weight maps saved to <output_dir>/
    Pixel value: 0=bg_confident, 128=uncertain, 255=fg_confident
"""
import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add mmsegmentation to path
sys.path.insert(0, '/mnt/data/liufengyang/mmsegmentation')

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import init_model, inference_model


def generate_semantic_weights(
    config_path='/mnt/data/liufengyang/mmsegmentation/work_dirs/mask2former_swinb_flower_h20_bs8/mask2former_swin-b_flower_h20_final_fixed_bs4.py',
    checkpoint_path='/mnt/data/liufengyang/mmsegmentation/work_dirs/mask2former_swinb_flower_h20_bs8/best_mIoU_iter_15000.pth',
    image_dir='/mnt/data/liufengyang/data/myvideo/images',
    output_dir='/mnt/data/liufengyang/data/myvideo/semantic_weights',
    device='cuda:0',
    num_classes=2,
):
    assert num_classes == 2, f"[WARNING] generate_semantic_weights_m2f_fixed.py currently only supports binary segmentation (num_classes=2). Got num_classes={num_classes}. Please adapt fg_prob = probs[1] for multi-class."
    os.makedirs(output_dir, exist_ok=True)

    # Init model
    print(f"Loading model from {checkpoint_path} ...")
    model = init_model(config_path, checkpoint_path, device=device)
    model.eval()
    print("Model loaded.")

    # Get all images
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')) +
                         glob.glob(os.path.join(image_dir, '*.jpg')) +
                         glob.glob(os.path.join(image_dir, '*.JPEG')))
    print(f"Found {len(image_paths)} images.")

    for img_path in tqdm(image_paths, desc="Generating semantic weights"):
        img_name = os.path.basename(img_path).split('.')[0]
        out_path = os.path.join(output_dir, img_name + '.png')
        if os.path.exists(out_path):
            continue

        # Inference
        result = inference_model(model, img_path)
        # result.seg_logits.data: [C, H, W] logits
        seg_logits = result.seg_logits.data  # torch.Tensor on GPU

        # Get foreground probability (class 1)
        # Apply softmax to get per-class probabilities
        seg_logits = seg_logits * 10.0  # temperature scaling for Mask2Former
        probs = torch.softmax(seg_logits, dim=0)  # [C, H, W]
        fg_prob = probs[1]  # [H, W], foreground probability

        # Convert to uint8: 0=bg(0.0), 128=uncertain(0.5), 255=fg(1.0)
        weight_uint8 = (fg_prob.cpu().numpy() * 255).astype(np.uint8)

        # Save
        Image.fromarray(weight_uint8).save(out_path)

    print(f"Done! Semantic weights saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate semantic weight maps')
    parser.add_argument('--image_dir', type=str, default='/mnt/data/liufengyang/data/myvideo/images')
    parser.add_argument('--output_dir', type=str, default='/mnt/data/liufengyang/data/myvideo/semantic_weights')
    parser.add_argument('--config_path', type=str, default='/mnt/data/liufengyang/mmsegmentation/work_dirs/mask2former_swinb_flower_h20_bs8/mask2former_swin-b_flower_h20_final_fixed_bs4.py')
    parser.add_argument('--checkpoint_path', type=str, default='/mnt/data/liufengyang/mmsegmentation/work_dirs/mask2former_swinb_flower_h20_bs8/best_mIoU_iter_15000.pth')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    generate_semantic_weights(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
